import argparse
import copy
import pickle
import re
from nltk import WordNetLemmatizer
import shutil
import numpy as np
import logging
import os
import random
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PYEVALB import scorer
from PYEVALB import parser
from torchtext.data import Dataset, Field
import torch
import torch.optim as optim
import timeit
from example import make_example_from_oracles
# from fields import ActionField
# from iterator import SimpleIterator
from models import DiscRNNG
from oracle import DiscOracle
from torchtext.data import Iterator
from torchtext import vocab
import utils
from tf_logger_class import Logger
import glob
import json
from action_prod_field import ActionRuleField
# from production import Production
from nltk.grammar import Production
import production

CACHE_DIR = './cache'


def load_pretrained_model(type, pretrained_file):
    cache_file = os.path.join('./cache/', type + '_vocab.pkl')
    if os.path.exists(cache_file):
        print('loading cached vocab from', cache_file)
        res = pickle.load(open(cache_file, 'rb'))
        print("Done.", len(res), " words loaded!")
        return res

    else:
        print("Loading Model from", pretrained_file)
        f = open(pretrained_file, 'r')
        pretrained_vocab = []
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            pretrained_vocab.append(word)

        with open(cache_file, 'wb') as f:
            pickle.dump(pretrained_vocab, f)
        print("Done.", len(pretrained_vocab), " words loaded!")
        return pretrained_vocab


class Trainer(object):
    def __init__(self,
                 id,
                 train_corpus,
                 test_corpus,
                 save_to,
                 emb_path,
                 dev_corpus,
                 emb_type,
                 train_grammar_file,
                 rnng_type='discriminative',
                 lower=True,
                 min_freq=1,
                 lemma=True,
                 word_embedding_size=100,
                 pos_embedding_size=10,
                 nt_embedding_size=60,
                 action_embedding_size=36,
                 input_size=128,
                 hidden_size=128,
                 num_layers=2,
                 dropout=0.2,
                 learning_rate=0.01,
                 max_epochs=1000,
                 seed=25122017,
                 log_interval=100,
                 cuda=False,
                 batch_size=1,
                 clip=10,
                 new_corpus=True,
                 # small_corpus=True,
                 debug_mode=False,
                 use_unk=True,
                 patience=5,
                 resume_dir=None,
                 optimizer='adam',
                 exclude_word_emb=False,
                 use_cache=False,
                 cache_path="./cache"):

        self.id = id
        self.use_cache = use_cache
        self.patience = patience
        self.use_unk = use_unk
        self.lemma = lemma
        self.emb_type = emb_type
        self.new_corpus = new_corpus
        self.clip = clip
        self.test_corpus = test_corpus
        self.cache_path = cache_path
        self.resume_dir = resume_dir
        self.optimizer_type = optimizer
        self.train_corpus = train_corpus
        self.exclude_word_embs = exclude_word_emb
        self.dev_corpus = dev_corpus
        self.rnng_type = rnng_type
        self.lower = lower
        self.min_freq = min_freq
        self.word_embedding_size = word_embedding_size if self.emb_type == 'glove' else 100
        self.pos_embedding_size = pos_embedding_size
        self.nt_embedding_size = nt_embedding_size
        self.action_embedding_size = action_embedding_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.cuda = cuda and torch.cuda.is_available()
        self.seed = seed
        self.log_interval = log_interval
        self.batch_size = batch_size
        if self.emb_type == 'glove':
            self.pretrained_emb_path = os.path.join(emb_path, 'glove.6B.' + str(self.word_embedding_size) + 'd.txt')
        else:
            self.pretrained_emb_path = os.path.join(emb_path, 'sskip.100.vectors')
        self.attributes_dict = self.__dict__.copy()
        print(self.attributes_dict)

        self.singletons = set()
        self.hper = 'id={};optimizer={};unk={};new={};emb_type={};lemma={};lr={:.4f};word={};clip={}'.format(
            self.id,
            self.optimizer_type,
            self.use_unk,
            self.new_corpus,
            self.emb_type,
            self.lemma,
            self.learning_rate,
            self.word_embedding_size,
            self.clip,
        )
        # self.hper = 'id={}'.format(self.id)
        self.save_to = save_to
        self.debug_mode = debug_mode
        self.train_productions = production.get_productions_from_file(train_grammar_file)  # type: List[Production]

    def set_random_seed(self) -> None:
        # self.logger.info('Setting random seed to %d', self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def prepare_output_dir(self) -> None:
        # logger
        self.logger_path = os.path.join(self.save_to, 'logger.txt')
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        if self.debug_mode:
            handler = logging.StreamHandler()
        else:
            handler = logging.FileHandler(self.logger_path)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s - %(levelname)s - %(funcName)10s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        self.logger = logger

        self.logger.info('Preparing output directory in %s', self.save_to)
        self.save_to = os.path.join(self.save_to, self.hper)
        if os.path.exists(self.save_to):
            self.logger.warning('There exists the same' + str(self.save_to))
        os.makedirs(self.save_to, exist_ok=True)
        self.fields_dict_path = os.path.join(self.save_to, 'fields_dict.pkl')
        self.model_metadata_path = os.path.join(self.save_to, 'model_metadata.json')

        with open(self.model_metadata_path, 'w') as fp:
            json.dump(self.attributes_dict, fp)
        self.saved_model_dir = os.path.join(self.save_to, 'saved_model')
        os.makedirs(self.saved_model_dir, exist_ok=True)

        # tensorboard
        self.tf_log_path = os.path.join(self.save_to, 'tf_log')
        if os.path.exists(self.tf_log_path):
            shutil.rmtree(self.tf_log_path)
            self.logger.warning('Removed ' + str(self.tf_log_path))
        self.tf_logger = Logger(self.tf_log_path)
        self.logger.info('TF logger into ' + self.tf_log_path)

    def init_fields(self) -> None:
        self.WORDS = Field(pad_token=None, lower=self.lower)
        self.POS_TAGS = Field(pad_token=None)
        self.NONTERMS = Field(pad_token=None)
        # self.ACTIONS = ActionField(self.NONTERMS)
        self.ACTIONS = ActionRuleField(self.NONTERMS, self.train_productions)
        self.RAWS = Field(lower=self.lower, pad_token=None)
        # self.LABELS = Field(use_vocab=False, sequential=False)
        self.fields = [
            ('actions', self.ACTIONS), ('nonterms', self.NONTERMS),
            ('pos_tags', self.POS_TAGS), ('words', self.WORDS),
            ('raws', self.RAWS),
            # ('labels', self.LABELS)
        ]

    def process_each_corpus(self, corpus, name=None, shuffle=True):
        assert corpus is not None
        dataset = self.make_dataset(corpus, name)
        iterator = Iterator(dataset,
                            shuffle=shuffle,
                            device=torch.device('cuda') if self.cuda else -1,
                            batch_size=self.batch_size,
                            repeat=False)
        return dataset, iterator

    def process_corpora(self) -> None:
        self.train_dataset, self.train_iterator = self.process_each_corpus(self.train_corpus, 'train', shuffle=True)
                                                                           # training=True)
        self.logger.info('Len of train = ' + str(len(self.train_iterator)))

        if self.dev_corpus:
            self.dev_dataset, self.dev_iterator = self.process_each_corpus(self.dev_corpus, 'dev', shuffle=False)
            self.logger.info('Len of dev = ' + str(len(self.dev_iterator)))

        if self.test_corpus:
            self.test_dataset, self.test_iterator = self.process_each_corpus(self.test_corpus, 'test', shuffle=False)
            self.logger.info('Len of test = ' + str(len(self.test_iterator)))

    def build_vocabularies(self) -> None:
        def extend_vocab(field, word_lst, using_vector=False):
            cnt_add_w = 0
            for w in word_lst:
                if w not in field.vocab.stoi:
                    cnt_add_w += 1
                    field.vocab.itos.append(w)
                    field.vocab.stoi[w] = len(field.vocab.itos) - 1
                else:
                    self.logger.warning(w + ' is already in the field')
            if using_vector:
                self.logger.info('Add ' + str(cnt_add_w) + ' zero vectors into vocab.vectors')
                field.vocab.vectors = torch.cat((field.vocab.vectors,
                                                 torch.zeros(cnt_add_w, self.word_embedding_size)), 0)

        self.logger.info('Building vocabularies')
        self.logger.info('Loading pretrained vectors from' + self.pretrained_emb_path)
        pretrained_vec = vocab.Vectors(os.path.basename(self.pretrained_emb_path),
                                       os.path.dirname(self.pretrained_emb_path))
        self.WORDS.build_vocab(self.train_dataset, min_freq=self.min_freq, vectors=pretrained_vec)
        extend_vocab(self.WORDS, self.singletons, using_vector=True)
        cnt_zero = 0
        zero_words = []

        for cnt, each_vec in enumerate(self.WORDS.vocab.vectors):
            if each_vec.sum().item() == 0:
                cnt_zero += 1
                cur_word = self.WORDS.vocab.itos[cnt]
                assert cur_word.startswith('unk') or cur_word == '<unk>' or cur_word in self.singletons
                self.WORDS.vocab.vectors[cnt] = np.random.normal(0, 0.05)
                zero_words.append(cur_word)

        self.logger.info('There are ' + str(cnt_zero) + ' zero embeddings')
        print('Zero words = ', zero_words)
        assert cnt_zero > 1

        self.POS_TAGS.build_vocab(self.train_dataset)
        self.NONTERMS.build_vocab(self.train_dataset)
        extend_vocab(self.NONTERMS, ['<w>'])

        self.ACTIONS.build_vocab()
        assert self.ACTIONS.vocab.itos[2] == 'NP(TOP -> S)'
        self.RAWS.build_vocab(self.train_dataset)

        self.num_words = len(self.WORDS.vocab)
        self.num_pos = len(self.POS_TAGS.vocab)
        self.num_nt = len(self.NONTERMS.vocab)
        self.num_actions = len(self.ACTIONS.vocab)
        self.logger.info('Found %d words, %d POS tags, %d nonterminals, %d actions',
                         self.num_words, self.num_pos, self.num_nt, self.num_actions)
        # self.logger.info('Saving fields dict to%s', self.fields_dict_path)
        # torch.save(dict(self.fields), self.fields_dict_path, pickle_module=dill)

    def build_model(self) -> None:
        self.logger.info('Building model')
        model_args = (self.num_words, self.num_pos, self.num_nt)
        model_kwargs = dict(
            word_embedding_size=self.word_embedding_size,
            pos_embedding_size=self.pos_embedding_size,
            nt_embedding_size=self.nt_embedding_size,
            action_embedding_size=self.action_embedding_size,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            pretrained_emb_vec=self.WORDS.vocab.vectors,
            productions=self.train_productions,
            nonterms=self.NONTERMS,
        )
        self.model = DiscRNNG(*model_args, **model_kwargs)
        if self.cuda:
            self.model.cuda()

    def preprocess_token(self, token):
        if token.startswith('unk') or token == '<unk>':
            return token

        if self.lemma:
            wordnet_lemmatizer = WordNetLemmatizer()
            processed_token = wordnet_lemmatizer.lemmatize(token)
        else:
            processed_token = token

        if processed_token in self.pretrainedVec:
            return processed_token
        else:
            if token in self.pretrainedVec:
                return token
            else:
                return '<unk>'

    def make_oracles(self, corpus: str):
        oracles = []
        f = open(corpus, 'r')
        line = f.readline()

        while line:
            assert line.startswith('# ')
            raw = line[2:].strip()
            pos_tag = f.readline().strip()
            token = f.readline().strip()
            # lower = f.readline().strip()
            unks = f.readline().strip()
            unk_lst = [self.preprocess_token(x.lower()) for x in unks.split()]
            raw_token_lst = [self.preprocess_token(x.lower()) for x in token.split()]

            # replace <unk> in raw with specific unk type in unk_lst
            for id, raw_token in enumerate(raw_token_lst):
                if raw_token == '<unk>':
                    raw_token_lst[id] = unk_lst[id]

            # find all unk types
            for id, w in enumerate(raw_token_lst):
                if unk_lst[id].startswith('unk'):
                    if unk_lst[id] == 'unk-num':
                        continue
                    if not w.startswith('unk'):
                        self.singletons.add(w)

            # get action seqs
            actions = []
            while True:
                line = f.readline().strip()
                if line == '':
                    break
                assert 'NP' in line or line == 'SHIFT' or line == 'REDUCE'
                actions.append(line)

            oracles.append(DiscOracle(actions, pos_tag.split(), unk_lst, raw_token_lst))
            # if training:
            #     if raw_token_lst != unk_lst:
            #         oracles.append(DiscOracle(actions, pos_tag.split(), raw_token_lst, raw_token_lst, False))

            line = f.readline()
            while line == '\n': line = f.readline()
        return oracles

    def make_dataset(self, corpus, name):
        cached_corpus = os.path.join(CACHE_DIR, 'cached_' + name + '.pkl')
        if self.use_cache:
            self.logger.info('Loading cached corpus from ' + cached_corpus)
            oracles = torch.load(cached_corpus)
        else:
            self.logger.info('Reading from %s', corpus)
            oracles = self.make_oracles(corpus)
            self.logger.info('Dumping cached corpus to ' + cached_corpus)
            torch.save(oracles, cached_corpus)

        examples = [make_example_from_oracles(x, self.fields) for x in oracles]
        res = Dataset(examples, self.fields)
        return res

    def save_model(self, epoch):
        self.saved_model_file = os.path.join(self.saved_model_dir, 'epoch_{}.pth'.format(epoch))
        self.logger.info('Save model parameters to %s', self.saved_model_file)
        torch.save(self.model.state_dict(), self.saved_model_file)

    def id2original(self, field, ids):
        if type(ids) == torch.Tensor:
            ids = ids.view(-1).cpu().numpy()
        return [field.vocab.itos[x] for x in ids]

    def get_eval_metrics(self, instance, pred_action_ids):
        assert type(pred_action_ids) == list
        pred_actions = self.id2original(self.ACTIONS, pred_action_ids)
        gold_actions = self.id2original(self.ACTIONS, instance.actions)

        tokens = self.id2original(self.WORDS, instance.words)
        pos_tags = self.id2original(self.POS_TAGS, instance.pos_tags)

        measure = scorer.Scorer()
        golden_tree_seq = utils.actions2treestr(gold_actions, tokens, pos_tags)
        gold_tree = parser.create_from_bracket_string(golden_tree_seq)
        try:
            pred_tree_seq = utils.actions2treestr(pred_actions, tokens, pos_tags)
            pred_tree = parser.create_from_bracket_string(pred_tree_seq)
            ret = measure.score_trees(gold_tree, pred_tree)
        except:
            return -1
        else:
            match_num = ret.matched_brackets
            gold_num = ret.gold_brackets
            pred_num = ret.test_brackets
            return match_num, gold_num, pred_num

    def build_optimizer(self):
        if self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.logger.info('Using ' + self.optimizer_type + ' as optimizer, lr = ' + str(self.learning_rate))

        self.scheduler = ReduceLROnPlateau(self.optimizer,
                                           mode='min', factor=0.75,
                                           patience=self.patience,
                                           verbose=True, threshold=0.001)
        self.losser = torch.nn.NLLLoss(reduction='sum')

    def training(self):
        self.logger.info('Start training ...')
        self.model.train()
        if torch.cuda.is_available() and not self.cuda:
            self.logger.warning("WARNING: You have a CUDA device, so you should probably run with --cuda")

        cnt_change = 0
        epoch_meter = utils.ParsingMeter()
        interval_meter = utils.ParsingMeter()
        total_step = 0
        best_dev_f1 = 0
        # cnt = 0
        for epoch in range(self.max_epochs):
            start_time = timeit.default_timer()
            for cnt, instance in enumerate(self.train_iterator):
                self.model.zero_grad()
                # if instance.labels.item() == False: continue

                # [value, batch_size] -> [value]
                # only have 1 batch
                unk_words = instance.words.view(-1)
                pos_tags = instance.pos_tags.view(-1)
                actions = instance.actions.view(-1)
                raw_words = instance.raws.view(-1)

                # replace unk
                origin_unk_words = self.id2original(self.WORDS, instance.words)
                origin_raw_words = self.id2original(self.RAWS, instance.raws)
                if origin_raw_words != origin_unk_words:
                    for id, word_id in enumerate(unk_words):
                        cur_unk_word = self.WORDS.vocab.itos[word_id]

                        if cur_unk_word.startswith('unk') and cur_unk_word != 'unk-num': #doesn't replace unk-num
                            singleton_word = self.RAWS.vocab.itos[raw_words[id]]
                            if singleton_word != '<unk>':
                                if random.random() > 0.5:
                                    unk_words[id] = self.WORDS.vocab.stoi[singleton_word]
                                    cnt_change += 1
                                    self.logger.info('Change from ' + cur_unk_word + ' into ' + singleton_word)

                self.log_logits, self.pred_action_ids = self.model.forward(unk_words, pos_tags, actions)

                self.training_loss = self.losser(self.log_logits, instance.actions.view(-1))
                self.training_loss.backward()
                if self.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                self.optimizer.step()
                res = self.get_eval_metrics(instance, self.pred_action_ids)
                epoch_meter.update(res)
                interval_meter.update(res)

                # logging
                if (cnt + 1) % self.log_interval == 0 or cnt == len(self.train_iterator) - 1:
                    elapsed = timeit.default_timer() - start_time
                    start_time = timeit.default_timer()
                    self.logger.info(
                        'Epoch [{}/{}], Step [{}/{}], Train_Loss: {:.4f}, F1_train: {:.4f}, Error_tree: {}, Time: {:.4f}'.format(
                            epoch + 1,
                            self.max_epochs,
                            cnt + 1,
                            len(self.train_iterator),
                            self.training_loss.item(),
                            epoch_meter.f1,
                            interval_meter.error_tree,
                            elapsed))

                    info = {'train_loss': self.training_loss.item(),
                            'interval_error_tree': interval_meter.error_tree,
                            'interval_F1': interval_meter.f1}
                    for tag, value in info.items():
                        self.tf_logger.scalar_summary(tag=tag, value=value, step=total_step + 1)
                    total_step += 1
                    interval_meter.reset()
                # cnt += 1

            # DEV
            # self.logger.info('Change ' + str(cnt_change) + ' unk tokens into singleton tokens')
            cnt_change = 0
            if self.dev_corpus:
                dev_meter = self.inference(self.dev_iterator, type_corpus='dev', step=epoch + 1, tf_board=True)
                if dev_meter.f1 > best_dev_f1:
                    best_dev_f1 = dev_meter.f1
                    self.logger.info('Best F1: ' + str(best_dev_f1))
                    saved_files = glob.glob(os.path.join(self.saved_model_dir, '*'))
                    for file in saved_files:
                        os.remove(file)
                        self.logger.warning('Removed' + str(file))
                    self.save_model(epoch + 1)
                self.model.train()
                self.scheduler.step(dev_meter.f1)

            epoch_meter.reset()
        self.save_model('final')
        self.logger.info('Finish training')

    def get_info_infer(self, iterator):
        infer_meter = utils.ParsingMeter()
        self.model.eval()
        with torch.no_grad():
            for instance in iterator:
                self.pred_action_ids, _ = self.model.decode(instance.words, instance.pos_tags)
                metric = self.get_eval_metrics(instance, self.pred_action_ids)
                infer_meter.update(metric)
        return infer_meter

    def inference(self, iterator, type_corpus, step=1, tf_board=True):
        self.logger.info('Testing on ' + type_corpus)
        infer_meter = self.get_info_infer(iterator)
        self.logger.info('F1: {}, Error tree {}'.format(infer_meter.f1, infer_meter.error_tree))

        if tf_board:
            test_info = {type_corpus + '_F1': infer_meter.f1,
                         type_corpus + '_error_tree': infer_meter.error_tree}
            for tag, value in test_info.items():
                self.tf_logger.scalar_summary(tag=tag, value=value, step=step)
        return infer_meter

    def load_model(self, resume_dir):
        resume_dir = re.sub('\'', '', resume_dir)
        self.logger.info('Get file from dir ' + str(resume_dir))
        self.resume_file_lst = glob.glob(os.path.join(resume_dir, '*'))
        assert len(self.resume_file_lst) == 1
        self.resume_file = self.resume_file_lst[0]
        self.logger.info('Loading model from ' + str(self.resume_file))

        if self.exclude_word_embs:
            self.logger.info('Excluding word embedding from pretrained model')
            pretrained_dict = torch.load(self.resume_file)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'word_embedding' not in k}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        else:
            self.model.load_state_dict(torch.load(self.resume_file))
        self.logger.info('Done loading.')

    def check_grad(self):
        self.model.train()
        instance = next(iter(self.train_iterator))
        self.model.zero_grad()
        self.log_logits, self.pred_action_ids = self.model.forward(instance.words.view(-1),
                                                                   instance.pos_tags.view(-1),
                                                                   instance.actions.view(-1))

        self.training_loss = self.losser(self.log_logits, instance.actions.view(-1))
        assert self.training_loss.item() != 0
        assert not torch.isinf(self.training_loss)
        self.logger.warning('Loss = ' + str(self.training_loss.item()))

        self.training_loss.backward()
        self.optimizer.step()

        for name, para in self.model.named_parameters():
            if (para.grad is None or para.grad.equal(torch.zeros_like(para.grad))) and para.requires_grad:
                raise ValueError('There is no grad at', name)

    def check_zero_embedding(self):
        for name in 'word pos nt action'.split():
            embedding = getattr(self.model, '{}_embedding'.format(name))
            # print (name, embedding)
            for row in embedding.weight.data:
                assert not row.equal(torch.zeros_like(row))

        for name, para in self.model.named_parameters():
            if para.equal(torch.zeros_like(para)):
                assert 'bias' in name or 'guard' in name or 'h0' in name or 'c0' in name

    def check_load(self):
        new_paras = list(self.model.named_parameters())
        for old_para, new_para in zip(self.old_paras, new_paras):
            if old_para.equal(new_para[1]):
                self.logger.warning('Same para at ' + new_para[0])

    def unit_test(self):
        self.check_zero_embedding()
        if self.resume_dir:
            self.check_load()
        self.check_grad()
        self.logger.info('Finish all unit tests')

    def run(self):
        self.pretrainedVec = load_pretrained_model(self.emb_type, self.pretrained_emb_path)
        self.set_random_seed()
        self.prepare_output_dir()
        self.init_fields()
        self.process_corpora()
        self.build_vocabularies()
        self.build_model()
        self.build_optimizer()
        self.old_paras = copy.deepcopy(list(self.model.parameters()))
        if self.resume_dir:
            self.load_model(self.resume_dir)
        self.unit_test()
        # self.inference(self.train_iterator, type_corpus='train', tf_board=True)
        self.training()
        self.inference(self.test_iterator, type_corpus='test', tf_board=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train RNNG network')
    parser.add_argument('-t', '--train-corpus', required=True, metavar='FILE', help='path to train corpus')
    parser.add_argument('--train_grammar_file', required=True, metavar='FILE')
    parser.add_argument('-s', '--save-to', required=True, metavar='DIR')
    parser.add_argument('--dev-corpus', metavar='FILE', help='path to dev corpus')
    parser.add_argument('--test-corpus', metavar='FILE')
    parser.add_argument('--emb-path', required=True, type=str, help='Pretrained word emb')
    parser.add_argument('--emb-type', type=str, default='glove', help='Type of embedding')
    parser.add_argument('--word-embedding-size', type=int, default=100, metavar='NUMBER')
    parser.add_argument('--learning-rate', type=float, default=0.05, metavar='NUMBER')
    parser.add_argument('--clip', type=float, default=10, metavar='NUMBER')
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--new-corpus', action='store_true')
    parser.add_argument('--lemma', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--max-epochs', type=int, default=1000, metavar='NUMBER')
    parser.add_argument('--patience', type=int, default=5, metavar='NUMBER')
    parser.add_argument('--resume_dir', type=str, default='')
    parser.add_argument('--exclude_word_emb', action='store_true')
    parser.add_argument('--use-cache', action='store_true')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--id', type=int, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)
    kwargs = vars(args)
    trainer = Trainer(**kwargs)
    trainer.run()

    # def check_input(self):
    #     def check_input_each_corpus(iterator, type):
    #         cache_file = os.path.join(self.cache_path, 'first_' + type + '_cached.pkl')
    #         cache_instance = torch.load(cache_file)
    #         instance = next(iter(iterator))
    #         for attr in 'words pos_tags actions'.split():
    #             attr_input = getattr(instance, attr)
    #
    #     _, unit_train_iterator = self.process_each_corpus(self.train_corpus, 'train', shuffle=False)
    #     if self.dev_corpus:
    #         _, unit_dev_iterator = self.process_each_corpus(self.dev_corpus, 'dev', shuffle=False)
    #     if self.test_corpus:
    #         _, unit_test_iterator = self.process_each_corpus(self.test_corpus, 'test', shuffle=False)
    #
    #     for type in 'train dev test'.split():
    #         # type_iterator = getattr(self, type + '_iterator')
    #         check_input_each_corpus(locals()['unit_' + type + '_iterator'], type)

    # parser.add_argument('--encoding', default='utf-8', help='file encoding to use (default: utf-8)')
    # parser.add_argument('--rnng-type', choices='discriminative'.split(), metavar='TYPE', default='discriminative', help='type of RNNG to train (default: discriminative)')
    # parser.add_argument('--no-lower', action='store_false', dest='lower', help='whether not to lowercase the words')
    # parser.add_argument('--min-freq', type=int, default=2, metavar='NUMBER', help='minimum word frequency to be included in the vocabulary (default: 2)')
    # parser.add_argument('--pos-embedding-size', type=int, default=12, metavar='NUMBER', help='dimension of POS tag embeddings (default: 12)')
    # parser.add_argument('--nt-embedding-size', type=int, default=60, metavar='NUMBER', help='dimension of nonterminal embeddings (default: 60)')
    # parser.add_argument('--action-embedding-size', type=int, default=16, metavar='NUMBER', help='dimension of action embeddings (default: 16)')
    # parser.add_argument('--input-size', type=int, default=128, metavar='NUMBER', help='input dimension of the LSTM parser state encoders (default: 128)')
    # parser.add_argument('--hidden-size', type=int, default=128, metavar='NUMBER',help='hidden dimension of the LSTM parser state encoders (default: 128)')
    # parser.add_argument('--num-layers', type=int, default=2, metavar='NUMBER',help='number of layers of the LSTM parser state encoders and composers (default: 2)')
    # parser.add_argument('--dropout', type=float, default=0.2, metavar='NUMBER', help='dropout rate (default: 0.2)')
    # parser.add_argument('--log-interval', type=int, default=100, metavar='NUMBER', help='print logs every this number of iterations (default: 10)')
    # parser.add_argument('--seed', type=int, default=25122017, help='random seed (default: 25122017)')
    # parser.add_argument('--cuda', dest='cuda', help='whether use CUDA', action='store_true')
