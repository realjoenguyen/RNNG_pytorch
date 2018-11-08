from nltk.tree import Tree
import pprint
import argparse
from typing import List, NamedTuple, Optional, Sequence, Sized, Tuple, Union, cast
import copy
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
import shutil
import numpy as np
import logging
import random
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PYEVALB import scorer
from PYEVALB import parser
from torchtext.data import Dataset, Field, RawField
import torch
import torch.optim as optim
import timeit
from example import make_example_from_oracles
# from models import DiscRNNG
from models_c import DiscRNNG
from oracle import DiscOracle
from torchtext.data import Iterator
from torchtext import vocab
import utils
from tf_logger_class import Logger
import glob
import json
from action_prod_field import ActionRuleField
from production import get_productions_from_file, Production
import os
from cls import CyclicLR
from timeit import default_timer as timer

CACHE_DIR = './cache'
from nltk.corpus import wordnet

#TODO: change device
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def get_wordnet_pos(pos_tag: str):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def load_pretrained_model(type, pretrained_file):
    cache_file = os.path.join('./cache/', type + '_vocab.pkl')
    print (glob.glob('./*'))
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
                 lemma=False,
                 word_embedding_size=100,
                 pos_embedding_size=10,
                 nt_embedding_size=60,
                 action_embedding_size=36,
                 rule_embedding_size=100,
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
                 debug_mode=False,
                 use_unk=True,
                 patience=5,
                 resume_dir=None,
                 optimizer='adam',
                 exclude_word_emb=False,
                 use_cache=False,
                 rule_emb=False,
                 cyclic_lr=False,
                 cache_path="./cache"):

        self.id = id
        self.cyclic_lr = cyclic_lr
        self.use_cache = use_cache
        self.patience = patience
        self.use_unk = use_unk
        self.lemma = lemma
        self.emb_type = emb_type
        self.clip = clip
        self.test_corpus = test_corpus
        self.cache_path = cache_path
        self.resume_dir = resume_dir
        self.optimizer_type = optimizer
        self.train_corpus = train_corpus
        self.exclude_word_emb = exclude_word_emb
        self.dev_corpus = dev_corpus
        self.rnng_type = rnng_type
        self.lower = lower
        self.min_freq = min_freq
        self.word_embedding_size = word_embedding_size if self.emb_type == 'glove' else 100
        self.pos_embedding_size = pos_embedding_size
        self.nt_embedding_size = nt_embedding_size
        self.action_embedding_size = action_embedding_size
        self.rule_embedding_size = rule_embedding_size
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
        self.rule_emb = rule_emb
        if self.emb_type == 'glove':
            self.pretrained_emb_path = os.path.join(emb_path, 'glove.6B.' + str(self.word_embedding_size) + 'd.txt')
        else:
            self.pretrained_emb_path = os.path.join(emb_path, 'sskip.100.vectors')
        self.attributes_dict = self.__dict__.copy()
        pprint.pprint(self.attributes_dict)

        self.singletons = set()
        self.hper = 'id={};rule_emb={};optimizer={};unk={};emb_type={};lemma={};lr={:.4f};word={};clip={}'.format(
            self.id,
            self.rule_emb,
            self.optimizer_type,
            self.use_unk,
            # self.new_corpus,
            self.emb_type,
            self.lemma,
            self.learning_rate,
            self.word_embedding_size,
            self.clip,
        )
        # self.hper = 'id={}'.format(self.id)
        self.save_to = save_to
        self.debug_mode = debug_mode
        self.grammar_file = train_grammar_file

    def get_grammar(self):
        self.productions = get_productions_from_file(self.grammar_file)  # type: List[Production]
        self.logger.info('Done loading grammar from ' + self.grammar_file)

    def set_random_seed(self) -> None:
        # self.logger.info('Setting random seed to %d', self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def prepare_output_dir(self) -> None:
        # logger
        # self.logger.info('Preparing output directory in %s', self.save_to)
        print('Preparing output directory in', self.save_to)
        self.save_to = os.path.join(self.save_to, self.hper)
        if os.path.exists(self.save_to):
            # self.logger.warning('There exists the same' + str(self.save_to))
            print('There exists the same', self.save_to)
        os.makedirs(self.save_to, exist_ok=True)

        self.logger_path = os.path.join(self.save_to, 'logger.txt')
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        if self.debug_mode:
            handler = logging.StreamHandler()
        else:
            handler = logging.FileHandler(self.logger_path)
            print('Logging into', self.logger_path)

        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s - %(levelname)s - %(funcName)10s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        self.logger = logger
        self.fields_dict_path = os.path.join(self.save_to, 'fields_dict.pkl')
        self.model_metadata_path = os.path.join(self.save_to, 'model_metadata.json')

        with open(self.model_metadata_path, 'w') as fp:
            json.dump(self.attributes_dict, fp)
        self.saved_model_dir = os.path.join(self.save_to, 'saved_model')
        os.makedirs(self.saved_model_dir, exist_ok=True)
        self.logger.info('START A NEW PROCESS!')

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
        self.ACTIONS = ActionRuleField(self.NONTERMS, self.productions)
        # self.RAWS = Field(lower=self.lower, pad_token=None)
        self.RAWS = RawField()
        self.SEQ = RawField()
        self.fields = [
            ('raw_seq', self.SEQ),
            ('actions', self.ACTIONS), ('nonterms', self.NONTERMS),
            ('pos_tags', self.POS_TAGS), ('words', self.WORDS),
            ('raws', self.RAWS),
        ]

    def process_each_corpus(self, corpus: str, name: str, shuffle=True):
        assert corpus is not None
        dataset = self.make_dataset(corpus, name)
        iterator = Iterator(dataset,
                            shuffle=shuffle,
                            device=torch.device('cuda') if self.cuda else -1,
                            batch_size=self.batch_size,
                            repeat=False)
        return dataset, iterator

    def get_singletons(self, examples, corpus):
        self.cached_singleton_file = os.path.join(CACHE_DIR, os.path.basename(corpus) + '_singleton.pkl')
        if not self.lemma:
            self.cached_singleton_file += '_notlemma'
        if os.path.exists(self.cached_singleton_file):
            self.logger.info('Loading self.singleton from ' + self.cached_singleton_file)
            self.singletons = torch.load(self.cached_singleton_file)
        else:
            # add singleton into self.singleton
            self.logger.info('Geting singleton from' + str(corpus))
            for example in examples:
                raw_token_lst = example.raws
                unk_lst = example.words
                pos_tag_lst = example.pos_tags
                for id, w in enumerate(raw_token_lst):
                    if unk_lst[id].startswith('unk'):
                        if unk_lst[id] == 'unk-num':  # doesn't replace unk-num with number
                            continue
                        # if not w.startswith('unk'):
                        added_singleton = self.preprocess_token(w, pos_tag_lst[id])
                        self.singletons.add(added_singleton)

            self.logger.info('Dumping cached self.singleton into ' + self.cached_singleton_file)
            torch.save(self.singletons, self.cached_singleton_file)

        assert len(self.singletons) > 0
        self.logger.info('Len singleton = ' + str(len(self.singletons)))

    def process_corpora(self):
        self.train_dataset, self.train_iterator = self.process_each_corpus(self.train_corpus, 'train', shuffle=True)
        self.logger.info('Len of train = ' + str(len(self.train_iterator)))
        if self.dev_corpus:
            self.dev_dataset, self.dev_iterator = self.process_each_corpus(self.dev_corpus, 'dev', shuffle=False)
            self.logger.info('Len of dev = ' + str(len(self.dev_iterator)))
        self.get_singletons(self.train_dataset, self.train_corpus)

        if self.test_corpus:
            self.test_dataset, self.test_iterator = self.process_each_corpus(self.test_corpus, 'test', shuffle=False)
            self.logger.info('Len of test = ' + str(len(self.test_iterator)))

    def build_vocab(self) -> None:
        def extend_vocab(field, word_lst, using_vector=False):
            cnt_add_w = 0
            for w in word_lst:
                if w not in field.vocab.stoi:
                    cnt_add_w += 1
                    field.vocab.itos.append(w)
                    field.vocab.stoi[w] = len(field.vocab.itos) - 1
                # else:
                #     self.logger.warning(w + ' is already in the field')
            if using_vector:
                # self.logger.info('Add ' + str(cnt_add_w) + ' zero vectors into vocab.vectors')
                field.vocab.vectors = torch.cat((field.vocab.vectors,
                                                 torch.zeros(cnt_add_w, self.word_embedding_size)), 0)

        self.logger.info('Building vocabularies')
        self.logger.info('Loading pretrained vectors from' + self.pretrained_emb_path)
        pretrained_vec = vocab.Vectors(os.path.basename(self.pretrained_emb_path),
                                       os.path.dirname(self.pretrained_emb_path))
        self.WORDS.build_vocab(self.train_dataset, min_freq=self.min_freq, vectors=pretrained_vec)
        extend_vocab(self.WORDS, self.singletons, using_vector=True)

        # print vocab to file
        f_write = open(os.path.join(self.save_to, 'vocab.txt'), 'w')
        for w in self.WORDS.vocab.itos:
            f_write.write(w + '\n')

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
        print('Zero words = ', zero_words[-25:] + zero_words[:25])
        assert cnt_zero > 1

        self.POS_TAGS.build_vocab(self.train_dataset)
        self.NONTERMS.build_vocab(self.train_dataset)
        extend_vocab(self.NONTERMS, ['<w>'])

        self.ACTIONS.build_vocab()
        assert self.ACTIONS.vocab.itos[2] == 'NP(TOP -> S)'

        self.num_words = len(self.WORDS.vocab)
        self.num_pos = len(self.POS_TAGS.vocab)
        self.num_nt = len(self.NONTERMS.vocab)
        self.num_actions = len(self.ACTIONS.vocab)
        self.logger.info('Found %d words, %d POS tags, %d nonterminals, %d actions',
                         self.num_words, self.num_pos, self.num_nt, self.num_actions)

    def build_model(self) -> None:
        self.logger.info('Building model')
        model_args = (self.num_words, self.num_pos, self.num_nt)
        model_kwargs = dict(
            word_embedding_size=self.word_embedding_size,
            pos_embedding_size=self.pos_embedding_size,
            nt_embedding_size=self.nt_embedding_size,
            action_embedding_size=self.action_embedding_size,
            rule_embedding_size=self.rule_embedding_size,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            pretrained_emb_vec=self.WORDS.vocab.vectors,
            productions=self.productions,
            nonterms=self.NONTERMS,
            words=self.WORDS,
            rule_emb=self.rule_emb,
        )

        self.model = DiscRNNG(*model_args, **model_kwargs)
        if self.cuda:
            self.model.cuda()

    def preprocess_token(self, token: str, pos_tag: str):
        token = token.lower()
        if token.startswith('unk') or token == '<unk>':
            return token

        if self.lemma:
            wordnet_pos_tag = get_wordnet_pos(pos_tag)
            wordnet_lemmatizer = WordNetLemmatizer()
            if wordnet_pos_tag != '':
                processed_token = wordnet_lemmatizer.lemmatize(token, wordnet_pos_tag)
            else:
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

    def make_oracles(self, corpus: str, name: str):
        oracles = []
        f = open(corpus, 'r')
        line = f.readline()

        while line:
            assert line.startswith('# ')
            raw_seq = line[2:].strip()
            pos_tag_str = f.readline().strip()
            raw_token_str = f.readline().strip()
            # lower = f.readline().strip()
            unks_str = f.readline().strip()
            unk_lst = [self.preprocess_token(x.lower(), tag) for x, tag in zip(unks_str.split(), pos_tag_str.split())]
            raw_token_lst = [x for x in raw_token_str.split()]

            # # replace <unk> in raw with specific unk type in unk_lst
            # for id, raw_token in enumerate(raw_token_lst):
            #     if raw_token == '<unk>':
            #         raw_token_lst[id] = unk_lst[id]

            # get action seqs
            actions = []
            while True:
                line = f.readline().strip()
                if line == '':
                    break
                assert 'NP' in line or line == 'SHIFT' or line == 'REDUCE'
                actions.append(line)

            oracles.append(DiscOracle(raw_seq, actions, pos_tag_str.split(), unk_lst, raw_token_lst))
            line = f.readline()
            while line == '\n': line = f.readline()
        return oracles

    def make_dataset(self, corpus, name):
        corpus_file_name = os.path.basename(corpus)
        if not self.lemma:
            corpus_file_name += '_notlemma'
        cached_corpus = os.path.join(CACHE_DIR, corpus_file_name + '.pkl')
        if self.use_cache:
            self.logger.info('Loading cached corpus from ' + cached_corpus)
            oracles = torch.load(cached_corpus)
        else:
            self.logger.info('Reading from %s', corpus)
            oracles = self.make_oracles(corpus, name)
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

        tokens = instance.raws[0]
        pos_tags = self.id2original(self.POS_TAGS, instance.pos_tags)

        measure = scorer.Scorer()
        golden_tree_seq = instance.raw_seq[0]
        gold_tree = parser.create_from_bracket_string(golden_tree_seq)
        try:
            pred_tree_seq = utils.action2treestr(pred_actions, tokens, pos_tags)
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

        if self.cyclic_lr:
            self.logger.info('Using cyclic learning rate')
            self.scheduler = CyclicLR(self.optimizer, base_lr=0.001, max_lr=0.1, step_size=16000, mode='triangular')
        else:
            self.scheduler = ReduceLROnPlateau(self.optimizer,
                                               mode='min', factor=0.75,
                                               patience=self.patience,
                                               verbose=True, threshold=0.001)

        self.losser = torch.nn.NLLLoss(reduction='sum')

    def training(self):
        if self.rule_emb:
            self.logger.info('Using rule embedding')
        else:
            self.logger.info('Using rule composition')
        self.logger.info('Start training ...')

        self.model.train()
        if torch.cuda.is_available() and not self.cuda:
            self.logger.warning("WARNING: You have a CUDA device, so you should probably run with --cuda")

        cnt_change = 0
        epoch_meter = utils.ParsingMeter()
        interval_meter = utils.ParsingMeter()
        total_step = 0
        self.best_dev_f1 = 0
        interval_loss = 0
        for epoch in range(self.max_epochs):
            start_time = timeit.default_timer()
            for cnt, instance in enumerate(self.train_iterator):
                # start_instance = timer()
                self.model.zero_grad()
                if self.cyclic_lr:
                    self.scheduler.batch_step()

                # [value, batch_size] -> [value]
                # only have 1 batch
                unk_words = instance.words.view(-1)
                pos_tags = instance.pos_tags.view(-1)
                actions = instance.actions.view(-1)

                # replace unk
                origin_unk_words = self.id2original(self.WORDS, instance.words)
                origin_raw_words = instance.raws[0]  # type: List[str]
                if origin_raw_words != origin_unk_words:
                    for id, word_id in enumerate(unk_words):
                        cur_unk_word = self.WORDS.vocab.itos[word_id]
                        if cur_unk_word.startswith('unk') and cur_unk_word != 'unk-num':  # doesn't replace unk-num
                            pos_tag_singleton = self.POS_TAGS.vocab.itos[pos_tags[id]]
                            singleton = self.preprocess_token(origin_raw_words[id], pos_tag_singleton)
                            if singleton != '<unk>' and not singleton.startswith('unk'):
                                if random.random() > 0.5:
                                    unk_words[id] = self.WORDS.vocab.stoi[singleton]
                                    cnt_change += 1
                                    # self.logger.info('Change from ' + cur_unk_word + ' into ' + singleton)

                # print ('before forward =', timer() - start_instance)
                # start = timer()
                word_condi = not unk_words.equal(torch.zeros_like(unk_words))
                pos_tag_condi = not pos_tags.equal(torch.zeros_like(pos_tags))
                action_condi = not actions.equal(torch.zeros_like(actions))
                assert word_condi or pos_tag_condi or action_condi

                self.log_logits, self.pred_action_ids = self.model.forward(instance, unk_words, pos_tags, actions)
                # print ('forward = ', timer() - start)
                # after_forward = timer()
                # print (self.log_logits.size(), instance.actions.view(-1).size())
                self.training_loss = self.losser(self.log_logits, instance.actions.view(-1))
                assert not torch.isinf(self.training_loss)
                assert not torch.isnan(self.training_loss)
                interval_loss += self.training_loss.item()
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
                            interval_loss / self.log_interval,
                            interval_meter.f1,
                            interval_meter.error_tree,
                            elapsed))

                    info = {'train_loss': self.training_loss.item(),
                            'interval_error_tree': interval_meter.error_tree,
                            'interval_F1': interval_meter.f1}
                    for tag, value in info.items():
                        self.tf_logger.scalar_summary(tag=tag, value=value, step=total_step + 1)
                    total_step += 1
                    interval_loss = 0
                    interval_meter.reset()

            # DEV
            # self.logger.info('Change ' + str(cnt_change) + ' unk tokens into singleton tokens')
            cnt_change = 0
            if self.dev_corpus:
                if self.best_dev_f1 == 0:
                    self.save_model(epoch + 1)
                dev_meter = self.inference(self.dev_iterator, type_corpus='dev', step=epoch + 1, tf_board=True)
                if dev_meter.f1 > self.best_dev_f1:
                    self.best_dev_f1 = dev_meter.f1
                    self.logger.info('Best F1: ' + str(self.best_dev_f1))
                    saved_files = glob.glob(os.path.join(self.saved_model_dir, '*'))
                    for file in saved_files:
                        os.remove(file)
                        self.logger.warning('Removed' + str(file))
                    self.save_model(epoch + 1)

                self.model.train()
                if not self.cyclic_lr:
                    self.scheduler.step(dev_meter.f1)

            epoch_meter.reset()
        self.logger.info('Best F1: {}'.format(self.best_dev_f1))
        self.logger.info('Finish training')

    # def get_info_infer(self, iterator):
    #     return infer_meter

    def inference(self, iterator, type_corpus, step=1, tf_board=True):
        self.logger.info('Testing on ' + type_corpus)
        result_file = "./post_processing/pred_dev_seq.txt"
        true_file = "./post_processing/dev_seqs.txt"
        f_write = open(result_file, 'w')
        f2_write = open(true_file, 'w')

        self.model.eval()
        infer_meter = utils.ParsingMeter()
        with torch.no_grad():
            for cnt, instance in enumerate(iterator):
                try:
                    self.pred_action_ids, pred_tree = self.model.decode(instance)
                except:
                    print('Decode error at', instance.raw_seq)
                metric = self.get_eval_metrics(instance, self.pred_action_ids)
                infer_meter.update(metric)
                action_str_lst = [self.ACTIONS.vocab.itos[e] for e in self.pred_action_ids]
                pos_tag_lst = [self.POS_TAGS.vocab.itos[e] for e in instance.pos_tags.view(-1)]
                pred_seq = utils.action2treestr(action_str_lst, instance.raws[0], pos_tag_lst)
                f_write.write(pred_seq + "\n")
                f2_write.write(instance.raw_seq[0] + "\n")

                # if metric != 1 and (metric[0] != metric[1] or metric[1] != metric[2]):
                #     f_write.write(instance.raw_seq[0] + "\n")
                #     f_write.write(str(metric) + "\n")
                #     Tree.fromstring(instance.raw_seq[0]).pretty_print(stream=f_write)
                #     Tree.fromstring(pred_seq).pretty_print(stream=f_write)
                #     f_write.write('\n')

                # if (cnt + 1) % 100 == 0:
                #     print ('Current', cnt + 1, infer_meter.f1)
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

        # TODO: notice this

        if self.exclude_word_emb:
            self.logger.warning('Excluding word embedding from pretrained model')
            pretrained_dict = torch.load(self.resume_file)
            model_dict = self.model.state_dict()
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'word_embedding' not in k}
            # model_dict.update(pretrained_dict)
            # self.model.load_state_dict(model_dict)

            # get part of word emb
            self.logger.info('Get part of pretrained word emb')
            cur_word_emb_size = self.model.state_dict()['word_embedding.weight'].size()[0]
            pretrain_word_emb_size = pretrained_dict['word_embedding.weight'].size()[0]
            if cur_word_emb_size < pretrain_word_emb_size:
                pretrained_dict['word_embedding.weight'] = pretrained_dict['word_embedding.weight'][:cur_word_emb_size]
            else:
                model_dict['word_embedding.weight'][:pretrain_word_emb_size] = pretrained_dict['word_embedding.weight']
                pretrained_dict['word_embedding.weight'] = model_dict['word_embedding.weight']

            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        else:
            pretrained_dict = torch.load(self.resume_file)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

        # self.model.load_state_dict(torch.load(self.resume_file))
        self.logger.info('Done loading.')

    def check_grad(self):
        self.model.train()
        instance = next(iter(self.train_iterator))
        self.model.zero_grad()
        for id in instance.words.view(-1).data.tolist():
            assert id < self.num_words and id >= 0
        for id in instance.pos_tags.view(-1).data.tolist():
            assert id < self.num_pos and id >= 0
        for id in instance.actions.view(-1).data.tolist():
            assert id < self.num_actions and id >= 0

        self.log_logits, self.pred_action_ids = self.model.forward(instance,
                                                                   instance.words.view(-1),
                                                                   instance.pos_tags.view(-1),
                                                                   instance.actions.view(-1))

        self.training_loss = self.losser(self.log_logits, instance.actions.view(-1))
        assert self.training_loss.item() != 0
        assert not torch.isinf(self.training_loss)
        assert not torch.isnan(self.training_loss)
        self.logger.warning('Loss = ' + str(self.training_loss.item()))

        self.training_loss.backward()
        self.optimizer.step()

        for name, para in self.model.named_parameters():
            if (para.grad is None or para.grad.equal(torch.zeros_like(para.grad))) and para.requires_grad:
                # if just rule emb -> no grad at nt_embedding
                if self.rule_emb:
                    if 'rule_fwd_composer' or 'rule_bwd_composer' or 'nt_embedding' in name:
                        continue
                else:
                    if 'rule_emb' in name or 'rule2encoder':
                        continue
                raise ValueError('There is no grad at', name)

    def check_zero_embedding(self):
        self.logger.info('Checking zero emb')
        str_check = 'word pos nt action'
        if self.rule_emb: str_check += ' rule'
        for name in str_check.split():
            embedding = getattr(self.model, '{}_embedding'.format(name))
            for id, row in enumerate(embedding.weight.data):
                if row.equal(torch.zeros_like(row)):
                    raise ValueError('Not zero:', name, id)

        self.logger.info('Checking zero para')
        for name, para in self.model.named_parameters():
            if para.equal(torch.zeros_like(para)):
                assert 'bias' in name or 'guard' in name or 'h0' in name or 'c0' in name

    def check_load(self):
        new_paras = list(self.model.named_parameters())
        for old_para, new_para in zip(self.old_paras, new_paras):
            if old_para.equal(new_para[1]):
                self.logger.warning('Same para at ' + new_para[0])

    def check_action2treeseq(self):
        instance = next(iter(self.train_iterator))
        action_str_lst = self.id2original(self.ACTIONS, instance.actions)
        pos_tags = self.id2original(self.POS_TAGS, instance.pos_tags)
        converted_seq = utils.action2treestr(action_str_lst, instance.raws[0], pos_tags)

        measure = scorer.Scorer()
        golden_seq = instance.raw_seq[0]
        # print ('raws =', instance.raws[0])
        # print ('converted_seq =', converted_seq)
        # print ('golden_seq =', golden_seq)

        gold_tree = parser.create_from_bracket_string(golden_seq)
        converted_tree = parser.create_from_bracket_string(converted_seq)
        ret = measure.score_trees(gold_tree, converted_tree)
        match_num = ret.matched_brackets
        gold_num = ret.gold_brackets
        pred_num = ret.test_brackets
        assert match_num == gold_num
        assert match_num == pred_num

    def unit_test(self):
        self.logger.info('Unit test')
        self.check_zero_embedding()
        if self.resume_dir:
            self.check_load()
        self.check_grad()
        self.check_action2treeseq()
        self.logger.info('Finish all unit tests')

    def run(self):
        self.pretrainedVec = load_pretrained_model(self.emb_type, self.pretrained_emb_path)
        self.set_random_seed()
        self.prepare_output_dir()
        self.get_grammar()
        self.init_fields()
        self.process_corpora()
        self.build_vocab()
        self.build_model()
        self.build_optimizer()
        self.old_paras = copy.deepcopy(list(self.model.parameters()))
        if self.resume_dir:
            self.load_model(self.resume_dir)
        self.unit_test()
        self.inference(self.dev_iterator, type_corpus='dev', tf_board=True)
        # self.training()
        # self.inference(self.test_iterator, type_corpus='test', tf_board=True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train RNNG network')
    parser.add_argument('-t', '--train_corpus', required=True, metavar='FILE', help='path to train corpus')
    parser.add_argument('--train_grammar_file', required=True, metavar='FILE')
    parser.add_argument('-s', '--save_to', required=True, metavar='DIR')
    parser.add_argument('--dev_corpus', metavar='FILE', help='path to dev corpus')
    parser.add_argument('--test_corpus', metavar='FILE')
    parser.add_argument('--emb_path', required=True, type=str, help='Pretrained word emb')
    parser.add_argument('--emb_type', type=str, default='glove', help='Type of embedding')
    parser.add_argument('--word_embedding_size', type=int, default=100, metavar='NUMBER')
    parser.add_argument('--learning_rate', type=float, default=0.05, metavar='NUMBER')
    parser.add_argument('--clip', type=float, default=10, metavar='NUMBER')
    parser.add_argument('--debug_mode', action='store_true')
    # parser.add_argument('--new_corpus', action='store_true')
    parser.add_argument('--lemma', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--max_epochs', type=int, default=1000, metavar='NUMBER')
    parser.add_argument('--patience', type=int, default=5, metavar='NUMBER')
    parser.add_argument('--resume_dir', type=str, default='')
    parser.add_argument('--exclude_word_emb', action='store_true')
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--rule_emb', action='store_true')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--id', type=int, required=True)
    parser.add_argument('--cyclic_lr', action='store_true')
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

    # if self.exclude_word_embs:
    #     self.logger.info('Excluding word embedding from pretrained model')
    #     pretrained_dict = torch.load(self.resume_file)
    #     model_dict = self.model.state_dict()
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'word_embedding' not in k}
    #     model_dict.update(pretrained_dict)
    #     self.model.load_state_dict(model_dict)
    # else:
    # pretrained_dict = torch.load(self.resume_file)
    # model_dict = self.model.state_dict()
    # exclude_lst = [k for k in pretrained_dict.keys() if pretrained_dict[k].size() != model_dict[k].size()]
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if v.size() == model_dict[k].size()}
    # print('WARNING: Excluding', exclude_lst)
    # model_dict.update(pretrained_dict)
    # self.model.load_state_dict(model_dict)
