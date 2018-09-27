# coding=utf-8
from timeit import default_timer as timer
import nltk
from typing import List, NamedTuple, Optional, Sequence, Sized, Tuple, Union, cast
from typing import Dict  # noqa

from nltk.tree import Tree
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from self_typing import WordId, NTId, ActionId
from production import Production

# Production = NamedTuple('Production', [('production', nltk.grammar.Production),
#                                        ('lhs', str),
#                                        ('rhs', List[str]),
#                                        ])

StackElement = NamedTuple('StackElement', [('subtree', Union[WordId, Tree]),
                                           ('production', Union[Production, str]),
                                           ('is_open_np', bool),
                                           ('emb', Variable),
                                           ])

EarlyElement = NamedTuple('EarlyElement', [('production', Production),
                                           ('next_open_nt_id', int),
                                           ])


class EmptyStackError(Exception):
    def __init__(self):
        super().__init__('stack is already empty')


class StackLSTM(nn.Module, Sized):
    BATCH_SIZE = 1
    SEQ_LEN = 1

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.):
        if input_size <= 0:
            raise ValueError('nonpositive input size: {}'.format(input_size))
        if hidden_size <= 0:
            raise ValueError('nonpositive hidden size: {}'.format(hidden_size))
        if num_layers <= 0:
            raise ValueError('nonpositive number of layers: {}'.format(num_layers))
        if dropout < 0. or dropout >= 1.:
            raise ValueError('invalid dropout rate: {}'.format(dropout))

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout)
        self.h0 = nn.Parameter(torch.Tensor(num_layers, self.BATCH_SIZE, hidden_size))
        self.c0 = nn.Parameter(torch.Tensor(num_layers, self.BATCH_SIZE, hidden_size))
        init_states = (self.h0, self.c0)
        self._states_hist = [init_states]
        self._outputs_hist = []  # type: List[Variable]
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.lstm.named_parameters():
            if name.startswith('weight'):
                init.orthogonal_(param)
            else:
                assert name.startswith('bias')
                init.constant_(param, 0.)
        init.constant_(self.h0, 0.)
        init.constant_(self.c0, 0.)

    def forward(self, inputs):
        if inputs.size() != (self.input_size,):
            raise ValueError('expected input to have size ({},), got {}'.format(self.input_size, tuple(inputs.size())))
        assert self._states_hist
        # Set seq_len and batch_size to 1
        inputs = inputs.view(self.SEQ_LEN, self.BATCH_SIZE, inputs.numel())
        next_outputs, next_states = self.lstm(inputs, self._states_hist[-1])
        self._states_hist.append(next_states)
        self._outputs_hist.append(next_outputs)
        return next_states

    def push(self, *args, **kwargs):
        return self(*args, **kwargs)

    def pop(self) -> Tuple[Variable, Variable]:
        if len(self._states_hist) > 1:
            self._outputs_hist.pop()
            return self._states_hist.pop()
        else:
            raise EmptyStackError()

    @property
    def top(self) -> Variable:
        # outputs: hidden_size
        return self._outputs_hist[-1].squeeze() if self._outputs_hist else None

    def __repr__(self) -> str:
        res = ('{}(input_size={input_size}, hidden_size={hidden_size}, '
               'num_layers={num_layers}, dropout={dropout})')
        return res.format(self.__class__.__name__, **self.__dict__)

    def __len__(self):
        return len(self._outputs_hist)


def log_softmax(inputs: Variable, restrictions: Optional[torch.LongTensor] = None) -> Variable:
    if restrictions is None:
        return F.log_softmax(inputs)

    assert restrictions.dim() == 1
    addend = Variable(inputs.data.new(inputs.size()).zero_().index_fill_(
        inputs.dim() - 1,
        restrictions,
        -float('inf')))
    return F.log_softmax(inputs + addend)


def is_scan_prod(prod: Production) -> bool:
    # return [str(e) for e in prod.rhs()] == ['<w>']
    return prod.rhs == ['<w>']


import torchtext


class DiscRNNG(nn.Module):
    MAX_OPEN_NP = 100
    REDUCE_ID = 0
    SHIFT_ID = 1

    def __init__(self,
                 num_words: int,
                 num_pos: int,
                 num_nt: int,
                 word_embedding_size,
                 pos_embedding_size,
                 nt_embedding_size,
                 action_embedding_size,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 pretrained_emb_vec,
                 productions: List[Production],
                 nonterms: torchtext.data.Field,
                 words: torchtext.data.Field,
                 ) -> None:
        super().__init__()
        self.num_words = num_words
        self.num_pos = num_pos
        self.num_nt = num_nt
        self.word_embedding_size = word_embedding_size
        self.pos_embedding_size = pos_embedding_size
        self.nt_embedding_size = nt_embedding_size
        self.action_embedding_size = action_embedding_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.productions = productions
        self.NONTERMS = nonterms
        self.WORDS = words

        # Parser states
        self._stack = []  # type: List[StackElement]
        self._buffer = []  # type: List[WordId]
        self._history = []  # type: List[ActionId]
        self._early_stack = []  # type: List[EarlyElement]
        self._num_open_np = 0

        # Embeddings
        self.word_embedding = nn.Embedding(self.num_words, self.word_embedding_size)
        self.word_embedding.weight.data.copy_(pretrained_emb_vec)

        self.pos_embedding = nn.Embedding(self.num_pos, self.pos_embedding_size)
        self.nt_embedding = nn.Embedding(self.num_nt, self.nt_embedding_size)
        self.action_embedding = nn.Embedding(self.num_actions, self.action_embedding_size)

        # Parser state encoders
        self.stack_encoder = StackLSTM(self.input_size,
                                       self.hidden_size,
                                       num_layers=self.num_layers,
                                       dropout=self.dropout)
        self.stack_guard = nn.Parameter(torch.Tensor(self.input_size))

        self.buffer_encoder = StackLSTM(self.input_size,
                                        self.hidden_size,
                                        num_layers=self.num_layers,
                                        dropout=self.dropout)
        self.buffer_guard = nn.Parameter(torch.Tensor(self.input_size))

        self.history_encoder = StackLSTM(self.input_size,
                                         self.hidden_size,
                                         num_layers=self.num_layers,
                                         dropout=self.dropout)
        self.history_guard = nn.Parameter(torch.Tensor(self.input_size))

        # Compositions
        self.fwd_composer = nn.LSTM(self.input_size,
                                    self.input_size,
                                    num_layers=self.num_layers,
                                    dropout=self.dropout)
        self.bwd_composer = nn.LSTM(self.input_size,
                                    self.input_size,
                                    num_layers=self.num_layers,
                                    dropout=self.dropout)

        # Transformations
        self.word2encoder = nn.Sequential(
            nn.Linear(self.word_embedding_size + self.pos_embedding_size, self.hidden_size),
            nn.ReLU(),
            # nn.Tanh(),
        )
        self.nt2encoder = nn.Sequential(
            nn.Linear(self.nt_embedding_size, self.hidden_size),
            nn.ReLU(),
        )
        self.action2encoder = nn.Sequential(
            nn.Linear(self.action_embedding_size, self.hidden_size),
            nn.ReLU(),
        )
        self.fwdbwd2composed = nn.Sequential(
            nn.Linear(2 * self.input_size, self.input_size),
            nn.ReLU(),
        )
        self.encoders2summary = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(3 * self.hidden_size, self.hidden_size),
            nn.Tanh(),
        )
        self.summary2actionlogprobs = nn.Linear(self.hidden_size, self.num_actions)

        # Final embeddings
        self._word_emb = {}  # type: Dict[WordId, Variable]
        self._nt_emb = {}  # type: Dict[NTId, Variable]
        self._action_emb = {}  # type: Dict[ActionId, Variable]

        self.reset_parameters()

    @property
    def num_actions(self) -> int:
        return len(self.productions) + 2

    def next_early_open_nt(self) -> str:
        assert len(self._early_stack) > 0
        top = self._early_stack[-1]  # type: EarlyElement
        assert top.next_open_nt_id < len(top.production.data)
        return top.production.rhs[top.next_open_nt_id]

    @property
    def finished(self) -> bool:
        return len(self._stack) == 1 and not self._stack[0].is_open_np \
               and len(self._buffer) == 0

    def reset_parameters(self) -> None:
        # Embeddings
        for name in 'pos nt action'.split():
            embedding = getattr(self, '{}_embedding'.format(name))
            embedding.reset_parameters()

        # Encoders
        for name in 'stack buffer history'.split():
            encoder = getattr(self, '{}_encoder'.format(name))
            encoder.reset_parameters()

        # Compositions
        for name in 'fwd bwd'.split():
            lstm = getattr(self, '{}_composer'.format(name))
            for pname, pval in lstm.named_parameters():
                if pname.startswith('weight'):
                    init.orthogonal_(pval)
                else:
                    assert pname.startswith('bias')
                    init.constant_(pval, 0.)

        # Transformations
        gain = init.calculate_gain('relu')
        for name in 'word nt action'.split():
            layer = getattr(self, '{}2encoder'.format(name))
            init.xavier_uniform_(layer[0].weight, gain=gain)
            init.constant_(layer[0].bias, 1.)

        init.xavier_uniform_(self.fwdbwd2composed[0].weight, gain=gain)
        init.constant_(self.fwdbwd2composed[0].bias, 1.)
        init.xavier_uniform_(self.encoders2summary[1].weight, gain=gain)
        init.constant_(self.encoders2summary[1].bias, 1.)
        init.xavier_uniform_(self.summary2actionlogprobs.weight)
        init.constant_(self.summary2actionlogprobs.bias, 0.)

        # Guards
        for name in 'stack buffer history'.split():
            guard = getattr(self, '{}_guard'.format(name))
            init.constant_(guard, 0.)

    def forward(self,
                instance: torchtext.data.Batch,
                words: torch.Tensor,
                pos_tags: torch.Tensor,
                actions: torch.Tensor):

        self.instance = instance
        assert words.dim() == 1
        assert words.size() == pos_tags.size()
        assert actions.dim() == 1

        self._start(words, pos_tags, actions=actions)
        # llh = 0.
        llh = []
        predicted_actions = []
        for action in actions:
            assert len(self.stack_encoder) == len(self._stack) + 1
            log_probs = self._compute_action_log_probs()
            llh.append(log_probs)
            max_action_id = torch.max(log_probs, dim=0)[1].item()
            predicted_actions.append(max_action_id)

            action_id = action.item()
            if action_id == self.SHIFT_ID:
                if self._check_shift():
                    self._shift()
                else:
                    break
            elif action_id == self.REDUCE_ID:
                if self._check_reduce():
                    # start = timer()
                    self._reduce()
                    # print ('reduce = ', timer() - start)
                else:
                    break
            else:
                if self._check_push_np():
                    # start = timer()
                    self._push_prod(self._get_prod_id(action_id))
                    # print ('push = ', timer() - start)
                else:
                    break
            self._append_history(action_id)

        llh = torch.stack(llh, dim=0)
        return llh, predicted_actions

    def decode(self, instance: torchtext.data.Batch) -> Tuple[List[ActionId], Tree]:
        self.instance = instance
        words = instance.words.view(-1)
        pos_tags = instance.pos_tags.view(-1)

        self._start(words, pos_tags)
        while not self.finished:
            log_probs = self._compute_action_log_probs()
            max_action_id = torch.max(log_probs, dim=0)[1].item()
            if max_action_id == self.SHIFT_ID:
                if self._check_shift():
                    self._shift()
                else:
                    print('Raw seq:', instance.raw_seq[0])
                    print('ERROR: shift is an illegal one\n')
                    break
            elif max_action_id == self.REDUCE_ID:
                if self._check_reduce():
                    self._reduce()
                else:
                    print('Raw seq:', instance.raw_seq[0])
                    print('ERROR: reduce is an illegal one\n')
                    break
            else:
                if self._check_push_np():
                    self._push_prod(self._get_prod_id(max_action_id))
                else:
                    print('Raw seq:', instance.raw_seq[0])
                    print('ERROR: push NP is an illegal one\n')
                    break
            self._append_history(max_action_id)
        return list(self._history), self._stack[0].subtree

    def _start(self,
               words: Variable,
               pos_tags: Variable,
               actions: Optional[Variable] = None) -> None:

        assert words.dim() == 1
        assert words.size() == pos_tags.size()
        if actions is not None:
            assert actions.dim() == 1

        self._stack = []
        self._buffer = []
        self._history = []
        self._early_stack = []
        self._num_open_np = 0

        while len(self.stack_encoder) > 0:
            self.stack_encoder.pop()
        while len(self.buffer_encoder) > 0:
            self.buffer_encoder.pop()
        while len(self.history_encoder) > 0:
            self.history_encoder.pop()

        # Feed guards as inputs
        self.stack_encoder.push(self.stack_guard)
        self.buffer_encoder.push(self.buffer_guard)
        self.history_encoder.push(self.history_guard)

        # Initialize input buffer and its LSTM encoder
        self._prepare_embeddings(words, pos_tags, actions=actions)
        for word_id in reversed(words.data.tolist()):
            self._buffer.append(word_id)
            assert word_id in self._word_emb
            self.buffer_encoder.push(self._word_emb[word_id])

    def _prepare_embeddings(self,
                            words: Variable,
                            pos_tags: Variable,
                            actions: Optional[Variable] = None) -> None:
        # words: (seq_length,)
        # pos_tags: (seq_length,)
        # actions: (action_seq_length,)

        assert words.dim() == 1
        assert words.size() == pos_tags.size()
        if actions is not None:
            assert actions.dim() == 1

        if actions is None:
            with torch.no_grad():
                actions = Variable(self._new(range(self.num_actions))).long()

        with torch.no_grad():
            nonterms = Variable(self._new(range(self.num_nt))).long()

        word_embs = self.word_embedding(
            words.view(1, -1)).view(-1, self.word_embedding_size)
        pos_embs = self.pos_embedding(
            pos_tags.view(1, -1)).view(-1, self.pos_embedding_size)
        nt_embs = self.nt_embedding(
            nonterms.view(1, -1)).view(-1, self.nt_embedding_size)
        action_embs = self.action_embedding(
            actions.view(1, -1)).view(-1, self.action_embedding_size)

        final_word_embs = self.word2encoder(torch.cat([word_embs, pos_embs], dim=1))
        final_nt_embs = self.nt2encoder(nt_embs)
        final_action_embs = self.action2encoder(action_embs)

        self._word_emb = dict(zip(words.data.tolist(), final_word_embs))
        self._nt_emb = dict(zip(nonterms.data.tolist(), final_nt_embs))
        self._action_emb = dict(zip(actions.data.tolist(), final_action_embs))

    def _compute_action_log_probs(self) -> Variable:
        assert self.stack_encoder.top is not None
        assert self.buffer_encoder.top is not None
        assert self.history_encoder.top is not None

        concatenated = torch.cat([self.stack_encoder.top,
                                  self.buffer_encoder.top,
                                  self.history_encoder.top]).view(1, -1)

        summary = self.encoders2summary(concatenated)
        illegal_actions = self._get_illegal_actions()
        return log_softmax(
            self.summary2actionlogprobs(summary),
            restrictions=illegal_actions
        ).view(-1)

    def _check_push_np(self) -> bool:
        return len(self._buffer) > 0 and self._num_open_np < self.MAX_OPEN_NP

    def _check_push_pred_np(self, action_id: int) -> bool:
        pushed_prod = self.productions[self._get_prod_id(action_id)]
        if len(self._early_stack) > 0:
            top_early = self._early_stack[-1]
            if top_early.next_open_nt_id >= len(top_early.production.data):
                # done all NT in rule
                return False
            # next predicted lhs of rule must == next open nt in early stack
            return pushed_prod.lhs == self.next_early_open_nt()
        else:
            # first rule must be TOP -> sth
            return pushed_prod.lhs == 'TOP'

    def _check_shift(self) -> bool:
        return len(self._buffer) > 0 and self._num_open_np > 0 and len(self._stack) > 0

    def _check_reduce(self) -> bool:
        tos_is_open_np = len(self._stack) > 0 and self._stack[-1].is_open_np
        old_cond = self._num_open_np > 0 and not tos_is_open_np \
                   and not (self._num_open_np < 2 and len(self._buffer) > 0) and len(self._stack) > 0

        # reduce only top early is NT -> * w or S -> A B *
        # for example REDUCE (S -> A * B) is illegal
        if len(self._early_stack) > 0:
            top = self._early_stack[-1]  # type: EarlyElement
            early_cond = is_scan_prod(top.production) or top.next_open_nt_id == len(top.production.data)
            return old_cond and early_cond
        else:
            return False

    def _append_history(self, action_id: ActionId) -> None:
        assert action_id in self._action_emb

        self._history.append(action_id)
        self.history_encoder.push(self._action_emb[action_id])

    # def _push_nt(self, nt_id: NTId) -> None:
    #     assert self._check_push_nt()
    #     assert nt_id in self._nt_emb
    #
    #     self._stack.append(StackElement(Tree(nt_id, []), self._nt_emb[nt_id], True))
    #     self.stack_encoder.push(self._nt_emb[nt_id])
    #     self._num_open_nt += 1

    def _push_prod(self, prod_id: int):
        assert self._check_push_np()

        # lhs_nt_id = self.NONTERMS.vocab.stoi[str(cur_prod.lhs())]
        # rhs_nt_ids = [self.NONTERMS.vocab.stoi[str(nt)] for nt in cur_prod.rhs()]

        cur_prod = self.productions[prod_id]
        lhs_nt_id = self.NONTERMS.vocab.stoi[cur_prod.lhs]
        rhs_nt_ids = [self.NONTERMS.vocab.stoi[nt] for nt in cur_prod.rhs]

        lhs_emb = self._nt_emb[lhs_nt_id]
        rhs_embs = [self._nt_emb[rhs_nt_id] for rhs_nt_id in rhs_nt_ids]
        composed_emb = self._compose(lhs_emb, rhs_embs)

        # self._stack.append(StackElement(Tree(lhs_nt_id, rhs_nt_ids), composed_emb, True))
        # self._stack.append(StackElement(cur_prod, composed_emb, True))
        self._stack.append(StackElement(Tree(lhs_nt_id, []), cur_prod, True, composed_emb))
        self.stack_encoder.push(composed_emb)
        self._num_open_np += 1

        # early_stack
        self._early_stack.append(EarlyElement(cur_prod, 0))

    def _shift(self) -> None:
        assert self._check_shift()
        assert len(self._buffer) > 0
        assert len(self.buffer_encoder) > 0
        assert self._buffer[-1] in self._word_emb

        word_id = self._buffer.pop()
        self.buffer_encoder.pop()
        self._stack.append(StackElement(word_id, self.WORDS.vocab.itos[word_id], False, self._word_emb[word_id], ))
        self.stack_encoder.push(self._word_emb[word_id])

    def _reduce(self) -> None:
        assert self._check_reduce()

        children = []  # type: StackElement
        while len(self._stack) > 0 and not self._stack[-1].is_open_np:
            children.append(self._stack.pop())
            self.stack_encoder.pop()

        assert len(children) > 0
        assert len(self._stack) > 0

        children.reverse()
        child_subtrees, child_prod, _, child_embs = zip(*children)
        open_np = self._stack.pop()
        self.stack_encoder.pop()

        # check if chilren label match with rhs of rule in open_nt
        # if open_np.production != None:
        if not is_scan_prod(open_np.production):
            rhs_nt = open_np.production.rhs
            label_children = [e.lhs for e in child_prod if type(e) == Production]
            assert rhs_nt == label_children

        assert isinstance(open_np.subtree, Tree)
        parent_subtree = cast(Tree, open_np.subtree)
        parent_subtree.extend(child_subtrees)
        composed_emb = self._compose(open_np.emb, child_embs)
        self._stack.append(StackElement(parent_subtree, open_np.production, False, composed_emb))
        self.stack_encoder.push(composed_emb)

        self._num_open_np -= 1
        assert self._num_open_np >= 0

        # early parsing
        top_early = self._early_stack[-1]
        self._early_stack[-1] = EarlyElement(top_early.production, top_early.next_open_nt_id + 1)
        top_early = self._early_stack[-1]

        # NT -> w *
        if is_scan_prod(top_early.production) or top_early.next_open_nt_id > len(top_early.production.data):
            # finish this rule
            self._early_stack.pop()
            if len(self._early_stack) > 0:
                top_early = self._early_stack[-1]
                self._early_stack[-1] = EarlyElement(top_early.production, top_early.next_open_nt_id + 1)

    def _compose(self, open_nt_emb: Variable, children_embs: Sequence[torch.Tensor]) -> Variable:
        assert open_nt_emb.size() == (self.input_size,)
        assert all(x.size() == (self.input_size,) for x in children_embs)

        fwd_input = [open_nt_emb]
        bwd_input = [open_nt_emb]
        for i in range(len(children_embs)):
            fwd_input.append(children_embs[i])
            bwd_input.append(children_embs[-i - 1])

        # (n_children + 1, 1, input_size)
        fwd_input = torch.stack(fwd_input).unsqueeze(1)
        bwd_input = torch.stack(bwd_input).unsqueeze(1)
        # (n_children + 1, 1, input_size)
        fwd_output, _ = self.fwd_composer(fwd_input)
        bwd_output, _ = self.bwd_composer(bwd_input)
        # (input_size,)
        fwd_emb = F.dropout(fwd_output[-1][0], p=self.dropout, training=self.training)
        bwd_emb = F.dropout(bwd_output[-1][0], p=self.dropout, training=self.training)
        # (input_size,)
        return self.fwdbwd2composed(torch.cat([fwd_emb, bwd_emb]).view(1, -1)).view(-1)

    def _get_illegal_actions(self):
        illegal_action_ids = [action_id for action_id in range(self.num_actions) if not self._is_legal(action_id)]
        if not illegal_action_ids:
            return None
        return self._new(illegal_action_ids).long()

    def _is_legal(self, action_id: int) -> bool:
        if action_id == self.SHIFT_ID:
            return self._check_shift()
        if action_id == self.REDUCE_ID:
            return self._check_reduce()
        return self._check_push_pred_np(action_id)

    # def _get_nt(self, action_id: int) -> int:
    #     assert action_id >= 2
    #     return action_id - 2

    def _get_prod_id(self, action_id: int) -> int:
        assert action_id >= 2
        return action_id - 2

    def _new(self, *args, **kwargs) -> torch.FloatTensor:
        return next(self.parameters()).data.new(*args, **kwargs)
