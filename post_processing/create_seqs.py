
from nltk.tree import *
from nltk.grammar import *

def get_seqs(raw_oracle_file, save_to):
    # if seqs==False:
    # raw_oracle_file = '/home/anh/rnng_all/rnng_self/data/oracle_new/dev.oracle'
    f_read = open(raw_oracle_file, 'r')

    raw_seqs = []
    pos_tokens = []
    raw_tokens = []
    unk_tokens = []
    line = f_read.readline()
    while line:
        if line.startswith('# '):
            raw_seqs.append(line[2:])
            # true_seqs.append(line)
            pos = f_read.readline()
            pos_tokens.append(pos.split())
            raw = f_read.readline()
            raw_tokens.append(raw.split())

            lower = f_read.readline()
            unk = f_read.readline()
            unk_tokens.append(unk.split())
            line = f_read.readline()
        else:
            line = f_read.readline()

    f_write = open(save_to, 'w')
    seqs = []
    for id, line in enumerate(raw_seqs):
        for raw, unk, pos_tag in zip(raw_tokens[id], unk_tokens[id], pos_tokens[id]):
            # if raw != unk:
            line = line.replace('(' + pos_tag + ' ' + raw + ')', '(XX ' + unk + ')')
            # line = line.replace('(' + pos_tag + ' ' + raw + ')', '')
        # print (train_line)

        f_write.write(line)
        seqs.append(line)
    # print (seqs[:10])
    assert seqs != []
    return seqs

get_seqs('/home/ta/Projects/RNNG_all/rnng_self/data/oracle_new/dev.oracle', 'dev_seqs.txt')
get_seqs('/home/ta/Projects/RNNG_all/rnng_self/data/oracle_new/test.oracle', 'test_seqs.txt')
get_seqs('/home/ta/Projects/RNNG_all/rnng_self/data/oracle_new/train.oracle', 'train_seqs.txt')
# get_seqs('/home/anh/rnng_all/rnng_self/data/pred_seqs.txt', 'pred_seqs.txt')

from nltk.tree import *

def get_grammar(seq_file):
    grammar = set()
    f = open(seq_file, 'r')
    seqs = f.readlines()
    print ('Len =', len(seqs))

    # for id, dev_line in enumerate(seqs):
    #     for raw, unk, pos_tag in zip(raw_tokens[id], unk_tokens[id], pos_tokens[id]):
    #         if raw != unk:
            # dev_line = dev_line.replace('(' + pos_tag + ' ' + raw + ')', '(XX ' + unk + ')')
        # print (train_line)
       # f_write.write(dev_line)
    for line in seqs:
        tree = Tree.fromstring(line)
        grammar.update(tree.productions())
    return grammar

