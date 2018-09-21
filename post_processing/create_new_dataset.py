import re
from nltk.tree import *
corpus = '/home/ta/Projects/RNNG_all/rnng_self/data/oracle_new/test.oracle'
seqs_file = '/home/ta/Projects/RNNG_all/rnng_self/data/post_processing/test_seqs.txt'
# new_corpus_file = '/home/ta/Projects/RNNG_all/rnng_self/data/oracle_rule/train_rule.oracle'
# f_write = open(new_corpus_file, 'w')
f_seqs = open(seqs_file, 'r')

f = open(corpus, 'r')
line = f.readline()

def NP(production):
    return 'NP(' + production + ')'

def NP_word(nonterm):
    return 'NP(' + nonterm + ' -> w)'

SHIFT = 'SHIFT'
REDUCE = 'REDUCE'

def get_no_XX_rule(prod):
    prod = str(prod).replace('XX', '')
    prod = re.sub(' +', ' ', prod)
    prod = prod.strip()
    return prod

def get_new_rule_lst(tree):
    # pass
    if tree.label() == 'XX':
        return [SHIFT]

    res = []
    root_prod = tree.productions()[0]
    no_XX_rule = get_no_XX_rule(root_prod)
    if not no_XX_rule.endswith('->'):
        res.append(NP(no_XX_rule))

    if all([tree[id].label() == 'XX' for id in range(len(tree))]):
        res.append(NP_word(tree.label()))

    for id in range(len(tree)):
        res.extend(get_new_rule_lst(tree[id]))

    res.append(REDUCE)
    return res

singletons = open('/home/ta/Projects/RNNG_all/rnng_self/data/singletons.txt', 'r').readlines()
singletons = [e[:-1].lower() for e in singletons]
print (singletons[:5])

while line:
    assert line.startswith('# ')
    # f_write.write(line)

    raw = line[2:].strip()
    pos_tag = f.readline().strip()
    token = f.readline().strip()
    lower = f.readline().strip()
    unks = f.readline().strip()
    # f_write.write(pos_tag + '\n')
    # f_write.write(token + '\n')
    # f_write.write(unks + '\n')

    # get action seqs
    actions = []
    while True:
        line = f.readline().strip()
        if line == '':
            break
        assert 'NT' in line or line == 'SHIFT' or line == 'REDUCE'
        actions.append(line)

    raw_seq_XX = f_seqs.readline()
    # raw_seq_XX = '(TOP (S (NP (A (XX a) (XX b)) (B (XX c))) (VP (XX d)) (XX !)))'
    tree = Tree.fromstring(raw_seq_XX)
    # print (tree.pretty_print())
    # print (tree.productions())
    # print (get_new_rule_lst(tree))
    # for action in get_new_rule_lst(tree):
    #     f_write.write(action + '\n')
    # f_write.write('\n')

    # print (unks)
    # for id, unk_token in enumerate(unks.split()):
    #     if unk_token.startswith('UNK'):
            # print(token.split()[id])
            # singletons.append(token.split()[id])

    for id, unk_token in enumerate(unks.split()):
        if unk_token.startswith('UNK'):
            if token.split()[id].lower() in singletons:
                print (token.split()[id])

    line = f.readline()
    while line == '\n': line = f.readline()
    # break

print ('Done')
# print('Done', new_corpus_file)
# pickle.dump(singletons, open('singletons', 'w'))
# torch.save(singletons, 'singletons'