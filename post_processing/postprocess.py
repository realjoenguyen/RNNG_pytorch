import sys
from collections import Counter, defaultdict

from PYEVALB import scorer
from PYEVALB import parser, tree

# pred_seqs_file = '/home/anh/rnng_all/rnng_self/data/post_processing/pred_seqs.txt'
# test_seqs_file = '/home/anh/rnng_all/rnng_self/data/post_processing/test_seqs.txt'
# train_seqs_file = '/home/anh/rnng_all/rnng_self/data/post_processing/train_seqs.txt'
# dev_seqs_file = '/home/anh/rnng_all/rnng_self/data/post_processing/dev_seqs.txt'
pred_seqs_file = '/home/ta/Projects/RNNG_all/rnng_self/data/post_processing/pred_seqs.txt'
test_seqs_file = '/home/ta/Projects/RNNG_all/rnng_self/data/post_processing/test_seqs.txt'
train_seqs_file = '/home/ta/Projects/RNNG_all/rnng_self/data/post_processing/train_seqs.txt'
dev_seqs_file = '/home/ta/Projects/RNNG_all/rnng_self/data/post_processing/dev_seqs.txt'
test_seqs = open(test_seqs_file, 'r').readlines()
pred_seqs = open(pred_seqs_file, 'r').readlines()

from nltk.tree import *
import re

# SKIP_XX = True

def no_X_grammar(grammar):
    no_X_grammar = []
    for rule in grammar:
        # print (rule)
        if str(rule.lhs()) == 'XX':
            # print ('XX -> sth = ', rule)
            continue
        rule = str(rule).replace('XX', '')
        rule = re.sub(' +', ' ', rule)
        rule = rule.strip()
        no_X_grammar.append(rule)
    return set(no_X_grammar)

def no_span_prod(rule):
    rule = re.sub('\(([^\)]+)\)', '', str(rule))
    rule = re.sub(' +', ' ', rule)
    rule = rule.strip()
    return rule

def get_diff_prods_no_span():
    print ('Getting diff between', test_seqs_file, 'and', pred_seqs_file)
    diff = set()
    id = 0
    from collections import Counter
    diff_prods_counter = Counter()
    diff_heights = defaultdict(list)

    for test_line, pred_line in zip(test_seqs, pred_seqs):
        # print ('true =', true_line)
        # print ('pred =', pred_line)
        measure = scorer.Scorer()
        gold_tree = parser.create_from_bracket_string(test_line)
        pred_tree = parser.create_from_bracket_string(pred_line)

        # print (id)
        # print(test_line, pred_line)
        # print (gold_tree.sentence)
        # print (pred_tree.sentence)
        # id += 1
        ret = measure.score_trees(gold_tree, pred_tree)
        match_num = ret.matched_brackets
        gold_num = ret.gold_brackets
        pred_num = ret.test_brackets

        if match_num < gold_num or match_num < pred_num:
            pred_grammar, pred_heights = gold_tree.productions(skip_XX=False, skip_span=False)
            true_grammar, _ = pred_tree.productions(skip_XX=False, skip_span=False)

            # print(pred_grammar)
            # print(true_grammar)
            # diff_prods = set(pred_grammar) - set(true_grammar)
            diff_prods = []
            diff_prods_heights = []
            for id, prod in enumerate(pred_grammar):
                if prod not in true_grammar:
                    diff_prods.append(prod)
                    diff_prods_heights.append(pred_heights[id])

            for id, prod in enumerate(diff_prods):
                diff_heights[no_span_prod(prod)].append(diff_prods_heights[id])
                # if pred_heights[id] == 0:
                    # print (test_line)
                    # print (pred_line)
                    # print ('Height 0 =', prod, no_span_prod(prod))
                    # sys.exit(0)

            diff_no_span_prods = set([no_span_prod(prod) for prod in diff_prods])
            diff.update(diff_no_span_prods)
            diff_prods_counter.update(diff_no_span_prods)

            # pred_tree_nltk.pretty_print()
            # true_tree_nltk.pretty_print()

    # diff_rule_count = dict([e for e in pred_rule_count.items() if e[0] in diff])
    # print ('Wrong rules')
    # print (diff_rule_count)
    # print ('Len wrong rules = ', len(diff))
    # assert len(diff) == len(diff_rule_count)

    print (diff_prods_counter.most_common(10))
    print ('There are', len(diff), 'different distint productions')
    print ('Done')
    print ('')
    return diff, diff_prods_counter, diff_heights

import re

def get_grammar(seq_file, no_XX):
    def no_X_rule(rule):
        if str(rule.lhs()) == 'XX':
            # print ('XX -> sth = ', rule)
            return None
        # rule = str(rule).replace('XX', '')
        rule = re.sub('XX', '', str(rule))
        rule = re.sub(' +', ' ', rule)
        rule = rule.strip()
        return rule

    grammar = set()
    f = open(seq_file, 'r')
    seqs = f.readlines()
    print ('Len of', seq_file, '=', len(seqs))
    print ('Getting grammar...')
    rule_counter = Counter()
    for line in seqs:
        tree = Tree.fromstring(line)
        rules = tree.productions()
        grammar.update(rules)
        if no_XX:
            no_X_rules = [no_X_rule(rule) for rule in tree.productions()]
            no_X_rules = [rule for rule in no_X_rules if rule is not None]
            rules = no_X_rules
        rule_counter.update(rules)

    if no_XX:
        assert len(no_X_grammar(grammar)) == len(rule_counter)
        return no_X_grammar(grammar), rule_counter
    else:
        return grammar, rule_counter

def get_grammar_from_file_new(seq_file):
    def no_quote_prod(prod):
        prod = re.sub('\'', '', str(prod))
        prod = re.sub(' +', ' ', prod)
        return prod.strip()

    print ('Getting grammar from', seq_file)
    f = open(seq_file, 'r')
    # grammar = None 
    prod_counter = Counter() 
    prods = []
    cnt_line = 0
    for seq in f:
        cnt_line += 1
        tree = parser.create_from_bracket_string(seq)
        this_seq_prods, _ = tree.productions(skip_XX=True, skip_span=True)
        this_seq_prods = [no_quote_prod(prod) for prod in this_seq_prods]
        this_seq_prods = [prod for prod in this_seq_prods if 'XX ->' not in prod]
        prods.extend(this_seq_prods) 
        prod_counter.update(this_seq_prods)

    print ('Done at', cnt_line, 'lines.')
    print ('There are', len(set(prods)), 'productions')
    print ('Top grammar:', prod_counter.most_common(10))
    print ('')
    return set(prods), prod_counter

# train_grammar, _ = get_grammar(train_seqs_file)
# train_grammar.update(get_grammar(dev_seqs_file)[0])
# test_grammar, test_counter = get_grammar(test_seqs_file)
train_grammar, _ = get_grammar_from_file_new(train_seqs_file)
train_grammar_file = '/home/ta/Projects/RNNG_all/rnng_self/post_processing/train_grammar.txt'
print ('Write train grammmar into', train_grammar_file)
f = open(train_grammar_file, 'w')
for prod in train_grammar:
   f.write(prod + '\n')

train_grammar.update(get_grammar_from_file_new(dev_seqs_file)[0])
test_grammar, test_counter = get_grammar_from_file_new(test_seqs_file)
diff_no_span, diff_counter, diff_heights = get_diff_prods_no_span()
print ('len diff =', len(diff_no_span))
print (diff_counter.most_common(10))
print ('')
# print ('diff_heights = ', diff_heights)

wrong_not_in_train = diff_no_span - train_grammar
print ('wrong grammar - train =', len(wrong_not_in_train))
wrong_grammar_counter = ([e for e in diff_counter.items() if e[0] in wrong_not_in_train])
assert len(wrong_grammar_counter) == len(wrong_not_in_train)
print (sorted(wrong_grammar_counter, key=lambda x:x[1], reverse=True))

print ('height for each prod')
diff_h_not_in_train_counter = Counter()
for prod in wrong_not_in_train:
    diff_h_not_in_train_counter.update(diff_heights[prod])
print ('heights wrong rule - train = ', diff_h_not_in_train_counter.most_common())
diff_h_counter = Counter()
for prod in diff_no_span:
    diff_h_counter.update(diff_heights[prod])
print ('heights wrong rule:', diff_h_counter)
record = []
for k, v in diff_h_counter.items():
    record.append((k, v, diff_h_not_in_train_counter[k], diff_h_not_in_train_counter[k] / v))
print (sorted(record, key=lambda x:x[3], reverse=True))

test_not_in_train = test_grammar - train_grammar
print('test - train grammar =', len(test_not_in_train))
test_not_counter = ([e for e in test_counter.items() if e[0] in test_not_in_train])
assert len(test_not_counter) == len(test_not_in_train)
print (sorted(test_not_counter, key=lambda x:x[1], reverse=True))