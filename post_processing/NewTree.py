from PYEVALB import tree
# import PYEVALB
from nltk.tree import Production, Tree

from PYEVALB import scorer
from PYEVALB import parser
import re

def no_span_prod(rule):
    rule = re.sub('\(([^\)]+)\)', '', str(rule))
    rule = re.sub(' +', ' ', rule)
    rule = rule.strip()
    return rule

# gold = '(IP (NP (PN 这里)) (VP (ADVP (AD 便)) (VP (VV 产生) (IP (NP (QP (CD 一) (CLP (M 个))) (DNP (NP (JJ 结构性)) (DEG 的)) (NP (NN 盲点))) (PU ：) (IP (VP (VV 臭味相投) (PU ，) (VV 物以类聚)))))) (PU 。))'
# test = '(IP (IP (NP (PN 这里)) (VP (ADVP (AD 便)) (VP (VV 产生) (NP (QP (CD 一) (CLP (M 个))) (DNP (ADJP (JJ 结构性)) (DEG 的)) (NP (NN 盲点)))))) (PU ：) (IP (NP (NN 臭味相投)) (PU ，) (VP (VV 物以类聚))) (PU 。))'
gold = '(TOP (S (INTJ (XX No)) (XX ,) (NP (XX it)) (VP (XX was) (XX nt) (NP (XX Black) (XX Monday))) (XX .)))'
test = '(TOP (S (ADVP (XX No) ) (XX ,) (NP (XX it) ) (VP (XX was) (XX nt) (NP (XX Black) (XX Monday) ) ) (XX .) ) )'

gold_tree = parser.create_from_bracket_string(gold)
test_tree = parser.create_from_bracket_string(test)
gold_prods, gold_heights = gold_tree.productions(skip_XX=False, skip_span=False)
test_prods, test_heights = test_tree.productions(skip_XX=False, skip_span=False)
gold_nltk_tree = Tree.fromstring(gold).pretty_print()
test_nltk_tree = Tree.fromstring(test).pretty_print()

print (gold_prods)
print (list(map(no_span_prod, gold_prods)))
print (gold_heights)
print (test_prods)
print (test_heights)
print ('Substract = ', set(gold_prods) - set(test_prods))

scorer = scorer.Scorer()
result = scorer.score_trees(gold_tree, test_tree)
print (result)
# print (gold_tree.non_terminal_labels)
# first = gold_tree.non_terminal_labels[0]
# print (first.children)
# print (gold_tree.root, first)
# print (all_labels.value)
# print (test_tree.non_terminals)