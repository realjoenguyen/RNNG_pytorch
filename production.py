# coding=utf-8

from nltk.tree import Tree
from nltk.grammar import Production
import nltk
from nltk.grammar import Nonterminal
train_grammar_file = './data/train_grammar.txt'

# class Production(nltk.grammar.Production):
#
#     @staticmethod
#     def fromstring(str):
#         prod_split = str.partition('->')
#         lhs = Nonterminal(prod_split[0].strip())
#         rhs = [Nonterminal(e.strip()) for e in prod_split[2].split()]
#         return super().__init__(lhs, rhs)
#
#     @staticmethod
#     def get_train_productions(train_grammar_file):
#         productions = open(train_grammar_file, 'r').readlines()
#         productions = [e.replace('\n', '') for e in productions]
#         productions = [Production.fromstring(e) for e in productions]
#         return productions
        # print('len = ', len(productions))
        # nonterms = []
        # for prod in productions:
        #     prod_split = prod.partition('->')
        #     nonterms.append(prod_split[0].strip())
        #     nonterms.extend([e.strip() for e in prod_split[2].split()])
        # for nt in set(nonterms):
        #     print(nt + ' -> w')
        # return productions
# print (Production.get_train_productions(train_grammar_file))

def str2production(str):
    prod_split = str.partition('->')
    lhs = Nonterminal(prod_split[0].strip())
    rhs = [Nonterminal(e.strip()) for e in prod_split[2].split()]
    # return super().__init__(lhs, rhs)
    return Production(lhs, rhs)

def get_productions_from_file(train_grammar_file):
    productions = open(train_grammar_file, 'r').readlines()
    productions = [e.replace('\n', '') for e in productions]
    productions = [str2production(e) for e in productions]
    return productions
