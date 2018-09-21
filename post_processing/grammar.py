from nltk import Tree
f = open('/home/anh/rnng_all/rnng_self/post_processing/true_seqs.txt', 'r')
grammar = set()
for line in f:
    tree = Tree.fromstring(line)
    grammar.update(tree.productions())

print (list(grammar)[0])
