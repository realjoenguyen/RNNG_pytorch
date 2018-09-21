from typing import List, Tuple

from torchtext.data import Example, Field

# from actions import is_nt, get_nonterm
from oracle import Oracle

def get_nonterm(a):
    assert a.startswith('NP')
    prod = a[3:-1]
    nts = [e for e in prod.split() if e.isalpha() and e != 'w']
    for e in nts:
        assert e.upper() == e
    return nts

def make_example_from_oracles(oracle: Oracle, fields: List[Tuple[str, Field]]):
    # nonterms = [get_nonterm(a) for a in oracle.actions if is_nt(a)]
    nonterms = [get_nonterm(a) for a in oracle.actions if a.startswith('NP')]
    nonterms = [item for sublist in nonterms for item in sublist]
    return Example.fromlist(
        # [oracle.actions, nonterms, oracle.pos_tags, oracle.words, oracle.raws, oracle.check_unk], fields
        [oracle.actions, nonterms, oracle.pos_tags, oracle.words, oracle.raws], fields
    )
