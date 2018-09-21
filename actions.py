from self_typing import Action, NTLabel, Word

# REDUCE: Action = 'REDUCE'
# SHIFT: Action = 'SHIFT'

REDUCE = 'REDUCE'
SHIFT ='SHIFT'

# def NT(label: NTLabel) -> Action:
#     return 'NT({})'.format(label)
#
# def GEN(word: Word) -> Action:
#     return 'GEN({})'.format(word)
#
# def get_nonterm(action: Action) -> NTLabel:
#     if action.startswith('NT(') and action.endswith(')'):
#         return action[3:-1]
#     raise ValueError('action {} is not an NT action'.format(action))
#
# def get_word(action: Action) -> Word:
#     if action.startswith('GEN(') and action.endswith(')'):
#         return action[4:-1]
#     raise ValueError('action {} is not a GEN action'.format(action))
#
# def is_nt(action: Action) -> bool:
#     return action.startswith('NT')
#
# def is_gen(action: Action) -> bool:
#     return action.startswith('GEN')

