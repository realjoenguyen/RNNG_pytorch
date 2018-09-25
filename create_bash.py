import numpy as np

def get_sample(size, low, high, discrete=False):
    if discrete:
        return np.random.randint(low, high + 1, size)
    else:
        return np.random.uniform(low, high, size)

num_experiments = 5
# word_emb = get_sample(num_experiments,)

# word_emb = np.random.choice([100, 200, 300], size=num_experiments)
word_emb = np.random.choice([100], size=num_experiments)
emb_type = np.random.choice(['glove', 'sskip'], size=num_experiments)
use_unk = np.random.choice([True, False], size=num_experiments)
use_lemma = np.random.choice([True, False], size=num_experiments)
learning_rate = get_sample(num_experiments, low=-3.04, high=-3) #from 0.0001 to 0.002
clip = get_sample(num_experiments, low=15, high=40, discrete=True)
prefix = 'python3.5 ./trainer.py --train-corpus ${TRAIN_ORACLE} --save-to ${SAVE_TO} --dev-corpus ${DEV_ORACLE} --emb-path ${WORD_EMBEM} --test-corpus ${TEST_ORACLE} --new-corpus --cuda '
for i in range(num_experiments):
    suffix = ('--emb-type {} --learning-rate {} --clip {} --word-embedding-size {}'.format(
        emb_type[i],
        10**learning_rate[i],
        clip[i],
        word_emb[i],
        i
    ))
    if use_lemma[i]:
        suffix += ' --lemma'
    if use_unk[i]:
        suffix += ' --use-unk'
    hper = 'unk={};new={};emb_type={};lemma={};lr={:.4f};word={};clip={}'.format(
        # small_corpus,
        use_unk[i],
        True,
        emb_type[i],
        use_lemma[i],
        10**learning_rate[i],
        word_emb[i],
        # pos_embedding_size,
        # action_embedding_size,
        # dropout,
        # hidden_size,
        clip[i],
    )

    suffix += ' > "./logs/' + hper + '"'

    print (prefix + suffix + ' &')