#!/usr/bin/env bash
#TRAIN_ORACLE=./data/oracle/train_small.oracle
#DEV_ORACLE=./data/oracle/dev_small.oracle
#TEST_ORACLE=./data/oracle/test.oracle

TRAIN_ORACLE=./data/oracle_new/train.oracle
DEV_ORACLE=./data/oracle_new/dev.oracle
TEST_ORACLE=./data/oracle_new/test.oracle

TRAIN_ORIGIN=./data/PTB/train.txt
TEST_ORGIN=./data/PTB/test.txt
DEV_ORGIN=./data/PTB/dev.txt
WORD_EMBEM=/home/anh/embedding/

SAVE_TO=./res/ 

#source ./config_path.sh
#rm -rf ${SAVE_TO}/*

python3.5 ./trainer.py \
--rule_emb \
--cuda \
--debug_mode \
--max_epochs 1000 --id 3 \
--train_corpus ./data/oracle_rule/train_rule.oracle \
--use_cache --lemma --save_to ./res/ --emb_path /home/anh/embedding/ --emb_type glove --word_embedding_size 100 \
--train_grammar_file ./data/train_grammar.txt \
--dev_corpus ./data/oracle_rule/dev_rule.oracle --test_corpus ./data/oracle_rule/test_rule.oracle \
--learning_rate 0.001 --clip 10 --optimizer adam \
--resume_dir './res/id=1;optimizer=adam;unk=True;emb_type=glove;lemma=True;lr=0.0010;word=100;clip=10.0/saved_model/'

#--use-cache --exclude_word_embs \
#--lemma \
#--max-epochs 1000 \
#--id 0 \
#--train-corpus ./data/oracle_new/train.oracle --dev-corpus ./data/oracle_new/dev.oracle --test-corpus ./data/oracle_new/test.oracle \
#--save-to ./res/ --emb-path /home/anh/embedding/ \
#--cuda --emb-type glove --learning-rate 0.001 --clip 20 --word-embedding-size 100 --optimizer adam --debug_mode \
#--resume_file '/home/anh/rnng_all/rnng_self/res/optimizer=adam;unk=True;new=True;emb_type=glove;lemma=False;lr=0.0010;word=100;clip=40.0/saved_model'