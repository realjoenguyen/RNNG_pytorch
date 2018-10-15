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


#python3.5 ./trainer.py \
#--exclude_word_emb \
#--cuda \
#--debug_mode \
#--max_epochs 1000 --id 22 \
#--cyclic_lr \
#\
#--train_corpus ./data/oracle_rule/train_rule.oracle \
#--use_cache \
#--save_to ./res/ --emb_path /home/anh/embedding/ --emb_type glove --word_embedding_size 100 \
#--train_grammar_file ./data/train_grammar.txt \
#--dev_corpus ./data/oracle_rule/dev_rule.oracle --test_corpus ./data/oracle_rule/test_rule.oracle \
#--learning_rate 0.01 --clip 10 --optimizer SGD \
#--resume_dir '/home/anh/rnng_self/res/id=20;rule_emb=False;optimizer=SGD;unk=True;emb_type=glove;lemma=True;lr=0.0100;word=100;clip=10.0/saved_model'

python3.5 ./trainer.py \
--exclude_word_emb \
--cuda \
--debug_mode \
--max_epochs 1000 --id 24 \
--cyclic_lr \
--lemma \
--train_corpus ./data/oracle_rule/train_rule.oracle \
--use_cache \
--save_to ./res/ --emb_path /home/anh/embedding/ --emb_type glove --word_embedding_size 100 \
--train_grammar_file ./data/train_grammar.txt \
--dev_corpus ./data/oracle_rule/dev_rule.oracle --test_corpus ./data/oracle_rule/test_rule.oracle \
--learning_rate 0.01 --clip 10 --optimizer SGD \
--resume_dir '/home/anh/rnng_self/res/id=20;rule_emb=False;optimizer=SGD;unk=True;emb_type=glove;lemma=True;lr=0.0100;word=100;clip=10.0/saved_model'


