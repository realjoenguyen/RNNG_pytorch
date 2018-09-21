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
#1
python3.5 ./trainer.py --use-cache --exclude_word_embs --lemma --max-epochs 1000 --id 4 --train-corpus ./data/oracle_new/train.oracle --dev-corpus ./data/oracle_new/dev.oracle --test-corpus ./data/oracle_new/test.oracle --save-to ./res/ --emb-path /home/anh/embedding/ --new-corpus --cuda --emb-type glove --learning-rate 0.001 --clip 20 --word-embedding-size 100 --optimizer adam --debug_mode --resume_file '/home/anh/rnng_all/rnng_self/res/optimizer=adam;unk=True;new=True;emb_type=glove;lemma=False;lr=0.0010;word=100;clip=40.0/saved_model'

