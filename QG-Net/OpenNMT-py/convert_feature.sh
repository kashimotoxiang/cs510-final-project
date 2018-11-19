#!/usr/bin/env bash

i=1 # number of sentences in a training context

# call OpenNMT preprocess routine to output files that OpenNMT can take
python3 preprocess.py \
-train_src ../data/info_retriaval/docs.txt \
-train_tgt ../data/info_retriaval/questions.txt \
-valid_src ../data/info_retriaval/tmp1.txt \
-valid_tgt ../data/info_retriaval/tmp2.txt \
-save_data ../data/data.feat.${i}sent \
-src_vocab_size 100000 -tgt_vocab_size 100000 \
-src_seq_length 10000 -tgt_seq_length 10000 \
-dynamic_dict

echo "PREPROCESSING ALL DONE!!!!!!"

