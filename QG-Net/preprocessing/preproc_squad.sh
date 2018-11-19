#!/usr/bin/env bash

i=0 # number of sentences in a training context

# process training set
python3 preproc_squad.py \
-data_dir ../preprocessing/ \
-out_dir ../test/input.for.test \
-split test \
-corenlp_path ../data/corenlp \
-num_sents $i
