# echo 'Jim Henson was a puppeteer' > ./tmp/input.txt

python extract_features.py \
  --output_file=./tmp/ \
  --input_file=./tmp/input.txt \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1,-2,-3,-4\
  --max_seq_length=128 \
  --batch_size=8