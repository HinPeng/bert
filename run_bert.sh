#!/bin/bash

export BERT_BASE_DIR=$HOME/px/bert/bert_base_uncase/
# export BERT_LARGE_DIR=/home/frog/maweiliang/models/bert-master/bert_models/bert_large_uncase/
rm -r ./output/pretraining_output

CUDA_VISIBLE_DEVICES=$1 \
python run_pretraining.py \
  --input_file=./output/tf_examples.tfrecord \
  --output_dir=./output/pretraining_output \
  --do_train=True \
  --do_eval=False \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --train_batch_size=$2 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5 \
  --save_checkpoints_steps=0 \
  --allow_growth=False \
  --master="grpc://localhost:29999"
  --build_cost_model=100 \
  --build_cost_model_after=20
