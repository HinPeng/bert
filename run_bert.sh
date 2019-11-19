#!/bin/bash

export BERT_BASE_DIR=/home/frog/maweiliang/models/bert-master/bert_models/bert_base_uncase/
# export BERT_LARGE_DIR=/home/frog/maweiliang/models/bert-master/bert_models/bert_large_uncase/
rm -r ./output/pretraining_output
# export TF_LOG_TENSOR_INFO=true
export TF_MODEL_NUM_NODES=7349 # bs=70
# export TF_MODEL_NUM_NODES=7391
# export TF_MODEL_NUM_NODES=7400
# export TF_MODEL_NUM_NODES=14033

# export SWAP_POLICY_FILE=/mnt/maweiliang/models/bert/swapping_decision.log
# export SWAP_POLICY_FILE=/mnt/maweiliang/work/TFCompGraphSim/bert_66_p100/swapping_decision.log
# export SWAP_POLICY_FILE=/mnt/maweiliang/work/TFCompGraphSim/bert_66_p100/swapping_decision_ondemand.log
# export SWAP_POLICY_FILE=/mnt/xq/TFCompGraphSim_backup/bert_66_p100/swapping_decision_ondemand_16.log
# export SWAP_POLICY_FILE=/mnt/vfonel/backup/TFCompGraphSim/bert_66_p100/swapping_decision.log

export RECOMPUTE_POLICY_FILE=/home/frog/maweiliang/work/TFCompGraphSim/bert_66_p100/recompute_400.log
# export RECOMPUTE_POLICY_FILE=/home/frog/maweiliang/tmp/TFCompGraphSim/bertlarge_8_p100/recompute.log
# export RECOMPUTE_POLICY_FILE=/home/frog/maweiliang/tmp/TFCompGraphSim/bertlarge_8_p100/recompute_only_once.log
# export RECOMPUTE_POLICY_FILE=/mnt/maweiliang/work/TFCompGraphSim/bert_66_p100/recompute_d0_70.log

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
#  --lognode_time=True \
#  --openai_opt=True \
#  --memory_optimization=SWAPPING_HEURISTICS
