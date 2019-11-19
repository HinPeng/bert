
# export TF_LOG_TENSOR_ACCESS=true

# export TF_CUDNN_USE_AUTOTUNE=0

export BERT_BASE_DIR=$HOME/vfonel/bert/bert_base_uncase
export TF_MODEL_NUM_NODES=7244

# export SWAP_POLICY_FILE=/home/frog/maweiliang/tmp/swap_policy.txt
# export SWAP_POLICY_FILE=/home/frog/maweiliang/models/bert/swap_blank.txt
# export SWAP_POLICY_FILE=/vpublic01/frog/vfonel/TFCompGraphSim/bert_66_p100/swapping_decision.log
# export SWAP_POLICY_FILE=/vpublic01/frog/vfonel/TFCompGraphSim/bert_66_p100/swap_blank.log
# export SWAP_POLICY_FILE=/home/frog/maweiliang/TFCompGraphSim/bert_66_p100/r_swap.log
# export RECOMPUTE_POLICY_FILE=/home/frog/maweiliang/tmp/recompute_policy.txt

rm -r ./output/pretraining_output/*

CUDA_VISIBLE_DEVICES=$1 python run_pretraining.py \
    --input_file=./output/tf_examples.tfrecord \
    --output_dir=./output/pretraining_output \
    --memory_optimization=NO_MEM_OPT \
    --do_train=True \
    --do_eval=False \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --train_batch_size=$2 \
    --save_checkpoints_steps=0 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --num_train_steps=30 \
    --num_warmup_steps=10 \
    --learning_rate=2e-5
