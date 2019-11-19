import os
import time
import pandas as pd
runtimes = 2
mem_size = 16.0


def run_swap_mimem(batch_size):
    runtime = dict()
    required_saving = 5
    frac = 8
    while(True):
        command = " export TF_EXTRA_REQUIRED_SAVINGS=-%s000000000 && \
                    CUDA_VISIBLE_DEVICES=0 python run_pretraining.py \
                    --input_file=./output/tf_examples.tfrecord \
                    --output_dir=./output/pretraining_output \
                    --memory_optimization=SWAPPING_HEURISTICS \
                    --gpu_memory_frac_for_testing=%f \
                    --do_train=True \
                    --do_eval=False \
                    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
                    --train_batch_size=%d \
                    --max_seq_length=128 \
                    --max_predictions_per_seq=20 \
                    --num_train_steps=20 \
                    --num_warmup_steps=0 \
                    --learning_rate=2e-5  > output/swap_mem/%d_%d_%d.txt 2>&1" % (required_saving, frac/mem_size, batch_size, batch_size, frac, required_saving)
        clear = "rm -rf ./output/pretraining_output/*"
        if os.system(clear):
            print "CLEAR ERROR"
            return None, None
        for i in range(runtimes):
            start = time.time()
            rst = os.system(command)
            end = time.time()
            if rst == 0:
                runtime[batch_size] = end-start
                return frac, runtime
        if rst:
            required_saving += 1
            if required_saving > 16:
                frac+=1
                required_saving = 5
                if frac > mem_size:
                    return None, None
            


def run_recomp_minmem(batch_size):
    runtime = dict()
    frac = 8
    while(True):
        command = " CUDA_VISIBLE_DEVICES=0 python run_pretraining.py \
                    --input_file=./output/tf_examples.tfrecord \
                    --output_dir=./output/pretraining_output \
                    --memory_optimization=RECOMPUTATION_HEURISTICS \
                    --gpu_memory_frac_for_testing=%f \
                    --do_train=True \
                    --do_eval=False \
                    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
                    --train_batch_size=%d \
                    --max_seq_length=128 \
                    --max_predictions_per_seq=20 \
                    --num_train_steps=20 \
                    --num_warmup_steps=0 \
                    --learning_rate=2e-5  > output/recomp_mem/%d_%d.txt 2>&1" % (frac/mem_size, batch_size, batch_size, frac)
        clear = "rm -rf ./output/pretraining_output/*"
        if os.system(clear):
            print "CLEAR ERROR"
            return None, None
        for i in range(runtimes):
            start = time.time()
            rst = os.system(command)
            end = time.time()
            if rst == 0:
                runtime[batch_size] = end-start
                return frac, runtime
        if rst:
            frac += 1
            if frac > mem_size:
                return None, None

if __name__ == "__main__":
#    frac, runtime = run_swap_mimem(66)
    frac, runtime = run_recomp_minmem(66)
    print frac
    print runtime
