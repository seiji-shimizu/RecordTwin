#!/bin/bash
n=1

max_length=350
dataset_dir="chunked/${max_length}"
model_name="emilyalsentzer/Bio_ClinicalBERT"
# Ensure task_type is set correctly; only one task type should be set at a time
task_type="i2b2_2012"
# task_type="i2b2_2010"  # Comment out the unused task type

if [ "$task_type" == "i2b2_2012" ]; then
    train_file="${dataset_dir}/i2b2_2012/train.json"
    dev_file="${dataset_dir}/i2b2_2012/dev.json"
    test_file="${dataset_dir}/i2b2_2012/test.json"
    echo "i2b2_2012"
    
elif [ "$task_type" == "i2b2_2010" ]; then
    train_file="${dataset_dir}/i2b2_2010/train.json"
    dev_file="${dataset_dir}/i2b2_2010/dev.json"
    test_file="${dataset_dir}/i2b2_2010/test.json"
    echo "i2b2_2010"
else
    echo "wrong task type: $task_type"
    exit 1  # Use exit instead of continue since there is no enclosing loop
fi

for ((i=1; i<=n; i++)); do
    output_path="output/${max_length}/${task_type}/seed_${i}"
    seed=$i
    
    # Training script
    CUDA_VISIBLE_DEVICES=0 \
    python run_ner_finetunning.py \
      --model_name_or_path "$model_name" \
      --num_train_epochs 4 \
      --do_train \
      --do_eval \
      --do_predict \
      --evaluation_strategy epoch \
      --train_file "$train_file" \
      --validation_file "$dev_file" \
      --test_file "$test_file" \
      --per_device_train_batch_size 4 \
      --per_device_eval_batch_size 8 \
      --output_dir "$output_path" \
      --learning_rate 2e-5 \
      --fp16 \
      --overwrite_cache \
      --overwrite_output_dir \
      --max_length $max_length
done
