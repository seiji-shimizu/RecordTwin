#!/bin/bash

#mode="generated" 
mode="original" 

## Example of pretraining BERT model entirely from scratch. Includes learning of a domain-specific tokenizer.
if [ "$mode" == "generated" ]; then
    model_dir="./generated/"
    device_id=0
    data_train="generated_corpus.json"
elif [ "$mode" == "original" ]; then
    model_dir="./original/"
    data_train="original_corpus.json"
    device_id=1
fi


## Example of continuing to pretrain an existing BERT model.

echo "~~~ Executing Pretraining Procedure ~~~"
CUDA_VISIBLE_DEVICES=$device_id python -u pretrain.py \
    --setting train \
    --model_dir "$model_dir" \
    --tokenizer_init google-bert/bert-base-uncased \
    --model_init google-bert/bert-base-uncased \
    --data_train "$data_train" \
    --data_eval_p 0.05 \
    --sample_random_state 42 \
    --learn_batch 16 \
    --learn_gradient_accumulation_steps 4 \
    --learn_sequence_length_data 128 \
    --learn_sequence_length_model 512 \
    --learn_sequence_overlap 0 \
    --learn_lr 0.00005 \
    --learn_mask_probability 0.15 \
    --learn_mask_max_per_sequence 10 \
    --learn_eval_strategy epoch \
    --learn_random_state 42 \
    --learn_early_stopping \
    --learn_early_stopping_patience 3 \
    --learn_early_stopping_tol 0.0 \
    --learn_save_total_limit 5 \
    --learn_epochs 10

echo "~~~ Visualizing Pretraining Loss Curves ~~~"
python -u pretrain.py \
    --setting plot \
    --model_dir "$model_dir" \
    --plot_prop 1.0
