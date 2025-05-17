#!/bin/bash

# This script is used to pretrain a En-Zh text translation model. We use the DoTA dataset for convinience.
base_dir=/path/to/M4Doc

trans_model_dir=$base_dir/models/trans_model

en_tokenizer_dir=$base_dir/utils/en_tokenizer
zh_tokenizer_dir=$base_dir/utils/zh_tokenizer

en_mmd_dir=$base_dir/DoTA_dataset/en_mmd
zh_mmd_dir=$base_dir/DoTA_dataset/zh_mmd

split_json_file_path=$base_dir/DoTA_dataset/split_dataset.json

deepspeed_config_file_path=$base_dir/utils/deepspeed_zero0.json

export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --num_processes 4 \
    --num_machines 1 codes/pretrain_trans.py \
    --en_tokenizer_dir $en_tokenizer_dir \
    --zh_tokenizer_dir $zh_tokenizer_dir \
    --en_mmd_dir $en_mmd_dir \
    --zh_mmd_dir $zh_mmd_dir \
    --split_json_file_path $split_json_file_path \
    --output_dir $trans_model_dir \
    --batch_size 64 \
    --batch_size_per_gpu 4 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --dataloader_num_workers 16 \
    --deepspeed $deepspeed_config_file_path