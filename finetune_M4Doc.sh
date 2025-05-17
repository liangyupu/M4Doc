#!/bin/bash

# This script is used to construct M4Doc model and finetune it on the DoTA dataset.
base_dir=/path/to/M4Doc

trans_model_dir=$base_dir/models/trans_model

deepspeed_config_file_path=$base_dir/utils/deepspeed_zero0.json

mllm_model_dir=$base_dir/pretrained_models/vary-llava80k
nougat_model_dir=$base_dir/pretrained_models/nougat-small

dataset_dir=$base_dir/DoTA_dataset
m4doc_model_dir=$base_dir/models/M4Doc_model

export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
    --num_processes 4 \
    --num_machines 1 codes/finetune_M4Doc_vary.py \
    --base_dir $base_dir \
    --mllm_model_dir $mllm_model_dir \
    --trans_model_dir $trans_model_dir \
    --nougat_model_dir $nougat_model_dir \
    --dataset_dir $dataset_dir \
    --output_dir $m4doc_model_dir \
    --batch_size 64 \
    --batch_size_per_gpu 2 \
    --dataloader_num_workers 16 \
    --deepspeed $deepspeed_config_file_path