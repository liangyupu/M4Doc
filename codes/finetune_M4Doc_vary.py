import os
import json
import jieba
import re
import argparse

import torch
from transformers import AutoTokenizer, CLIPImageProcessor, DonutProcessor, VisionEncoderDecoderModel, EncoderDecoderModel, EncoderDecoderConfig, TrainingArguments
from vary.model import *

from my_model import M4DocModel_Train
from my_dataset import DoTADataset, DoTADatasetDataCollator
from my_trainer import Trainer

import copy

def train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_type = torch.bfloat16
    MAX_LENGTH = 1536

    # load MLLM, processor and tokenizer

    mllm_tokenizer = AutoTokenizer.from_pretrained(args.mllm_model_dir, trust_remote_code=True)
    mllm_image_processor = CLIPImageProcessor.from_pretrained(os.path.join(args.base_dir, 'utils/clip-vit-large-patch14'))

    nougat_processor = DonutProcessor.from_pretrained(os.path.join(args.base_dir,'utils/donut-base'))
    nougat_processor.image_processor.size = {'height': 896, 'width': 672}
    nougat_processor.image_processor.image_mean = [0.485, 0.456, 0.406]
    nougat_processor.image_processor.image_std = [0.229, 0.224, 0.225]
    
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.base_dir, 'utils/zh_tokenizer'))

    # load model

    trans_model = EncoderDecoderModel.from_pretrained(args.trans_model_dir, device_map='cpu')
    trans_decoder = trans_model.decoder

    mllm_model = varyQwenForCausalLM.from_pretrained(args.mllm_model_dir, low_cpu_mem_usage=True, device_map='cpu', trust_remote_code=True)
    mllm_encoder = mllm_model.transformer

    nougat_model = VisionEncoderDecoderModel.from_pretrained(args.nougat_model_dir, device_map='cpu')
    nougat_encoder = nougat_model.encoder

    semantic_encoder = copy.deepcopy(nougat_encoder)

    config = EncoderDecoderConfig.from_pretrained(args.trans_model_dir)
    config.kd_loss_weight = 1.0
    model = M4DocModel_Train(config, mllm_encoder, semantic_encoder, nougat_encoder, trans_decoder)

    print(model)

    # frozen mllm_encoder
    for param in model.mllm_encoder.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total parameters: {total_params/1e6:.2f}M, Trainable parameters: {trainable_params/1e6:.2f}M, Percentage: {trainable_params/total_params*100:.2f}%')

    # load dataset

    json_file_path = os.path.join(args.dataset_dir, 'split_dataset.json')
    with open(json_file_path, 'r') as f:
        json_dict = json.load(f)
    train_name_list = json_dict['train_name_list']
    valid_name_list = json_dict['valid_name_list']
    
    image_dir = os.path.join(args.dataset_dir, 'imgs')
    en_txt_dir = os.path.join(args.dataset_dir, 'en_mmd')
    zh_txt_dir = os.path.join(args.dataset_dir, 'zh_mmd')

    train_dataset = DoTADataset(mllm_image_processor, mllm_tokenizer, nougat_processor, tokenizer, image_dir, en_txt_dir, zh_txt_dir, train_name_list)
    valid_dataset = DoTADataset(mllm_image_processor, mllm_tokenizer, nougat_processor, tokenizer, image_dir, en_txt_dir, zh_txt_dir, valid_name_list)
    my_data_collator = DoTADatasetDataCollator()

    # train args

    batch_size = args.batch_size
    batch_size_per_gpu = args.batch_size_per_gpu
    num_gpus = torch.cuda.device_count()
    gradient_accumulation_steps = batch_size // (batch_size_per_gpu * num_gpus)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=batch_size_per_gpu,
        per_device_eval_batch_size=batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=args.max_steps,
        logging_strategy='steps',
        logging_steps=1,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        bf16=args.bf16,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        deepspeed=args.deepspeed,
        report_to='tensorboard'
    )

    print(training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=my_data_collator,
    )

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str)
    parser.add_argument("--mllm_model_dir", type=str)
    parser.add_argument("--trans_model_dir", type=str)
    parser.add_argument("--nougat_model_dir", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--deepspeed", type=str)
    
    args = parser.parse_args()
    
    train(args)

