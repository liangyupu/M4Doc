import os
import json
import torch
import jieba
import re
from PIL import Image

from vary.model import *
from vary.utils.conversation import conv_templates
from vary.model.plug.transforms import test_transform

from transformers import PreTrainedModel, BertConfig, VisionEncoderDecoderModel, GenerationConfig
import torch.nn as nn
from transformers.modeling_outputs import Seq2SeqLMOutput, ModelOutput
from typing import Optional, Tuple, Union

def get_en_text(en_file_path):
    split_lines = []
    with open(en_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.strip() == '':
            continue
        split_lines.append(line.strip() + ' \n\n')
    return ' '.join(split_lines)

def get_zh_text(zh_file_path):
    split_lines = []
    with open(zh_file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.strip() == '':
            continue
        split_lines.append(' '.join(jieba.cut(line.strip())) + ' \n\n')
    return ' '.join(split_lines)

from torch.utils.data import Dataset
class DoTADatasetTrans(Dataset):
    def __init__(self, en_tokenizer, zh_tokenizer, en_txt_dir, zh_txt_dir, name_list, max_length):
        self.en_tokenizer = en_tokenizer
        self.zh_tokenizer = zh_tokenizer
        self.en_txt_dir = en_txt_dir
        self.zh_txt_dir = zh_txt_dir
        self.name_list = name_list
        self.max_length = max_length
    
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, index):
        encoding = {}
        en_file_path = os.path.join(self.en_txt_dir, self.name_list[index]+'.mmd')
        en_text = get_en_text(en_file_path)
        tokenizer_outputs = self.en_tokenizer(en_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoding['input_ids'] = tokenizer_outputs['input_ids'][0]
        encoding['attention_mask'] = tokenizer_outputs['attention_mask'][0]
        
        zh_file_path = os.path.join(self.zh_txt_dir, self.name_list[index]+'.mmd')
        zh_text = get_zh_text(zh_file_path)
        tokenizer_outputs = self.zh_tokenizer(zh_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        encoding['decoder_input_ids'] = tokenizer_outputs['input_ids'][0]
        encoding['decoder_attention_mask'] = tokenizer_outputs['attention_mask'][0]
        input_ids = tokenizer_outputs['input_ids'][0].tolist()
        labels = input_ids[1:] + [-100]*(self.max_length-len(input_ids)+1)
        encoding['labels'] = torch.tensor(labels, dtype=torch.long)
        
        return encoding

MAX_LENGTH=1536
MLLM_MAX_LENGTH = 2048
prompt_OCR = 'Convert the image to markdown format.'
image_processor_high = test_transform
image_token_len = 256

class DoTADataset(Dataset):
    def __init__(self, mllm_image_processor, mllm_tokenizer, nougat_processor, tokenizer, image_dir, en_txt_dir, zh_txt_dir, name_list):
        self.name_list = name_list
        self.mllm_image_processor = mllm_image_processor
        self.mllm_tokenizer = mllm_tokenizer
        self.nougat_processor = nougat_processor
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.en_txt_dir = en_txt_dir
        self.zh_txt_dir = zh_txt_dir
        self.name_list = name_list
        
    def __len__(self):
        return len(self.name_list)
    
    def __getitem__(self, index):
        encoding = {}
        
        image_file_path = os.path.join(self.image_dir, self.name_list[index]+'.png')
        image = Image.open(image_file_path)
        image = image.convert('RGB')
        image_1 = image.copy()
        image_tensor = self.mllm_image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor_1 = test_transform(image_1)

        encoding['image_low'] = image_tensor.unsqueeze(0)
        encoding['image_high'] = image_tensor_1.unsqueeze(0)

        en_mmd_file_path = os.path.join(self.en_txt_dir, self.name_list[index]+'.mmd')
        en_mmd = get_en_text(en_mmd_file_path)

        qs = '<img>' + '<imgpad>'*image_token_len + '</img>' + prompt_OCR
        conv = conv_templates['mpt'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        all_prompt = prompt + en_mmd + '<|im_end|>'
        
        tokenizer_outputs = self.mllm_tokenizer(all_prompt, return_tensors="pt", padding="max_length", max_length=MLLM_MAX_LENGTH, truncation=True)
        encoding['mllm_input_ids'] = tokenizer_outputs['input_ids'].squeeze(0)

        image_file_path = os.path.join(self.image_dir, self.name_list[index]+'.png')
        image = Image.open(image_file_path)
        image = image.convert('RGB')
        nougat_pixel_values = self.nougat_processor(image, return_tensors="pt").pixel_values.squeeze(0)
        encoding['nougat_pixel_values'] = nougat_pixel_values
        
        zh_mmd_file_path = os.path.join(self.zh_txt_dir, self.name_list[index]+'.mmd')
        zh_mmd = get_zh_text(zh_mmd_file_path)
        
        tokenizer_outputs = self.tokenizer(zh_mmd, return_tensors="pt", padding="max_length", max_length=MAX_LENGTH, truncation=True)
        encoding['decoder_input_ids'] = tokenizer_outputs['input_ids'][0]
        encoding['decoder_attention_mask'] = tokenizer_outputs['attention_mask'][0]
        tokenizer_outputs = self.tokenizer(zh_mmd, max_length=MAX_LENGTH, truncation=True)
        input_ids = tokenizer_outputs['input_ids']
        labels = input_ids[1:] + [-100]*(MAX_LENGTH-len(input_ids)+1)
        encoding['labels'] = torch.tensor(labels, dtype=torch.long)
        
        return encoding

from transformers import DefaultDataCollator
class DoTADatasetDataCollator(DefaultDataCollator):
    def __call__(self, batch):
        encoding = {}
        encoding['mllm_input_ids'] = torch.cat([sample['mllm_input_ids'].unsqueeze(0) for sample in batch], 0)
        encoding['images'] = [(sample['image_low'], sample['image_high']) for sample in batch]
        encoding['nougat_pixel_values'] = torch.cat([sample['nougat_pixel_values'].unsqueeze(0) for sample in batch], 0)
        encoding['decoder_input_ids'] = torch.cat([sample['decoder_input_ids'].unsqueeze(0) for sample in batch], 0)
        encoding['decoder_attention_mask'] = torch.cat([sample['decoder_attention_mask'].unsqueeze(0) for sample in batch], 0)
        encoding['labels'] = torch.cat([sample['labels'].unsqueeze(0) for sample in batch], 0)
        return encoding

class ExtendedSeq2SeqLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    trans_loss: Optional[torch.FloatTensor] = None
    kd_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    mllm_encoder_hidden_states: Optional[torch.FloatTensor] = None