import os
import json
import jieba
import re

import torch
from transformers import PreTrainedModel, GenerationConfig, DefaultDataCollator

import torch.nn as nn
from transformers.modeling_outputs import Seq2SeqLMOutput, ModelOutput
from typing import Optional, Tuple, Union

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

class M4DocModel_Train(PreTrainedModel):
    def __init__(self, config, mllm_encoder, semantic_encoder, nougat_encoder, trans_decoder):
        super().__init__(config)
        self.mllm_encoder = mllm_encoder
        self.semantic_encoder = semantic_encoder
        self.nougat_encoder = nougat_encoder
        self.trans_decoder = trans_decoder
        
        self.semantic_up_proj = nn.Linear(1024, 4096)
        self.expand_length_net = nn.Linear(588, 2048)
        self.semantic_down_proj = nn.Linear(4096, 512)
        self.nougat_down_proj = nn.Linear(1024, 512)
        
        self.kd_loss_weight = config.kd_loss_weight
    
    def forward(
        self,
        mllm_input_ids: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        nougat_pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ):
        mllm_encoder_forward_outputs = self.mllm_encoder(
            input_ids=mllm_input_ids,
            images=images,
            return_dict=True,
        )
        
        semantic_encoder_outputs = self.semantic_encoder(
            pixel_values=nougat_pixel_values,
            return_dict=True,
        )
        
        nougat_encoder_outputs = self.nougat_encoder(
            pixel_values=nougat_pixel_values,
            return_dict=True,
        )
        
        semantic_hidden_states_up = self.semantic_up_proj(semantic_encoder_outputs['last_hidden_state'])
        semantic_hidden_states_up_expand = self.expand_length_net(semantic_hidden_states_up.transpose(1, 2)).transpose(1, 2)
        
        
        # KD loss
        kd_loss_func = nn.CosineEmbeddingLoss()
        x_1 = semantic_hidden_states_up_expand.reshape(-1, 4096)
        x_2 = mllm_encoder_forward_outputs['last_hidden_state'].reshape(-1, 4096)
        y = torch.ones(x_1.shape[0]).to(device=semantic_hidden_states_up_expand.device, dtype=semantic_hidden_states_up_expand.dtype)
        kd_loss = kd_loss_func(x_1, x_2, y)
        
        semantic_hidden_states_down = self.semantic_down_proj(semantic_hidden_states_up_expand)
        nougat_hidden_states_down = self.nougat_down_proj(nougat_encoder_outputs['last_hidden_state'])
        
        
        decoder_outputs = self.trans_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=nougat_hidden_states_down,
            mllm_encoder_hidden_states=semantic_hidden_states_down,
            return_dict=True,
        )
        
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            trans_loss_func = nn.CrossEntropyLoss()
            trans_loss = trans_loss_func(logits.reshape(-1, self.trans_decoder.config.vocab_size), labels.reshape(-1).long())
            
            loss = trans_loss + kd_loss * self.kd_loss_weight
        
        
        return ExtendedSeq2SeqLMOutput(
            loss=loss,
            trans_loss=trans_loss,
            kd_loss=kd_loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_hidden_state=nougat_hidden_states_down,
            mllm_encoder_hidden_states=semantic_hidden_states_down,
        )
    
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        state_dict: Optional[dict] = None,
        safe_serialization: bool = False,
        **kwargs,
    ):
        save_dir = os.path.join(save_directory, 'state_dict')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.semantic_encoder.state_dict(), os.path.join(save_dir, 'semantic_encoder.bin'))
        torch.save(self.nougat_encoder.state_dict(), os.path.join(save_dir, 'nougat_encoder.bin'))
        torch.save(self.trans_decoder.state_dict(), os.path.join(save_dir, 'trans_decoder.bin'))
        
        torch.save(self.semantic_up_proj.state_dict(), os.path.join(save_dir, 'semantic_up_proj.bin'))
        torch.save(self.expand_length_net.state_dict(), os.path.join(save_dir, 'expand_length_net.bin'))
        torch.save(self.semantic_down_proj.state_dict(), os.path.join(save_dir, 'semantic_down_proj.bin'))
        torch.save(self.nougat_down_proj.state_dict(), os.path.join(save_dir, 'nougat_down_proj.bin'))

class M4DocModel_Inference(PreTrainedModel):
    def __init__(self, config, semantic_encoder, nougat_encoder, trans_decoder):
        super().__init__(config)
        self.semantic_encoder = semantic_encoder
        self.nougat_encoder = nougat_encoder
        self.trans_decoder = trans_decoder
        
        self.semantic_up_proj = nn.Linear(1024, 4096)
        self.expand_length_net = nn.Linear(588, 2048)
        self.semantic_down_proj = nn.Linear(4096, 512)
        self.nougat_down_proj = nn.Linear(1024, 512)
    
    def generate(
        self,
        nougat_pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        semantic_encoder_outputs = self.semantic_encoder(
            pixel_values=nougat_pixel_values,
            return_dict=True,
        )
        
        nougat_encoder_outputs = self.nougat_encoder(
            pixel_values=nougat_pixel_values,
            return_dict=True,
        )
        
        semantic_hidden_states_up = self.semantic_up_proj(semantic_encoder_outputs['last_hidden_state'])
        semantic_hidden_states_up_expand = self.expand_length_net(semantic_hidden_states_up.transpose(1, 2)).transpose(1, 2)
        
        semantic_hidden_states_down = self.semantic_down_proj(semantic_hidden_states_up_expand)
        nougat_hidden_states_down = self.nougat_down_proj(nougat_encoder_outputs['last_hidden_state'])
        
        generation_outputs = self.trans_decoder.generate(
            encoder_hidden_states=nougat_hidden_states_down,
            mllm_encoder_hidden_states=semantic_hidden_states_down,
            generation_config=generation_config,
        )
        
        return generation_outputs
