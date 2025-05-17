import os
import json
import jieba
import re
from PIL import Image
import argparse
import copy

import torch
from transformers import AutoTokenizer, DonutProcessor, VisionEncoderDecoderModel, EncoderDecoderModel, EncoderDecoderConfig, GenerationConfig

from my_model import M4DocModel_Inference


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_type = torch.bfloat16
    MAX_LENGTH = 1536

    # load processor and tokenizer

    nougat_processor = DonutProcessor.from_pretrained(os.path.join(args.base_dir, 'utils/donut-base'))
    nougat_processor.image_processor.size = {'height': 896, 'width': 672}
    nougat_processor.image_processor.image_mean = [0.485, 0.456, 0.406]
    nougat_processor.image_processor.image_std = [0.229, 0.224, 0.225]
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.base_dir, 'utils/zh_tokenizer'))

    # load model

    trans_model = EncoderDecoderModel.from_pretrained(args.trans_model_dir, device_map='cpu')
    trans_decoder = trans_model.decoder

    nougat_model = VisionEncoderDecoderModel.from_pretrained(args.nougat_model_dir, device_map='cpu')
    nougat_encoder = nougat_model.encoder
    
    semantic_encoder = copy.deepcopy(nougat_encoder)

    config = EncoderDecoderConfig.from_pretrained(args.trans_model_dir)
    model = M4DocModel_Inference(config, semantic_encoder, nougat_encoder, trans_decoder)

    print(model)

    # load state_dict
    state_dict_dir = os.path.join(args.checkpoint_dir, 'state_dict')
    model.semantic_encoder.load_state_dict(torch.load(os.path.join(state_dict_dir, 'semantic_encoder.bin'), map_location=torch.device('cpu')))
    model.nougat_encoder.load_state_dict(torch.load(os.path.join(state_dict_dir, 'nougat_encoder.bin'), map_location=torch.device('cpu')))
    model.trans_decoder.load_state_dict(torch.load(os.path.join(state_dict_dir, 'trans_decoder.bin'), map_location=torch.device('cpu')))

    model.semantic_up_proj.load_state_dict(torch.load(os.path.join(state_dict_dir, 'semantic_up_proj.bin'), map_location=torch.device('cpu')))
    model.expand_length_net.load_state_dict(torch.load(os.path.join(state_dict_dir, 'expand_length_net.bin'), map_location=torch.device('cpu')))
    model.semantic_down_proj.load_state_dict(torch.load(os.path.join(state_dict_dir, 'semantic_down_proj.bin'), map_location=torch.device('cpu')))
    model.nougat_down_proj.load_state_dict(torch.load(os.path.join(state_dict_dir, 'nougat_down_proj.bin'), map_location=torch.device('cpu')))

    model.to(device, dtype=data_type)
    model.eval()
    

    # generate
    image_dir = os.path.join(args.dataset_dir, 'imgs')

    json_file_path = os.path.join(args.dataset_dir, 'split_dataset.json')
    with open(json_file_path, 'r') as f:
        json_dict = json.load(f)
    test_name_list = json_dict['test_name_list']

    generation_config = GenerationConfig(
        max_length=MAX_LENGTH,
        early_stopping=True,
        num_beams=4,
        use_cache=True,
        length_penalty=1.0,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    from tqdm import tqdm
    for name in tqdm(test_name_list):
        image_file_path = os.path.join(image_dir, name+'.png')
        image = Image.open(image_file_path).convert('RGB')
        nougat_pixel_values = nougat_processor(image, return_tensors='pt')['pixel_values'].to(device, dtype=data_type)

        generation_outputs = model.generate(
            nougat_pixel_values=nougat_pixel_values,
            generation_config=generation_config,
        )
        
        zh_text = tokenizer.decode(generation_outputs[0])
        
        result_file_path = os.path.join(args.output_dir, name+'.mmd')
        with open(result_file_path, 'w', encoding='utf-8') as f:
            f.write(zh_text)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str)
    parser.add_argument("--base_dir", type=str)
    parser.add_argument("--trans_model_dir", type=str)
    parser.add_argument("--nougat_model_dir", type=str)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    
    args = parser.parse_args()
    
    main(args)

