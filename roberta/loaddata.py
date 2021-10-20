import json
import os
import re
from collections import defaultdict

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer


testdata_path = 'data/rule-reasoning-dataset-V2020.2.5.0/original/depth-{}/test.jsonl'
traindata_path = 'data/rule-reasoning-dataset-V2020.2.5.0/original/depth-{}/train.jsonl'

class train_dataset(Dataset):

    def __init__(self,inputs,max_length = 512):
        self.tensors = {'input_ids':[],'attention_mask':[],'labels':[]}
        self.max_length = max_length
        for idx in tqdm(range(len(next(iter(inputs.values()))))):
            padded_input = None
            if len(inputs['input_ids'][idx]) <= max_length:
                padded_input,attention_mask = self.padding(inputs['input_ids'][idx])
            else:
                print(len(inputs['input_ids'][idx]))
                exit(1)
            assert padded_input != None,"error"
            self.tensors['input_ids'].append(padded_input)
            self.tensors['attention_mask'].append(attention_mask)
            self.tensors['labels'].append(inputs['labels'][idx])
        for t in self.tensors:
            self.tensors[t] = torch.LongTensor(self.tensors[t][:])
        # self.tensors['labels'] = torch.nn.functional.one_hot(self.tensors['labels'],num_classes=2)
        
    def padding(self,input):
        attention_mask = [1]*self.max_length
        # attention_mask = [1]*len(input)+[0]*(self.max_length-len(input))
        return (input + [1]*(self.max_length-len(input))),attention_mask

    def cuda(self):
        for key in self.tensors:
            self.tensors[key] = self.tensors[key].cuda()
    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)

def load_orgin_data(deep=5):
    '''所有id下，所有rules'''
    datas = []
    with open(traindata_path.format(deep), "r+", encoding="utf8") as f:

        reader = jsonlines.Reader(f)
        for item in reader:
            datas.append(item)

    return datas


def load_context_ques_pair(datas):
    context_ques_pairs = []
    for data in datas:
        context_ques_pairs.append((data['context'], data['questions']))
    return context_ques_pairs


def load_inputs(context_ques_pairs):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    inputs = {'input_ids': [], 'labels': []}
    for pair in tqdm(context_ques_pairs):
        context = pair[0]
        for ques in pair[1]:
            label = ques['label']
            text = ques['text']
            input_ids = tokenizer.build_inputs_with_special_tokens(
                tokenizer(context)['input_ids'][1:-1], tokenizer(text)['input_ids'][1:-1])
            inputs['input_ids'].append(input_ids)
            inputs['labels'].append(label)
    return inputs


def save_inputs():
    print('preparing...')
    datas = load_orgin_data(2)
    context_ques_pairs = load_context_ques_pair(datas)
    inputs = load_inputs(context_ques_pairs)
    json_str = json.dumps(inputs)
    with open('train_data.json', 'w') as json_file:
        json_file.write(json_str)


def load_inputs_from_json():
    with open('train_data.json', 'r') as json_file:
        inputs = json.load(json_file)
    max_len = 0
    print(len(inputs['input_ids']))
    for i in inputs['input_ids']:
        if len(i) > max_len:
            max_len = len(i)
    print(max_len)
    return inputs
if __name__ == '__main__':
    save_inputs()

