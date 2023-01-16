import numpy as np
import os 
import pandas as pd
from transformers import AutoConfig, AutoTokenizer
import torch

from torch.utils.data import Dataset, DataLoader


class SentimentDataset(Dataset):
    """Class to create standard format data for 
    train/test dataset"""
    def __init__(self, filepath):
        file = pd.read_csv(filepath, encoding='utf-8', nrows=3200)
        self.tweets = file['text'].tolist()
        self.att_mask = []
        self.labels = file['sentiment'].tolist()
        
    def __len__(self):
        return len(self.tweets)
        
    def __getitem__(self, idx):
        return (self.tweets[idx], self.att_mask[idx], self.labels[idx])


def make_dataloader(filepath: str)-> DataLoader:
    dataset = SentimentDataset(filepath)
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    config = AutoConfig.from_pretrained( "cardiffnlp/twitter-roberta-base-sentiment-latest")
    for i in range(len(dataset)):
        if type(dataset.tweets[i]) != str:
            dataset.tweets[i] = ''
        dataset.labels[i] = torch.tensor(config.label2id[dataset.labels[i]])
            
    tokens = tokenizer(dataset.tweets, padding=True, return_tensors='pt')
    dataset.tweets = tokens.input_ids
    dataset.att_mask = tokens.attention_mask
    
    #dataset.tweets =  [tokens.input_ids, tokens.attention_mask]
    dataloader = DataLoader(dataset, batch_size=32)
    return dataloader 
