import pandas as pd
from transformers import AutoConfig, AutoTokenizer
import torch
from typing import Tuple
from torch.utils.data import Dataset, DataLoader


class SentimentDataset(Dataset):
    """Class to create standard format data for
    train/test dataset"""
    def __init__(self, filepath: str, n_rows: int = 320):
        file = pd.read_csv(filepath, encoding='utf-8', nrows=n_rows)
        self.tweets = file['text'].tolist()
        self.att_mask = []
        self.labels = file['sentiment'].tolist()

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx) -> Tuple[str, int, int]:
        return (self.tweets[idx], self.att_mask[idx], self.labels[idx])


def make_dataloader(filepath: str,
                    batch_size: int = 32,
                    n_rows: int = 320) -> DataLoader:
    dataset = SentimentDataset(filepath, n_rows)
    tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest")
    config = AutoConfig.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest")
    for i in range(len(dataset)):
        if type(dataset.tweets[i]) != str:
            dataset.tweets[i] = ''
        dataset.labels[i] = torch.tensor(config.label2id[dataset.labels[i]])
    tokens = tokenizer(dataset.tweets, padding=True, return_tensors='pt')
    dataset.tweets = tokens.input_ids
    dataset.att_mask = tokens.attention_mask
    # dataset.tweets =  [tokens.input_ids, tokens.attention_mask]
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader
