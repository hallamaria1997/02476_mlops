import numpy as np
import os 
import pandas as pd

from torch.utils.data import Dataset, DataLoader


class SentimentDataset(Dataset):
    """Class to create standard format data for 
    train/test dataset"""
    def __init__(self, filepath):
        file = pd.read_csv(filepath)
        self.tweets = file['text'].to_numpy()
        self.labels = file['sentiment'].to_numpy()
        
    def __len__(self):
        return len(self.tweets)
        
    def __getitem__(self, idx):
        return (self.tweets[idx], self.labels[idx])


def train(filepath: str)-> DataLoader:
    dataset = SentimentDataset(filepath)
    train = DataLoader(dataset, batch_size=32)
    return train
  
    
def test(filepath: str)-> DataLoader:
    dataset = SentimentDataset(filepath)
    test = DataLoader(dataset, batch_size=32)
    return test
 
 
train_data = train('data/processed/test.csv')

for tweets, labels in train_data:
    print(tweets)