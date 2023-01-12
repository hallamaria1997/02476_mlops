from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
import click
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from typing import Callable, Tuple, Union, Optional, List
from scipy.special import softmax
import numpy as np

from load_data import make_dataloader
from model import  SentimentModel    


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-4, help="learning rate to use for training")
@click.option("--epochs", default=30, help="number of epochs")
def train(lr:float, epochs:int)->None:
    """main training function for the model, calls the subsequent training function"""
    
    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    
    
    tokenizer = AutoTokenizer.from_pretrained( "cardiffnlp/twitter-roberta-base-sentiment-latest")
    config = AutoConfig.from_pretrained( "cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = SentimentModel()
    train_set = make_dataloader(filepath="C:/Users/Lenovo/Documents/02476_mlops/data/raw/train.csv")
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = epochs
    training_loss = []

    for e in tqdm(range(epochs)):
        cum_loss = 0
        for tweets, labels in train_set:
            optimizer.zero_grad()
            labels = labels[0]
            tweet = preprocess(tweets[0])
            # tweet = [tokenizer(t, return_tensors='pt') for t in tweets]
            # labels = [l for l in labels]
            print(tweet)
            input = tokenizer(tweet, return_tensors='pt')
            pred = model(**input)
            loss = criterion(pred[0][0], torch.tensor(config.label2id[labels]))
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
        print('Loss in epoch ' + str(e) + ': ' +
            str(cum_loss/len(train_set)))
        training_loss.append(cum_loss / len(train_set))

    torch.save(model.state_dict(), "models/checkpoint.pth")
    print("saved to model/checkpoint.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(loss)
    plt.title("Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("reports/figures/loss.png", dpi=200)


cli.add_command(train)

if __name__ == "__main__":
    cli()