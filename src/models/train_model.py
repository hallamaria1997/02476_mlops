import click
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from scipy.special import softmax
import numpy as np

from load_data import make_dataloader
from model import  SentimentModel    

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-4, help="learning rate to use for training")
@click.option("--epochs", default=4, help="number of epochs")
def train(lr:float, epochs:int)->None:
    """main training function for the model, calls the subsequent training function"""
    
    model = SentimentModel()
    train_set = make_dataloader(filepath="data/raw/train.csv")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = epochs
    training_loss = []

    for e in tqdm(range(epochs)):
        cum_loss = 0
        for tweets, att_mask, labels in train_set:
            optimizer.zero_grad()
            pred = model(tweets, att_mask)
            pred  = pred.logits
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
        print('Loss in epoch ' + str(e) + ': ' +
            str(cum_loss/len(train_set)))
        training_loss.append(cum_loss / len(train_set))

    torch.save(model.state_dict(), "models/checkpoint.pth")
    print("saved to model/checkpoint.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(training_loss)
    print('plotting')
    plt.title("Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("reports/figures/loss.png", dpi=200)


cli.add_command(train)

if __name__ == "__main__":
    cli()
