import click
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from typing import Callable, Tuple, Union, Optional, List



from src.models.load_data import make_dataloader
from src.models.model import  SentimentModel    


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=3e-3, help="learning rate to use for training")
@click.option("--epochs", default=50, help="number of epochs")
def train(lr:float, epochs:int)->None:
    """main training function for the model, calls the subsequent training function"""
    model = SentimentModel()
    train_set, _ = make_dataloader(data_path="data/processed/train_csv")
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = epochs
    training_loss = []

    for e in tqdm(range(epochs)):
        cum_loss = 0
        for tweets, labels in train_set:
            optimizer.zero_grad()
            pred = model(tweets)
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
    plt.plot(loss)
    plt.title("Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("reports/figures/loss.png", dpi=200)


cli.add_command(train)

if __name__ == "__main__":
    cli()