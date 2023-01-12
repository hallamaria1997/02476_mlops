import click
import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from typing import Callable, Tuple, Union, Optional, List
import hydra



import logging
from datetime import datetime

import hydra
import numpy as np
import torch
from datasets import load_metric
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

import wandb
from src import _PATH_MODELS
from src.data.make_dataset import load_data


from src.models.load_data import train   #fylla út fallið
from src.models.model import  SentimentModel      #fylla út fallið


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=3e-3, help="learning rate to use for training")
@click.option("--epochs", default=50, help="number of epochs")
def train(lr:float, epochs:int)->None:
    """main training function for the model, calls the subsequent training function"""
    model = SentimentModel()
    train_set, _ = train(data_path="data/processed/train_csv")
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = epochs
    model_out, loss = training(model, train_set, criterion, optimizer, epochs=epochs)

    torch.save(model_out.state_dict(), "models/checkpoint.pth")
    print("saved to model/checkpoint.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(loss)
    plt.title("Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("reports/figures/loss.png", dpi=200)


def training(model: nn.Module, 
    train_set:  torch.utils.data.DataLoader, 
    criterion : Union[Callable, nn.Module], 
    optimizer Optional[torch.optim.Optimizer] = None, 
    epochs: int = 5) -> :
    """Training function for the model

    Args:
        model (nn.Module): model to train
        train_set (torch.utils.data.DataLoader): training set
        criterion (nn.Module): loss function
        optimizer (torch.optim): optimizer to use for training
    Returns:
        model (nn.Module): trained model
        running_loss_l (list): list of training losses
    """
    pbar = tqdm(range(epochs))
    running_loss_l = []
    for e in pbar:
        running_loss = 0
        for images, labels in train_set:

            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        pbar.set_postfix({"Training loss": running_loss / len(train_set)})
        running_loss_l.append(running_loss / len(train_set))
    return model, running_loss_l


cli.add_command(train)

if __name__ == "__main__":
    cli()