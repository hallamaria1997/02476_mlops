import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from load_data import make_dataloader
from model import SentimentModel
import hydra
import wandb


@hydra.main(config_path="config", config_name='default_config.yaml')
def train(config: DictConfig) -> None:
    """main training function for the model,
    calls the subsequent training function"""

    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hyparams = config.experiment.hyper_parameters
    # torch.manual_seed(hyparams["seed"])
    epochs = hyparams['n_epochs']
    lr = hyparams['lr']
    batch_size = hyparams['batch_size']
    n_rows = hyparams['n_rows']

    wandb.init(mode=config.experiment.wandb.mode)

    model = SentimentModel()
    train_set = make_dataloader(filepath="data/raw/train.csv",
                                batch_size=batch_size,
                                n_rows=n_rows)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = epochs
    training_loss = []


    
    for e in tqdm(range(epochs)):
        cum_loss = 0
        for tweets, att_mask, labels in train_set:
            optimizer.zero_grad()
            pred = model(tweets, att_mask)
            pred = pred.logits
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
        print('Loss in epoch ' + str(e) + ': ' + str(cum_loss/len(train_set)))
        training_loss.append(cum_loss / len(train_set))

    torch.save(model.state_dict(), "C:/Users/Rebekka/Desktop/DTU/MLOps/02476_mlops/models/checkpoint.pth")
    print("saved to model/checkpoint.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(training_loss)
    print('plotting')
    plt.title("Training loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("C:/Users/Rebekka/Desktop/DTU/MLOps/02476_mlops/reports/figures/loss.png", dpi=200)


if __name__ == "__main__":
    train()
