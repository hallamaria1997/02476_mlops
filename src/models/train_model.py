import torch
from omegaconf import DictConfig, OmegaConf
from load_data import make_dataloader
from model import SentimentModel
import hydra
from hydra.utils import to_absolute_path
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler
from google.cloud import storage


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

    model = SentimentModel()

    wandb.init(mode=config.experiment.wandb.mode,
               config=config.experiment.hyper_parameters)

    train_data = make_dataloader(filepath="/data/raw/train.csv",
                                 batch_size=batch_size,
                                 n_rows=n_rows)
    val_data = make_dataloader(filepath="/data/raw/test.csv",
                               batch_size=batch_size,
                               n_rows=n_rows)

    wandb.watch(model, log_freq=100)
    config_wandb = {
        "model": model,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs
    }
    wandb_logger = WandbLogger(
        project=config.experiment.wandb.model_dir,
        entity=config.experiment.wandb.entity,
        config=config_wandb
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=3, verbose=True, mode="min"
    )

    profiler = SimpleProfiler()

    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=epochs,
        precision=32,
        accelerator='cpu',
        logger=wandb_logger,
        default_root_dir=to_absolute_path(config.experiment.wandb.model_dir),
        profiler=profiler
    )

    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=val_data)

    torch.save(model.state_dict(), "/models/checkpoint.pth")
    print("Model saved to models/checkpoint.pth")

    storage_client = storage.Client(project='dtumlops-tweet-sentiment')
    bucket = storage_client.bucket("trained-twitter-model")
    blob = bucket.blob("/models/checkpoint.pth")

    print("Uploading to cloud")
    blob.upload_from_filename("/models/checkpoint.pth", timeout=14400)

    print("Model saved to GCP bucket!")


if __name__ == "__main__":
    train()
