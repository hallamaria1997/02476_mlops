import torch
from transformers import AutoModelForSequenceClassification
from torch import nn
from pytorch_lightning import LightningModule
import wandb


class SentimentModel(LightningModule):
    "Class for Model creation"
    # The different parameters are initialized and
    # utilized through save_hyperparmeters() function
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        learning_rate: float = 1e-4,
        batch_size: int = 32,
    ):
        super().__init__()
        # define the model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, tweets: torch.Tensor, att_mask: torch.Tensor) -> any:
        return self.model(tweets, att_mask)

    def training_step(self, batch, batch_idx) -> any:
        tweets, att_mask, labels = batch
        pred = self(tweets, att_mask)
        pred = pred.logits
        loss = self.criterion(pred, labels)
        return loss

    def training_epoch_end(self, outputs: any) -> None:
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", loss, prog_bar=False)
        wandb.log({"train_loss": loss})

    def validation_step(self, batch, batch_idx) -> any:
        tweets, att_mask, labels = batch
        pred = self(tweets, att_mask)
        pred = pred.logits
        val_loss = self.criterion(pred, labels)
        self.log("loss", val_loss, prog_bar=False)

        with torch.no_grad():
            preds = nn.functional.softmax(pred, dim=1).argmax(1)
            correct = (preds == labels).sum()
            accuracy = correct / len(labels)
        self.log("val_accuracy", accuracy, prog_bar=False)
        return {"val_loss": val_loss, "accuracy": accuracy,
                "predictions": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        accuracy = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=False)
        self.log("accuracy", accuracy)

        wandb.log({"val_loss": loss})
        wandb.log({"val_accuracy": accuracy})

        return loss

    def configure_optimizers(self, lr=1e-5) -> any:
        return torch.optim.Adam(self.parameters(), lr=lr)
