import torch
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from torch import nn

class SentimentModel(nn.Module):
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

    def forward(self, **input):
        return self.model(**input)
    
