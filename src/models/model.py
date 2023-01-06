import torch
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer

class SentimentModel():
    "Class for Model creation"
    # The different parameters are initialized and
    # utilized through save_hyperparmeters() function
    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
        learning_rate: float = 5e-5,
        batch_size: int = 32,
    ):
        super().__init__()
        # save all hyperparameters
        self.save_hyperparameters()
        # define the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3
        )

    def forward(self, **inputs):
        return self.model(**inputs)
    