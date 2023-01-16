import sys
sys.path.insert(1,'src/models')
from model import SentimentModel
import pytest
import torch
import random

model = SentimentModel()

@pytest.mark.skipif(not model.parameters, reason="Model not created")
def test_model_is_Roberta():
    assert 'RobertaForSequenceClassification' in str(type(model.model)), 'Model is not of type: RobertaForSequenceClassification'
    
@pytest.mark.skipif(not model.parameters, reason="Model not created")
def test_model_output_shape():
    batch_size = random.randint(0,64)
    tweet = torch.rand(batch_size,72).type(torch.LongTensor)
    att_mask = torch.ones(batch_size,72)
    output = model(tweet, att_mask)
    assert output.logits.size(dim=0)==batch_size and output.logits.size(dim=1)==3, 'Model output is of incorrect shape'
