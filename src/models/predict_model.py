from src.models.model import SentimentModel
import torch
from scipy.special import softmax
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from typing import Callable, Tuple, Union, Optional, List

class Predict():
	"""
		Uses the SentimentModel with pre-trained weights to acquire prediction
		for a given data (data_path).
	"""
	def __init__(self, model_path:str = '../../models/checkpoint.pth'):
		if model_path[-4:] != '.pth':
			model_path += '.pth'
		state_dict = torch.load(model_path)
		self.model = SentimentModel()
		self.model.load_state_dict(state_dict)

	def predict(self, tweet: str='')-> Tuple[int, str]:	
		tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
		config = AutoConfig.from_pretrained( "cardiffnlp/twitter-roberta-base-sentiment-latest")
		
		tokens = tokenizer(tweet, padding=True, return_tensors='pt')
		tweet_tokens = tokens.input_ids
		att_mask = tokens.attention_mask
		pred = self.model(tweet_tokens, att_mask)

		pred_id = np.argmax(softmax(pred[0][0].detach().numpy()))
		pred_label = config.id2label[pred_id]

		return pred_id, pred_label

if __name__ == '__main__':
	print('')