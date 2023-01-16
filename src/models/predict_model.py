from model import SentimentModel
import torch, click
from scipy.special import softmax
import numpy as np
from transformers import AutoTokenizer

@click.command()
@click.option('--model_path', default='models/checkpoint.pth')
@click.option('--tweet', default='')
def predict(model_path: str, tweet: str):
	"""
		Uses the SentimentModel with pre-trained weights to acquire prediction
		for a given data (data_path).
	"""
	if model_path[-4:] != '.pth':
		model_path += '.pth'
	
	state_dict = torch.load(model_path)
	model = SentimentModel()
	model.load_state_dict(state_dict)

	tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
	
	tokens = tokenizer(tweet, padding=True, return_tensors='pt')
	tweet_tokens = tokens.input_ids
	att_mask = tokens.attention_mask
	pred = model(tweet_tokens, att_mask)

	pred_label = np.argmax(softmax(pred[0][0].detach().numpy()))
	print(pred_label)
	return pred_label

if __name__ == '__main__':
    predict()