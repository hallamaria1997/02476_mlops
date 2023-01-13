from model import SentimentModel
import torch, click
from load_data import make_dataloader
from scipy.special import softmax
import numpy as np

@click.command()
@click.argument('model_path', default='models/checkpoint.pth')
@click.argument('data_path', default='data/processed/test.csv')
def predict(model_path: str, data_path: str):
	"""
		Uses the SentimentModel with pre-trained weights to acquire prediction
		for a given data (data_path).
	"""
	if model_path[-4:] != '.pth':
		model_path += '.pth'
	
	state_dict = torch.load(model_path)
	test_set = make_dataloader(data_path)
	model = SentimentModel()
	model.load_state_dict(state_dict)

	correct, total = 0, 0
	for tweets, att_mask, labels in test_set:		
		preds = model(tweets, att_mask)
		print(np.argmax(softmax(preds[0][0].detach().numpy())))
		pred_labels = np.argmax(softmax(preds[0][0].detach().numpy()))
		
		correct += (pred_labels == labels).sum().item()
		total += labels.numel()

	print(labels)
	print('Test set accuracy:', correct/total)

if __name__ == '__main__':
    predict()