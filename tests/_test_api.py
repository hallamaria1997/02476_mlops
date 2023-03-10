from fastapi.testclient import TestClient
from src.API.main import app, save_tweet
import pytest
import os

@pytest.mark.skipif(not os.path.exists('models/checkpoint.pth'), reason="Checkpoint not found")
def test_root():
    client = TestClient(app)
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {'message': 'Call /predict/<tweet> to get a prediction.'}

@pytest.mark.skipif(not os.path.exists('models/checkpoint.pth'), reason="Checkpoint not found")
def test_save_tweet():
	client = TestClient(app)
	predictions_path = 'src\\api\\'
	with open(predictions_path + 'predictions.csv', 'r') as file:
		lines = file.readlines()
		n_lines_before = len(lines)

	response = client.get('/predict/?tweet=testing that this tweet will be saved')

	save_tweet('testing that this tweet will be saved', 1, 'neutral', predictions_path)

	with open(predictions_path + 'predictions.csv', 'r') as file:
		lines_after = file.readlines()
		last_line = lines_after[-1]
		n_lines_after = len(lines_after)
	
	with open(predictions_path + 'predictions.csv', 'w') as file:
		for l in lines:
			file.write(l)

	assert response.status_code == 200
	assert last_line.split(',')[0] == 'testing that this tweet will be saved'
	assert n_lines_after == n_lines_before + 1

@pytest.mark.skipif(not os.path.exists('models/checkpoint.pth'), reason="Checkpoint not found")
def test_empty_tweet():
    client = TestClient(app)
    response = client.get('/predict/')
    assert response.status_code == 422
