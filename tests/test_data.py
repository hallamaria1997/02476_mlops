import sys
sys.path.insert(1,'src/models')
from load_data import make_dataloader
import os.path
import pytest

train_path = 'data/raw/train.csv'
test_path = 'data/raw/test.csv'
N_train = 3200 # 100 batches of 32 observations
N_test = 3200 # 100 batches of 32 observations
batch_size = 32

@pytest.mark.skipif(not os.path.exists(train_path), reason="Train data files not found")
def test_train_data():
    train_set = make_dataloader(train_path, batch_size=batch_size, n_rows=N_train)
    assert len(train_set) == N_train/batch_size, "Train Data: dataset did not have the correct number of samples"
    for tweets, att_mask, labels in train_set:
        assert tweets.size(dim=0)==batch_size and tweets.size(dim=1)==72, "Train Data: Size of tweets attributes are incorrect"
        assert att_mask.size(dim=0)==batch_size and att_mask.size(dim=1)==72, "Train Data: Size of att_mask attribute is incorrect"
        assert labels.size(dim=0)==32, "Train Data: Size of labels attribute is incorrect"
        assert len(set(labels.tolist())), "Train Data: Not all batches hold all possible labels"

@pytest.mark.skipif(not os.path.exists(test_path), reason="Test data files not found")
def test_test_data():
    test_set = make_dataloader(test_path, batch_size=batch_size, n_rows=N_test)
    assert len(test_set) == N_train/batch_size, "Test Data: dataset did not have the correct number of samples"
    for tweets, att_mask, labels in test_set:
        assert tweets.size(dim=0)==batch_size and tweets.size(dim=1)==70, "Test Data: Size of tweets attributes are incorrect"
        assert att_mask.size(dim=0)==batch_size and att_mask.size(dim=1)==70, "Test Data: Size of att_mask attribute is incorrect"
        assert labels.size(dim=0)==32, "Test Data: Size of labels attribute is incorrect"
        assert len(set(labels.tolist())), "Test Data: Not all batches hold all possible labels"
