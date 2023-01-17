from fastapi import FastAPI, BackgroundTasks
import uvicorn
import sys
from src.models.predict_model import Predict
import os


app = FastAPI()

if os.path.exists('models/checkpoint.pth'):
    if sys.argv[1] == 'False':
        p = Predict(model_path='../../models/checkpoint.pth')
    else:
        p = Predict()


def save_tweet(tweet: str, pred_id: int, pred_label: str,
               file_path: str = '') -> None:
    """Saves the received tweet to a database, along with the predicted
    id and label."""
    with open(file_path + 'predictions.csv', 'a') as file:
        file.write('\n' + tweet + ',' + str(pred_id) + ',' + pred_label)


@app.get("/")
def root() -> dict:
    """The root."""
    return {"message": "Call /predict/<tweet> to get a prediction."}


@app.get("/predict/{tweet}")
def predict(tweet: str, background_tasks: BackgroundTasks) -> dict:
    """Returns a classification of an input sentence.

    Parameters:
        tweet (string). Inserted as a parameter in the URL.
    Returns:
        pred_id (int): A numeral representation of the class (0, 1 or 2).
        pred_label (string): A text representation of the class
        (negative, neutral, positive)."""
    tweet = tweet.replace(',', '')
    pred_id, pred_label = p.predict(tweet=tweet)
    background_tasks.add_task(save_tweet, tweet, pred_id, pred_label)
    return {"pred_id": str(pred_id),
            "pred_label": pred_label}


if __name__ == '__main__':
    uvicorn.run("main:app", port=8000, reload=True)
