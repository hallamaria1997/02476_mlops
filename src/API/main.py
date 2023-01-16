from fastapi import FastAPI, BackgroundTasks
from src.models.predict_model import Predict


app = FastAPI()
p = Predict()


def save_tweet(tweet: str, pred_id: int, pred_label: str):
    with open('predictions.csv', 'a') as file:
        file.write('\n' + tweet + ',' + str(pred_id) + ',' + pred_label)


@app.get("/")
def root():
    return {"Hello": "World"}


@app.get("/predict/{tweet}")
def predict(tweet: str, background_tasks: BackgroundTasks):
    tweet = tweet.replace(',', '')
    pred_id, pred_label = p.predict(tweet=tweet)
    background_tasks.add_task(save_tweet, tweet, pred_id, pred_label)
    return {"pred_id": str(pred_id),
            "pred_label": pred_label}
