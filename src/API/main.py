from fastapi import FastAPI
from src.models.predict_model import Predict

app = FastAPI()
p = Predict()

@app.get("/")
def root():
   return {"Hello": "World"}

@app.get("/predict/{tweet}")
def predict(tweet: str):
	pred_id, pred_label = p.predict(tweet=tweet)
	return {"pred_id": str(pred_id),
			"pred_label": pred_label}