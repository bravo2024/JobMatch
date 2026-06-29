
"""api.py - FastAPI serving. Loads models/model.pkl (run train.py first), else trains on synthetic."""
import os, numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from src.data import make_synthetic
from src.model import fit_and_evaluate, predict_proba
from src.persist import load_model, save_model
app=FastAPI(title="JobMatch - LinkedIn API")
if os.path.exists("models/model.pkl"):
    MODEL=load_model()
else:
    MODEL,_=fit_and_evaluate(make_synthetic()); save_model(MODEL)
class Features(BaseModel):
    values: list
@app.get("/health")
def health(): return {"status":"ok"}
@app.post("/predict")
def predict_endpoint(f: Features):
    x=np.array([f.values],float)
    return {"probability": float(predict_proba(MODEL,x)[0])}
