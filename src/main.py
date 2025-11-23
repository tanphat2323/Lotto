from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import json
from src.ingestion import DataLoader
from src.features import FeatureEngine

app = FastAPI(title="Lotto 5/35 AI Predictor")

# Constants
MODEL_PATH = "models/lotto_model.keras"
DATA_PATH = "data/dataset.csv"
METRICS_PATH = "models/latest_metrics.json"
LOOKBACK = 10

# Load Model & Data on Startup
print("Loading model and data...")
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None
    print(f"Warning: Model not found at {MODEL_PATH}")

loader = DataLoader(DATA_PATH)
df_history = loader.load_data() # Ensure data is loaded
history_list = loader.get_draw_history()
feature_engine = FeatureEngine(lookback=LOOKBACK)

class PredictionResponse(BaseModel):
    draw_date: str
    predicted_main_top10: list[int]
    predicted_special_top3: list[int]
    main_probabilities: dict[int, float]
    special_probabilities: dict[int, float]

@app.get("/")
def home():
    return {"message": "Lotto 5/35 AI Predictor API is running."}

@app.get("/stats")
def get_stats():
    """Returns the metrics from the latest training run."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            return json.load(f)
    return {"error": "Metrics not found."}

@app.get("/predict/next")
def predict_next_draw():
    """Predicts the probability distribution for the NEXT upcoming draw."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    # Get last N draws
    if len(history_list) < LOOKBACK:
         raise HTTPException(status_code=400, detail="Not enough history data")

    last_sequence_draws = history_list[-LOOKBACK:]

    # Encode
    vectors = [feature_engine.encode_draw(d['main'], d['special']) for d in last_sequence_draws]
    input_seq = np.array([vectors]) # Shape (1, 10, 47)

    # Predict
    preds = model.predict(input_seq, verbose=0)
    pred_main = preds[0][0] # Shape (35,)
    pred_special = preds[1][0] # Shape (12,)

    # Process Results
    # Main: Get top 10
    top_10_main_idx = np.argsort(pred_main)[-10:][::-1]
    top_10_main = [int(i + 1) for i in top_10_main_idx]

    # Special: Get top 3
    top_3_special_idx = np.argsort(pred_special)[-3:][::-1]
    top_3_special = [int(i + 1) for i in top_3_special_idx]

    # Format probabilities
    main_probs = {int(i+1): float(prob) for i, prob in enumerate(pred_main)}
    special_probs = {int(i+1): float(prob) for i, prob in enumerate(pred_special)}

    return {
        "prediction_type": "next_draw",
        "input_last_date": last_sequence_draws[-1]['date'],
        "predicted_main_top10": top_10_main,
        "predicted_special_top3": top_3_special,
        "main_probabilities": main_probs,
        "special_probabilities": special_probs
    }
