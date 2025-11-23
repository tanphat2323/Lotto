import os
import json
import numpy as np
import tensorflow as tf
from src.ingestion import DataLoader
from src.features import FeatureEngine
from src.model import build_lotto_model
from src.metrics import calculate_top_k_hit_rate, calculate_special_accuracy

# Constants
DATA_PATH = "data/dataset.csv"
MODEL_DIR = "models"
LOOKBACK = 10
EPOCHS = 100
BATCH_SIZE = 16

def train():
    # 1. Load Data
    print("Loading data...")
    loader = DataLoader(DATA_PATH)
    history = loader.get_draw_history()

    # 2. Prepare Features
    print("Preparing features...")
    engine = FeatureEngine(lookback=LOOKBACK)
    X, y_main, y_special = engine.create_dataset(history)

    # Simple Time-based Split (Train 80%, Test 20%)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_main_train, y_main_test = y_main[:split_idx], y_main[split_idx:]
    y_special_train, y_special_test = y_special[:split_idx], y_special[split_idx:]

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 3. Build Model
    model = build_lotto_model(lookback=LOOKBACK)

    # 4. Train
    print("Starting training...")
    history_callback = model.fit(
        X_train,
        {'output_main': y_main_train, 'output_special': y_special_train},
        validation_data=(X_test, {'output_main': y_main_test, 'output_special': y_special_test}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # 5. Evaluate Custom Metrics
    print("Evaluating custom metrics on Test set...")
    preds = model.predict(X_test)
    pred_main = preds[0]
    pred_special = preds[1]

    avg_hits, hit_prob = calculate_top_k_hit_rate(y_main_test, pred_main, k=10)
    special_acc_top3 = calculate_special_accuracy(y_special_test, pred_special, k=3)

    metrics = {
        "test_avg_hits_top10_main": float(avg_hits),
        "test_hit_rate_any_top10_main": float(hit_prob),
        "test_special_acc_top3": float(special_acc_top3)
    }
    print("Metrics:", json.dumps(metrics, indent=2))

    # 6. Save Artifacts
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model_path = os.path.join(MODEL_DIR, "lotto_model.keras")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    metrics_path = os.path.join(MODEL_DIR, "latest_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    train()
