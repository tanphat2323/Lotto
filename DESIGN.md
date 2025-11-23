# Lotto 5/35 AI Predictive System - Design Document

## 1. System Overview
This system is an AI-powered engine designed to analyze and predict the outcomes of the "Lotto 5/35" lottery game. The core utilizes a Deep Learning model (LSTM) to learn temporal patterns from historical draw data.

**Phase 1 Goal:** Establish a working MVP that predicts the probability of winning numbers (5 Main + 1 Special) based on historical sequences, exposed via a REST API.
**Phase 2 Goal (Future):** Integrate Expected Value (EV) calculation, Rolldown simulation, and Reinforcement Learning (RL) for betting strategy optimization.

## 2. Architecture

### 2.1 Technology Stack
*   **Language:** Python 3.10+
*   **Web Framework:** FastAPI (Async, High Performance)
*   **ML Engine:** TensorFlow / Keras
*   **Data Processing:** Pandas, NumPy
*   **Storage:** Local Files (CSV for data, `.h5`/`.keras` for models) -> *Extensible to SQLite/Postgres*

### 2.2 Directory Structure
```
/
├── data/               # Raw datasets (dataset.csv)
├── models/             # Saved model artifacts (*.keras, metrics.json)
├── src/
│   ├── main.py         # FastAPI Entry point
│   ├── ingestion.py    # Data loading and cleaning
│   ├── features.py     # Feature engineering (Sliding windows, One-hot)
│   ├── model.py        # LSTM Architecture definition
│   ├── train.py        # Training pipeline
│   └── metrics.py      # Custom evaluation metrics (Hit Rate)
├── requirements.txt    # Dependencies
└── README.md           # Instructions (Windows focus)
```

## 3. Data Pipeline
The system treats the lottery draw history as a **Time-Series Classification** problem.

### 3.1 Inputs
*   **Source:** `dataset.csv`
*   **Schema:** `Date`, `Draw ID`, `Main1`...`Main5` (1-35), `Special` (1-12).

### 3.2 Feature Engineering
*   **Representation:**
    *   **Main Numbers:** Multi-hot encoding vector of size 35 (0 or 1).
    *   **Special Number:** One-hot encoding vector of size 12 (0 or 1).
    *   **Combined Feature Vector:** Size 47 per time step.
*   **Sequence Generation:**
    *   **Lookback Window:** $N$ previous draws (e.g., $N=10$) are used to predict the $N+1$ draw.
    *   **Shape:** `(Batch_Size, Lookback_Window, 47)`

## 4. Model Architecture (LSTM)
We use a multi-output Neural Network to handle the two distinct parts of the game (Main numbers vs Special number).

*   **Input Layer:** Shape `(Lookback, 47)`
*   **Backbone:**
    *   Bidirectional LSTM (64 units) -> Return Sequences
    *   LSTM (64 units) -> Last Hidden State
*   **Head 1 (Main Numbers):**
    *   Dense(35 units)
    *   Activation: `sigmoid` (Independent probabilities for each number 1-35)
    *   Loss: `binary_crossentropy`
*   **Head 2 (Special Number):**
    *   Dense(12 units)
    *   Activation: `softmax` (Probability distribution over 1-12)
    *   Loss: `categorical_crossentropy`

## 5. Evaluation Metrics (Phase 1)
Since "Accuracy" is misleading in lottery (low probability), we use:
1.  **Hit Rate (Recall):** Percentage of actual winning numbers found in the Top-K predicted numbers.
2.  **Top-K Accuracy:**
    *   Does the actual Special Number fall within the Top-3 predicted probabilities?
    *   Do the actual 5 Main Numbers fall within the Top-10 predicted probabilities?

## 6. Future Extensibility (Phase 2 Preparation)
*   **EV Calculation:** The API response structure will include a placeholder for `ev_analysis`. Currently `null`, but ready to be populated when Prize/Jackpot data is available.
*   **RL Agent:** The `src/model.py` can be wrapped into a Gymnasium environment where the State is the `(Lookback, 47)` vector and Action is the ticket selection.
