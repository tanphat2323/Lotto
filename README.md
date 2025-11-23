# Lotto 5/35 AI Predictive System (MVP)

*[Tiếng Việt]: Xem hướng dẫn cài đặt chi tiết tại [GUIDE_VI.md](GUIDE_VI.md)*

This project is an AI-powered prediction engine for the "Lotto 5/35" lottery game. It uses a **Bidirectional LSTM** neural network to analyze historical draw sequences and predict the most likely numbers for the next draw.

## Features (Phase 1 MVP)

*   **Deep Learning Model:** LSTM-based architecture trained on historical sequences.
*   **Dual-Head Prediction:**
    *   **Main Numbers:** Predicts probabilities for all 35 main numbers (Multi-label).
    *   **Special Number:** Predicts probabilities for the 12 special numbers (Multi-class).
*   **REST API:** Fast & Async API to serve predictions and statistics.
*   **Windows Compatible:** Fully tested instructions for Windows environments.

## Prerequisites

*   **Python 3.10+** (Ensure Python is added to your PATH).
*   **Windows PowerShell** or **Command Prompt**.

## Installation

1.  **Clone or Download** this repository to your local machine.

2.  **Create a Virtual Environment** (Recommended):
    Open your terminal in the project folder and run:
    ```powershell
    python -m venv .venv
    ```

3.  **Activate the Environment**:
    ```powershell
    .venv\Scripts\activate
    ```
    *(You should see `(.venv)` at the beginning of your command prompt)*

4.  **Install Dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```

## Usage

### 1. Training the Model
Before running the API, you must train the model on the provided dataset.

```powershell
python src/train.py
```
*   This will read `data/dataset.csv`.
*   Train the LSTM model for 100 epochs.
*   Save the best model to `models/lotto_model.keras`.
*   Save performance metrics to `models/latest_metrics.json`.

### 2. Running the API
Start the web server to serve predictions:

```powershell
uvicorn src.main:app --reload
```

You should see output indicating the server is running at `http://127.0.0.1:8000`.

### 3. Getting Predictions
You can interact with the API using a browser or tools like `curl`.

*   **Predict Next Draw:**
    Open your browser to: `http://127.0.0.1:8000/predict/next`

    Response Example:
    ```json
    {
      "prediction_type": "next_draw",
      "predicted_main_top10": [5, 12, 33, 1, 28, ...],
      "predicted_special_top3": [7, 12, 1],
      "main_probabilities": { ... },
      "special_probabilities": { ... }
    }
    ```

*   **View Model Stats:**
    `http://127.0.0.1:8000/stats`

## Project Structure

*   `data/`: Contains `dataset.csv`.
*   `src/`: Source code.
    *   `ingestion.py`: Data loading.
    *   `features.py`: Sequence generation (sliding window).
    *   `model.py`: Neural Network definition (Keras).
    *   `train.py`: Training script.
    *   `main.py`: API server.
*   `models/`: Stores trained model files.

## Future Plans (Phase 2)
*   **EV & Rolldown:** Integration of jackpot values to calculate Expected Value.
*   **Reinforcement Learning:** An agent to optimize betting strategies based on risk/reward.
*   **React Dashboard:** A full UI for visualization.
