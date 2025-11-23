# Lotto 5/35 Predictive System

## Project Overview
This project implements a predictive system for the Lotto 5/35 game using Deep Learning (LSTM/Transformer), Reinforcement Learning (RL), and Genetic Algorithms (GA).

## Components
*   **Data Ingestion**: Handles loading and cleaning of historical draw data.
*   **Feature Engineering**: Calculates statistical features (hot/cold, parity, sums, etc.).
*   **Prediction Model**: A Bi-LSTM model that outputs probabilities for the 5 main numbers and the special number.
*   **RL Agent**: Determines the betting strategy (skip, buy single, buy wheel).
*   **GA Optimizer**: Generates optimized ticket sets (wheels) based on model probabilities and coverage constraints.
*   **Backtesting**: Simulates performance over historical data.

## Setup
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Place `dataset_lotto_535.xlsx` in the `data/` directory.

## Usage
Run the main pipeline:
```bash
python src/main.py
```

## Docker
Build and run with Docker:
```bash
docker build -t lotto-predictor .
docker run lotto-predictor
```
