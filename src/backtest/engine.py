import pandas as pd
import numpy as np
from src.ingestion.loader import load_data
from src.features.engineer import calculate_features, create_sequences
from src.model.lotto_model import LottoPredictor
from src.ga.optimizer import WheelOptimizer
from src.rl.environment import LottoEnv
from src.rl.agent import PolicyGradientAgent

def check_win(ticket, result_numbers):
    """
    ticket: list of 5 numbers
    result_numbers: list of 5 numbers
    """
    match = len(set(ticket) & set(result_numbers))
    return match

def calculate_payout(match_count, prize_table):
    return prize_table.get(match_count, 0)

def run_backtest(data_path='data/dataset_lotto_535.xlsx', lookback=10):
    # 1. Load Data
    print("Loading data...")
    df = load_data(data_path)
    df = calculate_features(df)

    # 2. Split Data (Train / Test)
    # We will do a rolling window backtest or just a simple split for the skeleton
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # 3. Initial Train
    print("Training initial model...")
    X_train, y_main_train, y_special_train = create_sequences(train_df, lookback=lookback)

    # Input shape is now (lookback, 43) because of 35 multi-hot + 8 engineered features
    input_features = X_train.shape[2]
    model = LottoPredictor(input_shape=(lookback, input_features))
    model.train(X_train, y_main_train, y_special_train, epochs=2, verbose=0) # Fast train for test

    # 4. Backtest Loop
    total_cost = 0
    total_winnings = 0

    prize_table = {
        5: 12000000000,
        4: 3000000,
        3: 30000,
        2: 0
    }

    results_log = []

    print("Starting backtest loop...")
    # Iterate through test set
    # Note: We need 'lookback' previous frames to predict current.
    # The 'test_df' starts at split_idx. The input for predicting split_idx comes from split_idx-lookback...split_idx-1

    full_seq_X, _, _ = create_sequences(df, lookback=lookback)
    # create_sequences returns X starting from lookback index.
    # X[0] predicts y[lookback]
    # We want to predict for indices in test_df.
    # test_df starts at row `split_idx`.
    # This corresponds to y index `split_idx` in the original df.
    # In the `create_sequences` output, y starts at `lookback`.
    # So y[i] corresponds to df index `lookback + i`.
    # We want df index `k` where `k >= split_idx`.
    # So `lookback + i >= split_idx` => `i >= split_idx - lookback`.

    start_seq_idx = split_idx - lookback
    if start_seq_idx < 0: start_seq_idx = 0

    # Prepare RL agent
    env = LottoEnv(df, prize_table=prize_table)
    # Obs shape is (lookback * 35), wait, environment creates a flattened obs.
    # Env uses self.lookback * 35.
    # But now our sequences are 113 features.
    # The Env is currently returning zeros(lookback*35).
    # We should update Env to match features or just use what Env returns.
    # The agent expects input_dim.
    agent = PolicyGradientAgent(env.action_space, input_dim=env.observation_space.shape[0])

    for i in range(start_seq_idx, len(full_seq_X)):
        # Current input sequence
        current_X = full_seq_X[i:i+1] # shape (1, lookback, 113)

        # Actual result for this draw
        original_idx = lookback + i
        actual_row = df.iloc[original_idx]
        actual_numbers = [actual_row[f'num{k}'] for k in range(1, 6)]

        # 1. Get Model Probabilities
        preds = model.predict(current_X)
        probs_main = preds[0][0] # shape (35,)
        probs_special = preds[1][0]

        # 2. RL Agent Decision
        # Get obs from env (simulated step or actual data)
        # In backtest we need to feed the ACTUAL state to the agent.
        # The Env class is designed for training loop where it steps internally.
        # Here we are iterating externally.
        # Let's construct a flattened state from current_X.
        # current_X is (1, lookback, 113).
        # We'll flatten it to match agent input.
        # Note: Env.observation_space is lookback*35, but we have lookback*113.
        # We need to resize agent or slice features.
        # Let's just use the full flattened features.

        current_state = current_X.flatten()
        # Resize Agent if needed (hack for skeleton)
        if agent.input_dim != len(current_state):
             agent.input_dim = len(current_state)
             agent.W = np.resize(agent.W, (len(current_state), agent.action_space.n))

        action = agent.predict(current_state)

        # 3. Execute Decision
        tickets_bought = []

        if action == 0: # Skip
            pass
        elif action == 1: # Buy 1
            # Simple strategy: top 5 probs
            top_indices = np.argsort(probs_main)[-5:]
            ticket = sorted([x + 1 for x in top_indices])
            tickets_bought.append(ticket)
        elif action == 2 or action == 3: # Wheel
            budget = 5 if action == 2 else 10
            ga = WheelOptimizer(probs_main, probs_special, ticket_budget=budget, generations=5)
            best_wheel, _ = ga.optimize()
            tickets_bought.extend(best_wheel)

        # 4. Calculate PnL
        step_cost = len(tickets_bought) * 10000
        step_winnings = 0

        for t in tickets_bought:
            matches = check_win(t, actual_numbers)
            win_amt = calculate_payout(matches, prize_table)
            step_winnings += win_amt

        total_cost += step_cost
        total_winnings += step_winnings

        # Give reward to Agent
        # Reward = Winnings - Cost
        agent.store_reward(step_winnings - step_cost)

        # Update Agent (every step or every episode? usually episode, but here continuous)
        # Let's update every 10 steps to simulate mini-batches or 'episodes'
        if i % 10 == 0:
            agent.update()

        results_log.append({
            'date': actual_row['date'],
            'action': action,
            'cost': step_cost,
            'winnings': step_winnings,
            'matches': [check_win(t, actual_numbers) for t in tickets_bought] if tickets_bought else []
        })

        # 5. Online Learning and Drift Detection
        # Update model every step (Incremental Learning)
        # We need the ground truth for this step: y_main, y_special

        # Prepare targets
        # y_main: (1, 35) multi-hot
        y_main_step = np.zeros((1, 35))
        for num in actual_numbers:
            y_main_step[0, num-1] = 1

        y_special_step = np.array([actual_row['special'] - 1])

        # Monitor Drift: Calculate loss on this new sample before training
        # If loss is high, it indicates drift or anomaly
        # For simplicity, we just use the evaluate method or predict prob
        # loss = model.model.evaluate(current_X, {'main_output': y_main_step, 'special_output': y_special_step}, verbose=0)[0]
        # if loss > threshold: print("Drift detected!")

        # Train on this single new sample (Online Learning)
        # In practice, use a small buffer/batch, but here SGD style
        # Disable validation split for single sample update
        model.train(current_X, y_main_step, y_special_step, epochs=1, batch_size=1, validation_split=0.0, verbose=0)

    # Summary
    roi = (total_winnings - total_cost) / total_cost if total_cost > 0 else 0
    print(f"Backtest Complete.")
    print(f"Total Cost: {total_cost}")
    print(f"Total Winnings: {total_winnings}")
    print(f"ROI: {roi:.2%}")

    return pd.DataFrame(results_log)

if __name__ == "__main__":
    run_backtest()
