import pandas as pd
import numpy as np

def is_prime(n):
    if n <= 1: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

PRIMES = set([i for i in range(1, 36) if is_prime(i)])

def calculate_features(df):
    """
    Calculates features for the Lotto 5/35 dataset.
    df: DataFrame with num1..num5 and special
    """
    df = df.copy()

    # Main numbers columns
    main_cols = ['num1', 'num2', 'num3', 'num4', 'num5']

    # 1. Sum
    df['sum_main'] = df[main_cols].sum(axis=1)

    # 2. Parity Ratio (Count of Even numbers)
    df['even_count'] = df[main_cols].apply(lambda x: sum(1 for n in x if n % 2 == 0), axis=1)
    df['odd_count'] = 5 - df['even_count']

    # 3. Prime Count
    df['prime_count'] = df[main_cols].apply(lambda x: sum(1 for n in x if n in PRIMES), axis=1)

    # 4. Consecutive Count
    # Sort numbers first to be safe (though usually sorted in lotto results)
    def count_consecutive(row):
        nums = sorted(row.values)
        count = 0
        for i in range(len(nums)-1):
            if nums[i+1] == nums[i] + 1:
                count += 1
        return count

    df['consecutive_count'] = df[main_cols].apply(count_consecutive, axis=1)

    # 5. Decade Distribution
    # 1-10, 11-20, 21-30, 31-35
    df['decade_1'] = df[main_cols].apply(lambda x: sum(1 for n in x if 1 <= n <= 10), axis=1)
    df['decade_2'] = df[main_cols].apply(lambda x: sum(1 for n in x if 11 <= n <= 20), axis=1)
    df['decade_3'] = df[main_cols].apply(lambda x: sum(1 for n in x if 21 <= n <= 30), axis=1)
    df['decade_4'] = df[main_cols].apply(lambda x: sum(1 for n in x if 31 <= n <= 35), axis=1)

    # 6. Frequency / Hotness (Rolling count over last N draws)
    # We need to construct a "presence" matrix for this
    # Let's create a DataFrame where cols are 1-35 and value is 1 if present

    # One-hot encode the main numbers for history tracking
    one_hot = np.zeros((len(df), 36)) # 0 index unused
    for idx, row in df.iterrows():
        for col in main_cols:
            val = row[col]
            one_hot[idx, val] = 1

    presence_df = pd.DataFrame(one_hot, index=df.index, columns=[f'num_{i}' for i in range(36)])

    # Calculate rolling frequency (e.g., last 10, 20 draws)
    # Note: shift(1) because we want features based on *past* draws, not including current
    # But for training, we use past history to predict current.
    # Usually we construct the dataset such that X_t comes from draws [t-k, ..., t-1].
    # So we compute rolling sums on the presence_df and then shift.

    windows = [10, 30]
    for w in windows:
        rolling_sum = presence_df.rolling(window=w).sum()
        # The rolling sum at index t includes t. For feature at t (predicting t), we need info up to t-1.
        # But wait, standard feature engineering adds columns to the row.
        # If we use these columns as features for target `row`, they must be derived from `row-1` backwards.
        # So we will shift *after* calculating, or just calculate on shifted data.
        # Let's calculate the "hotness coming into this draw"
        # So we shift the rolling sum by 1.

        shifted_rolling = rolling_sum.shift(1)

        # Add summary stats of hotness as features (e.g. average hotness of winning numbers - cheating?)
        # No, we want features describing the state of the game.
        # So we can add "Hotness of number X" as 35 features? That's a lot.
        # The prompt asks for "Feature pipeline" and "normalize/scale".
        # For the LSTM, we usually feed the sequence of raw numbers or one-hot vectors.
        # The "features" listed in PDF might be auxiliary inputs or used for RL state.

        # Let's attach the specific "Recency" / "Gap" for all 35 numbers?
        # That's a 35-dim vector.
        pass

    # 6. Hotness (Frequency in last 10 draws)
    # We calculated presence_df earlier.
    presence_df = pd.DataFrame(one_hot, index=df.index, columns=[f'num_{i}' for i in range(36)])
    rolling_freq = presence_df.rolling(window=10).sum().shift(1).fillna(0)
    # This gives 36 columns (0 unused). We want columns 1-35.
    hotness_cols = [f'hotness_{i}' for i in range(1, 36)]
    hotness_df = pd.DataFrame(rolling_freq.values[:, 1:], index=df.index, columns=hotness_cols)
    df = pd.concat([df, hotness_df], axis=1)

    # 7. Gap / Recency
    # For each number, how many draws since it last appeared?
    # This is a cumulative count that resets when number appears.
    gaps = np.zeros((len(df), 36))
    last_seen = np.zeros(36) - 1 # initialized to -1

    # Iterate to fill gaps
    # This is slow in pure python but fine for 250 rows.
    for i in range(len(df)):
        # Calculate gaps for this row based on history
        # (Gap is distance from last_seen to i)
        # If never seen, it's i + some large number or just i.

        current_gaps = np.zeros(36)
        for n in range(1, 36):
            if last_seen[n] == -1:
                current_gaps[n] = i # effectively 'infinity' or just 'all time'
            else:
                current_gaps[n] = i - last_seen[n] - 1

        gaps[i] = current_gaps

        # Update last_seen for the numbers in this draw
        for col in main_cols:
            val = int(df.iloc[i][col])
            last_seen[val] = i

    # Shift gaps because gap at time t should be based on history t-1
    # Actually the loop calculated gaps *before* updating last_seen with current draw?
    # No, I calculated current_gaps using last_seen (past), THEN updated last_seen.
    # So gaps[i] represents the gap *coming into* draw i. This is correct for features.

    gap_df = pd.DataFrame(gaps[:, 1:], index=df.index, columns=[f'gap_{i}' for i in range(1, 36)])

    # Concatenate features
    df = pd.concat([df, gap_df], axis=1)

    return df

def create_sequences(df, lookback=50, is_training=True, scaler=None):
    """
    Prepares sequences for LSTM.
    X: (samples, lookback, features)
    y: (samples, 35) (multi-hot for main) + (samples, 1) (special)
    """
    # Features to include
    # Now including Hotness (35) and Gap (35) and basic stats (8)
    basic_cols = [
        'sum_main', 'even_count', 'prime_count', 'consecutive_count',
        'decade_1', 'decade_2', 'decade_3', 'decade_4'
    ]
    hotness_cols = [f'hotness_{i}' for i in range(1, 36)]
    gap_cols = [f'gap_{i}' for i in range(1, 36)]

    feature_cols = basic_cols + hotness_cols + gap_cols

    # Check if features exist, if not calculate them
    if 'sum_main' not in df.columns:
        df = calculate_features(df)

    # Normalize features
    # To avoid data leakage, we should use a scaler provided from outside or fit on valid data
    # For this MVP, we will do a simple robust scaling (value / max_possible)
    # Sum: max 35*5 approx 160.
    # Even count: max 5
    # Hotness: max window (10)
    # Gap: max len(df)

    scaled_df = df.copy()

    # Manual scaling to keep it simple and stateless-ish for this MVP function
    # In production, use sklearn StandardScaler loaded from disk

    scaled_df['sum_main'] = scaled_df['sum_main'] / 175.0
    scaled_df['even_count'] = scaled_df['even_count'] / 5.0
    scaled_df['prime_count'] = scaled_df['prime_count'] / 5.0
    scaled_df['consecutive_count'] = scaled_df['consecutive_count'] / 5.0
    for c in ['decade_1', 'decade_2', 'decade_3', 'decade_4']:
        scaled_df[c] = scaled_df[c] / 5.0

    for c in hotness_cols:
        scaled_df[c] = scaled_df[c] / 10.0 # window size

    for c in gap_cols:
        scaled_df[c] = np.log1p(scaled_df[c]) / 5.0 # Log scale for gaps

    main_cols = ['num1', 'num2', 'num3', 'num4', 'num5']
    special_col = 'special'

    # Convert main numbers to multi-hot encoding
    # shape: (N, 35)
    data_len = len(df)
    multi_hot = np.zeros((data_len, 35))

    for i in range(data_len):
        for col in main_cols:
            val = df.iloc[i][col] - 1 # 0-indexed
            multi_hot[i, val] = 1

    special = df[special_col].values - 1 # 0-indexed for 1-12

    # Extract auxiliary features
    aux_features = scaled_df[feature_cols].values # shape (N, num_features)

    # Combine Multi-hot + Aux Features
    # Input vector at each timestep t will be [Multi-hot (35) | Aux Features (8)]
    combined_features = np.concatenate([multi_hot, aux_features], axis=1)

    X = []
    y_main = []
    y_special = []

    for i in range(lookback, data_len):
        X.append(combined_features[i-lookback:i]) # The sequence of past draws
        y_main.append(multi_hot[i])       # The current draw (target)
        y_special.append(special[i])

    return np.array(X), np.array(y_main), np.array(y_special)

if __name__ == "__main__":
    from ingestion.loader import load_data
    df = load_data()
    df_feat = calculate_features(df)
    print("Features calculated.")
    print(df_feat.head())

    X, y1, y2 = create_sequences(df_feat, lookback=10)
    print(f"Sequences shape: X={X.shape}, y_main={y1.shape}, y_special={y2.shape}")
