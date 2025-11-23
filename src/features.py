import numpy as np
import pandas as pd

class FeatureEngine:
    def __init__(self, main_range=35, special_range=12, lookback=10):
        """
        Args:
            main_range: Max number for main set (35).
            special_range: Max number for special set (12).
            lookback: Number of past draws to use for prediction.
        """
        self.main_range = main_range
        self.special_range = special_range
        self.lookback = lookback
        self.total_features = main_range + special_range # 35 + 12 = 47

    def encode_draw(self, main_numbers, special_number):
        """
        Encodes a single draw into a vector of size 47.
        Part 1 (0-34): Multi-hot for main numbers (1-35 mapped to 0-34 indices).
        Part 2 (35-46): One-hot for special number (1-12 mapped to 0-11 indices).
        """
        # Initialize vector
        vector = np.zeros(self.total_features, dtype=np.float32)

        # Encode Main Numbers (1-35 -> indices 0-34)
        for num in main_numbers:
            if 1 <= num <= self.main_range:
                vector[int(num) - 1] = 1.0

        # Encode Special Number (1-12 -> indices 35-46)
        # Offset by self.main_range
        if 1 <= special_number <= self.special_range:
            vector[self.main_range + int(special_number) - 1] = 1.0

        return vector

    def create_dataset(self, draw_history):
        """
        Converts a list of draw dictionaries into X (sequences) and y (targets).

        X shape: (num_samples, lookback, 47)
        y_main shape: (num_samples, 35)
        y_special shape: (num_samples, 12)
        """
        # 1. Convert all draws to vectors
        vectors = []
        for draw in draw_history:
            vec = self.encode_draw(draw['main'], draw['special'])
            vectors.append(vec)

        vectors = np.array(vectors) # Shape: (total_draws, 47)

        X = []
        y_main = []
        y_special = []

        # 2. Create sequences
        # We need 'lookback' previous draws to predict the 'current' draw
        for i in range(self.lookback, len(vectors)):
            # Input sequence: i-lookback to i-1
            seq_x = vectors[i - self.lookback : i]

            # Target: i
            target_vec = vectors[i]

            # Split target into Main and Special parts
            # Main part: first 35
            target_main = target_vec[:self.main_range]
            # Special part: last 12
            target_special = target_vec[self.main_range:]

            X.append(seq_x)
            y_main.append(target_main)
            y_special.append(target_special)

        return np.array(X), np.array(y_main), np.array(y_special)

if __name__ == "__main__":
    # Test
    from src.ingestion import DataLoader
    loader = DataLoader("data/dataset.csv")
    history = loader.get_draw_history()

    engine = FeatureEngine(lookback=5)
    X, y1, y2 = engine.create_dataset(history)

    print("X shape:", X.shape)
    print("y_main shape:", y1.shape)
    print("y_special shape:", y2.shape)

    # Validation
    print("\nSample check:")
    print("Last input sequence (last step):\n", X[-1][-1]) # Should match 2nd to last draw
    print("Last target main:\n", y1[-1]) # Should match last draw
