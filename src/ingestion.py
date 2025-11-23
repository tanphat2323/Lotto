import pandas as pd
import numpy as np
import os
from datetime import datetime

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_data = None
        self.processed_data = None

    def load_data(self):
        """Loads the CSV file and performs basic cleaning."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        # Load CSV
        # Expected columns: date, id, result/0..4, db
        df = pd.read_csv(self.filepath)

        # Rename columns for clarity
        # result/0 -> main_1, ..., db -> special
        column_mapping = {
            'result/0': 'main_1',
            'result/1': 'main_2',
            'result/2': 'main_3',
            'result/3': 'main_4',
            'result/4': 'main_5',
            'db': 'special',
            'id': 'draw_id'
        }
        df.rename(columns=column_mapping, inplace=True)

        # Sort by date/id just in case
        df.sort_values(by=['draw_id'], inplace=True)
        self.raw_data = df
        return df

    def get_draw_history(self):
        """Returns the chronological list of draws."""
        if self.raw_data is None:
            self.load_data()

        # Extract only the numbers
        main_cols = ['main_1', 'main_2', 'main_3', 'main_4', 'main_5']
        special_col = 'special'

        history = []
        for _, row in self.raw_data.iterrows():
            draw = {
                'draw_id': row['draw_id'],
                'date': row['date'],
                'main': [row[c] for c in main_cols],
                'special': row[special_col]
            }
            history.append(draw)
        return history

if __name__ == "__main__":
    # Test execution
    loader = DataLoader("data/dataset.csv")
    df = loader.load_data()
    print("Columns:", df.columns)
    print("First 5 rows:\n", df.head())
