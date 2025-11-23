import argparse
import sys
import os

# Ensure project root is in path for absolute imports (e.g. from src.ingestion...)
project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.append(project_root)

# Also append src if relative imports are used there, but better to rely on absolute 'src.x'
sys.path.append(os.path.join(project_root, 'src'))

from src.ingestion.loader import load_data
from src.backtest.engine import run_backtest

def main():
    parser = argparse.ArgumentParser(description="Lotto 5/35 Predictive System")
    parser.add_argument('--mode', type=str, choices=['backtest', 'train', 'predict'], default='backtest', help='Mode to run')
    parser.add_argument('--data', type=str, default='data/dataset_lotto_535.xlsx', help='Path to dataset')

    args = parser.parse_args()

    if args.mode == 'backtest':
        print("Running Backtest...")
        try:
            results = run_backtest(data_path=args.data)
            print("Backtest finished successfully.")
            print(results.tail())
        except Exception as e:
            print(f"Error during backtest: {e}")
            import traceback
            traceback.print_exc()

    elif args.mode == 'train':
        print("Training mode not fully exposed in CLI yet. See notebooks.")

    elif args.mode == 'predict':
        print("Predict mode not fully exposed in CLI yet. See notebooks.")

if __name__ == "__main__":
    main()
