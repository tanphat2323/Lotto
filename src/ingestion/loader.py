import pandas as pd
import os

def load_data(filepath='data/dataset_lotto_535.xlsx'):
    """
    Loads the lotto dataset.
    Handles potential format issues where CSV data is pasted into an Excel column.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        # Fallback if it's actually a CSV file named .xlsx (unlikely but possible)
        try:
            df = pd.read_csv(filepath)
        except:
            raise e

    # Check if data is mashed into one column
    if len(df.columns) == 1 and isinstance(df.iloc[0, 0], str) and ',' in df.iloc[0, 0]:
        # Expand the single column
        col_name = df.columns[0]
        # It seems the header itself is also in the first column name 'date,id,...'
        # Let's try to parse it.

        # If the header is actually the column name
        header_str = df.columns[0]
        headers = header_str.split(',')

        # Split the data
        expanded_data = df.iloc[:, 0].str.split(',', expand=True)

        if len(headers) == expanded_data.shape[1]:
            expanded_data.columns = headers
            df = expanded_data
        else:
            # Fallback: maybe the first row is data
            pass

    # Clean up column names
    df.columns = [c.strip() for c in df.columns]

    # Rename columns to standard format if needed
    # Expected: date, id, result/0...result/4, db
    # We want: date, id, num1, num2, num3, num4, num5, special

    # Standardize column names
    # Possible formats: 'result/0'.. or 'result1'..

    cols = df.columns
    rename_map = {}

    if 'result/0' in cols:
        rename_map.update({
            'result/0': 'num1', 'result/1': 'num2', 'result/2': 'num3',
            'result/3': 'num4', 'result/4': 'num5'
        })
    elif 'result1' in cols:
        rename_map.update({
            'result1': 'num1', 'result2': 'num2', 'result3': 'num3',
            'result4': 'num4', 'result5': 'num5'
        })

    if 'db' in cols:
        rename_map['db'] = 'special'

    df = df.rename(columns=rename_map)

    # Convert types
    for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'special']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

    # Drop rows with NaN in critical columns
    df = df.dropna(subset=['num1', 'num2', 'num3', 'num4', 'num5', 'special'])

    # Ensure integers
    for col in ['num1', 'num2', 'num3', 'num4', 'num5', 'special']:
        df[col] = df[col].astype(int)

    return df.reset_index(drop=True)

if __name__ == "__main__":
    df = load_data()
    print("Data loaded successfully.")
    print(df.info())
    print(df.head())
