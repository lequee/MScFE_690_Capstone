import yfinance as yf
import pandas as pd
import numpy as np
import glob
import os

# Download data
def download_data(tickers, start_date, end_date):
    raw_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
    adj_close = pd.DataFrame()
    for ticker in tickers:
        try:
            adj_close[ticker] = raw_data[ticker]['Close']
        except:
            print(f"Skipping {ticker} due to missing data.")
    adj_close = adj_close.dropna(axis=1)
    return np.log(adj_close.replace(0, np.nan))


def read_csv_by_pattern(pattern="mean_reversion_full_report_*.csv"):
    file_pattern = '../modelling_result/'+ pattern
    csv_files = glob.glob(file_pattern)
    all_dfs = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df["SourceFile"] = os.path.basename(file)  # Optional
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    else:
        print("No matching CSV files found.")
        return pd.DataFrame()