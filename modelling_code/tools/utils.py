import yfinance as yf
import pandas as pd
import numpy as np
import glob
import os
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc

# Download data
def download_data(tickers, start_date, end_date):
    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)
        adj_close = pd.DataFrame()
        for ticker in tickers:
            try:
                adj_close[ticker] = raw_data[ticker]['Close']
            except:
                print(f"Skipping {ticker} due to missing data.")
    except Exception as e:
        print(f"Yahoo download failed completely: {e}")
        adj_close = pd.DataFrame()
    adj_close = adj_close.dropna(axis=1)
    adj_close = np.log(adj_close.replace(0, np.nan))
    # If data exists but is empty (e.g., no rows), treat as failed
    if adj_close.empty or adj_close.shape[0] == 0:
        print("Yahoo Finance returned no data. Falling back to local CSV.")
        adj_close = pd.DataFrame()  # Clear and fallback
    # If online data is missing or incomplete, fall back to local CSV
    missing_tickers = [t for t in tickers if t not in adj_close.columns]
    if missing_tickers:
        try:
            local_data = pd.read_csv(f"../data_prices/prices_{start_date[:4]}_{end_date[:4]}.csv")
            for ticker in missing_tickers:
                if ticker in local_data.columns:
                    adj_close[ticker] = local_data[ticker]
                else:
                    print(f"{ticker} not found in local CSV.")
        except Exception as e:
            print(f"Failed to load local CSV: {e}")
    adj_close = adj_close.dropna(axis=1, how='all')
    return adj_close

def get_sector_info(ticker_list):
    data = []
    for ticker in ticker_list:
        try:
            info = yf.Ticker(ticker).info
            data.append({
                'Ticker': ticker,
                'Sector': info.get('sector', 'N/A'),
                'Industry': info.get('industry', 'N/A')
            })
        except Exception as e:
            data.append({
                'Ticker': ticker,
                'Sector': 'Error',
                'Industry': str(e)
            })
    
    return pd.DataFrame(data)


# Mean Reversion Analysis
def adf_test(series):
    result = adfuller(series)
    return result[1]  # p-value

def calculate_hurst_exponent(series, strategy_type):
    if strategy_type == "price":
        H, _, _ = compute_Hc(series, kind="price", simplified=True)
    if strategy_type == "spread":
        H, _, _ = compute_Hc(series, kind="change", simplified=True)
    return H

def fit_ornstein_uhlenbeck(series):
    price_diff = np.diff(series)
    price_lag = series[:-1]
    beta = np.polyfit(price_lag, price_diff, 1)[0]
    half_life = -np.log(2) / beta if beta != 0 else np.nan
    return {"half_life": half_life}


## Run analysis
def run_mean_reversion_analysis(prices, metadata_df=None, strategy_type = "price"):
    results = []

    for ticker in prices.columns:
        series = prices[ticker].dropna()
        try:
            adf_pval = adf_test(series)
            hurst = calculate_hurst_exponent(series, strategy_type = strategy_type)
            ou_params = fit_ornstein_uhlenbeck(series)
            volatility = series.diff().std()

            results.append({
                "Ticker": ticker,
                "ADF p-value": round(adf_pval, 4),
                "Hurst": round(hurst, 4),
                "Half-life": round(ou_params["half_life"], 4),
                "Volatility": round(volatility, 4)
            })

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    df_result = pd.DataFrame(results)

    if metadata_df is not None:
        df_result = df_result.merge(metadata_df, on="Ticker", how="left")

    return df_result


# Result analysis
def read_csv_by_pattern(pattern="mean_reversion_individual_full_report_*.csv"):
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
    
#%%
