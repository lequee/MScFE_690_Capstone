# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc
import matplotlib.pyplot as plt


# Data definition
start_date = '2018-01-01'
end_date = '2024-12-31'

large_cap_tickers = ['AAPL', 'MSFT', 'JPM', 'NVDA', 'XOM', 'JNJ', 'UNH', 'PG', 'V', 'MA', 'HD', 'COST', 'AVGO', 'LLY', 'BAC', 'MRK', 'ADBE']
mid_cap_tickers = ['TFX', 'HES', 'NTNX', 'WU', 'FIVE', 'GNRC', 'WING', 'CHDN', 'FND', 'HWM', 'CROX', 'ENPH', 'FICO', 'ROK', 'LII']
small_cap_tickers = ['INSM', 'NEOG', 'ACLS', 'PRDO', 'ORGO', 'IMMR', 'CVCO', 'GPRO', 'STRL', 'TPC', 'GHC', 'FIZZ', 'EVTC', 'CMTL', 'MGEE']
tickers = large_cap_tickers + mid_cap_tickers + small_cap_tickers

ticker_cap_mapping = {ticker: 'Large' for ticker in large_cap_tickers}
ticker_cap_mapping.update({ticker: 'Mid' for ticker in mid_cap_tickers})
ticker_cap_mapping.update({ticker: 'Small' for ticker in small_cap_tickers})

# Download and prepare data
raw_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

adj_close = pd.DataFrame()
for ticker in tickers:
    try:
        adj_close[ticker] = raw_data[ticker]['Close']
    except:
        print(f"Skipping {ticker} due to missing data.")

adj_close = adj_close.dropna(axis=1)
log_prices = np.log(adj_close.replace(0, np.nan))

# Step 1: Statistical tests
def adf_test(series):
    result = adfuller(series)
    return result[1]  # p-value

def calculate_hurst_exponent(series):
    H, _, _ = compute_Hc(series, kind="price", simplified=True)
    return H

# Step 2: OU process
def compute_half_life(series):
    price_diff = np.diff(series)
    price_lag = series[:-1]
    beta = np.polyfit(price_lag, price_diff, 1)[0]
    half_life = -np.log(2) / beta if beta != 0 else np.nan
    return half_life

def fit_ornstein_uhlenbeck(series):
    mu = np.mean(series)
    sigma = np.std(series)
    half_life = compute_half_life(series)
    return {"mu": mu, "sigma": sigma, "half_life": half_life}

# Step 3: Run Analysis
results = []

for ticker in log_prices.columns:
    series = log_prices[ticker].dropna()
    try:
        adf_pval = adf_test(series)
        hurst = calculate_hurst_exponent(series)
        ou_params = fit_ornstein_uhlenbeck(series)
        volatility = series.diff().std()

        results.append({
            "Ticker": ticker,
            "CapSize": ticker_cap_mapping.get(ticker, "Unknown"),
            "ADF p-value": round(adf_pval, 4),
            "Hurst": round(hurst, 4),
            "Mu": round(ou_params["mu"], 4),
            "Sigma": round(ou_params["sigma"], 4),
            "Half-life": round(ou_params["half_life"], 4),
            "Volatility": round(volatility, 4)
        })

    except Exception as e:
        print(f"Error for {ticker}: {e}")

result_df = pd.DataFrame(results)
result_df.to_csv("mean_reversion_results.csv", index=False)
print("Analysis complete. Results saved to mean_reversion_results.csv.")

# Step 4: Trading Strategy
def bollinger_bands_strategy(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (num_std * rolling_std)
    lower_band = rolling_mean - (num_std * rolling_std)
    buy_signal = series < lower_band
    sell_signal = series > upper_band
    return buy_signal, sell_signal

def backtest_strategy(series, buy_signal, sell_signal, initial_capital=10000):
    capital = initial_capital
    position = 0
    for i in range(1, len(series)):
        if buy_signal[i]:
            position = capital / series[i]
            capital = 0
        elif sell_signal[i] and position > 0:
            capital = position * series[i]
            position = 0
    final_value = capital if capital > 0 else position * series.iloc[-1]
    return final_value, (final_value / initial_capital - 1) * 100

def plot_trading_signals(df, buy_signal, sell_signal):
    plt.figure(figsize=(12,6))
    plt.plot(df, label="Price", alpha=0.5)
    plt.scatter(df.index[buy_signal], df[buy_signal], marker="^", color="g", label="Buy", alpha=1)
    plt.scatter(df.index[sell_signal], df[sell_signal], marker="v", color="r", label="Sell", alpha=1)
    plt.title("Trading Signals")
    plt.legend()
    plt.show()