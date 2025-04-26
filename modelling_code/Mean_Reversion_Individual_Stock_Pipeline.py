#%%
# !pip install yfinance hurst matplotlib statsmodels
# !pip install hurst
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from tools.utils import download_data, read_csv_by_pattern
warnings.filterwarnings("ignore")

#%%
# Mean Reversion Analysis
def adf_test(series):
    result = adfuller(series)
    return result[1]  # p-value

def calculate_hurst_exponent(series):
    H, _, _ = compute_Hc(series, kind="price", simplified=True)
    return H

# OU process
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

# Run analysis
def run_mean_reversion_analysis(prices, ticker_cap_mapping):
    results = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
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
            print(f"Error processing {ticker}: {e}")
    return pd.DataFrame(results)

# Strategy and Backtest
def mean_reversion_strategy(prices, window, z_thresh=1):
    mean = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std().replace(0, 1e-10)
    z_scores = (prices - mean) / std
    signals = pd.Series(0, index=prices.index)
    signals[z_scores < -z_thresh] = 1
    signals[z_scores > z_thresh] = -1
    return signals

def backtest_strategy(prices, signals):
    shifted_signals = signals.shift(1).fillna(0)
    log_returns = np.log(prices / prices.shift(1)).fillna(0)
    strategy_log_returns = shifted_signals * log_returns
    benchmark_total_return = np.exp(log_returns.sum()) - 1
    # cumulative_strategy = strategy_log_returns.cumsum().apply(np.exp)
    # cumulative_asset = log_returns.cumsum().apply(np.exp)
    return strategy_log_returns, benchmark_total_return #, cumulative_strategy, cumulative_asset

def calculate_performance(log_returns):
    total_log_return = log_returns.sum()
    annualized_return = total_log_return * (252 / len(log_returns))
    annualized_volatility = log_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan
    cumulative = log_returns.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()
    return pd.DataFrame([{
        "Total Return": np.exp(total_log_return) - 1,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": 1 - np.exp(max_drawdown)
    }])

# Run backtest
def run_backtest_all(prices, tickers, window_list, z_thresh=1.5):
    backtest_results = []
    for ticker in tickers:
        series = prices[ticker].dropna()
        for window in window_list:
            try:
                signals = mean_reversion_strategy(series, window, z_thresh)
                strat_returns, benchmark_total_return = backtest_strategy(series, signals)
                perf = calculate_performance(strat_returns).iloc[0]
                perf["Ticker"] = ticker
                perf["Window"] = window
                perf["Benchmark Total Return"] = benchmark_total_return
                backtest_results.append(perf)
            except Exception as e:
                print(f"Backtest failed for {ticker} with window {window}: {e}")
    return pd.DataFrame(backtest_results)


#%% Full pipeline
def full_mean_reversion_pipeline(tickers, start_date, end_date, ticker_cap_mapping, window_list, z_thresh=1.5):
    prices = download_data(tickers, start_date, end_date)
    # prices = pd.read_csv(f"prices_{start_date[:4]}_{end_date[:4]}.csv")
    # prices = prices[tickers]
    prices.to_csv(f"../modelling_result/prices_{start_date[:4]}_{end_date[:4]}.csv", index=False)
    stat_df = run_mean_reversion_analysis(prices, ticker_cap_mapping)
    backtest_df = run_backtest_all(prices, prices.columns.tolist(), window_list, z_thresh)
    df_report = pd.merge(stat_df, backtest_df, on="Ticker", how="left")
    df_report['Start'] = start_date
    df_report['End'] = end_date
    cols = ['Ticker', 'Start', 'End', 'Window', 'Benchmark Total Return'] + [col for col in df_report.columns if col not in ['Ticker', 'Start', 'End', 'Window', 'Benchmark Total Return']]
    df_report = df_report[cols]
    df_report.to_csv(f"../modelling_result/mean_reversion_full_report_{start_date[:4]}_{end_date[:4]}.csv", index=False)
    print("Full pipeline complete. Saved to CSV.")
    return df_report

#%% Compare results
def compare_results(df_report):
    df_report["Mean_Reverting"] = ((df_report['ADF p-value'] < 0.05) | (df_report['Hurst'] < 0.5))
    df_report["Outperform"] = df_report["Total Return"] > df_report["Benchmark Total Return"]
    comparison = df_report.groupby(["Mean_Reverting", "Outperform", "CapSize"]).agg(
        Count=("Ticker", "count"),
        Sharpe_Mean=("Sharpe Ratio", "mean"),
        Hurst_Mean=("Hurst", "mean"),
        ADF_pval_Mean=("ADF p-value", "mean"),
        Half_life_Mean=("Half-life", "mean"),
        Window_mean=("Window", "mean"),
        Volatility_Mean=("Volatility", "mean")
    )
    return comparison



#%% Parameters
start_date = '2020-01-01'
end_date = '2024-12-31'
large_cap_tickers = ['AAPL', 'MSFT', 'JPM', 'NVDA', 'XOM', 'JNJ', 'UNH', 'PG', 'V', 'MA', 'HD', 'COST', 'AVGO', 'LLY', 'BAC', 'MRK', 'ADBE']
mid_cap_tickers = ['TFX', 'HES', 'NTNX', 'WU', 'FIVE', 'GNRC', 'WING', 'CHDN', 'FND', 'HWM', 'CROX', 'ENPH', 'FICO', 'ROK', 'LII']
small_cap_tickers = ['INSM', 'NEOG', 'ACLS', 'PRDO', 'ORGO', 'IMMR', 'CVCO', 'GPRO', 'STRL', 'TPC', 'GHC', 'FIZZ', 'EVTC', 'CMTL', 'MGEE']
tickers = large_cap_tickers + mid_cap_tickers + small_cap_tickers

ticker_cap_mapping = {ticker: 'Large' for ticker in large_cap_tickers}
ticker_cap_mapping.update({ticker: 'Mid' for ticker in mid_cap_tickers})
ticker_cap_mapping.update({ticker: 'Small' for ticker in small_cap_tickers})
window_list = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

df_report = full_mean_reversion_pipeline(tickers, start_date, end_date, ticker_cap_mapping, window_list, z_thresh=1.5)
combined_df_report = read_csv_by_pattern(pattern="mean_reversion_full_report_*.csv")
result_comparision = compare_results(combined_df_report)


# %%
