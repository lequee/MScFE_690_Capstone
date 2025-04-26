# %% Pair Trading
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc
from statsmodels.tsa.stattools import coint
from tools.utils import download_data

def run_pair_trading_analysis(prices, max_pairs=20):
    tickers = prices.columns
    pairs_results = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            y, x = prices[t1], prices[t2]
            y, x = y.dropna(), x.dropna()
            common_idx = y.index.intersection(x.index)
            y, x = y.loc[common_idx], x.loc[common_idx]

            if len(y) < 100:  # skip short series
                continue

            try:
                coint_pval = coint(y, x)[1]
                if coint_pval < 0.05:
                    spread = y - np.polyfit(x, y, 1)[0] * x
                    zscore = (spread - spread.mean()) / spread.std()
                    sharpe = (zscore.shift(1) * spread.diff()).mean() / (spread.diff().std() + 1e-8) * np.sqrt(252)

                    pairs_results.append({
                        "Stock1": t1,
                        "Stock2": t2,
                        "Cointegration p-value": round(coint_pval, 4),
                        "Sharpe (Z*ΔSpread)": round(sharpe, 4),
                        "Mean Spread": round(spread.mean(), 4),
                        "Std Spread": round(spread.std(), 4),
                        "Start": spread.index.min(),
                        "End": spread.index.max()
                    })

            except Exception as e:
                print(f"Error processing pair ({t1}, {t2}): {e}")

    return pd.DataFrame(sorted(pairs_results, key=lambda x: x["Sharpe (Z*ΔSpread)"], reverse=True)[:max_pairs])


# Pair Trading Backtest Function
def backtest_pair_strategy(y, x, entry_z=1.5, exit_z=0.5):
    # Align and regress to get spread
    df = pd.concat([y, x], axis=1).dropna()
    y, x = df.iloc[:, 0], df.iloc[:, 1]
    beta = np.polyfit(x, y, 1)[0]
    spread = y - beta * x
    zscore = (spread - spread.mean()) / spread.std()

    # Generate signals
    signals = pd.Series(0, index=spread.index)
    signals[zscore > entry_z] = -1  # Short spread
    signals[zscore < -entry_z] = 1  # Long spread
    signals[np.abs(zscore) < exit_z] = 0  # Exit

    # Forward fill signal positions
    positions = signals.replace(to_replace=0, method='ffill').shift(1).fillna(0)

    # PnL from spread changes
    pnl = positions * spread.diff().fillna(0)
    cumret = (1 + pnl).cumprod()
    log_returns = np.log(cumret).diff().fillna(0)

    # Metrics
    sharpe = log_returns.mean() / (log_returns.std() + 1e-8) * np.sqrt(252)
    max_drawdown = (cumret / cumret.cummax() - 1).min()

    return {
        "Sharpe Ratio": sharpe,
        "Total Return": cumret.iloc[-1] - 1,
        "Max Drawdown": max_drawdown,
        "Beta": beta
    }

def backtest_top_pairs(prices, pair_df, top_n=10):
    results = []
    for _, row in pair_df.head(top_n).iterrows():
        try:
            y = prices[row["Stock1"]]
            x = prices[row["Stock2"]]
            stats = backtest_pair_strategy(y, x)

            results.append({
                "Stock1": row["Stock1"],
                "Stock2": row["Stock2"],
                **stats
            })
        except Exception as e:
            print(f"Failed backtest for {row['Stock1']} & {row['Stock2']}: {e}")
    return pd.DataFrame(results)

#%%
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

prices = download_data(tickers, start_date, end_date)
pair_results_df = run_pair_trading_analysis(prices)
pair_results_df.to_csv("../modelling_result/top_cointegrated_pairs.csv", index=False)

pair_backtest_df = backtest_top_pairs(prices, pair_results_df, top_n=10)
pair_backtest_df.to_csv("../modelling_result/pair_trading_backtest_results.csv", index=False)
