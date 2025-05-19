#%% Pair Trading
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from tools.utils import download_data, get_sector_info, run_mean_reversion_analysis
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
import math

#%%
def rolling_beta(y, x, window=60):
    """
    Computes rolling hedge ratio (beta) using OLS.
    """
    betas = pd.Series(index=y.index, dtype='float64')
    for i in range(window, len(y)):
        yi = y.iloc[i - window:i]
        xi = x.iloc[i - window:i]
        if len(yi.dropna()) == window and len(xi.dropna()) == window:
            X = add_constant(xi)
            model = OLS(yi, X).fit()
            betas.iloc[i] = model.params[1]
    return betas.fillna(method='bfill')

def run_pair_trading_analysis(prices, max_pairs=50):
    tickers = prices.columns
    pairs_results = []
    spreads_dict = {}

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
                    # beta = np.polyfit(x, y, 1)[0]
                    beta = rolling_beta(y, x)
                    spread = y - beta * x
                    zscore = (spread - spread.mean()) / spread.std()
                    sharpe = (zscore.shift(1) * spread.diff()).mean() / (spread.diff().std() + 1e-8) * np.sqrt(252)

                    ticker_name = f"{t1}_{t2}"
                    pairs_results.append({
                        "Ticker": ticker_name,
                        "Stock1": t1,
                        "Stock2": t2,
                        "Cointegration p-value": round(coint_pval, 4),
                        "Sharpe (Z*ΔSpread)": round(sharpe, 4),
                        "Mean Spread": round(spread.mean(), 4),
                        "Std Spread": round(spread.std(), 4),
                        # "Start": spread.index.min(),
                        # "End": spread.index.max()
                    })
                    spreads_dict[ticker_name] = spread

            except Exception as e:
                print(f"Error processing pair ({t1}, {t2}): {e}")

    # Top pairs
    sorted_results = sorted(pairs_results, key=lambda x: x["Sharpe (Z*ΔSpread)"], reverse=True)[:max_pairs]
    pair_results_df = pd.DataFrame(sorted_results)
    # spread_df
    top_spread_series = {d["Ticker"]: spreads_dict[d["Ticker"]] for d in sorted_results}
    spread_df = pd.DataFrame(top_spread_series)
    
    return pair_results_df, spread_df




def benchmark_buy_and_hold(y, x, transaction_cost = 0.001):
    df = pd.concat([y, x], axis=1).dropna()
    daily_returns = df.diff().fillna(0)
    combined_returns = daily_returns.mean(axis=1)
    combined_returns.iloc[0] -= transaction_cost
    combined_returns.iloc[-1] -= transaction_cost
    cum_bench = combined_returns.cumsum()
    # cum_bench = cum_bench - 2 * transaction_cost

    return {
        "BuyHold Total Log Return": cum_bench
    }


# Pair Trading Backtest Function
def backtest_pair_strategy(y, x, entry_z=1.5, exit_z=0.5, cost=0.001):
    # Align and regress to get spread
    df = pd.concat([y, x], axis=1).dropna()
    y, x = df.iloc[:, 0], df.iloc[:, 1]
    beta = rolling_beta(y, x)
    # beta = np.polyfit(x, y, 1)[0]
    spread = y - beta * x
    spread_mean = spread.rolling(60).mean().shift(-1) # to avoid lookahead bias
    spread_std = spread.rolling(60).std().shift(-1)
    zscore = (spread - spread_mean) / spread_std
    
    spread = spread.loc[zscore.index]
    beta = beta.loc[zscore.index]

    # Generate signals
    signals = pd.Series(0, index=spread.index)
    signals[zscore > entry_z] = -1  # Short spread
    signals[zscore < -entry_z] = 1  # Long spread
    signals[np.abs(zscore) < exit_z] = 0  # Exit

    # Forward fill signal positions
    positions = signals.replace(to_replace=0, method='ffill').shift(1).fillna(0)

    # PnL from spread changes (adding costs)
    # ret_y = y.diff().fillna(0)
    # ret_x = x.diff().fillna(0)
    # net_log_return = positions * (ret_y - beta * ret_x)

    net_log_return = positions * spread.diff().fillna(0)
    trades = positions.diff().abs().fillna(0)
    costs = trades * cost
    net_log_return -= costs
    cum_log_return = net_log_return.cumsum()
    

    # Metrics
    sharpe = net_log_return.mean() / (net_log_return.std() + 1e-8) * np.sqrt(252)
    total_log_return = cum_log_return.iloc[-1]
    annualized_return = total_log_return * (252 / len(net_log_return))
    max_drawdown = (cum_log_return / cum_log_return.cummax() - 1).min()

    return {   
        "Total Log Return": total_log_return,
        "Annualized Return": annualized_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Beta": beta,
        "Cummulative Return": cum_log_return
    }

def backtest_top_pairs(prices, pair_df, top_n=10):
    results = []
    for _, row in pair_df.head(top_n).iterrows():
        try:
            y = prices[row["Stock1"]]
            x = prices[row["Stock2"]]
            stats = backtest_pair_strategy(y, x)
            benchmark = benchmark_buy_and_hold(y, x)

            results.append({
                "Stock1": row["Stock1"],
                "Stock2": row["Stock2"],
                **stats,
                **benchmark
            })
        except Exception as e:
            print(f"Failed backtest for {row['Stock1']} & {row['Stock2']}: {e}")
    return pd.DataFrame(results)

def plot_top_pairs_grid(prices, pair_df, top_n=10, transaction_cost=0.001):
    n_rows = math.ceil(top_n / 2)
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=False)
    axes = axes.flatten()

    for idx, (_, row) in enumerate(pair_df.head(top_n).iterrows()):
        stock1, stock2 = row["Stock1"], row["Stock2"]
        y = prices[stock1]
        x = prices[stock2]

        # Use predefined strategy functions
        pair_stats = backtest_pair_strategy(y, x, cost=transaction_cost)
        bh_stats = benchmark_buy_and_hold(y, x, transaction_cost=transaction_cost)

        cum_strategy = pair_stats["Cummulative Return"]
        cum_bench = bh_stats["BuyHold Total Log Return"]

        # Plot
        ax = axes[idx]
        ax.plot(cum_strategy, label="Pair Strategy", linewidth=2)
        ax.plot(cum_bench, label="Buy & Hold", linestyle="--", linewidth=2)
        ax.set_title(f"{stock1} vs {stock2}", fontsize=11)
        ax.legend()
        ax.grid(True)

    # Hide unused subplots
    for ax in axes[top_n:]:
        ax.set_visible(False)

    fig.suptitle("Top Pair Trading Strategies vs. Buy & Hold: Cummulative Log Return", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

#%%
start_date = '2010-01-01'
end_date = '2019-12-31'
large_cap_tickers = ['AAPL', 'MSFT', 'JPM', 'NVDA', 'XOM', 'JNJ', 'UNH', 'PG', 'V', 'MA', 'HD', 'COST', 'AVGO', 'LLY', 'BAC', 'MRK', 'ADBE']
mid_cap_tickers = ['TFX', 'HES', 'NTNX', 'WU', 'FIVE', 'GNRC', 'WING', 'CHDN', 'FND', 'HWM', 'CROX', 'ENPH', 'FICO', 'ROK', 'LII']
small_cap_tickers = ['INSM', 'NEOG', 'ACLS', 'PRDO', 'ORGO', 'IMMR', 'CVCO', 'GPRO', 'STRL', 'TPC', 'GHC', 'FIZZ', 'EVTC', 'CMTL', 'MGEE']
tickers = large_cap_tickers + mid_cap_tickers + small_cap_tickers
window_list = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
prices = download_data(tickers, start_date, end_date)

#%% metadata
ticker_cap_mapping = {ticker: 'Large' for ticker in large_cap_tickers}
ticker_cap_mapping.update({ticker: 'Mid' for ticker in mid_cap_tickers})
ticker_cap_mapping.update({ticker: 'Small' for ticker in small_cap_tickers})

sector_df = get_sector_info(tickers)
sector_df['CapSize'] = sector_df['Ticker'].map(ticker_cap_mapping)
metadata_df = sector_df[['Ticker', 'Sector', 'Industry', 'CapSize']]

#%%
def build_pair_metadata(pair_results_df, metadata_df):
    # Make metadata_df searchable
    meta = metadata_df.set_index("Ticker")

    records = []
    for _, row in pair_results_df.iterrows():
        t1, t2 = row["Stock1"], row["Stock2"]
        ticker_pair = row["Ticker"]

        # Get sector/cap info safely
        sector1 = meta.loc[t1]["Sector"] if t1 in meta.index else "N/A"
        sector2 = meta.loc[t2]["Sector"] if t2 in meta.index else "N/A"
        industry1 = meta.loc[t1]["Industry"] if t1 in meta.index else "N/A"
        industry2 = meta.loc[t2]["Industry"] if t2 in meta.index else "N/A"
        cap1 = meta.loc[t1]["CapSize"] if t1 in meta.index else "N/A"
        cap2 = meta.loc[t2]["CapSize"] if t2 in meta.index else "N/A"

        records.append({
            "Ticker": ticker_pair,
            # "Stock1": t1,
            # "Stock2": t2,
            "Sector1": sector1,
            "Sector2": sector2,
            "Industry1": industry1,
            "Industry2": industry2,
            "CapSize1": cap1,
            "CapSize2": cap2,
            # "CombinedSector": f"{sector1}_{sector2}"
        })

    return pd.DataFrame(records)

#%% Cointegration analysis
pair_results_df, pair_spread = run_pair_trading_analysis(prices)
# pair_results_df.to_csv(f"../modelling_result/pair_trading_top_cointegrated_pairs_{start_date[:4]}_{end_date[:4]}.csv", index=False)
# pair_spread.to_csv(f"../modelling_result/pair_trading_spread_series_{start_date[:4]}_{end_date[:4]}.csv", index=False)

#%% Mean Reversion Analysis
pair_metadata_df = build_pair_metadata(pair_results_df, metadata_df)
stat_df = run_mean_reversion_analysis(pair_spread, pair_metadata_df, strategy_type="spread")

#%%
pair_backtest_df = backtest_top_pairs(prices, pair_results_df, top_n=48)
pair_backtest_df.to_csv(f"../modelling_result/pair_trading_backtest_results_{start_date[:4]}_{end_date[:4]}.csv", index=False)

# %%
plot_top_pairs_grid(prices, pair_results_df, top_n=48)


#%% Histogram plots
# Plotting histograms for ADF p-value, Hurst exponent, and Half-life
import matplotlib.pyplot as plt
import seaborn as sns

df = stat_df.copy()
# Set style
sns.set(style='whitegrid')

# Set up the figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram for ADF p-value with threshold
sns.histplot(df['ADF p-value'], bins=30, kde=True, ax=axes[0], color='skyblue')
axes[0].axvline(0.05, color='red', linestyle='--', linewidth=2, label='0.05 threshold')
axes[0].set_title('ADF p-value Distribution')
axes[0].set_xlabel('ADF p-value')
axes[0].legend()

# Histogram for Hurst exponent with thresholds
sns.histplot(df['Hurst'], bins=30, kde=True, ax=axes[1], color='salmon')
axes[1].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Random walk (0.5)')
axes[1].axvline(0.4, color='blue', linestyle='--', linewidth=2, label='Mean-reversion (0.4)')
axes[1].set_title('Hurst Exponent Distribution')
axes[1].set_xlabel('Hurst Exponent')
axes[1].legend()

# Histogram for Half-life with thresholds
sns.histplot(df['Half-life'], bins=30, kde=True, ax=axes[2], color='lightgreen')
axes[2].axvline(10, color='blue', linestyle='--', linewidth=2, label='Strong (<10)')
axes[2].axvline(50, color='red', linestyle='--', linewidth=2, label='Weak (<50)')
axes[2].set_title('Half-life Distribution')
axes[2].set_xlabel('Half-life')
axes[2].legend()

# Final layout
plt.tight_layout()
plt.savefig(f"../modelling_result/plots_spread_distribution")
plt.show()
# %%