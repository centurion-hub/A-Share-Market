import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FUND_DAILY = r"D:\A_share_market\20240910_fund_dayReturn.csv"
INDEX_DAILY = r"D:\A_share_market\20240910_index_return.csv"

def strip_leading_zeros_to_nan(col: pd.Series) -> pd.Series:
    # Identify first non-zero, non-NaN and mask all earlier zeros as NaN
    mask_valid = col.notna() & (col != 0)
    if not mask_valid.any():
        return pd.Series(np.nan, index=col.index)  # never traded
    first_trading_day = mask_valid.idxmax()
    col = col.copy()
    col.loc[:first_trading_day] = col.loc[:first_trading_day].where(mask_valid.loc[:first_trading_day], np.nan)
    # everything strictly before first_trading_day -> NaN
    col.loc[:first_trading_day - pd.Timedelta(days=1)] = np.nan
    return col

def monthly_from_daily_ignoring_na(df: pd.DataFrame) -> pd.DataFrame:
    # Compute monthly returns as product(1+r) - 1, ignoring missing days
    return df.add(1).groupby(pd.Grouper(freq='M')).prod().sub(1)

def calculate_max_drawdown(series: pd.Series) -> float:
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return dd.min()

def calculate_performance_stats(returns_series: pd.Series) -> dict:
    ann_ret = returns_series.mean() * 252
    ann_vol = returns_series.std() * np.sqrt(252)
    sharpe = np.nan if ann_vol == 0 else ann_ret / ann_vol
    mdd = calculate_max_drawdown(returns_series)
    return {"Annualized Return": ann_ret, "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe, "Maximum Drawdown": mdd}

def run_strategy_and_get_excess_returns(fund_returns_df, index_returns_df, beta, index_col='000300.SH'):
    # 1) Compute monthly returns robustly
    monthly_fund = monthly_from_daily_ignoring_na(fund_returns_df)

    # 2) Rank each month by past 12M cumulative return (require full 12 months of data)
    top_funds_by_month = {}
    for i in range(12, len(monthly_fund)):
        past12 = monthly_fund.iloc[i-12:i]
        have_full_hist = past12.notna().sum(axis=0) == 12
        if have_full_hist.sum() == 0: continue
        cum12 = (1 + past12.loc[:, have_full_hist]).prod() - 1
        n = max(int(len(cum12) * 0.20), 1)
        top_funds = cum12.nlargest(n).index.tolist()
        top_funds_by_month[monthly_fund.index[i]] = top_funds

    # 3) Hold NEXT month’s daily returns for the selected basket (no look-ahead)
    excess_chunks = []
    for month_end, names in top_funds_by_month.items():
        month = month_end.to_period('M')
        in_month = fund_returns_df.index.to_period('M') == month
        port_daily = fund_returns_df.loc[in_month, names].mean(axis=1, skipna=True)
        idx_daily = index_returns_df.loc[in_month, index_col]
        daily_excess = port_daily - beta * idx_daily
        excess_chunks.append(daily_excess)

    if not excess_chunks:
        return pd.Series(dtype=float)

    excess = pd.concat(excess_chunks).sort_index()
    return excess.loc[~excess.index.duplicated()]

def main():
    fund_df = pd.read_csv(FUND_DAILY)
    index_df = pd.read_csv(INDEX_DAILY)

    fund_df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
    fund_df['date'] = pd.to_datetime(fund_df['date'])
    index_df['date'] = pd.to_datetime(index_df['date'], format='%Y%m%d')

    fund_df.set_index('date', inplace=True)
    index_df.set_index('date', inplace=True)

    common = fund_df.index.intersection(index_df.index)
    fund_df = fund_df.loc[common]
    index_df = index_df.loc[common]

    if fund_df.abs().stack().quantile(0.999) > 1:
        fund_df = fund_df / 100.0
    if index_df.abs().stack().quantile(0.999) > 1:
        index_df = index_df / 100.0

    fund_df = fund_df.apply(strip_leading_zeros_to_nan)

    betas = [0.6, 0.8, 1.0, 1.2, 1.4]
    performance_stats, cumulative_returns = {}, {}

    for beta in betas:
        excess = run_strategy_and_get_excess_returns(fund_df, index_df, beta, index_col='000300.SH')
        performance_stats[beta] = calculate_performance_stats(excess)
        cumulative_returns[beta] = (1 + excess).cumprod()

    plt.figure(figsize=(12,7))
    for beta, cr in cumulative_returns.items():
        plt.plot(cr.index, cr.values, label=f'Beta = {beta}')
    plt.title('Cumulative Excess Returns of Market Neutral Strategies')
    plt.xlabel('Date'); plt.ylabel('Cumulative Returns')
    plt.legend(loc='upper left'); plt.grid(True); plt.tight_layout()
    out_png = Path(__file__).resolve().parents[1] / 'results' / 'cumulative_returns_all.png'
    plt.savefig(out_png, dpi=150)
    print('Saved plot:', out_png)

    stats_df = pd.DataFrame(performance_stats).T
    out_csv = Path(__file__).resolve().parents[1] / 'results' / 'short_strategy_stats.csv'
    stats_df.to_csv(out_csv)
    print('Saved stats:', out_csv)

if __name__ == '__main__':
    main()