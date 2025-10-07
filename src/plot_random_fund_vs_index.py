import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path

FUND_PATH = r'D:\A_share_market\20240910_fund_dayReturn.csv'
INDEX_PATH = r'D:\A_share_market\20240910_index_return.csv'

def replace_leading_zeros_with_nan(series: pd.Series) -> pd.Series:
    # Determine the first non-zero return index
    non_zero_mask = series.ne(0)
    first_valid_idx = non_zero_mask.idxmax()
    if series.loc[first_valid_idx] == 0:
        return series.mask(series == 0)
    # Set zeros before the first non-zero as missing (pre-listing period)
    series.loc[:first_valid_idx] = series.loc[:first_valid_idx].replace(0.0, pd.NA)
    return series

def main():
    # Load fund daily returns
    fund_df = pd.read_csv(FUND_PATH)
    fund_df.rename(columns={fund_df.columns[0]: 'date'}, inplace=True)
    fund_df['date'] = pd.to_datetime(fund_df['date'], format='%Y-%m-%d')
    fund_df.set_index('date', inplace=True)
    if fund_df.abs().stack().quantile(0.999) > 1:
        fund_df = fund_df / 100.0
    fund_df = fund_df.apply(replace_leading_zeros_with_nan)

    # Load index returns
    index_df = pd.read_csv(INDEX_PATH, encoding='gbk')
    index_df.rename(columns={index_df.columns[0]: 'date'}, inplace=True)
    index_df['date'] = pd.to_datetime(index_df['date'].astype(str), format='%Y%m%d')
    index_df.set_index('date', inplace=True)
    if index_df.abs().stack().quantile(0.999) > 1:
        index_df = index_df / 100.0

    # Randomly pick a fund & a reference index
    random_fund = random.choice(fund_df.columns)
    reference_index = index_df.columns[0]

    # Align dates and compute cumulative returns
    fund_series = fund_df[random_fund].dropna()
    index_series = index_df[reference_index].dropna()
    common_dates = fund_series.index.intersection(index_series.index)

    cumulative_fund = (1 + fund_series.loc[common_dates]).cumprod() - 1
    cumulative_index = (1 + index_series.loc[common_dates]).cumprod() - 1

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_fund.index, cumulative_fund, label=f'Fund: {random_fund}', linewidth=2)
    plt.plot(cumulative_index.index, cumulative_index, label=f'Index: {reference_index}', linewidth=2)
    plt.title('Cumulative Return: Fund vs Index')
    plt.xlabel('Date'); plt.ylabel('Cumulative Return')
    plt.legend(); plt.grid(True); plt.tight_layout()

    out = Path(__file__).resolve().parents[1] / 'results' / 'fund_vs_index.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print('Saved plot:', out)

if __name__ == '__main__':
    main()