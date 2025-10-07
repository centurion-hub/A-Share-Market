import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FUND_DAILY = r"D:\A_share_market\20240910_fund_dayReturn.csv"
INDEX_DAILY = r"D:\A_share_market\20240910_index_return.csv"

def main():
    # Load daily returns
    fund = pd.read_csv(FUND_DAILY)
    idx = pd.read_csv(INDEX_DAILY)

    # Parse dates
    fund = fund.rename(columns={'Unnamed: 0': 'date'})
    fund['date'] = pd.to_datetime(fund['date'], errors='coerce')
    idx['date'] = pd.to_datetime(idx['date'], format='%Y%m%d', errors='coerce')

    # Align by common dates
    common = pd.merge(fund['date'], idx['date'], on='date')['date']
    fund = fund[fund['date'].isin(common)].set_index('date')
    idx = idx[idx['date'].isin(common)].set_index('date')

    # Convert % to decimals if needed
    if fund.abs().stack().quantile(0.999) > 1:
        fund = fund / 100.0
    if idx.abs().stack().quantile(0.999) > 1:
        idx = idx / 100.0

    # Benchmark index = average of HS300, ZZ500, ZZ800
    idx['Benchmark_Index'] = idx[['000300.SH','000905.SH','000906.SH']].mean(axis=1)

    # Pre-listing masking: set days prior to first non-zero to NaN
    start_dates = {}
    for c in fund.columns:
        nz = fund[c].ne(0).idxmax()
        if pd.notna(nz) and fund.loc[nz, c] != 0:
            start_dates[c] = nz
        else:
            start_dates[c] = fund.index.max()
    for c, sdt in start_dates.items():
        fund.loc[fund.index < sdt, c] = np.nan

    # Monthly compounding from daily
    m_fund = fund.resample('M').apply(lambda x: (1 + x).prod() - 1)
    m_idx = idx['Benchmark_Index'].resample('M').apply(lambda x: (1 + x).prod() - 1)

    res = pd.DataFrame(index=m_fund.index)
    res['Top_Portfolio_Return'] = np.nan
    res['Index_Return'] = m_idx

    # 6-month lookback, rank by mean, next-month hold
    for i in range(6, len(m_fund)-1):
        cur = m_fund.index[i]
        nxt = m_fund.index[i+1]
        past = m_fund.iloc[i-6:i]
        perf = past.mean().sort_values(ascending=False).dropna()
        if len(perf) == 0: 
            continue
        k = max(int(len(perf) * 0.2), 1)
        top = perf.head(k).index
        res.loc[nxt, 'Top_Portfolio_Return'] = m_fund.loc[nxt, top].mean()

    # Hedge with various betas
    betas = [0.6, 0.8, 1.0, 1.2, 1.4]
    cum = {}
    for b in betas:
        res[f'Hedged_Excess_Beta_{b}'] = res['Top_Portfolio_Return'] - b * res['Index_Return']
        cum[b] = (1 + res[f'Hedged_Excess_Beta_{b}'].fillna(0)).cumprod()

    plt.figure(figsize=(14,8))
    for b in betas:
        plt.plot(res.index, cum[b], label=f'Beta={b}', linewidth=2)
    plt.title('Cumulative Excess Returns with Different Hedge Ratios')
    plt.xlabel('Date'); plt.ylabel('Cumulative Return')
    plt.grid(True); plt.legend(); plt.tight_layout()

    out_png = Path(__file__).resolve().parents[1] / 'results' / 'hedge_ratio_comparison.png'
    plt.savefig(out_png, dpi=150)
    print('Saved plot:', out_png)

    out_csv = Path(__file__).resolve().parents[1] / 'results' / 'market_neutral_results.csv'
    res.to_csv(out_csv)
    print('Saved results:', out_csv)

if __name__ == '__main__':
    main()