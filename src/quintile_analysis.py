import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FUND_DAILY = r"D:\A share market\20240910_fund_dayReturn.csv"
INDEX_DAILY = r"D:\A share market\20240910_index_return.csv"

def main():
    # Load daily returns
    fund_returns = pd.read_csv(FUND_DAILY)
    index_returns = pd.read_csv(INDEX_DAILY)

    # Parse dates
    fund_returns = fund_returns.rename(columns={'Unnamed: 0': 'date'})
    fund_returns['date'] = pd.to_datetime(fund_returns['date'], format='%Y-%m-%d')
    index_returns['date'] = pd.to_datetime(index_returns['date'], format='%Y%m%d')

    # Align dates
    common_dates = pd.merge(fund_returns['date'], index_returns['date'], on='date')['date']
    fund = fund_returns[fund_returns['date'].isin(common_dates)].set_index('date')
    index = index_returns[index_returns['date'].isin(common_dates)].set_index('date')

    # Convert % to decimals if needed
    if fund.abs().stack().quantile(0.999) > 1:
        fund = fund / 100.0
    if index.abs().stack().quantile(0.999) > 1:
        index = index / 100.0

    # Benchmark as the average of three indices
    index['Benchmark_Index'] = index[['000300.SH', '000905.SH', '000906.SH']].mean(axis=1)

    # -------- Key fix: precise start date per fund --------
    # 1) Find the first non-zero return date for each fund (trading start)
    fund_start_dates = {}
    for col in fund.columns:
        non_zero_idx = fund[col].ne(0).idxmax()
        if pd.notna(non_zero_idx) and fund.loc[non_zero_idx, col] != 0:
            fund_start_dates[col] = non_zero_idx
        else:
            # If no valid start date, push to max date (ignore this fund)
            fund_start_dates[col] = fund.index.max()

    # 2) Mask pre-start data as NaN
    for col, start_date in fund_start_dates.items():
        mask = fund.index < start_date
        fund.loc[mask, col] = np.nan

    # 3) Compute average performance per fund only after start date
    perf = {}
    for col in fund.columns:
        sdt = fund_start_dates[col]
        perf[col] = fund.loc[sdt:, col].mean()

    perf = pd.Series(perf).sort_values(ascending=False).dropna()

    # Split into 5 equal groups by average return
    names_sorted = perf.index.tolist()
    splits = np.array_split(names_sorted, 5)
    quintiles = {f'Quintile_{i+1}': list(g) for i, g in enumerate(splits)}

    # Daily equal-weight average per quintile (only valid trading funds each day)
    quintile_daily = pd.DataFrame(index=fund.index)
    for qname, flist in quintiles.items():
        daily_means = pd.Series(index=fund.index, dtype=float)
        for dt in fund.index:
            valid = []
            for c in flist:
                if dt >= fund_start_dates[c] and not pd.isna(fund.loc[dt, c]):
                    valid.append(c)
            daily_means.loc[dt] = fund.loc[dt, valid].mean() if len(valid) else np.nan
        quintile_daily[qname] = daily_means

    # Forward fill gaps, then set initial NAs to zero
    quintile_daily.fillna(method='ffill', inplace=True)
    quintile_daily.fillna(0, inplace=True)

    # Cumulative curves
    cumulative = (1 + quintile_daily).cumprod()
    cumulative['Benchmark_Index'] = (1 + index['Benchmark_Index']).cumprod()
    cumulative = cumulative / cumulative.iloc[0]

    plt.figure(figsize=(14, 8))
    for col in cumulative.columns:
        plt.plot(cumulative.index, cumulative[col], label=col)
    plt.title('Cumulative Returns of Fund Quintiles and Benchmark Index')
    plt.xlabel('Date'); plt.ylabel('Cumulative Return')
    plt.grid(True); plt.legend(); plt.tight_layout()

    out_png = Path(__file__).resolve().parents[1] / 'results' / 'cumulative_returns_quintiles.png'
    plt.savefig(out_png, dpi=150)
    print('Saved plot:', out_png)

    # Performance table for the top tranche vs benchmark
    top = quintile_daily['Quintile_1']
    bm = index['Benchmark_Index']

    def ann_ret(r):
        r = r.dropna()
        return (1 + r).prod()**(252/len(r)) - 1 if len(r) else np.nan
    def ann_vol(r):
        r = r.dropna()
        return r.std() * np.sqrt(252) if len(r) else np.nan
    def mdd(r):
        r = r.dropna()
        if len(r) == 0: return np.nan
        cum = (1 + r).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd.min()

    stats = pd.DataFrame({
        'Metric': ['Annualized Return', 'Annualized Volatility', 'Sharpe (rf=0)', 'Max Drawdown'],
        'Top Tranche': [ann_ret(top), ann_vol(top), (ann_ret(top)/(ann_vol(top) or np.nan)), mdd(top)],
        'Benchmark'  : [ann_ret(bm),  ann_vol(bm),  (ann_ret(bm)/(ann_vol(bm) or np.nan)),  mdd(bm)]
    })
    out_csv = Path(__file__).resolve().parents[1] / 'results' / 'quintile_stats.csv'
    stats.to_csv(out_csv, index=False)
    print('Saved stats:', out_csv)

if __name__ == '__main__':
    main()