import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def run_window(start_date: str, end_date: str,
               fund_monthly_csv: str, index_monthly_csv: str, out_png: Path):
    # Read monthly returns (decimal)
    fund_monthly = pd.read_csv(fund_monthly_csv, index_col=0, parse_dates=True)
    index_monthly = pd.read_csv(index_monthly_csv, index_col=0, parse_dates=True)

    # Subset window
    fund_sub = fund_monthly.loc[start_date:end_date].copy()
    index_sub = index_monthly.loc[start_date:end_date].copy()

    # Treat zero returns as missing for funds (exclude not-yet-listed funds)
    fund_sub_cleaned = fund_sub.copy()
    fund_sub_cleaned[fund_sub_cleaned == 0] = pd.NA
    fund_mean = fund_sub_cleaned.mean(axis=1, skipna=True)

    # Cumulative returns
    fund_cumulative = (1 + fund_mean).cumprod()
    index_cumulative = (1 + index_sub).cumprod()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(fund_cumulative, label='Fund Average')
    for col in index_cumulative.columns:
        plt.plot(index_cumulative[col], label=col)

    plt.title(f'Fund Average vs Index Cumulative ({start_date}–{end_date})')
    plt.ylabel('Cumulative Growth Multiple')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print('Saved:', out_png)

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    fm = root / 'results' / 'monthly_fund_returns.csv'
    im = root / 'results' / 'monthly_index_returns.csv'
    run_window('2006-01-01', '2009-12-31', str(fm), str(im), root / 'results' / 'fund_vs_index_2006_2009.png')
    run_window('2014-01-01', '2016-12-31', str(fm), str(im), root / 'results' / 'fund_vs_index_2014_2016.png')