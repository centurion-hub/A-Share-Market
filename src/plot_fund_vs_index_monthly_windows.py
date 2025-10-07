import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def run_window(start_date: str, end_date: str,
               fund_monthly_csv: str, index_monthly_csv: str, out_png: Path):
    """
    Compare fund vs. index cumulative performance within a specific time window.

    Parameters
    ----------
    start_date : str
        Start date of the window (inclusive, e.g. '2006-01-01').
    end_date : str
        End date of the window (inclusive, e.g. '2009-12-31').
    fund_monthly_csv : str
        Path to the monthly fund returns CSV file.
    index_monthly_csv : str
        Path to the monthly index returns CSV file.
    out_png : Path
        Output path for the PNG plot.
    """

    # --- 1. Load monthly returns ---
    fund_monthly = pd.read_csv(fund_monthly_csv, index_col=0, parse_dates=True)
    index_monthly = pd.read_csv(index_monthly_csv, index_col=0, parse_dates=True)

    # --- 2. Subset the desired window ---
    fund_sub = fund_monthly.loc[start_date:end_date].copy()
    index_sub = index_monthly.loc[start_date:end_date].copy()

    # --- 3. Mask pre-listing periods for funds ---
    # Once a fund shows a non-zero return, all prior periods are considered "not yet listed"
    # Those pre-listing values are replaced with NaN
    has_started = (fund_sub != 0).cummax(axis=0)
    fund_sub_cleaned = fund_sub.mask(~has_started)

    # --- 4. Compute equal-weighted average fund return ---
    # Skip NaNs when averaging; fill remaining NaN with 0 to keep cumprod continuous
    fund_mean = fund_sub_cleaned.mean(axis=1, skipna=True).fillna(0.0)

    # --- 5. Compute cumulative returns (growth multiple) ---
    fund_cumulative = (1.0 + fund_mean).cumprod()
    index_cumulative = (1.0 + index_sub).cumprod()

    # --- 6. Safety checks ---
    assert fund_cumulative.notna().any(), "No valid fund data in the given window."
    assert index_cumulative.notna().any().any(), "No valid index data in the given window."

    # --- 7. Plot cumulative performance ---
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.plot(fund_cumulative, label='Fund Average')
    for col in index_cumulative.columns:
        ax.plot(index_cumulative[col], label=col)

    ax.axhline(1.0, linewidth=1, linestyle='--', alpha=0.6)
    ax.set_title(f'Fund Average vs Index Cumulative ({start_date}–{end_date})')
    ax.set_ylabel('Cumulative Growth (×)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    # --- 8. Save figure ---
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print('Saved:', out_png)


if __name__ == '__main__':
    # Define project root and file paths
    root = Path(__file__).resolve().parents[1]
    fm = root / 'results' / 'monthly_fund_returns.csv'
    im = root / 'results' / 'monthly_index_returns.csv'

    # Run selected analysis windows
    run_window('2006-01-01', '2009-12-31', str(fm), str(im),
               root / 'results' / 'fund_vs_index_2006_2009.png')

    run_window('2014-01-01', '2016-12-31', str(fm), str(im),
               root / 'results' / 'fund_vs_index_2014_2016.png')
