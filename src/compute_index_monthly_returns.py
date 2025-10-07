import pandas as pd
from pathlib import Path

DEFAULT_INPUT = r"D:\A_share_market\20240910_index_return.csv"
OUTPUT_CSV = Path(__file__).resolve().parents[1] / 'results' / 'monthly_index_returns.csv'

def replace_leading_zeros_with_nan(series: pd.Series) -> pd.Series:
    # Determine the first non-zero return index
    non_zero_mask = series.ne(0)
    first_valid_idx = non_zero_mask.idxmax()
    if series.loc[first_valid_idx] == 0:
        return series.mask(series == 0)
    # Set zeros before the first non-zero as missing (pre-trading period)
    series.loc[:first_valid_idx] = series.loc[:first_valid_idx].replace(0.0, pd.NA)
    return series

def main(input_path: str = DEFAULT_INPUT):
    df = pd.read_csv(input_path)
    df.rename(columns={df.columns[0]: 'date'}, inplace=True)
    # index dates are often numeric yyyymmdd
    try:
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    except Exception:
        df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Convert percentage to decimal if needed
    if df.abs().stack().quantile(0.999) > 1:
        df = df / 100.0

    df_cleaned = df.apply(replace_leading_zeros_with_nan)
    monthly_returns = (1 + df_cleaned).resample('M').prod() - 1
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    monthly_returns.to_csv(OUTPUT_CSV, encoding='utf-8-sig')
    print('Saved:', OUTPUT_CSV)

if __name__ == '__main__':
    main()