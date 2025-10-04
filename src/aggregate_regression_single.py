import pandas as pd
import statsmodels.api as sm
from pathlib import Path

FUND_DAILY = r"D:\A share market\20240910_fund_dayReturn.csv"
INDEX_DAILY = r"D:\A share market\20240910_index_return.csv"
MARKET_INDEX = '000905.SH'  # ZZ500

def main():
    # Read datasets
    fund_df = pd.read_csv(FUND_DAILY)
    index_df = pd.read_csv(INDEX_DAILY)

    # Parse dates and merge
    fund_df = fund_df.rename(columns={'Unnamed: 0': 'date'})
    fund_df['date'] = pd.to_datetime(fund_df['date'])
    try:
        index_df['date'] = pd.to_datetime(index_df['date'].astype(str), format='%Y%m%d')
    except Exception:
        index_df['date'] = pd.to_datetime(index_df['date'])
    merged_df = pd.merge(fund_df, index_df, on='date', how='inner')

    # Remove rows where all funds are zeros (non-trading days)
    fund_cols = fund_df.columns[1:]
    merged_df = merged_df[~(merged_df[fund_cols].sum(axis=1) == 0)]

    # Set pre-listing zeros to NaN per fund
    cleaned = merged_df[fund_cols].copy()
    for col in cleaned.columns:
        s = cleaned[col]
        first_nonzero_idx = s.ne(0).idxmax()
        cleaned.loc[:first_nonzero_idx - 1, col] = pd.NA
    merged_df[fund_cols] = cleaned

    # Convert percentages to decimals if needed
    if merged_df[fund_cols].abs().stack().quantile(0.999) > 1:
        merged_df[fund_cols] = merged_df[fund_cols] / 100.0
    if merged_df[MARKET_INDEX].abs().quantile(0.999) > 1:
        merged_df[MARKET_INDEX] = merged_df[MARKET_INDEX] / 100.0

    merged_df['fund_avg_return'] = merged_df[fund_cols].mean(axis=1, skipna=True)

    X = sm.add_constant(merged_df[MARKET_INDEX])
    y = merged_df['fund_avg_return']
    model = sm.OLS(y, X, missing='drop').fit()

    print("=== Aggregate Fund Regression (vs ZZ500) ===")
    print(f"Alpha: {model.params['const']:.6f} (t = {model.tvalues['const']:.2f})")
    print(f"Beta : {model.params[MARKET_INDEX]:.6f} (t = {model.tvalues[MARKET_INDEX]:.2f})")
    print(f"R^2  : {model.rsquared:.4f}")

    out = Path(__file__).resolve().parents[1] / 'results' / 'fund_data_cleaned_single.csv'
    merged_df.to_csv(out, index=False)
    print('Saved cleaned merge:', out)

if __name__ == '__main__':
    main()