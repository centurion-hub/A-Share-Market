# A-share Fund Analysis & Market-Neutral Strategies

This repository contains a reproducible Python pipeline for:
- Cleaning and transforming daily fund & index returns
- Aggregating to monthly returns
- Plotting fund vs. index cumulative performance
- Single- and two-factor regressions (aggregate fund vs HS300/ZZ500)
- Quintile ranking backtests
- A short/hedged daily implementation using past-12M momentum for monthly selection

> **Data expectation**: Two CSVs are expected:
> - `D:\A_share_market\20240910_fund_dayReturn.csv` (daily % returns; first column is date)
> - `D:\A_share_market\20240910_index_return.csv` (daily % returns; first column is date; contains `000300.SH`, `000905.SH`, `000906.SH`)


## How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place data files**
   Make sure the two CSV files are located as expected:
   ```
   D:\A_share_market\20240910_fund_dayReturn.csv
   D:\A_share_market\20240910_index_return.csv
   ```

3. **Run scripts in `src/`**
   From the project root directory, run for example:
   ```bash
   python src/aggregate_regression_single.py
   python src/aggregate_regression_twofactor.py
   python src/quintile_analysis.py
   python src/short_sell_strategy.py
   ```

4. **Check outputs**
   Results (cleaned data, regression outputs, plots, backtest statistics) will be saved under the `results/` folder as CSV or image files.

5. **Adjust paths if needed**
   If your data files are in a different location, modify the `FUND_DAILY` and `INDEX_DAILY` variables at the top of the scripts accordingly.

