# A-share Fund Analysis & Market-Neutral Strategies

This repository contains a reproducible Python pipeline for:
- Cleaning and transforming daily fund & index returns
- Aggregating to monthly returns
- Plotting fund vs. index cumulative performance
- Single- and two-factor regressions (aggregate fund vs HS300/ZZ500)
- Quintile ranking backtests
- A short/hedged daily implementation using past-12M momentum for monthly selection

> **Data expectation**: Two CSVs are expected:
> - `D:\A share market\20240910_fund_dayReturn.csv` (daily % returns; first column is date)
> - `D:\A share market\20240910_index_return.csv` (daily % returns; first column is date; contains `000300.SH`, `000905.SH`, `000906.SH`)
