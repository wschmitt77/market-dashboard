# Market Signals Dashboard — data feed and metric engine

This repository powers a daily QQQ / IVV / TQQQ signals dashboard built in Claude.

- `fetch/fetch_data.py` runs on GitHub Actions every morning (`.github/workflows/daily-data.yml`, 10:15 UTC) and
  commits raw public data under `data/raw/`: daily prices (Yahoo Finance, Stooq fallback), FRED macro series,
  Cboe volatility indices, S&P 500 and Nasdaq-100 constituent closes for breadth, AAA gas prices, Atlanta Fed GDPNow,
  the Cleveland Fed inflation nowcast, ISM PMI, Cboe put/call ratios, Invesco QQQ holdings, multpl.com valuation
  tables and the FOMC / CPI / payrolls calendars. `data/manifest.json` records what succeeded.
- `pipeline/pipeline.py` turns the raw files into the dashboard's 50 metrics, three composite scores (regime light,
  dip-buy score, heat score), sector rotation table, backtests and the insight scorecard. It runs inside the scheduled
  Claude task, which also writes the ten daily insights and refreshes the dashboard.
- `config.json` holds tickers, FRED series, thresholds and the example TQQQ sizing ladder. Edit it to change the rules.
- `pipeline/INSIGHTS_FORMAT.md` and `pipeline/MORNING_RUN.md` document the daily procedure.
- `pipeline/make_synthetic.py` writes fake data in the same layout for offline testing only.

Nothing here is investment advice. All data comes from public sources and may be delayed or revised.

## Manual run

```bash
pip install -r fetch/requirements.txt
python fetch/fetch_data.py            # ~5–10 minutes; writes data/raw and data/manifest.json
python pipeline/pipeline.py --repo . --out out
```

## Allocation plan

`pipeline/allocation_backtest.py` tests the owner's plan — an unlevered ladder at 14/20/27% QQQ drawdowns, a leveraged
sleeve sized by depth that opens only after two weekly closes above the 200-day, and a reserve of bills, Treasuries and
gold — across the 2000, 2008, 2020 and 2022 bears and across rolling seven-year windows since 2000, with 2x/3x funds
simulated before their inception. `pipeline/plan.py` applies the same rules to a private portfolio file (kept in the
dashboard's store, never in this repo) and writes the "Recommended allocation" datasets the dashboard shows at the top.
Starting amounts are command-line parameters; the defaults are placeholders. None of it is advice.
