#!/usr/bin/env python3
"""
Developer fixture: writes SYNTHETIC raw files in the same layout as fetch_data.py
so the metric engine can be exercised offline. Never use its output as market data.

    python pipeline/make_synthetic.py /path/to/fake_repo_root
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("synthetic_repo")
raw = root / "data" / "raw"
cfg = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
rng = np.random.default_rng(7)

bdays = pd.bdate_range("2004-01-02", "2026-10-28")
n = len(bdays)

# Market factor with regime shifts
mkt = rng.normal(0.0004, 0.012, n)
vix = np.empty(n)
v = 18.0
for i in range(n):
    v = max(9.0, v + 0.08 * (18 - v) + rng.normal(0, 1.2) - 25 * mkt[i])
    vix[i] = v


def path(mu, sigma, beta=1.0, start=100.0, extra=None):
    r = mu + beta * mkt + rng.normal(0, sigma, n) + (extra if extra is not None else 0)
    p = start * np.exp(np.cumsum(r))
    return p, r


def write_price(sym, close, vol_scale=1e7, start_idx=0):
    c = close[start_idx:]
    d = bdays[start_idx:]
    o = c * (1 + rng.normal(0, 0.003, len(c)))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, len(c))))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, len(c))))
    df = pd.DataFrame({"date": d.strftime("%Y-%m-%d"), "open": o.round(2), "high": h.round(2), "low": lo.round(2),
                       "close": c.round(2), "adj_close": c.round(2),
                       "volume": (rng.integers(5, 15, len(c)) * vol_scale).astype(int)})
    (raw / "prices").mkdir(parents=True, exist_ok=True)
    df.to_csv(raw / "prices" / f"{sym.replace('^', '_')}.csv", index=False)
    return df


qqq, rq = path(0.0002, 0.006, beta=1.15, start=40)
ivv, _ = path(0.0002, 0.003, beta=1.0, start=110)
spy, _ = path(0.0002, 0.003, beta=1.0, start=110)
# TQQQ: 3x daily QQQ return minus fees, from 2010-02-11
rt = 3 * rq - 0.0000375
tq = 10 * np.exp(np.cumsum(rt))
write_price("QQQ", qqq); write_price("IVV", ivv); write_price("SPY", spy)
write_price("TQQQ", tq, start_idx=int(np.searchsorted(bdays, pd.Timestamp("2010-02-11"))))
for s in cfg["sector_etfs"]:
    p, _ = path(0.0002, 0.007, beta=rng.uniform(0.6, 1.3), start=rng.uniform(30, 150))
    write_price(s, p)
for s in cfg["ratio_tickers"]:
    beta = 0.1 if s in ("HYG", "IEF", "TLT", "UUP", "GLD") else rng.uniform(0.7, 1.3)
    p, _ = path(0.0001, 0.004, beta=beta, start=rng.uniform(30, 150))
    write_price(s, p)
write_price("^VIX", vix); write_price("^VXN", vix * 1.15); write_price("^MOVE", 60 + vix * 2.5); write_price("^NDX", qqq * 100); write_price("^GSPC", ivv * 10)

(raw / "cboe").mkdir(parents=True, exist_ok=True)
for idx, mult in (("VIX", 1.0), ("VIX3M", 1.08), ("VIX9D", 0.97), ("VXN", 1.15), ("SKEW", 7.0)):
    c = vix * mult if idx != "SKEW" else 120 + (vix - 18) * 0.8 + rng.normal(0, 4, n)
    pd.DataFrame({"date": bdays.strftime("%Y-%m-%d"), "open": c, "high": c * 1.02, "low": c * 0.98, "close": c.round(2)}).to_csv(raw / "cboe" / f"{idx}.csv", index=False)

(raw / "fred").mkdir(parents=True, exist_ok=True)
def rw(start, sigma, lo, hi, freq="D"):
    idx = bdays if freq == "D" else pd.date_range("2004-01-02", "2026-10-28", freq=freq)
    x = np.clip(start + np.cumsum(rng.normal(0, sigma, len(idx))), lo, hi)
    return pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "value": np.round(x, 3)})

series = {
    "DGS10": rw(4.0, 0.03, 0.5, 6), "DGS2": rw(3.8, 0.03, 0.1, 6), "DGS1": rw(3.9, 0.03, 0.1, 6), "DGS6MO": rw(4.0, 0.03, 0.1, 6),
    "DGS3MO": rw(4.2, 0.03, 0.0, 6), "DTB3": rw(4.1, 0.03, 0.0, 6), "DFII10": rw(1.8, 0.02, -1.5, 3), "T10Y2Y": rw(0.2, 0.02, -1.2, 2.5),
    "T10Y3M": rw(-0.2, 0.02, -2, 3), "T10YIE": rw(2.3, 0.01, 1, 3.5), "T5YIFR": rw(2.2, 0.01, 1, 3.5), "BAMLH0A0HYM2": rw(3.5, 0.03, 2.5, 9),
    "BAMLC0A0CM": rw(1.0, 0.01, 0.6, 3), "VIXCLS": pd.DataFrame({"date": bdays.strftime("%Y-%m-%d"), "value": vix.round(2)}),
    "DTWEXBGS": rw(120, 0.2, 100, 135), "DCOILWTICO": rw(70, 0.8, 30, 130), "DFF": rw(4.3, 0.005, 0, 6),
    "WALCL": rw(7_000_000, 5000, 4_000_000, 9_000_000, "W-WED"), "WTREGEN": rw(700_000, 8000, 100_000, 1_200_000, "W-WED"),
    "RRPONTSYD": rw(200_000, 5000, 0, 2_500_000), "NFCI": rw(-0.4, 0.02, -1, 2, "W-FRI"), "ICSA": rw(220_000, 2500, 180_000, 400_000, "W-SAT"),
    "CCSA": rw(1_850_000, 8000, 1_300_000, 2_500_000, "W-SAT"), "UMCSENT": rw(65, 1.5, 50, 100, "MS"), "MICH": rw(3.0, 0.1, 2, 6, "MS"),
    "GASREGW": rw(3.1, 0.03, 2.0, 5.0, "W-MON"), "CPIAUCSL": pd.DataFrame({"date": pd.date_range("2004-01-01", "2026-09-01", freq="MS").strftime("%Y-%m-%d"), "value": np.round(180 * np.exp(np.cumsum(rng.normal(0.0022, 0.0015, 273))), 3)}),
    "CPILFESL": pd.DataFrame({"date": pd.date_range("2004-01-01", "2026-09-01", freq="MS").strftime("%Y-%m-%d"), "value": np.round(190 * np.exp(np.cumsum(rng.normal(0.0021, 0.001, 273))), 3)}),
    "PCEPILFE": pd.DataFrame({"date": pd.date_range("2004-01-01", "2026-08-01", freq="MS").strftime("%Y-%m-%d"), "value": np.round(90 * np.exp(np.cumsum(rng.normal(0.0018, 0.001, 272))), 3)}),
    "UNRATE": rw(4.2, 0.05, 3.3, 10, "MS"), "PAYEMS": rw(158_000, 150, 130_000, 170_000, "MS"),
    "SP500": pd.DataFrame({"date": bdays[-2500:].strftime("%Y-%m-%d"), "value": (ivv[-2500:] * 10).round(2)}),
    "NASDAQ100": pd.DataFrame({"date": bdays[-2500:].strftime("%Y-%m-%d"), "value": (qqq[-2500:] * 100).round(2)}),
}
for k, df in series.items():
    df.to_csv(raw / "fred" / f"{k}.csv", index=False)

# breadth: 500 / 100 synthetic constituents, 2 years
(raw / "breadth").mkdir(parents=True, exist_ok=True)
for label, count in (("spx", 500), ("ndx", 100)):
    days = bdays[-520:]
    cols = {}
    for i in range(count):
        p, _ = path(0.0002, 0.012, beta=rng.uniform(0.5, 1.5), start=rng.uniform(20, 300))
        cols[f"SYM{i:03d}"] = p[-520:].round(2)
    wide = pd.DataFrame(cols, index=days.strftime("%Y-%m-%d"))
    wide.index.name = "date"
    wide.reset_index().to_csv(raw / "breadth" / f"{label}_closes.csv", index=False)
    pd.DataFrame({"symbol": list(cols)}).to_csv(raw / "breadth" / f"{label}_constituents.csv", index=False)

(raw / "misc").mkdir(parents=True, exist_ok=True)
pd.DataFrame({"date": ["2026-10-27", "2026-10-28"], "regular": [3.05, 3.07], "yesterday": [3.04, 3.05], "week_ago": [3.10, 3.11],
              "month_ago": [3.20, 3.19], "year_ago": [3.15, 3.14], "diesel": [3.60, 3.61]}).to_csv(raw / "misc" / "aaa_gas.csv", index=False)
pd.DataFrame({"date": ["2026-10-17", "2026-10-28"], "estimate": [2.9, 3.1], "quarter": ["2026 Q3", "2026 Q3"], "fetched": ["2026-10-27", "2026-10-28"]}).to_csv(raw / "misc" / "gdpnow.csv", index=False)
pd.DataFrame([
    {"date": "2026-10-28", "table": 0, "row": "CPI", "col": "October 2026 month-over-month percent change", "value": 0.25},
    {"date": "2026-10-28", "table": 0, "row": "CPI", "col": "October 2026 year-over-year percent change", "value": 2.9},
    {"date": "2026-10-28", "table": 0, "row": "Core CPI", "col": "October 2026 month-over-month percent change", "value": 0.28},
    {"date": "2026-10-28", "table": 0, "row": "Core CPI", "col": "October 2026 year-over-year percent change", "value": 3.0},
    {"date": "2026-10-28", "table": 0, "row": "PCE", "col": "October 2026 month-over-month percent change", "value": 0.22},
    {"date": "2026-10-28", "table": 0, "row": "PCE", "col": "October 2026 year-over-year percent change", "value": 2.6},
    {"date": "2026-10-28", "table": 0, "row": "Core PCE", "col": "October 2026 month-over-month percent change", "value": 0.24},
    {"date": "2026-10-28", "table": 0, "row": "Core PCE", "col": "October 2026 year-over-year percent change", "value": 2.8},
]).to_csv(raw / "misc" / "cleveland_nowcast.csv", index=False)
pd.DataFrame({"date": ["2026-10-01"], "pmi": [49.2], "new_orders": [48.5], "services_pmi": [52.1], "release_month": ["September 2026"]}).to_csv(raw / "misc" / "ism.csv", index=False)
pcd = bdays[-60:]
pd.DataFrame({"date": pcd.strftime("%Y-%m-%d"), "equity_pc": np.round(np.clip(0.62 + rng.normal(0, 0.08, 60), 0.35, 1.3), 2),
              "index_pc": np.round(1.2 + rng.normal(0, 0.15, 60), 2), "total_pc": np.round(0.9 + rng.normal(0, 0.1, 60), 2), "data_date": pcd.strftime("%Y-%m-%d")}).to_csv(raw / "misc" / "cboe_putcall.csv", index=False)
hold = pd.DataFrame({"date": "2026-10-28", "ticker": [f"T{i}" for i in range(100)], "name": [f"Company {i}" for i in range(100)],
                     "weight": np.round(np.sort(rng.dirichlet(np.ones(100) * 0.3) * 100)[::-1], 3)})
hold.to_csv(raw / "misc" / "qqq_holdings.csv", index=False)
pd.DataFrame({"date": ["2026-10-28"], "top10_weight": [float(hold["weight"].head(10).sum())], "n_holdings": [100]}).to_csv(raw / "misc" / "qqq_top10_history.csv", index=False)
months = pd.date_range("2004-01-01", "2026-10-01", freq="MS")
pd.DataFrame({"date": months.strftime("%Y-%m-%d"), "value": np.round(np.clip(18 + np.cumsum(rng.normal(0, 0.4, len(months))), 12, 40), 2)}).to_csv(raw / "misc" / "sp500_pe.csv", index=False)
pd.DataFrame({"date": months.strftime("%Y-%m-%d"), "value": np.round(np.clip(26 + np.cumsum(rng.normal(0, 0.4, len(months))), 15, 45), 2)}).to_csv(raw / "misc" / "shiller_cape.csv", index=False)
(raw / "misc" / "calendar.json").write_text(json.dumps({"fetched": "2026-10-28", "fomc": ["2026-10-28", "2026-12-09", "2027-01-27"],
                                                         "cpi": ["2026-11-10", "2026-12-10"], "nfp": ["2026-11-06", "2026-12-04"], "notes": []}))
(root / "data" / "manifest.json").write_text(json.dumps({"run_at": "2026-10-29T10:20:00+00:00", "sources": {"prices/QQQ": {"status": "ok", "last_date": "2026-10-28"}},
                                                          "summary": {"ok": 1, "total": 1, "failed": []}}))
print("synthetic raw data written under", raw)
