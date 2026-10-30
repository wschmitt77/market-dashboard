#!/usr/bin/env python3
"""
Data feed for the Market Signals Dashboard.

Runs on GitHub Actions (full internet) and writes raw, source-faithful CSV/JSON
files under data/raw/. Every source is independent: a failure is logged in
data/manifest.json and the previous file is kept, so one broken site never
blanks the dashboard.

Sources (all public, no API keys):
  prices    Yahoo Finance chart API (fallback: Stooq)        -> data/raw/prices/<TICKER>.csv
  fred      FRED fredgraph.csv                                -> data/raw/fred/<SERIES>.csv
  cboe      Cboe index history (VIX, VIX3M, VIX9D, VXN, SKEW) -> data/raw/cboe/<INDEX>.csv
  breadth   S&P 500 / Nasdaq-100 constituents (Wikipedia) and
            their daily closes (Yahoo spark)                  -> data/raw/breadth/*.csv
  misc      AAA gas, Atlanta Fed GDPNow, Cleveland Fed inflation nowcast,
            ISM PMI, Cboe put/call, Invesco QQQ holdings, multpl P/E & CAPE,
            FOMC / CPI / payrolls calendars                   -> data/raw/misc/*
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import threading
import time
import traceback
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CONFIG = json.loads((ROOT / "config.json").read_text())

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept": "text/html,application/json,text/csv,*/*;q=0.8",
           "Accept-Language": "en-US,en;q=0.9"}

MANIFEST: dict = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sources": {}}
ET = ZoneInfo("America/New_York")
FRED_API_KEY = os.environ.get("FRED_API_KEY", "").strip()


def drop_incomplete_today(df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo includes the in-progress session as a bar; drop a bar dated today when the US close hasn't happened."""
    if df is None or df.empty or "date" not in df.columns:
        return df
    now_et = datetime.now(ET)
    if now_et.hour < 16 or (now_et.hour == 16 and now_et.minute < 15):
        today = now_et.strftime("%Y-%m-%d")
        df = df[df["date"].astype(str) != today]
    return df


# ----------------------------------------------------------------------------- helpers
LOG_PATH = ROOT / "data" / "fetch_log.txt"


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a") as fh:
            fh.write(line + "\n")
    except Exception:  # noqa: BLE001
        pass


def save_manifest() -> None:
    try:
        ok = sum(1 for v in MANIFEST["sources"].values() if v["status"] == "ok")
        MANIFEST["summary"] = {"ok": ok, "total": len(MANIFEST["sources"]),
                               "failed": [k for k, v in MANIFEST["sources"].items() if v["status"] not in ("ok", "partial")]}
        (ROOT / "data").mkdir(parents=True, exist_ok=True)
        (ROOT / "data" / "manifest.json").write_text(json.dumps(MANIFEST, indent=1))
    except Exception:  # noqa: BLE001
        pass


def http_get(url: str, *, params=None, timeout=20, retries=2, sleep=1.5, headers=None) -> requests.Response:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers or HEADERS, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r
            last = RuntimeError(f"HTTP {r.status_code} for {url}")
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(sleep * (attempt + 1))
    raise last  # type: ignore[misc]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def merge_by_date(new: pd.DataFrame, path: Path, keep_days: int | None = None) -> pd.DataFrame:
    """Union an existing file with new rows; new values win on the same date."""
    if path.exists():
        try:
            old = pd.read_csv(path)
            if "date" in old.columns and len(old):
                both = pd.concat([old, new], ignore_index=True)
                both["date"] = pd.to_datetime(both["date"]).dt.strftime("%Y-%m-%d")
                both = both.drop_duplicates(subset="date", keep="last").sort_values("date")
                new = both
        except Exception:  # noqa: BLE001
            pass
    new = new.sort_values("date").reset_index(drop=True)
    if keep_days:
        new = new.tail(keep_days).reset_index(drop=True)
    return new


def record(name: str, status: str, **kw) -> None:
    entry = {"status": status, "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    entry.update({k: v for k, v in kw.items() if v is not None})
    MANIFEST["sources"][name] = entry
    log(f"{name}: {status} " + " ".join(f"{k}={v}" for k, v in kw.items() if v is not None and k != "error"))
    if "error" in kw and kw["error"]:
        log(f"   error: {str(kw['error'])[:300]}")
    save_manifest()


def guarded(name: str, fn, *args, budget: float | None = None, **kwargs):
    """Run a fetcher in a daemon thread with a time budget; on failure or timeout record it and keep going."""
    budget = budget or float(CONFIG.get("source_time_budget_s", 120))
    box: dict = {}

    def runner():
        try:
            box["result"] = fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            box["error"] = e

    th = threading.Thread(target=runner, daemon=True, name=name)
    t0 = time.time()
    th.start()
    th.join(budget)
    if th.is_alive():
        record(name, "timeout", error=f"gave up after {budget:.0f}s (source hanging or rate-limited)")
        return None
    if "error" in box:
        e = box["error"]
        record(name, "failed", error=f"{type(e).__name__}: {e}")
        return None
    log(f"   ({name} took {time.time() - t0:.1f}s)")
    return box.get("result")


# ----------------------------------------------------------------------------- prices
def yahoo_chart(symbol: str, rng: str = "max") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(symbol)}"
    r = http_get(url, params={"range": rng, "interval": "1d", "events": "div,split",
                              "includeAdjustedClose": "true"}, retries=1)
    js = r.json()
    res = js["chart"]["result"][0]
    ts = res.get("timestamp") or []
    q = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose") or [None] * len(ts)
    off = res.get("meta", {}).get("gmtoffset", 0) or 0
    rows = []
    for i, t in enumerate(ts):
        c = q["close"][i] if i < len(q["close"]) else None
        if c is None:
            continue
        d = datetime.fromtimestamp(t + off, tz=timezone.utc).strftime("%Y-%m-%d")
        rows.append({"date": d, "open": q["open"][i], "high": q["high"][i], "low": q["low"][i],
                     "close": c, "adj_close": adj[i] if adj[i] is not None else c,
                     "volume": q["volume"][i]})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Yahoo returned no rows")
    return df


def stooq_daily(symbol: str) -> pd.DataFrame:
    sym = symbol.lower().replace("^", "^")
    url = f"https://stooq.com/q/d/l/?s={sym}.us&i=d" if not symbol.startswith("^") else f"https://stooq.com/q/d/l/?s={sym}&i=d"
    r = http_get(url)
    text = r.text
    if "Exceeded" in text[:200] or "No data" in text[:200]:
        raise RuntimeError(f"Stooq: {text[:80]}")
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.lower() for c in df.columns]
    if "close" not in df.columns:
        raise RuntimeError("Stooq: unexpected columns " + ",".join(df.columns))
    df = df.rename(columns={"date": "date"})
    df["adj_close"] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = None
    return df[["date", "open", "high", "low", "close", "adj_close", "volume"]]


YAHOO_RAW_FAILURES = 0          # circuit breaker for the raw query1 endpoints (they 429 cloud runners)
YAHOO_RAW_LIMIT = 4


def yahoo_raw_available() -> bool:
    return YAHOO_RAW_FAILURES < YAHOO_RAW_LIMIT


def note_yahoo_raw(ok: bool) -> None:
    global YAHOO_RAW_FAILURES
    YAHOO_RAW_FAILURES = 0 if ok else YAHOO_RAW_FAILURES + 1


def yf_bulk(symbols: list[str], period: str = "max", chunk: int = 80) -> dict[str, pd.DataFrame]:
    """Download daily bars for many symbols with yfinance (browser impersonation, shared session).
    Returns {symbol: DataFrame(date, open, high, low, close, adj_close, volume)}; symbols that fail are left out."""
    try:
        import yfinance as yf  # noqa: WPS433
    except ImportError:
        log("yfinance not installed; skipping bulk download")
        return {}
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        try:
            df = yf.download(batch, period=period, interval="1d", auto_adjust=False, actions=False,
                             group_by="ticker", threads=True, progress=False, timeout=30)
        except Exception as e:  # noqa: BLE001
            log(f"yfinance batch {i // chunk + 1}: {type(e).__name__}: {e}")
            time.sleep(3)
            continue
        if df is None or df.empty:
            time.sleep(2)
            continue
        multi = isinstance(df.columns, pd.MultiIndex)
        for sym in batch:
            try:
                sub = df[sym] if multi else df
            except KeyError:
                continue
            if "Close" not in sub.columns:
                continue
            sub = sub.dropna(subset=["Close"])
            if sub.empty:
                continue
            idx = pd.to_datetime(sub.index)
            try:
                idx = idx.tz_localize(None)
            except (TypeError, AttributeError):
                pass
            rows = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "open": sub["Open"].values, "high": sub["High"].values,
                                 "low": sub["Low"].values, "close": sub["Close"].values,
                                 "adj_close": sub["Adj Close"].values if "Adj Close" in sub.columns else sub["Close"].values,
                                 "volume": sub["Volume"].values if "Volume" in sub.columns else None})
            rows = rows.drop_duplicates("date", keep="last")
            out[sym] = rows
        time.sleep(1.0)
    return out


def fetch_price(symbol: str, bulk: pd.DataFrame | None = None) -> None:
    name = f"prices/{symbol}"
    path = RAW / "prices" / f"{symbol.replace('^', '_')}.csv"
    df, src, err = None, None, None
    if bulk is not None and len(bulk):
        df, src = bulk, "yfinance"
    else:
        chain = []
        if yahoo_raw_available():
            chain.append(("yahoo", yahoo_chart))
        chain.append(("stooq", stooq_daily))
        for label, fn in chain:
            try:
                df = fn(symbol)
                src = label
                if label == "yahoo":
                    note_yahoo_raw(True)
                break
            except Exception as e:  # noqa: BLE001
                err = f"{label}: {type(e).__name__}: {e}"
                if label == "yahoo":
                    note_yahoo_raw(False)
                time.sleep(1.0)
    if df is None:
        record(name, "kept_previous" if path.exists() else "failed", error=err)
        return
    df = drop_incomplete_today(df)
    df = merge_by_date(df, path)
    # a previously saved partial bar for the same (still open) session is dropped too
    df = drop_incomplete_today(df)
    write_csv(df, path)
    record(name, "ok", source=src, rows=len(df), last_date=str(df["date"].iloc[-1]))


# ----------------------------------------------------------------------------- FRED
FRED_FAILURES = 0


def fred_api(series: str) -> pd.DataFrame:
    r = http_get("https://api.stlouisfed.org/fred/series/observations",
                 params={"series_id": series, "api_key": FRED_API_KEY, "file_type": "json",
                         "observation_start": "1990-01-01"}, timeout=25, retries=2)
    obs = r.json().get("observations", [])
    df = pd.DataFrame([{"date": o["date"], "value": o["value"]} for o in obs])
    if df.empty:
        raise RuntimeError("FRED API returned no observations")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def fred_graph(series: str) -> pd.DataFrame:
    r = http_get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series}, timeout=15, retries=1)
    df = pd.read_csv(io.StringIO(r.text))
    dcol = "observation_date" if "observation_date" in df.columns else df.columns[0]
    df = df.rename(columns={dcol: "date", series: "value"})
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])[["date", "value"]]


def fetch_fred(series: str) -> None:
    global FRED_FAILURES
    name = f"fred/{series}"
    path = RAW / "fred" / f"{series}.csv"
    if FRED_FAILURES >= 3:
        record(name, "kept_previous" if path.exists() else "failed", error="skipped: FRED unreachable from this runner (3 earlier timeouts)")
        return
    errs = []
    df = None
    if FRED_API_KEY:
        try:
            df = fred_api(series)
        except Exception as e:  # noqa: BLE001
            errs.append(f"api: {type(e).__name__}: {e}")
    if df is None:
        try:
            df = fred_graph(series)
        except Exception as e:  # noqa: BLE001
            errs.append(f"fredgraph: {type(e).__name__}: {e}")
    if df is None:
        FRED_FAILURES += 1
        record(name, "kept_previous" if path.exists() else "failed", error="; ".join(errs)[:300])
        return
    FRED_FAILURES = 0
    df = merge_by_date(df, path)
    write_csv(df, path)
    record(name, "ok", source="fred_api" if FRED_API_KEY else "fredgraph", rows=len(df), last_date=str(df["date"].iloc[-1]))
    time.sleep(0.2)


# ----------------------------------------------------------------------------- fallbacks for FRED series
def write_fallback_series(series: str, df: pd.DataFrame, source: str) -> None:
    """Fill fred/<series>.csv from an alternative source when FRED itself failed."""
    name = f"fred/{series}"
    path = RAW / "fred" / f"{series}.csv"
    if MANIFEST["sources"].get(name, {}).get("status") == "ok":
        return
    df = df.dropna(subset=["value"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").round(4)
    df = drop_incomplete_today(df[["date", "value"]]).drop_duplicates("date", keep="last")
    if df.empty:
        return
    df = merge_by_date(df, path)
    write_csv(df, path)
    record(name, "ok", source=source, rows=len(df), last_date=str(df["date"].iloc[-1]))


def treasury_curve(kind: str, years: list[int]) -> pd.DataFrame:
    frames = []
    for y in years:
        url = (f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/all/{y}"
               f"?type={kind}&field_tdr_date_value={y}&page&_format=csv")
        try:
            r = http_get(url, timeout=25, retries=1)
            t = pd.read_csv(io.StringIO(r.text))
            if "Date" in t.columns and len(t):
                frames.append(t)
        except Exception as e:  # noqa: BLE001
            log(f"treasury {kind} {y}: {type(e).__name__}: {e}")
        time.sleep(0.5)
    if not frames:
        raise RuntimeError("no Treasury CSV downloaded")
    df = pd.concat(frames, ignore_index=True)
    dates = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.drop(columns=["Date"])
    df.columns = [re.sub(r"[^a-z0-9]", "", str(c).lower()) for c in df.columns]
    df.insert(0, "date", dates)
    return df.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")


def fetch_fred_fallbacks() -> None:
    missing = {s for s in CONFIG["fred_series"] if MANIFEST["sources"].get(f"fred/{s}", {}).get("status") != "ok"}
    if not missing:
        return
    this_year = date.today().year
    years = [this_year - 3, this_year - 2, this_year - 1, this_year]
    curve_targets = {"DGS3MO": "3mo", "DGS6MO": "6mo", "DGS1": "1yr", "DGS2": "2yr", "DGS10": "10yr", "DTB3": "3mo"}
    if missing & (set(curve_targets) | {"T10Y2Y", "T10Y3M"}):
        try:
            cv = treasury_curve("daily_treasury_yield_curve", years)
            for series, col in curve_targets.items():
                if series in missing and col in cv.columns:
                    write_fallback_series(series, cv.rename(columns={col: "value"})[["date", "value"]], "treasury.gov daily yield curve")
            if "10yr" in cv.columns and "2yr" in cv.columns and "T10Y2Y" in missing:
                write_fallback_series("T10Y2Y", pd.DataFrame({"date": cv["date"], "value": pd.to_numeric(cv["10yr"], errors="coerce") - pd.to_numeric(cv["2yr"], errors="coerce")}), "treasury.gov (10y − 2y)")
            if "10yr" in cv.columns and "3mo" in cv.columns and "T10Y3M" in missing:
                write_fallback_series("T10Y3M", pd.DataFrame({"date": cv["date"], "value": pd.to_numeric(cv["10yr"], errors="coerce") - pd.to_numeric(cv["3mo"], errors="coerce")}), "treasury.gov (10y − 3m)")
        except Exception as e:  # noqa: BLE001
            log(f"treasury yield curve fallback failed: {type(e).__name__}: {e}")
    if "DFII10" in missing:
        try:
            rv = treasury_curve("daily_treasury_real_yield_curve", years)
            if "10yr" in rv.columns:
                write_fallback_series("DFII10", rv.rename(columns={"10yr": "value"})[["date", "value"]], "treasury.gov daily real yield curve")
        except Exception as e:  # noqa: BLE001
            log(f"treasury real yield fallback failed: {type(e).__name__}: {e}")
    # Yahoo proxies for what is still missing
    still = {s for s in missing if MANIFEST["sources"].get(f"fred/{s}", {}).get("status") != "ok"}
    proxies = {"DGS10": ("^TNX", "Yahoo ^TNX (10y yield)", None), "DTB3": ("^IRX", "Yahoo ^IRX (13-week bill)", None),
               "DGS3MO": ("^IRX", "Yahoo ^IRX (13-week bill)", None), "DCOILWTICO": ("CL=F", "Yahoo CL=F (WTI front month)", None),
               "DTWEXBGS": ("DX-Y.NYB", "Yahoo DXY (dollar index; proxy for the broad index)", None),
               "DFF": ("ZQ=F", "Yahoo ZQ=F (30-day fed funds futures, 100 − price)", lambda s: 100 - s),
               "VIXCLS": ("^VIX", "Yahoo ^VIX", None)}
    wanted = {s: proxies[s] for s in still if s in proxies}
    if wanted:
        bulk = yf_bulk(sorted({v[0] for v in wanted.values()}), period="5y")
        for series, (sym, label, fn) in wanted.items():
            df = bulk.get(sym)
            if df is None or df.empty:
                continue
            vals = pd.to_numeric(df["close"], errors="coerce")
            if fn:
                vals = fn(vals)
            write_fallback_series(series, pd.DataFrame({"date": df["date"], "value": vals}), label)
    # Chicago Fed NFCI CSV
    if "NFCI" in still:
        try:
            r = http_get("https://www.chicagofed.org/~/media/publications/nfci/nfci-data-series-csv.csv", timeout=25, retries=1)
            t = pd.read_csv(io.StringIO(r.text))
            dcol = next((c for c in t.columns if "date" in str(c).lower() or "friday" in str(c).lower()), t.columns[0])
            vcol = next((c for c in t.columns if str(c).strip().upper() == "NFCI"), None)
            if vcol:
                write_fallback_series("NFCI", pd.DataFrame({"date": pd.to_datetime(t[dcol], errors="coerce"), "value": pd.to_numeric(t[vcol], errors="coerce")}).dropna(), "Chicago Fed NFCI csv")
        except Exception as e:  # noqa: BLE001
            log(f"NFCI fallback failed: {type(e).__name__}: {e}")
    # BLS public API (no key: ~3 years of monthly data) for CPI, core CPI, unemployment, payrolls
    bls_ids = {"CPIAUCSL": "CUSR0000SA0", "CPILFESL": "CUSR0000SA0L1E", "UNRATE": "LNS14000000", "PAYEMS": "CES0000000001"}
    for series, sid in bls_ids.items():
        if series not in still:
            continue
        try:
            r = http_get(f"https://api.bls.gov/publicAPI/v2/timeseries/data/{sid}", timeout=25, retries=1,
                         headers={**HEADERS, "User-Agent": "market-dashboard/1.0 (+https://github.com/wschmitt77/market-dashboard)"})
            js = r.json()
            rows = []
            for s in js.get("Results", {}).get("series", []):
                for d in s.get("data", []):
                    if str(d.get("period", "")).startswith("M") and d["period"] != "M13":
                        rows.append({"date": f"{d['year']}-{int(d['period'][1:]):02d}-01", "value": d["value"]})
            df = pd.DataFrame(rows)
            if df.empty:
                raise RuntimeError("BLS API returned no monthly data: " + ", ".join(js.get("message", [])[:2]))
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            write_fallback_series(series, df, f"BLS public API {sid}")
        except Exception as e:  # noqa: BLE001
            log(f"BLS fallback {series} failed: {type(e).__name__}: {e}")
        time.sleep(0.5)


# ----------------------------------------------------------------------------- Cboe
def fetch_cboe(index: str) -> None:
    name = f"cboe/{index}"
    path = RAW / "cboe" / f"{index}.csv"
    try:
        r = http_get(f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{index}_History.csv")
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip().lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        if "close" not in df.columns:  # e.g. SKEW_History.csv has DATE,SKEW
            vcol = next((c for c in df.columns if c != "date"), None)
            if vcol is None:
                raise RuntimeError("no value column in " + ",".join(df.columns))
            df = df.rename(columns={vcol: "close"})
        keep = [c for c in ["date", "open", "high", "low", "close"] if c in df.columns]
        df = df[keep]
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["close"])
        df = merge_by_date(df, path)
        write_csv(df, path)
        record(name, "ok", rows=len(df), last_date=str(df["date"].iloc[-1]))
    except Exception as e:  # noqa: BLE001
        record(name, "kept_previous" if path.exists() else "failed", error=f"{type(e).__name__}: {e}")


# ----------------------------------------------------------------------------- breadth
def wiki_tickers(url: str, col_candidates=("Symbol", "Ticker"), min_rows: int = 50) -> list[str]:
    r = http_get(url)
    tables = pd.read_html(io.StringIO(r.text))
    best: list[str] = []
    for t in tables:
        if len(t) < min_rows:
            continue
        t.columns = [" ".join(str(x) for x in c) if isinstance(c, tuple) else str(c) for c in t.columns]
        for c in t.columns:
            lc = c.strip().lower()
            if any(cand.lower() in lc for cand in col_candidates):
                syms = t[c].astype(str).str.strip().str.replace(".", "-", regex=False)
                syms = [s for s in syms if re.fullmatch(r"[A-Z0-9\-]{1,6}", s)]
                if len(syms) >= min_rows and len(syms) > len(best):
                    best = syms
    if best:
        return list(dict.fromkeys(best))
    raise RuntimeError(f"no constituents table found ({len(tables)} tables)")


def holdings_tickers() -> list[str]:
    """Nasdaq-100 members from the Invesco QQQ holdings file written earlier in the run."""
    p = RAW / "misc" / "qqq_holdings.csv"
    if not p.exists():
        raise RuntimeError("qqq_holdings.csv not available")
    h = pd.read_csv(p)
    syms = [str(s).strip().replace(".", "-") for s in h["ticker"].astype(str)]
    syms = [s for s in syms if re.fullmatch(r"[A-Z0-9\-]{1,6}", s)]
    if len(syms) < 80:
        raise RuntimeError(f"only {len(syms)} tickers in holdings file")
    return list(dict.fromkeys(syms))


def yahoo_spark(symbols: list[str], rng: str = "2y") -> dict[str, pd.Series]:
    url = "https://query1.finance.yahoo.com/v8/finance/spark"
    r = http_get(url, params={"symbols": ",".join(symbols), "range": rng, "interval": "1d"}, retries=1)
    js = r.json()
    out: dict[str, pd.Series] = {}
    for item in js.get("spark", {}).get("result", []):
        sym = item.get("symbol")
        resp = (item.get("response") or [None])[0]
        if not resp or not resp.get("timestamp"):
            continue
        off = resp.get("meta", {}).get("gmtoffset", 0) or 0
        closes = resp["indicators"]["quote"][0].get("close") or []
        idx = [datetime.fromtimestamp(t + off, tz=timezone.utc).strftime("%Y-%m-%d") for t in resp["timestamp"]]
        s = pd.Series(closes, index=idx, dtype="float64").dropna()
        s = s[~s.index.duplicated(keep="last")]
        if len(s):
            out[sym] = s
    return out


def fetch_breadth(label: str, url: str, col_candidates) -> None:
    name = f"breadth/{label}"
    wide_path = RAW / "breadth" / f"{label}_closes.csv"
    cons_path = RAW / "breadth" / f"{label}_constituents.csv"
    try:
        syms = None
        errs = []
        if label == "ndx":
            try:
                syms = holdings_tickers()
            except Exception as e:  # noqa: BLE001
                errs.append(f"holdings: {e}")
        if syms is None:
            try:
                syms = wiki_tickers(url, col_candidates, min_rows=90 if label == "ndx" else 400)
            except Exception as e:  # noqa: BLE001
                errs.append(f"wikipedia: {e}")
        if syms is None and cons_path.exists():
            syms = pd.read_csv(cons_path)["symbol"].astype(str).tolist()
            errs.append("using previous constituent list")
        if syms is None:
            raise RuntimeError("; ".join(errs))
        if errs:
            log(f"{name}: {'; '.join(errs)}")
        write_csv(pd.DataFrame({"symbol": syms}), cons_path)
        series: dict[str, pd.Series] = {}
        bulk = yf_bulk(syms, period="2y")
        for sym, df in bulk.items():
            s = pd.Series(df["close"].values, index=df["date"].values, dtype="float64").dropna()
            if len(s):
                series[sym] = s
        missing = [s for s in syms if s not in series]
        if missing and len(missing) <= 200 and yahoo_raw_available():
            for i in range(0, len(missing), 40):
                batch = missing[i:i + 40]
                try:
                    series.update(yahoo_spark(batch))
                    note_yahoo_raw(True)
                except Exception:  # noqa: BLE001
                    note_yahoo_raw(False)
                    if not yahoo_raw_available():
                        break
                time.sleep(0.6)
        if len(series) < len(syms) * 0.6:
            raise RuntimeError(f"only {len(series)} of {len(syms)} constituents downloaded")
        wide = pd.DataFrame(series)
        wide.index.name = "date"
        wide = wide.sort_index().reset_index()
        wide = drop_incomplete_today(wide)
        wide = merge_by_date(wide, wide_path, keep_days=CONFIG.get("breadth_history_days", 800))
        wide = drop_incomplete_today(wide)
        write_csv(wide, wide_path)
        record(name, "ok", symbols=len(syms), with_data=len(series), rows=len(wide), last_date=str(wide["date"].iloc[-1]))
    except Exception as e:  # noqa: BLE001
        record(name, "kept_previous" if wide_path.exists() else "failed", error=f"{type(e).__name__}: {e}")


# ----------------------------------------------------------------------------- misc sources
def today_iso() -> str:
    return date.today().isoformat()


def append_row(path: Path, row: dict, key: str = "date") -> pd.DataFrame:
    df = pd.DataFrame([row])
    if path.exists():
        try:
            old = pd.read_csv(path)
            df = pd.concat([old, df], ignore_index=True)
        except Exception:  # noqa: BLE001
            pass
    df = df.drop_duplicates(subset=key, keep="last").sort_values(key).reset_index(drop=True)
    write_csv(df, path)
    return df


def fetch_aaa_gas() -> None:
    name, path = "misc/aaa_gas", RAW / "misc" / "aaa_gas.csv"
    try:
        r = http_get("https://gasprices.aaa.com/")
        html = r.text
        tables = pd.read_html(io.StringIO(html))
        found = None
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if any("regular" in c for c in cols) and len(t) >= 3:
                found = t
                break
        if found is None:
            raise RuntimeError("national average table not found")
        found.columns = [str(c).strip().lower() for c in found.columns]
        label_col = found.columns[0]
        reg_col = [c for c in found.columns if "regular" in c][0]
        dsl_col = next((c for c in found.columns if "diesel" in c), None)

        def val(row_label: str):
            m = found[found[label_col].astype(str).str.lower().str.contains(row_label)]
            if m.empty:
                return None
            v = str(m.iloc[0][reg_col]).replace("$", "").strip()
            return float(v) if re.fullmatch(r"\d+\.\d+", v) else None

        def dval(row_label: str):
            if not dsl_col:
                return None
            m = found[found[label_col].astype(str).str.lower().str.contains(row_label)]
            if m.empty:
                return None
            v = str(m.iloc[0][dsl_col]).replace("$", "").strip()
            return float(v) if re.fullmatch(r"\d+\.\d+", v) else None

        row = {"date": today_iso(), "regular": val("current"), "yesterday": val("yesterday"),
               "week_ago": val("week"), "month_ago": val("month"), "year_ago": val("year"),
               "diesel": dval("current")}
        if row["regular"] is None:
            raise RuntimeError("could not parse current average")
        df = append_row(path, row)
        record(name, "ok", rows=len(df), regular=row["regular"])
    except Exception as e:  # noqa: BLE001
        record(name, "kept_previous" if path.exists() else "failed", error=f"{type(e).__name__}: {e}")


def fetch_gdpnow() -> None:
    name, path = "misc/gdpnow", RAW / "misc" / "gdpnow.csv"
    try:
        r = http_get("https://www.atlantafed.org/cqer/research/gdpnow")
        text = re.sub(r"<[^>]+>", " ", r.text)
        text = re.sub(r"\s+", " ", text)
        m = re.search(r"Latest estimate:?\s*(-?\d+(?:\.\d+)?)\s*percent", text, flags=re.I)
        if not m:
            i = text.lower().find("latest estimate")
            raise RuntimeError("estimate not found; page text near 'Latest estimate': " + (text[max(0, i - 40):i + 200] if i >= 0 else text[:200]))
        est = float(m.group(1))
        dm = re.search(r"([A-Z][a-z]+ \d{1,2}, \d{4})", text[m.end():m.end() + 120])
        as_of = pd.to_datetime(dm.group(1)).strftime("%Y-%m-%d") if dm else today_iso()
        q = re.search(r"for (?:real GDP growth .*?in the )?(first|second|third|fourth) quarter of (\d{4})", text)
        quarter = f"{q.group(2)} Q{['first', 'second', 'third', 'fourth'].index(q.group(1)) + 1}" if q else ""
        df = append_row(path, {"date": as_of, "estimate": est, "quarter": quarter, "fetched": today_iso()})
        record(name, "ok", rows=len(df), estimate=est, as_of=as_of)
    except Exception as e:  # noqa: BLE001
        record(name, "kept_previous" if path.exists() else "failed", error=f"{type(e).__name__}: {e}")


def fetch_cleveland_nowcast() -> None:
    name, path = "misc/cleveland_nowcast", RAW / "misc" / "cleveland_nowcast.csv"
    try:
        r = http_get("https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting")
        tables = pd.read_html(io.StringIO(r.text))
        rows = []
        for ti, t in enumerate(tables):
            t = t.copy()
            t.columns = [" ".join(str(x) for x in c) if isinstance(c, tuple) else str(c) for c in t.columns]
            first = t.columns[0]
            for _, rr in t.iterrows():
                label = str(rr[first]).strip()
                for c in t.columns[1:]:
                    v = str(rr[c]).strip()
                    if re.fullmatch(r"-?\d+(\.\d+)?%?", v):
                        rows.append({"date": today_iso(), "table": ti, "row": label, "col": c,
                                     "value": float(v.rstrip("%"))})
        if not rows:
            raise RuntimeError("no numeric nowcast tables parsed")
        new = pd.DataFrame(rows)
        if path.exists():
            old = pd.read_csv(path)
            old = old[old["date"] != today_iso()]
            new = pd.concat([old, new], ignore_index=True)
        new = new.sort_values(["date", "table"]).reset_index(drop=True)
        # keep the file small: last 120 days of snapshots
        keep_dates = sorted(new["date"].unique())[-120:]
        new = new[new["date"].isin(keep_dates)]
        write_csv(new, path)
        record(name, "ok", rows=len(rows), tables=len(tables))
    except Exception as e:  # noqa: BLE001
        record(name, "kept_previous" if path.exists() else "failed", error=f"{type(e).__name__}: {e}")


def fetch_ism() -> None:
    name, path = "misc/ism", RAW / "misc" / "ism.csv"
    try:
        out = {"date": today_iso(), "pmi": None, "new_orders": None, "services_pmi": None, "release_month": ""}
        candidates = ["https://www.ismworld.org/supply-management-news-and-reports/reports/ism-pmi-reports/pmi/",
                      "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/pmi/",
                      "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-pmi-reports/"]
        r = None
        for u in candidates:
            try:
                r = http_get(u, timeout=20, retries=1)
                base = u.rsplit("/", 2)[0] + "/"
                break
            except Exception as e:  # noqa: BLE001
                log(f"ism: {u} -> {type(e).__name__}: {e}")
        if r is None:
            raise RuntimeError("no ISM report page reachable")
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text))
        m = re.search(r"Manufacturing PMI[^.]{0,80}?(\d{2}\.\d)\s*percent", text) or \
            re.search(r"\bPMI\W{0,3}[^.]{0,60}?(\d{2}\.\d)\s*percent", text)
        if m:
            out["pmi"] = float(m.group(1))
        n = re.search(r"New Orders Index[^.]{0,120}?(\d{2}\.\d)\s*percent", text)
        if n:
            out["new_orders"] = float(n.group(1))
        mo = re.search(r"([A-Z][a-z]+ \d{4}) Manufacturing", text)
        if mo:
            out["release_month"] = mo.group(1)
        try:
            r2 = http_get(base + "services/")
            t2 = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r2.text))
            s = re.search(r"Services PMI[^.]{0,80}?(\d{2}\.\d)\s*percent", t2)
            if s:
                out["services_pmi"] = float(s.group(1))
        except Exception:  # noqa: BLE001
            pass
        if out["pmi"] is None and out["services_pmi"] is None:
            raise RuntimeError("no PMI number parsed from ISM pages")
        df = append_row(path, out)
        record(name, "ok", rows=len(df), pmi=out["pmi"], services=out["services_pmi"])
    except Exception as e:  # noqa: BLE001
        record(name, "kept_previous" if path.exists() else "failed", error=f"{type(e).__name__}: {e}")


def fetch_cboe_putcall() -> None:
    name, path = "misc/cboe_putcall", RAW / "misc" / "cboe_putcall.csv"
    try:
        r = http_get("https://www.cboe.com/us/options/market_statistics/daily/")
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text))
        def grab(label):
            m = re.search(label + r"\s*(\d\.\d{2})", text, flags=re.I)
            return float(m.group(1)) if m else None
        row = {"date": today_iso(), "equity_pc": grab(r"EQUITY PUT/CALL RATIO"),
               "index_pc": grab(r"INDEX PUT/CALL RATIO"), "total_pc": grab(r"TOTAL PUT/CALL RATIO"),
               "data_date": ""}
        if row["equity_pc"] is None and row["total_pc"] is None:
            raise RuntimeError("put/call ratios not found on page")
        df = append_row(path, row)
        record(name, "ok", rows=len(df), equity_pc=row["equity_pc"])
    except Exception as e:  # noqa: BLE001
        record(name, "kept_previous" if path.exists() else "failed", error=f"{type(e).__name__}: {e}")


def fetch_qqq_holdings() -> None:
    name = "misc/qqq_holdings"
    path = RAW / "misc" / "qqq_holdings.csv"
    hist = RAW / "misc" / "qqq_top10_history.csv"
    try:
        url = ("https://www.invesco.com/us/financial-products/etfs/holdings/main/holdings/0"
               "?audienceType=Investor&action=download&ticker=QQQ")
        r = http_get(url)
        lines = r.text.splitlines()
        hdr = next((i for i, ln in enumerate(lines) if "Holding Ticker" in ln or ("Weight" in ln and "Ticker" in ln)), 0)
        df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), engine="python", on_bad_lines="skip")
        df.columns = [str(c).strip() for c in df.columns]
        wcol = next((c for c in df.columns if "weight" in c.lower()), None)
        tcol = next((c for c in df.columns if c.lower() in ("holding ticker", "ticker", "symbol")), None)
        ncol = next((c for c in df.columns if c.lower() in ("name", "security name", "holding name")), None)
        if not wcol or not tcol:
            raise RuntimeError("weight/ticker columns not found: " + ",".join(df.columns))
        out = pd.DataFrame({"date": today_iso(), "ticker": df[tcol].astype(str).str.strip(),
                            "name": df[ncol].astype(str) if ncol else "",
                            "weight": pd.to_numeric(df[wcol].astype(str).str.replace("%", ""), errors="coerce")})
        out = out.dropna(subset=["weight"]).sort_values("weight", ascending=False).reset_index(drop=True)
        write_csv(out, path)
        top10 = float(out["weight"].head(10).sum())
        append_row(hist, {"date": today_iso(), "top10_weight": top10, "n_holdings": len(out)})
        record(name, "ok", holdings=len(out), top10_weight=round(top10, 2))
    except Exception as e:  # noqa: BLE001
        record(name, "kept_previous" if path.exists() else "failed", error=f"{type(e).__name__}: {e}")


def fetch_multpl(slug: str, fname: str) -> None:
    name, path = f"misc/{fname}", RAW / "misc" / f"{fname}.csv"
    try:
        r = http_get(f"https://www.multpl.com/{slug}/table/by-month")
        tables = pd.read_html(io.StringIO(r.text))
        t = max(tables, key=len)
        t.columns = [str(c).strip().lower() for c in t.columns]
        dcol = next(c for c in t.columns if "date" in c)
        vcol = next(c for c in t.columns if c != dcol)
        df = pd.DataFrame({"date": pd.to_datetime(t[dcol], errors="coerce"),
                           "value": pd.to_numeric(t[vcol].astype(str).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")})
        df = df.dropna().sort_values("date")
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        write_csv(df, path)
        record(name, "ok", rows=len(df), last_date=str(df["date"].iloc[-1]), last_value=float(df["value"].iloc[-1]))
    except Exception as e:  # noqa: BLE001
        record(name, "kept_previous" if path.exists() else "failed", error=f"{type(e).__name__}: {e}")


MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                      "August", "September", "October", "November", "December"], 1)}


def fetch_calendar() -> None:
    name, path = "misc/calendar", RAW / "misc" / "calendar.json"
    cal = {"fetched": today_iso(), "fomc": [], "cpi": [], "nfp": [], "notes": []}
    # FOMC decision days (second day of each meeting)
    try:
        r = http_get("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
        html = r.text
        blocks = re.split(r"(\d{4}) FOMC Meetings", html)
        for i in range(1, len(blocks) - 1, 2):
            year = int(blocks[i])
            body = blocks[i + 1]
            for mm, dd in re.findall(r'fomc-meeting__month[^>]*>\s*<strong>\s*([A-Za-z/]+)\s*</strong>.*?fomc-meeting__date[^>]*>\s*([\d\-–/ ]+?)\s*[*<]',
                                     body, flags=re.S):
                months = [m for m in re.split(r"/", mm) if m]
                days = [int(x) for x in re.findall(r"\d+", dd)]
                if not months or not days:
                    continue
                month = months[-1]
                month_num = MONTHS.get(month) or MONTHS.get({"Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April",
                                                             "Jun": "June", "Jul": "July", "Aug": "August", "Sep": "September",
                                                             "Sept": "September", "Oct": "October", "Nov": "November",
                                                             "Dec": "December"}.get(month, month))
                if not month_num:
                    continue
                day = days[-1]
                try:
                    cal["fomc"].append(date(year, month_num, day).isoformat())
                except ValueError:
                    continue
        cal["fomc"] = sorted(set(cal["fomc"]))
        if not cal["fomc"]:
            raise RuntimeError("no FOMC dates parsed")
    except Exception as e:  # noqa: BLE001
        cal["notes"].append(f"fomc: {type(e).__name__}: {e}")
        cal["fomc"] = CONFIG.get("calendar_fallback", {}).get("fomc", [])

    def bls_dates(url: str) -> list[str]:
        r = http_get(url, timeout=20, retries=1, headers={**HEADERS, "Referer": "https://www.bls.gov/",
                                                           "User-Agent": "market-dashboard/1.0 (+https://github.com/wschmitt77/market-dashboard)"})
        tables = pd.read_html(io.StringIO(r.text))
        out = []
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            rc = next((c for c, lc in zip(t.columns, cols) if "release date" in lc), None)
            if rc is None:
                continue
            for v in t[rc].astype(str):
                v2 = re.sub(r"\.", "", v).strip()
                d = pd.to_datetime(v2, errors="coerce")
                if pd.notna(d):
                    out.append(d.strftime("%Y-%m-%d"))
        if not out:
            raise RuntimeError("no release dates found")
        return sorted(set(out))

    for key, url in (("cpi", "https://www.bls.gov/schedule/news_release/cpi.htm"),
                     ("nfp", "https://www.bls.gov/schedule/news_release/empsit.htm")):
        try:
            cal[key] = bls_dates(url)
        except Exception as e:  # noqa: BLE001
            cal["notes"].append(f"{key}: {type(e).__name__}: {e}")
            cal[key] = CONFIG.get("calendar_fallback", {}).get(key, [])
        time.sleep(1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal, indent=1))
    status = "ok" if (cal["fomc"] and (cal["cpi"] or cal["nfp"])) else ("partial" if cal["fomc"] or cal["cpi"] or cal["nfp"] else "failed")
    record(name, status, fomc=len(cal["fomc"]), cpi=len(cal["cpi"]), nfp=len(cal["nfp"]),
           error="; ".join(cal["notes"]) if cal["notes"] else None)


# ----------------------------------------------------------------------------- main
def main(argv: list[str]) -> int:
    only = set(a for a in argv[1:] if not a.startswith("-"))
    RAW.mkdir(parents=True, exist_ok=True)

    tickers = list(dict.fromkeys(CONFIG["core_tickers"] + list(CONFIG["sector_etfs"]) + CONFIG["ratio_tickers"]
                                 + CONFIG.get("yahoo_indices", [])))
    try:
        LOG_PATH.write_text("")
    except Exception:  # noqa: BLE001
        pass
    log(f"fetch start; {len(tickers)} tickers, {len(CONFIG['fred_series'])} FRED series")
    if not only or "prices" in only:
        bulk = guarded("prices/bulk", yf_bulk, tickers, "max", budget=300) or {}
        log(f"yfinance bulk: {len(bulk)}/{len(tickers)} tickers")
        for t in tickers:
            guarded(f"prices/{t}", fetch_price, t, bulk.get(t), budget=45)
    if not only or "fred" in only:
        log("FRED via " + ("API key" if FRED_API_KEY else "fredgraph.csv (no FRED_API_KEY secret set)"))
        for s in CONFIG["fred_series"]:
            guarded(f"fred/{s}", fetch_fred, s, budget=40)
        guarded("fred/fallbacks", fetch_fred_fallbacks, budget=300)
    if not only or "cboe" in only:
        for idx in CONFIG["cboe_indices"]:
            guarded(f"cboe/{idx}", fetch_cboe, idx, budget=45)
    if not only or "misc" in only:
        guarded("misc/qqq_holdings", fetch_qqq_holdings, budget=60)
    if not only or "breadth" in only:
        guarded("breadth/spx", fetch_breadth, "spx", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ("Symbol",), budget=420)
        guarded("breadth/ndx", fetch_breadth, "ndx", "https://en.wikipedia.org/wiki/Nasdaq-100", ("Ticker", "Symbol"), budget=240)
    if not only or "misc" in only:
        guarded("misc/aaa_gas", fetch_aaa_gas, budget=60)
        guarded("misc/gdpnow", fetch_gdpnow, budget=60)
        guarded("misc/cleveland_nowcast", fetch_cleveland_nowcast, budget=60)
        guarded("misc/ism", fetch_ism, budget=90)
        guarded("misc/cboe_putcall", fetch_cboe_putcall, budget=60)
        guarded("misc/sp500_pe", fetch_multpl, "s-p-500-pe-ratio", "sp500_pe", budget=60)
        guarded("misc/shiller_cape", fetch_multpl, "shiller-pe", "shiller_cape", budget=60)
        guarded("misc/calendar", fetch_calendar, budget=120)

    save_manifest()
    ok = MANIFEST["summary"]["ok"]
    log(f"done: {ok}/{len(MANIFEST['sources'])} sources ok")
    sys.stdout.flush()
    os._exit(0)  # don't wait on abandoned (timed-out) network threads


if __name__ == "__main__":
    sys.exit(main(sys.argv))
