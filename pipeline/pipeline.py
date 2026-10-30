#!/usr/bin/env python3
"""
Metric engine for the Market Signals Dashboard.

  compute stage (every morning, first):
      python pipeline/pipeline.py --repo <repo root> --out <dir> [--history history.json] [--as-of YYYY-MM-DD]
      -> writes <dir>/summary.json, metrics.json, series.json, sectors.json, backtest.json,
         backtest_curve.json, events.json, sources.json, signals.json, brief.md

  finalize stage (after Claude has written <dir>/insights_today.json):
      python pipeline/pipeline.py --repo <repo root> --out <dir> --stage finalize --insights <dir>/insights_today.json [--history history.json]
      -> writes <dir>/insights.json, insight_log.json, scorecard.json, history.json

Everything is computed from the raw files the GitHub feed commits under data/raw/.
Nothing here is investment advice: thresholds are starting points, in config.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE.parent / "config.json").read_text())
TZ = ZoneInfo("America/Chicago")
SOURCE_PRICES = "Yahoo Finance / Stooq daily bars via the GitHub data feed"
SOURCE_FRED = "FRED (St. Louis Fed) via the GitHub data feed"
SOURCE_CBOE = "Cboe index history via the GitHub data feed"


# ============================================================================ loading
class Data:
    def __init__(self, repo: Path):
        self.repo = repo
        self.raw = repo / "data" / "raw"
        self.manifest = self._json(repo / "data" / "manifest.json") or {"sources": {}}
        self._cache: dict[str, pd.DataFrame | None] = {}

    @staticmethod
    def _json(p: Path):
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            return None

    def _csv(self, p: Path, dedupe: bool = True) -> pd.DataFrame | None:
        key = f"{p}|{dedupe}"
        if key not in self._cache:
            try:
                df = pd.read_csv(p)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df = df.dropna(subset=["date"]).sort_values("date", kind="stable")
                    if dedupe:
                        df = df.drop_duplicates("date", keep="last")
                    df = df.set_index("date")
                self._cache[key] = df
            except Exception:  # noqa: BLE001
                self._cache[key] = None
        return self._cache[key]

    def _incomplete_session_date(self) -> str | None:
        """The trading date whose bar may be partial: the feed's run date when it ran before the 4 PM ET close."""
        try:
            run_at = pd.Timestamp(self.manifest.get("run_at"))
            if run_at.tzinfo is None:
                run_at = run_at.tz_localize("UTC")
            et = run_at.tz_convert("America/New_York")
            if et.hour < 16 or (et.hour == 16 and et.minute < 15):
                return et.strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001
            return None
        return None

    def prices(self, sym: str) -> pd.DataFrame | None:
        df = self._csv(self.raw / "prices" / f"{sym.replace('^', '_')}.csv")
        if df is None:
            return None
        for c in ("open", "high", "low", "close", "adj_close", "volume"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        bad = self._incomplete_session_date()
        if bad is not None and len(df) and pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d") == bad:
            df = df.iloc[:-1]
        return df.dropna(subset=["close"])

    def close(self, sym: str, adj: bool = False) -> pd.Series | None:
        df = self.prices(sym)
        if df is None:
            return None
        col = "adj_close" if adj and "adj_close" in df.columns else "close"
        return df[col].dropna()

    def fred(self, series: str) -> pd.Series | None:
        df = self._csv(self.raw / "fred" / f"{series}.csv")
        if df is None or "value" not in df.columns:
            return None
        return pd.to_numeric(df["value"], errors="coerce").dropna()

    def cboe(self, idx: str) -> pd.Series | None:
        df = self._csv(self.raw / "cboe" / f"{idx}.csv")
        if df is None or "close" not in df.columns:
            # fall back to Yahoo index copy
            alt = self.close("^" + idx) if idx in ("VIX", "VXN") else None
            return alt
        return pd.to_numeric(df["close"], errors="coerce").dropna()

    def wide(self, label: str) -> pd.DataFrame | None:
        df = self._csv(self.raw / "breadth" / f"{label}_closes.csv")
        if df is None:
            return None
        df = df.apply(pd.to_numeric, errors="coerce")
        bad = self._incomplete_session_date()
        if bad is not None and len(df) and pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d") == bad:
            df = df.iloc[:-1]
        return df

    def misc(self, name: str) -> pd.DataFrame | None:
        return self._csv(self.raw / "misc" / f"{name}.csv", dedupe=name not in ("cleveland_nowcast", "qqq_holdings"))

    def misc_json(self, name: str):
        return self._json(self.raw / "misc" / f"{name}.json")


# ============================================================================ math helpers
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(ad != 0, 100.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def pct_b(s: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    m = sma(s, n)
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return (s - (m - k * sd)) / (2 * k * sd)


def drawdown(s: pd.Series, window: int | None = 252) -> pd.Series:
    peak = s.rolling(window, min_periods=1).max() if window else s.cummax()
    return s / peak - 1


def realized_vol(s: pd.Series, n: int) -> pd.Series:
    r = np.log(s).diff()
    return r.rolling(n, min_periods=n).std(ddof=0) * math.sqrt(252) * 100


def pctile_of_last(s: pd.Series, window: int = 252) -> float | None:
    s = s.dropna()
    if len(s) < 30:
        return None
    w = s.iloc[-window:]
    return float((w < w.iloc[-1]).mean() * 100)


def zscore_last(s: pd.Series, window: int = 252) -> float | None:
    s = s.dropna()
    if len(s) < 30:
        return None
    w = s.iloc[-window:]
    sd = w.std(ddof=0)
    return float((w.iloc[-1] - w.mean()) / sd) if sd and not math.isnan(sd) else None


def last(s: pd.Series | None, lag: int = 0) -> float | None:
    if s is None:
        return None
    s = s.dropna()
    if len(s) <= lag:
        return None
    v = s.iloc[-1 - lag]
    return None if pd.isna(v) else float(v)


def last_date(s: pd.Series | pd.DataFrame | None) -> str | None:
    if s is None or len(s) == 0:
        return None
    return pd.Timestamp(s.dropna().index[-1]).strftime("%Y-%m-%d") if isinstance(s, pd.Series) else pd.Timestamp(s.index[-1]).strftime("%Y-%m-%d")


def change(s: pd.Series | None, lag: int, pct: bool = False) -> float | None:
    a, b = last(s), last(s, lag)
    if a is None or b is None:
        return None
    return (a / b - 1) * 100 if pct else a - b


def scale(x: float | None, x0: float, x100: float) -> float | None:
    """Linear 0..100 clamp: x0 -> 0, x100 -> 100 (either direction)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    t = (x - x0) / (x100 - x0)
    return float(max(0.0, min(1.0, t)) * 100)


def f(v, spec: str = "{:.1f}", none: str = "n/a") -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return none
    return spec.format(v)


def pct(v, signed=True, d=1):
    return f(v, ("{:+." + str(d) + "f}%") if signed else ("{:." + str(d) + "f}%"))


def bp(v):
    return f(v, "{:+.0f} bp")


def consecutive_down(s: pd.Series) -> int:
    d = np.sign(s.diff().dropna().values)
    n = 0
    for x in d[::-1]:
        if x < 0:
            n += 1
        else:
            break
    return n


def days_since(cond: pd.Series) -> int | None:
    idx = np.where(cond.fillna(False).values)[0]
    if len(idx) == 0:
        return None
    return int(len(cond) - 1 - idx[-1])


# ============================================================================ composites
def dip_components(dd=None, rsi14=None, rsi2=None, pb=None, vix=None, vix_pct=None, vix_ts=None,
                   breadth200=None, pc10=None, hy_chg30=None):
    comps = {
        "drawdown": (scale(dd, -3, -20), 25),
        "rsi14": (scale(rsi14, 50, 25), 15),
        "rsi2": (scale(rsi2, 30, 5), 10),
        "pct_b": (scale(pb, 0.5, -0.2), 10),
        "vix": (None if vix is None and vix_pct is None else max(x for x in (scale(vix, 18, 35), scale(vix_pct, 50, 95)) if x is not None), 15),
        "vix_term": (scale(vix_ts, 0.9, 1.05), 10),
        "breadth": (scale(breadth200, 50, 20), 10),
        "put_call": (scale(pc10, 0.7, 1.0), 5),
    }
    penalty = 0.0
    if hy_chg30 is not None and hy_chg30 > 50:
        penalty = (scale(hy_chg30, 50, 150) or 0) * 0.15
    return comps, penalty


def heat_components(dist200=None, rsi14=None, pb=None, vix=None, pc10=None, breadth200=None, ad_div=None,
                    qqq_ivv_z=None, days_since_pullback=None):
    comps = {
        "dist_200d": (scale(dist200, 5, 15), 25),
        "rsi14": (scale(rsi14, 60, 80), 15),
        "pct_b": (scale(pb, 0.8, 1.1), 10),
        "vix": (scale(vix, 18, 12), 15),
        "put_call": (scale(pc10, 0.7, 0.5), 5),
        "breadth": (scale(breadth200, 65, 85), 10),
        "ad_divergence": (None if ad_div is None else (100.0 if ad_div else 0.0), 10),
        "qqq_vs_ivv_z": (scale(qqq_ivv_z, 1, 2.5), 5),
        "complacency": (scale(days_since_pullback, 60, 180), 5),
    }
    return comps, 0.0


def composite(comps: dict, penalty: float = 0.0):
    avail = {k: v for k, v in comps.items() if v[0] is not None}
    if not avail:
        return None, [], list(comps)
    wsum = sum(w for _, w in avail.values())
    score = sum(v * w for v, w in avail.values()) / wsum - penalty
    score = max(0.0, min(100.0, score))
    parts = [{"component": k, "value": round(v, 1), "weight": w, "contribution": round(v * w / wsum, 1)} for k, (v, w) in avail.items()]
    return round(score, 1), parts, [k for k in comps if k not in avail]


def dip_series(close: pd.Series, high: pd.Series | None, low: pd.Series | None, vix: pd.Series | None, vix3m: pd.Series | None) -> pd.Series:
    """Historical dip score from price + volatility components only (for backtests)."""
    dd = drawdown(close) * 100
    r14, r2, pb = rsi(close, 14), rsi(close, 2), pct_b(close)
    df = pd.DataFrame({"dd": dd, "r14": r14, "r2": r2, "pb": pb})
    if vix is not None:
        v = vix.reindex(df.index).ffill()
        df["vix"] = v
        df["vix_pct"] = v.rolling(252, min_periods=60).apply(lambda w: (w[:-1] < w[-1]).mean() * 100, raw=True)
    if vix is not None and vix3m is not None:
        df["ts"] = (vix / vix3m.reindex(vix.index)).reindex(df.index).ffill()
    out = []
    cols = df.columns
    for row in df.itertuples(index=False):
        d = dict(zip(cols, row))
        comps, _ = dip_components(dd=d["dd"], rsi14=d["r14"], rsi2=d["r2"], pb=d["pb"], vix=d.get("vix"), vix_pct=d.get("vix_pct"), vix_ts=d.get("ts"))
        s, _, _ = composite(comps)
        out.append(s)
    return pd.Series(out, index=df.index, dtype="float64")


def heat_series(close: pd.Series, vix: pd.Series | None) -> pd.Series:
    dist = (close / sma(close, 200) - 1) * 100
    r14, pb = rsi(close, 14), pct_b(close)
    dd = drawdown(close, None)
    pull = (dd <= -0.05)
    # days since the last 5% pullback from the running high
    ds = pd.Series(np.nan, index=close.index)
    lastp = None
    for i, (ts, flag) in enumerate(pull.items()):
        if flag:
            lastp = i
        ds.iloc[i] = (i - lastp) if lastp is not None else np.nan
    df = pd.DataFrame({"dist": dist, "r14": r14, "pb": pb, "ds": ds})
    if vix is not None:
        df["vix"] = vix.reindex(df.index).ffill()
    out = []
    cols = df.columns
    for row in df.itertuples(index=False):
        d = dict(zip(cols, row))
        comps, _ = heat_components(dist200=d["dist"], rsi14=d["r14"], pb=d["pb"], vix=d.get("vix"), days_since_pullback=d["ds"])
        s, _, _ = composite(comps)
        out.append(s)
    return pd.Series(out, index=df.index, dtype="float64")


# ============================================================================ the engine
class Engine:
    def __init__(self, data: Data, as_of: date):
        self.d = data
        self.as_of = as_of
        self.metrics: list[dict] = []
        self.notes: list[str] = []
        self.n = 0

    # ---- metric row helper
    def add(self, layer_key, layer, name, *, value=None, fmt=None, value2=None, fmt2=None, col1="QQQ", col2="IVV",
            unit="", signal="neutral", state="neutral", reading="", threshold="", detail="", pctile=None, z=None,
            chg_1d=None, chg_5d=None, as_of=None, status=None, source=SOURCE_PRICES, id=None):
        self.n += 1
        status = status or ("live" if value is not None else "unavailable")
        if value is None:
            signal, state = "n/a", "neutral"
            reading = "Source unavailable in today's feed."
            detail = ""
        row = {"n": self.n, "id": id or f"m{self.n}", "layer_key": layer_key, "layer": layer, "name": name,
               "value": None if value is None else round(float(value), 4), "fmt": fmt or f(value),
               "value2": None if value2 is None else round(float(value2), 4), "fmt2": fmt2 or ("" if value2 is None else f(value2)),
               "col1": col1, "col2": col2, "unit": unit, "signal": signal, "state": state, "reading": reading,
               "display": (f"{col1} {fmt or f(value)}" if col1 else (fmt or f(value))) + (f" · {col2} {fmt2}" if (fmt2 or (value2 is not None)) and col2 else (f" · {fmt2}" if fmt2 and not col2 else "")),
               "threshold": threshold, "detail": detail,
               "pctile_1y": None if pctile is None else int(round(float(pctile))),
               "z_1y": None if z is None else round(float(z), 2),
               "chg_1d": None if chg_1d is None else round(float(chg_1d), 3),
               "chg_5d": None if chg_5d is None else round(float(chg_5d), 3),
               "as_of": as_of or "", "status": status, "source": source}
        self.metrics.append(row)
        return row

    # ---------------------------------------------------------------- run
    def run(self) -> dict:
        d = self.d
        out: dict = {}
        qqq_df, ivv_df, tq_df = d.prices("QQQ"), d.prices("IVV"), d.prices("TQQQ")
        if qqq_df is None or ivv_df is None:
            raise SystemExit("QQQ/IVV price files missing: nothing to compute")
        qqq, ivv = qqq_df["close"], ivv_df["close"]
        tqqq = tq_df["close"] if tq_df is not None else None
        data_through = last_date(qqq)
        self.data_through = data_through

        vix = d.cboe("VIX")
        vix3m = d.cboe("VIX3M")
        vxn = d.cboe("VXN")
        skew = d.cboe("SKEW")
        move = d.close("^MOVE")
        tbill = d.fred("DTB3")
        hy = d.fred("BAMLH0A0HYM2")
        claims = d.fred("ICSA")
        t10y3m, t10y2y = d.fred("T10Y3M"), d.fred("T10Y2Y")
        real10 = d.fred("DFII10")

        # ---------------- A. trend & regime -----------------------------------------
        L, LK = "Trend & regime", "trend"
        trend = {}
        for sym, s in (("QQQ", qqq), ("IVV", ivv)):
            s50, s200 = sma(s, 50), sma(s, 200)
            dist = (s / s200 - 1) * 100
            slope = (s200 / s200.shift(20) - 1) * 100
            cross = np.sign(s50 - s200)
            cross_chg = cross.ne(cross.shift())
            cross_dates = cross.index[cross_chg.fillna(False) & cross.notna()]
            trend[sym] = {"dist": last(dist), "slope": last(slope), "above": (last(s) or 0) > (last(s200) or 0),
                          "golden": (last(s50) or 0) > (last(s200) or 0),
                          "cross_date": pd.Timestamp(cross_dates[-1]).strftime("%Y-%m-%d") if len(cross_dates) else "",
                          "dist_series": dist, "s50": s50, "s200": s200}
        tq = trend["QQQ"]
        sig = "bearish" if not tq["above"] else ("extended" if (tq["dist"] or 0) > 12 else "bullish")
        state = "bad" if not tq["above"] else ("warn" if sig == "extended" else "ok")
        self.add(LK, L, "Distance from 200-day SMA", id="dist_200d",
                 value=tq["dist"], fmt=pct(tq["dist"]), value2=trend["IVV"]["dist"], fmt2=pct(trend["IVV"]["dist"]), unit="%",
                 signal=sig, state=state,
                 reading=("Below the 200-day: regime filter is off; leveraged dip-buying has historically done its worst damage here."
                          if not tq["above"] else ("More than 12% above the 200-day: extended; trimming zone for TQQQ." if sig == "extended"
                                                   else "Above the 200-day with the average %s: trend regime intact." % ("rising" if (tq["slope"] or 0) > 0 else "flattening"))),
                 threshold="Below 200d = regime off; > +12% = extended",
                 detail=f"QQQ 200d slope {pct(tq['slope'])} per 20 sessions; 50/200 {'golden' if tq['golden'] else 'death'} cross since {tq['cross_date'] or 'n/a'}. IVV slope {pct(trend['IVV']['slope'])}.",
                 pctile=pctile_of_last(tq["dist_series"]), z=zscore_last(tq["dist_series"]), chg_1d=change(tq["dist_series"], 1), chg_5d=change(tq["dist_series"], 5), as_of=data_through)

        # 2. 12-1 momentum vs T-bill
        mom = {}
        for sym, s in (("QQQ", qqq), ("IVV", ivv)):
            m = (s.shift(21) / s.shift(252) - 1) * 100
            mom[sym] = (last(m), m)
        tb = last(tbill)
        hurdle = (tb or 0) * 11 / 12
        v = mom["QQQ"][0]
        self.add(LK, L, "12-1 month momentum vs T-bill", id="momentum_12_1",
                 value=v, fmt=pct(v), value2=mom["IVV"][0], fmt2=pct(mom["IVV"][0]), unit="%",
                 signal="risk-on" if v is not None and v > hurdle else "risk-off", state="ok" if v is not None and v > hurdle else "bad",
                 reading=("Trailing 12-month return (skipping the last month) beats the T-bill hurdle: absolute momentum is positive." if v is not None and v > hurdle
                          else "Trailing 12-1 return is below the T-bill hurdle: absolute momentum says risk-off."),
                 threshold=f"Hurdle = 3m T-bill ({f(tb, '{:.2f}')}%) prorated", detail=f"Hurdle {pct(hurdle)}; QQQ 12-1 {pct(v)}, IVV {pct(mom['IVV'][0])}.",
                 pctile=pctile_of_last(mom["QQQ"][1]), z=zscore_last(mom["QQQ"][1]), chg_5d=change(mom["QQQ"][1], 5), as_of=data_through)

        # 3. Drawdown from 52-week high
        dd = {sym: drawdown(s) * 100 for sym, s in (("QQQ", qqq), ("IVV", ivv))}
        ddq = last(dd["QQQ"])
        tier = sum(1 for t in (-5, -10, -15, -20) if ddq is not None and ddq <= t)
        hi52 = qqq.iloc[-252:].max()
        hi52_date = pd.Timestamp(qqq.iloc[-252:].idxmax()).strftime("%Y-%m-%d")
        self.add(LK, L, "Drawdown from 52-week high", id="drawdown_52w",
                 value=ddq, fmt=pct(ddq), value2=last(dd["IVV"]), fmt2=pct(last(dd["IVV"])), unit="%",
                 signal=["at highs", "pullback", "correction", "deep correction", "bear zone"][tier] if ddq is not None else "n/a",
                 state="ok" if tier == 0 else ("warn" if tier <= 2 else "bad"),
                 reading=f"Tier {tier} of 4 on the dip ladder (−5/−10/−15/−20%)." + (" No tranche trigger from drawdown alone." if tier == 0 else f" Drawdown has crossed {tier} tranche level(s)."),
                 threshold="Tranche levels −5 / −10 / −15 / −20%",
                 detail=f"QQQ 52-week high {f(hi52, '{:.2f}')} on {hi52_date}.", pctile=pctile_of_last(dd["QQQ"]), z=zscore_last(dd["QQQ"]),
                 chg_1d=change(dd["QQQ"], 1), chg_5d=change(dd["QQQ"], 5), as_of=data_through)

        # 4. Drawdown percentile vs own history
        ddall = drawdown(qqq, None) * 100
        cur = last(ddall)
        deeper_share = float((ddall.dropna() <= cur).mean() * 100) if cur is not None else None
        ddall_i = drawdown(ivv, None) * 100
        cur_i = last(ddall_i)
        deeper_i = float((ddall_i.dropna() <= cur_i).mean() * 100) if cur_i is not None else None
        since_year = pd.Timestamp(qqq.index[0]).year
        self.add(LK, L, "How rare is today's dip (share of sessions with a deeper drawdown)", id="drawdown_percentile",
                 value=deeper_share, fmt=pct(deeper_share, False, 0), value2=deeper_i, fmt2=pct(deeper_i, False, 0), unit="%",
                 signal="rare dip" if deeper_share is not None and deeper_share < 15 else ("ordinary" if deeper_share is not None and deeper_share < 60 else "near highs"),
                 state="warn" if deeper_share is not None and deeper_share < 15 else "neutral",
                 reading=((f"Only {f(deeper_share, '{:.0f}')}% of sessions since {since_year} had a drawdown from the all-time high this deep or deeper: a historically rare dip."
                           if deeper_share < 15 else f"Drawdowns at least this deep occurred on {f(deeper_share, '{:.0f}')}% of sessions since {since_year}: " + ("an ordinary dip." if deeper_share < 60 else "the index is near its highs."))
                          if deeper_share is not None else ""),
                 threshold="< 15th percentile = historically rare dip", detail=f"QQQ drawdown from all-time high {pct(cur)}; IVV {pct(cur_i)}.", as_of=data_through)

        # 5. Distribution days & up/down volume
        dist_days = {}
        for sym, df in (("QQQ", qqq_df), ("IVV", ivv_df)):
            c, vol = df["close"], df["volume"]
            down = (c < c.shift() * (1 - 0.002)) & (vol > vol.shift())
            dist_days[sym] = int(down.iloc[-25:].sum()) if vol.notna().iloc[-25:].all() else None
            upv = vol.where(c > c.shift(), 0).rolling(50).sum()
            dnv = vol.where(c < c.shift(), 0).rolling(50).sum()
            dist_days[sym + "_ud"] = last(upv / dnv.replace(0, np.nan))
        v = dist_days["QQQ"]
        self.add(LK, L, "Distribution days (last 25 sessions)", id="distribution_days",
                 value=v, fmt=f(v, "{:.0f}"), value2=dist_days["IVV"], fmt2=f(dist_days["IVV"], "{:.0f}"), unit="days",
                 signal="selling pressure" if v is not None and v >= 5 else "normal", state="bad" if v is not None and v >= 6 else ("warn" if v is not None and v >= 5 else "ok"),
                 reading=("Five or more down days on rising volume in 25 sessions: institutions are distributing." if v is not None and v >= 5 else "Institutional selling pressure looks contained."),
                 threshold="≥ 5–6 = market under pressure", detail=f"50-day up/down volume ratio QQQ {f(dist_days['QQQ_ud'], '{:.2f}')}, IVV {f(dist_days['IVV_ud'], '{:.2f}')}.", as_of=data_through)

        # ---------------- B. mean reversion -----------------------------------------
        L, LK = "Mean reversion", "meanrev"
        r14 = {sym: rsi(s, 14) for sym, s in (("QQQ", qqq), ("IVV", ivv))}
        r2 = {sym: rsi(s, 2) for sym, s in (("QQQ", qqq), ("IVV", ivv))}
        v = last(r14["QQQ"])
        self.add(LK, L, "RSI(14)", id="rsi14", value=v, fmt=f(v), value2=last(r14["IVV"]), fmt2=f(last(r14["IVV"])),
                 signal="oversold" if v is not None and v < 30 else ("overbought" if v is not None and v > 70 else "neutral"),
                 state="warn" if v is not None and (v < 30 or v > 70) else "ok",
                 reading="Oversold territory: momentum is washed out." if v is not None and v < 30 else ("Overbought: momentum is stretched." if v is not None and v > 70 else "Momentum in the neutral band."),
                 threshold="< 30 oversold, > 70 overbought", pctile=pctile_of_last(r14["QQQ"]), chg_1d=change(r14["QQQ"], 1), chg_5d=change(r14["QQQ"], 5), as_of=data_through)
        v2 = last(r2["QQQ"])
        cdq, cdi = consecutive_down(qqq), consecutive_down(ivv)
        self.add(LK, L, "RSI(2) and consecutive down days", id="rsi2", value=v2, fmt=f(v2), value2=last(r2["IVV"]), fmt2=f(last(r2["IVV"])),
                 signal="dip trigger" if v2 is not None and v2 < 10 and tq["above"] else ("washed out" if v2 is not None and v2 < 10 else ("hot" if v2 is not None and v2 > 90 else "neutral")),
                 state="ok" if v2 is not None and v2 < 10 and tq["above"] else ("warn" if v2 is not None and (v2 < 10 or v2 > 90) else "neutral"),
                 reading=("RSI(2) under 10 while above the 200-day: the classic short-term dip entry." if v2 is not None and v2 < 10 and tq["above"]
                          else ("RSI(2) under 10 but below the 200-day: washed out inside a downtrend, size small." if v2 is not None and v2 < 10
                                else ("RSI(2) above 90: short-term exhaustion." if v2 is not None and v2 > 90 else "No short-term extreme."))),
                 threshold="< 10 above 200d = dip entry; > 90 = exit", detail=f"Consecutive down closes: QQQ {cdq}, IVV {cdi}.", chg_1d=change(r2["QQQ"], 1), as_of=data_through)
        pbq, pbi = pct_b(qqq), pct_b(ivv)
        v = last(pbq)
        self.add(LK, L, "Bollinger %B (20, 2)", id="pct_b", value=v, fmt=f(v, "{:.2f}"), value2=last(pbi), fmt2=f(last(pbi), "{:.2f}"),
                 signal="oversold" if v is not None and v < 0 else ("stretched" if v is not None and v > 1 else "neutral"),
                 state="warn" if v is not None and (v < 0 or v > 1) else "ok",
                 reading="Closing below the lower band." if v is not None and v < 0 else ("Closing above the upper band: stretched." if v is not None and v > 1 else "Inside the bands."),
                 threshold="< 0 oversold, > 1 stretched", pctile=pctile_of_last(pbq), chg_1d=change(pbq, 1), as_of=data_through)
        stq = (qqq - sma(qqq, 20)) / atr(qqq_df, 14)
        sti = (ivv - sma(ivv, 20)) / atr(ivv_df, 14)
        v = last(stq)
        self.add(LK, L, "Stretch from 20-day mean in ATR units", id="atr_stretch", value=v, fmt=f(v, "{:+.2f} ATR"), value2=last(sti), fmt2=f(last(sti), "{:+.2f} ATR"),
                 signal="extreme" if v is not None and abs(v) >= 3 else ("notable" if v is not None and abs(v) >= 2 else "neutral"),
                 state="warn" if v is not None and abs(v) >= 2 else "ok",
                 reading=f"{f(abs(v) if v is not None else None, '{:.1f}')} ATRs {'below' if (v or 0) < 0 else 'above'} the 20-day mean." + (" Beyond ±2 is notable, ±3 extreme." if v is not None and abs(v) >= 2 else ""),
                 threshold="±2 notable, ±3 extreme", pctile=pctile_of_last(stq), chg_1d=change(stq, 1), as_of=data_through)
        macd_h = {}
        for sym, s in (("QQQ", qqq), ("IVV", ivv)):
            line = ema(s, 12) - ema(s, 26)
            hist = line - ema(line, 9)
            macd_h[sym] = hist
        h = macd_h["QQQ"]
        v = last(h)
        rising = (v or 0) > (last(h, 1) or 0)
        zero_cross = days_since(np.sign(h).ne(np.sign(h.shift())) & h.notna())
        self.add(LK, L, "MACD histogram (12, 26, 9)", id="macd_hist", value=v, fmt=f(v, "{:+.2f}"), value2=last(macd_h["IVV"]), fmt2=f(last(macd_h['IVV']), "{:+.2f}"),
                 signal="turning up" if v is not None and v < 0 and rising else ("turning down" if v is not None and v > 0 and not rising else ("positive" if (v or 0) > 0 else "negative")),
                 state="ok" if (v is not None and (v > 0 or rising)) else "warn",
                 reading=("Histogram rising from below zero: the bounce is being confirmed." if v is not None and v < 0 and rising
                          else ("Histogram falling from above zero: momentum rolling over." if v is not None and v > 0 and not rising
                                else ("Momentum positive and rising." if rising else "Momentum negative and falling."))),
                 threshold="Up-turn from < 0 confirms a bounce", detail=f"Last zero-line cross {zero_cross} sessions ago." if zero_cross is not None else "", chg_1d=change(h, 1), as_of=data_through)

        # ---------------- C. volatility -----------------------------------------
        L, LK = "Volatility & TQQQ", "vol"
        rv20, rv60 = realized_vol(qqq, 20), realized_vol(qqq, 60)
        rv20i = realized_vol(ivv, 20)
        v, v60 = last(rv20), last(rv60)
        self.add(LK, L, "Realized volatility (20d / 60d, annualized)", id="realized_vol", value=v, fmt=pct(v, False), value2=v60, fmt2=pct(v60, False), col1="20d", col2="60d", unit="%",
                 signal="high" if v is not None and v > 30 else ("low" if v is not None and v < 15 else "normal"), state="warn" if v is not None and v > 30 else "ok",
                 reading=f"QQQ 20-day realized vol at the {f(pctile_of_last(rv20), '{:.0f}')}th percentile of the past year.",
                 threshold="> 30% high (decay accelerates), < 15% calm", pctile=pctile_of_last(rv20), z=zscore_last(rv20), chg_1d=change(rv20, 1), chg_5d=change(rv20, 5), as_of=data_through)
        drag20 = 3 * (v / 100) ** 2 * 100 if v is not None else None
        drag60 = 3 * (v60 / 100) ** 2 * 100 if v60 is not None else None
        self.add(LK, L, "Estimated TQQQ volatility drag (3·σ², annualized)", id="vol_drag", value=drag20, fmt=pct(drag20, False), value2=drag60, fmt2=pct(drag60, False), col1="from 20d vol", col2="from 60d vol", unit="%/yr",
                 signal="severe" if drag20 is not None and drag20 > 30 else ("elevated" if drag20 is not None and drag20 > 15 else "moderate"),
                 state="bad" if drag20 is not None and drag20 > 30 else ("warn" if drag20 is not None and drag20 > 15 else "ok"),
                 reading=f"At current realized vol a daily-rebalanced 3× fund bleeds roughly {f(drag20, '{:.0f}')}% a year versus 3× buy-and-hold." if drag20 is not None else "",
                 threshold="≈3×σ²: 20% vol → 12%/yr, 40% vol → 48%/yr", as_of=data_through)
        gap63 = gap126 = None
        gap_series = None
        if tqqq is not None:
            tqa, qqa = d.close("TQQQ", adj=True), d.close("QQQ", adj=True)
            both = pd.concat([tqa, qqa], axis=1, keys=["t", "q"], sort=True).dropna()
            gap_series = ((both["t"] / both["t"].shift(63) - 1) - 3 * (both["q"] / both["q"].shift(63) - 1)) * 100
            g126 = ((both["t"] / both["t"].shift(126) - 1) - 3 * (both["q"] / both["q"].shift(126) - 1)) * 100
            gap63, gap126 = last(gap_series), last(g126)
        self.add(LK, L, "TQQQ return minus 3× QQQ return (3m / 6m)", id="tqqq_gap", value=gap63, fmt=f(gap63, "{:+.1f} pp"), value2=gap126, fmt2=f(gap126, "{:+.1f} pp"), col1="3 months", col2="6 months", unit="pp",
                 signal="decay" if gap63 is not None and gap63 < -3 else ("compounding tailwind" if gap63 is not None and gap63 > 3 else "small"),
                 state="warn" if gap63 is not None and gap63 < -5 else "ok",
                 reading=("TQQQ has lagged three times QQQ's return over the last quarter: realized decay." if gap63 is not None and gap63 < -3
                          else ("Trending market: daily compounding has added to TQQQ beyond 3× (positive convexity)." if gap63 is not None and gap63 > 3 else "Realized gap versus 3× QQQ is small.")),
                 threshold="Below −3 pp = decay regime", as_of=data_through if tqqq is not None else None, status=None if tqqq is not None else "unavailable")
        vv, vxv = last(vix), last(vxn)
        self.add(LK, L, "VIX and VXN", id="vix_vxn", value=vv, fmt=f(vv), value2=vxv, fmt2=f(vxv), col1="VIX", col2="VXN",
                 signal="panic" if vv is not None and vv >= 30 else ("elevated" if vv is not None and vv >= 20 else ("complacent" if vv is not None and vv < 14 else "normal")),
                 state="warn" if vv is not None and (vv >= 30 or vv < 14) else "ok",
                 reading=("VIX at or above 30: historically a dip-buy zone, but size for wide swings." if vv is not None and vv >= 30
                          else ("VIX under 14: complacency; options are cheap and TQQQ trims have historically paid here." if vv is not None and vv < 14 else "Implied volatility in its normal range.")),
                 threshold="≥ 30 dip zone, < 14 complacency", pctile=pctile_of_last(vix) if vix is not None else None, z=zscore_last(vix) if vix is not None else None,
                 chg_1d=change(vix, 1), chg_5d=change(vix, 5), as_of=last_date(vix), source=SOURCE_CBOE)
        ts = None
        ts_series = None
        if vix is not None and vix3m is not None:
            ts_series = (vix / vix3m.reindex(vix.index)).dropna()
            ts = last(ts_series)
        self.add(LK, L, "VIX / VIX3M term structure", id="vix_term", value=ts, fmt=f(ts, "{:.3f}"), col1="ratio", col2="",
                 signal="backwardation" if ts is not None and ts > 1 else ("flat" if ts is not None and ts > 0.95 else "contango"),
                 state="warn" if ts is not None and ts > 1 else "ok",
                 reading=("Front-month fear above 3-month: backwardation, a capitulation signature." if ts is not None and ts > 1
                          else ("Term structure flattening: stress building." if ts is not None and ts > 0.95 else "Normal contango: no panic in the options market.")),
                 threshold="> 1.0 = backwardation (capitulation)", pctile=pctile_of_last(ts_series) if ts_series is not None else None, chg_1d=change(ts_series, 1), as_of=last_date(ts_series), source=SOURCE_CBOE)
        vrp = vrp_q = None
        if vv is not None and last(rv20i) is not None:
            vrp = vv - last(rv20i)
        if vxv is not None and v is not None:
            vrp_q = vxv - v
        self.add(LK, L, "Variance risk premium (implied − realized)", id="vrp", value=vrp, fmt=f(vrp, "{:+.1f} pts"), value2=vrp_q, fmt2=f(vrp_q, "{:+.1f} pts"), col1="VIX − IVV 20d RV", col2="VXN − QQQ 20d RV", unit="pts",
                 signal="panic" if vrp is not None and vrp < 0 else ("rich hedges" if vrp is not None and vrp > 8 else "normal"),
                 state="warn" if vrp is not None and vrp < 0 else "ok",
                 reading=("Realized volatility exceeds implied: the market is moving more than options priced, a panic signature." if vrp is not None and vrp < 0
                          else "Implied above realized, as usual: option sellers are being paid."),
                 threshold="< 0 = realized beating implied (panic)", as_of=data_through, source=SOURCE_CBOE)
        atrp = last(atr(qqq_df, 14) / qqq * 100)
        var_t = None
        if tqqq is not None:
            rvt = realized_vol(tqqq, 20)
            var_t = (last(rvt) or 0) / math.sqrt(252) * 2.33 if last(rvt) is not None else None
        self.add(LK, L, "ATR(14) % and 1-day 99% VaR", id="atr_var", value=atrp, fmt=pct(atrp, False, 2), value2=var_t, fmt2=pct(var_t, False), col1="QQQ ATR %", col2="TQQQ 1d VaR", unit="%",
                 signal="wide" if atrp is not None and atrp > 2.5 else "normal", state="warn" if atrp is not None and atrp > 2.5 else "ok",
                 reading=f"A $10,000 TQQQ position has a one-day 99% parametric VaR of about ${f((var_t or 0) * 100, '{:,.0f}')} at current volatility." if var_t is not None else "",
                 threshold="Use ATR for stops; VaR = 2.33 × daily σ", as_of=data_through)
        mv = last(move)
        self.add(LK, L, "MOVE index (Treasury volatility)", id="move", value=mv, fmt=f(mv, "{:.0f}"), col1="MOVE", col2="",
                 signal="stress" if mv is not None and mv > 120 else ("elevated" if mv is not None and mv > 100 else "calm"),
                 state="warn" if mv is not None and mv > 120 else "ok",
                 reading=("Bond volatility above 120: rates stress tends to spill into equity multiples." if mv is not None and mv > 120 else "Rates volatility contained."),
                 threshold="> 120 = rates stress", pctile=pctile_of_last(move) if move is not None else None, chg_5d=change(move, 5), as_of=last_date(move))

        # ---------------- D. breadth -----------------------------------------
        L, LK = "Breadth & internals", "breadth"
        spx_w, ndx_w = d.wide("spx"), d.wide("ndx")
        br = {}
        for label, w in (("spx", spx_w), ("ndx", ndx_w)):
            if w is None or w.shape[1] < 20:
                br[label] = None
                continue
            a200 = (w > w.rolling(200, min_periods=200).mean()).sum(axis=1) / w.notna().sum(axis=1) * 100
            a50 = (w > w.rolling(50, min_periods=50).mean()).sum(axis=1) / w.notna().sum(axis=1) * 100
            a200 = a200.where(w.rolling(200, min_periods=200).mean().notna().sum(axis=1) > w.shape[1] * 0.5)
            a50 = a50.where(w.rolling(50, min_periods=50).mean().notna().sum(axis=1) > w.shape[1] * 0.5)
            adv = (w.diff() > 0).sum(axis=1)
            dec = (w.diff() < 0).sum(axis=1)
            ratio = ((adv - dec) / (adv + dec).replace(0, np.nan) * 1000)
            mcc = ema(ratio, 19) - ema(ratio, 39)
            adline = (adv - dec).cumsum()
            hi = w.rolling(252, min_periods=252).max()
            lo = w.rolling(252, min_periods=252).min()
            nh = (w >= hi).sum(axis=1)
            nl = (w <= lo).sum(axis=1)
            nhnl = (nh - nl).where(hi.notna().sum(axis=1) > w.shape[1] * 0.5)
            br[label] = {"a200": a200, "a50": a50, "mcc": mcc, "adline": adline, "nhnl": nhnl, "n": int(w.shape[1])}
        bs = br["spx"]
        v = last(bs["a200"]) if bs else None
        self.add(LK, L, "% of S&P 500 above 200-day / 50-day", id="spx_breadth", value=v, fmt=pct(v, False, 0), value2=last(bs["a50"]) if bs else None, fmt2=pct(last(bs["a50"]) if bs else None, False, 0), col1="> 200d", col2="> 50d", unit="%",
                 signal="washout" if v is not None and v < 20 else ("frothy" if v is not None and v > 80 else ("weak" if v is not None and v < 40 else "healthy")),
                 state="warn" if v is not None and (v < 20 or v > 80) else ("bad" if v is not None and v < 40 else "ok"),
                 reading=("Fewer than a fifth of S&P stocks above their 200-day: washout territory." if v is not None and v < 20 else ("More than 80% above the 200-day: participation is frothy." if v is not None and v > 80 else "Participation is " + ("narrow." if v is not None and v < 40 else "broad."))),
                 threshold="< 20% washout, > 80% frothy", detail=f"Computed from {bs['n']} constituents' closes." if bs else "", pctile=pctile_of_last(bs["a200"]) if bs else None, chg_5d=change(bs["a200"], 5) if bs else None,
                 as_of=last_date(bs["a200"]) if bs else None, source="S&P 500 constituents (Wikipedia list) priced via Yahoo Finance; computed in the pipeline")
        nd = br["ndx"]
        v = last(nd["a50"]) if nd else None
        self.add(LK, L, "% of Nasdaq-100 above 50-day / 200-day", id="ndx_breadth", value=v, fmt=pct(v, False, 0), value2=last(nd["a200"]) if nd else None, fmt2=pct(last(nd["a200"]) if nd else None, False, 0), col1="> 50d", col2="> 200d", unit="%",
                 signal="washout" if v is not None and v < 20 else ("frothy" if v is not None and v > 85 else "normal"), state="warn" if v is not None and (v < 20 or v > 85) else "ok",
                 reading=f"{f(v, '{:.0f}')}% of Nasdaq-100 members are above their 50-day average." if v is not None else "",
                 threshold="< 20% washout, > 85% frothy", pctile=pctile_of_last(nd["a50"]) if nd else None, chg_5d=change(nd["a50"], 5) if nd else None,
                 as_of=last_date(nd["a50"]) if nd else None, source="Nasdaq-100 constituents (Wikipedia list) priced via Yahoo Finance; computed in the pipeline")
        mcc = last(bs["mcc"]) if bs else None
        ad_div = None
        if bs and len(bs["adline"].dropna()) > 25:
            price_hi = ivv.iloc[-1] >= ivv.iloc[-20:].max() * 0.999
            ad_hi = bs["adline"].iloc[-1] >= bs["adline"].iloc[-20:].max() * 0.999 if bs["adline"].iloc[-1] > 0 else bs["adline"].iloc[-1] >= bs["adline"].iloc[-20:].max()
            ad_div = bool(price_hi and not ad_hi)
        self.add(LK, L, "Advance-decline line and McClellan oscillator (S&P 500)", id="mcclellan", value=mcc, fmt=f(mcc, "{:+.0f}"), col1="McClellan", col2="",
                 signal="washout" if mcc is not None and mcc < -70 else ("overbought" if mcc is not None and mcc > 70 else ("divergence" if ad_div else "neutral")),
                 state="warn" if (mcc is not None and abs(mcc) > 70) or ad_div else "ok",
                 reading=("Breadth momentum deeply oversold." if mcc is not None and mcc < -70 else ("Breadth momentum overbought." if mcc is not None and mcc > 70
                          else ("Index at a 20-day high while the advance-decline line is not: a negative divergence." if ad_div else "No breadth divergence against price."))),
                 threshold="< −70 washout, > +70 overbought; A/D divergence = warning", detail="Ratio-adjusted McClellan from constituent advances/declines.", chg_5d=change(bs["mcc"], 5) if bs else None,
                 as_of=last_date(bs["mcc"]) if bs else None, source="S&P 500 constituents via Yahoo Finance; computed in the pipeline")
        nh = last(bs["nhnl"]) if bs else None
        self.add(LK, L, "52-week new highs minus new lows", id="nh_nl", value=nh, fmt=f(nh, "{:+.0f}"), value2=last(nd["nhnl"]) if nd else None, fmt2=f(last(nd["nhnl"]) if nd else None, "{:+.0f}"), col1="S&P 500", col2="Nasdaq-100",
                 signal="expanding lows" if nh is not None and nh < -20 else ("expanding highs" if nh is not None and nh > 40 else "neutral"), state="warn" if nh is not None and nh < -20 else "ok",
                 reading=("More stocks at 52-week lows than highs." if nh is not None and nh < 0 else "New highs outnumber new lows."), threshold="Negative on an up day = weak tape",
                 as_of=last_date(bs["nhnl"]) if bs else None, source="Constituent closes via Yahoo Finance; computed in the pipeline")
        rsp, qqqe = d.close("RSP"), d.close("QQQE")
        eq = {}
        for name, a, b in (("rsp_ivv", rsp, ivv), ("qqqe_qqq", qqqe, qqq)):
            if a is None:
                eq[name] = (None, None)
                continue
            ratio = (a / b.reindex(a.index)).dropna()
            eq[name] = (change(ratio, 21, pct=True), ratio)
        v = eq["rsp_ivv"][0]
        idx_up = change(ivv, 21, pct=True)
        self.add(LK, L, "Equal-weight vs cap-weight (RSP/IVV, QQQE/QQQ, 1-month change)", id="equal_weight", value=v, fmt=pct(v), value2=eq["qqqe_qqq"][0], fmt2=pct(eq["qqqe_qqq"][0]), col1="RSP/IVV", col2="QQQE/QQQ", unit="%",
                 signal="narrowing" if v is not None and v < -1 and (idx_up or 0) > 0 else ("broadening" if v is not None and v > 1 else "neutral"),
                 state="warn" if v is not None and v < -1 and (idx_up or 0) > 0 else "ok",
                 reading=("Equal weight lagging while the index rises: leadership is narrowing, fragile tape." if v is not None and v < -1 and (idx_up or 0) > 0
                          else ("Equal weight leading: participation broadening." if v is not None and v > 1 else "No strong concentration signal this month.")),
                 threshold="Falling ratio + rising index = narrow leadership", z=zscore_last(eq["rsp_ivv"][1]) if eq["rsp_ivv"][1] is not None else None, as_of=data_through)
        hold = d.misc("qqq_holdings")
        top10 = None
        names = ""
        top_src = "Invesco QQQ holdings file via the GitHub data feed"
        approx = ""
        if hold is not None and "weight" in hold.columns:
            h = hold.reset_index().sort_values("weight", ascending=False)
            top10 = float(h["weight"].head(10).sum())
            names = ", ".join(str(x) for x in h["ticker"].head(5))
        else:
            caps = self.d._csv(self.d.raw / "misc" / "ndx_marketcaps.csv", dedupe=False)
            if caps is not None and "market_cap" in caps.columns and len(caps) >= 90:
                c = caps.reset_index().copy()
                # dual-class listings (GOOGL/GOOG) report the whole company's cap on each line: split it between the classes
                base = c["name"].astype(str).str.lower().str.replace(r"\b(class [a-c]|common stock|ordinary shares|inc\.?|corporation|corp\.?|plc|ltd\.?|holdings?)\b", "", regex=True).str.replace(r"[^a-z0-9]", "", regex=True)
                c["market_cap"] = c["market_cap"] / base.map(base.value_counts())
                w = c["market_cap"] / c["market_cap"].sum()
                # Nasdaq-100 quarterly capping rule: if names above 4.5% sum to more than 48%, they are scaled to 40%
                big = w > 0.045
                if w[big].sum() > 0.48:
                    w = w.copy()
                    w[big] = w[big] * 0.40 / w[big].sum()
                    w[~big] = w[~big] * 0.60 / w[~big].sum()
                c["weight"] = w * 100
                c = c.sort_values("weight", ascending=False)
                top10 = float(c["weight"].head(10).sum())
                names = ", ".join(str(x) for x in c["ticker"].head(5))
                hold = caps
                top_src = "Nasdaq-100 member market caps (Nasdaq API) via the GitHub data feed, with the index's 4.5%/48%→40% capping rule applied; an approximation of the fund's weights"
                approx = " (approx.)"
        self.add(LK, L, "Top-10 holdings weight in QQQ", id="qqq_top10", value=top10, fmt=pct(top10, False) + approx, col1="top 10", col2="",
                 signal="concentrated" if top10 is not None and top10 > 50 else "normal", state="warn" if top10 is not None and top10 > 50 else "ok",
                 reading=f"Ten names are about {f(top10, '{:.0f}')}% of the index ({names}, …): single-stock risk is index risk." if top10 is not None else "",
                 threshold="> 50% = highly concentrated", as_of=last_date(hold) if hold is not None else None, source=top_src)

        # ---------------- E. sectors -----------------------------------------
        L, LK = "Relative pricing & rotation", "sectors"
        sectors = []
        rel_hist = {}
        for tkr, nm in CFG["sector_etfs"].items():
            s = d.close(tkr, adj=True)
            if s is None:
                continue
            bench = d.close("IVV", adj=True).reindex(s.index).ffill()
            rel = (s / bench)
            row = {"ticker": tkr, "sector": nm}
            for lbl, n in (("1m", 21), ("3m", 63), ("6m", 126), ("12m", 252)):
                row[f"ret_{lbl}"] = change(s, n, pct=True)
                row[f"rel_{lbl}"] = (change(s, n, pct=True) or 0) - (change(bench, n, pct=True) or 0) if change(s, n, pct=True) is not None else None
            rel_norm = rel / sma(rel, 126)
            row["rs_ratio"] = (last(rel_norm) - 1) * 100 if last(rel_norm) is not None else None
            row["rs_momentum"] = change(rel_norm, 10, pct=True)
            row["z_1y"] = zscore_last(rel)
            row["pctile_1y"] = pctile_of_last(rel)
            row["dist_200d"] = (last(s) / last(sma(s, 200)) - 1) * 100 if last(sma(s, 200)) else None
            q = ("leading" if (row["rs_ratio"] or 0) > 0 and (row["rs_momentum"] or 0) > 0 else "weakening" if (row["rs_ratio"] or 0) > 0
                 else "improving" if (row["rs_momentum"] or 0) > 0 else "lagging")
            row["quadrant"] = q
            rel_hist[tkr] = rel
            sectors.append(row)
        sectors.sort(key=lambda r: -(r["rel_3m"] if r["rel_3m"] is not None else -999))
        for i, r in enumerate(sectors, 1):
            r["rank_3m"] = i
        leaders = ", ".join(r["ticker"] for r in sectors[:3])
        laggards = ", ".join(r["ticker"] for r in sectors[-3:])
        self.add(LK, L, "Sector relative strength vs IVV (3-month, best / worst)", id="sector_rs", value=sectors[0]["rel_3m"] if sectors else None, fmt=f(sectors[0]["rel_3m"], "{:+.1f} pp") if sectors else "n/a",
                 value2=sectors[-1]["rel_3m"] if sectors else None, fmt2=f(sectors[-1]["rel_3m"], "{:+.1f} pp") if sectors else "", col1=sectors[0]["ticker"] if sectors else "", col2=sectors[-1]["ticker"] if sectors else "", unit="pp",
                 signal="rotation" if sectors else "n/a", state="neutral", reading=f"Leading: {leaders}. Lagging: {laggards}." if sectors else "",
                 threshold="RRG-style: ratio vs IVV and its momentum", detail="Full table in the Sectors section.", as_of=data_through)
        stretched = [r for r in sectors if r["z_1y"] is not None and abs(r["z_1y"]) >= 2]
        v = len(stretched)
        self.add(LK, L, "Sectors stretched ±2σ vs their own 1-year relative norm", id="sector_z", value=v, fmt=f(v, "{:.0f}"), col1="count", col2="",
                 signal="stretched" if v else "normal", state="warn" if v else "ok",
                 reading=("Stretched: " + ", ".join(f"{r['ticker']} ({r['z_1y']:+.1f}σ)" for r in stretched)) if stretched else "No sector is more than two standard deviations from its usual relationship to IVV.",
                 threshold="|z| ≥ 2 flags stretched or washed-out sectors", as_of=data_through)
        disp_series = None
        if len(sectors) >= 6:
            rets = pd.concat([(d.close(t, adj=True).pct_change(21) * 100) for t in CFG["sector_etfs"] if d.close(t) is not None], axis=1)
            disp_series = rets.std(axis=1, ddof=0).dropna()
        dv = last(disp_series)
        self.add(LK, L, "Cross-sector dispersion of 1-month returns", id="sector_dispersion", value=dv, fmt=f(dv, "{:.1f} pp"), col1="σ across 11 sectors", col2="", unit="pp",
                 signal="high rotation" if dv is not None and (pctile_of_last(disp_series) or 0) > 80 else ("low" if dv is not None and (pctile_of_last(disp_series) or 0) < 20 else "normal"),
                 state="warn" if dv is not None and (pctile_of_last(disp_series) or 0) > 80 else "ok",
                 reading=f"Dispersion at the {f(pctile_of_last(disp_series), '{:.0f}')}th percentile of the past year." + (" Rotation is active: relative bets matter more than index direction." if dv is not None and (pctile_of_last(disp_series) or 0) > 80 else ""),
                 threshold="> 80th percentile = rotation underway", pctile=pctile_of_last(disp_series), as_of=data_through)
        ratios = {}
        for key, a, b, nm in (("xly_xlp", "XLY", "XLP", "Offense/defense (XLY/XLP)"), ("sphb_splv", "SPHB", "SPLV", "High beta / low vol (SPHB/SPLV)"), ("smh_qqq", "SMH", "QQQ", "Semis / Nasdaq-100 (SMH/QQQ)")):
            sa, sb = d.close(a, adj=True), d.close(b, adj=True)
            if sa is None or sb is None:
                ratios[key] = None
                self.add(LK, L, nm, id=key, value=None)
                continue
            r = (sa / sb.reindex(sa.index)).dropna()
            ratios[key] = r
            above = (last(r) or 0) > (last(sma(r, 50)) or 0)
            c1m = change(r, 21, pct=True)
            self.add(LK, L, nm, id=key, value=c1m, fmt=pct(c1m), value2=zscore_last(r), fmt2=f(zscore_last(r), "{:+.2f}σ"), col1="1-month change", col2="z vs 1y", unit="%",
                     signal="risk-on" if above and (c1m or 0) > 0 else ("risk-off" if not above and (c1m or 0) < 0 else "mixed"),
                     state="ok" if above else "warn",
                     reading=(f"{a} is {'out' if (c1m or 0) > 0 else 'under'}performing {b} this month and the ratio is {'above' if above else 'below'} its 50-day average."),
                     threshold="Rising ratio above its 50d = risk appetite", pctile=pctile_of_last(r), chg_5d=change(r, 5, pct=True), as_of=data_through)
        qi = (qqq / ivv.reindex(qqq.index)).dropna()
        zq = zscore_last(qi)
        self.add(LK, L, "QQQ / IVV ratio z-score (1 year)", id="qqq_ivv_z", value=zq, fmt=f(zq, "{:+.2f}σ"), value2=change(qi, 21, pct=True), fmt2=pct(change(qi, 21, pct=True)), col1="z-score", col2="1-month change",
                 signal="QQQ stretched" if zq is not None and zq > 2 else ("QQQ washed out" if zq is not None and zq < -2 else "normal"), state="warn" if zq is not None and abs(zq) > 2 else "ok",
                 reading=("QQQ is unusually rich versus IVV: a rebalance tilt toward IVV is the mean-reversion play." if zq is not None and zq > 2
                          else ("QQQ unusually cheap versus IVV: tilt toward QQQ on a rebalance." if zq is not None and zq < -2 else "Growth vs broad market within its normal band.")),
                 threshold="|z| > 2 = rebalance signal between your two core holdings", pctile=pctile_of_last(qi), as_of=data_through)

        # ---------------- F. sentiment, valuation, calendar -----------------------------------------
        L, LK = "Sentiment, valuation & calendar", "sentiment"
        pc = d.misc("cboe_putcall")
        pc10 = None
        pc_series = None
        if pc is not None and "equity_pc" in pc.columns:
            pc_series = pd.to_numeric(pc["equity_pc"], errors="coerce").dropna()
            pc10 = last(pc_series.rolling(10, min_periods=1).mean())
        sk = last(skew)
        self.add(LK, L, "Equity put/call ratio (10-day avg) and SKEW", id="put_call", value=pc10, fmt=f(pc10, "{:.2f}"), value2=sk, fmt2=f(sk, "{:.0f}"), col1="put/call 10d", col2="SKEW",
                 signal="fear" if pc10 is not None and pc10 > 0.9 else ("complacent" if pc10 is not None and pc10 < 0.55 else "neutral"), state="warn" if pc10 is not None and (pc10 > 0.9 or pc10 < 0.55) else "ok",
                 reading=("Heavy put buying: fear extreme, contrarian bullish." if pc10 is not None and pc10 > 0.9 else ("Very light put demand: complacency." if pc10 is not None and pc10 < 0.55 else "Options sentiment unremarkable.")),
                 threshold="> 0.9 fear, < 0.55 complacency; SKEW > 150 heavy tail hedging", as_of=last_date(pc_series) if pc_series is not None else None,
                 source="Cboe daily market statistics and SKEW index via the GitHub data feed", status=None if pc10 is not None else "unavailable")
        pe, cape = d.misc("sp500_pe"), d.misc("shiller_cape")
        pev = last(pe["value"]) if pe is not None else None
        capev = last(cape["value"]) if cape is not None else None
        pe_pct = pctile_of_last(pe["value"], 120) if pe is not None else None
        self.add(LK, L, "S&P 500 P/E (trailing) and Shiller CAPE", id="valuation", value=pev, fmt=f(pev), value2=capev, fmt2=f(capev), col1="P/E", col2="CAPE",
                 signal="expensive" if pe_pct is not None and pe_pct > 80 else ("cheap" if pe_pct is not None and pe_pct < 20 else "mid-range"), state="warn" if pe_pct is not None and pe_pct > 80 else "ok",
                 reading=f"Trailing P/E at the {f(pe_pct, '{:.0f}')}th percentile of the last ten years." if pe_pct is not None else "",
                 threshold="> 80th percentile of 10 years = rich", pctile=pe_pct, as_of=last_date(pe["value"]) if pe is not None else None, source="multpl.com monthly tables via the GitHub data feed")
        dgs10 = d.fred("DGS10")
        erp = (100 / pev - (last(dgs10) or 0)) if pev else None
        self.add(LK, L, "Equity risk premium (earnings yield − 10-year)", id="erp", value=erp, fmt=f(erp, "{:+.2f} pp"), value2=(100 / pev) if pev else None, fmt2=pct(100 / pev if pev else None, False, 2), col1="ERP", col2="earnings yield", unit="pp",
                 signal="thin cushion" if erp is not None and erp < 1 else ("ample" if erp is not None and erp > 3 else "normal"), state="warn" if erp is not None and erp < 1 else "ok",
                 reading=("Stocks yield barely more than Treasuries: little valuation cushion if rates rise." if erp is not None and erp < 1 else "Equities still carry a yield premium over bonds."),
                 threshold="< 1 pp = thin cushion", as_of=last_date(dgs10), source="multpl.com P/E and FRED DGS10")
        tdd = tpos = None
        if tqqq is not None:
            tdd = last(drawdown(tqqq) * 100)
            lo52, hi52t = tqqq.iloc[-252:].min(), tqqq.iloc[-252:].max()
            tpos = (last(tqqq) - lo52) / (hi52t - lo52) * 100 if hi52t > lo52 else None
        self.add(LK, L, "TQQQ drawdown from 52-week high and range position", id="tqqq_drawdown", value=tdd, fmt=pct(tdd), value2=tpos, fmt2=pct(tpos, False, 0), col1="drawdown", col2="position in 52w range", unit="%",
                 signal="deep" if tdd is not None and tdd < -40 else ("down" if tdd is not None and tdd < -20 else "near highs"), state="warn" if tdd is not None and tdd < -40 else "neutral",
                 reading=f"TQQQ sits {f(tpos, '{:.0f}')}% of the way up its 52-week range." if tpos is not None else "", threshold="A −20% QQQ dip is roughly −50% in TQQQ", as_of=data_through, status=None if tdd is not None else "unavailable")
        mret = qqq.resample("ME").last().pct_change() * 100
        mret = mret.dropna()
        this_m, next_m = self.as_of.month, (self.as_of.month % 12) + 1
        same = mret[mret.index.month == this_m]
        nxt = mret[mret.index.month == next_m]
        pos_share = float((same > 0).mean() * 100) if len(same) else None
        phase = ["election year", "post-election year", "midterm year", "pre-election year"][(self.as_of.year - 2024) % 4]
        self.add(LK, L, "Seasonality: average QQQ return this month / next month", id="seasonality", value=float(same.mean()) if len(same) else None, fmt=pct(float(same.mean()) if len(same) else None),
                 value2=float(nxt.mean()) if len(nxt) else None, fmt2=pct(float(nxt.mean()) if len(nxt) else None), col1=date(2000, this_m, 1).strftime("%B"), col2=date(2000, next_m, 1).strftime("%B"), unit="%",
                 signal="tailwind" if len(same) and same.mean() > 1 else ("headwind" if len(same) and same.mean() < 0 else "neutral"), state="neutral",
                 reading=f"QQQ has finished {date(2000, this_m, 1).strftime('%B')} higher {f(pos_share, '{:.0f}')}% of the time since {mret.index[0].year}; cycle phase: {phase}." if len(same) else "",
                 threshold="Nov–Apr has historically been the stronger half; the 12 months after a midterm the strongest cycle stretch", as_of=data_through)
        cal = d.misc_json("calendar") or {}
        def next_event(key):
            ds = sorted(x for x in (cal.get(key) or []) + CFG.get("calendar_fallback", {}).get(key, []) if x >= self.as_of.isoformat())
            if not ds:
                return None, None
            nd_ = date.fromisoformat(ds[0])
            return nd_.isoformat(), (nd_ - self.as_of).days
        ev = {k: next_event(k) for k in ("cpi", "fomc", "nfp")}
        earn = sorted((v_, k) for k, v_ in CFG.get("earnings_watch", {}).items() if isinstance(v_, str) and len(v_) == 10 and v_ >= self.as_of.isoformat())
        self.add(LK, L, "Days to next CPI / FOMC decision", id="event_risk", value=ev["cpi"][1], fmt=f(ev["cpi"][1], "{:.0f} d"), value2=ev["fomc"][1], fmt2=f(ev["fomc"][1], "{:.0f} d"), col1="CPI", col2="FOMC", unit="days",
                 signal="event week" if any(x[1] is not None and x[1] <= 2 for x in ev.values()) else "clear", state="warn" if any(x[1] is not None and x[1] <= 2 for x in ev.values()) else "ok",
                 reading=("A binary macro print lands within two sessions: avoid adding leverage blind into it." if any(x[1] is not None and x[1] <= 2 for x in ev.values()) else "No major macro print in the next two sessions."),
                 threshold="≤ 2 days = event risk for new leverage",
                 detail=f"Payrolls {ev['nfp'][0] or 'n/a'}; CPI {ev['cpi'][0] or 'n/a'}; FOMC {ev['fomc'][0] or 'n/a'}." + (f" Next mega-cap earnings: {earn[0][1]} {earn[0][0]}." if earn else ""),
                 as_of=cal.get("fetched", ""), source="Federal Reserve and BLS release calendars via the GitHub data feed", status="live" if ev["cpi"][1] is not None or ev["fomc"][1] is not None else "unavailable")

        # ---------------- G. credit, rates, liquidity -----------------------------------------
        L, LK = "Credit, rates & liquidity", "credit"
        hyv, hy30 = last(hy), (change(hy, 21) or 0) * 100 if hy is not None and change(hy, 21) is not None else None
        hyg, ief = d.close("HYG", adj=True), d.close("IEF", adj=True)
        hyg_ief = change((hyg / ief.reindex(hyg.index)).dropna(), 21, pct=True) if hyg is not None and ief is not None else None
        self.add(LK, L, "High-yield credit spread (OAS) and 30-day change", id="hy_oas", value=hyv, fmt=pct(hyv, False, 2), value2=hy30, fmt2=bp(hy30), col1="OAS", col2="30d change", unit="%",
                 signal="stress" if (hyv is not None and hyv > 5) or (hy30 is not None and hy30 > 100) else ("widening" if hy30 is not None and hy30 > 50 else ("tight" if hyv is not None and hyv < 3.5 else "normal")),
                 state="bad" if (hyv is not None and hyv > 5) or (hy30 is not None and hy30 > 100) else ("warn" if hy30 is not None and hy30 > 50 else "ok"),
                 reading=("Credit is flashing stress: spreads above 5% or widening more than 100 bp in a month." if (hyv is not None and hyv > 5) or (hy30 is not None and hy30 > 100)
                          else ("Spreads widening fast: credit is leading equities lower." if hy30 is not None and hy30 > 50 else ("Spreads tight: credit sees no trouble (and offers no cushion)." if hyv is not None and hyv < 3.5 else "Credit calm."))),
                 threshold="> 5% or +100 bp/month = stress", detail=f"HYG/IEF ratio 1-month change {pct(hyg_ief)} (daily proxy).", pctile=pctile_of_last(hy) if hy is not None else None, z=zscore_last(hy) if hy is not None else None,
                 chg_5d=(change(hy, 5) or 0) * 100 if hy is not None and change(hy, 5) is not None else None, as_of=last_date(hy), source=SOURCE_FRED)
        c3, c2 = last(t10y3m), last(t10y2y)
        inv_recent = bool(t10y3m is not None and (t10y3m.iloc[-252:] < 0).any())
        resteep = bool(c3 is not None and inv_recent and c3 > 0 and (c3 - float(t10y3m.iloc[-252:].min())) > 0.5)
        self.add(LK, L, "Yield curve: 10y − 3m and 10y − 2y", id="yield_curve", value=c3, fmt=f(c3, "{:+.2f} pp"), value2=c2, fmt2=f(c2, "{:+.2f} pp"), col1="10y − 3m", col2="10y − 2y", unit="pp",
                 signal="inverted" if c3 is not None and c3 < 0 else ("re-steepening" if resteep else "normal"), state="warn" if (c3 is not None and c3 < 0) or resteep else "ok",
                 reading=("Curve inverted: the recession clock is running, though equities often rise for months after inversion." if c3 is not None and c3 < 0
                          else ("Curve has re-steepened out of inversion: historically the window when recessions actually began." if resteep else "Curve positively sloped with no recent inversion.")),
                 threshold="Inversion = warning; re-steepening after inversion = timing risk", chg_5d=(change(t10y3m, 5) or 0) * 100 if t10y3m is not None and change(t10y3m, 5) is not None else None, as_of=last_date(t10y3m), source=SOURCE_FRED)
        rv_, r1m = last(real10), (change(real10, 21) or 0) * 100 if real10 is not None and change(real10, 21) is not None else None
        self.add(LK, L, "10-year real yield (TIPS) and 1-month change", id="real_yield", value=rv_, fmt=pct(rv_, False, 2), value2=r1m, fmt2=bp(r1m), col1="real yield", col2="1m change", unit="%",
                 signal="rising fast" if r1m is not None and r1m > 50 else ("falling fast" if r1m is not None and r1m < -50 else "stable"), state="warn" if r1m is not None and r1m > 50 else "ok",
                 reading=("Real yields up more than 50 bp in a month: the main headwind for long-duration growth multiples." if r1m is not None and r1m > 50
                          else ("Real yields falling sharply: a tailwind for QQQ multiples." if r1m is not None and r1m < -50 else "Real yields steady.")),
                 threshold="+50 bp/month = pressure on QQQ", pctile=pctile_of_last(real10) if real10 is not None else None, as_of=last_date(real10), source=SOURCE_FRED)
        dff, dgs1, dgs6 = d.fred("DFF"), d.fred("DGS1"), d.fred("DGS6MO")
        path12 = (last(dgs1) - last(dff)) * 100 if last(dgs1) is not None and last(dff) is not None else None
        path6 = (last(dgs6) - last(dff)) * 100 if last(dgs6) is not None and last(dff) is not None else None
        self.add(LK, L, "Fed path priced by the curve (1y and 6m yield minus fed funds)", id="fed_path", value=path12, fmt=bp(path12), value2=path6, fmt2=bp(path6), col1="12 months", col2="6 months", unit="bp",
                 signal="cuts priced" if path12 is not None and path12 < -25 else ("hikes priced" if path12 is not None and path12 > 25 else "on hold"), state="neutral",
                 reading=f"Roughly {f(abs(path12) / 25 if path12 is not None else None, '{:.1f}')} quarter-point {'cuts' if (path12 or 0) < 0 else 'hikes'} implied over the next year (proxy from the 1-year Treasury)." if path12 is not None else "",
                 threshold="Proxy for futures-implied path; each 25 bp ≈ one move", detail=f"Effective fed funds {pct(last(dff), False, 2)}.", as_of=last_date(dgs1), source=SOURCE_FRED)
        walcl, tga, rrp = d.fred("WALCL"), d.fred("WTREGEN"), d.fred("RRPONTSYD")
        netliq = None
        nl_series = None
        if walcl is not None and tga is not None and rrp is not None:
            wk = pd.concat([walcl, tga, rrp.resample("W-WED").last()], axis=1, keys=["a", "t", "r"], sort=True).ffill().dropna()
            nl_series = (wk["a"] - wk["t"] - wk["r"]) / 1e6  # millions -> $ trillions
            netliq = last(nl_series)
        nl4 = (change(nl_series, 4) or 0) * 1000 if nl_series is not None and change(nl_series, 4) is not None else None
        self.add(LK, L, "Net liquidity (Fed assets − TGA − reverse repo)", id="net_liquidity", value=netliq, fmt=f(netliq, "${:.2f}T"), value2=nl4, fmt2=f(nl4, "{:+.0f}B / 4 wk"), col1="level", col2="4-week change", unit="$T",
                 signal="draining" if nl4 is not None and nl4 < -100 else ("adding" if nl4 is not None and nl4 > 100 else "flat"), state="warn" if nl4 is not None and nl4 < -100 else "ok",
                 reading=("Liquidity draining more than $100B over four weeks: a headwind for the Nasdaq." if nl4 is not None and nl4 < -100 else ("Liquidity expanding: supportive." if nl4 is not None and nl4 > 100 else "Net liquidity roughly flat.")),
                 threshold="±$100B / 4 weeks", as_of=last_date(nl_series), source=SOURCE_FRED)
        nfci = d.fred("NFCI")
        self.add(LK, L, "Chicago Fed financial conditions (NFCI)", id="nfci", value=last(nfci), fmt=f(last(nfci), "{:+.2f}"), value2=change(nfci, 4), fmt2=f(change(nfci, 4), "{:+.2f} / 4 wk"), col1="index", col2="4-week change",
                 signal="tight" if (last(nfci) or 0) > 0 else "loose", state="warn" if (last(nfci) or 0) > 0 else "ok",
                 reading=("Conditions tighter than average." if (last(nfci) or 0) > 0 else "Financial conditions looser than average."), threshold="> 0 = tighter than average", as_of=last_date(nfci), source=SOURCE_FRED)
        dxy = d.fred("DTWEXBGS")
        self.add(LK, L, "Dollar (trade-weighted broad index)", id="dollar", value=last(dxy), fmt=f(last(dxy)), value2=change(dxy, 21, pct=True), fmt2=pct(change(dxy, 21, pct=True)), col1="index", col2="1-month change", unit="%",
                 signal="strengthening" if (change(dxy, 21, pct=True) or 0) > 1.5 else ("weakening" if (change(dxy, 21, pct=True) or 0) < -1.5 else "stable"), state="warn" if (change(dxy, 21, pct=True) or 0) > 1.5 else "ok",
                 reading=("Dollar up sharply: a headwind for mega-cap tech earnings." if (change(dxy, 21, pct=True) or 0) > 1.5 else "Dollar not a major factor this month."), threshold="> +1.5% / month = headwind",
                 pctile=pctile_of_last(dxy) if dxy is not None else None, as_of=last_date(dxy), source=SOURCE_FRED)

        # ---------------- H. macro nowcasts -----------------------------------------
        L, LK = "Macro nowcasts & lead indicators", "macro"
        nc = d.misc("cleveland_nowcast")
        ncv = {}
        if nc is not None and len(nc):
            latest = nc.reset_index()
            latest = latest[latest["date"] == latest["date"].max()]
            def measure_of(txt: str):
                t = txt.lower().strip()
                if "core cpi" in t:
                    return "core_cpi"
                if "core pce" in t:
                    return "core_pce"
                if t in ("cpi", "headline cpi"):
                    return "cpi"
                if t in ("pce", "headline pce"):
                    return "pce"
                return None
            for _, r in latest.iterrows():
                row_t, col_t = str(r["row"]), str(r["col"])
                key = measure_of(col_t) or measure_of(row_t)
                if not key:
                    continue
                other = (row_t if measure_of(col_t) else col_t).lower()
                if "year" in other:
                    kind = "yoy"
                elif "month" in other:
                    kind = "mom"
                else:
                    kind = {0: "mom", 1: "yoy", 2: "qtr"}.get(int(r["table"]) if str(r["table"]).isdigit() else -1)
                if kind and f"{key}_{kind}" not in ncv:
                    ncv[f"{key}_{kind}"] = float(r["value"])
        cpi = d.fred("CPIAUCSL")
        cpi_yoy = (change(cpi, 12, pct=True)) if cpi is not None else None
        cpi_mom = change(cpi, 1, pct=True) if cpi is not None else None
        self.add(LK, L, "Cleveland Fed inflation nowcast (CPI y/y, core CPI m/m)", id="inflation_nowcast", value=ncv.get("cpi_yoy"), fmt=pct(ncv.get("cpi_yoy"), False, 2), value2=ncv.get("core_cpi_mom"), fmt2=pct(ncv.get("core_cpi_mom"), False, 2), col1="CPI y/y nowcast", col2="core CPI m/m nowcast", unit="%",
                 signal="hot" if ncv.get("core_cpi_mom") is not None and ncv["core_cpi_mom"] >= 0.3 else ("cool" if ncv.get("core_cpi_mom") is not None and ncv["core_cpi_mom"] <= 0.15 else "in line"),
                 state="warn" if ncv.get("core_cpi_mom") is not None and ncv["core_cpi_mom"] >= 0.3 else "ok",
                 reading=(f"Nowcast runs {f(ncv.get('cpi_yoy'), '{:.2f}')}% y/y versus the last published {f(cpi_yoy, '{:.2f}')}%: " + ("inflation re-accelerating into the print." if ncv.get("cpi_yoy") is not None and cpi_yoy is not None and ncv["cpi_yoy"] > cpi_yoy + 0.1 else "no upside surprise building.")) if ncv else "",
                 threshold="Core m/m ≥ 0.30% = hot print risk", detail=f"PCE y/y nowcast {pct(ncv.get('pce_yoy'), False, 2)}; core PCE m/m {pct(ncv.get('core_pce_mom'), False, 2)}. Last actual CPI m/m {pct(cpi_mom, False, 2)}.",
                 as_of=last_date(nc) if nc is not None else None, source="Cleveland Fed inflation nowcasting page via the GitHub data feed", status="live" if ncv else "unavailable")
        gas = d.misc("aaa_gas")
        gasw = d.fred("GASREGW")
        wti = d.fred("DCOILWTICO")
        gv = last(gas["regular"]) if gas is not None and "regular" in gas.columns else last(gasw)
        yago = last(gas["year_ago"]) if gas is not None and "year_ago" in gas.columns else (last(gasw, 52) if gasw is not None else None)
        gas_yoy = (gv / yago - 1) * 100 if gv and yago else None
        self.add(LK, L, "AAA national average gas (regular) and WTI crude", id="gas_oil", value=gv, fmt=f(gv, "${:.2f}"), value2=last(wti), fmt2=f(last(wti), "${:.0f}"), col1="gas $/gal", col2="WTI $/bbl",
                 signal="squeeze" if gas_yoy is not None and gas_yoy > 10 else ("relief" if gas_yoy is not None and gas_yoy < -10 else "steady"), state="warn" if gas_yoy is not None and gas_yoy > 10 else "ok",
                 reading=(f"Gas is {pct(gas_yoy)} year over year" + ("; a consumer squeeze and a headline-CPI tailwind." if gas_yoy is not None and gas_yoy > 10 else ("; consumer relief and headline-CPI drag." if gas_yoy is not None and gas_yoy < -10 else "; neutral for the consumer."))) if gas_yoy is not None else "",
                 threshold="±10% y/y moves headline CPI and XLE", detail=(f"Week ago ${f(last(gas['week_ago']), '{:.2f}')}, month ago ${f(last(gas['month_ago']), '{:.2f}')}, year ago ${f(yago, '{:.2f}')}." if gas is not None and "week_ago" in gas.columns else "") + f" WTI 1-month change {pct(change(wti, 21, pct=True))}.",
                 chg_5d=(gv - last(gas["week_ago"])) if gas is not None and "week_ago" in gas.columns and gv is not None and last(gas["week_ago"]) is not None else None,
                 as_of=last_date(gas) if gas is not None else last_date(gasw), source="AAA gasprices.aaa.com (daily) with EIA weekly via FRED as history; WTI from FRED")
        gdp = d.misc("gdpnow")
        gdp_src = "Atlanta Fed GDPNow page via the GitHub data feed"
        if gdp is None or "estimate" not in gdp.columns:
            g = d.fred("GDPNOW")
            if g is not None and len(g):
                gdp = pd.DataFrame({"estimate": g.values, "quarter": ""}, index=g.index)
                gdp_src = "Atlanta Fed GDPNow via FRED (series GDPNOW)"
        gdpv = last(gdp["estimate"]) if gdp is not None and "estimate" in gdp.columns else None
        gdp_prev = last(gdp["estimate"], 1) if gdp is not None and "estimate" in gdp.columns else None
        self.add(LK, L, "Atlanta Fed GDPNow", id="gdpnow", value=gdpv, fmt=pct(gdpv, False), value2=(gdpv - gdp_prev) if gdpv is not None and gdp_prev is not None else None, fmt2=f((gdpv - gdp_prev) if gdpv is not None and gdp_prev is not None else None, "{:+.1f} pp vs prior"), col1="real GDP nowcast", col2="revision", unit="%",
                 signal="strong" if gdpv is not None and gdpv > 3 else ("stall" if gdpv is not None and gdpv < 1 else "trend"), state="warn" if gdpv is not None and gdpv < 1 else "ok",
                 reading=(f"GDPNow tracks {f(gdpv)}% annualized for {str(gdp['quarter'].iloc[-1]) or 'the current quarter'}." if gdpv is not None else ""),
                 threshold="< 1% stall speed, > 3% hot", as_of=last_date(gdp["estimate"]) if gdp is not None else None, source=gdp_src)
        cl4 = claims.rolling(4).mean() if claims is not None else None
        clv = last(cl4)
        cl_low = float(cl4.iloc[-52:].min()) if cl4 is not None and len(cl4.dropna()) > 10 else None
        cl_off = (clv / cl_low - 1) * 100 if clv and cl_low else None
        self.add(LK, L, "Initial jobless claims (4-week average)", id="claims", value=clv / 1000 if clv else None, fmt=f(clv / 1000 if clv else None, "{:.0f}k"), value2=cl_off, fmt2=pct(cl_off), col1="4-wk avg", col2="vs 52-wk low", unit="k",
                 signal="labor warning" if cl_off is not None and cl_off > 15 else ("softening" if cl_off is not None and cl_off > 8 else "healthy"), state="bad" if cl_off is not None and cl_off > 15 else ("warn" if cl_off is not None and cl_off > 8 else "ok"),
                 reading=(f"Claims are {pct(cl_off)} above their 52-week low" + ("; past 15% has preceded most slowdowns." if cl_off is not None and cl_off > 15 else "; the labor market is holding.")) if cl_off is not None else "",
                 threshold="> +15% off the low = classic warning", as_of=last_date(claims), source=SOURCE_FRED)
        ism = d.misc("ism")
        pmi = last(ism["pmi"]) if ism is not None and "pmi" in ism.columns else None
        no = last(ism["new_orders"]) if ism is not None and "new_orders" in ism.columns else None
        svc = last(ism["services_pmi"]) if ism is not None and "services_pmi" in ism.columns else None
        self.add(LK, L, "ISM manufacturing PMI and new orders", id="ism", value=pmi, fmt=f(pmi), value2=no, fmt2=f(no), col1="PMI", col2="new orders",
                 signal="contraction" if pmi is not None and pmi < 50 else "expansion", state="warn" if pmi is not None and pmi < 48 else "ok",
                 reading=(f"Manufacturing {'contracting' if pmi < 50 else 'expanding'} ({f(pmi)}); services PMI {f(svc)}." if pmi is not None else (f"Services PMI {f(svc)}." if svc is not None else "")),
                 threshold="< 50 contraction; new orders lead", as_of=last_date(ism) if ism is not None else None, source="ISM Report on Business pages via the GitHub data feed", status="live" if pmi is not None or svc is not None else "unavailable")
        t5, mich, t10ie = d.fred("T5YIFR"), d.fred("MICH"), d.fred("T10YIE")
        self.add(LK, L, "Inflation expectations: 5y5y forward and Michigan 1-year", id="inflation_expectations", value=last(t5), fmt=pct(last(t5), False, 2), value2=last(mich), fmt2=pct(last(mich), False, 1), col1="5y5y breakeven", col2="Michigan 1y", unit="%",
                 signal="unanchored" if (last(t5) or 0) > 2.75 or (last(mich) or 0) > 4.5 else "anchored", state="warn" if (last(t5) or 0) > 2.75 or (last(mich) or 0) > 4.5 else "ok",
                 reading=("Long-run expectations drifting above the Fed's comfort zone." if (last(t5) or 0) > 2.75 else "Market expectations anchored near 2%.") + f" 10-year breakeven {pct(last(t10ie), False, 2)}.",
                 threshold="5y5y > 2.75% or Michigan 1y > 4.5% = unanchored", chg_5d=(change(t5, 5) or 0) * 100 if t5 is not None and change(t5, 5) is not None else None, as_of=last_date(t5), source=SOURCE_FRED)

        # ---------------- composites -----------------------------------------
        reg = CFG["composites"]["regime"]
        regime_parts = []
        rs_ = 0.0
        regime_parts.append({"component": "QQQ vs 200-day", "value": "above" if tq["above"] else "below", "points": 1.0 if tq["above"] else -1.0})
        regime_parts.append({"component": "200-day slope", "value": pct(tq["slope"]), "points": 0.5 if (tq["slope"] or 0) > 0 else -0.5})
        if hy30 is not None:
            p = 0.5 if hy30 <= 25 else (0.0 if hy30 <= 50 else -1.0)
            if hyv is not None and hyv > 5:
                p -= 0.5
            regime_parts.append({"component": "HY spread (30d change, level)", "value": f"{bp(hy30)}, {pct(hyv, False, 2)}", "points": p})
        if cl_off is not None:
            regime_parts.append({"component": "Claims vs 52-week low", "value": pct(cl_off), "points": 0.5 if cl_off <= 10 else (0.0 if cl_off <= 15 else -1.0)})
        if c3 is not None:
            regime_parts.append({"component": "10y − 3m curve", "value": f(c3, "{:+.2f} pp"), "points": -0.5 if c3 < 0 else 0.0})
        rs_ = sum(p["points"] for p in regime_parts)
        light = "green" if rs_ >= reg["green_at"] else ("red" if rs_ <= reg["red_at"] else "yellow")
        max_alloc = reg["max_tqqq_allocation_pct"][light]

        dip_comps, pen = dip_components(dd=ddq, rsi14=last(r14["QQQ"]), rsi2=last(r2["QQQ"]), pb=last(pbq), vix=vv, vix_pct=pctile_of_last(vix) if vix is not None else None,
                                        vix_ts=ts, breadth200=last(bs["a200"]) if bs else None, pc10=pc10, hy_chg30=hy30)
        dip_score, dip_parts, dip_missing = composite(dip_comps, pen)
        heat_comps, _ = heat_components(dist200=tq["dist"], rsi14=last(r14["QQQ"]), pb=last(pbq), vix=vv, pc10=pc10, breadth200=last(bs["a200"]) if bs else None,
                                         ad_div=ad_div, qqq_ivv_z=zq, days_since_pullback=days_since((drawdown(qqq, None) <= -0.05)))
        heat_score, heat_parts, heat_missing = composite(heat_comps)
        add_levels, trim_levels = CFG["composites"]["dip"]["add_tranche_at"], CFG["composites"]["heat"]["trim_tranche_at"]
        dip_tranches = sum(1 for lv in add_levels if dip_score is not None and dip_score >= lv)
        trim_tranches = sum(1 for lv in trim_levels if heat_score is not None and heat_score >= lv)
        if light == "red":
            action = (f"Regime red: no new TQQQ exposure. Dip score {dip_score} would otherwise flag {dip_tranches} tranche(s); wait for QQQ to reclaim the 200-day."
                      if dip_tranches else "Regime red: stay out of TQQQ; dip score is not elevated anyway.")
        elif trim_tranches:
            action = f"Heat score {heat_score}: trim tranche {trim_tranches} of 3 in TQQQ (levels {', '.join(map(str, trim_levels))})."
        elif dip_tranches:
            action = f"Dip score {dip_score}: add tranche {dip_tranches} of 3 in TQQQ, capped at {max_alloc}% of the portfolio for a {light} regime."
        else:
            action = f"No tranche trigger. Dip score {dip_score} (first add at {add_levels[0]}), heat score {heat_score} (first trim at {trim_levels[0]}); regime {light}, TQQQ cap {max_alloc}%."

        # ---------------- series for charts -----------------------------------------
        idx = qqq.index[-504:]
        ser = pd.DataFrame(index=idx)
        ser["qqq"], ser["ivv"] = qqq.reindex(idx), ivv.reindex(idx)
        ser["tqqq"] = tqqq.reindex(idx) if tqqq is not None else np.nan
        ser["qqq_sma50"], ser["qqq_sma200"] = trend["QQQ"]["s50"].reindex(idx), trend["QQQ"]["s200"].reindex(idx)
        ser["ivv_sma50"], ser["ivv_sma200"] = trend["IVV"]["s50"].reindex(idx), trend["IVV"]["s200"].reindex(idx)
        ser["qqq_dd"], ser["ivv_dd"] = dd["QQQ"].reindex(idx), dd["IVV"].reindex(idx)
        ser["tqqq_dd"] = (drawdown(tqqq) * 100).reindex(idx) if tqqq is not None else np.nan
        ser["qqq_rsi14"], ser["ivv_rsi14"], ser["qqq_rsi2"], ser["qqq_pctb"] = r14["QQQ"].reindex(idx), r14["IVV"].reindex(idx), r2["QQQ"].reindex(idx), pbq.reindex(idx)
        ser["qqq_rv20"], ser["qqq_rv60"] = rv20.reindex(idx), rv60.reindex(idx)
        ser["vol_drag"] = 3 * (rv20.reindex(idx) / 100) ** 2 * 100
        ser["tqqq_gap63"] = gap_series.reindex(idx) if gap_series is not None else np.nan
        for name, s_ in (("vix", vix), ("vxn", vxn), ("vix3m", vix3m), ("move", move), ("hy_oas", hy), ("ig_oas", d.fred("BAMLC0A0CM")), ("real10", real10), ("t10y3m", t10y3m), ("t10y2y", t10y2y),
                         ("dollar", dxy), ("wti", wti), ("nfci", nfci), ("gas_weekly", gasw), ("breakeven10", t10ie), ("fwd5y5y", t5)):
            ser[name] = s_.reindex(idx, method="ffill") if s_ is not None else np.nan
        ser["vix_ratio"] = ts_series.reindex(idx, method="ffill") if ts_series is not None else np.nan
        ser["claims4"] = (cl4 / 1000).reindex(idx, method="ffill") if cl4 is not None else np.nan
        ser["net_liquidity"] = nl_series.reindex(idx, method="ffill") if nl_series is not None else np.nan
        for lbl, key in (("spx", "spx"), ("ndx", "ndx")):
            b_ = br[key]
            ser[f"{lbl}_pct200"] = b_["a200"].reindex(idx) if b_ else np.nan
            ser[f"{lbl}_pct50"] = b_["a50"].reindex(idx) if b_ else np.nan
            ser[f"{lbl}_nhnl"] = b_["nhnl"].reindex(idx) if b_ else np.nan
        ser["mcclellan"] = bs["mcc"].reindex(idx) if bs else np.nan
        ser["ad_line"] = bs["adline"].reindex(idx) if bs else np.nan
        ser["rsp_ivv"] = eq["rsp_ivv"][1].reindex(idx) if eq["rsp_ivv"][1] is not None else np.nan
        ser["qqqe_qqq"] = eq["qqqe_qqq"][1].reindex(idx) if eq["qqqe_qqq"][1] is not None else np.nan
        for k in ("xly_xlp", "sphb_splv", "smh_qqq"):
            ser[k] = ratios[k].reindex(idx) if ratios.get(k) is not None else np.nan
        ser["qqq_ivv"] = qi.reindex(idx)
        ser["put_call10"] = pc_series.rolling(10, min_periods=1).mean().reindex(idx, method="ffill") if pc_series is not None else np.nan
        ser["sector_dispersion"] = disp_series.reindex(idx) if disp_series is not None else np.nan
        # historical composite scores (price + vol components) for the chart
        dip_hist = dip_series(qqq.iloc[-900:], None, None, vix, vix3m)
        heat_hist = heat_series(qqq.iloc[-900:], vix)
        ser["dip_score_core"], ser["heat_score_core"] = dip_hist.reindex(idx), heat_hist.reindex(idx)
        ser = ser.replace([np.inf, -np.inf], np.nan)
        series_rows = []
        for ts_, row in ser.iterrows():
            rec = {"date": ts_.strftime("%Y-%m-%d")}
            for k, v_ in row.items():
                rec[k] = None if pd.isna(v_) else round(float(v_), 3)
            series_rows.append(rec)

        # ---------------- backtests -----------------------------------------
        backtest_rows, curve_rows, bt_stats = self.backtests(qqq, tqqq, vix, vix3m, tbill, dip_hist=None)

        # ---------------- events -----------------------------------------
        events = []
        for key, label in (("cpi", "CPI release"), ("fomc", "FOMC decision"), ("nfp", "Payrolls report")):
            ds = sorted(set(x for x in (cal.get(key) or []) + CFG.get("calendar_fallback", {}).get(key, []) if x >= self.as_of.isoformat()))[:3]
            for x in ds:
                events.append({"event": label, "date": x, "days_to": (date.fromisoformat(x) - self.as_of).days, "kind": key})
        for dt_, tk in earn[:8]:
            events.append({"event": f"{tk} earnings", "date": dt_, "days_to": (date.fromisoformat(dt_) - self.as_of).days, "kind": "earnings"})
        events.sort(key=lambda r: r["date"])
        for e in events:
            e["days_to_fmt"] = f"in {e['days_to']} days" if e["days_to"] > 1 else ("tomorrow" if e["days_to"] == 1 else "today")

        # ---------------- sources -----------------------------------------
        src_rows = []
        for name, info in sorted(self.d.manifest.get("sources", {}).items()):
            src_rows.append({"source": name, "status": info.get("status", ""), "last_date": str(info.get("last_date", "") or ""),
                             "fetched_at": str(info.get("fetched_at", "") or "")[:16].replace("T", " "), "rows": info.get("rows", None),
                             "note": str(info.get("error", "") or "")[:160]})
        feed_run = str(self.d.manifest.get("run_at", ""))[:16].replace("T", " ")

        live = sum(1 for m in self.metrics if m["status"] == "live")
        unavailable = [m["name"] for m in self.metrics if m["status"] == "unavailable"]
        summary = {
            "as_of": self.as_of.isoformat(), "data_through": data_through, "feed_run_utc": feed_run,
            "generated_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M %Z"),
            "regime_light": light, "regime_score": round(rs_, 2), "max_tqqq_alloc_pct": max_alloc,
            "dip_score": dip_score, "heat_score": heat_score, "dip_tranches": dip_tranches, "trim_tranches": trim_tranches,
            "action": action,
            "qqq_close": last(qqq), "qqq_chg_1d": change(qqq, 1, pct=True), "qqq_chg_5d": change(qqq, 5, pct=True), "qqq_ytd": (last(qqq) / qqq[qqq.index.year == qqq.index[-1].year].iloc[0] - 1) * 100 if len(qqq) else None,
            "ivv_close": last(ivv), "ivv_chg_1d": change(ivv, 1, pct=True), "ivv_chg_5d": change(ivv, 5, pct=True), "ivv_ytd": (last(ivv) / ivv[ivv.index.year == ivv.index[-1].year].iloc[0] - 1) * 100 if len(ivv) else None,
            "tqqq_close": last(tqqq), "tqqq_chg_1d": change(tqqq, 1, pct=True), "tqqq_chg_5d": change(tqqq, 5, pct=True), "tqqq_ytd": ((last(tqqq) / tqqq[tqqq.index.year == tqqq.index[-1].year].iloc[0] - 1) * 100) if tqqq is not None and len(tqqq) else None,
            "vix": vv, "vix_chg_1d": change(vix, 1), "qqq_dist_200d": tq["dist"], "qqq_drawdown": ddq, "qqq_rsi14": last(r14["QQQ"]), "hy_oas": hyv,
            "metrics_live": live, "metrics_total": len(self.metrics), "metrics_unavailable": "; ".join(unavailable),
            "dip_missing": ", ".join(dip_missing), "heat_missing": ", ".join(heat_missing),
            "takeaway": self.takeaway(light, dip_score, heat_score, tq, ddq, vv, hyv, hy30),
        }
        for k, v_ in list(summary.items()):
            if isinstance(v_, float):
                summary[k] = round(v_, 4)
        fm = {"qqq_close": "{:,.2f}", "ivv_close": "{:,.2f}", "tqqq_close": "{:,.2f}", "vix": "{:.1f}", "hy_oas": "{:.2f}%", "regime_score": "{:+.1f}",
              "dip_score": "{:.0f}", "heat_score": "{:.0f}", "qqq_dist_200d": "{:+.1f}%", "qqq_drawdown": "{:+.1f}%", "qqq_rsi14": "{:.0f}", "vix_chg_1d": "{:+.1f}",
              "max_tqqq_alloc_pct": "{:.0f}%"}
        for k in ("qqq_chg_1d", "qqq_chg_5d", "qqq_ytd", "ivv_chg_1d", "ivv_chg_5d", "ivv_ytd", "tqqq_chg_1d", "tqqq_chg_5d", "tqqq_ytd"):
            fm[k] = "{:+.1f}%"
        for k, spec in fm.items():
            summary[k + "_fmt"] = f(summary.get(k), spec)
        summary["regime_label"] = {"green": "Green", "yellow": "Yellow", "red": "Red"}[light]
        summary["dip_tranches_fmt"] = f"{dip_tranches} of 3"
        summary["trim_tranches_fmt"] = f"{trim_tranches} of 3"
        summary["metrics_live_fmt"] = f"{live} of {len(self.metrics)} metrics live"
        out["summary"] = [summary]
        parts = []
        for p_ in regime_parts:
            parts.append({"key": "regime/" + p_["component"], "composite": "regime", "component": p_["component"], "value": str(p_["value"]), "weight": None, "contribution": p_["points"], "contribution_fmt": f"{p_['points']:+.1f} pts"})
        for p_ in dip_parts:
            parts.append({"key": "dip/" + p_["component"], "composite": "dip", "component": p_["component"], "value": f"{p_['value']:.0f}", "weight": p_["weight"], "contribution": p_["contribution"], "contribution_fmt": f"{p_['contribution']:.1f}"})
        for p_ in heat_parts:
            parts.append({"key": "heat/" + p_["component"], "composite": "heat", "component": p_["component"], "value": f"{p_['value']:.0f}", "weight": p_["weight"], "contribution": p_["contribution"], "contribution_fmt": f"{p_['contribution']:.1f}"})
        out["composite_parts"] = parts
        out["metrics"] = self.metrics
        out["series"] = series_rows
        sec_rows = []
        for r in sectors:
            row = {k: (round(v_, 3) if isinstance(v_, float) else v_) for k, v_ in r.items()}
            for k in ("ret_1m", "ret_3m", "ret_6m", "ret_12m", "rel_1m", "rel_3m", "rel_6m", "rel_12m", "dist_200d"):
                row[k + "_fmt"] = pct(r.get(k))
            row["z_1y_fmt"] = f(r.get("z_1y"), "{:+.1f}σ")
            row["rs_ratio_fmt"] = f(r.get("rs_ratio"), "{:+.1f}")
            row["rs_momentum_fmt"] = f(r.get("rs_momentum"), "{:+.1f}")
            sec_rows.append(row)
        out["sectors"] = sec_rows
        out["backtest"] = backtest_rows
        out["backtest_curve"] = curve_rows
        out["backtest_stats"] = bt_stats
        out["events"] = events
        out["sources"] = src_rows
        out["composites"] = {"regime": {"light": light, "score": round(rs_, 2), "parts": regime_parts, "max_tqqq_alloc_pct": max_alloc},
                             "dip": {"score": dip_score, "parts": dip_parts, "missing": dip_missing, "penalty": round(pen, 1)},
                             "heat": {"score": heat_score, "parts": heat_parts, "missing": heat_missing}}
        return out

    @staticmethod
    def takeaway(light, dip, heat, tq, ddq, vv, hyv, hy30):
        bits = [f"Regime {light}: QQQ {'above' if tq['above'] else 'below'} its 200-day ({pct(tq['dist'])})"]
        if ddq is not None:
            bits.append(f"{pct(ddq)} from the 52-week high")
        if vv is not None:
            bits.append(f"VIX {f(vv)}")
        if hyv is not None:
            bits.append(f"HY spread {pct(hyv, False, 2)}" + (f" ({bp(hy30)} in a month)" if hy30 is not None and abs(hy30) > 25 else ""))
        return "; ".join(bits) + f". Dip score {dip}, heat score {heat}."

    # ---------------------------------------------------------------- backtests
    def backtests(self, qqq: pd.Series, tqqq: pd.Series | None, vix, vix3m, tbill, dip_hist=None):
        cfg = CFG["backtest"]
        horizons, edges = cfg["horizons_days"], cfg["score_buckets"]
        # simulated 3x series (daily rebalanced) to extend TQQQ history before 2010-02
        r = qqq.pct_change()
        tb = (tbill.reindex(qqq.index, method="ffill") / 100 / 252).fillna(0) if tbill is not None else pd.Series(0.0, index=qqq.index)
        sim = (3 * r - 2 * tb - 0.0095 / 252).fillna(0)
        sim_px = 100 * (1 + sim).cumprod()
        if tqqq is not None:
            first = tqqq.index[0]
            scale_ = tqqq.iloc[0] / sim_px.reindex([first], method="ffill").iloc[0]
            ext = pd.concat([sim_px[sim_px.index < first] * scale_, tqqq])
        else:
            ext = sim_px
        dip = dip_series(qqq, None, None, vix, vix3m)
        heat = heat_series(qqq, vix)
        rows = []
        for score_name, sc in (("dip", dip), ("heat", heat)):
            for asset, px in (("QQQ", qqq), ("TQQQ*", ext.reindex(qqq.index))):
                for h in horizons:
                    fwd = (px.shift(-h) / px - 1) * 100
                    df = pd.DataFrame({"s": sc, "f": fwd}).dropna()
                    base_n, base_mean, base_hit = len(df), float(df["f"].mean()), float((df["f"] > 0).mean() * 100)
                    rows.append({"score": score_name, "asset": asset, "horizon": h, "bucket": "all", "lo": 0, "hi": 100, "n": base_n,
                                 "mean_ret": round(base_mean, 2), "median_ret": round(float(df["f"].median()), 2), "hit_rate": round(base_hit, 1), "edge_vs_all": 0.0})
                    for lo, hi in zip(edges[:-1], edges[1:]):
                        sub = df[(df["s"] >= lo) & ((df["s"] < hi) if hi < 100 else (df["s"] <= hi))]
                        if len(sub) < 20:
                            rows.append({"score": score_name, "asset": asset, "horizon": h, "bucket": f"{lo}–{hi}", "lo": lo, "hi": hi, "n": len(sub),
                                         "mean_ret": None, "median_ret": None, "hit_rate": None, "edge_vs_all": None})
                            continue
                        rows.append({"score": score_name, "asset": asset, "horizon": h, "bucket": f"{lo}–{hi}", "lo": lo, "hi": hi, "n": len(sub),
                                     "mean_ret": round(float(sub["f"].mean()), 2), "median_ret": round(float(sub["f"].median()), 2),
                                     "hit_rate": round(float((sub["f"] > 0).mean() * 100), 1), "edge_vs_all": round(float(sub["f"].mean()) - base_mean, 2)})
        for rw_ in rows:
            rw_["mean_ret_fmt"] = pct(rw_["mean_ret"], True, 2)
            rw_["hit_rate_fmt"] = pct(rw_["hit_rate"], False, 0)
            rw_["edge_fmt"] = f(rw_["edge_vs_all"], "{:+.2f} pp")
            rw_["label"] = f"{rw_['score']} {rw_['bucket']}"
        # equity curves: QQQ, TQQQ* buy & hold, TQQQ* held only when QQQ > 200d (else T-bills), dip-ladder variant
        s200 = sma(qqq, 200)
        pos = (qqq > s200).astype(float).shift(1).fillna(0)
        ext_r = ext.reindex(qqq.index).pct_change().fillna(0)
        strat = pos * ext_r + (1 - pos) * tb
        curves = pd.DataFrame({"QQQ": (1 + r.fillna(0)).cumprod(), "TQQQ* buy & hold": (1 + ext_r).cumprod(), "TQQQ* only above 200d": (1 + strat).cumprod()})
        curves = curves[curves.index >= pd.Timestamp("2005-01-01")] if (curves.index >= pd.Timestamp("2005-01-01")).any() else curves
        curves = curves / curves.iloc[0] * 100
        monthly = curves.resample("ME").last()
        curve_rows = [{"date": ts.strftime("%Y-%m-%d"), **{k: round(float(v), 2) for k, v in row.items()}} for ts, row in monthly.iterrows()]
        stats = []
        years = (curves.index[-1] - curves.index[0]).days / 365.25
        for col in curves.columns:
            c = curves[col]
            daily = c.pct_change().dropna()
            cagr = (c.iloc[-1] / c.iloc[0]) ** (1 / years) - 1 if years > 0 else None
            mdd = float((c / c.cummax() - 1).min() * 100)
            vol = float(daily.std(ddof=0) * math.sqrt(252) * 100)
            stats.append({"strategy": col, "cagr": round(cagr * 100, 2) if cagr is not None else None, "cagr_fmt": pct(cagr * 100 if cagr is not None else None, False), "max_drawdown": round(mdd, 1), "max_drawdown_fmt": pct(mdd),
                          "volatility": round(vol, 1), "volatility_fmt": pct(vol, False),
                          "sharpe": round((cagr * 100 - float(tb.mean() * 252 * 100)) / vol, 2) if cagr is not None and vol else None,
                          "sharpe_fmt": f((cagr * 100 - float(tb.mean() * 252 * 100)) / vol if cagr is not None and vol else None, "{:.2f}"),
                          "start": curves.index[0].strftime("%Y-%m-%d"), "end": curves.index[-1].strftime("%Y-%m-%d"),
                          "note": "TQQQ* = actual TQQQ from Feb 2010, simulated 3× daily QQQ (less 2× T-bill financing and 0.95% fee) before."})
        return rows, curve_rows, stats


# ============================================================================ insights, history, scorecard
CATEGORIES = ["regime", "dip_buy", "heat", "sector", "volatility", "breadth", "credit_rates", "macro", "calendar", "risk"]
CATEGORY_LABEL = {"regime": "Regime", "dip_buy": "Dip buy", "heat": "Heat", "sector": "Sectors", "volatility": "Volatility", "breadth": "Breadth",
                  "credit_rates": "Credit & rates", "macro": "Macro", "calendar": "Calendar", "risk": "Risk"}


def validate_insights(doc: dict, as_of: date) -> list[dict]:
    items = doc.get("insights") if isinstance(doc, dict) else doc
    if not isinstance(items, list) or not items:
        raise SystemExit("insights file must hold a non-empty 'insights' list")
    clean = []
    for i, it in enumerate(items[:10], 1):
        pred = it.get("prediction") or {"type": "none"}
        ptype = pred.get("type", "none")
        if ptype not in ("direction", "relative", "level", "none"):
            ptype = "none"
        conf = pred.get("confidence")
        try:
            conf = float(conf) if conf is not None else None
        except Exception:  # noqa: BLE001
            conf = None
        if conf is not None:
            conf = max(0.5, min(0.95, conf))
        h = pred.get("horizon_days")
        h = int(h) if h in (5, 10, 20, 60) or (isinstance(h, (int, float)) and 1 <= int(h) <= 120) else (None if ptype == "none" else 10)
        clean.append({"rank": i, "date": as_of.isoformat(), "category": it.get("category") if it.get("category") in CATEGORIES else "risk",
                      "title": str(it.get("title", ""))[:120], "text": str(it.get("text", ""))[:600], "action": str(it.get("action", ""))[:300],
                      "metric_ids": [str(x) for x in (it.get("metric_ids") or [])][:6],
                      "prediction": {"type": ptype, "asset": str(pred.get("asset", "") or "").upper()[:12], "versus": str(pred.get("versus", "") or "").upper()[:12],
                                     "direction": str(pred.get("direction", "") or "").lower(), "horizon_days": h, "confidence": conf}})
    return clean


def series_for(data: Data, asset: str) -> tuple[pd.Series | None, pd.Series | None]:
    """(open, close) for a price asset, or (None, close) for an index/series."""
    asset = asset.upper()
    if asset in ("VIX", "VXN", "VIX3M", "SKEW"):
        s = data.cboe(asset)
        return None, s
    if asset in ("HY", "HY_OAS", "BAMLH0A0HYM2"):
        return None, data.fred("BAMLH0A0HYM2")
    if asset in ("DGS10", "10Y", "US10Y"):
        return None, data.fred("DGS10")
    if asset in ("DFII10", "REAL10"):
        return None, data.fred("DFII10")
    if asset in ("MOVE", "^MOVE"):
        return None, data.close("^MOVE")
    df = data.prices(asset)
    if df is None:
        return None, None
    return (df["open"] if "open" in df.columns else None), df["adj_close"] if "adj_close" in df.columns else df["close"]


def score_insight(data: Data, ins: dict, today: date) -> dict:
    p = ins["prediction"]
    res = {"status": "n/a", "realized_pct": None, "resolves_on": "", "base": None, "target": None}
    if p["type"] == "none" or not p.get("asset") or not p.get("horizon_days"):
        return res
    h = int(p["horizon_days"])
    d0 = pd.Timestamp(ins["date"])

    def base_and_target(asset):
        op, cl = series_for(data, asset)
        if cl is None or len(cl) < 10:
            return None, None, None, None
        cl = cl.dropna()
        after = cl.index[cl.index >= d0]
        if len(after) == 0:
            return "pending", None, None, None
        b_idx = cl.index.get_loc(after[0])
        base = float(op.iloc[b_idx]) if (op is not None and b_idx < len(op) and not pd.isna(op.iloc[b_idx])) else float(cl.iloc[b_idx - 1]) if b_idx > 0 else float(cl.iloc[b_idx])
        t_idx = b_idx + h - 1
        if t_idx >= len(cl):
            return "pending", base, None, cl.index[b_idx]
        return "ok", base, float(cl.iloc[t_idx]), cl.index[t_idx]

    st, base, tgt, tdate = base_and_target(p["asset"])
    if st is None:
        res["status"] = "unscorable"
        return res
    if st == "pending":
        res["status"] = "pending"
        est = d0 + pd.tseries.offsets.BDay(h)
        res["resolves_on"] = est.strftime("%Y-%m-%d")
        res["base"] = base
        return res
    ret = (tgt / base - 1) * 100
    if p["type"] == "relative" and p.get("versus"):
        st2, b2, t2, _ = base_and_target(p["versus"])
        if st2 != "ok":
            res["status"] = "pending" if st2 == "pending" else "unscorable"
            res["resolves_on"] = (d0 + pd.tseries.offsets.BDay(h)).strftime("%Y-%m-%d")
            return res
        ret = ret - (t2 / b2 - 1) * 100
    direction = p.get("direction", "")
    want_up = direction in ("up", "higher", "outperform", "rise", "bullish")
    want_down = direction in ("down", "lower", "underperform", "fall", "bearish")
    if not (want_up or want_down):
        res["status"] = "unscorable"
        return res
    hit = (ret > 0) if want_up else (ret < 0)
    res.update({"status": "hit" if hit else "miss", "realized_pct": round(ret, 2), "resolves_on": pd.Timestamp(tdate).strftime("%Y-%m-%d"),
                "base": round(base, 4), "target": round(tgt, 4), "signed_pct": round(ret if want_up else -ret, 2)})
    return res


def pred_label(p: dict) -> str:
    t = p.get("type", "none")
    if t == "none" or not p.get("asset"):
        return "context"
    arrow = "↑" if p.get("direction") in ("up", "higher", "outperform", "rise", "bullish") else "↓"
    tgt = f"{p['asset']} vs {p['versus']}" if t == "relative" and p.get("versus") else p["asset"]
    return f"{tgt} {arrow} {p.get('horizon_days')}d"


def build_scorecard(log: list[dict]) -> list[dict]:
    rows = []
    def agg(group_type, group, items):
        resolved = [x for x in items if x["status"] in ("hit", "miss")]
        pending = [x for x in items if x["status"] == "pending"]
        hits = sum(1 for x in resolved if x["status"] == "hit")
        hr = round(hits / len(resolved) * 100, 1) if resolved else None
        signed = [x["signed_pct"] for x in resolved if x.get("signed_pct") is not None]
        confs = [(x["confidence"], 1.0 if x["status"] == "hit" else 0.0) for x in resolved if x.get("confidence") is not None]
        brier = round(sum((c - o) ** 2 for c, o in confs) / len(confs), 3) if confs else None
        rows.append({"group_type": group_type, "group": group, "n_total": len(items), "n_resolved": len(resolved), "n_pending": len(pending),
                     "n_untestable": sum(1 for x in items if x["status"] in ("n/a", "unscorable")), "hits": hits, "hit_rate": hr, "hit_rate_fmt": pct(hr, False, 0),
                     "avg_signed_pct": round(sum(signed) / len(signed), 2) if signed else None, "avg_signed_fmt": f(sum(signed) / len(signed) if signed else None, "{:+.2f}%"),
                     "brier": brier, "brier_fmt": f(brier, "{:.3f}")})
    agg("all", "All calls", log)
    for h in sorted({x["horizon_days"] for x in log if x.get("horizon_days")}):
        agg("horizon", f"{h}-day", [x for x in log if x.get("horizon_days") == h])
    for c in CATEGORIES:
        items = [x for x in log if x["category"] == c]
        if items:
            agg("category", CATEGORY_LABEL.get(c, c), items)
    for lo, hi, lbl in ((0.5, 0.6, "50–60%"), (0.6, 0.7, "60–70%"), (0.7, 0.8, "70–80%"), (0.8, 0.96, "80%+")):
        items = [x for x in log if x.get("confidence") is not None and lo <= x["confidence"] < hi]
        if items:
            agg("confidence", lbl, items)
    for a in sorted({x["asset"] for x in log if x.get("asset")}):
        items = [x for x in log if x.get("asset") == a]
        if len(items) >= 3:
            agg("asset", a, items)
    return rows


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else str(o)))


def brief_text(out: dict, prev_log: list[dict], as_of: date) -> str:
    s = out["summary"][0]
    comp = out["composites"]
    lines = [f"# Morning brief — {as_of.isoformat()} (data through {s['data_through']}, feed run {s['feed_run_utc']} UTC)", "",
             f"Regime light: {s['regime_light'].upper()} (score {s['regime_score']}); dip-buy score {s['dip_score']}; heat score {s['heat_score']}. TQQQ cap {s['max_tqqq_alloc_pct']}%.",
             f"Suggested action: {s['action']}", f"Takeaway: {s['takeaway']}", "",
             f"QQQ {f(s['qqq_close'], '{:.2f}')} ({pct(s['qqq_chg_1d'])} 1d, {pct(s['qqq_chg_5d'])} 5d, {pct(s['qqq_ytd'])} YTD) | IVV {f(s['ivv_close'], '{:.2f}')} ({pct(s['ivv_chg_1d'])} 1d) | TQQQ {f(s['tqqq_close'], '{:.2f}')} ({pct(s['tqqq_chg_1d'])} 1d, {pct(s['tqqq_ytd'])} YTD) | VIX {f(s['vix'])} ({f(s['vix_chg_1d'], '{:+.1f}')})",
             "", "## Composite components"]
    lines.append("Regime: " + "; ".join(f"{p['component']} {p['value']} ({p['points']:+.1f})" for p in comp["regime"]["parts"]))
    lines.append("Dip: " + "; ".join(f"{p['component']} {p['value']} (w{p['weight']})" for p in comp["dip"]["parts"]) + (f"; penalty {comp['dip']['penalty']}" if comp["dip"]["penalty"] else "") + (f"; missing: {', '.join(comp['dip']['missing'])}" if comp["dip"]["missing"] else ""))
    lines.append("Heat: " + "; ".join(f"{p['component']} {p['value']} (w{p['weight']})" for p in comp["heat"]["parts"]) + (f"; missing: {', '.join(comp['heat']['missing'])}" if comp["heat"]["missing"] else ""))
    lines += ["", "## The 50 metrics (id | value | signal | pctile 1y | reading)"]
    cur_layer = None
    for m in out["metrics"]:
        if m["layer"] != cur_layer:
            cur_layer = m["layer"]
            lines.append(f"### {cur_layer}")
        extra = f" | {m['col2']} {m['fmt2']}" if m["fmt2"] else ""
        pz = f" | pct {m['pctile_1y']:.0f}" if m["pctile_1y"] is not None else ""
        lines.append(f"- {m['n']}. {m['id']} [{m['status']}] {m['name']}: {m['col1']} {m['fmt']}{extra} | {m['signal']}{pz} | {m['reading']} {('(' + m['detail'] + ')') if m['detail'] else ''}")
    lines += ["", "## Notable (|z| ≥ 1.5, extreme percentiles, warning states)"]
    notable = [m for m in out["metrics"] if (m["z_1y"] is not None and abs(m["z_1y"]) >= 1.5) or (m["pctile_1y"] is not None and (m["pctile_1y"] <= 10 or m["pctile_1y"] >= 90)) or m["state"] in ("warn", "bad")]
    lines += [f"- {m['id']}: {m['signal']} (z {m['z_1y']}, pct {m['pctile_1y']}) — {m['reading']}" for m in notable] or ["- nothing beyond normal ranges"]
    lines += ["", "## Sectors (3m relative to IVV, pp; quadrant; z vs 1y)"]
    lines += [f"- {r['ticker']} {r['sector']}: rel3m {f(r['rel_3m'], '{:+.1f}')} | rel1m {f(r['rel_1m'], '{:+.1f}')} | {r['quadrant']} | z {f(r['z_1y'], '{:+.1f}')}" for r in out["sectors"]]
    lines += ["", "## Upcoming events"] + [f"- {e['event']} on {e['date']} ({e['days_to']} days)" for e in out["events"][:8]]
    lines += ["", "## Backtest headline (dip score ≥ 60 vs all, 20-day forward)"]
    for r in out["backtest"]:
        if r["horizon"] == 20 and r["bucket"] in ("60–80", "80–100", "all") and r["score"] == "dip":
            lines.append(f"- {r['asset']} dip {r['bucket']}: n={r['n']} mean {f(r['mean_ret'], '{:+.2f}')}% hit {f(r['hit_rate'], '{:.0f}')}% (edge {f(r['edge_vs_all'], '{:+.2f}')})")
    for r in out["backtest"]:
        if r["horizon"] == 20 and r["bucket"] in ("60–80", "80–100") and r["score"] == "heat":
            lines.append(f"- {r['asset']} heat {r['bucket']}: n={r['n']} mean {f(r['mean_ret'], '{:+.2f}')}% hit {f(r['hit_rate'], '{:.0f}')}% (edge {f(r['edge_vs_all'], '{:+.2f}')})")
    if prev_log:
        recent = [x for x in prev_log if x["status"] in ("hit", "miss")][-10:]
        pend = [x for x in prev_log if x["status"] == "pending"]
        lines += ["", f"## Scorecard so far: {sum(1 for x in prev_log if x['status']=='hit')} hits / {sum(1 for x in prev_log if x['status'] in ('hit','miss'))} resolved, {len(pend)} pending"]
        lines += [f"- {x['date']} #{x['rank']} {x['title']}: {x['status']} ({f(x.get('realized_pct'), '{:+.2f}')}%)" for x in recent]
    lines += ["", "## Data quality", f"- {s['metrics_live']}/{s['metrics_total']} metrics live." + (f" Unavailable: {s['metrics_unavailable']}" if s["metrics_unavailable"] else "")]
    failed = [r for r in out["sources"] if r["status"] not in ("ok", "")]
    lines += [f"- Source issues: " + "; ".join(f"{r['source']} ({r['status']})" for r in failed)] if failed else ["- All feed sources fetched."]
    return "\n".join(lines)


# ============================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["compute", "finalize"], default="compute")
    ap.add_argument("--history", default=None, help="previous history.json (asset of the dashboard)")
    ap.add_argument("--insights", default=None, help="today's insights JSON (finalize stage)")
    ap.add_argument("--as-of", default=None)
    args = ap.parse_args()

    repo, out_dir = Path(args.repo), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    as_of = date.fromisoformat(args.as_of) if args.as_of else datetime.now(TZ).date()
    data = Data(repo)
    history = {"version": 1, "days": []}
    if args.history and Path(args.history).exists():
        try:
            history = json.loads(Path(args.history).read_text())
            if "days" not in history:
                history = {"version": 1, "days": []}
        except Exception:  # noqa: BLE001
            history = {"version": 1, "days": []}

    if args.stage == "compute":
        eng = Engine(data, as_of)
        out = eng.run()
        for key in ("summary", "metrics", "series", "sectors", "backtest", "backtest_curve", "backtest_stats", "events", "sources", "composite_parts"):
            write_json(out_dir / f"{key}.json", out[key])
        write_json(out_dir / "composites.json", out["composites"])
        # score prior insights now so the brief can mention them
        prev_log = []
        for day in history["days"]:
            for ins in day.get("insights", []):
                sc = score_insight(data, ins, as_of)
                prev_log.append({**{k: ins[k] for k in ("date", "rank", "category", "title")}, **sc,
                                 "asset": ins["prediction"].get("asset", ""), "horizon_days": ins["prediction"].get("horizon_days"), "confidence": ins["prediction"].get("confidence")})
        (out_dir / "brief.md").write_text(brief_text(out, prev_log, as_of))
        print(f"compute done: {out['summary'][0]['metrics_live']}/{out['summary'][0]['metrics_total']} metrics live; regime {out['summary'][0]['regime_light']}, dip {out['summary'][0]['dip_score']}, heat {out['summary'][0]['heat_score']}")
        print(f"brief: {out_dir / 'brief.md'}")
        return 0

    # finalize
    if not args.insights or not Path(args.insights).exists():
        raise SystemExit("--insights file required for finalize")
    summary = json.loads((out_dir / "summary.json").read_text())[0]
    insights = validate_insights(json.loads(Path(args.insights).read_text()), as_of)
    snapshot = {"date": as_of.isoformat(), "data_through": summary["data_through"], "regime_light": summary["regime_light"], "regime_score": summary["regime_score"],
                "dip_score": summary["dip_score"], "heat_score": summary["heat_score"], "qqq_close": summary["qqq_close"], "ivv_close": summary["ivv_close"],
                "tqqq_close": summary["tqqq_close"], "vix": summary["vix"], "qqq_dist_200d": summary["qqq_dist_200d"], "qqq_drawdown": summary["qqq_drawdown"],
                "hy_oas": summary["hy_oas"], "action": summary["action"], "insights": insights}
    history["days"] = [d_ for d_ in history["days"] if d_["date"] != as_of.isoformat()] + [snapshot]
    history["days"].sort(key=lambda d_: d_["date"])
    write_json(out_dir / "history.json", history)
    log = []
    for day in history["days"]:
        for ins in day.get("insights", []):
            sc = score_insight(data, ins, as_of)
            log.append({"date": ins["date"], "rank": ins["rank"], "category": ins["category"], "category_label": CATEGORY_LABEL.get(ins["category"], ins["category"]), "title": ins["title"], "text": ins["text"], "action": ins["action"],
                        "asset": ins["prediction"].get("asset", ""), "versus": ins["prediction"].get("versus", ""), "direction": ins["prediction"].get("direction", ""),
                        "horizon_days": ins["prediction"].get("horizon_days"), "confidence": ins["prediction"].get("confidence"), "type": ins["prediction"].get("type", "none"),
                        "status": sc["status"], "realized_pct": sc.get("realized_pct"), "signed_pct": sc.get("signed_pct"), "resolves_on": sc.get("resolves_on", ""),
                        "base": sc.get("base"), "target": sc.get("target"), "prediction_fmt": pred_label(ins["prediction"]),
                        "confidence_fmt": f((ins["prediction"].get("confidence") or 0) * 100, "{:.0f}%") if ins["prediction"].get("confidence") else "",
                        "realized_fmt": f(sc.get("realized_pct"), "{:+.2f}%") if sc.get("realized_pct") is not None else ""})
    log.sort(key=lambda x: (x["date"], x["rank"]), reverse=True)
    write_json(out_dir / "insight_log.json", log)
    write_json(out_dir / "scorecard.json", build_scorecard(log))
    today_rows = [{**{k: ins[k] for k in ("rank", "date", "category", "title", "text", "action")}, "category_label": CATEGORY_LABEL.get(ins["category"], ins["category"]), "metric_ids": ", ".join(ins["metric_ids"]),
                   "asset": ins["prediction"].get("asset", ""), "versus": ins["prediction"].get("versus", ""), "direction": ins["prediction"].get("direction", ""),
                   "horizon_days": ins["prediction"].get("horizon_days"), "confidence": ins["prediction"].get("confidence"), "type": ins["prediction"].get("type", "none"),
                   "testable": "yes" if ins["prediction"].get("type", "none") != "none" else "no",
                   "prediction_fmt": pred_label(ins["prediction"]), "confidence_fmt": f((ins["prediction"].get("confidence") or 0) * 100, "{:.0f}%") if ins["prediction"].get("confidence") else ""} for ins in insights]
    write_json(out_dir / "insights.json", today_rows)
    # score-history dataset: composites per day with forward QQQ returns where known (live calibration)
    qqq = data.close("QQQ", adj=True)
    sh = []
    for d_ in history["days"]:
        row = {k: d_.get(k) for k in ("date", "regime_light", "dip_score", "heat_score", "qqq_close", "tqqq_close", "vix")}
        for h in CFG["insight_horizons_days"]:
            row[f"qqq_fwd_{h}d"] = None
            if qqq is not None:
                after = qqq.index[qqq.index >= pd.Timestamp(d_["date"])]
                if len(after):
                    b = qqq.index.get_loc(after[0])
                    if b + h - 1 < len(qqq) and b > 0:
                        row[f"qqq_fwd_{h}d"] = round(float(qqq.iloc[b + h - 1] / qqq.iloc[b - 1] - 1) * 100, 2)
        sh.append(row)
    write_json(out_dir / "score_history.json", sh)
    resolved = [x for x in log if x["status"] in ("hit", "miss")]
    print(f"finalize done: {len(insights)} insights saved for {as_of}; log {len(log)} calls, {len(resolved)} resolved, "
          f"hit rate {f(sum(1 for x in resolved if x['status']=='hit') / len(resolved) * 100 if resolved else None, '{:.0f}')}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
