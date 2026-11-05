#!/usr/bin/env python3
"""
Exposure plan: a trend- and volatility-scaled leveraged position in the Nasdaq-100 (or whichever of QQQ / S&P leads),
tested against holding 100% QQQ and 100% SPY.

    python3 -I pipeline/exposure_backtest.py --repo <repo> [--years 7] [--asset QQQ] [--mode rolling|bears|both]

The rules (decided on weekly closes, executed at the close):
  * Core asset: QQQ, or the stronger of QQQ and SPY by 12-month-minus-1-month return, checked monthly ("MOM").
  * Target exposure E (times the portfolio) from the trend and volatility of the core asset:
      - above its 200-day and 20-day realized vol under vol_hi   -> e_hi   (leveraged)
      - above its 200-day, vol at or above vol_hi               -> e_mid  (usually 1.0: fully invested, no leverage)
      - below its 200-day (3% hysteresis to re-enter)           -> e_lo   (part defensive)
      - after a bear of at least 25% (deepest close), two weekly closes back above the 200-day add reclaim_bonus until the
        drawdown closes to -10%
      - optional ladder: while below the 200-day, each drawdown level adds exposure (buying the dip unlevered)
  * E above 1 is built with the core asset plus a 2x fund on it (QLD for QQQ, SSO for SPY), simulated as 2x daily minus
    financing (T-bill) minus a 0.95% fee; E below 1 leaves the rest in a defensive mix (T-bills, or bills/IEF/GLD).
  * Rebalance only when the live exposure drifts more than `band` from the target.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P  # noqa: E402
import allocation_backtest as AB  # noqa: E402

BEARS = AB.BEARS


def load(repo: Path, asset: str = "QQQ") -> dict:
    """Arrays for the asset under test (A), SPY, cash rate and defensive-mix returns, all on A's calendar."""
    base = AB.load(repo, asset)  # qqq-role = asset; also spy, tbd, bond, gold
    d = P.Data(repo)
    idx = base["idx"]
    spy = pd.Series(base["spy"], index=idx)
    a = pd.Series(base["qqq"], index=idx)
    out = dict(base)
    out["ra"] = a.pct_change().fillna(0.0).values
    out["rs"] = spy.pct_change().fillna(0.0).values
    out["sma200_a"] = out["sma200"]
    out["sma200_s"] = P.sma(spy, 200).values
    out["rv_a"] = out["rv20"]
    out["rv_s"] = (np.log(spy).diff().rolling(20).std() * math.sqrt(252) * 100).values
    out["hi_a"], out["hi_s"] = out["hi2y"], out["hi2y_spy"]
    out["dd_a"], out["dd_s"] = out["dd"], out["dd_spy"]
    # 12-1 momentum: return over the past 252 sessions excluding the latest 21
    mom_a = (a.shift(21) / a.shift(252) - 1).values
    mom_s = (spy.shift(21) / spy.shift(252) - 1).values
    out["mom_a"], out["mom_s"] = mom_a, mom_s
    month_end = np.r_[idx[1:].month != idx[:-1].month, True]
    out["month_end"] = month_end
    return out


def simulate(D: dict, i0: int, i1: int, *, start_val: float = 1_000_000.0, core: str = "A", e_hi: float = 1.5, e_mid: float = 1.0, e_lo: float = 0.6,
             vol_hi: float = 30.0, exit_pct: float = 0.03, reclaim_bonus: float = 0.5, reclaim_depth: float = -0.25, ladder=None,
             lev: float = 2.0, fee: float = 0.0095, defensive=None, band: float = 0.10, constant: float | None = None, vol_crash: float = 60.0) -> dict:
    """core: 'A' (the asset under test), 'S' (SPY) or 'MOM' (monthly 12-1 switch between them). constant: fixed exposure, no rules."""
    idx = D["idx"]
    tbd, bond, gold, week_end, month_end = D["tbd"], D["bond"], D["gold"], D["week_end"], D["month_end"]
    mix = defensive or {"tbill": 1.0}
    w_tb, w_bd, w_gd = mix.get("tbill", 0.0), mix.get("bond", 0.0), mix.get("gold", 0.0)
    ladder = ladder or []
    px = {"A": D["qqq"], "S": D["spy"]}
    r = {"A": D["ra"], "S": D["rs"]}
    sma = {"A": D["sma200_a"], "S": D["sma200_s"]}
    rv = {"A": D["rv_a"], "S": D["rv_s"]}
    dd = {"A": D["dd_a"], "S": D["dd_s"]}
    pick = "S" if core == "S" else "A"
    if core == "MOM":
        ma, ms = D["mom_a"][i0], D["mom_s"][i0]
        pick = "A" if (math.isnan(ma) or math.isnan(ms) or ma >= ms) else "S"
    units_core = 0.0
    units_lev, lev_px = 0.0, 100.0
    defensive_usd = start_val
    trend_up = px[pick][i0] > sma[pick][i0] if not math.isnan(sma[pick][i0]) else True
    st = {"deepest": min(0.0, dd[pick][i0]), "wk_above": 0, "reclaim_on": False, "cooloff": 0}
    target = None
    vals = np.empty(i1 - i0 + 1)
    trades = 0
    exposures = []
    for k, i in enumerate(range(i0, i1 + 1)):
        if k > 0:
            lev_px *= 1 + (lev * r[pick][i] - (lev - 1) * tbd[i] - fee / 252)
            lev_px = max(lev_px, 1e-9)
            defensive_usd *= 1 + (w_tb * tbd[i] + w_bd * bond[i] + w_gd * gold[i])
        total = units_core * px[pick][i] + units_lev * lev_px + defensive_usd
        # monthly core switch
        if core == "MOM" and month_end[i]:
            ma, ms = D["mom_a"][i], D["mom_s"][i]
            new_pick = pick if (math.isnan(ma) or math.isnan(ms)) else ("A" if ma >= ms else "S")
            if new_pick != pick:
                # liquidate into the defensive sleeve, re-establish on the new asset at the same exposure
                defensive_usd = total
                units_core, units_lev, lev_px = 0.0, 0.0, 100.0
                pick = new_pick
                trend_up = px[pick][i] > sma[pick][i] if not math.isnan(sma[pick][i]) else True
                st["deepest"] = min(0.0, dd[pick][i]); st["wk_above"] = 0; st["reclaim_on"] = False
                target = None  # force a rebalance below
                trades += 1
        d_now = dd[pick][i]
        st["deepest"] = min(st["deepest"], d_now)
        if d_now > -0.03:
            st["deepest"] = 0.0
            st["reclaim_on"] = False
        if constant is not None:
            tgt = constant
            decide = week_end[i]
        else:
            decide = week_end[i]
            if decide:
                s200 = sma[pick][i]
                if not math.isnan(s200):
                    if trend_up and px[pick][i] < s200 * (1 - exit_pct):
                        trend_up = False
                    elif not trend_up and px[pick][i] > s200:
                        trend_up = True
                above = (not math.isnan(s200)) and px[pick][i] > s200
                st["wk_above"] = st["wk_above"] + 1 if above else 0
                if st["deepest"] <= reclaim_depth and st["wk_above"] >= 2 and d_now <= -0.10:
                    st["reclaim_on"] = True
                if d_now > -0.10 or not trend_up:
                    st["reclaim_on"] = False
                v = rv[pick][i]
                if trend_up:
                    tgt = e_hi if (not math.isnan(v) and v < vol_hi) else e_mid
                else:
                    tgt = e_lo
                    for lvl, add in ladder:
                        if d_now <= lvl:
                            tgt += add
                    tgt = min(tgt, 1.0)
                if st["reclaim_on"]:
                    tgt = min(2.0, tgt + reclaim_bonus)
                if not math.isnan(v) and v >= vol_crash:
                    tgt = min(tgt, e_lo)  # crash regime: no leverage whatever the trend
            else:
                tgt = target if target is not None else (e_hi if trend_up else e_lo)
        live = (units_core * px[pick][i] + lev * units_lev * lev_px) / total if total > 0 else 0.0
        if decide and (target is None or abs(live - tgt) > band or abs(tgt - (target or 0)) > 1e-9 and abs(live - tgt) > 0.02):
            target = tgt
            if tgt >= 1.0:
                lev_share = min(1.0, (tgt - 1.0) / (lev - 1.0))
                core_share = 1.0 - lev_share
                def_share = 0.0
            else:
                lev_share, core_share, def_share = 0.0, tgt, 1.0 - tgt
            units_core = core_share * total / px[pick][i]
            units_lev = lev_share * total / lev_px
            defensive_usd = def_share * total
            trades += 1
        vals[k] = units_core * px[pick][i] + units_lev * lev_px + defensive_usd
        exposures.append((units_core * px[pick][i] + lev * units_lev * lev_px) / vals[k] if vals[k] > 0 else 0.0)
    v = pd.Series(vals, index=idx[i0:i1 + 1])
    years = (i1 - i0) / 252
    peak = v.cummax()
    return {"final_pct": (v.iloc[-1] / start_val - 1) * 100, "cagr": ((v.iloc[-1] / start_val) ** (1 / years) - 1) * 100 if years > 0 else float("nan"),
            "max_dd": float((v / peak - 1).min() * 100), "min_pct": (v.min() / start_val - 1) * 100, "trades": trades, "avg_exposure": float(np.mean(exposures)),
            "path": v, "end_pick": pick, "end_target": target}


def replay_state(D: dict, i0: int, i1: int, *, e_hi: float = 1.5, e_mid: float = 1.0, e_lo: float = 0.6, vol_hi: float = 30.0, vol_crash: float = 60.0,
                 exit_pct: float = 0.03, reclaim_bonus: float = 0.5, reclaim_depth: float = -0.25, pick: str = "A", decision_mask=None) -> dict:
    """The weekly decision state of the exposure rules at i1, replayed on prices only from i0 (no money involved).
    decision_mask: boolean array of decision days (default: the week-end flags)."""
    px, sma, rv, dd = D["qqq"] if pick == "A" else D["spy"], D["sma200_a"] if pick == "A" else D["sma200_s"], D["rv_a"] if pick == "A" else D["rv_s"], D["dd_a"] if pick == "A" else D["dd_s"]
    week_end = D["week_end"] if decision_mask is None else decision_mask
    trend_up = px[i0] > sma[i0] if not math.isnan(sma[i0]) else True
    st = {"deepest": min(0.0, dd[i0]), "wk_above": 0, "reclaim_on": False}
    target, last_decision, reason = None, None, ""
    for i in range(i0, i1 + 1):
        st["deepest"] = min(st["deepest"], dd[i])
        if dd[i] > -0.03:
            st["deepest"] = 0.0
            st["reclaim_on"] = False
        if not week_end[i]:
            continue
        s200 = sma[i]
        if not math.isnan(s200):
            if trend_up and px[i] < s200 * (1 - exit_pct):
                trend_up = False
            elif not trend_up and px[i] > s200:
                trend_up = True
        above = (not math.isnan(s200)) and px[i] > s200
        st["wk_above"] = st["wk_above"] + 1 if above else 0
        if st["deepest"] <= reclaim_depth and st["wk_above"] >= 2 and dd[i] <= -0.10:
            st["reclaim_on"] = True
        if dd[i] > -0.10 or not trend_up:
            st["reclaim_on"] = False
        v = rv[i]
        if trend_up:
            if not math.isnan(v) and v < vol_hi:
                target, reason = e_hi, f"above the {'200'}-day with realized vol {v:.0f}% under {vol_hi:.0f}%"
            else:
                target, reason = e_mid, f"above the 200-day but realized vol {v:.0f}% is at or above {vol_hi:.0f}%"
        else:
            target, reason = e_lo, "below the 200-day (needs a weekly close back above it)"
        if st["reclaim_on"]:
            target = min(2.0, target + reclaim_bonus); reason += f"; reclaim bonus +{reclaim_bonus:.2f} after a {st['deepest'] * 100:.0f}% bear"
        if not math.isnan(v) and v >= vol_crash:
            target = min(target, e_lo); reason += f"; vol {v:.0f}% ≥ {vol_crash:.0f}%: no leverage"
        last_decision = D["idx"][i]
    return {"target": target, "trend_up": trend_up, "wk_above": st["wk_above"], "deepest": st["deepest"], "reclaim_on": st["reclaim_on"], "last_decision": last_decision, "reason": reason}


def windows(D, years):
    idx = D["idx"]
    horizon = int(252 * years)
    first_ok = int(np.argmax(~np.isnan(D["sma200_a"]) & ~np.isnan(D["mom_a"])))
    months = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).first()
    starts = [idx.get_loc(t) for t in months if idx.get_loc(t) >= max(first_ok, 252) and idx.get_loc(t) + horizon < len(idx)]
    return starts, horizon


def rolling(D, starts, horizon, **kw):
    c, m, f, e = [], [], [], []
    for i0 in starts:
        res = simulate(D, i0, i0 + horizon, **kw)
        c.append(res["cagr"]); m.append(res["max_dd"]); f.append(res["final_pct"]); e.append(res["avg_exposure"])
    return np.array(c), np.array(m), np.array(f), np.array(e)


def variants() -> dict:
    L = [(-0.14, 0.15), (-0.20, 0.15), (-0.27, 0.10)]
    V = {
        "100% QQQ": dict(constant=1.0),
        "100% SPY": dict(core="S", constant=1.0),
        "constant 1.5x QQQ (no rules)": dict(constant=1.5),
        "T1  trend 1.5x / 1.0 / 0.6": dict(e_hi=1.5, e_mid=1.0, e_lo=0.6),
        "T2  trend 1.5x / 1.0 / 0.6 + ladder": dict(e_hi=1.5, e_mid=1.0, e_lo=0.6, ladder=L),
        "T3  trend 1.75x / 1.0 / 0.5": dict(e_hi=1.75, e_mid=1.0, e_lo=0.5),
        "T4  trend 2.0x / 1.0 / 0.5": dict(e_hi=2.0, e_mid=1.0, e_lo=0.5),
        "T5  trend 1.5x / 1.0 / 1.0 (never defensive)": dict(e_hi=1.5, e_mid=1.0, e_lo=1.0),
        "T6  trend 1.5x / 1.2 / 0.6, vol cap 40": dict(e_hi=1.5, e_mid=1.2, e_lo=0.6, vol_hi=40),
        "T7  trend 1.5x / 1.0 / 0.3": dict(e_hi=1.5, e_mid=1.0, e_lo=0.3),
        "M1  MOM core, trend 1.5x / 1.0 / 0.6": dict(core="MOM", e_hi=1.5, e_mid=1.0, e_lo=0.6),
        "M2  MOM core, trend 1.5x / 1.0 / 0.6 + ladder": dict(core="MOM", e_hi=1.5, e_mid=1.0, e_lo=0.6, ladder=L),
        "M3  MOM core, trend 1.75x / 1.0 / 0.5": dict(core="MOM", e_hi=1.75, e_mid=1.0, e_lo=0.5),
        "D1  T1 with defensive 60/20/20 bills/IEF/gold": dict(e_hi=1.5, e_mid=1.0, e_lo=0.6, defensive={"tbill": 0.6, "bond": 0.2, "gold": 0.2}),
        "D2  M1 with defensive 60/20/20": dict(core="MOM", e_hi=1.5, e_mid=1.0, e_lo=0.6, defensive={"tbill": 0.6, "bond": 0.2, "gold": 0.2}),
        "X1  T1 with 3x fund": dict(e_hi=1.5, e_mid=1.0, e_lo=0.6, lev=3.0),
    }
    return V


def run_rolling(D, a):
    starts, horizon = windows(D, a.years)
    print(f"\n=== rolling {a.years:.0f}-year windows on {a.asset}: {len(starts)} monthly starts {D['idx'][starts[0]].date()} .. {D['idx'][starts[-1]].date()} ===")
    res = {}
    for name, kw in variants().items():
        res[name] = rolling(D, starts, horizon, **kw)
    q, s = res["100% QQQ"][0], res["100% SPY"][0]
    rows = []
    for name, (c, m, f, e) in res.items():
        rows.append({"variant": name, "median CAGR": np.median(c), "10th pct": np.percentile(c, 10), "worst window": f.min(), "median maxDD": np.median(m), "worst maxDD": m.min(),
                     "> QQQ %": float((c > q).mean() * 100), "med. excess vs QQQ": np.median(c - q), "> SPY %": float((c > s).mean() * 100), "med. excess vs SPY": np.median(c - s),
                     "> both %": float(((c > q) & (c > s)).mean() * 100), "avg exposure": np.mean(e)})
    df = pd.DataFrame(rows).set_index("variant")
    print(df.round(1).to_string())
    return df


def run_bears(D, a):
    idx = D["idx"]
    dd = pd.Series(D["dd_a"], index=idx)
    for name, (s, e) in BEARS.items():
        seg = dd[s:e]
        hits = seg[seg <= -0.14]
        if hits.empty:
            continue
        i0 = idx.get_loc(hits.index[0]); i1 = min(i0 + int(252 * a.years), len(idx) - 1)
        print(f"\n=== {name} bear from {idx[i0].date()} ({dd.iloc[i0] * 100:.0f}%), window to {idx[i1].date()} ===")
        rows = []
        for vn, kw in variants().items():
            r = simulate(D, i0, i1, **kw)
            rows.append({"variant": vn, "end": r["final_pct"], "CAGR": r["cagr"], "worst vs start": r["min_pct"], "max DD": r["max_dd"], "trades": r["trades"], "avg exposure": r["avg_exposure"]})
        print(pd.DataFrame(rows).set_index("variant").round(1).to_string())


CHOSEN = dict(e_hi=1.5, e_mid=1.0, e_lo=0.5, vol_hi=30.0, exit_pct=0.03, reclaim_bonus=0.5, lev=2.0, defensive={"tbill": 0.6, "bond": 0.2, "gold": 0.2})


def run_robust(repo: Path, a):
    """The chosen rules on other assets (vs 100% of that asset) and one-at-a-time parameter changes on QQQ."""
    print("\n=== chosen rules on other assets: trend-scaled 1.5x/1.0/0.6 on the asset vs holding 100% of it (core stays that asset; SPY is the switch alternative only in MOM mode) ===")
    rows = []
    for asset in ["QQQ", "SPY", "IWM", "XLK", "XLF", "XLE", "XLV", "XLY", "SMH"]:
        try:
            Da = load(repo, asset)
        except Exception as e:  # noqa: BLE001
            print(f"  {asset}: no data ({e})"); continue
        st, hz = windows(Da, a.years)
        if len(st) < 24:
            continue
        c, m, f, e = rolling(Da, st, hz, **CHOSEN)
        b, bm, bf, _ = rolling(Da, st, hz, constant=1.0)
        s_, sm, sf, _ = rolling(Da, st, hz, core="S", constant=1.0)
        rows.append({"asset": asset, "windows": len(st), "rules CAGR": np.median(c), "100% asset CAGR": np.median(b), "> asset %": float((c > b).mean() * 100), "med. excess": np.median(c - b),
                     "> SPY %": float((c > s_).mean() * 100), "rules 10th pct": np.percentile(c, 10), "asset 10th pct": np.percentile(b, 10), "rules worst DD": m.min(), "asset worst DD": bm.min()})
    print(pd.DataFrame(rows).set_index("asset").round(1).to_string())
    D = load(repo, "QQQ")
    starts, horizon = windows(D, a.years)
    q = rolling(D, starts, horizon, constant=1.0)[0]
    s_ = rolling(D, starts, horizon, core="S", constant=1.0)[0]
    print("\n=== one change at a time on QQQ ===")
    variants_ = {"chosen": {}, "e_hi 1.3": {"e_hi": 1.3}, "e_hi 1.75": {"e_hi": 1.75}, "e_mid 0.8": {"e_mid": 0.8}, "e_mid 1.2": {"e_mid": 1.2}, "e_lo 0.4": {"e_lo": 0.4}, "e_lo 0.8": {"e_lo": 0.8},
                 "vol_hi 25": {"vol_hi": 25.0}, "vol_hi 35": {"vol_hi": 35.0}, "vol_hi 45": {"vol_hi": 45.0}, "exit 1%": {"exit_pct": 0.01}, "exit 5%": {"exit_pct": 0.05},
                 "no reclaim bonus": {"reclaim_bonus": 0.0}, "reclaim bonus 0.25": {"reclaim_bonus": 0.25}, "band 5": {"band": 0.05}, "band 15": {"band": 0.15},
                 "defensive bills only": {"defensive": {"tbill": 1.0}}, "3x fund": {"lev": 3.0}, "150-day instead of 200-day": {"_sma": 150}, "250-day instead of 200-day": {"_sma": 250}}
    rows = []
    for name, ch in variants_.items():
        kw = dict(CHOSEN); ch = dict(ch)
        Dv = D
        if "_sma" in ch:
            n = ch.pop("_sma"); Dv = dict(D)
            qa = pd.Series(D["qqq"], index=D["idx"]); Dv["sma200_a"] = P.sma(qa, n).values
        kw.update(ch)
        c, m, f, e = rolling(Dv, starts, horizon, **kw)
        rows.append({"variant": name, "median CAGR": np.median(c), "10th pct": np.percentile(c, 10), "worst window": f.min(), "worst maxDD": m.min(), "> QQQ %": float((c > q).mean() * 100), "> SPY %": float((c > s_).mean() * 100), "med. excess vs QQQ": np.median(c - q)})
    print(pd.DataFrame(rows).set_index("variant").round(1).to_string())
    # bull-only windows: starts 2009-07 onward (no 2000/2008 bear inside)
    late = [i for i in starts if D["idx"][i] >= pd.Timestamp("2009-07-01")]
    c, m, f, e = rolling(D, late, horizon, **CHOSEN); qb = rolling(D, late, horizon, constant=1.0)[0]
    print(f"\nwindows starting 2009-07 .. {D['idx'][late[-1]].date()} ({len(late)}, mostly bull): rules median {np.median(c):.1f}% vs QQQ {np.median(qb):.1f}%, rules ahead in {(c > qb).mean() * 100:.0f}%, median excess {np.median(c - qb):+.1f}%/yr, worst maxDD {m.min():.0f}% vs QQQ {rolling(D, late, horizon, constant=1.0)[1].min():.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--asset", default="QQQ")
    ap.add_argument("--years", type=float, default=7.0)
    ap.add_argument("--mode", choices=["rolling", "bears", "both", "robust"], default="both")
    a = ap.parse_args()
    pd.set_option("display.width", 260)
    if a.mode == "robust":
        run_robust(Path(a.repo), a); return
    D = load(Path(a.repo), a.asset)
    print("data:", "; ".join(D["notes"]))
    if a.mode in ("rolling", "both"):
        run_rolling(D, a)
    if a.mode in ("bears", "both"):
        run_bears(D, a)


if __name__ == "__main__":
    main()
