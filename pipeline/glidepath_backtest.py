#!/usr/bin/env python3
"""
Backtest of the bear-market glide path for a QQQ/IVV investor with a TQQQ sleeve.

    python3 -I pipeline/glidepath_backtest.py --repo <repo root> [--start-dd -0.14] [--ivv <$> --cash <$> --tranche-frac 0.25]

Portfolio starts with IVV (proxied by SPY, which has history to 1993) and cash earning the 3-month T-bill; pass your own
starting amounts on the command line (the defaults are round placeholder numbers). Unlevered tranches, each a fraction of
the starting cash, buy QQQ/SPY at drawdown levels, blind. The TQQQ sleeve follows a target share that scales with
QQQ's drawdown from its running (2-year) peak and is only raised when a confirmation holds; it is cut back one rung
when the exit rule fires. TQQQ is simulated as 3x daily QQQ less financing (2x T-bill) and a 0.95% fee, which tracks
the real fund closely since 2010 and is the only way to see 2000-02 and 2008.

Strategies compared on every bear:
  A  unlevered ladder only
  B  depth-only glide (no confirmation, no exit rule)
  C  all-in: 100% TQQQ at the first -50% close (the original idea), unlevered ladder before that
  D  confirmed glide path (depth x 50/200-day confirmation x vol scaling, weekly decisions)
  E  unlevered ladder, then TQQQ only after two weekly closes above the 200-day, sized by how deep the bear went
  F  same as E with a 50% rung once the bear has reached -50%
  Z  hold (no tranches, no TQQQ): the benchmark
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

RUNGS = [(-0.15, 0.10), (-0.25, 0.15), (-0.35, 0.30), (-0.50, 0.45)]  # (drawdown at or below, target TQQQ share of portfolio)
TRANCHES = [(-0.14, 2 / 3), (-0.20, 2 / 3), (-0.27, 1 / 2)]  # (QQQ drawdown that fires the tranche, QQQ share of it; the rest buys SPY)
SPY_TRIGGER = {-0.14: -0.07, -0.20: -0.10, -0.27: -0.15}     # the same tranche also fires on this SPY drawdown
BEARS = {"2000-02": ("2000-03-01", "2003-06-30"), "2007-09": ("2007-10-01", "2009-12-31"), "2020": ("2020-02-15", "2020-09-30"), "2022": ("2021-11-15", "2023-06-30")}


def target_share(dd: float) -> float:
    t = 0.0
    for lvl, share in RUNGS:
        if dd <= lvl:
            t = share
    return t


SIZING_E = [(-0.15, 0.10), (-0.25, 0.20), (-0.35, 0.30)]           # (deepest drawdown reached, TQQQ share once the 200-day is reclaimed)
SIZING_F = [(-0.15, 0.10), (-0.25, 0.20), (-0.35, 0.30), (-0.50, 0.50)]  # same, but a 50% rung for a -50% bear
NAMES = {"Z": "Z hold (no tranches, no TQQQ)", "A": "A unlevered ladder", "B": "B depth-only glide", "C": "C all-in TQQQ at -50%",
         "D": "D vol-scaled confirmed glide", "E": "E TQQQ after 200d reclaim (weekly)", "F": "F same, 50% rung at -50%",
         "G": "G = E, only once the 200d has stopped rising"}


def size_for(deepest: float, table) -> float:
    out = 0.0
    for lvl, share in table:
        if deepest <= lvl:
            out = share
    return out


def simulate(data: dict, start: pd.Timestamp, end: pd.Timestamp, strategy: str, ivv0: float, cash0: float, verbose: bool = False, tranche_gap: int = 0, tranche_frac: float = 0.25) -> dict:
    qqq, spy, vix, tbd = data["qqq"], data["spy"], data["vix"], data["tbd"]
    r3 = data["r3"]
    idx = qqq[start:end].index
    hi = data["hi2y"]; sma50 = data["sma50"]; sma200 = data["sma200"]; rv = data["rv20"]
    pos = {"spy": ivv0 / spy[start], "qqq": 0.0, "tqqq": 0.0}  # units; tqqq in "simulated fund" units priced by r3 path
    tq_px = 100.0
    cash = cash0
    done_tranches = set()
    current_rung = 0.0
    allin_done = False
    log = []
    values = []
    prev_tq = tq_px
    st = {"above50_streak": 0, "cooloff": 0, "wk_above200": 0, "deepest": 0.0, "last_tranche_i": -10_000, "glided": False}
    sma200_lag = sma200.shift(20)
    for i, t in enumerate(idx):
        # advance prices
        tq_px = prev_tq * (1 + r3[t]) if i > 0 else tq_px
        prev_tq = tq_px
        if i > 0:
            cash *= (1 + tbd[t])
        total = cash + pos["spy"] * spy[t] + pos["qqq"] * qqq[t] + pos["tqqq"] * tq_px
        dd = qqq[t] / hi[t] - 1
        dd_spy = spy[t] / data["hi2y_spy"][t] - 1
        week_end = (i + 1 >= len(idx)) or (idx[i + 1].isocalendar()[1] != t.isocalendar()[1])
        # ---- unlevered tranches (all strategies but Z)
        if strategy != "Z":
            for k, (lvl, q_share) in enumerate(TRANCHES):
                if k not in done_tranches and (dd <= lvl or dd_spy <= SPY_TRIGGER[lvl]) and cash > 1000 and i - st["last_tranche_i"] >= tranche_gap:
                    st["last_tranche_i"] = i
                    spend = min(tranche_frac * cash0, cash)
                    q_spend, s_spend = spend * q_share, spend * (1 - q_share)
                    pos["qqq"] += q_spend / qqq[t]; pos["spy"] += s_spend / spy[t]; cash -= spend
                    done_tranches.add(k)
                    log.append((t.date(), f"tranche {k + 1}: ${spend:,.0f} (QQQ dd {dd * 100:.0f}%)"))
        total = cash + pos["spy"] * spy[t] + pos["qqq"] * qqq[t] + pos["tqqq"] * tq_px
        # ---- leveraged sleeve
        st["above50_streak"] = st["above50_streak"] + 1 if qqq[t] > sma50[t] else 0
        st["deepest"] = min(st["deepest"], dd)
        desired = current_rung
        if strategy in ("Z", "A"):
            desired = 0.0
        elif strategy == "C":
            desired = 1.0 if (dd <= -0.5 or allin_done) else 0.0
            if dd <= -0.5:
                allin_done = True
        elif strategy == "B":
            desired = target_share(dd)
        elif strategy == "D" and week_end:
            # depth target x trend confirmation x vol scaling, decided weekly, 5-point rebalance band
            st["cooloff"] = max(0, st["cooloff"] - 1)
            tgt = target_share(dd)
            volmult = min(1.0, 25.0 / rv[t]) if rv[t] and not math.isnan(rv[t]) else 0.0
            big_ok = qqq[t] > sma200[t]
            small_ok = st["above50_streak"] >= 10
            allowed = (tgt if big_ok else (0.15 if small_ok else 0.0)) if tgt >= 0.30 else (tgt if small_ok else 0.0)
            allowed = allowed * volmult if rv[t] < 45 else 0.0
            exit_small = qqq[t] < sma50[t] * 0.95
            exit_big = qqq[t] < sma200[t]
            if current_rung > 0 and (exit_small or (current_rung >= 0.30 and exit_big)):
                desired = 0.0 if exit_big else 0.5 * current_rung
                st["cooloff"] = 4
            elif allowed > current_rung + 0.05 and st["cooloff"] == 0:
                desired = allowed
            elif allowed < current_rung - 0.05:
                desired = allowed
            desired = round(desired, 2)
        elif strategy in ("E", "F", "G") and week_end:
            # unlevered buying at depth; TQQQ only after two weekly closes above the 200-day, sized by how deep the bear went;
            # out on a weekly close 3% below the 200-day, four weeks' cool-off; glide out as the drawdown closes
            table = SIZING_F if strategy == "F" else SIZING_E
            trend_ok = True if strategy != "G" else bool(sma200[t] <= sma200_lag[t])
            st["cooloff"] = max(0, st["cooloff"] - 1)
            st["wk_above200"] = st["wk_above200"] + 1 if qqq[t] > sma200[t] else 0
            size = size_for(st["deepest"], table)
            if current_rung > 0 and qqq[t] < sma200[t] * 0.97:
                desired = 0.0
                st["cooloff"] = 4
            elif st["wk_above200"] >= 2 and rv[t] < 45 and st["cooloff"] == 0 and size > current_rung and trend_ok and (not st["glided"] or dd <= -0.15):
                desired = size
            if dd > -0.10 and desired > 0.10:
                desired = 0.10          # profit-take: the drawdown has mostly closed
                st["glided"] = True
            if dd > -0.03 and desired > 0.0:
                desired = 0.0
            if dd > -0.03:
                st["deepest"] = 0.0; st["glided"] = False
        if abs(desired - current_rung) > 0.005:
            want = desired * total
            have = pos["tqqq"] * tq_px
            delta = want - have
            if delta > 0:
                # fund from cash, then from QQQ, then SPY
                take = min(delta, cash); cash -= take; delta_left = delta - take
                if delta_left > 0 and pos["qqq"] * qqq[t] > 0:
                    sell = min(delta_left, pos["qqq"] * qqq[t]); pos["qqq"] -= sell / qqq[t]; delta_left -= sell; take += sell
                if delta_left > 0 and pos["spy"] * spy[t] > 0:
                    sell = min(delta_left, pos["spy"] * spy[t]); pos["spy"] -= sell / spy[t]; delta_left -= sell; take += sell
                pos["tqqq"] += take / tq_px
                log.append((t.date(), f"TQQQ -> {desired * 100:.0f}% (buy ${take:,.0f}; QQQ dd {dd * 100:.0f}%, vol {rv[t]:.0f}%)"))
            else:
                sell = min(-delta, have); pos["tqqq"] -= sell / tq_px
                # proceeds: back to cash while the bear is on (dry powder), into QQQ once the drawdown has mostly closed
                if dd < -0.10:
                    cash += sell; where = "cash"
                else:
                    pos["qqq"] += sell / qqq[t]; where = "QQQ"
                log.append((t.date(), f"TQQQ -> {desired * 100:.0f}% (sell ${sell:,.0f} into {where}; QQQ dd {dd * 100:.0f}%)"))
            current_rung = desired
        total = cash + pos["spy"] * spy[t] + pos["qqq"] * qqq[t] + pos["tqqq"] * tq_px
        values.append(total)
    v = pd.Series(values, index=idx)
    start_val = ivv0 + cash0
    after_min = v[v.idxmin():]
    be = after_min[after_min >= start_val]
    out = {"strategy": strategy, "start": start.date(), "end": end.date(), "final": v.iloc[-1], "final_pct": (v.iloc[-1] / start_val - 1) * 100,
           "min": v.min(), "min_pct": (v.min() / start_val - 1) * 100, "min_date": v.idxmin().date(),
           "breakeven": (be.index[0] - start).days / 365.25 if not be.empty else float("nan"),
           "n_tqqq_trades": sum(1 for _, m in log if m.startswith("TQQQ")), "log": log, "path": v}
    for y in (1, 2, 3, 5):
        n = 252 * y
        out[f"y{y}_pct"] = (v.iloc[n] / start_val - 1) * 100 if n < len(v) else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--start-dd", type=float, default=-0.14, help="start each test at the first close at or below this QQQ drawdown")
    ap.add_argument("--ivv", type=float, default=1_000_000, help="starting IVV (SPY proxy) value")
    ap.add_argument("--cash", type=float, default=1_000_000, help="starting cash")
    ap.add_argument("--tranche-frac", type=float, default=0.25, help="each unlevered tranche as a fraction of starting cash")
    ap.add_argument("--years", type=float, default=7.0)
    ap.add_argument("--tranche-gap", type=int, default=0, help="minimum trading days between unlevered tranches")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    d = P.Data(Path(a.repo))
    qqq = d.close("QQQ", adj=True).dropna(); spy = d.close("SPY", adj=True).dropna().reindex(qqq.index).ffill()
    vix = d.cboe("VIX").reindex(qqq.index).ffill()
    tb = d.fred("DTB3"); tbd = (tb.reindex(qqq.index, method="ffill") / 100 / 252).fillna(0.0)
    r = qqq.pct_change().fillna(0.0)
    data = {"qqq": qqq, "spy": spy, "vix": vix, "tbd": tbd, "r3": (3 * r - 2 * tbd - 0.0095 / 252).clip(lower=-1.0),
            "hi2y": qqq.rolling(504, min_periods=1).max(), "hi2y_spy": spy.rolling(504, min_periods=1).max(),
            "sma50": P.sma(qqq, 50), "sma200": P.sma(qqq, 200), "rv20": (np.log(qqq).diff().rolling(20).std() * math.sqrt(252) * 100)}
    dd = qqq / data["hi2y"] - 1
    pd.set_option("display.width", 220)
    for name, (s, e) in BEARS.items():
        seg = dd[s:e]
        hits = seg[seg <= a.start_dd]
        if hits.empty:
            print(f"\n{name}: QQQ never reached {a.start_dd:.0%} in this window"); continue
        start = hits.index[0]
        end = min(start + pd.Timedelta(days=int(365.25 * a.years)), qqq.index[-1])
        trough = qqq[start:e].idxmin()
        print(f"\n=== {name} bear: start {start.date()} (QQQ {qqq[start]:.2f}, {dd[start] * 100:.0f}% from peak); trough {trough.date()} at {(qqq[trough] / data['hi2y'][start] - 1) * 100:.0f}% from peak; window to {end.date()} ===")
        rows = []
        results = {}
        for strat in NAMES:
            res = simulate(data, start, end, strat, a.ivv, a.cash, tranche_gap=a.tranche_gap, tranche_frac=a.tranche_frac)
            results[strat] = res
            rows.append({"strategy": NAMES[strat],
                         "1y": res["y1_pct"], "2y": res["y2_pct"], "3y": res["y3_pct"], "5y": res["y5_pct"], "end": res["final_pct"],
                         "worst": res["min_pct"], "worst date": res["min_date"], "breakeven yrs": res["breakeven"], "TQQQ trades": res["n_tqqq_trades"]})
        df = pd.DataFrame(rows).set_index("strategy")
        print(df.round(1).to_string())
        if a.verbose:
            for strat in ("E", "G"):
                print(f"  -- {strat} trade log --")
                for t, m in results[strat]["log"]:
                    print(f"     {t} {m}")
    # today's standing of the glide path
    t = qqq.index[-1]
    ep = dd[dd > -0.03].index[-1] if (dd > -0.03).any() else dd.index[0]  # start of the current drawdown episode
    deepest = dd[ep:].min()
    print(f"\nToday: QQQ {qqq[t]:.2f}, drawdown from 2y peak {dd[t] * 100:.1f}% (deepest this episode {deepest * 100:.1f}% on {dd[ep:].idxmin().date()}), 50d {data['sma50'][t]:.1f}, 200d {data['sma200'][t]:.1f}, 20d vol {data['rv20'][t]:.0f}%")
    print(f"  rule E: TQQQ size once two weekly closes are back above the 200-day = {size_for(deepest, SIZING_E) * 100:.0f}% (above 200d now: {bool(qqq[t] > data['sma200'][t])})")


if __name__ == "__main__":
    main()
