#!/usr/bin/env python3
"""
How much to trust the plan: benchmarks, pseudo-out-of-sample assets and parameter sensitivity.

    python3 -I pipeline/robustness.py --repo <repo> [--ivv <$> --cash <$> --tranche-frac 0.23] [--years 7]

1. Rolling windows on QQQ: the rules vs holding the start mix, vs 100% SPY, 100% QQQ and 60/40 SPY/QQQ fully invested.
2. The same rules, untouched, applied to other assets (SPY, IWM, sector ETFs) with a simulated 2x fund on that asset:
   if the edge is structural it should survive; if it is a QQQ artefact it should not.
3. One-at-a-time parameter changes on QQQ: ladder levels, rung sizes, weekly-close count, exit distance, vol cap, reserve mix.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import allocation_backtest as AB  # noqa: E402


def windows(D, years):
    idx = D["idx"]
    horizon = int(252 * years)
    first_ok = int(np.argmax(~np.isnan(D["sma200"])))
    months = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).first()
    return [idx.get_loc(t) for t in months if idx.get_loc(t) >= max(first_ok, 252) and idx.get_loc(t) + horizon < len(idx)], horizon


def rolling(D, starts, horizon, **kw):
    cagr, mdd, fin = [], [], []
    for i0 in starts:
        r = AB.simulate(D, i0, i0 + horizon, **kw)
        cagr.append(r["cagr"]); mdd.append(r["max_dd"]); fin.append(r["final_pct"])
    return np.array(cagr), np.array(mdd), np.array(fin)


def row(name, c, m, f, ref=None):
    out = {"strategy": name, "median CAGR": np.median(c), "10th pct": np.percentile(c, 10), "90th pct": np.percentile(c, 90), "worst window": f.min(), "median maxDD": np.median(m), "worst maxDD": m.min()}
    if ref is not None:
        d = c - ref
        out["vs rules: median diff"] = -np.median(d)  # positive = rules ahead
        out["rules ahead %"] = float((d < 0).mean() * 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ivv", type=float, default=1_000_000)
    ap.add_argument("--cash", type=float, default=1_300_000)
    ap.add_argument("--tranche-frac", type=float, default=0.25)
    ap.add_argument("--years", type=float, default=7.0)
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    eq = a.ivv + a.cash
    RULES = dict(ivv0=a.ivv, qqq0=0.0, cash0=a.cash, tranche_frac=a.tranche_frac, sizing=AB.SIZING_50, lev=2.0, fund_core=True, reserve_mix={"tbill": 0.6, "bond": 0.2, "gold": 0.2})

    # ---------------------------------------------------------------- 1. benchmarks on QQQ
    D = AB.load(Path(a.repo), "QQQ")
    starts, horizon = windows(D, a.years)
    print(f"\n=== 1. QQQ, {len(starts)} rolling {a.years:.0f}-year windows {D['idx'][starts[0]].date()} .. {D['idx'][starts[-1]].date()} ===")
    rc, rm, rf = rolling(D, starts, horizon, **RULES)
    rows = [row("Rules (ladder + QLD sleeve + 60/20/20 reserve)", rc, rm, rf)]
    for name, kw in [("Hold start mix", dict(ivv0=a.ivv, qqq0=0.0, cash0=a.cash, ladder=False, sleeve=False)),
                     ("100% SPY", dict(ivv0=eq, qqq0=0.0, cash0=0.0, ladder=False, sleeve=False)),
                     ("100% QQQ", dict(ivv0=0.0, qqq0=eq, cash0=0.0, ladder=False, sleeve=False)),
                     ("60/40 SPY/QQQ", dict(ivv0=0.6 * eq, qqq0=0.4 * eq, cash0=0.0, ladder=False, sleeve=False)),
                     ("Rules, sleeve off (ladder + reserve only)", dict(RULES, sleeve=False)),
                     ("Rules, ladder off (sleeve + reserve only)", dict(RULES, ladder=False))]:
        c, m, f = rolling(D, starts, horizon, **kw)
        rows.append(row(name, c, m, f, ref=rc if not name.startswith("Rules") else None))
    print(pd.DataFrame(rows).set_index("strategy").round(1).to_string())

    # ---------------------------------------------------------------- 2. other assets, same rules
    print(f"\n=== 2. Same rules on other assets (ladder buys the asset, sleeve = simulated 2x of it; core stays SPY) ===")
    rows = []
    for asset in ["QQQ", "SPY", "IWM", "XLK", "XLF", "XLE", "XLV", "XLY", "SMH"]:
        try:
            Da = AB.load(Path(a.repo), asset)
        except Exception as e:  # noqa: BLE001
            print(f"  {asset}: no data ({e})"); continue
        st, hz = windows(Da, a.years)
        if len(st) < 24:
            print(f"  {asset}: only {len(st)} windows, skipped"); continue
        c, m, f = rolling(Da, st, hz, **RULES)
        h, hm, hf = rolling(Da, st, hz, ivv0=a.ivv, qqq0=0.0, cash0=a.cash, ladder=False, sleeve=False)
        fi, fim, fif = rolling(Da, st, hz, ivv0=0.6 * eq, qqq0=0.4 * eq, cash0=0.0, ladder=False, sleeve=False)
        rows.append({"asset": asset, "windows": len(st), "from": str(Da["idx"][st[0]].date()), "rules CAGR": np.median(c), "hold CAGR": np.median(h), "60/40 CAGR": np.median(fi),
                     "rules > hold %": float((c > h).mean() * 100), "rules > 60/40 %": float((c > fi).mean() * 100), "rules 10th pct": np.percentile(c, 10), "60/40 10th pct": np.percentile(fi, 10),
                     "rules worst DD": m.min(), "60/40 worst DD": fim.min()})
    print(pd.DataFrame(rows).set_index("asset").round(1).to_string())

    # ---------------------------------------------------------------- 3. parameter sensitivity on QQQ
    print(f"\n=== 3. One change at a time (QQQ): does the result depend on the exact parameters? ===")
    base_ladder, base_sizing = list(AB.LADDER), list(AB.SIZING_50)
    variants = {
        "base": {},
        "ladder shallower (-11/-17/-24)": {"ladder": [(-0.11, -0.055, 2 / 3), (-0.17, -0.085, 2 / 3), (-0.24, -0.13, 1 / 2)]},
        "ladder deeper (-17/-24/-32)": {"ladder": [(-0.17, -0.085, 2 / 3), (-0.24, -0.12, 2 / 3), (-0.32, -0.18, 1 / 2)]},
        "ladder all IVV": {"ladder": [(l, s, 0.0) for l, s, _ in base_ladder]},
        "ladder all QQQ": {"ladder": [(l, s, 1.0) for l, s, _ in base_ladder]},
        "rungs x0.67 (10/20/30/50 QLD)": {"sizing": [(l, s * 0.67) for l, s in base_sizing]},
        "rungs x1.33": {"sizing": [(l, s * 1.33) for l, s in base_sizing]},
        "no 50% rung (cap 30%)": {"sizing": AB.SIZING_30},
        "1 weekly close to enter": {"weeks": 1},
        "3 weekly closes to enter": {"weeks": 3},
        "exit 2% below 200d": {"exit_pct": 0.02},
        "exit 5% below 200d": {"exit_pct": 0.05},
        "vol cap 35%": {"vol_cap": 35.0},
        "vol cap 60%": {"vol_cap": 60.0},
        "reserve all bills": {"reserve_mix": {"tbill": 1.0}},
        "reserve 50/50 bills/IEF": {"reserve_mix": {"tbill": 0.5, "bond": 0.5}},
        "3x fund (TQQQ) instead of 2x": {"lev": 3.0},
        "tranche 15% of reserve": {"tranche_frac": 0.15},
        "tranche 33% of reserve": {"tranche_frac": 0.33},
    }
    h, hm, hf = rolling(D, starts, horizon, ivv0=a.ivv, qqq0=0.0, cash0=a.cash, ladder=False, sleeve=False)
    fi, fim, fif = rolling(D, starts, horizon, ivv0=0.6 * eq, qqq0=0.4 * eq, cash0=0.0, ladder=False, sleeve=False)
    rows = []
    for name, ch in variants.items():
        kw = dict(RULES)
        ladder = ch.pop("ladder", None) if "ladder" in ch else None
        extra = {}
        for k in ("weeks", "exit_pct", "vol_cap"):
            if k in ch:
                extra[k] = ch.pop(k)
        kw.update(ch)
        old_ladder = AB.LADDER
        if ladder is not None:
            AB.LADDER = ladder
        c, m, f = rolling(D, starts, horizon, **kw, **extra)
        AB.LADDER = old_ladder
        rows.append({"variant": name, "median CAGR": np.median(c), "10th pct": np.percentile(c, 10), "worst window": f.min(), "worst maxDD": m.min(), "> hold %": float((c > h).mean() * 100), "> 60/40 %": float((c > fi).mean() * 100)})
    print(pd.DataFrame(rows).set_index("variant").round(1).to_string())


if __name__ == "__main__":
    main()
