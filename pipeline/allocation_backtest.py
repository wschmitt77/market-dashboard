#!/usr/bin/env python3
"""
Allocation backtest for a core (IVV/QQQ) + reserve + leveraged-sleeve plan.

    python3 -I pipeline/allocation_backtest.py --repo <repo> --mode bears   [--ivv <$> --cash <$> --tranche-frac 0.25]
    python3 -I pipeline/allocation_backtest.py --repo <repo> --mode rolling [--years 7]

Rules (the "reclaim" rule from glidepath_backtest.py, generalised):
  * Unlevered ladder: tranches of tranche_frac x the reserve at the start of the drawdown episode, bought when QQQ is
    14 / 20 / 27 % below its 2-year high (or SPY 7 / 10 / 15 %), split QQQ/SPY 2/3-1/3, 2/3-1/3, 1/2-1/2.
  * Leveraged sleeve: an L x QQQ fund (simulated: L x daily QQQ - (L-1) x T-bill - fee), sized by how deep the episode
    went, bought only after two weekly closes above the 200-day with 20-day realized vol under 45 %; sold on a weekly
    close 3 % below the 200-day (four-week cool-off); cut to the smallest rung when the drawdown closes to -10 % and to
    zero at -3 %. Sale proceeds go back to the reserve while the drawdown is deeper than -10 %, otherwise into QQQ.
  * Reserve: T-bills, or a mix with Treasuries (IEF; synthetic from the 10-year yield before 2002) and gold
    (GLD; gold futures before Nov 2004), rebalanced daily. An episode ends when QQQ is back within 3 % of its high;
    the ladder re-arms on the reserve that is left.
Variants compare leverage (3x vs 2x at equal exposure), reserve mixes, sizing tables and the starting equity/cash split.
Starting amounts default to round placeholder numbers; pass your own.
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

LADDER = [(-0.14, -0.07, 2 / 3), (-0.20, -0.10, 2 / 3), (-0.27, -0.15, 1 / 2)]  # (QQQ dd, SPY dd, QQQ share of the tranche)
SIZING_30 = [(-0.15, 0.10), (-0.25, 0.20), (-0.35, 0.30)]
SIZING_50 = [(-0.15, 0.10), (-0.25, 0.20), (-0.35, 0.30), (-0.50, 0.50)]
BEARS = {"2000-02": ("2000-03-01", "2003-06-30"), "2007-09": ("2007-10-01", "2009-12-31"), "2020": ("2020-02-15", "2020-09-30"), "2022": ("2021-11-15", "2023-06-30")}


def size_for(deepest: float, table) -> float:
    out = 0.0
    for lvl, share in table:
        if deepest <= lvl:
            out = share
    return out


# ---------------------------------------------------------------------------------------------------------- data
def load(repo: Path) -> dict:
    d = P.Data(repo)
    qqq = d.close("QQQ", adj=True).dropna()
    idx = qqq.index
    spy = d.close("SPY", adj=True).dropna().reindex(idx).ffill()
    # cash rate: 3-month bill (FRED/Treasury), else Yahoo ^IRX
    tb = d.fred("DTB3")
    irx = d.close("^IRX")
    rate = pd.Series(np.nan, index=idx)
    if tb is not None:
        rate = tb.reindex(idx, method="ffill")
    if irx is not None:
        rate = rate.fillna(irx.reindex(idx, method="ffill"))
    first_rate = rate.first_valid_index()
    rate = rate.ffill()
    tbd = (rate / 100 / 252).fillna(0.0)  # before the first rate observation cash earns nothing (reported in the notes)
    rate_note = f"T-bill rate from {first_rate.date() if first_rate is not None else 'n/a'} (DTB3 {'yes' if tb is not None else 'no'}, ^IRX {'yes' if irx is not None else 'no'})"
    # bonds: IEF total return; synthetic 7-10y Treasury from the 10-year yield before it exists
    ief = d.close("IEF", adj=True)
    y10 = d.fred("DGS10")
    tnx = d.close("^TNX")
    bond = pd.Series(np.nan, index=idx)
    if ief is not None:
        bond = ief.reindex(idx).ffill().pct_change()
    y = pd.Series(np.nan, index=idx)
    if y10 is not None:
        y = y10.reindex(idx, method="ffill")
    if tnx is not None:
        y = y.fillna(tnx.reindex(idx, method="ffill"))
    y = y.ffill()
    dur, conv = 7.6, 70.0
    synth = (y.shift(1) / 100 / 252) - dur * (y.diff() / 100) + 0.5 * conv * (y.diff() / 100) ** 2
    bond = bond.fillna(synth).fillna(0.0)
    # gold: GLD, else gold futures; else T-bill (so a mix never breaks)
    gld = d.close("GLD", adj=True)
    gcf = d.close("GC=F")
    gold = pd.Series(np.nan, index=idx)
    if gld is not None:
        gold = gld.reindex(idx).ffill().pct_change()
    if gcf is not None:
        gold = gold.fillna(gcf.reindex(idx).ffill().pct_change())
    gold_first = gold.first_valid_index()
    gold = gold.fillna(tbd)
    r = qqq.pct_change().fillna(0.0)
    data = {"idx": idx, "qqq": qqq.values, "spy": spy.values, "tbd": tbd.values, "r": r.values, "bond": bond.values, "gold": gold.values,
            "hi2y": qqq.rolling(504, min_periods=1).max().values, "hi2y_spy": spy.rolling(504, min_periods=1).max().values,
            "sma200": P.sma(qqq, 200).values, "rv20": (np.log(qqq).diff().rolling(20).std() * math.sqrt(252) * 100).values,
            "notes": [rate_note, f"IEF from {ief.index[0].date() if ief is not None else 'n/a'} (synthetic 10y before)", f"gold from {gold_first.date() if gold_first is not None else 'n/a'}"]}
    data["dd"] = data["qqq"] / data["hi2y"] - 1
    data["dd_spy"] = data["spy"] / data["hi2y_spy"] - 1
    wk = idx.isocalendar().week.values
    data["week_end"] = np.r_[wk[1:] != wk[:-1], True]
    return data


# ---------------------------------------------------------------------------------------------------------- simulation
def simulate(D: dict, i0: int, i1: int, *, ivv0: float, qqq0: float, cash0: float, tranche_frac: float = 0.25, sizing=SIZING_50,
             lev: float = 3.0, fee: float = 0.0095, reserve_mix=None, ladder: bool = True, sleeve: bool = True, fund_core: bool = False) -> dict:
    """Simulate from index i0 to i1 (inclusive). reserve_mix: {"tbill": w, "bond": w, "gold": w}. fund_core: fund the sleeve from
    core holdings (QQQ then SPY) when the reserve is empty, otherwise only the reserve pays for it."""
    qqq, spy, tbd, r, bond, gold = D["qqq"], D["spy"], D["tbd"], D["r"], D["bond"], D["gold"]
    dd, dd_spy, sma200, rv, week_end = D["dd"], D["dd_spy"], D["sma200"], D["rv20"], D["week_end"]
    mix = reserve_mix or {"tbill": 1.0}
    w_tb, w_bd, w_gd = mix.get("tbill", 0.0), mix.get("bond", 0.0), mix.get("gold", 0.0)
    exposure_scale = 3.0 / lev  # a 2x fund needs 1.5x the dollars for the same exposure
    u_spy = ivv0 / spy[i0]
    u_qqq = qqq0 / qqq[i0]
    u_lev, lev_px = 0.0, 100.0
    reserve = cash0
    reserve_base = cash0
    done = set()
    rung = 0.0
    st = {"cooloff": 0, "wk_above200": 0, "deepest": min(0.0, dd[i0]), "glided": False}
    vals = np.empty(i1 - i0 + 1)
    n_trades = 0
    log = []
    for k, i in enumerate(range(i0, i1 + 1)):
        if k > 0:
            lev_px *= 1 + (lev * r[i] - (lev - 1) * tbd[i] - fee / 252)
            lev_px = max(lev_px, 1e-9)
            reserve *= 1 + (w_tb * tbd[i] + w_bd * bond[i] + w_gd * gold[i])
        total = reserve + u_spy * spy[i] + u_qqq * qqq[i] + u_lev * lev_px
        d_now = dd[i]
        st["deepest"] = min(st["deepest"], d_now)
        # episode over: re-arm the ladder on what is left of the reserve
        if d_now > -0.03:
            if done:
                done.clear()
            reserve_base = reserve
            st["deepest"] = 0.0
            st["glided"] = False
        # unlevered ladder
        if ladder:
            for t, (lvl, lvl_spy, q_share) in enumerate(LADDER):
                if t not in done and (d_now <= lvl or dd_spy[i] <= lvl_spy) and reserve > 1000:
                    spend = min(tranche_frac * reserve_base, reserve)
                    u_qqq += spend * q_share / qqq[i]
                    u_spy += spend * (1 - q_share) / spy[i]
                    reserve -= spend
                    done.add(t)
                    log.append((D["idx"][i].date(), f"tranche {t + 1} ${spend:,.0f} (QQQ {d_now * 100:.0f}%)"))
        total = reserve + u_spy * spy[i] + u_qqq * qqq[i] + u_lev * lev_px
        # leveraged sleeve, weekly
        desired = rung
        if sleeve and week_end[i]:
            st["cooloff"] = max(0, st["cooloff"] - 1)
            above = qqq[i] > sma200[i] if not math.isnan(sma200[i]) else False
            st["wk_above200"] = st["wk_above200"] + 1 if above else 0
            size = size_for(st["deepest"], sizing) * exposure_scale
            if rung > 0 and not math.isnan(sma200[i]) and qqq[i] < sma200[i] * 0.97:
                desired = 0.0
                st["cooloff"] = 4
            elif st["wk_above200"] >= 2 and not math.isnan(rv[i]) and rv[i] < 45 and st["cooloff"] == 0 and size > rung and (not st["glided"] or d_now <= -0.15):
                desired = size
            if d_now > -0.10 and desired > 0.10 * exposure_scale:
                desired = 0.10 * exposure_scale
                st["glided"] = True
            if d_now > -0.03 and desired > 0.0:
                desired = 0.0
        if abs(desired - rung) > 0.005:
            want = desired * total
            have = u_lev * lev_px
            delta = want - have
            if delta > 0:
                take = min(delta, reserve)
                reserve -= take
                left = delta - take
                if fund_core and left > 0:
                    sell = min(left, u_qqq * qqq[i])
                    u_qqq -= sell / qqq[i]
                    take += sell
                    left -= sell
                    if left > 0:
                        sell = min(left, u_spy * spy[i])
                        u_spy -= sell / spy[i]
                        take += sell
                u_lev += take / lev_px
                if take > 0:
                    log.append((D["idx"][i].date(), f"sleeve -> {desired * 100:.0f}% (buy ${take:,.0f}; QQQ {d_now * 100:.0f}%)"))
                    n_trades += 1
                rung = (u_lev * lev_px) / total if total > 0 else 0.0  # what we could actually afford
            else:
                sell = min(-delta, have)
                u_lev -= sell / lev_px
                if d_now < -0.10:
                    reserve += sell
                else:
                    u_qqq += sell / qqq[i]
                if sell > 0:
                    log.append((D["idx"][i].date(), f"sleeve -> {desired * 100:.0f}% (sell ${sell:,.0f}; QQQ {d_now * 100:.0f}%)"))
                    n_trades += 1
                rung = desired
        vals[k] = reserve + u_spy * spy[i] + u_qqq * qqq[i] + u_lev * lev_px
    start_val = ivv0 + qqq0 + cash0
    v = pd.Series(vals, index=D["idx"][i0:i1 + 1])
    peak = v.cummax()
    mdd = float((v / peak - 1).min() * 100)
    years = (i1 - i0) / 252
    after_min = v[v.idxmin():]
    be = after_min[after_min >= start_val]
    out = {"final_pct": (v.iloc[-1] / start_val - 1) * 100, "cagr": ((v.iloc[-1] / start_val) ** (1 / years) - 1) * 100 if years > 0 else float("nan"),
           "min_pct": (v.min() / start_val - 1) * 100, "min_date": v.idxmin().date(), "max_dd": mdd,
           "breakeven": (be.index[0] - v.index[0]).days / 365.25 if not be.empty else float("nan"), "trades": n_trades, "log": log, "path": v,
           "end_reserve_pct": reserve / v.iloc[-1] * 100, "end_lev_pct": u_lev * lev_px / v.iloc[-1] * 100,
           "positions": {"spy_usd": u_spy * spy[i1], "qqq_usd": u_qqq * qqq[i1], "lev_usd": u_lev * lev_px, "reserve_usd": reserve, "total": float(vals[-1]),
                         "tranches_done": sorted(done), "rung": rung, "state": dict(st), "reserve_base": reserve_base}}
    for yv in (1, 2, 3, 5):
        n = 252 * yv
        out[f"y{yv}"] = (v.iloc[n] / start_val - 1) * 100 if n < len(v) else float("nan")
    return out


# ---------------------------------------------------------------------------------------------------------- variants
def variants(a) -> dict:
    eq = a.ivv + a.cash  # total
    base = dict(ivv0=a.ivv, qqq0=0.0, cash0=a.cash, tranche_frac=a.tranche_frac)
    V = {
        "Z hold 43/57, nothing":                dict(base, ladder=False, sleeve=False),
        "A ladder only":                        dict(base, sleeve=False),
        "E ladder + 3x sleeve (30% cap)":       dict(base, sizing=SIZING_30, lev=3.0, fund_core=True),
        "F ladder + 3x sleeve (50% at -50%)":   dict(base, sizing=SIZING_50, lev=3.0, fund_core=True),
        "F2 same exposure via 2x fund":         dict(base, sizing=SIZING_50, lev=2.0, fund_core=True),
        "F2, sleeve paid from reserve only":    dict(base, sizing=SIZING_50, lev=2.0, fund_core=False),
        "F2 + reserve 70/30 bills/IEF":         dict(base, sizing=SIZING_50, lev=2.0, fund_core=True, reserve_mix={"tbill": 0.7, "bond": 0.3}),
        "F2 + reserve 70/30 bills/gold":        dict(base, sizing=SIZING_50, lev=2.0, fund_core=True, reserve_mix={"tbill": 0.7, "gold": 0.3}),
        "F2 + reserve 60/20/20 bills/IEF/gold": dict(base, sizing=SIZING_50, lev=2.0, fund_core=True, reserve_mix={"tbill": 0.6, "bond": 0.2, "gold": 0.2}),
        "FI 60/40 IVV/QQQ fully invested":      dict(ivv0=0.6 * eq, qqq0=0.4 * eq, cash0=0.0, ladder=False, sleeve=False),
        "FI + 2x sleeve funded from core":      dict(ivv0=0.6 * eq, qqq0=0.4 * eq, cash0=0.0, ladder=False, sizing=SIZING_50, lev=2.0, fund_core=True),
        "70/30 split + F2":                     dict(ivv0=0.7 * eq, qqq0=0.0, cash0=0.3 * eq, tranche_frac=a.tranche_frac, sizing=SIZING_50, lev=2.0, fund_core=True),
    }
    return V


def run_bears(D, a):
    idx = D["idx"]
    dd = pd.Series(D["dd"], index=idx)
    V = variants(a)
    for name, (s, e) in BEARS.items():
        seg = dd[s:e]
        hits = seg[seg <= a.start_dd]
        if hits.empty:
            print(f"\n{name}: QQQ never reached {a.start_dd:.0%}")
            continue
        start = hits.index[0]
        i0 = idx.get_loc(start)
        i1 = min(i0 + int(252 * a.years), len(idx) - 1)
        print(f"\n=== {name} bear: start {start.date()} ({dd[start] * 100:.0f}% from the 2y high); trough {seg.min() * 100:.0f}%; window to {idx[i1].date()} ===")
        rows = []
        for vn, kw in V.items():
            res = simulate(D, i0, i1, **kw)
            rows.append({"variant": vn, "1y": res["y1"], "2y": res["y2"], "3y": res["y3"], "5y": res["y5"], "end": res["final_pct"], "CAGR": res["cagr"],
                         "worst vs start": res["min_pct"], "breakeven yrs": res["breakeven"], "trades": res["trades"]})
            if a.verbose and vn.startswith("F2 same"):
                for t, m in res["log"]:
                    print(f"     {t} {m}")
        print(pd.DataFrame(rows).set_index("variant").round(1).to_string())


def run_rolling(D, a):
    idx = D["idx"]
    V = variants(a)
    horizon = int(252 * a.years)
    # first trading day of each month, from the first date with a 200-day average
    first_ok = int(np.argmax(~np.isnan(D["sma200"])))
    months = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).first()
    starts = [idx.get_loc(t) for t in months if idx.get_loc(t) >= max(first_ok, 252) and idx.get_loc(t) + horizon < len(idx)]
    print(f"\n=== rolling {a.years:.0f}-year windows, {len(starts)} monthly starts {idx[starts[0]].date()} .. {idx[starts[-1]].date()} ===")
    table = {}
    paths = {}
    for vn, kw in V.items():
        fin, mdd, cagr = [], [], []
        for i0 in starts:
            res = simulate(D, i0, i0 + horizon, **kw)
            fin.append(res["final_pct"]); mdd.append(res["max_dd"]); cagr.append(res["cagr"])
        fin, mdd, cagr = np.array(fin), np.array(mdd), np.array(cagr)
        table[vn] = {"median CAGR": np.median(cagr), "10th pct CAGR": np.percentile(cagr, 10), "90th pct CAGR": np.percentile(cagr, 90), "worst window": fin.min(),
                     "median max DD": np.median(mdd), "worst max DD": mdd.min(), "% windows > hold": np.nan}
        paths[vn] = cagr
    hold = paths["Z hold 43/57, nothing"]
    for vn in table:
        table[vn]["% windows > hold"] = float((paths[vn] > hold).mean() * 100)
    df = pd.DataFrame(table).T
    print(df.round(1).to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--mode", choices=["bears", "rolling", "both"], default="both")
    ap.add_argument("--ivv", type=float, default=1_000_000)
    ap.add_argument("--cash", type=float, default=1_300_000)
    ap.add_argument("--tranche-frac", type=float, default=0.25)
    ap.add_argument("--start-dd", type=float, default=-0.14)
    ap.add_argument("--years", type=float, default=7.0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    D = load(Path(a.repo))
    print("data:", "; ".join(D["notes"]))
    pd.set_option("display.width", 250)
    if a.mode in ("bears", "both"):
        run_bears(D, a)
    if a.mode in ("rolling", "both"):
        run_rolling(D, a)


if __name__ == "__main__":
    main()
