#!/usr/bin/env python3
"""
Recommended allocation for today: the plan's rules applied to the portfolio on file.

    python3 -I pipeline/plan.py --repo <repo> --portfolio portfolio.json --out out [--no-backtest]

Writes out/allocation.json (one row per fund: target, held, trade) and out/plan.json (one row per rule state and note).
The portfolio file is private (it lives in the dashboard's store, not in the repo):

    {"as_of": "2026-11-04",
     "holdings": {"IVV": 100000, "QQQ": 0, "QLD": 0, "TQQQ": 0, "TBILL": 100000, "IEF": 0, "GLD": 0},   # dollars on as_of (example)
     "baseline": {... the same shape: what was held when the current drawdown episode began ...},
     "rules": {"tranche_frac": 0.23, "vehicle": "QLD", "reserve_mix": {"tbill": 0.6, "bond": 0.2, "gold": 0.2}}}

Rules (see allocation_backtest.py for the tests behind them):
  * Episode: starts the day after QQQ was last within 3% of its 2-year high; the baseline is what was held then.
  * Ladder: tranches of tranche_frac x the baseline reserve, due when QQQ closes 14 / 20 / 27 % below its 2-year high
    (or IVV/SPY 7 / 10 / 15 %), split QQQ/IVV 2/3-1/3, 2/3-1/3, 1/2-1/2.
  * Leveraged sleeve: QLD (2x) at 15 / 30 / 45 / 75 % of the portfolio by the deepest close of the episode
    (-15 / -25 / -35 / -50 %), or TQQQ at two-thirds of those dollars, bought only after two weekly closes above the
    200-day with realized vol under 45 %, sold on a weekly close 3 % below the 200-day (four-week cool-off), cut to the
    smallest rung once the drawdown closes to -10 % and to zero at -3 %. Paid for from the reserve, then QQQ, then IVV.
  * Reserve: 60 % T-bills, 20 % IEF, 20 % GLD (configurable).
Nothing here is advice: these are the owner's own rules, applied mechanically.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P  # noqa: E402
import allocation_backtest as AB  # noqa: E402
import exposure_backtest as X  # noqa: E402

DEFAULT_RULES = {"tranche_frac": 0.23, "vehicle": "QLD", "reserve_mix": {"tbill": 0.6, "bond": 0.2, "gold": 0.2},
                 "sizing": AB.SIZING_50, "ladder": AB.LADDER, "fund_core": True}
EXPOSURE_RULES = {"plan": "exposure", "e_hi": 1.5, "e_mid": 1.0, "e_lo": 0.5, "vol_hi": 30.0, "vol_crash": 60.0, "exit_pct": 0.03, "reclaim_bonus": 0.5,
                  "reclaim_depth": -0.25, "band": 0.10, "vehicle": "QLD", "defensive_mix": {"tbill": 0.6, "bond": 0.2, "gold": 0.2}}
VEHICLE_LEV = {"QLD": 2.0, "TQQQ": 3.0}
FUND_NAME = {"IVV": "iShares Core S&P 500", "QQQ": "Invesco Nasdaq-100", "QLD": "ProShares Ultra QQQ (2x)", "TQQQ": "ProShares UltraPro QQQ (3x)",
             "TBILL": "T-bills / money market (SGOV, BIL)", "IEF": "iShares 7–10 Year Treasury", "GLD": "SPDR Gold"}
ROLE = {"IVV": "core", "QQQ": "core", "QLD": "leveraged sleeve", "TQQQ": "leveraged sleeve", "TBILL": "reserve", "IEF": "reserve", "GLD": "reserve"}


def usd(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    sign = "−" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000:
        return f"{sign}${x / 1_000_000:.2f}M"
    if x >= 1000:
        return f"{sign}${x / 1000:,.0f}k"
    return f"{sign}${x:,.0f}"


def pct(x: float, d: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}%"


def sleeve_state(D: dict, i0: int, i1: int, sizing, exposure_scale: float) -> dict:
    """Replay the sleeve rule on prices only, from the episode start to the last close."""
    qqq, dd, sma200, rv, week_end = D["qqq"], D["dd"], D["sma200"], D["rv20"], D["week_end"]
    st = {"cooloff": 0, "wk_above200": 0, "deepest": min(0.0, dd[i0]), "glided": False}
    rung = 0.0
    events = []
    for i in range(i0, i1 + 1):
        st["deepest"] = min(st["deepest"], dd[i])
        if not week_end[i]:
            continue
        st["cooloff"] = max(0, st["cooloff"] - 1)
        above = (not math.isnan(sma200[i])) and qqq[i] > sma200[i]
        st["wk_above200"] = st["wk_above200"] + 1 if above else 0
        size = AB.size_for(st["deepest"], sizing) * exposure_scale
        desired = rung
        if rung > 0 and not math.isnan(sma200[i]) and qqq[i] < sma200[i] * 0.97:
            desired = 0.0
            st["cooloff"] = 4
            events.append((D["idx"][i].date(), "exit: weekly close 3% below the 200-day"))
        elif st["wk_above200"] >= 2 and not math.isnan(rv[i]) and rv[i] < 45 and st["cooloff"] == 0 and size > rung and (not st["glided"] or dd[i] <= -0.15):
            desired = size
            events.append((D["idx"][i].date(), f"entry: sleeve to {size * 100:.0f}% after the 200-day reclaim"))
        if dd[i] > -0.10 and desired > 0.10 * exposure_scale:
            desired = 0.10 * exposure_scale
            st["glided"] = True
        if dd[i] > -0.03 and desired > 0:
            desired = 0.0
        rung = desired
    # what the rule would allow at the next weekly close if nothing changes
    size_now = AB.size_for(st["deepest"], sizing) * exposure_scale
    return {"rung": rung, "size_if_confirmed": size_now, "events": events, **st}


def build(repo: Path, portfolio: dict, run_backtest: bool = True) -> tuple[list[dict], list[dict]]:
    if (portfolio.get("rules") or {}).get("plan") == "exposure":
        return build_exposure(repo, portfolio, run_backtest)
    rules = dict(DEFAULT_RULES)
    rules.update(portfolio.get("rules") or {})
    sizing = [tuple(x) for x in rules["sizing"]]
    ladder = [tuple(x) for x in rules["ladder"]]
    vehicle = rules["vehicle"] if rules["vehicle"] in VEHICLE_LEV else "QLD"
    lev = VEHICLE_LEV[vehicle]
    exposure_scale = 3.0 / lev
    mix = rules["reserve_mix"]
    data = P.Data(repo)
    D = AB.load(repo)
    idx = D["idx"]
    i1 = len(idx) - 1
    last = idx[i1]
    dd_s = pd.Series(D["dd"], index=idx)
    # ---- episode
    if portfolio.get("episode_start"):
        i0 = idx.get_indexer([pd.Timestamp(portfolio["episode_start"])], method="bfill")[0]
    else:
        ok = dd_s[dd_s > -0.03]
        i0 = idx.get_loc(ok.index[-1]) + 1 if len(ok) else 0
        i0 = min(i0, i1)
    ep_start = idx[i0]
    deepest_i = int(np.argmin(D["dd"][i0:i1 + 1])) + i0
    deepest = float(D["dd"][deepest_i])
    deepest_spy = float(np.min(D["dd_spy"][i0:i1 + 1]))
    hi = float(D["hi2y"][i1])
    hi_date = idx[int(np.argmax(D["qqq"][max(0, i1 - 503):i1 + 1])) + max(0, i1 - 503)]
    qqq_last = float(D["qqq"][i1])
    dd_last = float(D["dd"][i1])
    sma200 = float(D["sma200"][i1])
    rv = float(D["rv20"][i1])
    # ---- prices for marking the portfolio (actual closes, not adjusted, for the fund rows)
    closes = {}
    for t in ("IVV", "QQQ", "QLD", "TQQQ", "IEF", "GLD", "BIL", "SPY"):
        s = data.close(t)
        closes[t] = s if s is not None and len(s) else None
    as_of = pd.Timestamp(portfolio.get("as_of") or last)
    tbill = D["tbd"]

    def mtm(holdings: dict) -> dict:
        out = {}
        for t, v in (holdings or {}).items():
            v = float(v or 0)
            if t == "TBILL":
                j0 = idx.get_indexer([as_of], method="ffill")[0]
                out[t] = v * float(np.prod(1 + tbill[j0 + 1:i1 + 1])) if j0 < i1 else v
            elif closes.get(t) is not None:
                s = closes[t]
                p0 = float(s[:as_of].iloc[-1]) if len(s[:as_of]) else float(s.iloc[-1])
                out[t] = v / p0 * float(s.iloc[-1])
            else:
                out[t] = v
        return out

    held = mtm(portfolio.get("holdings") or {})
    base = mtm(portfolio.get("baseline") or portfolio.get("holdings") or {})
    reserve_keys = ("TBILL", "IEF", "GLD")
    base_reserve = sum(base.get(k, 0.0) for k in reserve_keys)
    reserve_base = float(portfolio.get("reserve_base") or base_reserve)
    tranche_usd = rules["tranche_frac"] * reserve_base
    total_held = sum(held.values())
    # ---- ladder: which tranches are due (closes within the episode)
    seg_dd = D["dd"][i0:i1 + 1]
    seg_dds = D["dd_spy"][i0:i1 + 1]
    due = []
    for k, (lvl, lvl_spy, q_share) in enumerate(ladder):
        hit_q = np.where(seg_dd <= lvl)[0]
        hit_s = np.where(seg_dds <= lvl_spy)[0]
        first = min([h for h in (hit_q[0] if len(hit_q) else None, hit_s[0] if len(hit_s) else None) if h is not None], default=None)
        due.append({"k": k + 1, "lvl": lvl, "lvl_spy": lvl_spy, "q_share": q_share, "due": first is not None,
                    "date": idx[i0 + first].date().isoformat() if first is not None else None, "price": hi * (1 + lvl)})
    deployed = min(sum(tranche_usd for d in due if d["due"]), reserve_base)
    qqq_add = sum(tranche_usd * d["q_share"] for d in due if d["due"])
    ivv_add = sum(tranche_usd * (1 - d["q_share"]) for d in due if d["due"])
    scale = deployed / (qqq_add + ivv_add) if (qqq_add + ivv_add) > deployed > 0 else 1.0
    qqq_add *= scale
    ivv_add *= scale
    # ---- sleeve
    ss = sleeve_state(D, i0, i1, sizing, exposure_scale)
    sleeve_pct = ss["rung"]
    sleeve_usd = sleeve_pct * total_held
    # ---- targets
    target = {"IVV": base.get("IVV", 0.0) + ivv_add, "QQQ": base.get("QQQ", 0.0) + qqq_add, vehicle: 0.0}
    reserve_left = max(0.0, base_reserve - deployed)
    fund = min(sleeve_usd, reserve_left)
    reserve_left -= fund
    short = sleeve_usd - fund
    if short > 0 and rules.get("fund_core", True):
        take = min(short, target["QQQ"]); target["QQQ"] -= take; short -= take
        take = min(short, target["IVV"]); target["IVV"] -= take; short -= take
    target[vehicle] = sleeve_usd - max(0.0, short)
    for k, w in (("TBILL", mix.get("tbill", 0.0)), ("IEF", mix.get("bond", 0.0)), ("GLD", mix.get("gold", 0.0))):
        target[k] = reserve_left * w
    other_lev = "TQQQ" if vehicle == "QLD" else "QLD"
    if held.get(other_lev, 0.0) > 0:
        target[other_lev] = 0.0
    total_target = sum(target.values())
    # ---- fund rows
    order = ["IVV", "QQQ", vehicle, other_lev, "TBILL", "IEF", "GLD"]
    rows = []
    for t in order:
        if t not in target and held.get(t, 0.0) <= 0:
            continue
        tg = target.get(t, 0.0)
        hd = held.get(t, 0.0)
        tr = tg - hd
        note = ""
        if t == vehicle:
            alt = sleeve_usd * (lev / VEHICLE_LEV[other_lev])
            note = (f"or {usd(alt)} of {other_lev} for the same exposure" if sleeve_usd > 0 else
                    (f"would be {ss['size_if_confirmed'] * 100:.0f}% ({usd(ss['size_if_confirmed'] * total_held)}) once two weekly closes are back above the 200-day" if ss["size_if_confirmed"] > 0 else "no leverage until the episode is at least 15% deep and the 200-day is reclaimed"))
        elif t == "QQQ" and qqq_add > 0:
            note = f"includes {usd(qqq_add)} from {sum(d['due'] for d in due)} ladder tranche(s)"
        elif t == "IVV" and ivv_add > 0:
            note = f"includes {usd(ivv_add)} from the ladder"
        elif t in reserve_keys:
            w = {"TBILL": mix.get("tbill", 0.0), "IEF": mix.get("bond", 0.0), "GLD": mix.get("gold", 0.0)}[t]
            note = f"{w * 100:.0f}% of the reserve"
        rows.append({"ticker": t, "name": FUND_NAME.get(t, t), "role": ROLE.get(t, ""), "target_usd": round(tg), "held_usd": round(hd), "trade_usd": round(tr),
                     "target_pct": round(tg / total_target * 100, 1) if total_target else 0.0, "held_pct": round(hd / total_held * 100, 1) if total_held else 0.0,
                     "target_fmt": usd(tg), "held_fmt": usd(hd), "trade_fmt": ("buy " if tr > 0 else "sell ") + usd(abs(tr)) if abs(tr) >= 1000 else "—",
                     "target_pct_fmt": pct(tg / total_target * 100, 0) if total_target else "—", "held_pct_fmt": pct(hd / total_held * 100, 0) if total_held else "—", "note": note})
    # ---- headline
    trades = [(r["ticker"], r["trade_usd"]) for r in rows if abs(r["trade_usd"]) >= 1000]
    core = [(t, v) for t, v in trades if t not in reserve_keys]
    res_moves = [(t, v) for t, v in trades if t in reserve_keys]
    buys = [f"{usd(v)} {t}" for t, v in core if v > 0]
    sells = [f"{usd(-v)} {t}" for t, v in core if v < 0]
    parts = []
    if core:
        parts.append(("Buy " + ", ".join(buys) if buys else "") + ("; " if buys and sells else "") + ("sell " + ", ".join(sells) if sells else ""))
    else:
        parts.append("No ladder or sleeve trade today")
    if res_moves:
        into = [f"{usd(v)} to {t}" for t, v in res_moves if v > 0]
        outof = [f"{t}" for t, v in res_moves if v < 0]
        parts.append("reserve mix: move " + ", ".join(into) + (" from " + "/".join(outof) if outof else ""))
    headline = "; ".join(parts) + "."
    # ---- intraday watch line
    intraday = None
    try:
        q = json.loads((repo / "data" / "intraday" / "quotes.json").read_text())
        qq = next((x for x in q.get("quotes", []) if x.get("symbol") == "QQQ"), None)
        if qq and qq.get("session") and pd.Timestamp(qq["session"]) > last:
            live_dd = qq["last"] / hi - 1
            nxt = next((d for d in due if not d["due"]), None)
            s = f"QQQ {qq['last']:.2f} at {qq['as_of_et'][-5:]} ET ({qq['chg_pct']:+.1f}% on the day) is {live_dd * 100:.1f}% from the high"
            if nxt:
                s += f"; tranche {nxt['k']} fires on a close at or below {nxt['price']:.2f}" + (" — in play now" if live_dd <= nxt["lvl"] else "")
            if not math.isnan(sma200):
                s += f"; 200-day at {sma200:.1f} ({'above' if qq['last'] > sma200 else 'below'})"
            intraday = s
    except Exception:  # noqa: BLE001
        intraday = None
    # ---- backtest line for the rule set in force
    bt = None
    if run_backtest:
        try:
            horizon = 252 * 7
            first_ok = int(np.argmax(~np.isnan(D["sma200"])))
            months = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).first()
            starts = [idx.get_loc(t) for t in months if idx.get_loc(t) >= max(first_ok, 252) and idx.get_loc(t) + horizon < len(idx)]
            eq = max(total_held, 1.0)
            ivv0, cash0 = base.get("IVV", 0.0) + base.get("QQQ", 0.0), base_reserve
            if ivv0 + cash0 <= 0:
                ivv0, cash0 = 0.43 * eq, 0.57 * eq
            kw_rule = dict(ivv0=ivv0, qqq0=0.0, cash0=cash0, tranche_frac=rules["tranche_frac"], sizing=sizing, lev=lev, fund_core=rules.get("fund_core", True), reserve_mix=mix)
            kw_hold = dict(ivv0=ivv0, qqq0=0.0, cash0=cash0, ladder=False, sleeve=False)
            cag_r, cag_h, mdd_r, fin_r = [], [], [], []
            for s0 in starts:
                r_ = AB.simulate(D, s0, s0 + horizon, **kw_rule); h_ = AB.simulate(D, s0, s0 + horizon, **kw_hold)
                cag_r.append(r_["cagr"]); cag_h.append(h_["cagr"]); mdd_r.append(r_["max_dd"]); fin_r.append(r_["final_pct"])
            cag_r, cag_h = np.array(cag_r), np.array(cag_h)
            bt = (f"Rules vs holding the starting mix, {len(starts)} rolling 7-year windows from {idx[starts[0]].date()} to {idx[starts[-1]].date()}: "
                  f"median {np.median(cag_r):.1f}% a year vs {np.median(cag_h):.1f}%, 10th percentile {np.percentile(cag_r, 10):.1f}% vs {np.percentile(cag_h, 10):.1f}%, "
                  f"worst window {min(fin_r):+.0f}% total, worst drawdown {min(mdd_r):.0f}%, ahead of holding in {np.mean(cag_r > cag_h) * 100:.0f}% of windows. "
                  f"In-sample; leveraged funds simulated before 2010.")
        except Exception as e:  # noqa: BLE001
            bt = f"Backtest line unavailable: {type(e).__name__}"
    # ---- plan rows
    nxt_due = next((d for d in due if not d["due"]), None)
    plan = [
        {"key": "headline", "label": "Today", "value": headline, "detail": f"Targets follow the plan's rules from the baseline held on {ep_start.date()}; trades are against the holdings on file as of {as_of.date()}.", "state": "warn" if trades else "ok"},
        {"key": "episode", "label": "Drawdown episode", "value": f"QQQ {dd_last * 100:.1f}% from its 2-year high ({hi:.2f} on {hi_date.date()})",
         "detail": f"Episode began {ep_start.date()}; deepest close {deepest * 100:.1f}% on {idx[deepest_i].date()}; IVV/SPY deepest {deepest_spy * 100:.1f}%.", "state": "bad" if dd_last <= -0.14 else ("warn" if dd_last <= -0.07 else "ok")},
    ]
    for d in due:
        dist = (d["price"] / qqq_last - 1) * 100
        plan.append({"key": f"tranche_{d['k']}", "label": f"Ladder tranche {d['k']} ({usd(tranche_usd)}: {usd(tranche_usd * d['q_share'])} QQQ / {usd(tranche_usd * (1 - d['q_share']))} IVV)",
                     "value": ("due since " + d["date"]) if d["due"] else f"QQQ close at or below {d['price']:.2f} ({dist:+.1f}% from the last close) or IVV {d['lvl_spy'] * 100:.0f}% off its high",
                     "detail": f"Fires at {d['lvl'] * 100:.0f}% QQQ drawdown or {d['lvl_spy'] * 100:.0f}% IVV drawdown.", "state": "warn" if d["due"] else ""})
    weeks = ss["wk_above200"]
    above_now = qqq_last > sma200 if not math.isnan(sma200) else False
    gate_open = weeks >= 2 and rv < 45 and ss["cooloff"] == 0
    reclaim_val = (f"QQQ {qqq_last:.2f} is {'above' if above_now else 'below'} its 200-day ({sma200:.1f}, {(qqq_last / sma200 - 1) * 100:+.1f}%)" if not math.isnan(sma200) else "200-day unavailable")
    if ss["size_if_confirmed"] <= 0 and sleeve_pct <= 0:
        reclaim_val += "; no rung to confirm yet (the episode's deepest close must reach −15% first)"
    else:
        reclaim_val += "; gate open" if gate_open else f"; weekly closes above so far: {weeks} of 2 needed" + (f"; cool-off {ss['cooloff']} more weeks" if ss["cooloff"] else "")
    plan.append({"key": "reclaim", "label": "200-day reclaim (gate for leverage)", "value": reclaim_val,
                 "detail": f"20-day realized vol {rv:.0f}% (must be under 45%). " + ("Exit rule: a weekly close 3% below the 200-day sells the sleeve." if sleeve_pct > 0 else ""),
                 "state": ("ok" if gate_open else "warn") if (ss["size_if_confirmed"] > 0 or sleeve_pct > 0) else ""})
    plan.append({"key": "sleeve", "label": f"Leveraged sleeve ({vehicle}{', or TQQQ at two-thirds of the dollars' if vehicle == 'QLD' else ''})",
                 "value": (f"{sleeve_pct * 100:.0f}% of the portfolio ({usd(sleeve_usd)})" if sleeve_pct > 0 else
                           (f"0% now; the rule allows {ss['size_if_confirmed'] * 100:.0f}% ({usd(ss['size_if_confirmed'] * total_held)}) once two weekly closes are back above the 200-day" if ss["size_if_confirmed"] > 0 else
                            f"0%: no rung yet (deepest close {deepest * 100:.1f}%; the first rung, 15%, needs −15%)")),
                 "detail": "Size by the deepest close of the episode: 15% at −15%, 30% at −25%, 45% at −35%, 75% at −50% (QLD; two-thirds of that in TQQQ). Cut to 15% when the drawdown closes to −10%, to zero at −3%.",
                 "state": "ok" if sleeve_pct > 0 else ""})
    plan.append({"key": "reserve", "label": "Reserve", "value": f"{usd(reserve_left)} after the ladder ({usd(deployed)} deployed of {usd(reserve_base)})" + (f"; {usd(fund)} funds the sleeve" if fund > 0 else ""),
                 "detail": f"Held as {mix.get('tbill', 0) * 100:.0f}% T-bills, {mix.get('bond', 0) * 100:.0f}% IEF, {mix.get('gold', 0) * 100:.0f}% GLD. Tranches are {rules['tranche_frac'] * 100:.0f}% of the reserve at the start of the episode.", "state": ""})
    if intraday:
        plan.append({"key": "intraday", "label": "Intraday watch", "value": intraday, "detail": "Rules act on closes; this is where the open session stands.", "state": "warn"})
    if bt:
        plan.append({"key": "backtest", "label": "Evidence", "value": bt, "detail": "", "state": ""})
    plan.append({"key": "holdings", "label": "Holdings on file", "value": ", ".join(f"{t} {usd(v)}" for t, v in held.items() if v > 0) + f" (as of {as_of.date()}, marked to {last.date()})",
                 "detail": "Tell Claude when you trade and the file is updated; the trade column then clears.", "state": ""})
    for r in plan:
        r["as_of"] = last.date().isoformat()
    return rows, plan


def build_exposure(repo: Path, portfolio: dict, run_backtest: bool = True) -> tuple[list[dict], list[dict]]:
    """Trend- and volatility-scaled exposure to the Nasdaq-100 (see exposure_backtest.py)."""
    rules = dict(EXPOSURE_RULES)
    rules.update(portfolio.get("rules") or {})
    vehicle = rules["vehicle"] if rules["vehicle"] in VEHICLE_LEV else "QLD"
    lev = VEHICLE_LEV[vehicle]
    other_lev = "TQQQ" if vehicle == "QLD" else "QLD"
    mix = rules["defensive_mix"]
    data = P.Data(repo)
    D = X.load(repo, "QQQ")
    idx = D["idx"]
    i1 = len(idx) - 1
    last = idx[i1]
    qqq_last, sma200, rv, dd_last, hi = float(D["qqq"][i1]), float(D["sma200_a"][i1]), float(D["rv_a"][i1]), float(D["dd_a"][i1]), float(D["hi2y"][i1])
    hi_date = idx[int(np.argmax(D["qqq"][max(0, i1 - 503):i1 + 1])) + max(0, i1 - 503)]
    # decision days: the last trading day of each week, plus the day the plan started (its entry check)
    wk = idx.isocalendar().week.values
    mask = np.r_[wk[1:] != wk[:-1], last.weekday() == 4]
    plan_start = pd.Timestamp(portfolio.get("plan_start")) if portfolio.get("plan_start") else None
    if plan_start is not None:
        j = idx.get_indexer([plan_start], method="bfill")[0]
        if j >= 0:
            mask[j] = True
    params = {k: rules[k] for k in ("e_hi", "e_mid", "e_lo", "vol_hi", "vol_crash", "exit_pct", "reclaim_bonus", "reclaim_depth")}
    i0 = max(0, i1 - 756)
    st = X.replay_state(D, i0, i1, decision_mask=mask, **params)
    target = float(st["target"] if st["target"] is not None else rules["e_lo"])
    decided_on = st["last_decision"].date() if st["last_decision"] is not None else None
    # ---- holdings marked to the last close
    closes = {t: data.close(t) for t in ("IVV", "QQQ", "QLD", "TQQQ", "IEF", "GLD")}
    as_of = pd.Timestamp(portfolio.get("as_of") or last)
    tbill = D["tbd"]

    def mtm(h: dict) -> dict:
        out = {}
        for t, v in (h or {}).items():
            v = float(v or 0)
            if t == "TBILL":
                j0 = idx.get_indexer([as_of], method="ffill")[0]
                out[t] = v * float(np.prod(1 + tbill[j0 + 1:i1 + 1])) if j0 < i1 else v
            elif closes.get(t) is not None and len(closes[t]):
                sr = closes[t]
                p0 = float(sr[:as_of].iloc[-1]) if len(sr[:as_of]) else float(sr.iloc[-1])
                out[t] = v / p0 * float(sr.iloc[-1])
            else:
                out[t] = v
        return out

    held = mtm(portfolio.get("holdings") or {})
    total = sum(held.values()) or 1.0
    live_exposure = (held.get("IVV", 0) + held.get("QQQ", 0) + 2 * held.get("QLD", 0) + 3 * held.get("TQQQ", 0)) / total
    # ---- provisional read from the open session, if newer than the last close
    prov = None
    try:
        q = json.loads((repo / "data" / "intraday" / "quotes.json").read_text())
        qq = next((x for x in q.get("quotes", []) if x.get("symbol") == "QQQ"), None)
        if qq and qq.get("session") and pd.Timestamp(qq["session"]) > last:
            p = float(qq["last"])
            logs = np.diff(np.log(D["qqq"][i1 - 19:i1 + 1]))
            rv_prov = float(np.std(np.r_[logs, math.log(p / qqq_last)], ddof=1) * math.sqrt(252) * 100)
            trend_up = st["trend_up"]
            if trend_up and p < sma200 * (1 - rules["exit_pct"]):
                trend_up = False
            elif not trend_up and p > sma200:
                trend_up = True
            if trend_up:
                t_prov = rules["e_hi"] if rv_prov < rules["vol_hi"] else rules["e_mid"]
                why = f"above the 200-day; realized vol {rv_prov:.0f}% {'under' if rv_prov < rules['vol_hi'] else 'at or above'} {rules['vol_hi']:.0f}%"
            else:
                t_prov = rules["e_lo"]
                why = f"below the 200-day exit line ({sma200 * (1 - rules['exit_pct']):.2f})"
            if rv_prov >= rules["vol_crash"]:
                t_prov = min(t_prov, rules["e_lo"]); why += f"; vol {rv_prov:.0f}% ≥ {rules['vol_crash']:.0f}%: no leverage"
            is_decision_day = pd.Timestamp(qq["session"]).weekday() == 4 or (plan_start is not None and pd.Timestamp(qq["session"]) == plan_start)
            prov = {"price": p, "as_of": qq["as_of_et"], "chg": float(qq["chg_pct"]), "dd": p / hi - 1, "rv": rv_prov, "target": float(t_prov), "why": why, "decision_day": is_decision_day, "session": qq["session"]}
    except Exception:  # noqa: BLE001
        prov = None
    # the headline follows the provisional read when today is a decision day (Friday, or the plan's first day) and the
    # session is newer than the last close; otherwise the standing weekly target
    use_prov = prov is not None and prov["decision_day"]
    E = prov["target"] if use_prov else target
    # ---- dollar targets
    if E >= 1.0:
        lev_share = min(1.0, (E - 1.0) / (lev - 1.0)); core_share = 1.0 - lev_share; def_share = 0.0
    else:
        lev_share, core_share, def_share = 0.0, E, 1.0 - E
    core_usd = core_share * total
    ivv_t = min(held.get("IVV", 0.0), core_usd)  # IVV stays as part of the 1x core; new core money goes to QQQ
    qqq_t = core_usd - ivv_t
    target_usd = {"IVV": ivv_t, "QQQ": qqq_t, vehicle: lev_share * total, "TBILL": def_share * total * mix.get("tbill", 0), "IEF": def_share * total * mix.get("bond", 0), "GLD": def_share * total * mix.get("gold", 0)}
    if held.get(other_lev, 0) > 0:
        target_usd[other_lev] = 0.0
    rows = []
    for t in ["IVV", "QQQ", vehicle, other_lev, "TBILL", "IEF", "GLD"]:
        if t not in target_usd:
            continue
        tg, hd = target_usd[t], held.get(t, 0.0)
        tr = tg - hd
        note = ""
        if t == vehicle:
            alt = lev_share * total * (lev - 1) / (VEHICLE_LEV[other_lev] - 1)
            note = (f"{lev_share * 100:.0f}% of the portfolio in the 2x fund = {E:.2f}x total exposure; or {usd(alt)} of {other_lev} with the rest in QQQ" if lev_share > 0 else
                    f"no leverage at {E:.2f}x; leverage returns at {rules['e_hi']:.1f}x when QQQ is above its 200-day with realized vol under {rules['vol_hi']:.0f}%")
        elif t == "IVV":
            note = "kept as part of the 1x core; the tested plan's core is QQQ, so new core money goes to QQQ"
        elif t == "QQQ":
            note = f"core equity {core_share * 100:.0f}% of the portfolio" + ("" if core_share >= 1 else f" ({E:.2f}x exposure below 1 leaves the rest defensive)")
        elif t in ("TBILL", "IEF", "GLD"):
            w = {"TBILL": mix.get("tbill", 0), "IEF": mix.get("bond", 0), "GLD": mix.get("gold", 0)}[t]
            note = f"{w * 100:.0f}% of the defensive sleeve ({def_share * 100:.0f}% of the portfolio)"
        rows.append({"ticker": t, "name": FUND_NAME.get(t, t), "role": "core" if t in ("IVV", "QQQ") else ("leverage" if t in VEHICLE_LEV else "defensive"),
                     "target_usd": round(tg), "held_usd": round(hd), "trade_usd": round(tr), "target_pct": round(tg / total * 100, 1), "held_pct": round(hd / total * 100, 1),
                     "target_fmt": usd(tg), "held_fmt": usd(hd), "trade_fmt": ("buy " if tr > 0 else "sell ") + usd(abs(tr)) if abs(tr) >= 1000 else "—",
                     "target_pct_fmt": pct(tg / total * 100, 0), "held_pct_fmt": pct(hd / total * 100, 0), "note": note})
    trades = [(r["ticker"], r["trade_usd"]) for r in rows if abs(r["trade_usd"]) >= 1000]
    defensive_keys = ("TBILL", "IEF", "GLD")
    core_tr = [(t, v) for t, v in trades if t not in defensive_keys]
    def_tr = [(t, v) for t, v in trades if t in defensive_keys]
    basis = (f"provisional on today's {prov['as_of'][-5:]} ET price, final at the close" if use_prov else f"set at the {decided_on} close")

    def phrase(tr):
        b = [f"{usd(v)} {t}" for t, v in tr if v > 0]
        s_ = [f"{usd(-v)} {t}" for t, v in tr if v < 0]
        return (("buy " + ", ".join(b)) if b else "") + ("; " if b and s_ else "") + (("sell " + ", ".join(s_)) if s_ else "")

    if core_tr:
        headline = f"Move to {E:.2f}x exposure ({basis}): {phrase(core_tr)}" + (f"; defensive sleeve: {phrase(def_tr)}" if def_tr else "") + "."
    elif def_tr:
        headline = f"Hold at {E:.2f}x exposure ({basis}); defensive sleeve: {phrase(def_tr)}."
    else:
        headline = f"Hold at {E:.2f}x exposure ({basis}); live exposure {live_exposure:.2f}x."
    # ---- rule rows
    s200_exit = sma200 * (1 - rules["exit_pct"])
    plan = [{"key": "headline", "label": "Today", "value": headline,
             "detail": f"Decisions are taken on weekly closes (and on the plan's first day); the target then holds until the next weekly close moves it more than {rules['band'] * 100:.0f} points. Holdings on file as of {as_of.date()}.",
             "state": "warn" if trades else "ok"},
            {"key": "exposure", "label": "Exposure", "value": f"Target {E:.2f}x" + (" (provisional: today's close decides)" if use_prov else "") + f" · live {live_exposure:.2f}x" + (f" · previous weekly target {target:.2f}x (set {decided_on})" if use_prov and decided_on else ""),
             "detail": f"Ladder of targets: {rules['e_hi']:.1f}x above the 200-day with 20-day realized vol under {rules['vol_hi']:.0f}%; {rules['e_mid']:.1f}x above it with higher vol; {rules['e_lo']:.1f}x below it; +{rules['reclaim_bonus']:.2f} after a {abs(rules['reclaim_depth']) * 100:.0f}% bear is reclaimed, until the drawdown closes to −10%; no leverage when vol ≥ {rules['vol_crash']:.0f}%.",
             "state": "ok" if E > 1 else ("warn" if E >= 1 else "bad")},
            {"key": "trend", "label": "Trend test (QQQ vs 200-day)",
             "value": f"Last close {qqq_last:.2f} vs 200-day {sma200:.1f} ({(qqq_last / sma200 - 1) * 100:+.1f}%): {'up' if st['trend_up'] else 'down'}; weekly closes above so far: {st['wk_above']}",
             "detail": f"While up, a weekly close below {s200_exit:.2f} (3% under the 200-day) turns it down; while down, a weekly close above the 200-day turns it up. {st['reason']}.",
             "state": "ok" if st["trend_up"] else "bad"},
            {"key": "vol", "label": "Volatility gate", "value": f"20-day realized vol {rv:.0f}% (leverage needs under {rules['vol_hi']:.0f}%; everything but the {rules['e_lo']:.1f}x floor is off at {rules['vol_crash']:.0f}%+)",
             "detail": "Volatility drag and whipsaw losses on leveraged funds grow with the square of volatility; the tests kept leverage only in calm uptrends.", "state": "ok" if rv < rules["vol_hi"] else ("warn" if rv < rules["vol_crash"] else "bad")},
            {"key": "episode", "label": "Drawdown episode", "value": f"QQQ {dd_last * 100:.1f}% from its 2-year high ({hi:.2f} on {hi_date.date()}); deepest close this episode {st['deepest'] * 100:.1f}%" + ("; reclaim bonus on" if st["reclaim_on"] else ""),
             "detail": f"After a close {abs(rules['reclaim_depth']) * 100:.0f}% or more below the high, two weekly closes back above the 200-day add +{rules['reclaim_bonus']:.2f}x until the drawdown is inside −10%.",
             "state": "bad" if dd_last <= -0.2 else ("warn" if dd_last <= -0.1 else "")},
            {"key": "defensive", "label": "Defensive sleeve", "value": f"{def_share * 100:.0f}% of the portfolio ({usd(def_share * total)}) as {mix.get('tbill', 0) * 100:.0f}% T-bills, {mix.get('bond', 0) * 100:.0f}% IEF, {mix.get('gold', 0) * 100:.0f}% GLD",
             "detail": "Held whenever exposure is below 1x; it is what gets redeployed when the trend turns back up.", "state": ""}]
    if prov is not None:
        plan.append({"key": "intraday", "label": "Intraday watch" + (" — decision day" if prov["decision_day"] else ""),
                     "value": f"QQQ {prov['price']:.2f} at {prov['as_of'][-5:]} ET ({prov['chg']:+.1f}% on the day), {prov['dd'] * 100:.1f}% from the high, {'above' if prov['price'] > sma200 else 'below'} the 200-day ({sma200:.1f}); provisional vol {prov['rv']:.0f}% → target {prov['target']:.2f}x ({prov['why']})",
                     "detail": ("Today's close sets the target." if prov["decision_day"] else "Not a decision day: the standing target holds until this week's close."), "state": "warn"})
    if run_backtest:
        try:
            starts, hz = X.windows(D, 7)
            kw = dict(e_hi=rules["e_hi"], e_mid=rules["e_mid"], e_lo=rules["e_lo"], vol_hi=rules["vol_hi"], exit_pct=rules["exit_pct"], reclaim_bonus=rules["reclaim_bonus"], reclaim_depth=rules["reclaim_depth"], lev=lev, defensive=mix, band=rules["band"], vol_crash=rules["vol_crash"])
            c, m, f, e = X.rolling(D, starts, hz, **kw)
            qc = X.rolling(D, starts, hz, constant=1.0); sc = X.rolling(D, starts, hz, core="S", constant=1.0)
            bt = (f"{len(starts)} rolling 7-year windows ({idx[starts[0]].date()} to {idx[starts[-1]].date()} starts): rules median {np.median(c):.1f}% a year vs QQQ {np.median(qc[0]):.1f}% and SPY {np.median(sc[0]):.1f}%; "
                  f"ahead of QQQ in {(c > qc[0]).mean() * 100:.0f}% of windows (median +{np.median(c - qc[0]):.1f}%/yr) and of SPY in {(c > sc[0]).mean() * 100:.0f}%; 10th-percentile {np.percentile(c, 10):.1f}% vs QQQ {np.percentile(qc[0], 10):.1f}%; "
                  f"worst window {f.min():+.0f}% vs QQQ {qc[2].min():+.0f}%; worst drawdown {m.min():.0f}% vs QQQ {qc[1].min():.0f}%. In-sample; 2x fund simulated before 2006; no taxes.")
        except Exception as e:  # noqa: BLE001
            bt = f"Backtest line unavailable: {type(e).__name__}"
        plan.append({"key": "backtest", "label": "Evidence", "value": bt, "detail": "", "state": ""})
    plan.append({"key": "holdings", "label": "Holdings on file", "value": ", ".join(f"{t} {usd(v)}" for t, v in held.items() if v > 0) + f" (as of {as_of.date()}, marked to {last.date()})",
                 "detail": "Tell Claude when you trade and the file is updated; the trade column then clears.", "state": ""})
    for r in plan:
        r["as_of"] = last.date().isoformat()
    return rows, plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--portfolio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-backtest", action="store_true")
    a = ap.parse_args()
    portfolio = json.loads(Path(a.portfolio).read_text())
    rows, plan = build(Path(a.repo), portfolio, run_backtest=not a.no_backtest)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "allocation.json").write_text(json.dumps(rows, indent=1, default=str))
    (out / "plan.json").write_text(json.dumps(plan, indent=1, default=str))
    print(plan[0]["value"])
    for r in rows:
        print(f"  {r['ticker']:5s} target {r['target_fmt']:>8s} ({r['target_pct_fmt']:>4s})  held {r['held_fmt']:>8s}  {r['trade_fmt']:<14s} {r['note']}")
    for r in plan[1:]:
        print(f"  - {r['label']}: {r['value']}")


if __name__ == "__main__":
    main()
