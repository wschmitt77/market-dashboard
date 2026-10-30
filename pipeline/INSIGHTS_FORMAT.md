# Daily insights file (written by Claude each morning)

The morning job reads `out/brief.md`, then writes `out/insights_today.json` in this shape and
runs `pipeline.py --stage finalize --insights out/insights_today.json`.

```json
{
  "date": "2026-10-29",
  "insights": [
    {
      "category": "dip_buy",
      "title": "Dip score crossed the first tranche level",
      "text": "QQQ is 7% off its high with RSI(2) at 4 while still above the 200-day; VIX/VIX3M is flat. Historically the dip score band 60–80 has preceded a positive 20-day QQQ return 68% of the time.",
      "action": "Add TQQQ tranche 1 of 3, capped at 20% of the portfolio while the regime light is green.",
      "metric_ids": ["drawdown_52w", "rsi2", "vix_term"],
      "prediction": {"type": "direction", "asset": "QQQ", "direction": "up", "horizon_days": 20, "confidence": 0.66}
    }
  ]
}
```

Rules that make the scorecard meaningful:

- Exactly 10 insights, ranked by importance (rank 1 = the most actionable).
- `category` is one of: regime, dip_buy, heat, sector, volatility, breadth, credit_rates, macro, calendar, risk.
- Every insight names the metric ids it rests on (`metric_ids`, from the brief) and has a concrete `action`
  (what to do, or explicitly "no action").
- At least 6 of the 10 carry a **testable prediction**; the rest use `{"type": "none"}` (context, risk flags, calendar).
  - `direction`: `asset` (QQQ, IVV, TQQQ, a sector ETF, SMH, IWM, HYG…) and `direction` up|down over `horizon_days` (5, 10 or 20).
    Scored from the asset's **open on the insight date** to the **close `horizon_days` sessions later**.
  - `relative`: `asset` vs `versus` (e.g. XLE vs IVV), `direction` outperform|underperform.
  - `level`: a series such as VIX, HY (high-yield OAS), DGS10, MOVE, with `direction` up|down.
- `confidence` is the honest probability the call is right (0.5–0.95). The Brier score on the dashboard rewards calibration, not bravado.
- Never restate the same prediction twice (e.g. "QQQ up 20d" in two insights).
- Keep `text` to two or three sentences a person can read on a phone; cite the numbers from the brief, not from memory.
- No advice language ("you should"); describe the signal and the rule-based action the dashboard's ladder implies.

Scoring (done by the pipeline, not by Claude):

- A prediction is `pending` until its horizon has passed, then `hit` or `miss` by the sign of the realized (relative) move.
- The scorecard shows hit rate, average signed return and Brier score, overall and by horizon, category, confidence band and asset.
