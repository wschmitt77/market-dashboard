# The 6 AM run — step by step

The scheduled Claude task follows this procedure in a fresh session. Replace `OWNER/REPO` and the dashboard URL
with the values in the task prompt.

1. **Get the data and the code**
   ```bash
   cd "$SCRATCH" && rm -rf repo out && git clone --depth 1 https://github.com/OWNER/REPO.git repo && mkdir out
   python3 -c "import json;m=json.load(open('repo/data/manifest.json'));print(m['run_at'], m['summary'])"
   ```
   If `run_at` is older than about 30 hours, the GitHub Action did not run: still build the dashboard, but say so in the summary.

2. **Fetch yesterday's history file** (the record of past insights and scores). Read the dashboard's
   `datasets/score_history` document with `ArtifactData get`; its `history_url` field is `/_blob/<asset id>`.
   `Artifact read` with `url` = dashboard URL and `path` = that asset id saves the file locally; pass its path as
   `--history`. On the very first run there is no history: omit `--history`.

3. **Compute**
   ```bash
   python3 repo/pipeline/pipeline.py --repo repo --out out --history <history.json>
   ```
   Read `out/brief.md`.

4. **Write the ten insights** to `out/insights_today.json` following `pipeline/INSIGHTS_FORMAT.md`: ranked, each with
   metric ids, a concrete action and (for at least six) a testable prediction with honest confidence. Optionally run one
   web search for overnight market news to add context to one or two items — keep every prediction tied to the metrics.

5. **Finalize**
   ```bash
   python3 repo/pipeline/pipeline.py --repo repo --out out --stage finalize --insights out/insights_today.json --history <history.json>
   ```

6. **Upload the data files** as assets of the dashboard, one `Artifact` publish call per file with `asset: true`,
   `url` = dashboard URL and `file_path` = each of:
   `summary.json, metrics.json, series.json, sectors.json, composite_parts.json, backtest.json, backtest_curve.json,
   backtest_stats.json, events.json, sources.json, insights.json, insight_log.json, scorecard.json, score_history.json, history.json`.
   Collect each result's `url` (`/_blob/<id>`) into `out/assets.json` as `{"summary.json": "/_blob/...", ...}`.

7. **Point the datasets at the new files.** Read the current documents with `ArtifactData list` (`collection`: `datasets`,
   `out_dir`: `store`). The result lists each document with its `version N`: write those into `store/versions.json` as
   `{"summary": 2, "metrics": 2, ...}` (the saved files do not contain the version). Then
   ```bash
   python3 repo/pipeline/store_batch.py --assets out/assets.json --store store --versions store/versions.json --out store_out
   ```
   and send one `ArtifactData batch` whose `writes` is the array in `store_out/batch_writes.json` (each entry already
   carries `file_path` and `if_version`). If the batch reports a version conflict, list again and rebuild the writes.

8. **Summarize for the owner** in three to five lines: regime light, dip and heat scores, the rule-based action, the top
   three insights in one line each, and any data problem (stale feed, sources that failed). That summary is the push notification.

Never invent numbers. If the feed is broken for a source, the dashboard shows it as unavailable; say so rather than guessing.
