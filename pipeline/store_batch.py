#!/usr/bin/env python3
"""
Build the dataset documents for the dashboard's store.

    python pipeline/store_batch.py --assets assets.json --out store_out [--store existing_store_dir] [--by "Claude (scheduled run)"]

assets.json maps file names to the asset urls returned by the uploads, e.g.
    {"summary.json": "/_blob/0123...", "history.json": "/_blob/abcd...", ...}

--store is the directory an `ArtifactData list` (collection "datasets", out_dir=...) saved the current documents to
(store_dir/datasets/<id>.json); when given, each write is an "update" pinned to that document's version, otherwise a
full "set" that creates the document.

Writes <out>/<id>.json (one document payload each) and <out>/batch_writes.json: the `writes` array for an
`ArtifactData batch` call, with file_path pointing at those payload files.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "datasets.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", default=None)
    ap.add_argument("--by", default="Claude (scheduled run)")
    args = ap.parse_args()
    assets = json.loads(Path(args.assets).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    writes = []
    for ds_id, meta in SPEC["datasets"].items():
        fname = f"{ds_id}.json"
        if fname not in assets:
            print(f"skip {ds_id}: no asset url for {fname}")
            continue
        existing = None
        if args.store:
            p = Path(args.store) / "datasets" / f"{ds_id}.json"
            if p.exists():
                existing = json.loads(p.read_text())
        payload = {"source": {"kind": "file", "url": assets[fname], "name": fname, "format": "json"},
                   "updated": {"at": now, "by": args.by}}
        if ds_id == "score_history" and "history.json" in assets:
            payload["history_url"] = assets["history.json"]
        if existing is None:
            payload.update({"title": meta["title"], "description": meta["description"],
                            "schedule": SPEC["schedule"], "staleAfterHours": SPEC["staleAfterHours"]})
            op = "set"
        else:
            op = "update"
        (out / fname).write_text(json.dumps(payload, indent=1))
        entry = {"op": op, "collection": "datasets", "doc_id": ds_id, "file_path": str((out / fname).resolve())}
        if existing is not None:
            ver = existing.get("version") or (existing.get("_meta") or {}).get("version")
            if ver is None:
                # a saved document may wrap the content; look one level down
                for v in existing.values():
                    if isinstance(v, dict) and "version" in v:
                        ver = v["version"]
                        break
            if ver is not None:
                entry["if_version"] = int(ver)
        writes.append(entry)
    (out / "batch_writes.json").write_text(json.dumps(writes, indent=1))
    print(f"{len(writes)} writes -> {out / 'batch_writes.json'}")


if __name__ == "__main__":
    main()
