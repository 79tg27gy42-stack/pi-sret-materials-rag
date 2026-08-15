from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    inputs = sorted(ROOT.glob("data/sources/arxiv_materials_*_v2.csv"))
    if not inputs:
        raise RuntimeError("No arxiv_materials_*_v2.csv files found.")
    frames = []
    for path in inputs:
        frame = pd.read_csv(path).fillna("")
        frame["collector_file"] = path.name
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["source_id"]).sort_values(["query", "published", "source_id"], ascending=[True, False, True])
    records = df.to_dict("records")
    output_stem = ROOT / "data/sources/arxiv_materials_sources_v2"
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    with output_stem.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with output_stem.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Merged {len(df)} unique arXiv v2 sources from {len(inputs)} collector files.")
    print(df["query"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
