from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sret_materials_rag.utils.env import load_dotenv


def _require_key() -> str:
    api_key = os.environ.get("MP_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "MP_API_KEY is not set. Export it in your shell or create an untracked .env file."
        )
    return api_key


def _mp_api_available() -> bool:
    try:
        import mp_api  # noqa: F401
    except ImportError:
        return False
    return True


def collect_with_mp_api(api_key: str, limit: int) -> list[dict]:
    from mp_api.client import MPRester

    fields = [
        "material_id",
        "formula_pretty",
        "band_gap",
        "formation_energy_per_atom",
        "energy_above_hull",
        "is_stable",
    ]
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            fields=fields,
            num_chunks=1,
            chunk_size=limit,
        )

    records = []
    for doc in docs:
        row = {field: getattr(doc, field, None) for field in fields}
        row["source_type"] = "materials_project"
        row["source_url"] = f"https://materialsproject.org/materials/{row['material_id']}"
        records.append(row)
    return records


def collect_with_rest(api_key: str, limit: int) -> list[dict]:
    fields = [
        "material_id",
        "formula_pretty",
        "band_gap",
        "formation_energy_per_atom",
        "energy_above_hull",
        "is_stable",
    ]
    params = urllib.parse.urlencode(
        {
            "_fields": ",".join(fields),
            "_limit": limit,
        }
    )
    url = f"https://api.materialsproject.org/materials/summary/?{params}"
    request = urllib.request.Request(
        url,
        headers={
            "X-API-KEY": api_key,
            "Accept": "application/json",
            "User-Agent": "sret-materials-rag-research/0.1",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(
            f"Materials Project REST request failed with HTTP {error.code}. "
            f"Response body: {body}"
        ) from error

    data = payload.get("data", payload)
    records = []
    for item in data:
        row = {field: item.get(field) for field in fields}
        row["source_type"] = "materials_project"
        row["source_url"] = f"https://materialsproject.org/materials/{row['material_id']}"
        records.append(row)
    return records


def write_outputs(records: list[dict], output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_stem.with_suffix(".jsonl")
    csv_path = output_stem.with_suffix(".csv")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    if records:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect candidate material property records from Materials Project.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--backend", choices=["auto", "rest", "mp-api"], default="auto")
    parser.add_argument(
        "--output-stem",
        default=str(ROOT / "data/sources/materials_project_candidates"),
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api_key = _require_key()
    if args.backend == "rest":
        records = collect_with_rest(api_key, args.limit)
    elif args.backend == "mp-api":
        records = collect_with_mp_api(api_key, args.limit)
    else:
        try:
            records = collect_with_rest(api_key, args.limit)
        except RuntimeError as rest_error:
            if not _mp_api_available():
                raise
            print(f"REST API failed; trying mp-api client: {rest_error}")
            records = collect_with_mp_api(api_key, args.limit)
        except ImportError:
            records = collect_with_rest(api_key, args.limit)
    write_outputs(records, Path(args.output_stem))
    print(f"Wrote {len(records)} Materials Project records to {args.output_stem}.jsonl/.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
