from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATOM = "{http://www.w3.org/2005/Atom}"


def _text(element: ET.Element, name: str) -> str:
    child = element.find(f"{ATOM}{name}")
    return "" if child is None or child.text is None else child.text.strip()


def fetch_arxiv(query: str, max_results: int) -> list[dict]:
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"https://export.arxiv.org/api/query?{params}"
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = response.read()

    root = ET.fromstring(payload)
    records: list[dict] = []
    for entry in root.findall(f"{ATOM}entry"):
        authors = [
            _text(author, "name")
            for author in entry.findall(f"{ATOM}author")
        ]
        arxiv_id = _text(entry, "id").rsplit("/", 1)[-1]
        records.append(
            {
                "source_id": arxiv_id,
                "source_type": "arxiv",
                "title": " ".join(_text(entry, "title").split()),
                "summary": " ".join(_text(entry, "summary").split()),
                "published": _text(entry, "published"),
                "updated": _text(entry, "updated"),
                "authors": "; ".join(authors),
                "url": _text(entry, "id"),
                "query": query,
            }
        )
    return records


def write_outputs(records: list[dict], output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_stem.with_suffix(".jsonl")
    csv_path = output_stem.with_suffix(".csv")

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if records:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect public arXiv materials-science source metadata.")
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument(
        "--query",
        default='cat:cond-mat.mtrl-sci AND (band gap OR formation energy OR "materials project" OR stability)',
    )
    parser.add_argument(
        "--output-stem",
        default=str(ROOT / "data/sources/arxiv_materials_sources"),
    )
    args = parser.parse_args()

    records = fetch_arxiv(args.query, args.max_results)
    # Be polite if this script is expanded to multiple queries later.
    time.sleep(1)
    write_outputs(records, Path(args.output_stem))
    print(f"Wrote {len(records)} arXiv source records to {args.output_stem}.jsonl/.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

