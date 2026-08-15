from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sret_materials_rag.evaluation.ragas_baseline import run_ragas_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAGAS or RAGAS-compatible baseline metrics.")
    parser.add_argument("--input", default=str(ROOT / "data/processed/deepseek_r1_naive_full_standard.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/ragas_baseline_deepseek_r1_naive"))
    parser.add_argument("--backend", choices=["ragas", "proxy"], default="proxy")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--allow-proxy-fallback",
        action="store_true",
        help="Allow backend=ragas to fall back to proxy after a RAGAS failure. Do not use for paper results.",
    )
    args = parser.parse_args()
    result = run_ragas_baseline(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        backend=args.backend,
        limit=args.limit,
        allow_proxy_fallback=args.allow_proxy_fallback,
    )
    print("RAGAS baseline complete.")
    for key, value in result.metrics.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
