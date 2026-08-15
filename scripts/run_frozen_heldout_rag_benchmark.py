from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sret_materials_rag.evaluation.constraints import evaluate_material_qa_constraints  # noqa: E402
from sret_materials_rag.evaluation.faithfulness_evaluators import (  # noqa: E402
    EmbeddingSimilarityFaithfulnessEvaluator,
    LexicalOverlapFaithfulnessEvaluator,
    NLIHeuristicFaithfulnessEvaluator,
)


DATA = ROOT / "data" / "processed" / "canonical_v3"
OUT = ROOT / "results" / "frozen_heldout_rag_v3"
SYSTEMS = ["local_rule_rag", "qwen-max", "deepseek-r1"]
MODE = "naive_context"
BOOTSTRAP = 5000
SEED = 42


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-score the released fixed-context held-out benchmark.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA / "frozen_heldout_rag_756_v3.jsonl",
        help="Released JSONL split containing 756 cached responses.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    heldout = _read_jsonl(args.input)
    _validate_heldout(heldout)

    heldout = _score(heldout)
    heldout = heldout.sort_values(["raw_sample_id", "model"]).reset_index(drop=True)
    heldout.to_csv(args.output_dir / "scores.csv", index=False)

    summary = _system_summary(heldout)
    summary.to_csv(args.output_dir / "system_summary.csv", index=False)
    pairwise = _pairwise(heldout)
    pairwise.to_csv(args.output_dir / "pairwise_comparisons.csv", index=False)

    freeze = _freeze_manifest(heldout, args.input)
    (args.output_dir / "freeze_manifest.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n")
    paired_prompts = heldout.drop_duplicates("raw_sample_id")
    metrics = {
        "design": {
            "name": "post_freeze_heldout_rag_benchmark_v3",
            "scope": "external_to_development_pools_not_external_data_source",
            "answer_generation": "pre_existing_cached_outputs_scored_after_rule_freeze",
            "retrieval": "fixed_retrieved_context_shared_across_systems",
            "systems": SYSTEMS,
            "mode": MODE,
            "response_records": int(len(heldout)),
            "paired_prompt_contexts": int(heldout["raw_sample_id"].nunique()),
            "base_task_clusters": int(heldout["base_task_id"].nunique()),
            "excluded_development_base_tasks": 1653,
            "base_task_overlap_with_development": 0,
            "document_status_counts": paired_prompts["document_status"].value_counts().sort_index().to_dict(),
            "constraint_family_counts": paired_prompts["constraint_family"].value_counts().sort_index().to_dict(),
            "constraint_family_metadata_harmonized_records": int(
                heldout["original_constraint_family"].ne(heldout["constraint_family"]).sum()
            ),
            "bootstrap_iterations": BOOTSTRAP,
            "bootstrap_cluster": "base_task_id",
            "seed": SEED,
        },
        "system_summary": summary.to_dict(orient="records"),
        "pairwise_comparisons": pairwise.to_dict(orient="records"),
        "freeze": freeze,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(_report(metrics), encoding="utf-8")
    print(json.dumps(metrics["design"], ensure_ascii=False, indent=2))
    return 0


def _read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def _validate_heldout(frame: pd.DataFrame) -> None:
    if len(frame) != 756 or frame["raw_sample_id"].nunique() != 252:
        raise ValueError(f"Expected 756 records/252 paired prompts, observed {len(frame)}/{frame['raw_sample_id'].nunique()}")
    counts = frame.groupby("raw_sample_id")["model"].agg(["nunique", "size"])
    if not (counts["nunique"].eq(3) & counts["size"].eq(3)).all():
        raise ValueError("Every prompt must contain exactly one answer per system")
    for _, group in frame.groupby("raw_sample_id"):
        for column in ["question", "retrieved_context", "base_task_id", "document_status"]:
            if group[column].astype(str).nunique() != 1:
                raise ValueError(f"Paired systems are not aligned on {column}")


def _score(frame: pd.DataFrame) -> pd.DataFrame:
    score_columns = [
        "pi_sret_constraint_score",
        "pi_sret_violation",
        "pi_sret_violations",
        "lexical_support",
        "embedding_support",
        "nli_support",
    ]
    frame = frame.drop(columns=score_columns, errors="ignore")
    lexical = LexicalOverlapFaithfulnessEvaluator()
    embedding = EmbeddingSimilarityFaithfulnessEvaluator()
    nli = NLIHeuristicFaithfulnessEvaluator()
    rows = []
    for row in frame.itertuples(index=False):
        kwargs = {
            "question": str(row.question),
            "answer": str(row.answer),
            "retrieved_context": str(row.retrieved_context),
        }
        constraint = evaluate_material_qa_constraints(str(row.question), str(row.answer))
        rows.append(
            {
                "pi_sret_constraint_score": constraint.score,
                "pi_sret_violation": constraint.score < 1.0,
                "pi_sret_violations": ";".join(constraint.violations),
                "lexical_support": lexical.score(**kwargs).score,
                "embedding_support": embedding.score(**kwargs).score,
                "nli_support": nli.score(**kwargs).score,
            }
        )
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _cluster_ci(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    groups = {key: group.index.to_numpy() for key, group in frame.groupby("base_task_id", sort=False)}
    keys = np.array(list(groups), dtype=object)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(BOOTSTRAP):
        selected = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[key] for key in selected])
        values.append(float(frame.loc[idx, column].astype(float).mean()))
    return tuple(np.percentile(values, [2.5, 97.5]).tolist())


def _system_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "lexical_support",
        "embedding_support",
        "nli_support",
        "pi_sret_constraint_score",
        "pi_sret_violation",
    ]
    rows = []
    for system in SYSTEMS:
        group = frame[frame["model"].eq(system)].reset_index(drop=True)
        row = {
            "system": system,
            "n_responses": len(group),
            "n_prompt_contexts": group["raw_sample_id"].nunique(),
            "n_base_tasks": group["base_task_id"].nunique(),
        }
        for metric in metrics:
            low, high = _cluster_ci(group, metric)
            row[f"mean_{metric}"] = float(group[metric].astype(float).mean())
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _pairwise(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["lexical_support", "embedding_support", "nli_support", "pi_sret_constraint_score"]
    rows = []
    for left_idx, left in enumerate(SYSTEMS):
        for right in SYSTEMS[left_idx + 1 :]:
            left_frame = frame[frame["model"].eq(left)].set_index("raw_sample_id")
            right_frame = frame[frame["model"].eq(right)].set_index("raw_sample_id")
            paired = left_frame.join(right_frame, lsuffix="_left", rsuffix="_right", validate="one_to_one")
            cluster_map = paired["base_task_id_left"]
            keys = np.array(cluster_map.unique(), dtype=object)
            groups = {key: np.flatnonzero(cluster_map.to_numpy() == key) for key in keys}
            for metric in metrics:
                difference = paired[f"{metric}_left"].astype(float).to_numpy() - paired[f"{metric}_right"].astype(float).to_numpy()
                rng = np.random.default_rng(SEED)
                boot = []
                for _ in range(BOOTSTRAP):
                    selected = rng.choice(keys, size=len(keys), replace=True)
                    idx = np.concatenate([groups[key] for key in selected])
                    boot.append(float(difference[idx].mean()))
                low, high = np.percentile(boot, [2.5, 97.5])
                rows.append(
                    {
                        "system_left": left,
                        "system_right": right,
                        "metric": metric,
                        "paired_prompts": len(paired),
                        "mean_difference_left_minus_right": float(difference.mean()),
                        "ci_low": float(low),
                        "ci_high": float(high),
                    }
                )
    return pd.DataFrame(rows)


def _freeze_manifest(frame: pd.DataFrame, input_path: Path) -> dict[str, object]:
    files = [
        ROOT / "src/sret_materials_rag/evaluation/constraints.py",
        ROOT / "src/sret_materials_rag/evaluation/faithfulness_evaluators.py",
        Path(__file__).resolve(),
    ]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rule_version": "canonical_v3_frozen_after_expert_adjudication",
        "file_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "heldout_response_ids_sha256": hashlib.sha256("\n".join(sorted(frame["response_id"].astype(str))).encode()).hexdigest(),
        "heldout_base_task_ids_sha256": hashlib.sha256("\n".join(sorted(frame["base_task_id"].astype(str).unique())).encode()).hexdigest(),
        "released_input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "excluded_development_base_tasks": 1653,
    }


def _write_jsonl(frame: pd.DataFrame, path: Path) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in frame.to_dict(orient="records")) + "\n")


def _report(metrics: dict[str, object]) -> str:
    lines = [
        "# Frozen Held-Out Three-System RAG Benchmark",
        "",
        "## Design boundary",
        "",
        "This benchmark is external to all declared canonical_v3 development/evaluation pools at the base-task level, but it is not an external data source or institutional validation. It scores pre-existing cached answers after freezing the evaluator. Retrieval context is fixed across systems, so the comparison isolates answer generation rather than retrieval ranking. The cached provider labels are not immutable API snapshot identifiers, so the results compare stored outputs rather than current vendor endpoints.",
        "",
        f"- Response records: {metrics['design']['response_records']}",
        f"- Paired prompt-contexts: {metrics['design']['paired_prompt_contexts']}",
        f"- Base-task clusters: {metrics['design']['base_task_clusters']}",
        f"- Systems: {', '.join(metrics['design']['systems'])}",
        f"- Document status counts (paired prompts): {metrics['design']['document_status_counts']}",
        f"- Constraint-family counts (paired prompts): {metrics['design']['constraint_family_counts']}",
        "",
        "## System summary",
        "",
        _markdown_table(pd.DataFrame(metrics["system_summary"])),
        "",
        "## Pairwise comparisons",
        "",
        _markdown_table(pd.DataFrame(metrics["pairwise_comparisons"])),
        "",
    ]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
