"""Prepare and optionally score a frozen Materials Project external evaluation.

The default mode creates a provenance-rich external question/context set from
the Materials Project candidate table and records the frozen evaluator state.
It does not fabricate prospective model answers. To score an actual external
run, provide a CSV with ``external_id`` and ``answer`` columns generated after
the freeze.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MP_SOURCE = ROOT / "data" / "sources" / "materials_project_candidates_v3.csv"
DEFAULT_OUT = ROOT / "results" / "external_mp_prospective"

import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sret_materials_rag.evaluation.constraints import evaluate_material_qa_constraints  # noqa: E402
from sret_materials_rag.evaluation.faithfulness_evaluators import (  # noqa: E402
    LexicalOverlapFaithfulnessEvaluator,
    NLIHeuristicFaithfulnessEvaluator,
)
from sret_materials_rag.utils.llm_client import build_qwen_client  # noqa: E402


PROMPT_VERSION = "external_mp_context_answer_v1"
GENERATION_PROMPT = (
    "Answer the materials-science question using only the provided context. "
    "Be concise and preserve units. If the context is insufficient, state that the context does not specify the answer."
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def _context(row: pd.Series) -> str:
    stable = "stable" if bool(row["is_stable"]) else "not stable"
    return (
        f"Materials Project record {row.material_id} reports {row.formula_pretty} with "
        f"band gap {row.band_gap} eV, formation energy {row.formation_energy_per_atom} eV/atom, "
        f"energy above hull {row.energy_above_hull} eV/atom, and stability status {stable}. "
        f"Source: {row.source_url}"
    )


def build_external_set(n: int, seed: int) -> pd.DataFrame:
    source = pd.read_csv(MP_SOURCE)
    source = source.drop_duplicates("material_id").reset_index(drop=True)
    if n > len(source):
        raise ValueError(f"Requested {n} rows but only {len(source)} MP candidates are available.")
    sample = source.sample(n=n, random_state=seed).reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for index, row in enumerate(sample.itertuples(index=False), start=1):
        base = row._asdict()
        external_id = f"mp_external_{index:04d}"
        for task, question in [
            ("band_gap", f"What band gap is reported for {row.formula_pretty}?"),
            ("formation_energy", f"What formation energy is reported for {row.formula_pretty}?"),
            ("stability", f"Is {row.formula_pretty} stable according to this record?"),
        ]:
            rows.append(
                {
                    "external_id": f"{external_id}_{task}",
                    "material_id": row.material_id,
                    "formula_pretty": row.formula_pretty,
                    "constraint_family": task,
                    "question": question,
                    "retrieved_context": _context(pd.Series(base)),
                    "source_type": row.source_type,
                    "source_url": row.source_url,
                }
            )
    return pd.DataFrame(rows)


def score_answers(instances: pd.DataFrame, answers: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    merged = instances.merge(answers[["external_id", "answer"]], on="external_id", how="left", validate="one_to_one")
    if merged["answer"].isna().any():
        missing = merged.loc[merged["answer"].isna(), "external_id"].head(10).tolist()
        raise ValueError(f"Missing answers for {merged['answer'].isna().sum()} instances, examples={missing}")
    lexical = LexicalOverlapFaithfulnessEvaluator()
    nli = NLIHeuristicFaithfulnessEvaluator()
    scored = []
    for row in merged.itertuples(index=False):
        constraint = evaluate_material_qa_constraints(row.question, row.answer)
        lexical_score = lexical.score(question=row.question, answer=row.answer, retrieved_context=row.retrieved_context).score
        nli_score = nli.score(question=row.question, answer=row.answer, retrieved_context=row.retrieved_context).score
        scored.append(
            {
                **row._asdict(),
                "pi_sret_constraint_score": constraint.score,
                "pi_sret_violations": ";".join(constraint.violations),
                "lexical_support": lexical_score,
                "nli_proxy_support": nli_score,
            }
        )
    frame = pd.DataFrame(scored)
    metrics = {
        "n": int(len(frame)),
        "unique_materials": int(frame["material_id"].nunique()),
        "violation_rate": float((frame["pi_sret_constraint_score"] == 0).mean()),
        "constraint_pass_rate": float((frame["pi_sret_constraint_score"] == 1).mean()),
        "family_distribution": frame["constraint_family"].value_counts().sort_index().to_dict(),
        "mean_lexical_support": float(frame["lexical_support"].mean()),
        "mean_nli_proxy_support": float(frame["nli_proxy_support"].mean()),
        "evidence_level": "prospective_or_supplied_answers_scored_after_evaluator_freeze",
    }
    return frame, metrics


def generate_answers(
    instances: pd.DataFrame,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    output_path: Path,
) -> pd.DataFrame:
    client = build_qwen_client(model=model)
    rows: list[dict[str, object]] = []
    if output_path.exists():
        existing = pd.read_csv(output_path)
        rows.extend(existing.to_dict("records"))
        done = set(existing["external_id"].astype(str))
        instances = instances.loc[~instances["external_id"].astype(str).isin(done)].reset_index(drop=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    for index, row in enumerate(instances.itertuples(index=False), start=1):
        response = client.chat(
            [
                {"role": "system", "content": GENERATION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{row.question}\n\n"
                        f"Context:\n{row.retrieved_context}\n\n"
                        "Answer:"
                    ),
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        rows.append(
            {
                "external_id": row.external_id,
                "answer": response.content.strip(),
                "model": model,
                "provider": "DashScope/OpenAI-compatible",
                "generated_at_utc": generated_at,
                "prompt_version": PROMPT_VERSION,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompt": GENERATION_PROMPT,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
                "sequence_index": index,
            }
        )
        if index % 25 == 0:
            print(json.dumps({"generated": index, "total": len(instances)}, ensure_ascii=False))
        pd.DataFrame(rows).to_csv(output_path, index=False)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-materials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--answers", type=Path, help="CSV with external_id and answer columns from post-freeze generation.")
    parser.add_argument("--generate", action="store_true", help="Generate post-freeze answers with the configured Qwen-compatible chat API before scoring.")
    parser.add_argument("--model", default="qwen-max")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--limit-instances", type=int, help="Generate/score only the first N instances for a bounded prospective run.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    instances = build_external_set(args.n_materials, args.seed)
    if args.limit_instances is not None:
        instances = instances.head(args.limit_instances).reset_index(drop=True)
    instances.to_csv(args.output_dir / "external_mp_instances.csv", index=False)

    freeze = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "rule_version": "src/sret_materials_rag/evaluation/constraints.py",
        "thresholds": {
            "band_gap_upper_ev": 15.0,
            "formation_energy_lower_ev_atom": -4.0,
            "formation_energy_upper_ev_atom": 1.0,
            "conductivity_gap_ev": 0.1,
            "elemental_reference_tolerance_ev_atom": 0.05,
        },
        "source": str(MP_SOURCE),
        "n_instances": int(len(instances)),
        "n_materials": int(instances["material_id"].nunique()),
        "semantic_estimators": ["lexical_overlap", "nli_heuristic"],
        "prompt_version": PROMPT_VERSION,
        "generation_prompt": GENERATION_PROMPT,
    }
    (args.output_dir / "freeze_manifest.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    answers_path = args.answers
    if args.generate:
        answers_path = args.output_dir / "external_mp_answers.csv"
        generate_answers(
            instances,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output_path=answers_path,
        )

    if answers_path is None:
        summary = {
            **freeze,
            "status": "prepared_not_scored",
            "reason": "No post-freeze model answer CSV was supplied; no external performance metric is reported.",
            "required_answer_columns": ["external_id", "answer"],
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    scored, metrics = score_answers(instances, pd.read_csv(answers_path))
    metrics.update(
        {
            "status": "scored",
            "answers_file": str(answers_path),
            "model": args.model if args.generate else "user_supplied",
            "temperature": args.temperature if args.generate else None,
            "max_tokens": args.max_tokens if args.generate else None,
            "prompt_version": PROMPT_VERSION if args.generate else None,
            "high_support_low_g_lexical_n": int(((scored["lexical_support"] >= 0.8) & (scored["pi_sret_constraint_score"] < 1.0)).sum()),
            "high_support_low_g_nli_n": int(((scored["nli_proxy_support"] >= 0.8) & (scored["pi_sret_constraint_score"] < 1.0)).sum()),
            "high_support_low_g_lexical_rate": float(((scored["lexical_support"] >= 0.8) & (scored["pi_sret_constraint_score"] < 1.0)).mean()),
            "high_support_low_g_nli_rate": float(((scored["nli_proxy_support"] >= 0.8) & (scored["pi_sret_constraint_score"] < 1.0)).mean()),
        }
    )
    scored.to_csv(args.output_dir / "external_mp_scores.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
