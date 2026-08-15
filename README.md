# PI-SRET: Constraint-Aware Diagnostics for Materials RAG

This repository contains the code, released data, and reported outputs for PI-SRET, an executable diagnostic framework for checking materials-domain constraints in retrieval-augmented generation (RAG) responses. PI-SRET complements context-support measures by identifying cases where a response is supported by retrieved text but violates an implemented materials-science constraint.

## Contents

- `src/sret_materials_rag/`: PI-SRET extraction, constraint, support-proxy, and statistical evaluation code.
- `scripts/`: dataset preparation, evaluation, analysis, and figure-generation entry points.
- `configs/`: experiment and model configuration files.
- `data/processed/canonical_v3/`: canonical response-level evaluation dataset and fixed held-out split.
- `data/annotations/`: adjudicated blinded materials-review reference.
- `data/audits/`: extraction-audit gold annotations and sampling metadata.
- `data/sources/`: source records and provenance logs used to construct the evaluation material.
- `results/`: cached scores and aggregate outputs reported in the study.
- `docs/`: dataset schema, collection protocol, extraction-audit guidance, and rule specification.
- `tests/`: regression tests for the public implementation and released outputs.

## Installation

Python 3.10 or later is required. The project uses `uv` for environment management.

```bash
uv sync --extra dev
```

API credentials are only required for data collection or provider-backed generation. Copy `.env.example` to `.env` and set the relevant variables locally; `.env` is ignored by Git.

## Reproducibility Checks

Validate the canonical release:

```bash
uv run python scripts/validate_canonical_evaluation_dataset_v3.py
```

Recompute deterministic PI-SRET labels from the released cached answers and scores:

```bash
uv run python scripts/recompute_pi_sret_labels.py
```

Re-score the released fixed-context, three-system held-out split into a separate directory:

```bash
uv run python scripts/run_frozen_heldout_rag_benchmark.py \
  --output-dir results/reproduced_frozen_heldout
```

Recompute extraction-audit metrics:

```bash
uv run python scripts/evaluate_extraction_audit.py \
  --input data/audits/extraction_gold_audit.csv \
  --output-dir results/reproduced_extraction_audit
```

Run the regression suite:

```bash
uv run pytest
```

## Data Scope

`canonical_v3` separates base scientific tasks from response records and includes controlled Materials Project-derived records, source-linked arXiv records, cached generator responses, and fixed evaluation subsets. The released expert-review file contains adjudicated labels and rationale fields. Private reviewer-to-response linkage material and working annotation files are intentionally excluded.

The source-linked records retain provenance information. Users are responsible for complying with the terms of the underlying data providers when redistributing or extending the dataset.

## Rule Coverage

The released PI-SRET implementation covers diagnostic checks for band gap, formation energy, stability, chemical formula, temperature, pressure, conductivity/band-gap consistency, crystal structure, and superconductivity claims. The formal applicability, breach, exception, normalization, and threshold definitions are in [docs/rule_specification.md](docs/rule_specification.md).

## Citation

Citation metadata will be added when a persistent repository identifier is available.
