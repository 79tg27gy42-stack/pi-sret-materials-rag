from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sret_materials_rag.experiments.exp_a_h1_divergence import run


if __name__ == "__main__":
    print(run(ROOT / "configs/materials_qwen_max_naive.yaml"))
