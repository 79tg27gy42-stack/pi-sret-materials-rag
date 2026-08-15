"""Compatibility entry point for scoring the Phi extraction gold audit."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_extraction_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
