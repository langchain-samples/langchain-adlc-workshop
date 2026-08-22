#!/usr/bin/env python3
"""Day 1 setup check — run this before you start Day 1.

    uv run python day1/verify_setup.py

Checks only what Day 1 needs: model access, and the fixtures and tooling those labs read. A
three-day workshop edits its own data between days, so passing on Day 1 does not prove Day 1 will
run — re-run this each morning. Equivalent to `uv run python verify_setup.py --day 1` from the
repo root; `verify_setup.py` with no argument checks all three days at once.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import verify_setup  # noqa: E402  — the shared implementation lives at the repo root

if __name__ == "__main__":
    verify_setup.DAY = 1
    raise SystemExit(verify_setup.main())
