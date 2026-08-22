#!/usr/bin/env python3
"""Rebuild `tickets.db` from the committed JSON fixtures.

Not a workshop step: the database is committed, so participants never run this — the labs only read
it. It exists for whoever edits `tickets.json` or `users.json`, and as the record of how the
database was produced. Same split as `generate_pdfs.py`, and as the reference workshop's
`data/data_generation/`: generation lives outside anything the labs import, so opening a database
can never regenerate it as a side effect.

Run from the repo root, then commit the result:
    uv run python day1/data/build_ticket_db.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from day1.src.ticket_db import DB_PATH, build_database, connect  # noqa: E402


def main() -> int:
    # force=True: an explicit rebuild must not depend on mtimes, which are unreliable straight after
    # a `git` checkout (git writes files in name order, so the .db can look older than its sources).
    path = build_database(force=True)
    with connect() as con:
        tables = [r["name"] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        counts = {t: con.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"] for t in tables}
        orphans = con.execute(
            "SELECT COUNT(*) c FROM tickets t LEFT JOIN users u ON t.user_id = u.user_id "
            "WHERE u.user_id IS NULL").fetchone()["c"]

    print(f"built {path.relative_to(ROOT)}  ({path.stat().st_size:,} bytes)")
    for table, n in counts.items():
        print(f"  {table:<18} {n:>4} rows")
    if orphans:
        print(f"⛔ {orphans} ticket(s) reference a missing user — check the JSON fixtures")
        return 1
    print("✅ no orphan references. Commit the rebuilt database, then run: "
          "uv run python verify_setup.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
