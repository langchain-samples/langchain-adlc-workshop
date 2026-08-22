"""The Acme ticket system's **SQL database** — schema, build step, and query tools.

Why a real database
-------------------
The agenda asks for "ticket database / SQL lookup" and "SQL-style dynamic queries" in the Day 1
build, and the workshop this one is based on
([langsmith-agent-lifecycle-workshop](https://github.com/langchain-ai/langsmith-agent-lifecycle-workshop))
ships a real SQLite database (`data/structured/techhub.db`) queried by a dedicated SQL agent
(`agents/sql_agent.py`, `tools/database.py`). A real ticketing system is a database, so this is
that database rather than a keyword scan over a JSON list.

The teaching point is the *split*:

- **Structured questions** — "how many P1 tickets are still open?", "what is the mean time to
  resolution by category?" — are SQL. They need exact filters, joins, aggregates, and completeness.
  No amount of top-k similarity search answers them correctly.
- **Semantic questions** — "has anyone seen a VPN drop like this before?" — are RAG over the KB.

The Day 1 agent gets both, and choosing between them is the skill. See
`day3/src/06_knowledge_architecture.py` §1–2 for the decision procedure.

Data provenance — read this before trusting a column
----------------------------------------------------
`tickets.json` and `users.json` are the source of truth and are committed. This module builds
`day1/data/tickets.db` from them, and the database is **gitignored and regenerated on demand**, so
it can never drift from the JSON.

Three columns are *derived*, not source data, and are marked as such in `SCHEMA_DOC` so nobody
mistakes them for recorded fact:

- `tickets.resolved_at` — computed as `created_at` + the priority's SLA resolve target, for tickets
  in a terminal state only. It gives the aggregate queries something real to compute over.
- `tickets.queue` — looked up from the same category → queue routing table the MCP server uses
  (`ticket_mcp_server.QUEUE_ROUTING`), not invented here.
- `ticket_events` — one create row per ticket, plus one transition row for tickets that left
  `open`, reconstructed from the current status. It is a plausible audit trail, not a recorded one.

Everything else is copied verbatim from the JSON.

Security
--------
`execute_sql` is the general-purpose escape hatch, and is the one tool here that needs guarding:

- **read-only** — the connection is opened in SQLite's own read-only mode via a URI, so a write is
  rejected by the engine rather than by a string check that a determined prompt could talk its way
  around. The keyword check stays as a *fast, legible* first line of defence.
- **row-capped** — every result set is truncated, so a `SELECT *` over a large table cannot blow
  out the agent's context window.
- **parameterised** — every tool below binds user input with `?` placeholders. `execute_sql` takes
  a whole statement by design (that is the point of the tool), which is exactly why it runs on a
  read-only connection.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = _DATA_DIR / "tickets.db"
TICKETS_JSON = _DATA_DIR / "tickets.json"
USERS_JSON = _DATA_DIR / "users.json"

# Response/resolution targets by priority, in hours. Single source of truth, shared with the MCP
# server so the two never disagree about what a P1 promises.
SLA_HOURS: dict[str, dict[str, int]] = {
    "P1": {"respond": 1, "resolve": 4},
    "P2": {"respond": 4, "resolve": 24},
    "P3": {"respond": 8, "resolve": 72},
    "P4": {"respond": 24, "resolve": 120},
}

QUEUE_ROUTING: dict[str, str] = {
    "access": "IAM-ACCESS",
    "account": "IAM-ACCOUNT",
    "network": "NET-OPS",
    "hardware": "EUC-HARDWARE",
    "software": "EUC-SOFTWARE",
    "knowledge": "SERVICE-DESK",
    "security": "SOC-TRIAGE",
}

TERMINAL_STATUSES = ("resolved",)

SCHEMA_DDL = """
CREATE TABLE users (
    user_id     TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL,
    department  TEXT NOT NULL,
    role        TEXT NOT NULL
);

-- Permissions are normalised into their own table rather than a comma-joined string, so
-- entitlement checks are a JOIN instead of a LIKE. That is what makes row-level access control
-- expressible in SQL (see day3/src/05_governance.py §3).
CREATE TABLE user_permissions (
    user_id   TEXT NOT NULL REFERENCES users(user_id),
    category  TEXT NOT NULL,
    PRIMARY KEY (user_id, category)
);

CREATE TABLE tickets (
    ticket_id    TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(user_id),
    subject      TEXT NOT NULL,
    description  TEXT NOT NULL,
    category     TEXT NOT NULL,
    priority     TEXT NOT NULL,
    status       TEXT NOT NULL,
    resolution   TEXT,
    created_at   TEXT NOT NULL,
    resolved_at  TEXT,            -- DERIVED (see module docstring)
    queue        TEXT NOT NULL    -- DERIVED from category
);

-- A minimal audit trail. RECONSTRUCTED from current status, not recorded history.
CREATE TABLE ticket_events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id   TEXT NOT NULL REFERENCES tickets(ticket_id),
    ts          TEXT NOT NULL,
    actor       TEXT NOT NULL,
    from_status TEXT,
    to_status   TEXT NOT NULL,
    note        TEXT
);

CREATE INDEX idx_tickets_status   ON tickets(status);
CREATE INDEX idx_tickets_category ON tickets(category);
CREATE INDEX idx_tickets_priority ON tickets(priority);
"""

SCHEMA_DOC = """\
TABLE users(user_id PK, name, email, department, role)
TABLE user_permissions(user_id FK, category)          -- one row per permitted category
TABLE tickets(ticket_id PK, user_id FK, subject, description, category, priority,
              status, resolution, created_at, resolved_at*, queue*)
TABLE ticket_events(event_id PK, ticket_id FK, ts, actor, from_status, to_status, note)   -- *reconstructed*

  * = derived for this workshop, not recorded source data.

  status   ∈ open | in_progress | escalated | resolved
  priority ∈ P1 | P2 | P3 | P4      (resolve SLA: 4h | 24h | 72h | 120h)
  category ∈ access | account | hardware | knowledge | network | security | software
  Timestamps are ISO-8601 strings; use SQLite date functions (julianday, strftime) for arithmetic.
"""


def _parse_ts(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, tolerating a trailing 'Z' and a bare date."""
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return datetime.fromisoformat(text[:10])


def build_database(force: bool = False) -> Path:
    """Build `tickets.db` from the committed JSON fixtures. Idempotent.

    Maintainer tooling, not a lab step — the database ships, and nothing in the labs calls this.
    Run it after editing `tickets.json` or `users.json`, then commit the result;
    `verify_setup.py` checks the shipped copy against the JSON and will tell you if they diverge.

    Rebuilds when the database is missing or older than either JSON file. `force=True` rebuilds
    regardless, which is what the wrapper script uses — mtime ordering is not reliable straight
    after a `git` checkout, so an explicit rebuild should not depend on it.
    """
    sources = [p for p in (TICKETS_JSON, USERS_JSON) if p.exists()]
    if not force and DB_PATH.exists() and sources:
        if DB_PATH.stat().st_mtime >= max(p.stat().st_mtime for p in sources):
            return DB_PATH

    tickets = json.loads(TICKETS_JSON.read_text())
    users = json.loads(USERS_JSON.read_text())

    DB_PATH.unlink(missing_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        # SQLite ignores declared foreign keys unless this is switched on per connection. The build is
        # the only place anything is written, so this is where referential integrity is actually
        # enforceable: the INSERTs below would fail rather than quietly store a dangling user_id.
        # Readers open `file:...?mode=ro` and cannot write at all, so they need no equivalent.
        con.execute("PRAGMA foreign_keys = ON")
        con.executescript(SCHEMA_DDL)

        con.executemany(
            "INSERT INTO users (user_id, name, email, department, role) VALUES (?, ?, ?, ?, ?)",
            [(u["user_id"], u["name"], u["email"], u["department"], u["role"]) for u in users],
        )
        con.executemany(
            "INSERT INTO user_permissions (user_id, category) VALUES (?, ?)",
            [(u["user_id"], c) for u in users for c in u.get("permissions", [])],
        )

        rows, events = [], []
        for t in tickets:
            created = _parse_ts(t["created_at"])
            resolve_h = SLA_HOURS.get(t["priority"], {}).get("resolve", 24)
            terminal = t["status"] in TERMINAL_STATUSES
            resolved_at = (created + timedelta(hours=resolve_h)).isoformat() if terminal else None
            queue = QUEUE_ROUTING.get(t["category"], "SERVICE-DESK")

            rows.append((
                t["ticket_id"], t["user_id"], t["subject"], t["description"], t["category"],
                t["priority"], t["status"], t.get("resolution") or None,
                created.isoformat(), resolved_at, queue,
            ))
            events.append((t["ticket_id"], created.isoformat(), t["user_id"], None, "open", "ticket created"))
            if t["status"] != "open":
                ts = resolved_at or (created + timedelta(hours=resolve_h)).isoformat()
                events.append((t["ticket_id"], ts, queue, "open", t["status"], "status change (reconstructed)"))

        con.executemany(
            "INSERT INTO tickets (ticket_id, user_id, subject, description, category, priority, "
            "status, resolution, created_at, resolved_at, queue) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        con.executemany(
            "INSERT INTO ticket_events (ticket_id, ts, actor, from_status, to_status, note) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            events,
        )
        con.commit()
    finally:
        con.close()
    return DB_PATH


def connect(read_only: bool = True) -> sqlite3.Connection:
    """Open the shipped ticket database.

    `read_only=True` opens via SQLite's `file:...?mode=ro` URI, so writes are refused by the engine.
    That is the guarantee worth having: a prompt-injected `DROP TABLE` fails at the database, not at
    a string check in Python.

    Deliberately does NOT build. `tickets.db` is committed, so the labs only ever read it — the same
    split the reference workshop uses, where generation lives outside anything the labs import.
    An earlier version called `build_database()` here, which rebuilt on every fresh clone: `git`
    checks files out in name order, so `tickets.db` lands ~0.4 ms before `tickets.json`, and an
    mtime comparison read that write order as staleness. Regenerating derived data as a side effect
    of opening it is the wrong default anyway — see `day1/data/build_ticket_db.py` to rebuild.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"{DB_PATH} is missing. It ships with the repo — restore it with "
            f"`git checkout -- day1/data/tickets.db`, or rebuild it from the JSON fixtures with "
            f"`uv run python day1/data/build_ticket_db.py`."
        )
    if read_only:
        con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    else:
        con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _rows_to_text(rows: list[sqlite3.Row], empty: str) -> str:
    if not rows:
        return empty
    return "\n".join(
        " | ".join(f"{k}={r[k]}" for k in r.keys() if r[k] is not None) for r in rows
    )


# --------------------------------------------------------------------------------------------
# Query helpers. These are plain functions so they can be imported and unit-tested; each lab wraps
# the ones it needs with @tool, which is where the tool description (the thing the model actually
# reads) belongs.
# --------------------------------------------------------------------------------------------

MAX_ROWS = 25


def search_tickets(
    query: str = "",
    category: str | None = None,
    status: str | None = None,
    priority: str | None = None,
    limit: int = 5,
    for_user: str | None = None,
) -> str:
    """Search ticket history with real SQL filters.

    Every value is bound as a parameter — a ticket subject containing an apostrophe or a `--` is
    data, never syntax.

    `for_user` applies **row-level access control**: results are restricted to categories that user
    is entitled to, enforced in the WHERE clause. Filtering *before* the limit is what keeps recall
    honest — post-filtering a top-N would silently return fewer rows than the user is allowed.
    """
    sql = [
        "SELECT t.ticket_id, t.subject, t.category, t.priority, t.status, t.resolution, t.created_at",
        "FROM tickets t",
    ]
    where, params = [], []

    if for_user:
        sql.append("JOIN user_permissions p ON p.category = t.category AND p.user_id = ?")
        params.append(for_user)
    if query:
        where.append("(t.subject LIKE ? OR t.description LIKE ? OR IFNULL(t.resolution,'') LIKE ?)")
        params += [f"%{query}%"] * 3
    for column, value in (("t.category", category), ("t.status", status), ("t.priority", priority)):
        if value:
            where.append(f"{column} = ?")
            params.append(value)

    if where:
        sql.append("WHERE " + " AND ".join(where))
    # Live tickets first, then most recent — an agent triaging a ticket cares about open work.
    sql.append("ORDER BY CASE t.status WHEN 'escalated' THEN 0 WHEN 'open' THEN 1 "
               "WHEN 'in_progress' THEN 2 ELSE 3 END, t.created_at DESC")
    sql.append("LIMIT ?")
    params.append(max(1, min(int(limit), MAX_ROWS)))

    with connect() as con:
        rows = con.execute("\n".join(sql), params).fetchall()

    scoped = f" visible to {for_user}" if for_user else ""
    header = f"{len(rows)} ticket(s){scoped} [source: ticket database (SQL)]"
    return header + "\n" + _rows_to_text(rows, "  (no matching tickets)")


def ticket_stats(group_by: str = "category") -> str:
    """Aggregate statistics — the questions RAG cannot answer.

    Returns, per group: total, open, in_progress, escalated, resolved, unresolved, and mean hours
    to resolution. Every count is named for exactly what it counts, so no subtraction is needed to
    read the answer. `group_by` is validated against an allow-list because a column name cannot be
    a bound parameter.
    """
    allowed = {"category", "priority", "status", "queue"}
    col = group_by.strip().lower()
    if col not in allowed:
        return f"group_by must be one of {sorted(allowed)}; got {group_by!r}"

    # Column names are the tool's contract with the model. An earlier version returned a column
    # called `tickets` (meaning *total*) next to `resolved`, and the agent read it as "still open" —
    # answering "5 P1 tickets are open" when 1 was open and 2 unresolved. Name every column for
    # exactly what it counts, and return `open`/`unresolved` explicitly rather than making the model
    # subtract. A tool that requires arithmetic to interpret will eventually be misinterpreted.
    sql = f"""
        SELECT {col} AS grp,
               COUNT(*) AS total,
               SUM(status = 'open')        AS open_now,
               SUM(status = 'in_progress') AS in_progress,
               SUM(status = 'escalated')   AS escalated,
               SUM(status = 'resolved')    AS resolved,
               SUM(status != 'resolved')   AS unresolved,
               ROUND(AVG(CASE WHEN resolved_at IS NOT NULL
                    THEN (julianday(resolved_at) - julianday(created_at)) * 24 END), 1) AS avg_resolve_h
        FROM tickets
        GROUP BY {col}
        ORDER BY total DESC
    """
    with connect() as con:
        rows = con.execute(sql).fetchall()

    out = [f"ticket statistics by {col} [source: ticket database (SQL aggregate)]",
           "  columns: total = all tickets · open/in_progress/escalated = live · "
           "unresolved = total - resolved",
           f"  {'group':<12}{'total':>7}{'open':>6}{'in_prog':>9}{'escal':>7}"
           f"{'resolved':>10}{'unresolved':>12}{'avg resolve h':>15}"]
    for r in rows:
        avg = "—" if r["avg_resolve_h"] is None else f"{r['avg_resolve_h']:.1f}"
        out.append(f"  {r['grp']:<12}{r['total']:>7}{r['open_now']:>6}{r['in_progress']:>9}"
                   f"{r['escalated']:>7}{r['resolved']:>10}{r['unresolved']:>12}{avg:>15}")
    return "\n".join(out)


FORBIDDEN = ("insert", "update", "delete", "drop", "alter", "create", "replace",
             "truncate", "attach", "pragma")


def execute_sql(query: str, limit: int = MAX_ROWS) -> str:
    """Run an arbitrary read-only SELECT against the ticket database.

    This is the tool that makes "SQL-style dynamic queries" real: the model writes the query. Three
    layers keep that safe, and the order matters — the last one is the only one that is a guarantee.

    1. Multiple statements rejected outright (no `SELECT 1; DROP TABLE …`).
    2. Must start with SELECT or WITH (fast, legible).
    3. No mutating keyword anywhere.
    4. Every result set is row-capped, so a broad SELECT cannot flood the context window.
    5. The connection is opened **read-only at the SQLite level**, so even a statement that defeats
       1-3 cannot write. Checks 1-3 give a good error message; check 5 is the actual control.

    This follows the official guidance rather than the reference implementation. LangChain's
    security docs say to "consider using read-only credentials" and to "combine multiple layered
    security approaches rather than relying on any single layer of defense"; the SQL-agent tutorial
    additionally rejects multiple statements and caps rows. The workshop this repo is modelled on
    (`langsmith-agent-lifecycle-workshop`, `tools/database.py`) uses only checks 2-3 and builds its
    other queries with f-string interpolation — that is a teaching simplification, not a pattern to
    copy into an Acme deployment.
    """
    q = query.strip()

    # 1a. Reject MULTIPLE STATEMENTS before anything else. One trailing `;` is fine; a second
    # statement is not. Python's sqlite3 driver already refuses to run two statements in one
    # `execute()`, but relying on that is relying on a single undocumented layer — the official
    # SQL-agent guidance is explicit that defences must be layered, and that driver behaviour is
    # an implementation detail, not a security control. Checking here also produces an error the
    # model can act on rather than a driver exception.
    # Ref: https://docs.langchain.com/oss/python/langchain/agents  (security: limit permissions,
    # combine layered approaches) — see also the SQL agent tutorial's `sanitizeSqlQuery`.
    if q.count(";") > 1 or (";" in q.rstrip(";") ):
        return "Refused: multiple statements are not allowed. Send one SELECT."
    q = q.rstrip(";").strip()
    low = q.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return "Refused: only SELECT (or WITH … SELECT) queries are allowed."
    if any(f" {kw} " in f" {low} " or low.startswith(kw) for kw in FORBIDDEN):
        return f"Refused: query contains a non-read-only keyword. Allowed: SELECT/WITH over {sorted(_tables())}."

    try:
        with connect(read_only=True) as con:
            rows = con.execute(q).fetchmany(max(1, min(int(limit), MAX_ROWS)))
    except sqlite3.Error as exc:
        # Hand the model the real error: it can usually repair its own SQL from the message.
        return f"SQL error: {exc}\n\nSchema:\n{SCHEMA_DOC}"

    return (f"{len(rows)} row(s) [source: ticket database (SQL)]\n"
            + _rows_to_text(rows, "  (no rows)"))


def _tables() -> list[str]:
    with connect() as con:
        return [r["name"] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]


if __name__ == "__main__":
    # Reports on the shipped database; it does not rebuild. Rebuilding lives in
    # `day1/data/build_ticket_db.py` so that generation is never a side effect of inspection.
    path = DB_PATH
    with connect() as con:
        counts = {t: con.execute(f"SELECT COUNT(*) c FROM {t}").fetchone()["c"] for t in _tables()}
    print(f"shipped database: {path}  ({path.stat().st_size:,} bytes)")
    for table, n in counts.items():
        print(f"  {table:<18} {n:>4} rows")
    print("\n" + ticket_stats("category"))
