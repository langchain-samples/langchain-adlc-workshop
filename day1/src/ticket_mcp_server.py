"""A custom MCP server for the Acme ticket system.

Why this exists
---------------
The workshop agenda asks for an **MCP server integration pattern** in the Day 1 build, alongside RAG,
the ticket database, and the mock API. This is that server: a real
[Model Context Protocol](https://modelcontextprotocol.io) server, over stdio, exposing the ticket
system's *external-system* operations as MCP tools.

The point of the exercise is the boundary, not the tools. In Day 1 Lab 02 the agent's tools are Python
functions in the same process. Here they live behind a protocol: a separate process the agent talks to
over stdio, which it discovers at runtime. That is how an agent reaches a system it does not own —
an ITSM platform, an identity provider, SAP, a SQL database — and it is the shape Acme will actually
deploy, because the team that owns the ticket system is rarely the team that owns the agent.

What changes for the agent
--------------------------
Nothing. `MultiServerMCPClient.get_tools()` returns ordinary LangChain tools, so the same
`create_agent(...)` call works whether a tool is local or remote. That is the whole value of the
adapter, and the reason this lab is short.

Run it directly to serve over stdio (this is what the client spawns):

    uv run python day1/src/ticket_mcp_server.py

Security note: every tool below is a read or a *simulated* write over committed synthetic fixtures.
`update_ticket_status` mutates nothing on disk — it returns a receipt. A real deployment would put the
sensitive operations behind the same human-approval gate Lab 03 builds, on the agent side, because an
MCP server cannot know whether a human approved the call.
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Literal

# FastMCP logs every request at INFO and pydantic-settings emits a forward-ref warning on import.
# Both go to stderr, which the client surfaces — in a notebook that is a wall of noise around the
# one line the participant cares about. Quieten before importing the server.
warnings.filterwarnings("ignore", message=".*incomplete definition.*")
logging.getLogger("mcp").setLevel(logging.WARNING)

from mcp.server.fastmcp import FastMCP

logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TICKETS: list[dict] = json.loads((_DATA_DIR / "tickets.json").read_text())
USERS: list[dict] = json.loads((_DATA_DIR / "users.json").read_text())

mcp = FastMCP(
    "acme-ticket-system",
    log_level="WARNING",   # FastMCP logs every request at INFO; that is noise in a notebook
    instructions=(
        "The Acme IT ticket system. Use these tools to read ticket records, look up the queue a "
        "category routes to, check an SLA target, and record a status change. All data is synthetic."
    ),
)

# Which team owns which category — the kind of routing table that lives in the ITSM platform, not
# in the agent, and is exactly why you reach for it over a protocol rather than hardcoding it.
QUEUE_ROUTING: dict[str, dict[str, str]] = {
    "access":    {"queue": "IAM-ACCESS",      "owner": "Identity & Access Management"},
    "account":   {"queue": "IAM-ACCOUNT",     "owner": "Identity & Access Management"},
    "network":   {"queue": "NET-OPS",         "owner": "Network Operations"},
    "hardware":  {"queue": "EUC-HARDWARE",    "owner": "End User Computing"},
    "software":  {"queue": "EUC-SOFTWARE",    "owner": "End User Computing"},
    "knowledge": {"queue": "SERVICE-DESK",    "owner": "Service Desk (tier 1)"},
    "security":  {"queue": "SOC-TRIAGE",      "owner": "Security Operations Centre"},
}

# Response-time targets by priority, in hours. The agent should never invent these.
SLA_HOURS: dict[str, dict[str, int]] = {
    "P1": {"respond": 1, "resolve": 4},
    "P2": {"respond": 4, "resolve": 24},
    "P3": {"respond": 8, "resolve": 72},
    "P4": {"respond": 24, "resolve": 120},
}


@mcp.tool()
def get_ticket(ticket_id: str) -> str:
    """Fetch one ticket record by its ID (e.g. TKT-001) from the ticket system.

    Returns the full record: subject, description, category, priority, status and resolution.
    """
    row = next((t for t in TICKETS if t["ticket_id"].upper() == ticket_id.strip().upper()), None)
    if row is None:
        known = ", ".join(t["ticket_id"] for t in TICKETS[:3])
        return f"No ticket found for {ticket_id!r}. IDs look like {known}, …"
    return json.dumps(row, indent=2) + "\n[source: ticket system via MCP]"


@mcp.tool()
def route_to_queue(category: str) -> str:
    """Look up which support queue and owning team a ticket category routes to.

    Use this instead of guessing an owner — the routing table lives in the ticket system, and it
    changes without the agent being redeployed.
    """
    key = category.strip().lower()
    entry = QUEUE_ROUTING.get(key)
    if entry is None:
        return (f"Unknown category {category!r}. Valid categories: "
                f"{', '.join(sorted(QUEUE_ROUTING))}.")
    return (f"category: {key}\n  queue: {entry['queue']}\n  owning team: {entry['owner']}\n"
            f"  [source: ticket system routing table via MCP]")


@mcp.tool()
def get_sla(priority: Literal["P1", "P2", "P3", "P4"]) -> str:
    """Get the response and resolution SLA targets, in hours, for a ticket priority."""
    sla = SLA_HOURS.get(priority.strip().upper())
    if sla is None:
        return f"Unknown priority {priority!r}. Valid: P1, P2, P3, P4."
    return (f"priority: {priority.upper()}\n  respond within: {sla['respond']}h\n"
            f"  resolve within: {sla['resolve']}h\n  [source: ticket system SLA policy via MCP]")


@mcp.tool()
def update_ticket_status(
    ticket_id: str,
    new_status: Literal["open", "in_progress", "resolved", "escalated"],
    note: str = "",
) -> str:
    """Record a status change on a ticket. This is a WRITE to the ticket system.

    Returns a receipt for the audit trail. Because it changes state in an external system, gate it
    behind human approval on the agent side — the server cannot tell whether a human agreed.
    """
    row = next((t for t in TICKETS if t["ticket_id"].upper() == ticket_id.strip().upper()), None)
    if row is None:
        return f"No ticket found for {ticket_id!r}. Refusing to update an unknown ticket."
    return (
        f"⏳ STATUS CHANGE RECORDED (simulated)\n"
        f"  ticket: {row['ticket_id']} — {row['subject']}\n"
        f"  {row['status']} → {new_status}\n"
        f"  note: {note or 'none'}\n"
        f"  [source: ticket system via MCP · audit entry written]"
    )


@mcp.tool()
def whoami(user_id: str) -> str:
    """Resolve a user's role and permitted ticket categories from the ticket system's directory.

    The agent should call this before acting on someone's behalf. Exact-match only: a partial match
    that silently resolved to the wrong person would hand the agent the wrong authority.
    """
    q = user_id.strip().lower()
    user = next((u for u in USERS if q in (u["user_id"].lower(), u["name"].lower())), None) if q else None
    if user is None:
        return f"No user found for {user_id!r}. Ask for their user ID (e.g. USR-001)."
    return (f"{user['name']} ({user['user_id']}) — {user['department']}\n"
            f"  role: {user['role']}\n  permitted categories: {', '.join(user['permissions'])}\n"
            f"  [source: ticket system directory via MCP]")


if __name__ == "__main__":
    # stdio is the transport the workshop uses: the client spawns this file as a subprocess, so there
    # is no port, no auth and no network — the simplest thing that is still a real MCP server.
    # Anything printed to stdout would corrupt the protocol stream, so diagnostics go to stderr.
    # No banner on stdout OR stderr by default: the client spawns this process once per tool call
    # (there is no persistent session unless you use `client.session()`), so anything printed here
    # appears once per call and drowns the output the participant is reading.
    if "--verbose" in sys.argv:
        print("acme-ticket-system MCP server starting on stdio", file=sys.stderr)
    mcp.run(transport="stdio")
