"""Deployable ticket resolution agent — single source of truth for the workshop Day 1.

This graph is built over synthetic fixtures in `langchain_adlc_workshop/day1/data/` so the labs can teach
customer support ticket resolution without real Acme data.

In `langchain>=1.0`, `create_agent` returns a compiled LangGraph graph. The module-level graphs below
are registered in `langgraph.json` for LangGraph Studio and reused by notebooks/evals.

Security: all tools are pure-Python lookups/searches over committed JSON/Markdown fixtures (no SQL,
eval, network, shelling out). The system prompt is pulled with an offline-safe fallback via
`utils.prompts`; API keys are read only by the model provider at runtime.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

# The workshop root (this file is day1/src/<name>.py) must be importable before `utils.*` and
# `day1.src.*` resolve. langgraph dev / langgraph build load this file by path, so bootstrap it here
# rather than relying on the caller's cwd or PYTHONPATH. The `globals()` guard keeps the notebook
# render of this module runnable too — a Jupyter kernel has no `__file__`.
if "__file__" in globals():
    _WORKSHOP = Path(globals()["__file__"]).resolve().parent.parent.parent  # src -> day1 -> root
else:  # notebook render: walk up from the kernel's cwd instead
    _WORKSHOP = next(
        c for c in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
        if (c / "day1").is_dir() and (c / "utils").is_dir()
    )
if str(_WORKSHOP) not in sys.path:
    sys.path.insert(0, str(_WORKSHOP))

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from day1.src.models import get_embeddings, get_model
from utils.prompts import get_prompt

_DATA_DIR = _WORKSHOP / "day1" / "data"
KB_DIR = _DATA_DIR / "kb_tickets"
TICKETS_PATH = _DATA_DIR / "tickets.json"
USERS_PATH = _DATA_DIR / "users.json"
PROMPT_PATH = _DATA_DIR / "prompt_ticket.md"

PROMPT_NAME = "ticket-resolution"

# Ticket rows now live in the SQL database (day1/src/ticket_db.py builds it from this JSON);
# the path stays so the build step and the fixtures have one owner.
USERS: list = json.loads(USERS_PATH.read_text())



# --- Tool 1: search_kb (RAG over the bundled synthetic IT support KB articles) ---

def _build_kb_index() -> InMemoryVectorStore:
    docs = [
        Document(page_content=p.read_text(), metadata={"source": p.name})
        for p in sorted(KB_DIR.glob("*.md"))
    ]
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
    return InMemoryVectorStore.from_documents(chunks, get_embeddings())


_kb_index: InMemoryVectorStore | None = None


def _get_kb_index() -> InMemoryVectorStore:
    global _kb_index
    if _kb_index is None:
        _kb_index = _build_kb_index()
    return _kb_index


@tool
def search_kb(query: str) -> str:
    """Search IT support knowledge base articles by issue, error message, or keyword
    (MFA reset, VPN troubleshooting, software installation, account lockout, etc.).
    Returns cited snippets from the knowledge base."""
    hits = _get_kb_index().similarity_search(query, k=4)
    if not hits:
        return "No relevant KB articles found."
    return "\n\n---\n\n".join(f"[source: {h.metadata['source']}]\n{h.page_content}" for h in hits)


# --- Tool 2-4: the ticket DATABASE (SQL over day1/data/tickets.db) ---
#
# A real ticketing system is a database, and the agenda asks for "ticket database / SQL lookup"
# and "SQL-style dynamic queries". `day1/src/ticket_db.py` builds a normalised SQLite database
# from the committed JSON fixtures; these three tools are the agent's window onto it.
#
# The escalation is deliberate: a parameterised search, then aggregates, then a tool where the
# model writes SQL itself — each with a tighter guardrail than the last.

from day1.src.ticket_db import execute_sql as _execute_sql
from day1.src.ticket_db import search_tickets as _search_tickets
from day1.src.ticket_db import ticket_stats as _ticket_stats


@tool
def search_ticket_history(
    query: str = "",
    category: str | None = None,
    status: str | None = None,
    priority: str | None = None,
) -> str:
    """Search the ticket database for similar past or currently live tickets.

    Matches `query` against subject, description and resolution, and narrows by any of category
    (access, account, hardware, knowledge, network, security, software), status (open,
    in_progress, escalated, resolved) or priority (P1-P4). Escalated and open tickets are
    returned first: a live ticket on the same symptom is usually more useful than a closed one.
    """
    return _search_tickets(query=query, category=category, status=status, priority=priority)


@tool
def ticket_statistics(group_by: str = "category") -> str:
    """Aggregate ticket statistics: ticket counts, resolved counts and mean hours to resolution.

    Group by one of: category, priority, status, queue. Use this for "how many", "what share of"
    or "on average" questions — keyword search cannot answer those correctly or completely.
    """
    return _ticket_stats(group_by)


@tool
def query_ticket_db(query: str) -> str:
    """Run a read-only SQL SELECT against the ticket database, for questions the other tools cannot express.

    Use for joins and custom aggregates, e.g.
      SELECT u.department, COUNT(*) AS n FROM tickets t
      JOIN users u ON u.user_id = t.user_id GROUP BY u.department ORDER BY n DESC

    Schema:
    TABLE users(user_id, name, email, department, role)
    TABLE user_permissions(user_id, category)
    TABLE tickets(ticket_id, user_id, subject, description, category, priority, status,
                  resolution, created_at, resolved_at, queue)
    TABLE ticket_events(event_id, ticket_id, ts, actor, from_status, to_status, note)

    Only SELECT / WITH are permitted and the connection is opened read-only.
    """
    return _execute_sql(query)


# --- Tool 3: get_user_context (role + permissions lookup — authorization grounding) ---

def _find_user(users: list[dict], user_id: str) -> dict | None:
    """Resolve a user by exact `user_id`, else by full name.

    Deliberately NOT a substring match: `"a" in "ana costa"` is true, so a substring lookup lets a
    blank or one-character argument resolve to the first user in the file — which in this fixture is
    the admin. An authorization-grounding tool that can be nudged into the wrong identity is worse
    than no tool at all, so unknown input returns None and the caller says so.
    """
    q = user_id.strip().lower()
    if not q:
        return None
    for u in users:
        if q == u["user_id"].lower() or q == u["name"].lower():
            return u
    return None


@tool
def get_user_context(user_id: str) -> str:
    """Look up a user by ID (e.g. USR-001) or full name (e.g. "Ana Costa"). Returns their role,
    department, and the ticket categories they are permitted to access. Use this to verify the
    requester is authorized before returning ticket details or recommending account actions."""
    u = _find_user(USERS, user_id)
    if u is None:
        return f"No user found for {user_id!r}. Ask the requester for their user ID (e.g. USR-001)."
    return (
        f"{u['name']} ({u['user_id']}) — {u['department']} · role: {u['role']}\n"
        f"  email: {u['email']}\n"
        f"  permissions: {', '.join(u['permissions'])}\n"
        f"  [source: users.json]"
    )


# --- Tool 4: mock_api_action (sensitive account actions — always human-approved) ---

SENSITIVE_ACTIONS = ("mfa_reset", "account_unlock", "password_reset")


@tool
def mock_api_action(action: str, user_id: str, reason: str = "") -> str:
    """Execute a sensitive account action against the (mock) identity API: mfa_reset,
    account_unlock, or password_reset. These actions temporarily weaken account security, so this
    tool should always be gated by human approval (HumanInTheLoopMiddleware) before it runs.
    Returns a pending-action receipt for the audit log."""
    user = _find_user(USERS, user_id)
    if user is None:
        # Never guess the target of a sensitive action.
        return f"No user found for {user_id!r}. Refusing to queue {action!r} without a confirmed user."
    if action not in SENSITIVE_ACTIONS:
        return f"Unknown action {action!r}. Supported actions: {', '.join(SENSITIVE_ACTIONS)}."
    return (
        f"⏳ PENDING ACTION — {action} for {user['name']} ({user['user_id']})\n"
        f"  status: queued · requires: human approval\n"
        f"  reason: {reason or 'not specified'}\n"
        f"  audit: action logged for review\n"
        f"  [source: mock identity API]"
    )


def ticket_tools() -> list:
    """The agent's four grounding tools plus optional Tavily when key is set."""
    tools = [search_kb, search_ticket_history, ticket_statistics, query_ticket_db,
             get_user_context, mock_api_action]
    if os.getenv("TAVILY_API_KEY"):
        from langchain_tavily import TavilySearch

        tools.append(TavilySearch(max_results=3))
    return tools


# --- Structured output schema ---

class TicketResolution(BaseModel):
    """Structured output for the ticket resolution agent. Passed as `response_format=` so
    LangChain enforces it and returns a parsed object at `result["structured_response"]`."""

    issue_summary: str = Field(
        description="One-paragraph summary of the user's issue"
    )
    category: Literal["access", "account", "hardware", "knowledge", "network", "software", "security"] = Field(
        description="Ticket category"
    )
    kb_references: list[str] = Field(
        default_factory=list,
        description="KB article filenames cited in the resolution"
    )
    similar_tickets: list[str] = Field(
        default_factory=list,
        description="ticket_ids of similar past tickets"
    )
    recommended_action: str = Field(
        description="The recommended next step or resolution"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the resolution"
    )
    requires_hitl: bool = Field(
        description="True when the recommended action is sensitive (e.g. MFA reset, account unlock)"
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description="Information that would improve confidence if provided"
    )
    user_id: str = Field(
        description="The requesting user's ID, for the audit trail"
    )


# --- Build the agents ---

def build_agent(*, use_checkpointer: bool = False, interrupt_on: dict | None = None,
                response_format: type | None = None, middleware: list | None = None,
                name: str = "ticket_agent"):
    """Build the ticket resolution agent.

    Args:
        use_checkpointer: attach an in-memory checkpointer for local memory/threads + HITL.
        interrupt_on: optional HumanInTheLoopMiddleware config mapping tool name -> approval policy.
        response_format: optional Pydantic model for structured output.
        middleware: optional additional middleware list.
        name: the graph's internal name.
    """
    system_prompt = get_prompt(PROMPT_NAME, fallback=PROMPT_PATH.read_text().strip())

    mw = list(middleware) if middleware else []
    if interrupt_on:
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        mw.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    kwargs = {"middleware": mw} if mw else {}
    if use_checkpointer:
        from langgraph.checkpoint.memory import MemorySaver

        kwargs["checkpointer"] = MemorySaver()

    if response_format is not None:
        from langchain.agents.structured_output import ToolStrategy

        # `ToolStrategy` rather than bare `response_format=TicketResolution`. The default `AutoStrategy`
        # picks the provider's *native* JSON mode when available, and that path is measurably flaky here: in
        # clean-clone testing it raised `Native structured output expected valid JSON … Extra data` on roughly
        # one run in three, because the model appended prose after the JSON object. Tool-calling emits a
        # schema-validated tool call instead, and `handle_errors=True` feeds a parse failure back to the model
        # to retry rather than killing the lab mid-demo.
        response_format = ToolStrategy(response_format, handle_errors=True)

    return create_agent(
        get_model(),
        tools=ticket_tools(),
        system_prompt=system_prompt,
        response_format=response_format,
        name=name,
        **kwargs,
    )


# Module-level compiled graphs — importable by notebooks and deployable via langgraph.json.
# These are built at import time (requires API key) so langgraph dev can find them.
# If you don't have an API key set, import will fail — this is expected for langgraph dev.
#
# Note: langgraph dev handles persistence automatically — no custom checkpointer needed.
# For local testing with memory, use build_agent(use_checkpointer=True) directly.

graph = build_agent()  # plain ReAct agent
# `ticket_agent_mem` deliberately has NO custom checkpointer: LangGraph Server rejects graphs
# that bring their own (persistence is the platform's job, keyed by thread_id). For local
# notebook memory, call build_agent(use_checkpointer=True) — see Lab 03 §2.
graph_with_memory = build_agent(name="ticket_agent_mem")
graph_hitl = build_agent(
    interrupt_on={"mock_api_action": True},
    name="ticket_agent_hitl",
)
graph_structured = build_agent(response_format=TicketResolution, name="ticket_agent_structured")
