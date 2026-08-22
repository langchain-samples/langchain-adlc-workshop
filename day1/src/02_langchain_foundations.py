# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02 · LangChain Foundations + Middleware — Ticket Resolution
#
# **Workshop:** LangChain ADLC Workshop · **Day 1** · **ADLC stage:** Build
#
# > **Loop Engineering focus: Agent loop** — the model uses tools, instructions, and state/workflow
# > to complete a task. This lab builds the agent loop from scratch.
#
# A support engineer describes a user issue and expects a **cited, evidence-backed** resolution plan.
# We build that agent with `create_agent`: a ReAct loop that reasons, calls tools, and cites the
# knowledge base and ticket history.
#
# ```mermaid
# graph LR
#     A[Support engineer] -->|describes issue| B[Agent]
#     B -->|RAG query| C[search_kb]
#     B -->|similar tickets| D[search_ticket_history]
#     B -->|user role/permissions| E[get_user_context]
#     B -->|sensitive action| F[mock_api_action]
#     B -->|web search| G[tavily_search]
#     C --> H[KB ticket articles]
#     D --> I[(tickets.db · SQLite)]
#     E --> J[users.json]
#     F --> K[Mock external API]
#     B -->|structured output| L[TicketResolution]
# ```
#
# | Tool | Does |
# |---|---|
# | `search_kb` | RAG over IT support KB articles → cited snippets |
# | `search_ticket_history` | Lookup similar past tickets (resolved and live) by category and keyword |
# | `get_user_context` | User role and permissions lookup from `users.json` |
# | `mock_api_action` | Mock external API for sensitive actions (MFA reset, account unlock) — returns pending action requiring HITL |
# | `tavily_search` *(optional)* | live web for anything outside the bundled data |
#
# > Self-directed module · ~15 min. By the end you can:
# > - build a ticket resolution agent with `create_agent` (a ReAct loop over your tools)
# > - wire RAG + lookup + action tools and render the agent's graph
# > - add **LangChain middleware** for PII redaction and audit logging
# > - open a traced run and the prompt in LangSmith
# >
# > 🧭 **Builds on Lab 01; runs standalone.** Data is **synthetic**: tickets, users, and KB articles
# > are fictional and created for this workshop.


# %% [markdown]
# ### 📚 Stuck on syntax? Reference material
#
# You are not expected to write any of this from memory. When an API signature is the thing in your
# way, look it up — that is what a working engineer does, and every link below is the official source.
#
# | Need | Where to look |
# |---|---|
# | `create_agent(...)` signature | [API reference](https://reference.langchain.com/python/langchain/agents/factory/create_agent) |
# | Tools: `@tool`, docstrings, args | [Tools guide](https://docs.langchain.com/oss/python/langchain/tools) |
# | Middleware hooks (`@before_model`, PII, HITL) | [Middleware guide](https://docs.langchain.com/oss/python/langchain/middleware) |
# | `StateGraph`, `add_node`, `add_conditional_edges` | [Graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api) |
# | Checkpointers / persistence | [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) |
# | Interrupts and resuming | [Human-in-the-loop](https://docs.langchain.com/oss/python/langgraph/use-graph-api#human-in-the-loop) |
# | Studio: run + inspect state | [Studio quick start](https://docs.langchain.com/langsmith/quick-start-studio) |
# | A full worked example of this exact use case | [`langsmith-agent-lifecycle-workshop`](https://github.com/langchain-ai/langsmith-agent-lifecycle-workshop) — see `agents/`, `tools/database.py` |
# | Fundamentals, end to end | [`langgraph-101`](https://github.com/langchain-ai/langgraph-101) |
#
# > **Closest analogue to today's build:** `agents/supervisor_hitl_sql_agent_graph.py` in the
# > lifecycle workshop — supervisor + SQL + HITL, the same three pieces you are assembling.

# %% [markdown]
# ## 0. Setup

# %%
import json
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# No `override=True`: real environment variables must win over `.env`, otherwise the
# `export LANGSMITH_GATEWAY=true / export LANGSMITH_GATEWAY_API_KEY=...` gateway setup documented in
# the README (Model access section) and .env.example is silently clobbered by whatever `.env` happens to contain.
# `.env` fills the gaps; your shell wins.
load_dotenv(find_dotenv())

# No gateway key juggling here: `day1/src/models.py` routes both the chat model
# (`get_model`) and the embeddings client (`get_embeddings`) by inspecting the gateway env vars, and
# passes the gateway credential explicitly as `api_key=`. See the README (Model access section).

# Put the workshop root on sys.path so `day1.src.*`, `day2.src.*` and `utils.*` import cleanly.
# Works as a script (`__file__` is defined) and in a Jupyter kernel (it is not — fall back to the
# notebook's cwd), whether the workshop is checked out standalone or nested inside a parent repo.
def _find_workshop_root(start: Path) -> Path:
    for cand in (start.resolve(), *start.resolve().parents):
        if (cand / "day1").is_dir() and (cand / "utils").is_dir():
            return cand
        # Nested checkout: look for ANY child directory that holds day1/ and utils/, rather than
        # hardcoding the folder name — the repo can be cloned under any name.
        for child in sorted(p for p in cand.iterdir() if p.is_dir()) if cand.is_dir() else []:
            if (child / "day1").is_dir() and (child / "utils").is_dir():
                return child
    raise FileNotFoundError("Could not locate the workshop root (the directory holding day1/ and utils/).")


WORKSHOP = _find_workshop_root(Path(globals()["__file__"]).parent if "__file__" in globals() else Path.cwd())
sys.path.insert(0, str(WORKSHOP))
DATA = WORKSHOP / "day1" / "data"

print("workspace:", os.getenv("LANGSMITH_WORKSPACE_ID"), "| tracing:", os.getenv("LANGSMITH_TRACING"))

# %% [markdown]
# ## 1. Tool — search the knowledge base
#
# 📖 [Retrieval](https://docs.langchain.com/oss/python/langchain/retrieval)
#
# Chunk the synthetic KB articles, embed them, keep them in an in-memory store. The tool returns
# top matches **with their source filename** so the agent can cite.

# %%
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from day1.src.models import get_embeddings  # routes to gateway or direct provider — see the README (Model access section)

docs = [Document(page_content=p.read_text(), metadata={"source": p.name}) for p in sorted((DATA / "kb_tickets").glob("*.md"))]
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
kb_index = InMemoryVectorStore.from_documents(chunks, get_embeddings())
print(f"indexed {len(chunks)} chunks from {len(docs)} KB articles")

# %%
from langchain_core.tools import tool


@tool
def search_kb(query: str) -> str:
    """Search IT support knowledge base articles by topic, keyword, or symptom.
    Returns cited snippets from the KB ticket articles."""
    hits = kb_index.similarity_search(query, k=4)
    if not hits:
        return "No relevant KB articles found."
    return "\n\n---\n\n".join(f"[source: {h.metadata['source']}]\n{h.page_content}" for h in hits)


# %%
print(search_kb.invoke({"query": "MFA reset phone replacement"})[:320])

# %% [markdown]
# ## 2. Tools — the ticket **database** (SQL)
#
# 📖 [Tools](https://docs.langchain.com/oss/python/langchain/tools) · [reference](https://github.com/langchain-ai/langsmith-agent-lifecycle-workshop/blob/main/tools/database.py)
#
# A real ticketing system is a database, so this one is too. `day1/src/ticket_db.py` builds
# `day1/data/tickets.db` (SQLite) from the committed JSON fixtures — normalised into `tickets`,
# `users`, `user_permissions` and `ticket_events`, with indexes and foreign keys.
#
# This mirrors the workshop this one is based on
# ([langsmith-agent-lifecycle-workshop](https://github.com/langchain-ai/langsmith-agent-lifecycle-workshop)),
# which ships a real SQLite database and a dedicated SQL agent over it.
#
# **Why it matters here, and not just for realism:** structured questions and semantic questions
# need different machinery.
#
# | Question | Right tool | Why |
# |---|---|---|
# | *"Has anyone seen this VPN drop before?"* | `search_kb` (RAG) | Semantic match over prose |
# | *"Which P1 tickets are still open?"* | SQL | Exact filter, needs completeness |
# | *"Mean time to resolution by category?"* | SQL | Aggregate — no top-k answers this |
# | *"What can USR-004 actually see?"* | SQL | Row-level access control in the WHERE clause |
#
# Three tools, escalating in power:
#
# | Tool | Does | Guardrail |
# |---|---|---|
# | `search_ticket_history` | Filtered search, parameterised | Values bound with `?` — never string-formatted |
# | `ticket_stats` | `GROUP BY` aggregates | `group_by` validated against an allow-list |
# | `query_ticket_db` | **The model writes the SQL** | Read-only connection at the SQLite level |
#
# > **Name every column for exactly what it counts.** An earlier version of `ticket_statistics`
# > returned a column called `tickets` (meaning *total*) beside `resolved`. Asked "how many P1
# > tickets are still open?", the agent read `5` off the `tickets` column and answered "5 are open"
# > — when 1 was open and 2 unresolved. The SQL was right; the *column name* was ambiguous, and the
# > model resolved the ambiguity wrongly. It now returns `total`, `open`, `in_progress`,
# > `escalated`, `resolved` and `unresolved` as separate named columns, and answers correctly.
# > **A tool whose output needs arithmetic to interpret will eventually be misinterpreted** — that
# > is a tool-design bug, not a model failure, and no prompt change fixes it.
#
# > **The guardrail that counts.** `query_ticket_db` checks for mutating keywords *and* opens the
# > connection with `file:...?mode=ro`. The keyword check gives a good error message; the read-only
# > connection is the actual control, because it holds even against a query that talks its way past
# > a string check. Never rely on prompt instructions to keep an agent read-only.
#
# ### Two tiers, and why this lab only uses the first
#
# The reference workshop ships **both** patterns as separate agents, and so do we:
#
# | Tier | Pattern | Reference | Here |
# |---|---|---|---|
# | **Beginner** | Fixed, pre-written query tools. The model picks a tool and fills parameters. | [`agents/db_agent.py`](https://github.com/langchain-ai/langsmith-agent-lifecycle-workshop/blob/main/agents/db_agent.py) | **This lab** |
# | **Advanced** | The model authors SQL, with the schema in its system prompt. | [`agents/sql_agent.py`](https://github.com/langchain-ai/langsmith-agent-lifecycle-workshop/blob/main/agents/sql_agent.py) | **Lab 03 §3** |
#
# Start with fixed tools: they are predictable, cheap to trace, and impossible to misuse. Reach for
# model-authored SQL only when you have **evidence** the fixed tools are the bottleneck.
#
# That evidence is the point. In the reference workshop the SQL agent exists because *baseline
# evaluation showed the rigid tools could not handle complex queries* — the limitation was measured
# first, then fixed. You will reproduce that exact loop: build with fixed tools here, watch one fail
# below, and see it evaluated in Day 2 Lab 04.
#
# We still *define* `query_ticket_db` above so you can call it directly and see the guardrail work,
# but it is deliberately **not** in this lab's agent tool list.

# %%
from day1.src.ticket_db import execute_sql, search_tickets, ticket_stats
from day1.src.ticket_db import DB_PATH, SCHEMA_DOC

# `tickets.db` ships in the repo — there is nothing to build. The labs only ever read it, which is
# how the reference workshop does it too: data generation lives in a script the labs never import
# (`day1/data/build_ticket_db.py` here), so opening a database can never quietly regenerate it.
print(f"ticket database: {DB_PATH.name} — ships with the repo, opened read-only\n")
print(SCHEMA_DOC)


@tool
def search_ticket_history(
    keyword: str | None = None,
    category: str | None = None,
    status: str | None = None,
    priority: str | None = None,
) -> str:
    """Search the ticket database for past and live tickets.

    Filters are combined with AND. Resolved tickets give a proven fix; open or escalated ones flag
    that the same issue is already live, which is often the more useful answer.
    Categories: access, account, hardware, knowledge, network, security, software.
    Statuses: open, in_progress, escalated, resolved. Priorities: P1-P4.
    """
    return search_tickets(query=keyword or "", category=category, status=status, priority=priority)


@tool
def ticket_statistics(group_by: str = "category") -> str:
    """Aggregate ticket statistics: counts, resolved counts, and mean hours to resolution.

    Group by one of: category, priority, status, queue. Use this for "how many", "what share",
    or "on average" questions — a keyword search cannot answer them correctly.
    """
    return ticket_stats(group_by)


@tool
def query_ticket_db(query: str) -> str:
    """Run a read-only SQL SELECT against the ticket database for questions the other tools cannot express.

    Use for joins and custom aggregates, e.g.
      SELECT u.department, COUNT(*) n FROM tickets t
      JOIN users u ON u.user_id = t.user_id GROUP BY u.department ORDER BY n DESC

    Schema:
    TABLE users(user_id, name, email, department, role)
    TABLE user_permissions(user_id, category)
    TABLE tickets(ticket_id, user_id, subject, description, category, priority, status,
                  resolution, created_at, resolved_at, queue)
    TABLE ticket_events(event_id, ticket_id, ts, actor, from_status, to_status, note)

    Only SELECT/WITH are permitted; the connection is read-only.
    """
    return execute_sql(query)


# %%
print(search_ticket_history.invoke({"category": "access"}))  # includes the open TKT-023

# %%
print(search_ticket_history.invoke({"keyword": "VPN"}))

# %%
# The aggregate question — exact, complete, and impossible for similarity search to answer.
print(ticket_statistics.invoke({"group_by": "priority"}))

# %%
# And the model-authored query, with the read-only guard proving itself.
print(query_ticket_db.invoke({"query": (
    "SELECT u.department, COUNT(*) AS n FROM tickets t "
    "JOIN users u ON u.user_id = t.user_id GROUP BY u.department ORDER BY n DESC")}))
print()
print(query_ticket_db.invoke({"query": "DELETE FROM tickets"}))

# %% [markdown]
# ### Where the fixed tools run out
#
# `search_ticket_history` and `ticket_statistics` cover the common questions. Now ask one that
# needs a **join plus a filtered aggregate**: *"which department raises the most security or access
# tickets, and how many are still unresolved?"*
#
# There is no combination of the fixed tools' parameters that answers it — `ticket_statistics`
# groups by exactly one column and knows nothing about `users.department`. This is the measured
# limitation that justifies the advanced tier, and you will see the agent hit it in Lab 03.

# %%
print("fixed tools — closest possible attempt:")
print(ticket_statistics.invoke({"group_by": "category"})[:240])
print("\n  ↑ no department breakdown, no unresolved filter. The question is not expressible.\n")
print("model-authored SQL — the same question, answered:")
print(query_ticket_db.invoke({"query": (
    "SELECT u.department, COUNT(*) AS total, "
    "SUM(t.status != 'resolved') AS unresolved "
    "FROM tickets t JOIN users u ON u.user_id = t.user_id "
    "WHERE t.category IN ('security','access') "
    "GROUP BY u.department ORDER BY unresolved DESC, total DESC")}))

# %% [markdown]
# ## 3. Tool — mock API action (sensitive operations)
#
# 📖 [Tools](https://docs.langchain.com/oss/python/langchain/tools)
#
# A mock external API for sensitive actions like MFA reset and account unlock. Returns a pending
# action that requires human-in-the-loop (HITL) approval before execution. This demonstrates how an
# agent can flag sensitive operations for human review.

# %%
@tool
def mock_api_action(action: str, user_id: str, reason: str) -> str:
    """Perform a sensitive action via the external API. Supported actions: mfa_reset, account_unlock.
    Returns a pending action that requires human approval before execution."""
    supported = {"mfa_reset", "account_unlock"}
    if action not in supported:
        return f"Unsupported action {action!r}. Supported: {', '.join(sorted(supported))}"
    return (
        f"⏳ PENDING ACTION — {action}\n"
        f"  user_id: {user_id}\n"
        f"  reason: {reason}\n"
        f"  status: awaiting human approval\n"
        f"  [This action requires HITL review before it can be executed.]"
    )


# %%
print(mock_api_action.invoke({"action": "mfa_reset", "user_id": "USR-003", "reason": "Phone replaced, cannot receive MFA push"}))

# %% [markdown]
# ## 4. Tool — get user context
#
# 📖 [Tools](https://docs.langchain.com/oss/python/langchain/tools)
#
# A lookup from `users.json` that returns the user's role and permissions. The agent uses this to
# determine what actions the user is authorized to perform.

# %%
USERS = json.loads((DATA / "users.json").read_text())


@tool
def get_user_context(user_id: str) -> str:
    """Get a user's role and permissions by user ID (e.g. USR-001) or full name. Returns the user's
    name, department, role, and list of authorized permission categories."""
    # Exact match only. A substring match would let "a" resolve to "Ana Costa" — the admin — which
    # is precisely the wrong failure mode for the tool the agent uses to check authorization.
    q = user_id.strip().lower()
    for user in USERS:
        if q and (q == user["user_id"].lower() or q == user["name"].lower()):
            perms = ", ".join(user.get("permissions", []))
            return (
                f"{user['name']} ({user['user_id']}) — {user['department']}\n"
                f"  role: {user['role']}\n"
                f"  permissions: {perms}\n"
                f"  email: {user['email']}\n"
                f"  [source: users.json]"
            )
    return f"No user found for {user_id!r}. Ask for their user ID (e.g. USR-001)."


# %%
print(get_user_context.invoke({"user_id": "USR-002"}))

# %% [markdown]
# ## 5. The system prompt
#
# 📖 [Prompt engineering](https://docs.langchain.com/langsmith/prompt-engineering-quickstart)
#
# Managed in the **LangSmith Prompt Hub** (`ticket-resolution`), pulled at runtime with a
# local fallback.

# %%
from utils.prompts import get_prompt, prompt_url

PROMPT_NAME = "ticket-resolution"
system_prompt = get_prompt(PROMPT_NAME, fallback=(DATA / "prompt_ticket.md").read_text().strip())
print(system_prompt)

# %% [markdown]
# ## 6. Structured output schema
#
# 📖 [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
#
# The agent returns a `TicketResolution` — a structured resolution plan with citations, confidence,
# and HITL flags. Passed as `response_format=` so LangChain enforces it and returns a parsed object.

# %%
from typing import Literal

from pydantic import BaseModel, Field


class TicketResolution(BaseModel):
    """Structured output for the ticket resolution agent. Passed as `response_format=` so
    LangChain enforces it and returns a parsed object at `result["structured_response"]`."""

    issue_summary: str = Field(
        description="Brief summary of the user's issue"
    )
    category: Literal["access", "account", "hardware", "knowledge", "network", "software", "security"] = Field(
        description="Ticket category"
    )
    kb_references: list[str] = Field(
        description="KB article filenames that were cited as evidence"
    )
    similar_tickets: list[str] = Field(
        default_factory=list,
        description="Ticket IDs of similar resolved tickets"
    )
    recommended_action: str = Field(
        description="Recommended next step to resolve the issue"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the recommendation"
    )
    requires_hitl: bool = Field(
        description="True when the recommended action is sensitive (MFA reset, account unlock) and requires human approval"
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description="Information that would improve confidence if provided"
    )
    user_id: str = Field(
        description="The user ID associated with the ticket"
    )


# %% [markdown]
# ## 7. Build the agent
#
# 📖 [Agents (create_agent)](https://docs.langchain.com/oss/python/langchain/agents) · [reference](https://reference.langchain.com/python/langchain/agents/factory/create_agent)
#
# One clean keyword-arg call: model + tools + prompt in, a compiled ReAct agent out.

# %%
from langchain.agents import create_agent

from day1.src.models import get_model

model = get_model()

# Fixed, pre-written query tools only — see the note above on why the dynamic-SQL tool waits
# for Lab 03.
tools = [search_kb, search_ticket_history, ticket_statistics, mock_api_action, get_user_context]
if os.getenv("TAVILY_API_KEY"):
    from langchain_tavily import TavilySearch
    tools.append(TavilySearch(max_results=3))
print("tools:", [t.name for t in tools])

from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    response_format=ToolStrategy(TicketResolution, handle_errors=True),
)

# %% [markdown]
# **The agent's shape** — `create_agent` returns a compiled LangGraph graph; render it.

# %%
from IPython.display import Image, display

display(Image(agent.get_graph().draw_mermaid_png()))

# %% [markdown]
# ## 8. Run it
#
# 📖 [Agents (create_agent)](https://docs.langchain.com/oss/python/langchain/agents) · [reference](https://reference.langchain.com/python/langchain/agents/factory/create_agent)
#
# `ask()` invokes the agent and prints a clickable **LangSmith trace** link.

# %%
from utils.trace import ask

print(ask(agent, "My VPN keeps disconnecting every few minutes when I work from home. "
    "I'm USR-002. Can you help?"))

# %%
print(ask(agent, "I just got a new phone and my MFA isn't working. I'm locked out of everything. "
    "My user ID is USR-003."))

# %%
# Underspecified issue — the agent should generate follow-up questions
print(ask(agent, "Something is wrong with my computer."))

# %% [markdown]
# ## 9. Optional: Live API calls
#
# The agent has one optional tool that calls an external API. This is **not required** — the agent
# works without it — but it demonstrates how an agent can call live external services.
#
# | Tool | API | What it does |
# |---|---|---|
# | `tavily_search` | Tavily Search API | Live web search for solutions not in the bundled data |
#
# **To enable:**
# - Set `TAVILY_API_KEY` in `.env` — the agent adds `tavily_search` to its tool list
#
# > **Note:** Optional. The lab works without it — the synthetic data is comprehensive.
# > This is for participants who want to see how external API calls work in an agent.

# %%
# Check if the optional API is configured
tavily_enabled = bool(os.getenv("TAVILY_API_KEY"))

print(f"Tavily API key: {'✅ set' if tavily_enabled else '❌ not set (optional — agent works without it)'}")

# %%
# Demo: Tavily search (only if key is set)
if tavily_enabled:
    from langchain_tavily import TavilySearch
    tavily = TavilySearch(max_results=3)
    result = tavily.invoke({"query": "VPN troubleshooting enterprise"})
    print("Tavily search result:")
    # Tavily returns a list of results — print the first one
    if isinstance(result, list) and result:
        print(result[0].get("content", str(result[0]))[:500])
    else:
        print(str(result)[:500])
else:
    print("Tavily not enabled — set TAVILY_API_KEY in .env to enable live web search")

# %% [markdown]
# ## 10. LangChain middleware — PII redaction
#
# 📖 [Middleware](https://docs.langchain.com/oss/python/langchain/middleware)
#
# LangChain provides built-in middleware that hooks into the agent's execution lifecycle. The
# `PIIMiddleware` pattern intercepts messages before they reach the model and redacts sensitive
# patterns. This is the **correct** way to add PII protection — not a wrapper function.
#
# | Middleware | Purpose |
# |---|---|
# | `PIIMiddleware` | Redact PII (email, credit card, IP, MAC, URL) before the model sees it |
# | `HumanInTheLoopMiddleware` | Pause for human approval before tool calls (Lab 03) |
# | `ToolCallLimitMiddleware` | Cap runaway tool loops (Lab 05) |
#
# > `PIIMiddleware(pii_type, strategy=...)` ships with detectors for `email`, `credit_card`, `ip`,
# > `mac_address` and `url`, and four strategies: `redact`, `mask`, `hash`, `block`. By default it
# > rewrites the **input** to the model; `apply_to_output` / `apply_to_tool_results` extend it to
# > what comes back. See the [built-in middleware
# > docs](https://docs.langchain.com/oss/python/langchain/middleware/built-in).

# %%
from langchain.agents.middleware import PIIMiddleware
from langchain_core.messages import HumanMessage

pii_middleware = PIIMiddleware("email", strategy="redact")

# Run the middleware's own `before_model` hook to see exactly what the model would receive —
# no API call needed, and no guessing about what redaction does.
_before = HumanMessage(content="Contact support@acme.example for details.")
_after = pii_middleware.before_model({"messages": [_before]}, None)["messages"][0]
print("participant wrote :", _before.content)
print("model receives    :", _after.content)

# %% [markdown]
# ### Build the agent with PII middleware
#
# The middleware is passed to `create_agent` as a `middleware` list. Each middleware wraps the
# agent's execution — PII redaction runs before every model call.

# %%
agent_with_pii = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    response_format=ToolStrategy(TicketResolution, handle_errors=True),
    middleware=[pii_middleware],
)

# %%
# The middleware redacts PII before the model sees it
result = ask(agent_with_pii,
    "Look up the user context for Marco Rossi. My email is engineer@acme.example.")
print(result)

# %% [markdown]
# ## 11. LangSmith links
#
# Every run above printed its **trace** link. The **prompt** link opens the Hub entry.

# %%
_url = prompt_url(PROMPT_NAME)
if _url:
    print("📝 prompt:", _url)
else:
    print("📝 prompt: (local fallback in use — not synced to Prompt Hub)")

# %% [markdown]
# ## 12. Recap & next
#
# | Topic | API |
# |---|---|
# | Build a ReAct agent | `create_agent(model=, tools=, system_prompt=, response_format=)` |
# | RAG + lookup + action tools | `search_kb`, `search_ticket_history`, `mock_api_action`, `get_user_context` |
# | Render the graph | `agent.get_graph().draw_mermaid_png()` |
# | Traced runs + prompt link | `ask()` · `prompt_url()` |
# | Structured output | `response_format=TicketResolution` on `create_agent` |
# | PII redaction | `PIIMiddleware("email", strategy="redact")` — built-in LangChain middleware |
#
# **Next:** `03_langgraph_hitl.ipynb` — convert this into a controllable LangGraph workflow with
# memory and human-in-the-loop checkpoints.
