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
# # 01 · Setup — Acme Ticket Resolution Agent
#
# **Workshop:** LangChain ADLC Workshop · **Day 1** · **ADLC stage:** Setup
#
# > Self-directed module · ~15 min
#
# By the end you can:
# - Confirm your `.env` is loaded and pointing at the right LangSmith workspace
# - Verify the LLM and LangSmith are reachable before you start the track
# - Know where to look when a probe fails (Troubleshooting below)
#
# > 🧭 **First lab in the workshop — runs standalone.** All ticket data is **synthetic/fictional** —
# > created for this workshop, no real Acme data.


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
# ## 0. Environment setup
#
# > 💡 `override=True` lets the workshop `.env` win over globally-set keys. The printed **workspace**
# > is where every trace and prompt routes.

# %%
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# No `override=True`: real environment variables must win over `.env`, otherwise the
# `export LANGSMITH_GATEWAY=true / export LANGSMITH_GATEWAY_API_KEY=...` gateway setup documented in
# the README (Model access section) and .env.example is silently clobbered by whatever `.env` happens to contain.
# `.env` fills the gaps; your shell wins.
load_dotenv(find_dotenv())  # FIRST — before any model/embeddings client
# The model layer (day1/src/models.py) handles gateway vs direct API key routing.
# See the README (Model access section) for gateway setup instructions.

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

print("workspace:", os.getenv("LANGSMITH_WORKSPACE_ID"), "| tracing:", os.getenv("LANGSMITH_TRACING"))

# %% [markdown]
# ## 1. Status helpers
#
# Each probe prints ✅ / ⛔ so you can scan the whole cell at a glance. A down service never crashes
# the notebook — it prints a diagnostic line instead.

# %%
def _ok(label: str, detail: str = "") -> None:
    print(f"✅ {label}" + (f" — {detail}" if detail else ""))

def _fail(label: str, detail: str = "") -> None:
    print(f"⛔ {label}" + (f" — {detail}" if detail else ""))

def _probe(label: str):
    """Decorator: catch exceptions so a probe failure prints a ⛔ line, never a traceback."""
    def wrap(fn):
        def inner(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as e:
                _fail(label, str(e)[:120])
        return inner
    return wrap

# %% [markdown]
# ## 2. Probe — LLM reachability
#
# A one-token completion confirms the model endpoint + key work. The model is configured via
# `init_chat_model` — default is OpenAI, swappable via `LABS_MODEL` env var (e.g. `anthropic:claude-...`).

# %%
from day1.src.models import get_model

@_probe("LLM reachability")
def probe_llm() -> None:
    model = get_model()
    resp = model.invoke("Reply with the single word: ready")
    text = resp.content if hasattr(resp, "content") else str(resp)
    _ok("LLM reachability", text.strip()[:40])

probe_llm()

# %% [markdown]
# ## 3. Probe — LangSmith reachability
#
# Lists one workspace — confirms the API key and endpoint are correct. Traces from every lab in this
# workshop will appear under the printed project name.

# %%
@_probe("LangSmith reachability")
def probe_langsmith() -> None:
    if not os.getenv("LANGSMITH_API_KEY"):
        _fail("LangSmith reachability", "LANGSMITH_API_KEY not set")
        return
    from langsmith import Client
    client = Client()
    # Use list_projects instead of list_workspaces (which doesn't exist on the Client class)
    projects = list(client.list_projects(limit=1))
    if projects:
        _ok("LangSmith reachability", f"project: {projects[0].name}")
    else:
        _ok("LangSmith reachability", "connected (no projects listed)")

probe_langsmith()

# %% [markdown]
# ## 4. Probe — ticket data loaded
#
# Confirm the synthetic ticket dataset and KB articles are accessible.

# %%
DATA = WORKSHOP / "day1" / "data"
import json

@_probe("Ticket data")
def probe_data() -> None:
    tickets = json.loads((DATA / "tickets.json").read_text())
    ticket_count = len(tickets)
    kb_count = len(list((DATA / "kb_tickets").glob("*.md")))
    users = json.loads((DATA / "users.json").read_text())
    _ok("Ticket data", f"{ticket_count} tickets, {kb_count} KB articles, {len(users)} users")

probe_data()

# %% [markdown]
# ## 5. Troubleshooting
#
# | Symptom | Fix |
# |---|---|
# | ⛔ LLM reachability — 403 / auth error | Check `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) in `.env` |
# | ⛔ LLM — 403 gateway.smith.langchain.com | The gateway guard should drop it; verify `.env` loads first |
# | ⛔ LangSmith — API key not set | Set `LANGSMITH_API_KEY` in `.env` |
# | ⛔ LangSmith — wrong workspace | Set `LANGSMITH_WORKSPACE_ID` to your workshop workspace UUID |
# | ⛔ Ticket data — file not found | Run from the repo root; the path walker finds `langchain_adlc_workshop/day1/data/` |
#
# > If all probes are ✅, jump to `02_langchain_foundations.ipynb`.

# %% [markdown]
# ## 6. Recap & next
#
# | Check | What it proved |
# |---|---|
# | LLM probe | Model endpoint + key work |
# | LangSmith probe | Tracing + datasets will route correctly |
# | Ticket data probe | Synthetic data is loaded for the workshop |
#
# **Next:** `02_langchain_foundations.ipynb` — build the ticket resolution agent with LangChain.
