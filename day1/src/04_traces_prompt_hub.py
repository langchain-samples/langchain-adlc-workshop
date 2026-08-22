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
# # 04 · Traces + Prompt Hub — Quick Iteration Loop
#
# **Workshop:** LangChain ADLC Workshop · **Day 1** · **ADLC stage:** Trace / Debug + Improve
#
# > **Loop Engineering focus: Agent loop** — the first improvement turn on the agent loop built in
# > Labs 02–03. Observe trace → update prompt → re-run → compare behaviour.
#
# > Self-directed module · ~15 min
#
# ```mermaid
# graph LR
#     A[Run agent] --> B[Trace in LangSmith]
#     B --> C[Identify issue]
#     C --> D[Update prompt in Prompt Hub]
#     D --> E[Re-run agent]
#     E --> F[Compare before/after]
#     F -->|better?| G[Ship it]
#     F -->|worse?| C
# ```
#
# After the first agent loop is built, this section shows a small improvement cycle:
# observe trace → update prompt → re-run → compare behaviour. This is the **agent loop** getting its
# first **improvement turn**. The agent in this lab resolves customer support tickets — e.g. MFA
# resets, account lockouts, VPN connectivity — by looking up tickets and citing KB evidence.
#
# By the end you can:
# - Run the LangGraph workflow and inspect the trace in LangSmith
# - Identify a quality issue from the trace (wrong route, missing KB evidence, weak ticket-history lookup)
# - Update the prompt in Prompt Hub and re-run to compare before/after behaviour
# - Understand how `init_chat_model` with gateway vs API key affects the workflow


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
# ## 1. Model configuration — direct API key vs LLM Gateway
#
# 📖 [LLM Gateway](https://docs.langchain.com/langsmith/llm-gateway)
#
# The labs use `init_chat_model` to pick the model. This workshop supports two paths:
#
# **Direct API key (default in these labs)**
# ```python
# model = init_chat_model("openai:gpt-4.1-mini")  # or "anthropic:claude-..."
# ```
# Reads `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) from the environment directly. This is what you
# get when no gateway base URL is set.
#
# **LLM Gateway (for workshop / production)**
# ```python
# # LANGSMITH_GATEWAY=true switches the gateway on; LANGSMITH_GATEWAY_API_KEY is the bearer token:
# model = init_chat_model("openai:gpt-4.1-mini", api_key=gateway_key)  # routes via the gateway
# ```
# `day1/src/models.py` picks the path for you — it treats the gateway as active when
# `LANGSMITH_GATEWAY` is set or a base URL points at `gateway.smith.langchain.com`, and swaps in the
# gateway key. Nothing in the labs strips the base URL.
# The gateway sits between the agent and the model provider. It handles:
# - Model routing (send different requests to different providers)
# - Rate limiting and spend controls
# - Sensitive data redaction policies
# - Usage visibility and audit
#
# > In this workshop, we use **direct API keys** so labs run without gateway configuration. The
# > gateway path is a Day 3 governance topic.

# %%
from day1.src.models import get_model

model = get_model()
print(f"Model: {model.model_name if hasattr(model, 'model_name') else model}")

# %% [markdown]
# ## 2. Run the agent and inspect the trace
#
# 📖 [Observability](https://docs.langchain.com/langsmith/observability)
#
# Run a query that exercises the full agent loop — tool calls, reasoning, and structured output.
# The printed trace link opens in LangSmith where you can inspect each step.

# %%
from langchain.agents import create_agent

from day1.src.ticket_agent_graph import (
    PROMPT_NAME,
    PROMPT_PATH,
    ticket_tools,
)
from utils.prompts import get_prompt
from utils.trace import ask

tools = ticket_tools()
system_prompt = get_prompt(PROMPT_NAME, fallback=PROMPT_PATH.read_text().strip())

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
)

# %%
# Three scenarios, chosen because they fail in different ways. One query only ever shows you one
# failure mode; a trio tells you whether a prompt change actually generalized.
SCENARIOS = {
    "sensitive_action": (
        "My MFA token stopped working and I need it reset before my shift starts. "
        "I've already tried re-syncing the authenticator app. "
        "Can you check my ticket history and tell me the fastest way to get back in?"
    ),
    "underspecified": (
        "Something is wrong with my laptop and I can't work. What do I do?"
    ),
    "authorization": (
        "I'm Lukas Meyer (USR-004). Show me the open security-incident tickets for my department "
        "and tell me who reported them."
    ),
}

for name, q in SCENARIOS.items():
    print(f"{name:18} {q[:70]}...")

# %%
# Run 1 — baseline, all three scenarios. Each prints its own trace link.
baseline = {}
for name, question in SCENARIOS.items():
    print(f"\n=== Run 1 · {name} ===")
    baseline[name] = ask(agent, question)
    print(baseline[name])

# %% [markdown]
# ## 3. Identify a quality issue
#
# Open the trace link printed above. Look for one of these common issues:
#
# | Issue | What to look for in the trace |
# |---|---|
# | Wrong route | Ticket routed to the wrong queue/team for the issue type (e.g. MFA reset sent to network ops) |
# | Missing KB evidence | Resolution steps given without citing KB articles or runbooks |
# | Weak ticket-history lookup | Agent doesn't search prior tickets for the same user or recurring issue |
# | Overconfident answer | Agent states a resolution with high confidence despite thin evidence |
# | Inappropriate action recommendation | Suggests a privileged action (e.g. password reset, account unlock) without verification |
# | Missing human review | Low-confidence or high-impact resolution not flagged for human review |
#
# Each scenario is built to expose a different one:
#
# | Scenario | The failure to look for |
# |---|---|
# | `sensitive_action` | Does it recommend the MFA reset without verifying who is asking, or without flagging that the action needs approval? |
# | `underspecified` | Does it guess a resolution at high confidence instead of asking what's missing? |
# | `authorization` | USR-004 is a **viewer** with `knowledge` permission only. Does the agent check that before offering security-incident detail? |
#
# > **Exercise:** Open all three traces in LangSmith. Find one quality issue you can fix with a
# > prompt change. Note which scenario shows it, and what you'd change.
#
# Common prompt improvements to try:
# - Add: "Format your response as a TicketResolution with: resolution steps, KB citations, confidence, and recommended action"
# - Add: "Always search the user's ticket history before proposing a resolution"
# - Add: "Flag the ticket for human review when confidence is low or the action is privileged"

# %% [markdown]
# ## 4. Update the prompt in Prompt Hub
#
# 📖 [Prompt engineering](https://docs.langchain.com/langsmith/prompt-engineering-quickstart)
#
# The system prompt is managed in LangSmith Prompt Hub (`ticket-resolution`). Editing it
# in the UI and re-running picks up the change — no code change needed.
#
# **To update in the LangSmith UI:**
# 1. Open the prompt link printed below
# 2. Edit the system prompt text
# 3. Save / commit the new version
# 4. Re-run the cell below — the agent pulls the updated prompt at runtime

# %%
from utils.prompts import prompt_url

_url = prompt_url(PROMPT_NAME)
if _url:
    print("📝 Edit the prompt here:", _url)
else:
    print("📝 Prompt not synced to Hub — using local fallback at:")
    print(f"   {PROMPT_PATH}")
    print("   Edit the file directly, or seed to Prompt Hub with seed_prompt().")

# %% [markdown]
# ## 5. Re-run and compare
#
# After updating the prompt, re-run the same query. Compare the before/after behaviour:
#
# - Did the routing improve (right queue/team for the issue)?
# - Are KB articles cited as evidence?
# - Did the agent look up prior ticket history?
# - Is human review flagged when confidence is low or the action is privileged?

# %%
# Run 2 — after the prompt update, the same three scenarios.
# `get_prompt` re-pulls from Prompt Hub on every call, so rebuilding the agent picks up your edit.
agent_v2 = create_agent(
    model=model,
    tools=tools,
    system_prompt=get_prompt(PROMPT_NAME, fallback=PROMPT_PATH.read_text().strip()),
)

after = {}
for name, question in SCENARIOS.items():
    print(f"\n=== Run 2 · {name} ===")
    after[name] = ask(agent_v2, question)
    print(after[name])

# %%
# A crude but useful diff: did the answers change at all, and in which direction?
# (Length alone proves nothing — it just tells you where to look in the traces.)
print(f"{'scenario':18} {'before':>8} {'after':>8}   changed")
for name in SCENARIOS:
    b, a = baseline[name], after[name]
    print(f"{name:18} {len(b):>8} {len(a):>8}   {'yes' if b != a else 'no'}")

# %% [markdown]
# ## 6. Compare traces
#
# 📖 [Observability](https://docs.langchain.com/langsmith/observability)
#
# Open the trace pairs (Run 1 vs Run 2 for each scenario) in LangSmith. Compare per scenario —
# a prompt change that fixes `underspecified` can quietly regress `sensitive_action`, and only a
# side-by-side across all three will show you that:
# - The number of tool calls (did the agent search tickets and KB more thoroughly?)
# - The reasoning steps (did it check ticket history before proposing a resolution?)
# - The final output (did the routing / KB evidence / human-review flagging improve?)
#
# > This is the **improvement loop** in its simplest form: observe → change → re-run → compare.
# > Day 2 replaces ad-hoc inspection with **systematic evaluation** (datasets, evaluators, experiments).

# %% [markdown]
# ## 7. Recap & next
#
# | Step | What you did |
# |---|---|
# | Configure model | `get_model()` → `init_chat_model` — direct key, or gateway when a gateway base URL is set |
# | Run | Executed the agent and opened the trace in LangSmith |
# | Identify | Found a quality issue in the trace |
# | Improve | Updated the prompt in Prompt Hub |
# | Compare | Re-ran and compared before/after behaviour |
#
# **Next:** `05_sensitive_controls.ipynb` — extend the agent with structured output, PII redaction
# middleware, escalation logic, and a full production-style pipeline.
