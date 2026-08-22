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
# # 01 · Deployments — Ticket Resolution Agent
#
# **Workshop:** LangChain ADLC Workshop · **Day 3** · **ADLC stage:** Deploy
#
# > **Loop Engineering focus: Event-driven loop** — the ticket agent stops being a notebook cell
# > and becomes a **service**: a deployed LangGraph graph with a REST API, platform-managed
# > persistence, and production traces that feed the improvement loops built on Day 2.
#
# > Hands-on module · ~50 min
#
# ```mermaid
# graph LR
#     A[day1/langgraph.json<br/>4 ticket_agent variants] --> B{Deploy}
#     B -->|langgraph dev| C[Local server :2024<br/>in-memory persistence]
#     B -->|langgraph deploy / LangSmith UI| D[Cloud deployment<br/>managed Agent Server]
#     D --> E[LangGraph SDK<br/>threads + runs]
#     C --> E
#     E --> F[Production traces<br/>acme-ticket-deployment]
#     F -->|Lab 03| G[Online evals +<br/>review routing]
#     F -->|Day 2 evals| H[Offline regression<br/>before next revision]
# ```
#
# **What gets deployed:** the Day 1 ticket agent, unchanged. `day1/src/ticket_agent_graph.py` is the
# single source of truth and `day1/langgraph.json` registers four variants:
#
# | Graph ID | Module attribute | Variant |
# |---|---|---|
# | `ticket_agent` | `graph` | plain ReAct agent |
# | `ticket_agent_mem` | `graph_with_memory` | memory-ready (the platform supplies persistence) |
# | `ticket_agent_hitl` | `graph_hitl` | pauses for human approval on `mock_api_action` |
# | `ticket_agent_structured` | `graph_structured` | typed `TicketResolution` output |
#
# By the end you can:
# - pick a **graph variant** for deployment and explain the tradeoffs (default: `ticket_agent_hitl`)
# - run the deployable graph locally with `langgraph dev` and deploy it with `langgraph deploy`
#   (or the equivalent LangSmith UI flow)
# - configure **environment variables and the model endpoint** for a deployment (direct provider
#   key vs LLM Gateway)
# - invoke the deployed agent with the **LangGraph SDK** — threads, `runs.wait`, streaming, and
#   **resuming a HITL interrupt** with `Command(resume=...)`
# - inspect **deployment traces** in LangSmith
# - map this cloud deployment to **self-hosted / hybrid** paths (the Day 3 afternoon discussion)
#
# > 🧭 Run `day3/notebooks/00_setup.ipynb` first (or `uv run python day3/verify_setup.py`) — it
# > confirms `langgraph.json` still resolves to importable graphs and that the LangSmith SDK
# > authenticates, which is what the deploy steps below depend on. §0 is the per-lab `sys.path` and
# > `.env` bootstrap, not an environment check.
# >
# > 🧭 **Builds on Day 1 (graph) and Day 2 (evals); runs standalone for the SDK sections.** The
# > deploy steps are CLI + UI with a validation checkpoint in §1 — the notebook itself never shells
# > out. All ticket data is **synthetic/fictional** — created for this workshop, no real Acme data.


# %% [markdown]
# ### 📚 Stuck on syntax? Reference material
#
# You are not expected to write any of this from memory. When an API signature is the thing in your
# way, look it up — that is what a working engineer does, and every link below is the official source.
#
# | Need | Where to look |
# |---|---|
# | Deploying, environment variables, secrets | [Deployment](https://docs.langchain.com/langsmith/deployments) |
# | `langgraph.json` and the CLI | [CLI reference](https://docs.langchain.com/langsmith/cli) |
# | Calling a deployment from the SDK | [SDK](https://docs.langchain.com/langsmith/sdk) |
# | Online evals on production traces | [Online evaluations](https://docs.langchain.com/langsmith/online-evaluations) |
# | Dashboards, monitoring, drift | [Observability](https://docs.langchain.com/langsmith/observability) |
# | LLM Gateway: secrets, policies, redaction | [LLM Gateway](https://docs.langchain.com/langsmith/llm-gateway) |
# | Event-driven agents (Fleet) | [Fleet](https://docs.langchain.com/langsmith/fleet/code) |
# | A browser chat UI for end users | [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui) · [docs](https://docs.langchain.com/oss/python/langchain/ui) |
#
# > **Closest analogue:** the lifecycle workshop's `workshop_modules/module_3/` — production data
# > flywheel, SDK interaction, and a CI regression gate.

# %% [markdown]
# ## 0. Setup
#
# Same setup-cell pattern as Days 1–2: load `.env` first (so the model layer and the SDK see the
# right keys), then put the repo root on `sys.path` so `day1.src.*` and `utils.*` import cleanly.

# %%
import json
import os
import sys
from pathlib import Path

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

from dotenv import find_dotenv, load_dotenv

# No `override=True`: real environment variables must win over `.env`, otherwise the
# `export LANGSMITH_GATEWAY=true / export LANGSMITH_GATEWAY_API_KEY=...` gateway setup documented in
# the README (Model access section) and .env.example is silently clobbered by whatever `.env` happens to contain.
# `.env` fills the gaps; your shell wins.
load_dotenv(find_dotenv())  # FIRST — before any model/SDK client
# The model layer (day1/src/models.py) handles gateway vs direct API key routing.
# See the README (Model access section) for gateway setup instructions.

# No gateway key juggling here: `day1/src/models.py` routes both the chat model
# (`get_model`) and the embeddings client (`get_embeddings`) by inspecting the gateway env vars, and
# passes the gateway credential explicitly as `api_key=`. See the README (Model access section).

DAY1 = WORKSHOP / "day1"
LANGGRAPH_JSON = DAY1 / "langgraph.json"
assert LANGGRAPH_JSON.exists(), f"missing {LANGGRAPH_JSON} — the deployment reads it"

print("workspace:", os.getenv("LANGSMITH_WORKSPACE_ID"), "| tracing:", os.getenv("LANGSMITH_TRACING"))
print("langgraph.json:", LANGGRAPH_JSON)

# %% [markdown]
# ## 1. Deploy the ticket agent
#
# 📖 [Deployments / CLI](https://docs.langchain.com/langsmith/cli)
#
# A **deployment** is a hosted version of a LangGraph application: it runs 24/7 with autoscaling,
# exposes a REST API, manages persistence (threads, checkpoints) for you, and traces every run to
# its own LangSmith project. The deployable unit is exactly what Day 1 already built — no code
# changes:
#
# - `day1/langgraph.json` — declares the graphs, dependencies, Python version, and the env file
# - `day1/src/ticket_agent_graph.py` — the four compiled module-level graphs
#
# ### Which variant do we deploy?
#
# | Variant | Deploy? | Why |
# |---|---|---|
# | `ticket_agent` | optional | Simplest surface — good smoke test |
# | `ticket_agent_mem` | optional | Platform manages persistence; use threads to get memory |
# | `ticket_agent_hitl` | **default** | Production posture: sensitive account actions (`mfa_reset`, `account_unlock`, `password_reset`) always pause for human approval |
# | `ticket_agent_structured` | optional | When a caller (app, not a human) consumes the answer — typed `TicketResolution` output |
#
# All four can live in **one deployment**; the SDK call picks the graph by ID (§4).
#
# ### 1a. Local smoke test — `langgraph dev`
#
# Before deploying anywhere, prove the graph loads and runs as a server. In a terminal (not this
# notebook — the server is long-running):
#
# ```bash
# cd langchain_adlc_workshop/day1
# langgraph dev          # reads langgraph.json, serves all 4 graphs on http://127.0.0.1:2024
# ```
#
# `langgraph dev` gives you a local Agent Server with in-memory persistence plus a Studio link for
# interactive debugging — the same API shape as the cloud deployment, so §3–§6 work against **either**
# URL. Override the port with `--port 8123` when 2024 is busy.
#
# ### 1b. Deploy — CLI or UI (same result)
#
# **CLI** (from a machine logged in with `langgraph login`):
#
# ```bash
# cd langchain_adlc_workshop/day1
# langgraph deploy --name acme-ticket-deployment --config langgraph.json
# ```
#
# **UI** (equivalent — what most teams do first):
# 1. LangSmith → **Deployments** → **+ New Deployment**
# 2. Connect the workshop GitHub repo; pick the `day1` directory / `langgraph.json` config
# 3. Name: `acme-ticket-deployment`
# 4. **Environment variables** — see §2; paste from `.env` *minus* anything you don't want stored
# 5. Check **"Sharable through LangSmith Studio"** so you can drive the deployed agent from a web UI
# 6. Submit → wait for the ✅ (LangSmith pulls the code, builds an image, starts the server)
#
# > 💡 **Revisions.** Every redeploy creates a new revision — like a git branch for the running
# > service. Keep the graph ID stable (`ticket_agent_hitl`) so callers and evaluators don't change
# > when the code does.

# %% [markdown]
# ### Validation checkpoint — the four graphs registered
#
# Nothing to deploy from this cell: it just re-reads `day1/langgraph.json` so the notebook shows the
# exact graph IDs the deployment will serve, and fails loudly if a variant was renamed.

# %%
graphs = json.loads(LANGGRAPH_JSON.read_text())["graphs"]
for graph_id, path in graphs.items():
    print(f"  {graph_id:28s} → {path}")

TICKET_GRAPHS = [g for g in graphs if g.startswith("ticket_agent")]
assert len(TICKET_GRAPHS) == 4, "expected the 4 Day 1 ticket_agent variants in langgraph.json"
print(f"\n✔ {len(TICKET_GRAPHS)} ticket-agent variants registered; deploying graph id: ticket_agent_hitl")

# %% [markdown]
# ## 2. Environment variables & model endpoint
#
# What the deployed graph needs, and where each value goes:
#
# | Variable | Where it goes | Why |
# |---|---|---|
# | `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) | **Deployment env vars** (secrets, never in git) | The graph calls the model provider directly — the Day 1 default path |
# | `LABS_MODEL` | Deployment env var (optional) | One-knob agent-model swap (`openai:gpt-4.1-mini` default; `LABS_JUDGE_MODEL` swaps the grader tier separately) |
# | `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` | Deployment env vars | Every deployed run traces to the deployment's LangSmith project |
# | `LANGSMITH_PROJECT` | Deployment env var (optional) | Defaults to the deployment name — this is where §6 looks |
# | `LANGSMITH_GATEWAY` + `LANGSMITH_GATEWAY_API_KEY` | Deployment env vars (**production path**) | Route model calls through the **LLM Gateway**: provider key lives in LangSmith Provider Secrets, gateway enforces rate limits / spend controls / redaction. Note the key variable is deliberately *not* `LANGSMITH_API_KEY` — `langgraph deploy` strips that reserved name, so an agent reading its gateway credential from it deploys cleanly and then 401s at runtime |
# | `TAVILY_API_KEY` | Deployment env var (optional) | Adds live web search to the agent's tools when set |
#
# **Local vs deployed model endpoint — the same model layer, two contexts:**
#
# | Context | Default path | Production path |
# |---|---|---|
# | Your laptop (Days 1–2 labs) | `init_chat_model` reads the provider key directly | gateway env vars set locally (see the README (Model access section)) |
# | Cloud deployment (today) | provider key as a deployment secret | `LANGSMITH_GATEWAY=true`, bearer = `LANGSMITH_GATEWAY_API_KEY` |
#
# The graph code never changes — `day1/src/models.py::get_model()` checks the gateway env vars and
# picks the path at import time. That indirection is what makes the self-hosted/hybrid mapping in §7
# possible without code edits.
#
# > 🔐 **Secrets hygiene.** `.env` stays on your laptop (git-ignored). Deployment secrets are stored
# > by LangSmith and injected at runtime. Never paste a provider key into a notebook cell, a commit,
# > or the graph code.

# %% [markdown]
# ### Validation checkpoint — what *this* notebook will use
#
# The cells below need two coordinates: the **deployment URL** and a **LangSmith API key** the SDK
# can authenticate with. Everything is printed so a misconfiguration is visible before the first
# SDK call — and the SDK sections **skip cleanly** rather than traceback when no URL is set.

# %%
# Where is the agent? One of:
#   1. Cloud deployment URL  — from LangSmith → Deployments → acme-ticket-deployment (after §1)
#   2. http://127.0.0.1:2024 — `langgraph dev` from §1a (same API, local persistence)
DEPLOYMENT_URL = os.getenv("DEPLOYMENT_API_URL", "").strip()
LOCAL_DEV_URL = "http://127.0.0.1:2024"

if not DEPLOYMENT_URL:
    print("⚠ DEPLOYMENT_API_URL not set — set it to your deployment URL to run §3–§6 against the cloud.")
    print(f"  Falling back to the local dev server: {LOCAL_DEV_URL} (start it with `langgraph dev`).")

SDK_URL = DEPLOYMENT_URL or LOCAL_DEV_URL

# "Local" is a property of the URL, not of whether the variable happens to be set. Deriving it from
# `not DEPLOYMENT_URL` looked equivalent and was not: §3's own guidance below invites you to point
# DEPLOYMENT_API_URL at a local `langgraph dev` server, and doing exactly that used to mark the run
# as a cloud deployment — so §6 went looking for a cloud LangSmith project that does not exist and
# died with an unhandled LangSmithNotFoundError. The lab contradicted its own instructions.
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "0.0.0.0", "[::1]")
IS_LOCAL = not DEPLOYMENT_URL or any(h in SDK_URL for h in _LOCAL_HOSTS)

print("\nSDK target:", SDK_URL, "(local dev)" if IS_LOCAL else "(cloud deployment)")
print("LANGSMITH_API_KEY:", "set" if os.getenv("LANGSMITH_API_KEY") else "⛔ MISSING — required for cloud deployments")
from day1.src.models import DEFAULT_MODEL  # the resolved agent tier — see day1/src/models.py

# Ask the model layer, rather than re-deriving the check here — an inline copy of the detection
# drifts the moment a new way of switching the gateway on is supported, and then this line reports
# "gateway: False" while every model call is in fact being proxied.
from day1.src.models import _using_gateway

print("model:", os.getenv("LABS_MODEL") or f"{DEFAULT_MODEL} (default)",
      "| gateway:", _using_gateway())

# %% [markdown]
# ## 3. Connect with the LangGraph SDK
#
# `langgraph_sdk.get_client()` is the programmatic front door to a deployment — the same client
# pattern as the lifecycle workshop's module_3/section_2. This is how you:
#
# - build a custom UI or ITSM integration on top of the ticket agent
# - automate smoke tests against a deployment after each revision
# - handle HITL interrupts from code (§5) instead of the Studio UI
#
# **Auth:** cloud deployments authenticate with your `LANGSMITH_API_KEY`; the local `langgraph dev`
# server accepts any/no key. A missing key skips the SDK sections instead of raising.

# %%
# `get_sync_client` rather than `get_client`: the async client needs `await`, which is only legal at
# the top level of a *notebook* cell — a plain `python 01_deployments.py` would die on a
# `SyntaxError: 'await' outside function` before printing anything. The sync client has the same
# surface (`assistants.search()`, `threads.create()`, `runs.wait(...)`) and runs in both.
from langgraph_sdk import get_sync_client

_api_key = os.getenv("LANGSMITH_API_KEY")
if not IS_LOCAL and not _api_key:
    print("⛔ a cloud deployment needs LANGSMITH_API_KEY — set it in .env, or point")
    print("   DEPLOYMENT_API_URL at a local `langgraph dev` server (which needs no key).")
    client = None
else:
    client = get_sync_client(url=SDK_URL, **({} if IS_LOCAL else {"api_key": _api_key}))

ASSISTANT_ID = "ticket_agent_hitl"  # the graph ID from langgraph.json — the HITL variant

# Probe: list the assistants the server is actually serving. On the local dev server this always
# works; on a cloud deployment it confirms auth + connectivity in one call.
try:
    assert client is not None, "no client — see the message above"
    assistants = client.assistants.search()
    ids = sorted({a["graph_id"] for a in assistants})
    print("✔ connected — graphs served:", ", ".join(ids))
    assert ASSISTANT_ID in ids, f"{ASSISTANT_ID} not served by {SDK_URL} — check langgraph.json"
except Exception as e:
    print(f"⛔ cannot reach {SDK_URL}: {type(e).__name__}: {str(e)[:160]}")
    print("  → run `langgraph dev` in langchain_adlc_workshop/day1, or set DEPLOYMENT_API_URL to your deployment.")
    client = None

# %% [markdown]
# ## 4. Invoke the deployed agent
#
# 📖 [Deployments / CLI](https://docs.langchain.com/langsmith/cli)
#
# The core SDK loop — identical for local dev and cloud:
#
# 1. **Create a thread** (a conversation session — the platform persists its state)
# 2. **`runs.wait`** with the graph ID and an input message → blocks until the run finishes
# 3. Read the final message from the returned state
#
# First a non-sensitive question, so the run completes end-to-end without an interrupt.

# %%
if client is not None:
    thread = client.threads.create()
    thread_id = thread["thread_id"]

    response = client.runs.wait(
        thread_id,
        ASSISTANT_ID,
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "My VPN connection drops after about 2 minutes when I work from home. "
                               "What should I check? My user ID is USR-002.",
                }
            ]
        },
    )

    # `runs.wait` returns the run's final STATE, not a guaranteed message list: if the run errored
    # server-side it comes back as {"__error__": {...}}, and a graph that ends without producing a
    # message returns a state with no "messages" key at all. Indexing it blindly turns a readable
    # server-side error into a bare KeyError three frames away from the cause.
    if "__error__" in response:
        err = response["__error__"]
        print(f"⛔ the run failed server-side: {err.get('error', err)}")
        print("   check the deployment's logs in LangSmith → Deployments → Logs")
    elif response.get("messages"):
        print(f"Agent: {response['messages'][-1]['content']}")
    else:
        print(f"⚠️  run finished but returned no messages. Final state keys: {list(response)}")
    print("\nthread_id:", thread_id, "— reusing it keeps conversation state across runs")
else:
    print("⏭ skipped — no reachable deployment (see §3 output)")

# %% [markdown]
# ### What just happened?
#
# - The server compiled nothing, built nothing — the graph was already deployed; the SDK just
#   **addressed it by graph ID** and created a run on a thread
# - The agent called its tools (`search_kb`, `get_user_context`, `search_ticket_history`) *server-side*
# - The full run traced to the deployment's LangSmith project (§6)
#
# **Follow-up on the same thread** — thread state persists across runs, so the agent remembers the
# conversation (the platform supplies the checkpointer; no `MemorySaver` in deployed code):

# %%
if client is not None:
    response = client.runs.wait(
        thread_id,
        ASSISTANT_ID,
        input={"messages": [{"role": "user", "content": "Have we seen tickets like mine before? What fixed them?"}]},
    )
    print(f"Agent: {response['messages'][-1]['content']}")
else:
    print("⏭ skipped — no reachable deployment")

# %% [markdown]
# ## 5. HITL interrupt — approve a sensitive action programmatically
#
# `ticket_agent_hitl` wraps `mock_api_action` in `HumanInTheLoopMiddleware` — the run **pauses**
# before any MFA reset / account unlock / password reset. Studio lets you click approve/reject;
# the SDK pattern (module_3/section_2) is explicit:
#
# 1. `runs.wait(...)` returns with `__interrupt__` set — the run is *paused*, not failed
# 2. Inspect the interrupt payload (what action, for whom, why)
# 3. Resume on the **same thread** with `command=Command(resume=...)` — the resume value becomes the
#    return value of the interrupt inside the middleware
#
# > ⚠️ This is the production pattern for sensitive actions at Acme: the agent *prepares* the action
# > with full cited context; a human *releases* it. The approval decision is part of the trace.

# %%
from langgraph_sdk.schema import Command

if client is not None:
    hitl_thread = client.threads.create()
    hitl_thread_id = hitl_thread["thread_id"]

    result = client.runs.wait(
        hitl_thread_id,
        ASSISTANT_ID,
        input={
            "messages": [
                {
                    "role": "user",
                    # USR-002 (Marco Rossi) holds the `access` permission, so the agent is allowed
                    # to propose the reset and the HITL gate is what stops it. Pick a user without
                    # that permission (e.g. USR-003) and the agent correctly declines instead — a
                    # good thing, but then this section demonstrates authorization, not HITL.
                    "content": "Marco Rossi (USR-002) replaced his phone and his MFA token no longer "
                               "works — he is locked out of all systems. Verify his context, then "
                               "reset his MFA through the identity API.",
                }
            ]
        },
    )

    interrupts = result.get("__interrupt__", [])
    print("run status: paused" if interrupts else "run status: completed without interrupt")
    for i in interrupts:
        print("interrupt:", json.dumps(i.get("value", i), indent=2)[:800])
else:
    print("⏭ skipped — no reachable deployment")

# %% [markdown]
# ### Resume with an approval decision
#
# `HumanInTheLoopMiddleware` expects a resume value shaped like
# `{"decisions": [{"type": "approve" | "reject" | "edit", ...}]}` — one decision per queued action.
# Approving executes the (mock) identity API call; rejecting skips it and the agent explains the
# escalation path instead.

# %%
if client is not None and locals().get("result", {}).get("__interrupt__"):
    # One decision per pending action request — the middleware rejects a mismatched count.
    _pending = result["__interrupt__"][0]["value"]["action_requests"]
    resumed = client.runs.wait(
        hitl_thread_id,
        ASSISTANT_ID,
        command=Command(resume={"decisions": [{"type": "approve"} for _ in _pending]}),
    )
    print(f"Agent: {resumed['messages'][-1]['content']}")

    # Thread state shows the approved action in the history — auditable after the fact.
    state = client.threads.get_state(hitl_thread_id)
    print("\nmessages on thread:", len(state["values"]["messages"]))
else:
    print("⏭ skipped — no interrupt to resume (no deployment, or the agent answered without the tool)")

# %% [markdown]
# ## 6. Inspect deployment traces
#
# 📖 [Deployments / CLI](https://docs.langchain.com/langsmith/cli)
#
# Every SDK call above created a trace in the deployment's LangSmith project — this is the same
# trace surface used on Days 1–2, now fed by **production-style traffic** instead of lab cells.
#
# **In the UI** — LangSmith → **Projects** → the deployment project (defaults to the deployment
# name, e.g. `acme-ticket-deployment`):
#
# | What to check | Where | Why |
# |---|---|---|
# | Full conversation + tool calls | trace detail | Verify the agent searched KB / ticket history and cited sources |
# | The HITL pause + your approval | trace detail | The interrupt and resume are recorded — auditable sensitive-action trail |
# | Latency per step | trace timeline | Retrieval vs model time; where a slow run went |
# | Errors / tool failures | filtered view | First place to look when a caller reports a bad answer |
#
# **Programmatically** — pull the latest runs from the deployment project and confirm the two runs
# from §4–§5 landed (one completed, one paused → resumed):

# %%
if client is not None and not IS_LOCAL:
    from langsmith import Client as LangSmithClient

    from langsmith.utils import LangSmithNotFoundError

    ls_client = LangSmithClient()
    deployment_project = os.getenv("LANGSMITH_DEPLOYMENT_PROJECT", "acme-ticket-deployment")

    # `list_runs` resolves the project by NAME first, so a project that does not exist raises
    # rather than returning an empty list. Same trap as Day 2 Lab 05 §5 — and it fires on the
    # ordinary path here, because the project is named after a deployment you may not have created
    # yet. Catch it, or the helpful message below is unreachable for the person who needs it.
    try:
        runs = list(
            ls_client.list_runs(
                project_name=deployment_project,
                is_root=True,
                limit=10,
                select=["id", "name", "status", "start_time", "error"],
            )
        )
    except LangSmithNotFoundError:
        runs = None
        print(f"⚠ No LangSmith project named {deployment_project!r}.")
        print("  Either the deployment has not been created yet (§1), or its project has a different")
        print("  name — set LANGSMITH_DEPLOYMENT_PROJECT to match. Nothing else in the lab depends on this.")

    if runs is not None:
        print(f"latest root runs in '{deployment_project}':")
        for r in runs:
            print(f"  {r.status:10s} {r.name:24s} {r.start_time}")
        if not runs:
            print("  (none yet — traces can take a few seconds to land; re-run this cell)")
else:
    print("⏭ skipped — trace listing runs against the cloud deployment project;")
    print("   for `langgraph dev` inspect traces in your LANGSMITH_PROJECT workspace instead.")

# %% [markdown]
# **From trace to improvement.** A trace you don't like is not just a bug report — it's a dataset
# candidate. The Day 2 loop applies unchanged: send the run to an annotation queue, correct the
# output, add it to the eval dataset, and the next revision is gated on it. Lab 03 (online evals)
# automates the *detection* side of this loop for deployed runs.

# %% [markdown]
# ## 7. Mapping to self-hosted / hybrid deployment paths
#
# 📖 [Deployments / CLI](https://docs.langchain.com/langsmith/cli)
#
# The workshop deploys to managed LangSmith cloud. Acme's production reality is likely different —
# sensitive workloads, controlled networks, possibly air-gapped environments. The point of this
# lab's architecture is that **the graph, the SDK calls, and the eval loop are identical across all
# of these paths**; only *where the server runs* and *where secrets live* change.
#
# | Path | What changes | What stays the same |
# |---|---|---|
# | **Cloud (today)** | LangSmith runs the Agent Server; deployment secrets in LangSmith | graph code, `langgraph.json`, SDK calls, traces |
# | **Hybrid** | Agent Server runs in *your* infrastructure (your network, your data plane); LangSmith remains the control plane (traces, evals, deployments UI) | same graph + SDK; env vars point at your endpoints |
# | **Self-hosted** | LangSmith itself (control plane + Agent Server) deployed in your environment (e.g. Kubernetes via Helm); all secrets in your vault | same graph + SDK; base URLs point at your LangSmith |
# | **Air-gapped** | Self-hosted + no egress: model endpoint must be internal (self-hosted model or an internal gateway); offline tracing/eval | same graph; `LABS_MODEL` + `*_BASE_URL` retarget the model layer |
#
# **What carries over directly:**
#
# - **This notebook's SDK cells** — change `DEPLOYMENT_API_URL` and the API key; every call in
#   §3–§6 is unchanged (that's the payoff of addressing graphs by ID through the platform API)
# - **`langgraph.json` + the graph module** — the deployable unit is portable by design
# - **The model layer** — `day1/src/models.py::get_model()` already routes direct-key vs gateway;
#   an internal LiteLLM proxy is just another OpenAI-compatible `*_BASE_URL`
# - **The Day 2 eval loop** — offline evals gate a revision *before* deploy; online evals (Lab 03)
#   watch it *after*
#
# **What to decide per environment (the afternoon discussion):**
#
# - **Secrets** — provider key in LangSmith Provider Secrets vs your vault; gateway key issuance
# - **Model endpoint** — managed providers vs LLM Gateway vs LiteLLM vs self-hosted models
# - **Persistence** — platform Postgres (cloud) vs your Postgres (self-hosted Agent Server)
# - **Network** — endpoint allowlisting, egress rules, and what an interrupt/approval flow looks
#   like when the reviewer is on a different network segment than the agent
#
# > 🧭 **Rule of thumb:** develop against `langgraph dev`, validate against a cloud deployment,
# > and keep the graph free of environment-specific code — so the move to hybrid/self-hosted is a
# > *configuration* change, not a rewrite.

# %% [markdown]
# ## 8. Recap
#
# | Step | You did | Key artifact |
# |---|---|---|
# | Deploy | Registered 4 graph variants; `langgraph dev` locally, `langgraph deploy` / UI to cloud | `day1/langgraph.json` |
# | Configure | Mapped env vars + model endpoint (direct key vs gateway) to deployment secrets | §2 table |
# | Invoke | Threads + `runs.wait` via LangGraph SDK; follow-up on the same thread | `get_client`, `client.runs.wait` |
# | HITL | Paused on `mock_api_action`, resumed with `Command(resume={"decisions": ...})` | §5 |
# | Inspect | Deployment traces in the UI + `langsmith.Client.list_runs` | §6 |
# | Map | Cloud → hybrid → self-hosted → air-gapped without code changes | §7 table |
#
# **Next:** Lab 02 (`02_managed_deep_agents.py`) covers the *managed* deployment path for the Day 2
# deep agent; Lab 03 (`03_online_evals.py`) then attaches online evaluators to these traces and
# routes failures to review — the event-driven improvement loop this deployment now feeds.
#
# > 🧭 **Bridge to the afternoon:** keep your `DEPLOYMENT_API_URL` and thread IDs around — the
# > production improvement exercise (advanced exercise 4) builds on the runs you created here.
