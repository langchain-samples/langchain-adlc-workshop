# LangChain ADLC Workshop

A hands-on, three-day workshop that walks one agent through the whole **Agent Development
Lifecycle** — build it, trace and debug it, evaluate it, deploy it, then monitor, govern and
improve it in production.

The point is the *lifecycle*, not a demo. You build two agents against synthetic data committed to
this repo — an IT ticket-resolution agent and a vendor due-diligence Deep Agent — and then do the
work that usually gets skipped: measure whether they are actually any good, put a human in the loop
before anything sensitive happens, deploy them, and close the loop from production signals back
into the next improvement.

Built with LangChain, LangGraph, Deep Agents and LangSmith. Every lab is both a runnable script and
a notebook, and works with either a direct provider API key or the LangSmith LLM Gateway.

All data in this repository is synthetic. The vendors, tickets, users, knowledge-base articles, PDFs
and screening lists were created for the workshop and are not intended to represent any real
organisation, person, or sanctions listing; any resemblance to an actual entity is unintended. The
two optional integrations return live third-party data when you supply your own key.


## What This Workshop Teaches

Participants build two agents across the 3 days:

| Use Case | Day | What It Teaches |
|---|---|---|
| **Ticket resolution / customer support agent** | Day 1 | Agent loop: multi-agent orchestration, RAG/KB retrieval, ticket database lookup, API/MCP integration, HITL, user-scoped access, structured outputs, tracing/debugging |
| **Vendor Due Diligence** | Day 2–3 | Deep Agents harness, sub-agents, wiki memory, RubricMiddleware, evals, experiments, deployments, online evals, observability, governance |

The workshop follows the **ADLC** (Agent Development Lifecycle): build → trace/debug → evaluate → deploy → monitor → govern → improve. **Loop Engineering** is the complementary design lens: agent loop (Day 1) → verification loop (Day 2) → event-driven + hill-climbing loops (Day 3).

## The Two Use Cases

### 1. Ticket resolution — Day 1

A user submits a support ticket (*"I'm locked out and need an MFA reset"*) and the agent diagnoses
and resolves it, escalating to a human before anything sensitive happens.

1. **Searches** the support knowledge base (RAG over KB articles) for resolution guidance
2. **Queries the ticket database** — real SQLite, for prior tickets, aggregates and joins
3. **Checks the requester's entitlements** and returns only what they are authorised to see
4. **Proposes a sensitive action** (MFA reset, account unlock) — held for human approval
5. **Returns a structured outcome**: summary, category, cited sources, action taken, confidence

Day 1 builds this twice: once as a `create_agent` loop, then as an explicit `StateGraph` where
routing and source validation are properties of the code rather than the prompt.

### 2. Vendor due diligence — Days 2–3

A procurement officer asks whether a vendor is suitable. A **Deep Agent** supervisor delegates to
three specialists and synthesises one report — the longer-running, multi-step counterpart to Day 1.

1. **Evidence collector** gathers cited capability and certification evidence from the vendor KB
2. **Risk assessor** classifies risk from the vendor database and lists signals with severity
3. **Compliance screener** checks sanctions and watchlists — a match is an automatic escalation
4. **Operating rules** come from `AGENTS.md`; four `SKILL.md` workflows cover evidence review,
   source validation, risk classification and report drafting
5. **Durable notes** persist to a wiki the agent maintains across runs
6. **The report** carries a citation on every material claim, a screening verdict, a suitability
   recommendation, and a human-review flag

Days 2–3 then evaluate this agent — datasets, judges, trajectory checks, experiments, annotation
queues — and deploy it with online evals and monitoring.

### The ticket database — and why the agent gets two ways to query it

A real ticketing system is a database, so this one is too. `day1/src/ticket_db.py` builds a
normalised SQLite database (`tickets`, `users`, `user_permissions`, `ticket_events`) from the
committed JSON, rebuilding automatically whenever the JSON changes — so it can never drift, and it
is not committed as a binary.

The agent gets **two tiers**, which is the lesson:

| Tier | Tools | Where |
|---|---|---|
| **Fixed query tools** | `search_ticket_history` (parameterised filters), `ticket_statistics` (aggregates) | Day 1 Lab 02 |
| **Model-authored SQL** | `query_ticket_db` — the model writes the query, schema in its prompt | Day 1 Lab 03 §6 |

Start with fixed tools: predictable, cheap to trace, impossible to misuse. Reach for model-authored
SQL only when you have **evidence** the fixed tools are the bottleneck — Lab 02 demonstrates exactly
where they run out. This mirrors the
[reference workshop](https://github.com/langchain-ai/langsmith-agent-lifecycle-workshop), which
ships `db_agent.py` (fixed tools) and `sql_agent.py` (dynamic SQL) as separate agents.

**The SQL is guarded in five layers**, following LangChain's guidance to *"consider issuing
READ-ONLY credentials"* and *"combine multiple layered security approaches"*: multiple statements
rejected · `SELECT`/`WITH` only · mutating keywords blocked · results row-capped · and the
connection opened **read-only at the SQLite level** (`file:…?mode=ro`), so a write is refused by the
database engine rather than by a string check.

### Two ways to build a graph

Most of the workshop uses `create_agent`, which builds the graph for you — the model decides what
happens next. **Day 1 Lab 03 §4** drops a level and builds an explicit `StateGraph`: typed state,
conditional routing on *issue type* and on *confidence*, a source-validation node on every path to
`END`, `interrupt()` for approval, and a checkpointer for persistence.

The deciding question is *who is allowed to skip a step*. A validation step the model may skip is a
suggestion; a validation **node** is a control. Most production systems use both — an explicit graph
whose nodes are agents.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  DEMO UI (Next.js + LangGraph JS SDK)                              │
│  http://localhost:3000                                              │
│  - Streaming chat interface                                         │
│  - Tool call cards                                                  │
│  - Structured output display                                        │
│  - HITL approval buttons                                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LangGraph Server (:2024)                                           │
│  - ticket_agent (plain ReAct)                                       │
│  - ticket_agent_mem (with memory)                                   │
│  - ticket_agent_hitl (with HITL)                                    │
│  - ticket_agent_structured (with structured output)                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Agent (create_agent)                                               │
│  - Model: openai:gpt-4.1-mini (override with LABS_MODEL)            │
│  - 5 tools: search_kb, search_ticket_history, mock_api_action,     │
│    get_user_context, tavily_search (optional)                       │
│  - Middleware: PIIMiddleware, ToolCallLimitMiddleware,              │
│    custom @before_model (prompt injection guard)                    │
│  - Structured output: TicketResolution (Pydantic)                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Data Layer                                                         │
│  - tickets.json (sample support tickets)                            │
│  - kb_tickets/*.md (KB articles for ticket resolution)              │
│  - users.json (user entitlements / scopes for user-scoped access)   │
│  - vendors.json, kb/*.md, pdfs/*.pdf (retained for Day 2 DD)        │
│  - prompt.md (system prompt)                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Multi-Agent Pattern

Day 1 Lab 03 introduces the **supervisor + sub-agent pattern**:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Supervisor Agent (ticket_orchestrator)                             │
│  - Routes queries to the right specialist                           │
│  - Synthesizes responses from sub-agents                            │
│  - Calls mock_api_action for privileged actions (HITL)              │
└─────────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌─────────────────────────┐           ┌─────────────────────────┐
│  kb_search_specialist   │           │  ticket_history_specialist│
│  - search_kb             │           │  - search_ticket_history│
│  - tavily_search (opt.)  │           │  - get_user_context     │
│  Focus: KB / web search │           │  Focus: ticket DB / user│
└─────────────────────────┘           └─────────────────────────┘
```

**How it works:**
1. Each sub-agent is a `create_agent` with its own tools and system prompt
2. Each sub-agent is wrapped as a `@tool` function for the supervisor
3. The supervisor's system prompt tells it when to delegate to each specialist
4. The supervisor synthesizes the specialists' responses into a coherent answer

This is the **manual version** of what `create_deep_agent` automates on Day 2.

## How to Run

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- An **OpenAI API key** and a **LangSmith API key**

#### Optional third-party keys

Both are **genuinely optional** — every lab runs end to end without them, against the committed
synthetic fixtures. They exist so you can see the same tools hitting a live external service, which
is what a real deployment does.

| Service | What it adds | Used by | Free account |
|---|---|---|---|
| **Tavily** | Live web search when the bundled KB does not cover a question | `tavily_search`, added automatically when `TAVILY_API_KEY` is set (Day 1 Lab 02, vendor discovery) | Free tier — sign up at [tavily.com](https://tavily.com/) |
| **OpenSanctions** | Live sanctions/watchlist screening alongside the local synthetic list | `screen_vendor`, when `OPENSANCTIONS_API_KEY` is set (Day 2 due diligence) | Free trial — sign up at [opensanctions.org/account](https://www.opensanctions.org/account/) |

Set either in `.env` (see `.env.example`). Without them:

- `tavily_search` is simply absent from the tool list — the agent uses the KB only.
- `screen_vendor` still screens against `day1/data/sanctions_list.json`, which is deterministic and
  offline-safe. The sanctioned-vendor demo (VND-013) works either way.

> Both are third-party services with their own terms — review them before use beyond the free tier.

### Setup

Run these from the workshop root — the directory containing `day1/` and `utils/`.

```bash
# 1. Install dependencies
uv sync

# 2. Create your .env
cp .env.example .env
```

Fill in four values in `.env`:

```bash
PARTICIPANT=                        # your initials — see the note below
OPENAI_API_KEY=sk-proj-...          # or use the gateway instead — see "Model access" below
LANGSMITH_API_KEY=lsv2_pt_...       # tracing
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=langchain-adlc-workshop
```

> 👥 **Sharing a LangSmith workspace? Set `PARTICIPANT` to your initials.** Datasets, annotation
> queues and tracing projects are workspace-global, so without it everyone in the room seeds the same
> dataset and Day 2's before/after experiment comparison becomes meaningless. With `PARTICIPANT=tr`
> your artifacts become `vendor-due-diligence-eval-tr`, `langchain-adlc-workshop-tr`, and so on. Leave it blank
> when working alone.

```bash
# 3. Confirm the setup works BEFORE you start Lab 01
uv run python verify_setup.py
```

`verify_setup.py` resolves your model access path the same way the labs do, then proves each piece
actually answers — agent model, judge model, embeddings, LangSmith auth, the shipped data (ticket
database, vendor PDFs, RAG corpora, Day 2 skills and eval sets), the cross-references between them,
the RAG index, and the deployable graphs. If it prints `✅ Ready`, the labs will run. If it fails, it
tells you which piece and where to look.

**Re-check each morning.** The workshop edits its own fixtures as it goes, so passing on Day 1 does
not prove Day 3 will run. Each day has its own check that covers just that day's prerequisites:

```bash
uv run python day1/verify_setup.py     # or: uv run python verify_setup.py --day 1
uv run python day2/verify_setup.py     # Day 2 fixtures: agent skills, eval sets, wiki seeds
uv run python day3/verify_setup.py     # Day 3 also validates langgraph.json and the deploy tooling
```

Day 2 and Day 3 also open with a setup notebook, if you would rather see each check run in a cell:
`day2/notebooks/00_setup.ipynb` and `day3/notebooks/00_setup.ipynb`. They run the same checks and
then prove the day's central piece works — Day 2 builds the deep agent, Day 3 lists the graphs
`langgraph.json` declares and authenticates the LangSmith SDK. (Day 1's equivalent is Lab 01.)

Nothing needs building first. The ticket database and the vendor PDFs ship in the repo, so a fresh
clone is ready to run — `build_database()` still rebuilds automatically if you edit
`tickets.json` or `users.json`, so your own edits cannot leave the database stale.

<details>
<summary>Maintaining the synthetic data (not a workshop step)</summary>

Participants never regenerate anything — the fixtures, the ticket database and the vendor PDFs all
ship in the repo. Two generators exist for whoever edits the fixtures:

```bash
uv run python day1/data/generate_pdfs.py      # vendor PDFs, from vendors.json
uv run python day1/data/build_ticket_db.py   # tickets.db, from tickets.json + users.json
```

Both are deterministic and derive entirely from the committed JSON, so they are also the record of
how the synthetic data was produced. After editing a fixture, re-run the relevant generator and
`uv run python verify_setup.py` — it checks the shipped data against the JSON and will tell you if
they have diverged.

</details>

### Run the labs

Each lab exists twice, from one source: a Jupyter notebook and a plain script. Use whichever you
prefer — they cannot drift, because the notebooks are generated from the `.py` files.

```bash
# Notebooks (recommended for the workshop)
uv run jupyter notebook
#   → day1/notebooks/01_setup.ipynb, then 02 … 05 in order

# Or the same labs as scripts
uv run python day1/src/01_setup.py                  # environment probes
uv run python day1/src/02_langchain_foundations.py  # create_agent + 4 tools + RAG + structured output
uv run python day1/src/03_langgraph_hitl.py         # memory, human-in-the-loop, sub-agents
uv run python day1/src/04_traces_prompt_hub.py      # trace → diagnose → improve the prompt
uv run python day1/src/05_sensitive_controls.py     # middleware, PII, user-scoped access, escalation
```

Day 2 (`day2/src/`) covers the vendor due diligence Deep Agent, wiki memory, and evals. Day 3
(`day3/src/`) covers deployment, Managed Deep Agents, online evals, observability,
governance, and knowledge-base / large-data architecture. Both open with `00_setup` — run it first,
then `01` onward in order:

```bash
uv run python day2/src/00_setup.py    # Day 2 prerequisites, then builds the deep agent
uv run python day3/src/00_setup.py    # Day 3 prerequisites, then checks the deploy surface
```

### Open LangGraph Studio

```bash
cd day1
uv run langgraph dev
# Then open: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

Registers eight graphs: `ticket_agent`, `ticket_agent_mem`, `ticket_agent_hitl`,
`ticket_agent_structured` (Day 1's use case) and the four `vendor_discovery_agent_*` equivalents
(kept as the Day 2 reference).

### A browser chat UI for end users

Studio (above) is the *developer* view — graph, state, interrupts. For a **chat interface an end
user would actually use**, LangChain ships one, open source, and it needs no code from you:

```bash
# with `langgraph dev` already running on :2024, in a second terminal:
npx create-agent-chat-app@latest      # or: git clone https://github.com/langchain-ai/agent-chat-ui
```

Point it at `http://localhost:2024` and choose a graph (`ticket_agent`, or `ticket_agent_hitl` to
exercise approvals in the browser). [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui) is
a Next.js app that works with any `create_agent` or LangGraph agent and renders tool calls,
interrupts, and time-travel/state-forking — see the
[Agent Chat UI docs](https://docs.langchain.com/oss/python/langchain/ui).

| Want to show… | Use |
|---|---|
| Graph structure, state at each node, interrupts | **LangGraph Studio** (`langgraph dev`) |
| What a support engineer would see and type | **Agent Chat UI** |
| A one-question smoke test, no browser | `uv run python day1/src/02_langchain_foundations.py` |

**Approvals work in the browser, verified against the UI's own source.** Agent Chat UI ships an
*Agent Inbox* that renders human-in-the-loop interrupts as approve / edit / reject controls. Its
contract ([`agent-inbox/types.ts`](https://github.com/langchain-ai/agent-chat-ui/blob/main/src/components/thread/agent-inbox/types.ts))
expects `action_requests` and `review_configs`, and that is exactly what
`HumanInTheLoopMiddleware` emits here — checked field by field against a live interrupt from
`ticket_agent_hitl`:

```jsonc
{
  "action_requests": [{ "name": "mock_api_action",
                        "args": {"action": "mfa_reset", "user_id": "USR-002", "reason": "…"},
                        "description": "Tool execution requires approval…" }],
  "review_configs":  [{ "action_name": "mock_api_action",
                        "allowed_decisions": ["approve", "edit", "reject", "respond"] }]
}
```

So `ticket_agent_hitl` gives a working approval UI with no glue code. One caveat: our
`allowed_decisions` includes `respond`, which is not in the UI's `DecisionType`
(`approve | edit | reject`) — the three it knows all render; `respond` is simply ignored.

> Studio also has a **chat mode** for a simpler conversational view, available for graphs whose
> state extends `MessagesState` — which the `create_agent` graphs here do.
>
> This repo also ships `day1/demo-ui.config.json` (title, example prompts, tool labels) for the
> presenter UI used by the workshop team. It is optional — nothing in the labs depends on it.

## Model access: direct key or LLM Gateway

Every lab gets its model from one function, `get_model()` in [`day1/src/models.py`](day1/src/models.py).
It supports two paths and no lab needs to know which is active:

| | **Direct API key** (default) | **LangSmith LLM Gateway** |
|---|---|---|
| Configure | `OPENAI_API_KEY`, leave `OPENAI_BASE_URL` unset | `LANGSMITH_GATEWAY=true` + `LANGSMITH_GATEWAY_API_KEY`, and comment `OPENAI_API_KEY` out |
| Who holds the provider key | you | LangSmith, as a workspace Provider Secret |
| Policies (PII redaction, spend caps) | none | enforced by the gateway |
| Tracing | via `LANGSMITH_TRACING` | automatic for every proxied call |

Both paths are verified end to end, including embeddings.

> **Most customers should start on the direct-key path.** The gateway needs a workspace admin to
> add a Provider Secret and issue a service key, so unless that has already been done for you, use
> `OPENAI_API_KEY` and come back to the gateway later — every lab behaves identically either way.
>
> Official documentation for the gateway path:
> [LLM Gateway overview](https://docs.langchain.com/langsmith/llm-gateway) ·
> [Provider secrets & policies](https://docs.langchain.com/langsmith/llm-gateway) ·
> [Admin & API keys](https://docs.langchain.com/langsmith/administration-overview)

<details>
<summary><b>How the gateway path works</b> (click to expand)</summary>

On the gateway path your app holds **no provider key**. It authenticates to the gateway with a
LangSmith **service key**, and the gateway uses the workspace's stored Provider Secret to reach
OpenAI:

```
your agent ──HTTPS──> LangSmith LLM Gateway ──proxied──> OpenAI
   api_key=LANGSMITH_API_KEY_GATEWAY   │
                                       ├─ redact PII / secrets
                                       ├─ enforce spend caps
                                       └─ trace the call (redacted)
```

**Set up (one-time, by a workspace admin):**

1. *Settings → Workspace → Integrations → Provider Secrets* → add `OPENAI_API_KEY`. This is the only
   place the provider key lives.
2. *Settings → API Keys → + API Key* → Key Type **Service Key**, scoped to your workspace, role
   **`gateway-user`**.
3. Optionally *Settings → LLM Gateway → Policies* → enable Data Protection (PII/secret redaction)
   and Cost Controls.

**Then per machine:**

```bash
export LANGSMITH_GATEWAY="true"                        # or a regional URL, e.g. https://eu.gateway.smith.langchain.com
export LANGSMITH_GATEWAY_API_KEY="lsv2_sk_your_service_key"
# and comment OPENAI_API_KEY out of .env — you should not need it at all
```

Exported variables win over `.env` (the labs call `load_dotenv` without `override`), so the `export`
form above takes effect.

To route **one** provider through the gateway and leave the rest alone, set that provider's base URL
instead — a provider base URL takes precedence over the switch:

```bash
export OPENAI_BASE_URL="https://gateway.smith.langchain.com/openai"
```

**Three details that matter:**

- **Why its own variable name?** `langgraph deploy` strips the reserved `LANGSMITH_API_KEY` during
  upload, so an agent reading its gateway credential from it would deploy fine and then fail at
  runtime. `LANGSMITH_GATEWAY_API_KEY` survives. `models.py` also accepts the older
  `LANGSMITH_API_KEY_GATEWAY` and `LC_GATEWAY_KEY` spellings, so an existing `.env` keeps working.

- **The switch covers chat models, not embeddings.** `LANGSMITH_GATEWAY` is honoured by LangChain's
  chat models, which resolve the gateway endpoint themselves. `OpenAIEmbeddings` does **not** honour
  it — construct one with no provider key in the environment and it raises `OpenAIError: Missing
  credentials` before you ever make a request. That is why the RAG labs call `get_embeddings()`
  instead of building the client directly: it routes embeddings explicitly, so chat and retrieval
  stay on the same path.
- **The redaction lifecycle.** The gateway redacts before forwarding, so the *model never sees* the
  raw value; the trace stores only `[SAFE_TO_USE:EMAIL_ADDRESS_<hash>]` (a stable fingerprint, so you
  can still correlate); and the real value is substituted back into the response, so the end user sees
  it. You cannot get that from a client-side wrapper, because the client is the thing you are not
  trusting.

Embeddings route the same way — call `get_embeddings()` rather than constructing `OpenAIEmbeddings`
directly and it follows whichever path is active.

</details>

### Model tiers

| Tier | Model | Override | Used by |
|---|---|---|---|
| agent | `openai:gpt-4.1-mini` | `LABS_MODEL` | every lab's agent |
| heavy | `openai:gpt-4.1` | — | Day 2 Lab 04 A/B comparison |
| judge | `openai:gpt-5.4` | `LABS_JUDGE_MODEL` | LLM-as-judge evaluators, `RubricMiddleware` |

The judge tier is deliberately stronger than the agent it grades. The reasoning, and the
measurements behind it, are at the top of [`day1/src/models.py`](day1/src/models.py).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `langgraph dev` seems to serve the wrong graphs | **Port 2024 was already taken.** The CLI silently falls back to a random port and prints `Port 2024 is already in use, using port NNNNN instead.` — so requests to `:2024` hit somebody else's server. Read the port out of the startup log, or free 2024 first (`lsof -nP -iTCP:2024 -sTCP:LISTEN`). |
| `Missing credentials … set OPENAI_API_KEY` | Neither path configured. Set `OPENAI_API_KEY`, or configure the gateway — see the gateway section below. |
| `NameError: name '__file__' is not defined` | You are running a notebook cell copied out of context. Run the setup cell at the top of the notebook first. |
| `Could not locate the workshop root` | Run from inside the repo (the directory holding `day1/` and `utils/`). |
| `langgraph dev` won't start | Run it from `day1/`, where `langgraph.json` lives. |
| Traces don't appear in LangSmith | `LANGSMITH_TRACING=true` and a valid `LANGSMITH_API_KEY`; check `LANGSMITH_PROJECT`. |
| Evals fail but the agent works | On the gateway path, the model allow-list must include the judge tier. |
| **Gateway path: `402 … request blocked by gateway policies`** | **A spend policy has been hit — the workspace, key or user is over its cap.** Read the reason off the response header rather than guessing: `x-langsmith-gateway-metadata: {"outcome":"blocked","reason":"spend_limit","scopes":["workspace"],"limit_windows":["monthly"]}`. Raise or reset the cap at *Settings → Gateway → LLM Gateway* ([spend policies](https://docs.langchain.com/langsmith/llm-gateway-spend-policies)). Note that **`max_retries` does not help here** — the OpenAI SDK retries 408/409/429/5xx, not 402, so the first blocked call fails the cell outright. Budget for a room of participants running eval loops, not for one developer. |
| Gateway path: `Missing credentials … set OPENAI_API_KEY`, but only in the RAG labs | Something built `OpenAIEmbeddings(...)` directly. `LANGSMITH_GATEWAY` routes chat models only; call `get_embeddings()` so embeddings follow the same path. |

## What's in this repo

| Path | Purpose |
|---|---|
| `README.md` | This file — setup, how to run, model access, troubleshooting |
| `verify_setup.py` | First-run check — proves your credentials, data and RAG index work (`--day 1\|2\|3` to check one day) |
| `dayN/verify_setup.py` | Per-day check — run it each morning; the fixtures change as the workshop goes |
| `day1/` | Ticket-resolution agent: tools, RAG, structured output, HITL, middleware, user-scoped access |
| `day2/` | Vendor due diligence Deep Agent: context, skills, wiki memory, evals (incl. the full eval pyramid) |
| `day3/` | Deployment, Managed Deep Agents, online evals, observability, governance, knowledge-base & large-data architecture |
| `utils/` | Shared helpers — trace links, Prompt Hub access |
| `day1/langgraph.json` | Graph registry for `langgraph dev` / LangGraph Studio |

Each lab exists as both a `.py` script and a `.ipynb` notebook generated from it, so the two cannot
drift.

## Reference Repositories

| Repo | Named in the agenda | What this workshop took from it |
|---|---|---|
| [langsmith-agent-lifecycle-workshop](https://github.com/langchain-ai/langsmith-agent-lifecycle-workshop) | ✅ agenda | The **SQL ticket database**; the fixed-tool (`db_agent.py`) vs model-authored-SQL (`sql_agent.py`) split; supervisor + HITL via `interrupt()`; eval and simulation patterns |
| [deepagents](https://github.com/langchain-ai/deepagents) | ✅ agenda | The Day 2 harness — `create_deep_agent`, sub-agents, backends, and the first-class `skills=` / `memory=` parameters |
| [langgraph-101](https://github.com/langchain-ai/langgraph-101) | ✅ agenda | LangGraph fundamentals; conditional-edge variants; a deep agent that confirms the `skills=`/`memory=` usage |
| [lca-deepagents](https://github.com/langchain-ai/lca-deepagents) | ✅ agenda | LangChain Academy course on Deep Agents |
| [lca-lc-foundations](https://github.com/langchain-ai/lca-lc-foundations) | ✅ agenda | LangChain Academy foundations — models, prompting, tools, memory, MCP, state, multi-agent |
| [lca-reliable-agents](https://github.com/langchain-ai/lca-reliable-agents) | ✅ agenda | Reliability and eval patterns, including a schema-before-query evaluator |
| [agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui) | ✅ verified | The browser chat UI documented above; its Agent Inbox contract is what the HITL payload here was verified against |
| [openevals](https://github.com/langchain-ai/openevals) · [agentevals](https://github.com/langchain-ai/agentevals) | ✅ dependency | Direct dependencies: prebuilt judges, multi-turn simulators, trajectory evaluators |

`agenda` = named in the workshop agenda. `dependency` = imported directly by the labs
(`openevals` 8 import sites, `agentevals` 6; both pinned in `pyproject.toml`). `verified` =
`agent-chat-ui` is not a Python dependency, but its Agent Inbox contract was checked field-by-field
against a live interrupt from `ticket_agent_hitl`, so the browser approval flow is known to work.

## License

MIT License — see [LICENSE](LICENSE) for details.

**Copyright (c) 2026 LangChain, Inc.**
