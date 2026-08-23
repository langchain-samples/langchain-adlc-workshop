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
# # 03 · LangGraph Workflow + HITL — Ticket Resolution
#
# **Workshop:** LangChain ADLC Workshop · **Day 1** · **ADLC stage:** Orchestrate
#
# > **Loop Engineering focus: Agent loop** — LangGraph adds explicit state, routing, orchestration,
# > and human-in-the-loop checkpoints to the agent loop built in Lab 02.
#
# > Self-directed module · ~15 min
#
# ```mermaid
# graph TD
#     A[User query] --> B[Agent model]
#     B -->|tool call| C{Tool call?}
#     C -->|search_kb| D[IT support KB]
#     C -->|fixed tools / model-authored SQL| E[(tickets.db · SQLite)]
#     C -->|get_user_context| F[User context]
#     C -->|mock_api_action| G[Sensitive API action]
#     D --> B
#     E --> B
#     F --> B
#     G --> B
#     B -->|final response| I[Answer]
#
#     subgraph HITL["Human-in-the-loop (optional)"]
#     J[interrupt_on: mock_api_action] --> K[Human approval]
#     K -->|approve| L[Execute tool]
#     K -->|reject| M[Skip tool]
#     end
# ```
#
# By the end you can:
# - give the agent **memory** across turns with `checkpointer=MemorySaver()` and a `thread_id`
# - **pause for human approval** before executing sensitive account actions (MFA reset, account
#   unlock) with `HumanInTheLoopMiddleware`
# - open the *same* agent in **LangGraph Studio** and drive it (including an interrupt) by hand
#
# | What's new | How (one new `create_agent` kwarg / primitive) |
# |---|---|
# | Memory | `+ checkpointer=MemorySaver()` |
# | Human-in-the-loop | `+ middleware=[HumanInTheLoopMiddleware(...)]` |
# | Studio | `ticket_agent_graph.py` + `langgraph.json` |
#
# > 🧭 **Builds on Lab 02; runs standalone.** Tools + prompt are Lab 02's (imported from the deployable
# > graph module); what's *shown* here is the new runtime wiring — memory, approval, and Studio.


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
# ## 1. Build the baseline agent
#
# 📖 [Agents (create_agent)](https://docs.langchain.com/oss/python/langchain/agents) · [reference](https://reference.langchain.com/python/langchain/agents/factory/create_agent)
#
# The tools and prompt are Lab 02's (imported from the deployable graph module). The `create_agent`
# call is the lesson, so it's shown here and then repeated with one added kwarg for each capability.

# %%
from langchain.agents import create_agent

from day1.src.ticket_agent_graph import (
    PROMPT_NAME,
    PROMPT_PATH,
    ticket_tools,
)
from day1.src.models import get_model
from utils.prompts import get_prompt

model = get_model()
tools = ticket_tools()
system_prompt = get_prompt(PROMPT_NAME, fallback=PROMPT_PATH.read_text().strip())

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
)

# %%
from IPython.display import Image, display

display(Image(agent.get_graph().draw_mermaid_png()))

# %% [markdown]
# ## 2. Add memory with a checkpointer
#
# 📖 [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
#
# A **checkpointer** persists conversation state by `thread_id`. The follow-up can ask about "that
# user" or "that ticket" without repeating the issue details.
#
# > 💡 The graph **shape is unchanged** from §1 — a checkpointer adds *persistence keyed by
# > `thread_id`*, not new nodes. Memory is runtime state, not structure.

# %%
import uuid

from langgraph.checkpoint.memory import MemorySaver

agent_with_memory = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=MemorySaver(),  # ← only addition vs §1
)

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

# %%
from utils.trace import ask

r1 = ask(agent_with_memory,
    "Ingrid Johansson (USR-003) can't connect to the VPN from home. What troubleshooting steps do you recommend?",
    config=config)
print(r1[:200])

# %%
# Follow-up — the agent remembers the previous context
r2 = ask(agent_with_memory,
    "Have we seen similar VPN tickets before, and how were they resolved?",
    config=config)
print(r2[:200])

# %% [markdown]
# ## 3. Human-in-the-loop checkpoint
#
# 📖 [Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
#
# In a support workflow, some actions should be **reviewed by a human** before they execute —
# especially sensitive account actions like an MFA reset or account unlock. `HumanInTheLoopMiddleware`
# lets the agent **pause** before specific tool calls and wait for approval.
#
# The middleware intercepts tool calls and raises an **interrupt** — the run stops and returns the
# proposed tool call for human review. The human can approve, edit, or reject it.
#
# ```
# invoke → ⏸️ interrupt → decide → Command(resume=...) → ▶️ continue
# ```
#
# > **Two HITL payload shapes exist in the wild — know which one you are looking at.** If you search
# > for LangGraph human-in-the-loop examples you will meet both, and they are not interchangeable:
# >
# > | | Emitted by | Payload | Resume with |
# > |---|---|---|---|
# > | **Current** *(this lab)* | `HumanInTheLoopMiddleware` | `action_requests[]` + `review_configs[]` | `{"decisions": [{"type": "approve"}]}` |
# > | **Hand-rolled** | your own `interrupt([request])` | `action_request` *(singular)* + `config` with `allow_accept` / `allow_edit` / `allow_respond` / `allow_ignore` | `{"type": "response", "args": …}` |
# >
# > The ADLC workshop's own
# > [`supervisor_hitl_agent.py`](https://github.com/langchain-ai/langsmith-agent-lifecycle-workshop/blob/main/agents/supervisor_hitl_agent.py)
# > uses the hand-rolled form — a dedicated node that asks the user for something:
# >
# > ```python
# > def collect_email(state) -> Command[Literal["verify_customer"]]:
# >     user_input = interrupt(value="Please provide your email:")
# >     return Command(update={"messages": [HumanMessage(content=user_input)]}, goto="verify_customer")
# > ```
# >
# > That is the right tool when the gate is on something that is *not* a tool call. The middleware
# > form is what
# > [Agent Chat UI's Agent Inbox](https://github.com/langchain-ai/agent-chat-ui/blob/main/src/components/thread/agent-inbox/types.ts)
# > expects today, which is why the browser approval UI works here with no glue code.
# >
# > Rule of thumb: gating a **tool call** → middleware. Gating a **decision or a graph edge** →
# > hand-rolled `interrupt()`, as this lab's §4 `human_review` node does.
#
# | Decision | Effect |
# |---|---|
# | ✅ `approve` | run the tool as proposed |
# | ✏️ `edit` | change the args, then run |
# | ⛔ `reject` | skip it; tell the model why |
#
# > 💡 HITL needs persistence to pause/resume — hence the checkpointer here. In Studio, managed
# > persistence provides the same pause/resume behavior.

# %%
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command

from utils.trace import invoke_traced  # .invoke + printed trace link, returns the full result

hitl_agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=MemorySaver(),  # same as memory agent — HITL needs persistence to pause/resume
    middleware=[HumanInTheLoopMiddleware(interrupt_on={
        "mock_api_action": True,   # pause before any sensitive account action (MFA reset, unlock)
    })],
)

hitl_thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

# %% [markdown]
# **The shape changed.** Unlike the checkpointer, HITL is *middleware* — so it restructures the graph:
# a `HumanInTheLoopMiddleware.after_model` node now sits after the model, where the interrupt fires.
# Render it and compare to §1.

# %%
display(Image(hitl_agent.get_graph().draw_mermaid_png()))

# %% [markdown]
# **⏸️ Invoke → interrupt.** The run stops before `mock_api_action` and returns the proposed
# sensitive action.

# %%
result = invoke_traced(
    hitl_agent,
    {"messages": [{"role": "user", "content":
        "Marco Rossi (USR-002) replaced his phone and his MFA token isn't working — he's locked out "
        "of all systems. Verify his context and reset his MFA."}]},
    config=hitl_thread,
)

# One interrupt can carry SEVERAL proposed calls: the middleware batches every gated tool call the
# model asked for in that turn into a single `action_requests` list. Print all of them.
if result.get("__interrupt__"):
    requests = result["__interrupt__"][0].value["action_requests"]
    print(f"⏸️  paused — {len(requests)} action(s) awaiting approval:")
    for req in requests:
        print(f"    {req['name']} | args: {req['args']}")
else:
    print("No interrupt fired; final answer:\n", result["messages"][-1].content[:500])

# %% [markdown]
# **▶️ Decide → resume.** Approve until the run finishes; the model may request more than one action.

# %%
# The resume payload needs **one decision per pending action request** — the middleware raises
# "Number of human decisions (N) does not match number of hanging tool calls (M)" otherwise. Build
# the list from the interrupt rather than hardcoding a single approval.
approvals = 0
while result.get("__interrupt__"):
    pending = result["__interrupt__"][0].value["action_requests"]
    decisions = [{"type": "approve"} for _ in pending]
    approvals += len(decisions)
    result = hitl_agent.invoke(Command(resume={"decisions": decisions}), config=hitl_thread)

print(f"✅ {approvals} approval(s) → run complete")
print(result["messages"][-1].content[:400])

# %% [markdown]
# ### Discussion prompts
#
# | Question | Consider |
# |---|---|
# | What should require review? | MFA resets, account unlocks, access changes, P1/security tickets |
# | What should be redacted? | User emails, phone numbers, internal hostnames before sharing externally |
# | What should be logged? | Who approved, when, the original proposed action, the approved version |
# | What should not go to the model? | Other users' ticket details the requester isn't authorized to see |

# %% [markdown]
# ## 4. The explicit graph — state, nodes, conditional routing
#
# 📖 [Graph API — conditional branching](https://docs.langchain.com/oss/python/langgraph/use-graph-api#conditional-branching)
#
# Everything so far used `create_agent`, which builds a graph **for** you: a ReAct loop where the
# model decides what happens next. That is the right default. But the agenda's orchestration block
# asks for the layer underneath — *state, nodes, edges, routing, interrupts, persistence* — because
# some control flow should not be left to the model.
#
# ### Tool vs sub-agent vs graph node
#
# | Use a… | When | Who decides the next step | Example here |
# |---|---|---|---|
# | **Tool** | A single, well-defined capability | The model | `search_kb`, `query_ticket_db` |
# | **Sub-agent** | A specialist needing its own prompt, tools, and reasoning loop | The model (supervisor) | `ticket_sql_specialist` (§5) |
# | **Graph node** | A step that must happen, in order, every time | **You**, in code | `validate_sources` below |
#
# The deciding question is *who is allowed to skip this step.* A source-validation step that the
# model may skip when it is confident is not a control — it is a suggestion. Making it a node on
# the edge into `END` means every answer passes through it, on every path, whatever the model
# decided. That is why compliance-shaped requirements become nodes, not tool descriptions.
#
# ### The graph we are building
#
# ```mermaid
# graph TD
#     START([START]) --> T[triage<br/>classify + confidence]
#     T -->|category = access/account| A[access_review<br/>entitlement path]
#     T -->|category = knowledge| K[kb_lookup<br/>RAG]
#     T -->|anything else| D[db_lookup<br/>SQL]
#     A --> DR[draft_answer]
#     K --> DR
#     D --> DR
#     DR --> V[validate_sources<br/>every claim cited?]
#     V -->|confidence low OR ungrounded OR sensitive| H[human_review<br/>⏸ interrupt]
#     V -->|confident + grounded| E([END])
#     H --> E
# ```
#
# Two conditional edges, which is exactly what the agenda asks for: one routing **on issue type**,
# one routing **on confidence** (plus grounding and sensitivity).

# %%
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

# The graph's nodes call the same tools the agent used above, but directly — a node is plain
# Python, so it invokes a tool rather than asking a model to choose one.
from day1.src.ticket_agent_graph import search_kb, search_ticket_history

# The confidence floor for answering without a human. Below this the graph routes to review no
# matter how fluent the draft reads — a number you can tune and defend in an audit, rather than a
# judgement buried in a prompt.
CONFIDENCE_THRESHOLD = 0.70

SENSITIVE_CATEGORIES = {"access", "account", "security"}


class TicketState(TypedDict):
    """The graph's state. Every node reads and writes this one typed dict.

    Being explicit about state is the whole point: at any interrupt you can inspect exactly what the
    agent knew and why it routed the way it did — in Studio, in a trace, or in a test.
    """

    ticket: str
    category: str
    confidence: float
    evidence: str
    sources: list[str]
    draft: str
    grounded: bool
    route: str
    needs_human: bool
    decision: str


class Triage(BaseModel):
    """Structured triage result — the model's classification with a calibrated confidence."""

    category: Literal["access", "account", "hardware", "knowledge", "network", "security", "software"] = Field(
        description="The ticket's category."
    )
    confidence: float = Field(description="0.0-1.0 confidence in the category. Be honest; low is fine.")
    reason: str = Field(description="One sentence explaining the classification.")


def triage(state: TicketState) -> dict:
    """Classify the ticket and record a confidence. This is the only node that calls the model."""
    triager = model.with_structured_output(Triage)
    out = triager.invoke(
        "Classify this Acme IT support ticket into exactly one category and give a calibrated "
        "confidence.\n\nTicket: " + state["ticket"]
    )
    print(f"  triage      → {out.category} (confidence {out.confidence:.2f}) — {out.reason}")
    return {"category": out.category, "confidence": out.confidence}


def route_by_category(state: TicketState) -> Literal["access_review", "kb_lookup", "db_lookup"]:
    """Conditional edge #1 — route on ISSUE TYPE.

    Access and account tickets take the entitlement path, because they end in a permission
    decision rather than an answer. Knowledge requests go to RAG. Everything else queries the
    ticket database for prior art.
    """
    if state["category"] in {"access", "account"}:
        return "access_review"
    if state["category"] == "knowledge":
        return "kb_lookup"
    return "db_lookup"


def kb_lookup(state: TicketState) -> dict:
    """Retrieve from the knowledge base (semantic)."""
    hits = search_kb.invoke({"query": state["ticket"]})
    print("  kb_lookup   → knowledge base")
    return {"evidence": hits, "sources": ["knowledge base"], "route": "kb_lookup"}


def db_lookup(state: TicketState) -> dict:
    """Query the ticket database for similar prior tickets (structured)."""
    rows = search_ticket_history.invoke({"query": state["ticket"], "category": state["category"]})
    print("  db_lookup   → ticket database (SQL)")
    return {"evidence": rows, "sources": ["ticket database (SQL)"], "route": "db_lookup"}


def access_review(state: TicketState) -> dict:
    """The entitlement path: what has been granted before, and to whom."""
    rows = search_ticket_history.invoke({"query": state["ticket"], "category": state["category"]})
    print("  access_review → entitlement path (sensitive)")
    return {"evidence": rows, "sources": ["ticket database (SQL)"], "route": "access_review"}


def draft_answer(state: TicketState) -> dict:
    """Draft a response grounded in whatever the retrieval node found."""
    out = model.invoke(
        "You are an Acme IT support agent. Draft a short response to the ticket using ONLY the "
        "evidence below. Cite the source in square brackets. If the evidence does not answer it, "
        "say so plainly.\n\n"
        f"Ticket: {state['ticket']}\n\nEvidence:\n{state['evidence'][:2500]}"
    )
    return {"draft": out.content}


def validate_sources(state: TicketState) -> dict:
    """Source validation BEFORE the response goes out — a node, so it cannot be skipped.

    A cheap deterministic check: does the draft actually cite anything? Real deployments extend
    this to verify each citation resolves to a retrieved document. The important part is structural
    — it runs on every path into `END`.
    """
    grounded = "[" in state["draft"] and "]" in state["draft"]
    sensitive = state["category"] in SENSITIVE_CATEGORIES
    low_conf = state["confidence"] < CONFIDENCE_THRESHOLD
    needs_human = bool(low_conf or not grounded or sensitive)
    why = ", ".join(
        w for w, hit in (
            (f"confidence {state['confidence']:.2f} < {CONFIDENCE_THRESHOLD}", low_conf),
            ("draft cites no source", not grounded),
            (f"sensitive category '{state['category']}'", sensitive),
        ) if hit
    ) or "confident and grounded"
    print(f"  validate    → grounded={grounded} · needs_human={needs_human} ({why})")
    return {"grounded": grounded, "needs_human": needs_human, "decision": why}


def route_after_validation(state: TicketState) -> Literal["human_review", "__end__"]:
    """Conditional edge #2 — route on CONFIDENCE (and grounding, and sensitivity)."""
    return "human_review" if state["needs_human"] else END


def human_review(state: TicketState) -> dict:
    """Pause for a human. `interrupt()` suspends the graph and persists state via the checkpointer."""
    decision = interrupt({
        "action": "review_ticket_response",
        "category": state["category"],
        "confidence": state["confidence"],
        "why": state["decision"],
        "draft": state["draft"],
    })
    return {"decision": f"human: {decision}"}


ticket_graph = (
    StateGraph(TicketState)
    .add_node("triage", triage)
    .add_node("kb_lookup", kb_lookup)
    .add_node("db_lookup", db_lookup)
    .add_node("access_review", access_review)
    .add_node("draft_answer", draft_answer)
    .add_node("validate_sources", validate_sources)
    .add_node("human_review", human_review)
    .add_edge(START, "triage")
    # Conditional edge #1 — on issue type. The explicit target list is what lets LangGraph draw
    # the graph correctly (and lets Studio show you the branches before you run it).
    .add_conditional_edges("triage", route_by_category, ["access_review", "kb_lookup", "db_lookup"])
    .add_edge("kb_lookup", "draft_answer")
    .add_edge("db_lookup", "draft_answer")
    .add_edge("access_review", "draft_answer")
    .add_edge("draft_answer", "validate_sources")
    # Conditional edge #2 — on confidence / grounding / sensitivity.
    .add_conditional_edges("validate_sources", route_after_validation, ["human_review", END])
    .add_edge("human_review", END)
    # Persistence: required for interrupt() to be resumable. Same checkpointer idea as §2.
    .compile(checkpointer=MemorySaver())
)

display(Image(ticket_graph.get_graph().draw_mermaid_png()))

# %% [markdown]
# ### Run it — the knowledge path (no human needed)

# %%
cfg_a = {"configurable": {"thread_id": "graph-knowledge-1"}}
print("▶ knowledge request")
res_a = ticket_graph.invoke(
    {"ticket": "Where do I find the guide for setting up a new printer?"}, cfg_a
)
print(f"\nrouted via : {res_a['route']}\ndecision   : {res_a['decision']}")
print(f"draft      : {res_a['draft'][:200]}…")

# %% [markdown]
# ### Run it — the access path (always reviewed)
#
# Note this pauses **even if the model is confident**. `access` is a sensitive category, and that
# is a rule in the graph rather than a hope about the prompt.

# %%
cfg_b = {"configurable": {"thread_id": "graph-access-1"}}
print("▶ access request")
res_b = ticket_graph.invoke(
    {"ticket": "Please grant me write access to the procurement share drive. My user ID is USR-004."},
    cfg_b,
)

if "__interrupt__" in res_b:
    payload = res_b["__interrupt__"][0].value
    print(f"\n⏸️  paused for human review — {payload['why']}")
    print(f"    draft: {payload['draft'][:160]}…")
    resumed = ticket_graph.invoke(Command(resume="approved — entitlement matches the user's role"), cfg_b)
    print(f"▶️  resumed → {resumed['decision']}")
else:
    print("\n(no interrupt raised — inspect res_b['decision'])")

# %% [markdown]
# ### What the explicit graph bought us
#
# | Requirement | How the graph enforces it |
# |---|---|
# | Route on issue type | `add_conditional_edges("triage", route_by_category, [...])` |
# | Route on confidence | `add_conditional_edges("validate_sources", route_after_validation, [...])` |
# | Source validation before responding | `validate_sources` is a node on **every** path into `END` |
# | Human approval for sensitive actions | `interrupt()` inside `human_review` |
# | Persistence across the pause | `.compile(checkpointer=MemorySaver())` |
# | Inspectable state | `TicketState` — visible at every step in Studio and in traces |
#
# > **When to reach for this.** `create_agent` for open-ended reasoning; an explicit graph when a
# > step is mandatory, when routing must be auditable, or when a reviewer will ask *"can it skip
# > that?"* and "no" has to be a property of the code. Most production systems use both — an
# > explicit graph whose nodes are agents.
#
# ### Variants worth knowing — compare with `langgraph-101`
#
# [`music_store_supervisor_with_interrupt.py`](https://github.com/langchain-ai/langgraph-101/blob/main/agents/music_store/music_store_supervisor_with_interrupt.py)
# builds a graph with the same primitives but three different choices. None is more correct; they
# suit different jobs, and knowing which you want is the skill:
#
# | Choice | This lab | `langgraph-101` | When to prefer theirs |
# |---|---|---|---|
# | Conditional edge form | router returns the node name, targets as a **list** | router returns a label, mapped by **dict**: `{"continue": "supervisor", "interrupt": "human_input"}` | The label reads as intent (`"interrupt"`) rather than as a node name — clearer when routing logic is complex |
# | After the human replies | `human_review → END` — an **approval gate**; the decision ends the run | `human_input → verify_info` — a **retry loop**; it re-checks and can ask again | You are *gathering missing information* rather than authorising an action |
# | Node granularity | nodes are plain functions | `add_node("supervisor", supervisor)` — a **compiled agent as a node** | You want an explicit outer graph whose steps are themselves agents |
#
# That last one is the shape most production systems converge on: deterministic outer graph,
# agentic inner nodes. `create_agent` returns a compiled graph, so `add_node("triage", some_agent)`
# works directly — the subgraph shares the parent's state keys.
#
# They also pass `StateGraph(State, input_schema=InputState)` to constrain what callers may send.
# Worth doing once a graph is public API; omitted here so the state stays readable in one place.

# %% [markdown]
# ## 5. Sub-agents — a preview of Day 2
#
# 📖 [Multi-agent](https://docs.langchain.com/oss/python/langchain/multi-agent)
#
# So far we've built a **single agent** with multiple tools. The agent decides which tool to call
# based on the query. But what if different kinds of queries need different *expertise*?
#
# A **sub-agent** is a specialist agent with its own tools and system prompt. A **supervisor** agent
# routes queries to the right specialist. This is the pattern we'll use on Day 2 for the due diligence
# Deep Agent — but let's preview it here.
#
# **Why sub-agents?**
# - **Specialization:** Each sub-agent has a focused system prompt and a curated set of tools
# - **Context isolation:** The sub-agent's context stays clean — the supervisor only sees the final
#   response, not the intermediate tool calls
# - **Modularity:** Add, remove, or modify a sub-agent without changing the others
#
# **The pattern:** Wrap each sub-agent as a `@tool` so the supervisor can call it like any other tool.

# %%
from langchain_core.tools import tool

# Import the tools from the graph module (they're the same ones we used in Lab 02)
from day1.src.ticket_agent_graph import (
    get_user_context,
    mock_api_action,
    query_ticket_db,
    search_kb,
    search_ticket_history,
    ticket_statistics,
)
from day1.src.ticket_db import SCHEMA_DOC

# Optional live-web fallback for anything outside the bundled KB
extra_tools = [t for t in ticket_tools() if t.name not in
               {"search_kb", "search_ticket_history", "get_user_context", "mock_api_action"}]

# Build four specialist sub-agents
kb_agent = create_agent(
    model=model,
    tools=[search_kb, *extra_tools],
    name="kb_search_specialist",
    system_prompt=(
        "You are the knowledge base search specialist for Acme IT support. "
        "You answer how-to questions and troubleshooting queries by searching the IT support "
        "knowledge base (MFA reset guides, VPN troubleshooting, software installation, etc.), "
        "falling back to web search when the bundled KB doesn't cover the issue. "
        "You do NOT search ticket history, look up user context, or execute account actions. "
        "Always cite the KB article filename (or web source) in your answers. "
        "If no KB article covers the issue, say so clearly."
    ),
)

history_agent = create_agent(
    model=model,
    # BEGINNER TIER — fixed, pre-written query tools. Predictable and impossible to misuse.
    tools=[search_ticket_history, ticket_statistics, get_user_context],
    name="ticket_history_specialist",
    system_prompt=(
        "You are the ticket history specialist for Acme IT support. "
        "You find similar past tickets and their resolutions, and look up the requesting user's "
        "role and permissions to verify what they are authorized to access. "
        "You do NOT search the knowledge base or execute account actions. "
        "Always cite the ticket ID and source (ticket history or user record) in your answers. "
        "If a user is not authorized for a category, flag it instead of returning the details."
    ),
)

# ADVANCED TIER — the SQL specialist. The schema goes in the SYSTEM PROMPT (not just the tool
# docstring), which is how `agents/sql_agent.py` in the reference workshop does it: the model needs
# the table structure in front of it while it is *planning*, not only when it is calling.
#
# This sub-agent exists because the fixed tools have a measured limit (Lab 02 §2 demonstrates it):
# they cannot express a join across `tickets` and `users`, or a filtered aggregate. That is the
# eval-driven path — measure the limitation, then add capability to close it.
#
# That is not our framing; it is the reference implementation's own. From the module docstring of
# [`agents/sql_agent.py`](https://github.com/langchain-ai/langsmith-agent-lifecycle-workshop/blob/main/agents/sql_agent.py),
# verbatim:
#
# > *"This agent demonstrates eval-driven development: baseline evaluation revealed that rigid
# > tools couldn't handle complex queries, so we built a flexible SQL agent that generates queries
# > on-demand to handle any database question."*
#
# Three things there we deliberately match:
#
# | Their choice | Ours |
# |---|---|
# | Schema from `db.get_table_info()` embedded in the **system prompt at creation time** | `SCHEMA_DOC` in this agent's system prompt |
# | One tool: `SQL_AGENT_BASE_TOOLS = [execute_sql]` | one tool: `query_ticket_db` |
# | `db_agent` and `sql_agent` are **identical factories** — same params, same `state_schema`, same
#   checkpointer rule — differing only in `tools` and `system_prompt` | `ticket_history_specialist` (fixed tools) and `ticket_sql_specialist` (SQL) are sibling sub-agents with the same role |
# | `use_checkpointer=True` for dev, `False` for deployment — "platform handles it" | same rule; the deployed graphs in `ticket_agent_graph.py` carry no checkpointer |
#
# **Why the two specialists are deliberately parallel.** In the reference, `db_agent.py` and
# `sql_agent.py` are the *same* factory — same signature, same `state_schema`, same "checkpointer
# for dev, none for deployment" rule — and differ **only** in their tools
# (`[get_order_status, get_order_items, get_product_info, get_order_item_price]` vs
# `[execute_sql]`) and their prompt. That is what makes the comparison a measurement rather than an
# anecdote: swap one for the other and an eval isolates the *tool strategy*, because nothing else
# changed. Keep that discipline when you A/B your own agents — a comparison in which two things
# differ tells you nothing about either.
#
# Their fixed-tool prompt is three anti-hallucination rules worth copying almost verbatim:
# *"Always retrieve answers directly from the database using the available tools"*, *"If information
# is missing or not found, say so clearly"*, and *"Do NOT make assumptions or provide information
# not explicitly present in the database."*
#
# Where we go further: their `execute_sql` guards with a `startswith("SELECT")` check and a
# forbidden-keyword list only. Ours adds multi-statement rejection, parameter binding, a row cap,
# and an engine-level read-only connection — following LangChain's security guidance to *"consider
# issuing READ-ONLY credentials"* and to *"combine multiple layered security approaches"*.
sql_agent = create_agent(
    model=model,
    tools=[query_ticket_db],
    name="ticket_sql_specialist",
    system_prompt=(
        "You are the ticket database SQL specialist for Acme IT support. You answer questions from "
        "the orchestrator by writing read-only SQL against the ticket database.\n\n"
        "Schema:\n" + SCHEMA_DOC + "\n"
        "Rules:\n"
        "- Write a single SELECT (or WITH ... SELECT). Writes are rejected by the database itself.\n"
        "- Prefer an explicit JOIN over guessing; prefer an aggregate over returning many rows.\n"
        "- If the query errors, read the error, fix the SQL, and retry once.\n"
        "- Report the number the query returned. Never estimate, and never fill a gap from memory.\n"
        "- State the SQL you ran, then the answer, then '[source: ticket database (SQL)]'."
    ),
)

action_agent = create_agent(
    model=model,
    tools=[mock_api_action],
    name="api_action_specialist",
    system_prompt=(
        "You are the API action specialist for Acme IT support. "
        "You prepare sensitive account actions — MFA reset, account unlock, password reset — as "
        "mock API calls. These actions are always gated by human approval before they execute, "
        "so state the action, target user, and reason precisely. "
        "You do NOT search the knowledge base or ticket history."
    ),
)

print("Sub-agents created:")
print(f"  {kb_agent.name}: kb_search_specialist")
print(f"  {history_agent.name}: ticket_history_specialist (fixed tools)")
print(f"  {sql_agent.name}: ticket_sql_specialist (model-authored SQL)")
print(f"  {action_agent.name}: api_action_specialist")

# %% [markdown]
# ### Wrap the sub-agents as tools
#
# The key insight: **sub-agents become tools for the supervisor**. We wrap each sub-agent's `invoke()`
# in a `@tool` function with a clear description so the supervisor knows when to call it.

# %%
@tool("kb_search_specialist",
      description="Query the knowledge base search specialist for IT support articles, how-to "
                  "guides, and troubleshooting steps (MFA reset, VPN, software install, lockout). "
                  "Use for questions about procedures, error messages, and setup guides.")
def call_kb_specialist(query: str) -> str:
    """Call the KB search specialist subagent."""
    result = kb_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


@tool("ticket_history_specialist",
      description="Query the ticket history specialist for similar past tickets, their resolutions, "
                  "and the requesting user's role and permissions. "
                  "Use for questions about ticket IDs, past resolutions, and user authorization.")
def call_history_specialist(query: str) -> str:
    """Call the ticket history specialist subagent."""
    result = history_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


@tool("ticket_sql_specialist",
      description="Query the ticket database SQL specialist for anything the fixed ticket tools "
                  "cannot express: joins across tickets and users, filtered aggregates, counts by "
                  "department, mean time to resolution, or any 'how many / what share / compare' "
                  "question. It writes read-only SQL and reports the exact number.")
def call_sql_specialist(query: str) -> str:
    """Call the SQL specialist subagent."""
    result = sql_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


@tool("api_action_specialist",
      description="Query the API action specialist to execute a sensitive account action "
                  "(mfa_reset, account_unlock, password_reset) against the mock identity API. "
                  "Use only after KB steps have failed or the issue clearly requires the action. "
                  "Always human-approved before execution.")
def call_action_specialist(query: str) -> str:
    """Call the API action specialist subagent."""
    result = action_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


# %% [markdown]
# ### Build the supervisor
#
# The supervisor is a plain `create_agent` with the wrapped sub-agent tools. Its system prompt tells
# it when to delegate to each specialist.

# %%
supervisor = create_agent(
    model=model,
    tools=[call_kb_specialist, call_history_specialist, call_sql_specialist,
           call_action_specialist],
    name="ticket_orchestrator",
    system_prompt=(
        "You are the ticket orchestrator for Acme IT support. You coordinate four specialist "
        "sub-agents to resolve support tickets.\n\n"
        "Routing rules:\n"
        "- Use kb_search_specialist for how-to questions, troubleshooting steps, error messages, "
        "  and setup guides (MFA reset, VPN, software install, lockout).\n"
        "- Use ticket_history_specialist to find similar past tickets and resolutions, and to "
        "  verify the requesting user's role and permissions. It uses fixed query tools, so prefer "
        "  it for straightforward lookups — it is cheaper and more predictable.\n"
        "- Use ticket_sql_specialist when the question needs a join or an aggregate the fixed "
        "  tools cannot express: counts by department, share unresolved, mean time to resolution, "
        "  or any 'how many / what share / compare across' question. It writes read-only SQL.\n"
        "- Use api_action_specialist for sensitive account actions (MFA reset, account unlock, "
        "  password reset) — only after KB steps have failed or the issue clearly requires it. "
        "  These actions always require human approval.\n"
        "- You can call multiple tools for a single query if needed.\n\n"
        "Always synthesize the specialists' responses into a coherent answer. "
        "If a query doesn't need a specialist, answer directly."
    ),
)

# %% [markdown]
# **The supervisor's shape** — the supervisor's graph shows the tool nodes for each sub-agent.

# %%
display(Image(supervisor.get_graph().draw_mermaid_png()))

# %% [markdown]
# ### Test the supervisor
#
# **Query 1** — KB-only question (should route to the KB search specialist):

# %%
from utils.trace import ask

print(ask(supervisor, "How do I troubleshoot a VPN connection that drops after 2 minutes?"))

# %% [markdown]
# **Query 2** — history-only question (should route to the ticket history specialist):

# %%
print(ask(supervisor, "Have we resolved MFA issues for users who replaced their phones before? What fixed them?"))

# %% [markdown]
# **Query 3** — multi-specialist question (should route to more than one specialist):

# %%
print(ask(supervisor,
    "Ingrid Johansson (USR-003) is locked out after replacing her phone. Check what the KB says about "
    "MFA resets, find similar past tickets, verify her permissions, and recommend next steps."))

# %% [markdown]
# ### How this maps to Day 2
#
# The pattern shown here — supervisor + sub-agents as tools — is the foundation for Day 2's
# **Deep Agent** harness. Day 2 adds:
# - `create_deep_agent` with built-in planning, filesystem, and sub-agent middleware
# - `AGENTS.md` memory for always-relevant due diligence instructions
# - `SKILL.md` / skills directory for reusable capabilities
# - RubricMiddleware for runtime self-evaluation
# - Wiki memory for durable, source-grounded notes — hand-rolled in Day 2 Lab 02 via `AGENTS.md` +
#   filesystem tools. [OpenWiki](https://docs.langchain.com/oss/openwiki/overview) automates the
#   *writing* of a wiki like this for codebase documentation; a consuming agent reads either kind
#   the same way — through `AGENTS.md` and filesystem tools, not a special backend
#
# The supervisor pattern you just built is the manual version of what `create_deep_agent` automates.

# %% [markdown]
# ## 6. MCP — reaching a system you do not own
#
# 📖 [MCP with LangChain](https://docs.langchain.com/oss/python/langchain/mcp) · [reference](https://github.com/langchain-ai/langchain-mcp-adapters)
#
# Every tool so far has been a Python function in this process. Real Acme systems are not: the ticket
# platform, the identity provider, SAP, a SQL database — each is owned by a different team, on a
# different release cycle, behind an interface you do not control.
#
# **[MCP](https://modelcontextprotocol.io) (Model Context Protocol)** is the standard for that
# boundary. The system's owners publish a server exposing their operations as tools; your agent
# discovers and calls them at runtime. Neither side needs to know the other's internals.
#
# ```mermaid
# graph LR
#     A[Agent process] -->|stdio / HTTP| S[MCP server<br/>ticket system]
#     S --> T1[get_ticket]
#     S --> T2[route_to_queue]
#     S --> T3[get_sla]
#     S --> T4[update_ticket_status ⚠]
#     S --> T5[whoami]
#     A -.same create_agent call.-> L[local tools<br/>search_kb, …]
# ```
#
# This workshop ships a **custom MCP server** for the ticket system:
# [`day1/src/ticket_mcp_server.py`](ticket_mcp_server.py). It exposes the operations that belong to
# the *platform* rather than the agent:
#
# | MCP tool | Why it lives in the ticket system, not the agent |
# |---|---|
# | `get_ticket` | the record is the platform's, not a copy in your prompt |
# | `route_to_queue` | the routing table changes without redeploying the agent |
# | `get_sla` | SLA policy is owned by service management |
# | `update_ticket_status` | **a write to an external system** |
# | `whoami` | the directory is the authority on roles, not your JSON file |
#
# > 💡 **The key insight: nothing about `create_agent` changes.**
# > `MultiServerMCPClient.get_tools()` returns ordinary LangChain tools, so a remote tool and a local
# > one are indistinguishable to the agent. That is what makes MCP adoptable incrementally — you can
# > move one tool behind a protocol without touching the agent loop.
#
# > ⚠️ **Running this section in a notebook?** MCP is async, and this file is written to also run as
# > a plain script (`asyncio.run(...)`) — but Jupyter already has its own event loop running, so
# > `asyncio.run(...)` raises `RuntimeError: asyncio.run() cannot be called from a running event
# > loop`. **Fix: in the notebook, drop the `asyncio.run(...)` wrapper and just `await` the call
# > directly** — e.g. `mcp_tools = asyncio.run(discover())` becomes `mcp_tools = await discover()`.
# > Same swap applies to every `asyncio.run(...)` call in this section (4 total).

# %%
# The client spawns the server as a subprocess and speaks stdio — no port, no auth, no network.
# `transport="stdio"` is the right default for a workshop and for a locally-installed integration;
# use "streamable_http" when the server is a service someone else runs.
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient

MCP_SERVER = WORKSHOP / "day1" / "src" / "ticket_mcp_server.py"

mcp_client = MultiServerMCPClient({
    "ticket-system": {
        "command": sys.executable,          # the same interpreter running this notebook
        "args": [str(MCP_SERVER)],
        "transport": "stdio",
    }
})


async def discover() -> list:
    """MCP is async. In a notebook you can `await` directly; as a script, wrap in asyncio.run."""
    return await mcp_client.get_tools()


mcp_tools = asyncio.run(discover())
print(f"discovered {len(mcp_tools)} tools over MCP from 'ticket-system':")
for t in mcp_tools:
    print(f"  {t.name:22} {t.description.splitlines()[0][:78]}")

# %% [markdown]
# **They are LangChain tools.** Same `.name`, `.description`, `.args_schema` — so the same agent
# constructor accepts them. Call one directly to see the protocol round-trip:

# %%
_by_name = {t.name: t for t in mcp_tools}


def mcp_text(result) -> str:
    """MCP returns a list of typed content blocks, not a string. Pull the text out."""
    if isinstance(result, list):
        return "\n".join(b.get("text", "") for b in result if isinstance(b, dict))
    return str(result)


print(mcp_text(asyncio.run(_by_name["route_to_queue"].ainvoke({"category": "security"}))))
print()
print(mcp_text(asyncio.run(_by_name["get_sla"].ainvoke({"priority": "P1"}))))

# %% [markdown]
# ### An agent over local **and** MCP tools
#
# Now the point of the whole exercise: one agent, tools from two places, and a **HITL gate on the MCP
# write**. `update_ticket_status` changes state in a system we do not own — and the server cannot know
# whether a human approved, so the gate has to live here, on the agent side.

# %%
mcp_agent = create_agent(
    model=model,
    # local tools + MCP tools in one list — the agent cannot tell them apart
    tools=[search_kb, *mcp_tools],
    system_prompt=(
        "You are the Acme ticket triage agent.\n"
        "- Use the knowledge base (search_kb) for troubleshooting procedure.\n"
        "- Use the ticket-system tools for anything the platform owns: the ticket record "
        "(get_ticket), the owning queue (route_to_queue), SLA targets (get_sla), and who a user is "
        "(whoami). Never guess a queue name, an SLA, or someone's permissions.\n"
        "- update_ticket_status writes to the ticket system and requires human approval.\n"
        "Cite the source tag returned by each tool."
    ),
    checkpointer=MemorySaver(),
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"update_ticket_status": True})],
)

mcp_thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

# TKT-021 is a P1 that is still open (status: escalated in the fixture, unresolved). Pointing this
# at an already-resolved ticket makes the model reasonably ask "are you sure?" instead of acting —
# and then the approval gate has nothing to gate, which defeats the demonstration.
QUERY = ("Ticket TKT-024 is an open P1 security report. Look it up, tell me which queue owns it and "
         "what the SLA target is, then set its status to escalated with a note. Perform the status "
         "change — the approval gate will hold it for a human.")


async def run_with_approval() -> dict:
    """Drive the agent with `ainvoke`, approving any MCP write.

    **MCP tools are async-only.** Calling `mcp_agent.invoke(...)` raises
    `NotImplementedError: StructuredTool does not support sync invocation` the moment the agent
    reaches an MCP tool — the adapter wraps an async client session, so there is no sync path. Use
    `ainvoke` (and `asyncio.run` here, so the file still runs as a plain script; a notebook cell can
    `await` directly).
    """
    out = await mcp_agent.ainvoke({"messages": [{"role": "user", "content": QUERY}]},
                                  config=mcp_thread)
    while out.get("__interrupt__"):
        pending = out["__interrupt__"][0].value["action_requests"]
        print(f"⏸️  paused — {len(pending)} write(s) to the ticket system awaiting approval:")
        for req in pending:
            print(f"    {req['name']} | args: {req['args']}")
        out = await mcp_agent.ainvoke(
            Command(resume={"decisions": [{"type": "approve"} for _ in pending]}),
            config=mcp_thread,
        )
        print("✅ approved → resuming\n")
    return out


result = asyncio.run(run_with_approval())
print(result["messages"][-1].content[:700])

# %% [markdown]
# ### What to take away
#
# | Observation | Why it matters at Acme |
# |---|---|
# | The agent code is unchanged | you can put one existing tool behind MCP without rewriting the agent |
# | Tool descriptions come **from the server** | the system's owners control how their operations are described to every agent, in one place |
# | The write was gated here, not there | an MCP server cannot know whether a human approved — authorization and approval stay yours |
# | **MCP tools are async-only** | `agent.ainvoke(...)`, not `invoke` — a sync call raises `NotImplementedError` |
# | A new server process spawns per call | fine for stdio/local; for a hot path hold a session (`async with mcp_client.session("ticket-system")`) or use `streamable_http` |
#
# > ⚠️ **Trust boundary.** An MCP server supplies *tool descriptions* — text that goes into your
# > prompt. A malicious or compromised server can therefore attempt prompt injection through a tool
# > description. Treat a third-party MCP server the way you would any untrusted input: pin the
# > version, review what it exposes, and keep the guardrails from Lab 05 (`PIIMiddleware`, the
# > injection check, `ToolCallLimitMiddleware`) in the stack.

# %% [markdown]
# ## 7. Advanced exercise 1 — extend the orchestrator
#
# **Format:** ~20 minutes on your own, then a 10-minute walkthrough.
#
# The supervisor in §4 routes by prompt alone: it reads the query and picks a specialist. That is
# flexible but opaque — you cannot unit-test a prompt, and an access-control decision is exactly the
# kind of thing you want to be able to test. This exercise adds an **explicit routing function** in
# front of the supervisor, plus one new specialist.
#
# **Your task:** finish `route_ticket()` below. The action and history branches are written; the
# **access-review branch is missing**. An access-review ticket is one asking for permission to a
# system, folder, or facility — those must go to a dedicated `access_review_specialist` that always
# checks the requester's role before recommending anything.
#
# **Other options** (pick one instead if you prefer — each is a self-contained ~20 min change):
#
# | Option | What to change | Where |
# |---|---|---|
# | A. Access-review branch *(the starter below)* | Add the missing branch + an access specialist | this section |
# | B. Confidence-gated HITL | Only interrupt when the agent's confidence is below `medium` | §3 + `TicketResolution.confidence` |
# | C. MCP-style tool wrapper | Wrap `mock_api_action` behind a uniform `call_tool(server, tool, args)` façade | §4 |
# | D. Richer output schema | Add `priority` and `assigned_team` to `TicketResolution` and make the supervisor fill them | `ticket_agent_graph.py` |
#
# **Done when:** the self-check cell prints all ✅, and the supervisor routes an access request to
# the new specialist.

# %%
# --- STARTER -------------------------------------------------------------------------------
# Keyword sets are deliberately simple: routing should be legible and testable. In production you
# would classify with a small model or a trained classifier — the *shape* stays the same.

ACTION_KEYWORDS = {"reset", "unlock", "re-enable", "reactivate", "locked out"}
HISTORY_KEYWORDS = {"before", "previously", "past ticket", "similar", "recurring", "again"}
ACCESS_KEYWORDS = {"access to", "permission", "authorized", "authorisation", "authorization",
                   "grant me", "add me to", "share drive", "clearance"}


def route_ticket(ticket_text: str) -> list[str]:
    """Return the specialists that should handle this ticket, in call order.

    KB search is the default: every ticket gets a documented procedure. The other branches add
    specialists on top when the text calls for them.
    """
    text = ticket_text.lower()
    route: list[str] = ["kb_search_specialist"]

    if any(k in text for k in HISTORY_KEYWORDS):
        route.append("ticket_history_specialist")

    if any(k in text for k in ACTION_KEYWORDS):
        route.append("api_action_specialist")

    # TODO(exercise): add the access-review branch.
    #   When the ticket matches ACCESS_KEYWORDS, it must be handled by "access_review_specialist"
    #   INSTEAD of api_action_specialist — an access grant is a permission decision, not an
    #   account-recovery action, and it must never be auto-executed.
    #   Hint: append the specialist, and remove "api_action_specialist" if the starter added it.

    return route


# %%
# --- SELF-CHECK: run this to see where you are ---------------------------------------------
ROUTING_CASES = [
    ("How do I set up the VPN from home?",
     ["kb_search_specialist"]),
    ("My MFA token is dead and I'm locked out — please reset it.",
     ["kb_search_specialist", "api_action_specialist"]),
    ("Have we seen this printer error before?",
     ["kb_search_specialist", "ticket_history_specialist"]),
    ("I need access to the procurement share drive for the Quelmore file.",
     ["kb_search_specialist", "access_review_specialist"]),
    ("I was locked out again and I also need permission for the logistics folder.",
     ["kb_search_specialist", "ticket_history_specialist", "access_review_specialist"]),
]


def check_routing(fn) -> bool:
    ok = True
    for text, expected in ROUTING_CASES:
        actual = fn(text)
        passed = actual == expected
        ok = ok and passed
        print(f"{'✅' if passed else '❌'} {text[:52]:54} → {actual}")
        if not passed:
            print(f"   expected: {expected}")
    return ok


print("all cases pass:", check_routing(route_ticket))

# %% [markdown]
# ### Solution walkthrough
#
# Two things make the access branch different from the others:
#
# 1. **It is exclusive.** An access request is a *permission decision*, so it replaces
#    `api_action_specialist` rather than stacking with it. Ordering the branches so the access check
#    runs last — and letting it remove the action specialist — keeps that rule in one place.
# 2. **The specialist is narrow.** It gets `get_user_context` first, so it cannot recommend a grant
#    without having read the requester's role, and it has no action tool at all: the most it can do
#    is *recommend*, which is the point.

# %%
def route_ticket_solution(ticket_text: str) -> list[str]:
    """Reference solution — the access branch is exclusive with the action branch."""
    text = ticket_text.lower()
    route: list[str] = ["kb_search_specialist"]

    if any(k in text for k in HISTORY_KEYWORDS):
        route.append("ticket_history_specialist")

    if any(k in text for k in ACTION_KEYWORDS):
        route.append("api_action_specialist")

    if any(k in text for k in ACCESS_KEYWORDS):
        # A permission decision, not an account recovery — never auto-executed.
        route = [r for r in route if r != "api_action_specialist"]
        route.append("access_review_specialist")

    return route


print("all cases pass:", check_routing(route_ticket_solution))

# %% [markdown]
# **The new specialist.** Read-only by construction: user context + KB + history, no action tool.

# %%
access_agent = create_agent(
    model=model,
    tools=[get_user_context, search_kb, search_ticket_history],
    name="access_review_specialist",
    system_prompt=(
        "You are the access review specialist for Acme IT support. You handle requests for access "
        "to systems, folders, share drives, and facilities.\n"
        "Always call get_user_context FIRST and state the requester's role and permitted categories "
        "before anything else. Then cite the access-request procedure from the knowledge base.\n"
        "You may only RECOMMEND — approve, deny, or escalate — and you must say which. You have no "
        "tool that changes access, and access changes always require a human approver. "
        "If the requester's role does not already cover the category they are asking about, "
        "recommend escalation to their line manager and name the approval step."
    ),
)


@tool("access_review_specialist",
      description="Query the access review specialist for requests to grant access to a system, "
                  "folder, share drive, or facility. It verifies the requester's role and "
                  "permissions and recommends approve / deny / escalate. It cannot change access.")
def call_access_specialist(query: str) -> str:
    """Call the access review specialist subagent."""
    result = access_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content


# %% [markdown]
# **Wire the routing function to the supervisor.** The router decides *who* runs; the supervisor
# still decides *what to ask them*. Passing the route in as a hint keeps the graph unchanged — and
# because the route is a plain function, the decision is now unit-tested by the cell above.

# %%
routed_supervisor = create_agent(
    model=model,
    tools=[call_kb_specialist, call_history_specialist, call_action_specialist,
           call_access_specialist],
    name="ticket_orchestrator_routed",
    system_prompt=(
        "You are the ticket orchestrator for Acme IT support.\n"
        "Each request arrives with a ROUTE line naming the specialists selected for it by the "
        "routing policy. Call exactly those specialists, in the order given, then synthesize one "
        "coherent answer.\n"
        "Never call api_action_specialist for a request that asks for new access — access grants "
        "are permission decisions and go to access_review_specialist.\n"
        "Cite the sources your specialists returned."
    ),
)


def resolve(ticket_text: str) -> str:
    """Route, then ask the supervisor to run that route."""
    route = route_ticket_solution(ticket_text)
    print("route:", " → ".join(route))
    return ask(routed_supervisor, f"ROUTE: {', '.join(route)}\n\nTICKET: {ticket_text}")


# %%
print(resolve("Lukas Meyer (USR-004) needs access to the procurement share drive to review the "
              "Quelmore capability statement. Can you sort that out?"))

# %% [markdown]
# ## 8. LangGraph Studio
#
# 📖 [Studio quick start](https://docs.langchain.com/langsmith/quick-start-studio)
#
# The deployable graph is registered in `langgraph.json`. Open it in Studio to:
# - Visualize the graph structure
# - Inspect state transitions during a run
# - See tool calls and their results
# - Trigger interrupts and resume manually
#
# ```bash
# # From the langchain_adlc_workshop/day1/ directory:
# uv run langgraph dev     # serves at http://127.0.0.1:2024
# ```
#
# Studio: **https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024**
#
# | Graph | Try in Studio |
# |---|---|
# | `ticket_agent` | *"How do I set up the VPN for remote work from home?"* |
# | `ticket_agent_hitl` | *"Reset MFA for Ingrid Johansson (USR-003)."* → pauses before `mock_api_action`; click **Approve** |

# %% [markdown]
# ## 9. Recap & next
#
# | Topic | API |
# |---|---|
# | Memory across turns | `create_agent(…, checkpointer=MemorySaver())` + `config={"configurable": {"thread_id": …}}` |
# | Pause for approval | `create_agent(…, middleware=[HumanInTheLoopMiddleware(interrupt_on={...})])` |
# | Resume an interrupt | `agent.invoke(Command(resume={"decisions": [...]}), config=...)` — one decision per pending action |
# | Studio / deploy | `ticket_agent_graph.py` + `langgraph.json` → `uv run langgraph dev` |
#
# **Next:** `04_traces_prompt_hub.ipynb` — inspect traces, identify quality issues, improve with
# Prompt Hub.
