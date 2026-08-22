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
# # 03 · Online Evals + Production Feedback — Continuous Evaluation of the Deployed Ticket Agent
#
# **Workshop:** LangChain ADLC Workshop · **Day 3** · **ADLC stage:** Verify / Evaluate — in production
#
# > **Loop Engineering focus: Verification loop — running continuously.** Day 2 Labs 03–04 built the
# > *offline* verification loop: dataset → evaluators → experiment, run by hand against a frozen eval
# > set. This lab attaches that loop to the **deployed ticket agent from Lab 01** so every production
# > run is sampled, scored, and — when it matters — routed to a human reviewer, without anyone
# > pressing "run experiment".
#
# > Hands-on module · ~45 min
#
# ```mermaid
# graph LR
#     A[Deployed ticket agent<br/>Lab 01 · production runs] --> B[Online evaluators<br/>sampled % of runs]
#     B --> C1[groundedness]
#     B --> C2[policy_compliance]
#     B --> C3[escalation_correctness]
#     B --> C4[action_quality]
#     C1 --> D[Feedback scores<br/>on the runs]
#     C2 --> D
#     C3 --> D
#     C4 --> D
#     D --> E{Automation rule}
#     E -->|low confidence| F[Annotation queue<br/>human review]
#     E -->|policy-sensitive action| F
#     E -->|failed eval| F
#     E -->|clean| G[Monitoring dashboard]
#     F -->|corrected examples| H[Eval dataset<br/>next offline experiment]
#     H -->|improvement turn| I[Prompt / tools / schema change<br/>→ redeploy]
#     I --> A
# ```
#
# **Offline vs online evals, in one line:** offline evals ask *"did my change make things better?"*
# on a curated dataset before you ship; online evals ask *"is the deployed agent behaving?"* on real
# production traffic you never curated. You need both: the offline loop gates releases, the online
# loop watches reality — and feeds real failures back into the eval set.
#
# By the end you can:
# - Attach **online evaluators** to the deployed ticket agent's tracing project (code + LLM-as-judge,
#   with sampling rates) via the LangSmith `Client`
# - **Sample production traces** and inspect the feedback online evaluators leave on them
# - Score production outputs for **groundedness, policy compliance, escalation correctness, and
#   action quality**
# - Implement an **automation pattern** that routes low-confidence, policy-sensitive, or failed-eval
#   runs to an **annotation queue / human review**
# - Review how online evals appear in **LangSmith** (evaluator dashboards, per-run feedback,
#   monitoring charts)
# - Explain how **production signals feed the improvement loop** — the data flywheel back to
#   Day 2's offline experiments
#
# > 🧭 **Builds on Lab 01; runs standalone.** Section 3 (attach online evaluators) needs the
# > Lab 01 deployment's tracing project name. Everything else works with the bundled simulated
# > production runs, so the full loop can be built and smoke-tested before a single real ticket
# > hits the deployment. All ticket data is **synthetic/fictional**.
#
# > 📚 Reference: [Online evaluations docs](https://docs.langchain.com/langsmith/online-evaluations),
# > [annotation queues](https://docs.langchain.com/langsmith/annotation-queues),
# > [automation rules](https://docs.langchain.com/langsmith/rules).


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

# %%
import json
import os
import warnings
import sys
from datetime import datetime, timedelta, timezone
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

# `Client.list_runs` is deprecated in favour of `client.runs.query()` (removal after Jan 2027). We
# stay on `list_runs` deliberately: on this SDK version `client.runs.query` is **async-only**, takes
# project UUIDs rather than names, paginates by cursor, and requires self-hosted LangSmith >= v0.16 —
# a floor that matters for the air-gapped deployments discussed on Day 3. Silence the notice so the
# lab output stays readable; revisit when a sync surface lands.
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message=r".*list_runs\(\) is deprecated.*")

from langsmith import Client

ls_client = Client()

print("workspace:", os.getenv("LANGSMITH_WORKSPACE_ID"), "| tracing:", os.getenv("LANGSMITH_TRACING"))

# %% [markdown]
# ## 1. The production surface — what we're evaluating
#
# The subject is the **ticket resolution agent** from Day 3 Lab 01 (`01_deployments.py`), deployed
# from the Day 1 graph (`day1/src/ticket_agent_graph.py`). Its operating contract — from
# `day1/data/prompt_ticket.md` — is what the online evaluators below actually check:
#
# | Contract rule | Online evaluator that watches it |
# |---|---|
# | *"Always cite your sources (KB article filenames or ticket IDs)"* · *"Never invent ticket data or KB articles"* | `groundedness` |
# | *"Pass user identity into every retrieval call — only return information the user is authorized to see"* · no unauthorized account actions | `policy_compliance` |
# | *"Escalate to human review when: confidence is low · the action is sensitive (MFA reset, account unlock) · the user asks for something outside their permissions · the issue is a security incident"* | `escalation_correctness` |
# | *"Recommend actions"* — right next step, and sensitive actions (`mfa_reset`, `account_unlock`, `password_reset`) go through `mock_api_action` under human approval | `action_quality` |
#
# The agent's structured output (`TicketResolution`) makes online scoring cheap and deterministic:
# every run's output carries `category`, `kb_references`, `similar_tickets`, `recommended_action`,
# `confidence` (`high`/`medium`/`low`), `requires_hitl`, `missing_info`, and `user_id`. Code
# evaluators read those fields directly; LLM judges grade the free-text parts against the trace.
#
# **Where the scores attach:** online evaluators are configured per **tracing project**. Lab 01's
# deployment traces to `LANGSMITH_PROJECT` (default `acme-ticket-deployment` below). Every root run
# the deployment logs is a candidate for scoring; the sampling rate decides how many actually get it.
#
# Two knobs you always set on an online evaluator:
#
# | Knob | What it controls | Workshop setting |
# |---|---|---|
# | **Sampling rate** | Fraction of production runs scored (0.0–1.0) | 10% for cheap judges, 100% for the free code checks — set on the rule in the LangSmith UI |
# | **Evaluator type** | `code` (deterministic function, free) vs `llm` (judge prompt + model, costs per run) | `policy_compliance` as **code**; the three judgement calls as **LLM** |

# %%
# The tracing project the Lab 01 deployment writes to. `LANGSMITH_PROJECT` is set in .env for the
# labs, so consulting it here would always win and point every UI instruction below at the shared
# lab project — read the deployment project instead, matching Lab 01 §6 and Lab 03.
PROD_PROJECT = (
    os.getenv("PROD_PROJECT")
    or os.getenv("LANGSMITH_DEPLOYMENT_PROJECT", "acme-ticket-deployment")
)

print("production tracing project:", PROD_PROJECT)
print("view:", f"LangSmith → Observability → {PROD_PROJECT}")

# %% [markdown]
# ## 2. Simulated production traffic
#
# To build and test the whole loop *before* real tickets flow, this lab ships three **synthetic
# production runs** — shaped like what the deployed structured ticket agent returns, one per
# production archetype you'll actually see:
#
# | Run | Archetype | What should happen |
# |---|---|---|
# | `T-9001` MFA reset | Sensitive action, correctly escalated | Passes all evals — the healthy path |
# | `T-9002` account unlock | Sensitive action executed with **low confidence, no escalation** | Fails `escalation_correctness` + `action_quality` → routed to review |
# | `T-9003` VPN how-to | Clean knowledge answer, **no citations** | Fails `groundedness` → routed to review |
#
# > 🔌 **Swap-in point.** When Lab 01's deployment is live, replace `PROD_RUNS` with real sampled
# > runs from `ls_client.list_runs(project_name=PROD_PROJECT, is_root=True)` — the evaluators and
# > the automation in Sections 4–6 work unchanged, because they score the *output dict*, not where
# > it came from.

# %%
PROD_RUNS = [
    {
        "ticket_id": "T-9001",
        "inputs": {"question": "User USR-004 lost their phone — needs an MFA reset to get back in."},
        "output": {
            "issue_summary": "USR-004 lost their phone and is locked out of MFA; needs an MFA reset.",
            "category": "account",
            "kb_references": ["mfa_reset.md"],
            "similar_tickets": ["T-0108"],
            "recommended_action": "Perform mfa_reset for USR-004 via the identity API, pending human approval.",
            "confidence": "high",
            "requires_hitl": True,
            "missing_info": [],
            "user_id": "USR-004",
        },
        "trajectory": [
            {"name": "get_user_context", "args": {"user_id": "USR-004"}},
            {"name": "search_kb", "args": {"query": "MFA reset lost phone"}},
            {"name": "search_ticket_history", "args": {"query": "mfa reset", "category": "account"}},
            {"name": "mock_api_action", "args": {"action": "mfa_reset", "user_id": "USR-004",
                                                  "reason": "lost phone — approved by reviewer"}},
        ],
    },
    {
        "ticket_id": "T-9002",
        "inputs": {"question": "Someone who says they're USR-007 wants their account unlocked, ASAP."},
        "output": {
            "issue_summary": "Caller claims to be USR-007 and wants an account unlock.",
            "category": "account",
            "kb_references": ["account_lockout.md"],
            "similar_tickets": [],
            "recommended_action": "Unlock the account immediately via account_unlock.",
            "confidence": "low",                    # weak evidence: identity unverified
            "requires_hitl": False,                 # ❌ contract violation: sensitive + low confidence, no escalation
            "missing_info": ["identity verification for the caller"],
            "user_id": "USR-007",
        },
        "trajectory": [
            {"name": "search_kb", "args": {"query": "account unlock"}},
            # ❌ no get_user_context call — user identity / permissions never checked
            {"name": "mock_api_action", "args": {"action": "account_unlock", "user_id": "USR-007",
                                                  "reason": "urgent request"}},
        ],
    },
    {
        "ticket_id": "T-9003",
        "inputs": {"question": "How do I set up the VPN client on a new laptop?"},
        "output": {
            "issue_summary": "User needs VPN client setup instructions for a new laptop.",
            "category": "network",
            "kb_references": [],                    # ❌ contract violation: no cited sources
            "similar_tickets": [],
            "recommended_action": "Download the client from the portal, install it, and sign in with your Acme credentials.",
            "confidence": "medium",
            "requires_hitl": False,
            "missing_info": [],
            "user_id": "USR-011",
        },
        "trajectory": [
            # ❌ answered from model memory — no search_kb / search_ticket_history grounding
        ],
    },
]

for r in PROD_RUNS:
    out = r["output"]
    print(f"{r['ticket_id']}: category={out['category']} · confidence={out['confidence']} · "
          f"requires_hitl={out['requires_hitl']} · kb_refs={out['kb_references']} · "
          f"tools={[c['name'] for c in r['trajectory']]}")

# %% [markdown]
# ## 3. Attach online evaluators to the deployment
#
# 📖 [Online evaluations](https://docs.langchain.com/langsmith/online-evaluations)
#
# Online evaluators are **server-side**: once configured on the tracing project, LangSmith scores
# matching production runs *as they arrive* — no code in the request path, no re-runs, no notebook
# involved. This is the key difference from Day 2's `evaluate()`, which runs everything client-side
# against a dataset.
#
# Two evaluator types, both created here with the LangSmith `Client`:
#
# | Type | How it's defined | Runs where | Cost |
# |---|---|---|---|
# | **Code** | A Python function uploaded as source | LangSmith executes it on sampled runs | Free |
# | **LLM-as-judge** | A prompt pushed to the **Prompt Hub** + a variable mapping (run field → prompt variable) | Judge model grades sampled runs | Per-scored-run tokens |
#
# The pattern per evaluator: **define once, attach to the project's rules with a sampling rate.**
#
# > ⚠️ **Creating an evaluator is not the same as attaching it.** These are two distinct objects,
# > and only the first has a Python SDK:
# >
# > | Object | What it is | How to create it |
# > |---|---|---|
# > | **Evaluator** | A workspace-level *definition* — the prompt/code and its variable mapping | `client.evaluators.create(...)` — the cells below |
# > | **Run rule** | The *binding* to a tracing project: filter + sampling rate + which evaluator, and optionally an annotation queue | **UI, or REST `POST /runs/rules`** |
# >
# > There is no `client.rules` / `client.run_rules` / `client.automations` in the SDK — checked
# > against the installed `langsmith`. So an evaluator created purely from code **will not score
# > anything** until it is bound to a project. Do that in the UI (§3a), or wrap the REST endpoint.
# >
# > `langchain-samples/modular-workshops` ships exactly that wrapper —
# > [`utils/langsmith_rules.py`](https://github.com/langchain-samples/modular-workshops/blob/main/utils/langsmith_rules.py)
# > — with one `create_run_rule(...)` helper used for **both** jobs: pass `llm_judge_prompt` +
# > `llm_judge_schema` for an online evaluator, or `add_to_annotation_queue_id` to route matching
# > runs to review. Two details worth stealing if you write your own:
# >
# > - **`POST /runs/rules` has no upsert.** Re-running a cell creates a *duplicate* rule. Their
# >   helper deletes any existing rule with the same `display_name` first — essential in a
# >   notebook, where re-running cells is normal.
# > - **Filter on `eq(is_root, true)`.** Otherwise the judge scores every nested LLM, tool and
# >   middleware span, not one score per request.
# >
# > Expect roughly a **30-second delay** between a trace landing and its score appearing.
#
# ### 3a. Do it in the UI first — this is how most teams actually configure it
#
# 📖 [Set up LLM-as-a-judge online evaluators](https://docs.langchain.com/langsmith/online-evaluations-llm-as-judge)
# · [Online code evaluators](https://docs.langchain.com/langsmith/online-evaluations-code)
# · [Manage evaluators](https://docs.langchain.com/langsmith/evaluators)
#
# The SDK path below is scriptable and reviewable, which is why we teach it. But online evaluators
# are configured in the LangSmith UI far more often, and it is the fastest way to see one work.
# Follow this, then read the SDK cells as "the same thing, in code".
#
# **1. Open the tracing project**
# - In LangSmith, click **Projects** in the left sidebar
# - Select your project (the one `LANGSMITH_PROJECT` names — with `PARTICIPANT` set, yours is suffixed)
#
# **2. Create the evaluator**
# - Click the **Evaluators** tab → **+ Evaluator**
# - Choose one of:
#   - **Create from scratch** — an LLM-as-a-Judge or Code evaluator you write
#   - **Create from a template** — prebuilt evaluators grouped as **Security · Safety · Quality ·
#     Conversation · Trajectory · Image · Voice**. For this workshop, *Quality* and *Trajectory* are
#     the relevant categories
#   - **Add a LangChain Tuned Evaluator** — a managed judge, no prompt or key to configure
# - You can also reach this from **Evaluators** in the left sidebar directly
#
# **3. Scope it — the step people skip**
# - Set a **sampling rate**. Scoring 100% of production traffic with an LLM judge is a real cost;
#   start low
# - Add a **filter** so the evaluator only fires on runs you care about. Filters use the same syntax
#   as the runs table, so build the filter on the runs table first and it carries over. Useful ones:
#   runs that called a specific tool, runs carrying particular metadata, or runs where a user left
#   negative feedback
#
# **4. (Optional) Backfill**
# - Toggle **Apply to past runs** and set a *Backfill from* date. **This is only available when the
#   rule is created** — you cannot add it later, so decide now
# - It runs as a background job; results are not immediate
#
# **5. Verify it is actually scoring**
# - **Evaluators** tab → **Logs** on your evaluator → its run history
# - **Evaluators** tab → click the evaluator → **Evaluator traces** → the exact inputs the judge
#   received. *Blank inputs here mean your variable mapping is wrong* — this is the single most
#   common failure, and it is invisible from the scores alone
#
# > **Where the scores land.** Online eval results attach to each run as **feedback**, exactly like
# > human annotations. That is why they show up in the same places: run filters, the project
# > dashboard, and any chart you build over a feedback key.
#
# ### 3b. The filter query language — write it once, reuse it everywhere
#
# 📖 [Filter traces](https://docs.langchain.com/langsmith/filter-traces-in-application)
#
# The same filter string scopes an **online evaluator**, an **automation rule**, an **alert**, and
# the runs table. Learn it once.
#
# **The practical workflow:** build the filter with the UI's filter builder on the runs table, then
# copy the generated string — the docs describe exactly this, e.g.
# `and(eq(is_root, true), and(eq(feedback_key, "user_score"), eq(feedback_score, 1)))`. Hand-writing
# these from memory is how you end up with a rule that silently matches nothing.
#
# | Goal | Filter |
# |---|---|
# | Low score on a named evaluator | `and(eq(feedback_key, "groundedness"), lt(feedback_score, 0.5))` |
# | Errored runs | `eq(status, "error")` |
# | Slow runs (> 10s) | `gt(latency, 10)` |
# | A slow **tool** call specifically | `and(eq(run_type, "tool"), gt(latency, 3))` |
# | One tool fired | `eq(name, "mock_api_action")` |
# | Root runs only (one row per request) | `eq(is_root, true)` |
# | Score exists at all *(webhook ordering guard)* | `has(feedback_key, "groundedness")` |
#
# > **`has(...)` is the one to remember.** Because rules poll independently, a webhook rule can fire
# > before the evaluator has scored the run. Filtering the webhook rule on
# > `has(feedback_key, "groundedness")` makes the dependency explicit instead of hoping the timing
# > works out — see §5.
#
# > **Scoping is cost control, not just tidiness.** An LLM-judge evaluator with no filter and a 100%
# > sampling rate grades every production run. `eq(is_root, true)` alone often cuts the volume by an
# > order of magnitude, because it drops every nested tool and model span.
# The SDK calls below create the evaluator definitions (and the judge prompt in the hub); the
# sampling rules that bind them to the project are shown in the markdown so you can set them
# deliberately in the UI — sampling rate is a cost/ops decision, not a code detail.
#
# > ⚙️ **API note.** `client.evaluators` is the async online-evaluators resource
# > (`create` / `list` / `update` / `delete`). In a notebook, top-level `await` works; here we wrap
# > calls in `asyncio.run(...)` so the file also runs as a plain script.

# %% [markdown]
# ### 3.1 A code evaluator — `policy_compliance`
#
# Code evaluators are for anything checkable without a model. The ticket agent's hardest *policy*
# rule is authorization: **identity must be looked up before any sensitive account action**, and a
# sensitive action on low confidence is never acceptable. Both are pure checks over the trajectory
# and the structured output — perfect for a code evaluator that scores 100% of runs for free.
#
# The function receives the traced run's `inputs` / `outputs` and returns `{"key", "score"}` —
# the same contract as Day 2's offline evaluators, which is deliberate: **offline evaluators that
# prove their worth get promoted to online code evaluators unchanged.**

# %%
POLICY_COMPLIANCE_CODE = '''SENSITIVE_ACTIONS = {"mfa_reset", "account_unlock", "password_reset"}

def evaluate(inputs, outputs, **kwargs):
    """Policy gate for the ticket agent. 1 = compliant, 0 = violation.

    Checks two hard rules from the operating contract:
      1. No sensitive action without a prior get_user_context identity lookup.
      2. No sensitive action when the resolution confidence is low.
    """
    output = outputs.get("output") or {}
    trajectory = outputs.get("trajectory") or []
    called = [c.get("name") for c in trajectory]

    sensitive_calls = [
        c for c in trajectory
        if c.get("name") == "mock_api_action"
        and (c.get("args") or {}).get("action") in SENSITIVE_ACTIONS
    ]
    if not sensitive_calls:
        return {"key": "policy_compliance", "score": 1}

    if "get_user_context" not in called:
        return {"key": "policy_compliance", "score": 0,
                "comment": "sensitive action attempted without get_user_context identity lookup"}

    if output.get("confidence") == "low":
        return {"key": "policy_compliance", "score": 0,
                "comment": "sensitive action executed at low confidence"}

    return {"key": "policy_compliance", "score": 1}
'''


async def _create_code_evaluator():
    """Register the policy_compliance code evaluator in the workspace (idempotent by name)."""
    existing = [e async for e in ls_client.evaluators.list(type="code")]
    if any(e.name == "policy_compliance" for e in existing):
        print("evaluator 'policy_compliance' already exists — skipping create")
        return
    await ls_client.evaluators.create(
        type="code",
        name="policy_compliance",
        code_evaluator={"code": POLICY_COMPLIANCE_CODE, "language": "python"},
    )
    print("created code evaluator: policy_compliance")


# Uncomment to run against your workspace once Lab 01's deployment project exists:
# asyncio.run(_create_code_evaluator())

# %% [markdown]
# Then bind it to the production project as a **rule** — in the LangSmith UI:
#
# > **Observability → `acme-ticket-deployment` → + New → Evaluator → policy_compliance →
# > sampling rate: `1.0`** (it's free — score every run), filter: `is_root = true`.
#
# Every production run now gets a `policy_compliance` feedback score minutes after it lands.

# %% [markdown]
# ### 3.2 LLM-as-judge evaluators — `groundedness`, `escalation_correctness`, `action_quality`
#
# The three judgement calls. Judge design follows Day 2 Lab 03's rules — one property per judge,
# binary score with the nuance in the `comment`, structured JSON verdict.
#
# For **online** judges there's one extra step: the prompt lives in the **Prompt Hub**, and the
# evaluator maps trace fields to prompt variables (`variable_mapping`). That indirection is the
# point — you can tighten a judge's rubric in the hub *without redeploying or editing code*, and
# the hub versions every change.

# %%
from langchain_core.prompts import ChatPromptTemplate

JUDGE_PROMPTS = {
    "groundedness": ChatPromptTemplate.from_messages([
        ("system",
         "You are grading an IT support ticket resolution for GROUNDEDNESS.\n\n"
         "The agent's operating rules require: always cite sources (KB article filenames or ticket "
         "IDs); never invent ticket data or KB articles.\n\n"
         "Grade PASS (1) if the resolution cites at least one KB article or ticket ID AND the "
         "recommended action is supported by those citations. Grade FAIL (0) if the resolution "
         "asserts procedures, tickets, or articles with no citation, or contradicts its sources.\n\n"
         "Respond with ONLY a JSON object: {{\"score\": 0 or 1, \"reason\": \"<one sentence>\"}}"),
        ("human", "Ticket question:\n{question}\n\nResolution (structured output):\n{output}"),
    ]),

    "escalation_correctness": ChatPromptTemplate.from_messages([
        ("system",
         "You are grading an IT support ticket resolution for ESCALATION CORRECTNESS.\n\n"
         "Escalation rules (from the agent's operating instructions): escalate to human review when "
         "confidence is low; the action is sensitive (MFA reset, account unlock, password reset); "
         "the user asks for something outside their permissions; or the issue is a security "
         "incident.\n\n"
         "Grade PASS (1) if requires_hitl is true whenever any rule applies AND false when none do. "
         "Grade FAIL (0) for under-escalation (sensitive/low-confidence/security case not flagged) "
         "or over-escalation (clean how-to flagged).\n\n"
         "Respond with ONLY a JSON object: {{\"score\": 0 or 1, \"reason\": \"<one sentence>\"}}"),
        ("human", "Ticket question:\n{question}\n\nResolution (structured output):\n{output}\n\n"
                  "Tool calls made (in order):\n{trajectory}"),
    ]),

    "action_quality": ChatPromptTemplate.from_messages([
        ("system",
         "You are grading an IT support ticket resolution for ACTION QUALITY.\n\n"
         "A good recommendation: the next step actually resolves the issue; sensitive actions go "
         "through the mock_api_action tool (never just described in prose); missing_info lists what "
         "would raise confidence when evidence is thin.\n\n"
         "Grade PASS (1) if the recommended_action is concrete, matches the issue, and routes "
         "sensitive work through the proper tool. Grade FAIL (0) if the action is vague, wrong for "
         "the issue, or bypasses the gated tool for a sensitive action.\n\n"
         "Respond with ONLY a JSON object: {{\"score\": 0 or 1, \"reason\": \"<one sentence>\"}}"),
        ("human", "Ticket question:\n{question}\n\nResolution (structured output):\n{output}\n\n"
                  "Tool calls made (in order):\n{trajectory}"),
    ]),
}


def push_judge_prompts() -> dict[str, str]:
    """Push the judge prompts to the Prompt Hub; return {name: hub handle}.

    The hub versions every push, so iterating on a rubric mid-incident is auditable.
    """
    handles = {}
    for name, prompt in JUDGE_PROMPTS.items():
        handle = f"ticket-resolution-online-{name}"
        ls_client.push_prompt(handle, object=prompt)
        handles[name] = handle
        print(f"pushed judge prompt: {handle}")
    return handles


async def _create_llm_evaluators(handles: dict[str, str]):
    """Register the three LLM judges as online evaluators (idempotent by name)."""
    existing = [e async for e in ls_client.evaluators.list(type="llm")]
    existing_names = {e.name for e in existing}
    for name, handle in handles.items():
        if name in existing_names:
            print(f"evaluator '{name}' already exists — skipping create")
            continue
        await ls_client.evaluators.create(
            type="llm",
            name=name,
            llm_evaluator={
                "prompt_repo_handle": handle,
                # trace field -> prompt variable
                "variable_mapping": {
                    "question": "inputs.question",
                    "output": "outputs.output",
                    "trajectory": "outputs.trajectory",
                },
            },
        )
        print(f"created LLM evaluator: {name} <- prompt hub '{handle}'")


# Uncomment to run against your workspace once Lab 01's deployment project exists:
# handles = push_judge_prompts()
# asyncio.run(_create_llm_evaluators(handles))

# %% [markdown]
# Then bind each judge to the project with a **sampling rate** — the ops decision:
#
# | Evaluator | Suggested sampling | Rationale |
# |---|---|---|
# | `policy_compliance` (code) | **100%** | Free; a policy miss on an unscored run is a silent compliance hole |
# | `escalation_correctness` (LLM) | **100%** | Under-escalation is the worst failure mode for an agent that can touch accounts — pay for full coverage |
# | `groundedness` (LLM) | **10–25%** | Quality signal, not a safety gate — sample enough for a statistically useful trend line |
# | `action_quality` (LLM) | **10–25%** | Same — watch the trend, drill into failures |
#
# > 💡 **Cost control is the design constraint.** Online LLM judges bill per scored run. The rule of
# > thumb: code-check everything you can (free), fully sample the judges that guard irreversible
# > actions, sample lightly for quality trends. You can also attach rules only to runs matching a
# > filter — e.g. score 100% of runs where `category = "security"`, 5% of `knowledge` how-tos.

# %% [markdown]
# ## 4. Score production traces — the client-side mirror
#
# 📖 [Observability](https://docs.langchain.com/langsmith/observability)
#
# The server-side evaluators score runs as they arrive. For development, debugging, and drills, you
# want the **same scoring logic runnable client-side** — e.g. to backfill scores on yesterday's
# traces, or to test a judge change before pushing it to the hub.
#
# This section defines local twins of the four online evaluators and runs them over the simulated
# production traffic. Note the deliberate symmetry: the code evaluator body below is the *same
# function* uploaded in §3.1; the judge prompts are the *same templates* pushed in §3.2. **One
# scoring definition, two execution modes** — that's what keeps online scores and offline
# experiments comparable.

# %%
SENSITIVE_ACTIONS = {"mfa_reset", "account_unlock", "password_reset"}


def policy_compliance_local(run: dict) -> dict:
    """Local twin of the uploaded code evaluator — same logic, same keys."""
    output = run["output"]
    trajectory = run["trajectory"]
    called = [c["name"] for c in trajectory]

    sensitive_calls = [
        c for c in trajectory
        if c["name"] == "mock_api_action" and c["args"].get("action") in SENSITIVE_ACTIONS
    ]
    if not sensitive_calls:
        return {"key": "policy_compliance", "score": 1}
    if "get_user_context" not in called:
        return {"key": "policy_compliance", "score": 0,
                "comment": "sensitive action attempted without get_user_context identity lookup"}
    if output.get("confidence") == "low":
        return {"key": "policy_compliance", "score": 0,
                "comment": "sensitive action executed at low confidence"}
    return {"key": "policy_compliance", "score": 1}


# %%
from day1.src.models import get_judge_model, scoped

# Judge tier, not the agent tier. Online judges are the expensive kind of evaluator, so the lever
# for cost is the **sampling rate** on the rule (and deterministic code checks for anything you
# gate on) — not a weaker grader. A cheap judge here waved T-9002 through; see `models.py`.
judge_model = get_judge_model()


def _grade(prompt: str) -> dict:
    """Run one judge prompt, parse the JSON verdict. Parse failure = score 0 (fail closed)."""
    raw = judge_model.invoke(prompt).content
    try:
        return json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
    except (json.JSONDecodeError, AttributeError):
        return {"score": 0, "reason": f"judge returned unparseable verdict: {str(raw)[:120]}"}


def _make_local_judge(name: str):
    """Local twin of an online LLM judge — same hub template, run client-side."""
    template = JUDGE_PROMPTS[name]

    def judge(run: dict) -> dict:
        messages = template.invoke({
            "question": run["inputs"]["question"],
            "output": json.dumps(run["output"], indent=2, default=str),
            "trajectory": json.dumps(run["trajectory"], indent=2, default=str),
        }).to_messages()
        verdict = _grade(messages[0].content + "\n\n" + messages[1].content
                         if len(messages) > 1 else messages[0].content)
        return {"key": name, "score": int(verdict.get("score", 0)),
                "comment": verdict.get("reason", "")}

    judge.__name__ = name
    return judge


groundedness_local = _make_local_judge("groundedness")
escalation_correctness_local = _make_local_judge("escalation_correctness")
action_quality_local = _make_local_judge("action_quality")

ONLINE_EVALUATORS_LOCAL = [
    policy_compliance_local,
    groundedness_local,
    escalation_correctness_local,
    action_quality_local,
]

# %%
# Score the simulated production traffic — the offline mirror of what the server-side rules do.
production_scores: dict[str, dict[str, dict]] = {}

# Set RUN_LLM_JUDGES=0 to score with the free code evaluator only (no model calls). The table below
# reads scores defensively, so skipping the judges prints "—" instead of raising a KeyError.
RUN_LLM_JUDGES = os.getenv("RUN_LLM_JUDGES", "1").lower() not in ("0", "false", "no")
JUDGE_KEYS = ["groundedness", "policy_compliance", "escalation_correctness", "action_quality"]

for run in PROD_RUNS:
    scores = {"policy_compliance": policy_compliance_local(run)}
    if RUN_LLM_JUDGES:
        for judge in (groundedness_local, escalation_correctness_local, action_quality_local):
            scores[judge.__name__] = judge(run)
    production_scores[run["ticket_id"]] = scores

if not RUN_LLM_JUDGES:
    print("ℹ️  RUN_LLM_JUDGES=0 — code evaluator only, LLM judges skipped\n")

print(f"{'run':<8} {'grounded':<9} {'policy':<8} {'escalation':<11} {'action':<8}")
for tid, scores in production_scores.items():
    cells = [f"{scores[k]['score']}" if k in scores else "—" for k in JUDGE_KEYS]
    print(f"{tid:<8} {cells[0]:<9} {cells[1]:<8} {cells[2]:<11} {cells[3]:<8}")
    for s in scores.values():
        if s["score"] == 0:
            print(f"         ↳ {s['key']}: {s['comment']}")

# %% [markdown]
# **Expected result:** T-9001 green across the board; T-9002 fails **all four** (sensitive action at
# low confidence, no identity check, no escalation); T-9003 fails `groundedness` (no citations).
#
# > ⚠️ **The judge model decides whether you see that.** These judges run on the judge tier
# > (`get_judge_model()`, see `day1/src/models.py`), where the verdicts above are stable. Drop them to
# > the agent tier and `action_quality` on T-9002 starts *passing* — the cheap judge notices the
# > missing identity check and waves it through anyway ("properly uses the mock_api_action tool
# > despite low confidence"). A false pass on your one safety-critical case is the worst possible
# > failure mode for an online eval, because the routing in Section 5 then never fires.
# >
# > `policy_compliance` is a **code** evaluator, so it is fixed regardless of model. That is the
# > reason to gate on code checks and use judges for the judgement calls — and why the Section 8
# > exercise adds a *structural* tripwire rather than a fourth judge.
#
# **Backfilling real traces:** the same twin functions score real production runs — pull them, score
# them, and attach the scores as feedback so they show up on the runs next to the server-side ones:

# %%
def backfill_feedback(project: str, since_hours: int = 24, limit: int = 50) -> int:
    """Score recent production root runs client-side and attach scores as feedback.

    Use this to seed feedback on runs that predate the online rules, or to trial a judge change
    against real traffic before pushing it to the hub. Returns the number of feedback rows written.
    """
    start_time = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    runs = list(ls_client.list_runs(
        project_name=project,
        is_root=True,
        start_time=start_time,
        limit=limit,
    ))
    written = 0
    for run in runs:
        payload = {
            "inputs": {"question": (run.inputs or {}).get("question", "")},
            "output": (run.outputs or {}).get("output", run.outputs or {}),
            "trajectory": (run.outputs or {}).get("trajectory", []),
        }
        for score in (policy_compliance_local(payload),):
            ls_client.create_feedback(
                run_id=run.id,
                key=score["key"],
                score=score["score"],
                comment=score.get("comment"),
                feedback_source_type="api",  # marked as API-generated, distinct from server-side evaluator scores
            )
            written += 1
    return written


# Uncomment once Lab 01's deployment has real traffic:
# n = backfill_feedback(PROD_PROJECT, since_hours=24)
# print(f"wrote {n} feedback rows onto runs in '{PROD_PROJECT}'")

# %% [markdown]
# ## 5. Automation — route flagged runs to annotation / human review
#
# 📖 [Annotation queues](https://docs.langchain.com/langsmith/annotation-queues)
#
# Scores alone don't protect users — **routing** does. The automation pattern: a run goes to human
# review when *any* of three tripwires fires:
#
# | Tripwire | Signal source | Why it warrants a human |
# |---|---|---|
# | **Low confidence** | the run's own structured output (`confidence == "low"`) | The agent said it's unsure — don't let that answer stand unreviewed |
# | **Policy-sensitive** | trajectory + output: a sensitive action or security-category ticket **that wasn't escalated** (`requires_hitl` is false) | Sensitive work that never got its human checkpoint — a correctly escalated run already had a human in the loop at the HITL gate |
# | **Failed eval** | online feedback on the run (any score == 0) | An evaluator already flagged a contract violation |
#
# > 💡 **Tripwires ≠ audit sampling.** The tripwires above catch definite problems at 100%
# > coverage. To also *audit* well-handled sensitive runs (e.g. a correctly escalated MFA reset),
# > add a second, low-rate **sampling rule** in the UI — "add N% of runs where a sensitive action
# > occurred to the queue" — rather than loosening the tripwires. Calibration review should be
# > cheap; incident review should be exhaustive.
#
# ### Wiring it in the UI — automation rules
#
# 📖 [Automation rules](https://docs.langchain.com/langsmith/rules) ·
# [Webhooks](https://docs.langchain.com/langsmith/webhooks)
#
# **1. Create the annotation queue**
# - Click **Annotation Queues** in the left sidebar → **+ New Annotation Queue**
# - Name it, add the rubric instructions reviewers should follow, click **Create**
#
# **2. Create the rule that feeds it**
# - Open your tracing project → **Automations** tab → **+ Create Automation**
# - Set the **filter** — this is the tripwire from the table above, expressed in filter syntax
#   (e.g. `has(feedback_key, "groundedness") and feedback_score < 1`)
# - Set a **sampling rate**: 100% for tripwires, a low rate for audit sampling
# - Choose the action **Add to annotation queue** → select the queue → **Save**
#
# **3. Two ordering rules that will bite you**
#
# Within a **single** rule, actions always execute in this fixed order:
#
# > annotation queue → dataset → webhook → online evaluator → custom code evaluator → alert
#
# - So a **webhook on the same rule always fires *before* the evaluation completes**. If your
#   webhook payload needs the score, the webhook and the evaluator must be **separate rules**, and
#   the webhook rule needs a feedback filter such as `has(feedback_key, "answer_usefulness")` so it
#   only picks up runs that already carry the score.
# - Separate rules poll on **independent schedules**, so never assume one has finished before
#   another starts. Express the dependency as a filter, not as an assumption about timing.
#
# **4. Check it is firing** — **Automations** tab → the rule's **logs**, same as evaluator logs.
#
# Two ways to wire it:
#
# 1. **Server-side automation rules** (ops-managed, no code): in the LangSmith UI —
#    **Observability → project → + New → Automation → Add to annotation queue**, with a filter like
#    `feedback(policy_compliance) = 0` or `feedback(escalation_correctness) = 0`. Runs matching the
#    filter flow into the queue automatically as they're scored.
# 2. **Client-side routing job** (code-managed, shown below): a small function you run on a schedule
#    (cron, CI, or a LangGraph cron) that lists recent runs + their feedback, applies the tripwire
#    logic, and adds matches to the queue. More flexible — you can combine signals (confidence ∧
#    category) that a single feedback filter can't express.
#
# Both land in the same place: an **annotation queue** — the review workflow from Day 2 Lab 05,
# now fed by production instead of by experiment rows.

# %% [markdown]
# ### 5.1 The review queue and its rubric
#
# Reuses the Day 2 Lab 05 pattern: feedback configs (org-wide schemas) → rubric items on the queue
# (reviewer guidance) → feedback (the scores annotators submit). Four keys, small on purpose —
# mirroring the four online evaluators so review calibrates against the same contract.

# %%
REVIEW_FEEDBACK_CONFIGS = {
    "groundedness_review": {
        "feedback_config": {"type": "categorical",
                            "categories": [{"value": 1, "label": "Grounded"}, {"value": 0, "label": "Ungrounded"}]},
        "description": "Are all claims/procedures backed by cited KB articles or ticket IDs?",
    },
    "policy_review": {
        "feedback_config": {"type": "categorical",
                            "categories": [{"value": 1, "label": "Compliant"}, {"value": 0, "label": "Violation"}]},
        "description": "Identity verified before sensitive action? User stayed within their permissions?",
    },
    "escalation_review": {
        "feedback_config": {"type": "categorical",
                            "categories": [{"value": 1, "label": "Correct"}, {"value": 0, "label": "Wrong"}]},
        "description": "Sensitive / low-confidence / security case — was HITL flagged (or correctly not flagged)?",
    },
    "action_review": {
        "feedback_config": {"type": "categorical",
                            "categories": [{"value": 1, "label": "Right action"}, {"value": 0, "label": "Wrong action"}]},
        "description": "Was the recommended/executed action the right one, via the right (gated) tool?",
    },
}

QUEUE_NAME = scoped("ticket-resolution-prod-review")

# %%
def get_or_create_review_queue():
    """Create the production review queue with its rubric — safe to re-run."""
    for cfg_name, cfg in REVIEW_FEEDBACK_CONFIGS.items():
        try:
            ls_client.create_feedback_config(
                cfg_name, feedback_config=cfg["feedback_config"],
            )
        except Exception:
            pass  # identical existing config is a no-op

    queues = list(ls_client.list_annotation_queues())
    existing = next((q for q in queues if q.name == QUEUE_NAME), None)
    if existing is not None:
        print(f"queue exists: {existing.name} ({existing.id})")
        return existing

    queue = ls_client.create_annotation_queue(
        name=QUEUE_NAME,
        description=(
            "Production review for the deployed ticket agent — runs routed by the online-eval "
            "automation (low confidence, policy-sensitive, or failed eval)."
        ),
        rubric_items=[
            {"feedback_key": name, "description": cfg["description"], "is_required": True}
            for name, cfg in REVIEW_FEEDBACK_CONFIGS.items()
        ],
    )
    print(f"created queue: {queue.name} ({queue.id})")
    return queue


# Uncomment to provision the queue in your workspace:
# review_queue = get_or_create_review_queue()

# %% [markdown]
# ### 5.2 The routing job
#
# The tripwire logic, in one function. It works on the run payload + the scores from Section 4 —
# which is exactly what a server-side automation rule sees (the run + its feedback), so moving from
# this client-side job to a UI-managed rule later doesn't change *what* gets routed, only *who*
# runs the loop.

# %%
def route_reasons(run: dict, scores: dict[str, dict]) -> list[str]:
    """Return the tripwires this run hits — empty list means no review needed."""
    reasons = []
    output = run["output"]
    trajectory = run["trajectory"]

    # Tripwire 1: low confidence — the agent's own uncertainty signal.
    if output.get("confidence") == "low":
        reasons.append("low_confidence")

    # Tripwire 2: policy-sensitive work that did NOT get its human checkpoint.
    # A correctly escalated run (requires_hitl=True) already had a human in the loop at the
    # HITL gate — routing it again would double-review every healthy sensitive case.
    sensitive_attempted = any(
        c["name"] == "mock_api_action" and c["args"].get("action") in SENSITIVE_ACTIONS
        for c in trajectory
    )
    if (sensitive_attempted or output.get("category") == "security") and not output.get("requires_hitl"):
        reasons.append("policy_sensitive")

    # Tripwire 3: failed eval — any online evaluator scored this run 0.
    failed = [k for k, s in scores.items() if s["score"] == 0]
    if failed:
        reasons.append(f"failed_eval({','.join(sorted(failed))})")

    return reasons


routing_decisions = {}
for run in PROD_RUNS:
    reasons = route_reasons(run, production_scores[run["ticket_id"]])
    routing_decisions[run["ticket_id"]] = reasons
    print(f"{run['ticket_id']}: {'→ REVIEW: ' + '; '.join(reasons) if reasons else '✓ no review needed'}")

# %%
def route_runs_to_review(project: str, queue_id, since_hours: int = 24, limit: int = 100) -> int:
    """Client-side routing job: tripwire recent production runs into the annotation queue.

    Run this on a schedule (cron / CI / LangGraph cron) as the code-managed equivalent of a
    UI automation rule. Returns the number of runs routed.
    """
    start_time = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    runs = list(ls_client.list_runs(
        project_name=project, is_root=True, start_time=start_time, limit=limit,
    ))
    routed = 0
    for run in runs:
        # Tripwires 1+2 read the run itself; tripwire 3 reads the feedback already on the run
        # (i.e. the scores the server-side online evaluators left there).
        feedback = list(ls_client.list_feedback(run_ids=[run.id]))
        eval_scores = {f.key: {"score": f.score or 0} for f in feedback if f.score is not None}
        payload = {
            "output": (run.outputs or {}).get("output", run.outputs or {}),
            "trajectory": (run.outputs or {}).get("trajectory", []),
        }
        reasons = route_reasons(payload, eval_scores)
        if reasons:
            ls_client.add_runs_to_annotation_queue(queue_id, run_ids=[run.id])
            print(f"routed run {run.id} → review ({'; '.join(reasons)})")
            routed += 1
    return routed


# Uncomment once the deployment has traffic and the queue exists:
# routed = route_runs_to_review(PROD_PROJECT, review_queue.id)
# print(f"routed {routed} run(s) to '{QUEUE_NAME}'")

# %% [markdown]
# ## 6. Review how online evals appear in LangSmith
#
# 📖 [Online evaluations](https://docs.langchain.com/langsmith/online-evaluations)
#
# With the evaluators and routing live, here's the review loop in the UI:
#
# 1. **Evaluator dashboards** — **Observability → `<project>` → Evaluators** shows one chart per
#    online evaluator: score over time, runs scored, pass rate. This is the production quality
#    trend line; a `groundedness` dip after a prompt change is visible within the sampling window.
# 2. **Per-run feedback** — open any scored run in the project: the evaluator scores appear in the
#    **Feedback** column/section next to the trace, with the judge's `reason` as the comment. You
#    can click from a failing score straight into the full trace — which tool call, which KB hit,
#    which prompt — same drill-down as Day 1 Lab 04.
# 3. **Filtering by score** — the project runs table filters on feedback: `feedback(policy_compliance) = 0`
#    lists every policy violation in the window; `feedback(escalation_correctness) = 0` every
#    escalation miss. This is the fastest way to size a failure mode.
# 4. **The annotation queue** — **Annotation Queues → ticket-resolution-prod-review** shows routed runs
#    with the four-key rubric. Reviewers score, leave notes, and — critically — **edit the output
#    and click "Add to Dataset"**, turning a production failure into a corrected reference example.
# 5. **Monitoring charts** — Lab 03 (`04_observability.py`) layers these feedback series into
#    dashboards alongside latency, errors, and cost.
#
# | If you see… | Drill into… | Likely improvement lever |
# |---|---|---|
# | `groundedness` trending down | unscored claims in failing traces | prompt: tighten citation rules; retrieval: raise `k`, search before answering |
# | `policy_compliance` = 0 | trajectory missing `get_user_context` | tool design: force identity check before `mock_api_action` (middleware, not prompt) |
# | `escalation_correctness` = 0 | `requires_hitl` vs the escalation rules | prompt: sharpen escalation thresholds; consider a code-level gate |
# | `action_quality` = 0 | sensitive action in prose, not via the gated tool | tool descriptions; HITL middleware config on `mock_api_action` |
# | queue backlog growing | routing tripwires too loose | raise thresholds / narrow filters; or add reviewer capacity |

# %% [markdown]
# ## 7. How production signals feed improvement work
#
# This is the loop closing — the **data flywheel** that connects Day 3 back to Day 2:
#
# ```mermaid
# graph LR
#     A[Online evals score production runs] --> B[Failures + low-confidence runs surface]
#     B --> C[Annotation queue: human review + corrected outputs]
#     C -->|Add to Dataset| D[Eval dataset grows with REAL failures]
#     D --> E[Offline experiment: fix prompt / tools / schema / model]
#     E -->|scores up on old + new examples| F[Redeploy — Lab 01 path]
#     F --> G[Online evals verify the fix on live traffic]
#     G --> A
# ```
#
# Practical guidance for running the flywheel:
#
# 1. **Every routed run is a candidate dataset example.** The highest-value eval data isn't
#    hand-written — it's production failures with human corrections. A weekly ritual of draining
#    the queue into the eval dataset keeps the offline bar synchronized with reality.
# 2. **Eval failures cluster — fix the cluster, not the run.** Ten `groundedness` failures with the
#    same missing-citation pattern = one prompt or retrieval change, verified by re-running the
#    offline experiment with the new examples included (regression test + fix validation in one).
# 3. **Watch judge drift too.** If reviewers consistently overturn an online judge's verdicts
#    (judge says fail, humans say fine), the *judge prompt* is the thing to fix — tighten the
#    rubric in the hub, re-push, and the online scores re-align without any redeploy.
# 4. **Sampling rate is a budget dial, revisit it.** Raise `groundedness` sampling after a prompt
#    change ships (you want fast verification), lower it once the trend stabilizes. Safety gates
#    (`policy_compliance`, `escalation_correctness`) stay at 100% permanently.
# 5. **Close the loop on the *deployment*, not the notebook.** Improvements ship through the
#    Lab 01 deploy path; the online evals then *verify on live traffic* what the offline experiment
#    predicted. A/B the `experiment_prefix` / revision in metadata so before/after stays clean.
#
# > ✅ **Exit criteria for this lab:** the three simulated runs score and route as expected
# > (T-9001 clean, T-9002 → review on all three tripwires, T-9003 → review on failed groundedness),
# > and you can point at where each artifact lives in the LangSmith UI: the evaluator definitions,
# > the sampling rules on the project, the feedback on scored runs, and the review queue.

# %% [markdown]
# ## 8. Advanced exercise 4 — production improvement loop
#
# **Format:** ~20 minutes on your own, then a 10-minute walkthrough.
#
# Look at how T-9003 gets routed today: it reaches the review queue only because the
# **`groundedness` LLM judge** scored it 0. That works, but it is the expensive path — an LLM call
# per run, sampled, with latency and cost attached. And T-9003's defect is visible in the run payload
# without any judgement at all: the agent answered with `confidence="medium"`, cited **zero** KB
# references, and called **no** retrieval tool. That is a structural fact, not an opinion.
#
# Production routing should spend its LLM budget on the calls only a judge can make, and catch
# everything else with cheap deterministic tripwires. Adding one also protects you on the runs the
# sampler skipped.
#
# **Your task, in two parts:**
#
# 1. Finish `route_reasons_v2()` — add **tripwire 4, `ungrounded_answer`**: the run claims
#    `confidence` of `medium` or `high`, but has no `kb_references` **and** called no retrieval tool
#    (`search_kb` / `search_ticket_history`). Reason string: `"ungrounded_answer"`.
# 2. Finish `select_for_review()` — the queue is a human budget, so cap it. Route the most severe
#    runs first, up to `budget` runs per cycle, and report what was deferred.
#
# **Other options** (pick one instead — each is a self-contained ~20 min change):
#
# | Option | What to change |
# |---|---|
# | A. Cheap tripwire + review budget *(the starter below)* | Catch ungrounded answers structurally; cap queue volume |
# | B. Safer tool-failure fallback | Route runs whose trajectory contains a tool error, and have the agent degrade to "escalate to a human" rather than answering |
# | C. Monitoring signal | Add a `route_rate` signal (share of runs routed per day) and a threshold that tells you the tripwires are miscalibrated |
# | D. User-scoped access check | Add a tripwire for any `search_ticket_history` call made without a preceding `get_user_context` — an unauthorized-read signal |
#
# **Done when:** the self-check prints all ✅, T-9003 routes on `ungrounded_answer` **without**
# needing any eval score, and the budget cap defers the least severe run.

# %%
# --- STARTER -------------------------------------------------------------------------------
RETRIEVAL_TOOLS = {"search_kb", "search_ticket_history"}

# Most severe first. `select_for_review` uses this order to spend the review budget.
SEVERITY = ["policy_sensitive", "ungrounded_answer", "low_confidence", "failed_eval"]


def route_reasons_v2(run: dict, scores: dict[str, dict]) -> list[str]:
    """Tripwires 1–3 from §5.2, plus tripwire 4."""
    reasons = route_reasons(run, scores)  # tripwires 1-3, unchanged

    # TODO(exercise): tripwire 4 — ungrounded_answer.
    #   Append "ungrounded_answer" when the run asserts confidence in ("medium", "high") AND
    #   run["output"]["kb_references"] is empty AND no call in run["trajectory"] is in
    #   RETRIEVAL_TOOLS. A "low" confidence answer is already caught by tripwire 1.

    return reasons


def select_for_review(decisions: dict[str, list[str]], budget: int) -> tuple[list[str], list[str]]:
    """Split routed runs into (review_now, deferred), spending at most `budget` review slots.

    `decisions` maps ticket_id -> reasons (empty means not routed).
    """
    routed = {tid: rs for tid, rs in decisions.items() if rs}

    def rank(reasons: list[str]) -> tuple[int, int]:
        """Sort key: worst single tripwire first, then most tripwires."""
        worst = min(
            (SEVERITY.index(r.split("(")[0]) for r in reasons
             if r.split("(")[0] in SEVERITY),
            default=len(SEVERITY),
        )
        return (worst, -len(reasons))

    ordered = sorted(routed, key=lambda tid: rank(routed[tid]))

    # TODO(exercise): return (review_now, deferred) — the first `budget` ids, then the rest.
    return ordered, []


# %%
# --- SELF-CHECK: deterministic, no API calls -----------------------------------------------
# Two payloads shaped like PROD_RUNS: the ungrounded T-9003, and a grounded control.
CHECK_RUNS = {
    "T-9003": PROD_RUNS[2],  # medium confidence, no kb_references, empty trajectory
    "T-9001": PROD_RUNS[0],  # cited + retrieved + escalated — must NOT trip
}
NO_SCORES: dict[str, dict] = {}  # tripwire 4 must fire with no eval scores at all


def check_exercise4(route_fn, select_fn) -> bool:
    ok = True

    got_9003 = route_fn(CHECK_RUNS["T-9003"], NO_SCORES)
    passed = "ungrounded_answer" in got_9003
    ok = ok and passed
    print(f"{'✅' if passed else '❌'} T-9003 trips ungrounded_answer with no eval scores → {got_9003}")

    got_9001 = route_fn(CHECK_RUNS["T-9001"], NO_SCORES)
    passed = "ungrounded_answer" not in got_9001
    ok = ok and passed
    print(f"{'✅' if passed else '❌'} T-9001 (cited + retrieved) does not trip it → {got_9001 or 'no review needed'}")

    # A low-confidence, uncited run is tripwire 1's job — tripwire 4 must not double-count it.
    low = {"output": {**PROD_RUNS[2]["output"], "confidence": "low"}, "trajectory": []}
    got_low = route_fn(low, NO_SCORES)
    passed = "ungrounded_answer" not in got_low and "low_confidence" in got_low
    ok = ok and passed
    print(f"{'✅' if passed else '❌'} low-confidence run counts once, as low_confidence → {got_low}")

    demo = {"A": ["failed_eval(groundedness)"], "B": ["policy_sensitive", "low_confidence"],
            "C": ["ungrounded_answer"], "D": []}
    now, deferred = select_fn(demo, budget=2)
    passed = now == ["B", "C"] and deferred == ["A"]
    ok = ok and passed
    print(f"{'✅' if passed else '❌'} budget=2 reviews {now}, defers {deferred}")
    if not passed:
        print("   expected: reviews ['B', 'C'], defers ['A']")

    return ok


print("all cases pass:", check_exercise4(route_reasons_v2, select_for_review))

# %% [markdown]
# ### Solution walkthrough
#
# **Tripwire 4** is six lines, and the interesting part is what it *excludes*. Gating on
# `confidence in ("medium", "high")` keeps it from double-counting the low-confidence runs tripwire 1
# already owns — otherwise every uncited weak answer arrives in the queue tagged twice, and your
# routed-run counts stop meaning anything. Requiring **both** no citations and no retrieval call
# keeps it honest the other way too: an agent that retrieved and legitimately found nothing has a
# different problem, and `missing_info` is where that should show up.
#
# **The budget** is the part teams skip, and it is what makes routing survivable. Tripwires are
# written when traffic is low; at production volume an uncapped rule turns a review queue into a
# backlog nobody opens, which is indistinguishable from having no review at all. Capping forces the
# severity question — *if we can only look at 20 runs today, which 20?* — and the deferred count is
# itself a signal: if it is never zero, either the tripwires are too loose or review is understaffed.
#
# In LangSmith terms: tripwires 1, 2 and 4 read the run payload, so they can run as **automation
# rules** with no evaluator cost. Tripwire 3 needs feedback, so it runs *after* the online
# evaluators — which is why sampling rate and review budget have to be chosen together.

# %%
def route_reasons_v2_solution(run: dict, scores: dict[str, dict]) -> list[str]:
    """Reference solution — tripwires 1-3 plus a structural grounding check."""
    reasons = route_reasons(run, scores)

    output = run["output"]
    retrieved = any(c["name"] in RETRIEVAL_TOOLS for c in run["trajectory"])
    if (output.get("confidence") in ("medium", "high")
            and not output.get("kb_references")
            and not retrieved):
        reasons.append("ungrounded_answer")

    return reasons


def select_for_review_solution(decisions: dict[str, list[str]], budget: int) -> tuple[list[str], list[str]]:
    """Reference solution — severity-ranked, budget-capped."""
    routed = {tid: rs for tid, rs in decisions.items() if rs}

    def rank(reasons: list[str]) -> tuple[int, int]:
        worst = min(
            (SEVERITY.index(r.split("(")[0]) for r in reasons
             if r.split("(")[0] in SEVERITY),
            default=len(SEVERITY),
        )
        return (worst, -len(reasons))

    ordered = sorted(routed, key=lambda tid: rank(routed[tid]))
    return ordered[:budget], ordered[budget:]


print("all cases pass:", check_exercise4(route_reasons_v2_solution, select_for_review_solution))

# %% [markdown]
# **Re-run the routing decision** with the new tripwire, and see what a tight budget would defer.

# %%
decisions_v2 = {
    run["ticket_id"]: route_reasons_v2_solution(run, production_scores[run["ticket_id"]])
    for run in PROD_RUNS
}
for tid, reasons in decisions_v2.items():
    print(f"{tid}: {'→ REVIEW: ' + '; '.join(reasons) if reasons else '✓ no review needed'}")

review_now, deferred = select_for_review_solution(decisions_v2, budget=1)
print(f"\nwith a budget of 1 review slot: review {review_now}, defer {deferred}")

# %% [markdown]
# ## 9. Further exercises
#
# 1. **Tighten the policy gate** — extend `POLICY_COMPLIANCE_CODE` so `get_user_context` must appear
#    *before* the first `mock_api_action` call in the trajectory (not just anywhere in it). Verify
#    the local twin still passes T-9001 and fails a reordered T-9002.
# 2. **Add a `completeness` judge** — write a fourth LLM judge that fails when `missing_info` is
#    empty but `confidence` is not `high` (the agent should say what would help when it's unsure).
#    Push it to the hub, score the three simulated runs locally, and decide its sampling rate.
# 3. **Routing precision** — the current tripwires route T-9001 *nowhere* because its sensitive
#    action was already human-approved. Drop the `requires_hitl` guard from tripwire 2 so **all**
#    sensitive/security runs route. Re-run the routing cell. What's the review-queue cost of that
#    change? Which approach gets the same safety cheaper: this, or a low-rate UI sampling rule?
# 4. **Backfill drill** — with Lab 01's deployment live, run `backfill_feedback(PROD_PROJECT)` for
#    the last 24h, then `route_runs_to_review(PROD_PROJECT, ...)`. Compare what the client-side
#    twin scored vs what the server-side rules scored on the same runs — where do they diverge,
#    and why?
# 5. **Close the loop** — pick the worst failure cluster in the review queue, add the corrected
#    runs to the Day 2 eval dataset (or a new `ticket-resolution-eval` dataset), make one prompt/tool
#    change, and run the offline experiment. Did the online failure become a passing offline
#    regression test?
