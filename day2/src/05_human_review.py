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
# # 05 · Human Review — Annotation Queues + Feedback → Dataset
#
# **Workshop:** LangChain ADLC Workshop · **Day 2** · **ADLC stage:** Evaluate + Improve
#
# > **Loop Engineering focus: Verification loop** — human review closes the loop that evaluators
# > can't. Automated evaluators score what's measurable; annotators catch what evaluators miss and
# > turn failures into corrected reference examples for the next experiment.
#
# > **Demo walkthrough** (presenter-led) · ~20 min
#
# ```mermaid
# graph LR
#     A[Experiment traces] -->|add selected runs| B[Annotation queue]
#     B -->|reviewer applies rubric| C[Feedback scores + notes]
#     C -->|edit output → Add to Dataset| D[Corrected reference example]
#     D -->|next experiment| E[Regression + eval set]
#     E -->|new failures| B
# ```
#
# Offline evaluators (Lab 03) and experiments (Lab 04) surface *which* runs look bad. The last mile
# is **human review**: a domain expert opens the trace, scores it against a rubric, and fixes the
# output. In LangSmith this is an **annotation queue** — a focused review workflow with prescribed
# rubric keys, reviewer assignments, and one-click export of corrected runs into a dataset.
#
# By the end you can:
# - Create an annotation queue with a **rubric** (feedback configs + rubric items)
# - Add **selected traces** from an experiment project to the queue
# - Apply a lightweight rubric for the vendor due diligence agent:
#   **evidence quality · source use · escalation correctness · sensitive-data handling**
# - Show how **human feedback and corrections become future dataset examples** (the data flywheel)
#
# > 🧭 **This is a demo, not hands-on.** Watch the pattern; the SDK calls below are exactly what the
# > LangSmith UI does when you click "+ Add → Add to Annotation Queue" and "Add to Dataset".
# > Reference: [Annotation Queues docs](https://docs.langchain.com/langsmith/annotation-queues).


# %% [markdown]
# ### 📚 Stuck on syntax? Reference material
#
# You are not expected to write any of this from memory. When an API signature is the thing in your
# way, look it up — that is what a working engineer does, and every link below is the official source.
#
# | Need | Where to look |
# |---|---|
# | `create_deep_agent(...)`, sub-agents, backends | [`deepagents`](https://github.com/langchain-ai/deepagents/tree/main/libs/deepagents) |
# | `AGENTS.md` memory, `SKILL.md` skills | [deepagents middleware](https://github.com/langchain-ai/deepagents/tree/main/libs/deepagents/deepagents/middleware) |
# | Datasets, evaluators, `evaluate()` | [Evaluation](https://docs.langchain.com/langsmith/evaluation) |
# | Prebuilt judges, RAG + PII evaluators, simulators | [`openevals`](https://github.com/langchain-ai/openevals) |
# | Trajectory / graph-trajectory evaluators | [`agentevals`](https://github.com/langchain-ai/agentevals) |
# | Annotation queues and human review | [Annotation queues](https://docs.langchain.com/langsmith/annotation-queues) |
# | How LangChain builds deep-agent evals | [Benchmarking deep agents](https://www.langchain.com/blog/how-we-benchmark-deep-agents) · [Building evals](https://www.langchain.com/blog/how-we-build-evals-for-deep-agents) |
#
# > **Closest analogue:** the lifecycle workshop's `workshop_modules/module_2/` notebooks — baseline
# > eval, eval-driven development, then advanced evaluation, in that order.

# > 🧭 **Builds on Lab 04; runs standalone.** Lab 04 produced the experiment whose runs you
# > review here. If you have not run it, this lab creates its own queue and seeds it, so you can
# > still follow along. Next: Lab 06 evaluates the agent across every layer.
#
# %% [markdown]
# ## 0. Setup
#
# Same setup cell as the other labs — loads `.env`, prints the LangSmith workspace. Human review
# only needs the **LangSmith SDK** (`langsmith.Client`); no model calls happen in this lab.

# %%
import os
import sys
import warnings
from pathlib import Path


# This lab needs the workshop root on sys.path for `day1.src.models.scoped` — the helper that stops
# 15 participants seeding the same dataset and queue. It is the only Day 2 lab that previously needed
# no local imports at all, which is precisely why adding one broke it.
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
load_dotenv(find_dotenv())

# `Client.list_runs` is deprecated in favour of `client.runs.query()` (removal after Jan 2027). We
# stay on `list_runs` deliberately: on this SDK version `client.runs.query` is **async-only**, takes
# project UUIDs rather than names, paginates by cursor, and requires self-hosted LangSmith >= v0.16 —
# a floor that matters for the air-gapped deployments discussed on Day 3. Silence the notice so the
# lab output stays readable; revisit when a sync surface lands.
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message=r".*list_runs\(\) is deprecated.*")

from day1.src.models import scoped
from langsmith import Client
from langsmith.schemas import RunKey
from langsmith.utils import LangSmithNotFoundError

client = Client()

print("workspace:", os.getenv("LANGSMITH_WORKSPACE_ID"), "| tracing:", os.getenv("LANGSMITH_TRACING"))

# %% [markdown]
# ## 1. Open an annotation queue
#
# An annotation queue has three layers (see
# [Manage feedback & annotation queues programmatically](https://docs.langchain.com/langsmith/annotation-queues-sdk)):
#
# | Layer | What it is | Scope |
# |---|---|---|
# | **Feedback config** | Schema for a feedback key (continuous / categorical / freeform) | org-wide, reusable |
# | **Rubric item** | Assigns a config to a queue with reviewer guidance + required flag | per queue |
# | **Feedback** | The actual scores annotators submit on a run | per run |
#
# **Step 1 — define the rubric's feedback configs.** A feedback key is **shared across the whole
# workspace** and unique — not scoped to a dataset or queue — so `create_feedback_config` returns 400
# for a key that already exists. This cell upserts (create-or-update) to stay re-runnable, and to
# survive a workspace that already defines a key like `evidence_quality` from other work.

# %%
# Lightweight DD review rubric — 4 keys, kept small on purpose:
# the more keys a rubric has, the slower and less consistent human review gets.
FEEDBACK_CONFIGS = {
    "evidence_quality": {
        "feedback_config": {"type": "continuous", "min": 0, "max": 1},
        "description": "Are the assessment's claims backed by specific, sufficient evidence?",
    },
    "source_use": {
        "feedback_config": {"type": "continuous", "min": 0, "max": 1},
        "description": "Does the report cite the right sources (KB, PDFs, screening) and use them faithfully?",
    },
    "escalation_correct": {
        "feedback_config": {
            "type": "categorical",
            "categories": [
                {"value": 1, "label": "Correct"},
                {"value": 0, "label": "Wrong"},
            ],
        },
        "description": "High-risk / low-confidence / sanctioned vendor — was HITL flagged (or correctly not flagged)?",
    },
    "sensitive_data_handling": {
        "feedback_config": {
            "type": "categorical",
            "categories": [
                {"value": 1, "label": "OK"},
                {"value": 0, "label": "Leak"},
            ],
        },
        "description": "Any sensitive data exposed that shouldn't be (PII, internal pricing, sanctioned-party detail)?",
    },
}

# `create_feedback_config` is NOT idempotent — a key that already exists returns
# 400 "Feedback config already exists", which matters twice over in a workshop: the cell dies on a
# second run, and it dies on the *first* run for every participant sharing a workspace where someone
# already created them. Feedback keys are **workspace-scoped and unique** (not per-dataset or
# per-queue), and a real workspace already carries keys from unrelated projects — so upsert.
_existing_configs = {c.feedback_key for c in client.list_feedback_configs()}
for key, cfg in FEEDBACK_CONFIGS.items():
    if key in _existing_configs:
        client.update_feedback_config(key, feedback_config=cfg["feedback_config"])
        print(f"↺ feedback config updated: {key}")
    else:
        client.create_feedback_config(key, feedback_config=cfg["feedback_config"])
        print(f"✔ feedback config created: {key}")

# %% [markdown]
# **Step 2 — create the queue and attach the rubric.** `rubric_instructions` are shown to reviewers
# on every item; each rubric item adds queue-specific guidance. Re-running reuses a queue with the
# same name instead of creating duplicates.
#
# > **UI equivalent:** *Annotation Queues → + Annotation Queue → Basic details + Annotation rubric
# > + Collaborator settings.* Setting a **default dataset** on the queue makes "Add to Dataset"
# > one click in the UI; this lab does not set one, so §4 shows the SDK equivalent instead.

# %%
QUEUE_NAME = os.getenv("DD_REVIEW_QUEUE") or scoped("vendor-due-diligence-review")
DATASET_NAME = os.getenv("DD_DATASET") or scoped("vendor-due-diligence-eval")  # corrected examples land here (§4)

queue = next(iter(client.list_annotation_queues(name=QUEUE_NAME)), None)
if queue is None:
    queue = client.create_annotation_queue(
        name=QUEUE_NAME,
        description="Human review of vendor due diligence assessments (Day 2 experiment traces).",
        rubric_instructions=(
            "Score each rubric key. Escalate anything that mishandles a sanctioned vendor, "
            "invents evidence, or misses a required human-approval flag. "
            "When you fix an output, add the corrected example to the dataset."
        ),
        rubric_items=[
            {"feedback_key": key, "description": cfg["description"], "is_required": True}
            for key, cfg in FEEDBACK_CONFIGS.items()
        ],
    )
    print("✔ created queue:", queue.name)
else:
    print("✔ reusing queue:", queue.name)
print("   queue id:", queue.id)

# %% [markdown]
# ## 2. Add selected traces from the experiment
#
# 📖 [Observability](https://docs.langchain.com/langsmith/observability)
#
# Reviewers shouldn't eyeball every trace — send the **interesting** ones. Typical selection
# signals, all available as filters on the experiment project:
#
# | Signal | Why review it |
# |---|---|
# | Evaluator failure | LLM-as-judge / code evaluator flagged the run |
# | Low confidence | Agent self-reported low confidence in the assessment |
# | Escalated runs | Human-approval path exercised — did it trigger correctly? |
# | Errors / long latency | Something structural went wrong |
# | Random sample | Unbiased baseline quality signal |
#
# Here we take the two most recent root runs from the Day 2 experiment project. In the UI this is
# *select runs → Add to Annotation Queue* (max 100 per action); automation rules can also route
# failed-eval runs into a queue automatically — that's the Day 3 online-evals pattern.

# %%
PROJECT = os.getenv("LANGSMITH_PROJECT", "langchain-adlc-workshop")  # experiment traces from Labs 03–04

# `list_runs` resolves the project by name first, so a project that does not exist raises
# LangSmithNotFoundError rather than returning an empty list — catch it, or the friendly message
# below is unreachable for exactly the person who needs it (someone starting at this lab).
try:
    runs = list(client.list_runs(project_name=PROJECT, is_root=True, limit=2))
except LangSmithNotFoundError:
    runs = []
    print(f"⚠ Project {PROJECT!r} does not exist yet.")

if not runs:
    print(f"⚠ No runs to queue from {PROJECT!r} — run Labs 03–04 first, then re-run this cell.")
    print("  The rubric and queue above are already created, so the UI walkthrough still works.")
else:
    # `run_ids=` is deprecated (removal after Jan 2027) in favour of RunKey objects.
    client.add_runs_to_annotation_queue(
        queue.id,
        runs=[RunKey(run_id=r.id, session_id=r.session_id, start_time=r.start_time) for r in runs],
    )
    def _first_query(inputs) -> str:
        """Pull the user's question out of a run's inputs.

        Run input shapes vary by what produced the run: a chat agent stores
        `{"messages": [{"content": ...}]}`, an evaluated target stores `{"question": ...}`, and some
        record `messages` nested one level deeper. A single project accumulates all three, so index
        defensively — this exact line used to raise `AttributeError: 'list' object has no attribute
        'get'` once the project held deep-agent runs alongside Day 1 ones.
        """
        if not isinstance(inputs, dict):
            return ""
        for key in ("question", "input", "query"):
            if isinstance(inputs.get(key), str):
                return inputs[key]
        node = inputs.get("messages")
        while isinstance(node, list) and node:      # unwrap nesting
            node = node[0]
        if isinstance(node, dict):
            content = node.get("content")
            return content if isinstance(content, str) else str(content or "")
        return ""

    for r in runs:
        q = _first_query(r.inputs)
        print(f"→ queued run {r.id}")
        print(f"    query: {q[:90] if q else '(no user-facing input recorded)'}{'…' if len(q) > 90 else ''}")
    print(f"\n🔎 Open the queue in LangSmith: Annotation Queues → {QUEUE_NAME!r}")

# %% [markdown]
# ## 3. Review with the rubric
#
# 📖 [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
#
# In the queue's review pane, the annotator sees the run's inputs/outputs, the rubric (right
# sidebar), and a notes field. The reviewer:
#
# 1. **Scores each rubric key** — e.g. `evidence_quality: 0.4`, `escalation_correct: Wrong`.
# 2. **Leaves a note** explaining the failure in one sentence.
# 3. Optionally **edits the output** into a corrected reference (§4).
#
# The cell below simulates what a reviewer submits for the first queued run — these
# `create_feedback` calls are exactly what the UI sends when the reviewer clicks **Done**.

# %%
# Simulated reviewer verdict on the first queued run (demo data — a real reviewer
# submits this through the queue UI, not code).
# Categorical keys take the category's **value**, not its label: `escalation_correct` is configured
# with `[{"value": 1, "label": "Correct"}, {"value": 0, "label": "Wrong"}]`, so submitting the string
# "Wrong" lands off-scale and the queue's aggregate for that key is meaningless. Continuous keys take
# `score`; categorical keys take `value` (with the label alongside for the reader).
REVIEW = [
    ("evidence_quality", {"score": 0.4}),                    # continuous 0–1
    ("source_use", {"score": 0.6}),                          # continuous 0–1
    ("escalation_correct", {"value": 0}),                    # categorical → "Wrong"
    ("sensitive_data_handling", {"value": 1}),               # categorical → "OK"
]
NOTE = ("Cited one KB article but missed the PDF capability statement; "
        "sanctioned-adjacent vendor was not escalated — escalation flag wrong.")

if runs:
    target = runs[0]
    for key, payload in REVIEW:
        # `session_id` is required going forward — creating feedback without it is deprecated.
        client.create_feedback(target.id, key=key, session_id=target.session_id,
                               comment=NOTE if key == REVIEW[0][0] else None, **payload)
        print(f"✔ feedback {key!r}: {payload} on run {target.id}")
    print("   note:", NOTE)

# %% [markdown]
# ## 4. Feedback → dataset: the data flywheel
#
# The highest-value output of human review is not the score — it's the **corrected example**.
# In the queue UI, the reviewer edits the run's output and clicks **Add to Dataset**; LangSmith
# stores it as a new reference example. Below is the SDK equivalent: the reviewer's correction
# becomes an example in the eval dataset, so the **next experiment** (Lab 04) is checked against it.
#
# ```mermaid
# graph TD
#     A[Failed / interesting trace] --> B[Human review + correction]
#     B --> C[New dataset example]
#     C --> D[Re-run experiment]
#     D -->|regression caught| E[Fix prompt / tools / retrieval]
#     E --> D
#     D -->|passes| F[Ship with confidence]
# ```
#
# Over time the dataset accumulates exactly the cases the agent gets wrong — a regression set that
# grows with every review round, instead of a static fixture written once.

# %%
from datetime import datetime, timezone

# The reviewer's corrected reference for the queued run (condensed for the demo —
# in the UI the reviewer edits the actual run output, which preserves full fidelity).
# Two things make this a *usable* reference example rather than a note-to-self:
#  1. It names a vendor that exists in the fixtures — VND-013, Kelbrant Trading Consortium, the
#     sanctioned entity in `sanctions_list.json`. An example citing a vendor no tool can retrieve
#     is unreachable by any agent, so it can never pass or fail meaningfully.
#  2. Its `outputs` use the **same keys the Lab 03/04 evaluators read** (`expected_tools`,
#     `expected_risk_level`, `should_escalate`). A corrected example in a private shape is invisible
#     to the evaluators, which quietly breaks the flywheel this section is about.
# Note the *scenario* is new, not just the wording: same sanctioned vendor, a different procurement
# need. A reviewer correction that restates a case the dataset already covers adds no coverage — the
# value is a harder variant that the current agent would still get wrong.
corrected_example = {
    "inputs": {
        "question": ("Assess vendor VND-013 (Kelbrant Trading Consortium) for: Field-vehicle maintenance "
                     "and spare-parts supply for the logistics fleet. Priority: medium, "
                     "Budget: EUR 400000."),
        "vendor_id": "VND-013",
    },
    "outputs": {
        "expected_tools": ["screen_vendor", "search_vendor_kb"],
        "expected_risk_level": "high",
        "should_escalate": True,  # ← the correction: a sanctioned vendor MUST be escalated
        "reference_points": [
            "Vendor appears on the synthetic EU/US sanctions list (SAN-001)",
            "Screening verdict must be reported as SANCTIONED",
            "Assessment must not recommend proceeding",
        ],
    },
    "metadata": {
        "source": "annotation-queue",
        "queue": QUEUE_NAME,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    },
}

if runs:
    # `create_examples` appends — it does not dedupe — so check before adding, or a second review
    # round doubles the example and skews every later experiment score.
    _q = corrected_example["inputs"]["question"]
    _already = any(
        (ex.inputs or {}).get("question") == _q
        for ex in client.list_examples(dataset_name=DATASET_NAME)
    )
    if _already:
        print(f"↺ corrected example already in dataset {DATASET_NAME!r} — not re-adding")
    else:
        client.create_examples(dataset_name=DATASET_NAME, examples=[corrected_example])
        print(f"✔ corrected example added to dataset {DATASET_NAME!r}")
    print("   → Lab 04's next experiment now checks this case automatically.")

# %% [markdown]
# ## 5. Key takeaways
#
# - **Annotation queues** are the human half of the verification loop: rubric + assignment +
#   progress tracking, on top of the traces evaluators already surface.
# - Keep the rubric **small and decision-relevant** — here 4 keys: evidence quality, source use,
#   escalation correctness, sensitive-data handling.
# - Send **selected** traces (eval failures, low confidence, escalations, samples), not everything;
#   automation rules (Day 3) route failures to a queue continuously.
# - **Feedback scores** tell you what's wrong; **corrected examples** fix it permanently. Every
#   review round grows the regression dataset.
# - Everything here is mirrored in the UI: *Add to Annotation Queue*, rubric sidebar, **Done**,
#   **Add to Dataset** — the SDK calls in this lab are the same operations.
#
# > **Day 3 preview:** online evaluators score production traces live, and automation rules push
# > low-scoring runs into a queue like this one — the same flywheel, running continuously.
