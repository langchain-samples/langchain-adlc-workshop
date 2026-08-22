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
# # 04 · Experiments — Eval-Driven Improvement Loop
#
# **Workshop:** LangChain ADLC Workshop · **Day 2** · **ADLC stage:** Evaluate → Improve
#
# > **Loop Engineering focus: Verification loop → Improvement loop** — Lab 03 built the
# > verification loop (dataset + evaluators + first experiment). This lab *closes* the loop:
# > read the experiment results, localize the failure mode, change **one lever**, re-run the
# > experiment, and compare before/after scores, failure modes, and traces.
#
# > Hands-on module · ~60 min
#
# ```mermaid
# graph LR
#     A[Baseline experiment<br/>Lab 03: dd-scaffold-target] --> B[Read failures<br/>per evaluator]
#     B --> C{Failure mode?}
#     C -->|missing tools / wrong order| D[Prompt or<br/>tool descriptions]
#     C -->|weak evidence| E[Retrieval strategy]
#     C -->|missing fields / vague report| F[Output schema]
#     D --> G[Candidate experiment<br/>same dataset + evaluators]
#     E --> G
#     F --> G
#     G --> H[Compare: scores, failure modes,<br/>traces before vs after]
#     H -->|better| I[Keep change — commit it]
#     H -->|flat / worse| J[Revert, try next lever]
#     J --> B
# ```
#
# Day 1 Lab 04 showed the *ad-hoc* improvement loop (watch one trace → tweak prompt → re-run one
# query). This lab is the **systematic** version: the dataset and evaluators from Lab 03 hold the
# bar steady so every change is *measured*, not vibes.
#
# The four improvement levers, mapped to the failure modes Lab 03's evaluators detect:
#
# | Lever | Changes | Targeted by failure in… |
# |---|---|---|
# | **System prompt** | operating rules, escalation thresholds, field requirements | `required_fields_present`, `escalation_appropriateness` |
# | **Tool descriptions** | when/why to call each tool, argument guidance | `tool_use_expectations`, `trajectory_tool_coverage` |
# | **Retrieval strategy** | query form, `k`, evidence-before-conclusion ordering | `groundedness`, `evidence_quality` |
# | **Output schema** | required fields, evidence-with-verification structure | `report_schema_valid`, `required_fields_present` |
#
# By the end you can:
# - Re-run the Lab 03 baseline experiment and read failures **per evaluator**
# - Make a **targeted** change to one of the four levers — prompt, tool descriptions, retrieval,
#   output schema
# - Run a candidate experiment on the same dataset and compare scores and traces before vs after
# - Promote still-failing rows back into the dataset — failures become regression tests
#
# > 🧭 **Builds on Lab 03; runs standalone.** Lab 03's stub target passed everything by
# > construction; this lab's baseline agent is *deliberately under-specified* so real failures
# > appear, get diagnosed, and get fixed. All vendor data is **synthetic/fictional**.


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
DATA = WORKSHOP / "day2" / "data"

print("workspace:", os.getenv("LANGSMITH_WORKSPACE_ID"), "| tracing:", os.getenv("LANGSMITH_TRACING"))

# %% [markdown]
# ## 1. Hold the bar steady — dataset + evaluators from Lab 03
#
# Everything in this section stays **identical** across the baseline and candidate experiments.
# That's what makes the comparison fair:
#
# | Held constant | Where it comes from |
# |---|---|
# | Dataset | `vendor-due-diligence-eval` — seeded by Lab 03 from `day2/data/due_diligence_eval.json` |
# | Evaluators | Lab 03's three layers — code (`report_schema_valid`, `required_fields_present`, `tool_use_expectations`), LLM judges (`groundedness`, `evidence_quality`, `escalation_appropriateness`), trajectory (`trajectory_tool_coverage`, `trajectory_hitl_pause`) |
# | Model | `get_model()` — same agent model for baseline and candidate; the judge is held fixed on the stronger judge tier |
#
# > **Rule of the loop:** never tune the evaluators and the agent in the same turn. If the bar
# > moves while you're moving the agent, you can't attribute the score change.

# %%
from langsmith import Client

from day1.src.models import get_model

from day1.src.models import scoped

DATASET_NAME = scoped("vendor-due-diligence-eval")  # same scoping as Lab 03

ls_client = Client()


def _normalize_example(ex: dict) -> dict:
    """Bring an example into the evaluator-expected shape.

    Same normalizer as Lab 03 §1 — needed here too because this lab can be opened first. The
    committed `due_diligence_eval.json` uses `inputs.vendor_id` / `expected_outputs.*`, while the
    evaluators read `inputs.question` / `outputs.expected_tools`.
    """
    if "question" in ex.get("inputs", {}):
        return ex
    inp = ex["inputs"]
    exp = ex.get("expected_outputs", ex.get("outputs", {}))
    return {
        "inputs": {
            "question": (
                f"Assess vendor {inp['vendor_id']}"
                # vendor_name lives in expected_outputs in the committed fixture, not in inputs —
                # fall back cleanly rather than rendering an empty "VND-001 ()".
                + (f" ({_name})" if (_name := inp.get("vendor_name") or exp.get("vendor_name")) else "")
                + f" for: {inp['procurement_need']}. "
                f"Priority: {inp.get('priority', 'medium')}, "
                f"Budget: EUR {inp.get('budget_eur', 'N/A')}."
            ),
            "vendor_id": inp["vendor_id"],
        },
        "outputs": {
            "expected_tools": exp.get("expected_tools_called", exp.get("expected_tools", [])),
            "expected_risk_level": exp.get(
                "expected_risk_level",
                "low" if exp.get("suitability") == "high" else ("high" if exp.get("should_escalate") else "medium"),
            ),
            "should_escalate": exp.get("should_escalate", exp.get("requires_human_review", False)),
            "reference_points": ex.get("reference_points", []),
        },
    }


if ls_client.has_dataset(dataset_name=DATASET_NAME):
    EVAL_EXAMPLES = [
        _normalize_example({"inputs": ex.inputs, "outputs": ex.outputs})
        for ex in ls_client.list_examples(dataset_name=DATASET_NAME)
    ]
    print(f"using existing dataset {DATASET_NAME!r} with {len(EVAL_EXAMPLES)} example(s) — seeded by Lab 03")
else:
    # Lab 03 not run yet — fall back to the committed JSON so this lab still runs standalone.
    eval_file = DATA / "due_diligence_eval.json"
    EVAL_EXAMPLES = [_normalize_example(ex) for ex in json.loads(eval_file.read_text())]
    dataset = ls_client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Vendor due diligence eval set — see day2/data/due_diligence_eval.json (seeded from Lab 04 fallback).",
    )
    ls_client.create_examples(
        dataset_id=dataset.id,
        examples=[{"inputs": ex["inputs"], "outputs": ex["outputs"]} for ex in EVAL_EXAMPLES],
    )
    print(f"⚠️  Lab 03 dataset missing — seeded {len(EVAL_EXAMPLES)} example(s) from {eval_file.name}")

# The dataset is shared with Lab 05, which adds reviewer-corrected examples in a different shape.
# Read defensively so a human-review round trip can't break the experiment loop.
for i, ex in enumerate(EVAL_EXAMPLES, 1):
    out = ex.get("outputs") or {}
    print(f"\n--- example {i} ---")
    print("Q:", (ex.get("inputs") or {}).get("question", "(no question field)")[:100])
    if "expected_tools" in out:
        print("expected tools:", out["expected_tools"])
        print("expected risk:", out.get("expected_risk_level", "?"),
              "| escalate:", out.get("should_escalate", "?"))
    else:
        print("reviewer-corrected example (no expected_tools) — graded by the judges only")

# %% [markdown]
# Re-declare Lab 03's evaluators verbatim. They are copied here (not imported) so this lab runs
# standalone — but they must stay byte-for-byte equivalent in behaviour. If you change one here,
# change it in Lab 03 too and re-baseline.

# %%
from day1.src.models import JUDGE_MODEL, get_judge_model

judge_model = get_judge_model()  # same judge tier as Lab 03 — hold the grader fixed while the agent moves

REQUIRED_REPORT_FIELDS = [
    "vendor_summary",
    "evidence",
    "risk_signals",
    "risk_level",
    "suitability",
    "confidence",
    "follow_up_questions",
    "requires_human_review",
    "source_validation_status",
]


def report_schema_valid(outputs: dict, **_) -> dict:
    """Pass when the report is a structured dict (not free text / parse failure)."""
    report = outputs.get("report")
    if not isinstance(report, dict):
        return {"key": "report_schema_valid", "score": 0,
                "comment": f"report is {type(report).__name__}, expected structured dict"}
    return {"key": "report_schema_valid", "score": 1}


def required_fields_present(outputs: dict, **_) -> dict:
    """Pass when every AGENTS.md-required report field exists and is non-empty."""
    report = outputs.get("report")
    if not isinstance(report, dict):
        return {"key": "required_fields_present", "score": 0, "comment": "no structured report"}
    missing = [f for f in REQUIRED_REPORT_FIELDS if f not in report or report[f] is None]
    if missing:
        return {"key": "required_fields_present", "score": 0,
                "comment": f"missing fields: {missing}"}
    return {"key": "required_fields_present", "score": 1}


def tool_use_expectations(outputs: dict, reference_outputs: dict, **_) -> dict:
    """Pass when the agent called every tool the scenario expects (order-independent)."""
    called = {c["name"] for c in outputs.get("trajectory", [])}
    expected = set(reference_outputs.get("expected_tools", []))
    missing = sorted(expected - called)
    if missing:
        return {"key": "tool_use_expectations", "score": 0,
                "comment": f"expected tools not called: {missing}; called: {sorted(called)}"}
    return {"key": "tool_use_expectations", "score": 1}


JUDGE_PROMPTS = {
    "groundedness": """You are grading a vendor due diligence report for GROUNDEDNESS.

The agent's operating rules require: every assessment claim must cite a source (vendor KB,
vendor database, or web research) — no unsupported claims.

Report to grade:
{report}

Reference points the report must cover:
{reference_points}

Grade PASS (1) if every material claim in the report cites a source AND the report covers the
reference points without inventing facts. Grade FAIL (0) if any material claim is unsourced or
contradicts the evidence.

Respond with ONLY a JSON object: {{"score": 0 or 1, "reason": "<one sentence>"}}""",

    "evidence_quality": """You are grading a vendor due diligence report for EVIDENCE QUALITY.

Good evidence: each item has a named source, a verification status (verified / partially
verified / unverified), and unverified claims are flagged for follow-up rather than asserted.

Report to grade:
{report}

Reference points the report must cover:
{reference_points}

Grade PASS (1) if evidence items carry source + verification status, and unverified or partial
claims produce follow-up questions. Grade FAIL (0) if evidence is unsourced, unverified claims
are asserted as fact, or no follow-ups exist despite gaps.

Respond with ONLY a JSON object: {{"score": 0 or 1, "reason": "<one sentence>"}}""",

    "escalation_appropriateness": """You are grading a vendor due diligence report for ESCALATION
APPROPRIATENESS.

Escalation rules (from the agent's operating instructions): flag for human review when
- confidence is low, OR
- vendor risk is medium or high, OR
- source validation is incomplete, OR
- the procurement need is sensitive (high priority, high value).

Report to grade:
{report}

For this scenario, escalation was REQUIRED = {should_escalate}. The reference risk level is
{expected_risk_level}.

Grade PASS (1) if the report's requires_human_review flag matches the required behaviour AND
its risk_level is consistent with the reference. Grade FAIL (0) if it under-escalates a risky
case or over-escalates a clean one.

Respond with ONLY a JSON object: {{"score": 0 or 1, "reason": "<one sentence>"}}""",
}


def _grade(prompt: str) -> dict:
    """Run one judge prompt, parse the JSON verdict. Parse failure = score 0 (fail closed)."""
    raw = judge_model.invoke(prompt).content
    try:
        return json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
    except (json.JSONDecodeError, AttributeError):
        return {"score": 0, "reason": f"judge returned unparseable verdict: {str(raw)[:120]}"}


def _make_judge(name: str):
    """Build an evaluator function bound to one judge prompt."""
    def judge(outputs: dict, reference_outputs: dict, **_) -> dict:
        report = outputs.get("report")
        verdict = _grade(JUDGE_PROMPTS[name].format(
            report=json.dumps(report, indent=2, default=str),
            reference_points="\n".join(f"- {p}" for p in reference_outputs.get("reference_points", [])),
            should_escalate=reference_outputs.get("should_escalate"),
            expected_risk_level=reference_outputs.get("expected_risk_level"),
        ))
        return {"key": name, "score": int(verdict.get("score", 0)),
                "comment": verdict.get("reason", "")}
    judge.__name__ = name
    return judge


groundedness = _make_judge("groundedness")
evidence_quality = _make_judge("evidence_quality")
escalation_appropriateness = _make_judge("escalation_appropriateness")

EVIDENCE_TOOLS = {"search_vendor_kb", "screen_vendor", "tavily_search", "parse_vendor_pdf"}


def trajectory_tool_coverage(outputs: dict, reference_outputs: dict, **_) -> dict:
    """Expected tools present, and at least one evidence tool used before concluding."""
    trajectory = outputs.get("trajectory", [])
    called = [c["name"] for c in trajectory]
    expected = set(reference_outputs.get("expected_tools", []))

    missing = sorted(expected - set(called))
    used_evidence = [t for t in called if t in EVIDENCE_TOOLS]
    if missing:
        return {"key": "trajectory_tool_coverage", "score": 0,
                "comment": f"missing expected tools: {missing}"}
    if not used_evidence:
        return {"key": "trajectory_tool_coverage", "score": 0,
                "comment": "no evidence-gathering tool used — conclusion is unsupported"}
    return {"key": "trajectory_tool_coverage", "score": 1,
            "comment": f"evidence tools used: {used_evidence}"}


def trajectory_hitl_pause(outputs: dict, reference_outputs: dict, **_) -> dict:
    """Sensitive/risky scenarios must pause for human review; clean ones must not block."""
    required = bool(reference_outputs.get("should_escalate", False))
    paused = bool(outputs.get("paused_for_hitl", False))
    interrupted = any(
        c.get("name") in {"human_review", "request_human_approval"}
        for c in outputs.get("trajectory", [])
    )
    paused = paused or interrupted

    report = outputs.get("report") or {}
    flagged = isinstance(report, dict) and bool(report.get("requires_human_review"))

    if required and not (paused or flagged):
        return {"key": "trajectory_hitl_pause", "score": 0,
                "comment": "scenario required escalation but the run never paused or flagged for review"}
    if not required and paused:
        return {"key": "trajectory_hitl_pause", "score": 0,
                "comment": "clean scenario blocked on unnecessary human review"}
    return {"key": "trajectory_hitl_pause", "score": 1}


code_evaluators = [report_schema_valid, required_fields_present, tool_use_expectations]
llm_judges = [groundedness, evidence_quality, escalation_appropriateness]
trajectory_evaluators = [trajectory_tool_coverage, trajectory_hitl_pause]
ALL_EVALUATORS = code_evaluators + llm_judges + trajectory_evaluators
print(f"{len(ALL_EVALUATORS)} evaluators held constant:", [e.__name__ for e in ALL_EVALUATORS])

# %% [markdown]
# ## 2. Baseline agent — deliberately under-specified
#
# Lab 03's stub passed every evaluator by construction, which is right for a smoke test but
# teaches nothing about improvement. This baseline is a **real agent on the real tools** with a
# thin prompt and free-text output — the kind of v1 you ship before evals exist. Expect failures.
#
# The target function keeps Lab 03's contract: it returns
# `{"report": ..., "trajectory": [...], "paused_for_hitl": ...}` so every Lab 03 evaluator works
# unchanged. Because the baseline answers in free text, `report` is a string here —
# `report_schema_valid` should fail. **That failure is the first improvement backlog item.**

# %%
from langchain.agents import create_agent

from day1.src.vendor_discovery_graph import vendor_tools

model = get_model()

BASELINE_PROMPT = (
    "You are a vendor due diligence assistant for Acme procurement teams. "
    "Assess the vendor named in the request and say whether they are suitable. "
    "Use the tools available to look things up."
)

baseline_agent = create_agent(
    model=model,
    tools=vendor_tools(),
    system_prompt=BASELINE_PROMPT,
    name="dd_agent_baseline",
)


def _tool_calls_from_messages(messages) -> list[dict]:
    """Flatten tool calls from a LangGraph messages list into {name, args} dicts (Lab 03 helper)."""
    calls = []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            calls.append({"name": tc.get("name"), "args": tc.get("args", {})})
    return calls


def _make_target(agent) -> "callable":
    """Wrap an agent in Lab 03's target-function contract."""
    def target(inputs: dict) -> dict:
        result = agent.invoke({"messages": [{"role": "user", "content": inputs["question"]}]})
        structured = result.get("structured_response")
        report = structured.model_dump() if structured is not None else result["messages"][-1].content
        return {
            "report": report,
            "trajectory": _tool_calls_from_messages(result["messages"]),
            "paused_for_hitl": bool(result.get("__interrupt__")),
        }
    return target


baseline_target = _make_target(baseline_agent)

# %% [markdown]
# ## 3. Experiment 0 — baseline scores
#
# Run the under-specified agent against the same dataset + evaluators Lab 03 used. Note the
# `experiment_prefix` — `dd-agent-baseline` — so the comparison against Lab 03's stub run
# (`dd-scaffold-target`) and this lab's candidate (`dd-agent-v2-grounded`) stays clean in the LangSmith UI.

# %%
from langsmith.evaluation import evaluate

baseline_experiment = evaluate(
    baseline_target,
    data=DATASET_NAME,
    evaluators=ALL_EVALUATORS,
    experiment_prefix=scoped("dd-agent-baseline"),
    description="Under-specified v1 agent — thin prompt, free-text output, default tool docs.",
    metadata={
        "variant": "baseline",
        "prompt": "thin (3 sentences)",
        "output": "free text",
        "judge_model": JUDGE_MODEL,  # the grader, not the agent — LABS_MODEL is a different knob
        "data_file": "day2/data/due_diligence_eval.json",
    },
    max_concurrency=2,
)

# %%
print("baseline experiment:", baseline_experiment.experiment_name)
df_baseline = baseline_experiment.to_pandas()
feedback_cols = [c for c in df_baseline.columns if c.startswith("feedback.")]
print(df_baseline[["inputs.question"] + feedback_cols].to_string())

# %% [markdown]
# ## 4. Read the failures — localize the mode
#
# Group the red cells by evaluator and map each to its lever (same table as the header, now with
# what you'll actually see):
#
# | Failing evaluator | Likely failure in the trace | Lever |
# |---|---|---|
# | `report_schema_valid` | `report` is a string, not a dict | **Output schema** |
# | `required_fields_present` | report missing `source_validation_status`, `risk_signals`, … | **Output schema** or **prompt** |
# | `tool_use_expectations` / `trajectory_tool_coverage` | `get_risk_criteria` or `screen_vendor` never called | **Tool descriptions** (tool doesn't exist in descriptions the model reads) or **prompt** |
# | `groundedness` / `evidence_quality` | claims without sources; verification status absent | **Retrieval strategy** then **prompt** |
# | `escalation_appropriateness` / `trajectory_hitl_pause` | medium/high-risk scenario not flagged for review | **Prompt** (escalation rules) |
#
# > **Exercise:** before scrolling on, write down the failure mode you expect to dominate and the
# > single lever you'd pull first. Then check Sections 5–8 against your prediction.
# >
# > Change **one lever per experiment turn** in real work. Sections 5–8 show a candidate change
# > for *each* lever for teaching; Section 9 runs one combined v2. When you iterate for real,
# > pick the lever your worst failure mode points to and leave the others at baseline.

# %% [markdown]
# ## 5. Lever 1 — system prompt
#
# The baseline prompt says *"assess the vendor and say whether they are suitable"* — nothing about
# evidence, sources, verification, escalation, or the required report shape. The v2 prompt pulls
# the operating rules straight from `day2/data/agent/AGENTS.md`, which Lab 03's evaluators were
# built against. This is the cheapest lever and the first one to pull: prompt edits are additive,
# specific, and reversible.

# %%
AGENTS_MD = (DATA / "agent" / "AGENTS.md").read_text()

PROMPT_V2 = f"""You are a vendor due diligence agent for Acme procurement teams.

Your operating instructions follow verbatim — every report must satisfy them:

<operating_instructions>
{AGENTS_MD}
</operating_instructions>

Additional requirements for this evaluation harness:
- Always call `get_risk_criteria` before assigning a risk level, and cite it.
- Always call `screen_vendor` for the vendor before concluding — even when you
  expect no history. State explicitly when none exists.
- Gather evidence BEFORE assessing: `search_vendor_kb` (and `get_vendor`) results must be in
  hand before you state any capability or certification claim.
- Set `requires_human_review=true` whenever confidence is low, risk is medium or high, source
  validation is incomplete, or the procurement need is sensitive."""

print(PROMPT_V2[:600], "\n...\n")

# %% [markdown]
# ## 6. Lever 2 — tool descriptions
#
# The model picks tools from their **descriptions**, not from your prompt. The Day 1 tools were
# written for vendor *discovery*, so their docstrings say nothing about due diligence — the model
# has no signal that `get_vendor` should precede a risk assessment or that `search_vendor_kb` is
# the evidence source for certification claims.
#
# The v2 tools below keep the **same function bodies** and only change the descriptions — that's
# the point of the lever. Two copies get sharpened; the rest pass through unchanged.

# %%
from langchain_core.tools import tool

from day1.src import vendor_discovery_graph as vdg


@tool("search_vendor_kb")  # keep the Day 1 tool name — trajectory evaluators expect it
def search_vendor_kb_v2(query: str) -> str:
    """Search vendor profile pages for capability and certification EVIDENCE.

    This is your primary evidence source for due diligence claims — every capability or
    certification statement in your report must trace back to a snippet from this tool
    (cite the [source: ...] filename it returns). QUERY TIP: search for the capability or
    certification keyword (e.g. "QS avionics certification"), not the full user question;
    run multiple focused searches for multi-part requests.
    """
    return vdg.search_vendor_kb.invoke({"query": query})


@tool("get_vendor")  # keep the Day 1 tool name — trajectory evaluators expect it
def get_vendor_v2(vendor: str) -> str:
    """Get a single vendor's full structured record by vendor ID (e.g. VND-001) or name.

    Call this FIRST in any vendor assessment — the record carries risk level, compliance flags,
    certifications, and contract history that frame everything else. Cite it as
    [source: vendor database].
    """
    return vdg.get_vendor.invoke({"vendor": vendor})


@tool
def get_risk_criteria() -> str:
    """Get the risk criteria and scoring rubric for vendor due diligence.

    Call this before assigning a risk level — it defines critical/important/optional criteria
    for compliance, financial, operational, and security categories, plus the risk-level
    thresholds (low/medium/high/blocked) and required actions for each.
    """
    rc_path = DATA / "risk_criteria.json"  # DATA is resolved in the setup cell
    if rc_path.exists():
        return rc_path.read_text()
    return "Risk criteria file not found. Assess based on available evidence and flag missing criteria."


def make_tools_v2():
    """Day 1 tools with due-diligence-aware descriptions on the two evidence lookups."""
    return [
        search_vendor_kb_v2 if t.name == "search_vendor_kb"
        else get_vendor_v2 if t.name == "get_vendor"
        else t
        for t in vdg.vendor_tools()
    ] + [get_risk_criteria]


tools_v2 = make_tools_v2()
print("v2 tools:", [t.name for t in tools_v2])

# %% [markdown]
# ## 7. Lever 3 — retrieval strategy
#
# If `groundedness` / `evidence_quality` fail while `tool_use_expectations` passes, the agent
# called the right tools but worked from weak evidence — a retrieval problem, not a reasoning
# problem. Three cheap retrieval levers:
#
# | Lever | Change | When it helps |
# |---|---|---|
# | Query form | capability keywords, not raw question | raw questions embed poorly against profile pages |
# | `k` | more chunks per query | answers missing vendors that ARE in the KB |
# | Ordering | evidence tools before any conclusion | claims asserted before retrieval happened |
#
# Section 6's `search_vendor_kb_v2` description already pulls the query-form lever. Here we pull
# the second: bump `k` from 4 → 6 so more of the profile survives chunking. (Chunk size lives in
# the index builder — changing it means rebuilding the index, so treat it as a separate turn.)

# %%
@tool("search_vendor_kb")  # keep the Day 1 tool name — trajectory evaluators expect it
def search_vendor_kb_v2_wide(query: str) -> str:
    """Search vendor profile pages for capability and certification EVIDENCE.

    This is your primary evidence source for due diligence claims — every capability or
    certification statement in your report must trace back to a snippet from this tool
    (cite the [source: ...] filename it returns). QUERY TIP: search for the capability or
    certification keyword (e.g. "QS avionics certification"), not the full user question;
    run multiple focused searches for multi-part requests.
    """
    # `search_vendor_kb` is a @tool, so go through the tool boundary rather than the module's
    # private index builder — the retrieval knob we are changing is k, not the index.
    hits = vdg.vendor_kb_index().similarity_search(query, k=6)  # baseline: k=4
    if not hits:
        return "No relevant vendor profiles found. Try a broader capability keyword."
    return "\n\n---\n\n".join(f"[source: {h.metadata['source']}]\n{h.page_content}" for h in hits)


# Swap the wider retriever in under the stable tool name.
tools_v2 = [
    search_vendor_kb_v2_wide if t.name == "search_vendor_kb" else t
    for t in tools_v2
]
print("v2 retrieval: search_vendor_kb k=4 → 6 (query guidance in description)")
print("v2 tools:", [t.name for t in tools_v2])

# %% [markdown]
# ## 8. Lever 4 — output schema
#
# The baseline returns free text, so `report_schema_valid` and `required_fields_present` fail on
# every row. The fix: `response_format=` with a Pydantic model that *is* the AGENTS.md report
# shape. Structured output turns "please include these fields" (a prompt hope) into a guarantee
# the framework enforces — and it makes `required_fields_present` deterministic green.
#
# The schema also sharpens `evidence`: each item must carry `source` and `verification`, which is
# exactly what `evidence_quality` and `groundedness` grade. That's schema design driven by the
# evaluators, not by taste.

# %%
from typing import Literal

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    claim: str = Field(description="One factual claim about the vendor")
    source: str = Field(description="Where the claim came from: vendor KB filename, "
                                    "'vendor database', or web search")
    verification: Literal["verified", "partially verified", "unverified"] = Field(
        description="Verification status — unverified claims must NOT be asserted as fact"
    )


class RiskSignal(BaseModel):
    signal: str = Field(description="Specific risk signal, e.g. 'Expired critical certification'")
    severity: Literal["low", "medium", "high"]


class DueDiligenceReport(BaseModel):
    """The AGENTS.md report contract, enforced by response_format=."""

    vendor_summary: str = Field(description="One-paragraph vendor summary with vendor ID")
    evidence: list[EvidenceItem] = Field(description="All claims backing the assessment")
    risk_signals: list[RiskSignal] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = Field(
        description="Overall risk classification per get_risk_criteria"
    )
    suitability: Literal["high", "medium", "low"]
    confidence: Literal["high", "medium", "low"] = Field(
        description="'low' requires requires_human_review=True"
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Required when any evidence is unverified or partially verified"
    )
    requires_human_review: bool = Field(
        description="True when confidence is low, risk is medium+, source validation is "
                    "incomplete, or the need is sensitive"
    )
    source_validation_status: Literal["complete", "partial", "incomplete"]


# %% [markdown]
# ## 9. Experiment 1 — candidate agent (v2)
#
# All four levers applied at once (prompt + tool descriptions + retrieval + schema) — for the
# teaching narrative. The evaluators, dataset, model, and target contract are unchanged, so any
# score delta is attributable to the levers.
#
# > In your own iteration, prefer **one lever per experiment**. The metadata block records what
# > changed so a future reader can explain the delta without diffing code.

# %%
candidate_agent = create_agent(
    model=model,
    tools=tools_v2,
    system_prompt=PROMPT_V2,
    response_format=DueDiligenceReport,
    name="dd_agent_v2",
)
candidate_target = _make_target(candidate_agent)

# %%
candidate_experiment = evaluate(
    candidate_target,
    data=DATASET_NAME,
    evaluators=ALL_EVALUATORS,
    experiment_prefix=scoped("dd-agent-v2-grounded"),
    description="v2: AGENTS.md prompt + DD tool descriptions + wider KB retrieval + structured report schema.",
    metadata={
        "variant": "v2-all-levers",
        "lever_prompt": "AGENTS.md rules + evidence-first ordering + escalation thresholds",
        "lever_tool_descriptions": "search_vendor_kb_v2, get_vendor_v2",
        "lever_retrieval": "search_vendor_kb k=4→6 + keyword-query guidance",
        "lever_output_schema": "DueDiligenceReport (response_format=)",
        "judge_model": JUDGE_MODEL,  # the grader, not the agent — LABS_MODEL is a different knob
        "data_file": "day2/data/due_diligence_eval.json",
    },
    max_concurrency=2,
)

# %%
print("candidate experiment:", candidate_experiment.experiment_name)
df_candidate = candidate_experiment.to_pandas()
# Recompute against THIS frame rather than reusing the baseline's column list. An evaluator that
# errored on one run (a judge timeout, a gateway policy rejection) simply produces no column for it,
# and indexing the candidate with the baseline's names would raise KeyError instead of showing you
# the scores that did land.
cand_cols = [c for c in df_candidate.columns if c.startswith("feedback.")]
if missing := [c for c in feedback_cols if c not in cand_cols]:
    print(f"⚠️  no candidate scores for: {', '.join(c.removeprefix('feedback.') for c in missing)} "
          f"— that evaluator errored on this run; compare only the columns below.\n")
print(df_candidate[["inputs.question"] + cand_cols].to_string())

# %% [markdown]
# ## 10. Compare — before vs after
#
# Now the payoff: put the two experiments side by side. Three comparisons matter, in order:
#
# 1. **Score deltas per evaluator** — the table below pivots both experiments by example and
#    evaluator. A real improvement moves the evaluator that matches the failure mode you targeted.
# 2. **Failure-mode changes** — failures should *change shape*: baseline fails
#    `report_schema_valid` everywhere (free text); candidate should pass schema/fields and any
#    remaining red should be in the judgement calls (`groundedness`, `evidence_quality`) — a
#    smaller, more interesting problem.
# 3. **Trace diffs** — in the LangSmith UI, open the same dataset row in both experiments and
#    compare: which tools got called, in what order, and is the structured report fully populated
#    (vs. empty placeholder fields)?
#
# | Outcome | Meaning | Next action |
# |---|---|---|
# | Target evaluators up, others flat | the lever worked | keep the change, commit it, next failure mode |
# | Target evaluator flat | the lever missed the failure mode | re-read failing traces; pick a different lever |
# | Target up, another down | trade-off (e.g. strict schema hurt evidence prose) | inspect regressed rows; tune the change |
# | Everything moved | more than one thing changed, or judge variance | re-run both; change one lever per turn |
#
# > **Judge variance note:** the LLM judges (`groundedness`, `evidence_quality`,
# > `escalation_appropriateness`) have run-to-run variance. On a 3-example dataset, a one-row flip
# > can be noise — re-run with `num_repetitions=2` before declaring victory.

# %%
import pandas as pd  # noqa: F401 — langsmith's experiment.to_pandas() imports it; see pyproject note

_KEY = "inputs.question"


def _score_pivot(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("feedback.")]
    out = df[[_KEY] + cols].copy()
    out.columns = [_KEY] + [c.removeprefix("feedback.") for c in cols]
    out[_KEY] = out[_KEY].str.slice(0, 42)
    return out.set_index(_KEY)


pv_base = _score_pivot(df_baseline)
pv_cand = _score_pivot(df_candidate)

comparison = pv_base.join(pv_cand, lsuffix=" · baseline", rsuffix=" · v2")
print(comparison.to_string())

# %%
# Which evaluators actually moved? Mean score per evaluator, baseline vs v2.
means = pd.DataFrame({
    "baseline": pv_base.mean(numeric_only=True),
    "v2": pv_cand.mean(numeric_only=True),
})
means["delta"] = means["v2"] - means["baseline"]
print(means.round(2).to_string())

moved_up = means.index[means["delta"] > 0].tolist()
moved_down = means.index[means["delta"] < 0].tolist()
print("\nimproved:", moved_up or "none")
print("regressed:", moved_down or "none")

# %% [markdown]
# ### Why `groundedness` and `evidence_quality` don't move
#
# Expect those two to stay red after the v2 change, and that is the most useful row in the table.
# The v2 agent **does** cite a source and a verification status on every item in its `evidence`
# list — the schema change forced that. But its *narrative* sections (`vendor_summary`,
# `risk_signals`) still assert things like "established supplier" and "limited recent Acme contract
# history" with no citation, and the `groundedness` judge grades **every material claim**, not just
# the ones inside the evidence array.
#
# So the three levers this lab pulled — prompt, tool descriptions, output schema — cannot fix it.
# The fix is a *structural* one: require the narrative fields to reference evidence-item indices, or
# drop free-text narrative entirely. That is exactly the kind of finding an experiment pair is for:
# the score tells you a lever exists, the judge's **comment** tells you which one.
#
# > 💡 This is also why the judge runs on a stronger model than the agent (see `day1/src/models.py`).
# > A weaker judge on this same report scores it PASS — "all material claims are properly sourced" —
# > and you never learn the narrative sections are ungrounded.

# %% [markdown]
# ## 11. Close the loop — failures become regression tests
#
# The loop isn't closed until the remaining failures are captured:
#
# - **Keep the winning change where it lives** — prompt edits to Prompt Hub / `AGENTS.md`,
#   tool-description and retrieval changes to the graph module, schema changes to the
#   `response_format=` call. The experiment metadata records which lever moved which score.
# - **Promote still-failing rows into the dataset** — any row that's red in v2 is a ready-made
#   regression test. Copy the example, tighten its `reference_points` to what the fixed agent
#   *should* have said, and re-seed. The next iteration must clear a higher bar.
# - **Record the experiment pair** — baseline name, candidate name, lever changed, per-evaluator
#   deltas. That record is the audit trail for *why* the agent changed (Day 3 governance picks
#   this up).
#
# ```mermaid
# graph LR
#     F[Red cells in v2] -->|tighten reference_points| D[Bigger, harder eval dataset]
#     D --> E[Next experiment pair]
#     E -->|winning lever| P[Prompt Hub / AGENTS.md / graph code]
#     P --> E
# ```

# %%
# Example: promote a v2 failure back into the dataset as a sharper regression test.
# (Dry-run — inspect before upserting in real work.)
still_failing = means.index[means["v2"] < 1.0].tolist()
print("evaluators still below 1.0 in v2:", still_failing or "none — all green")
print("\nFor each red row: copy the example, sharpen reference_points to the behaviour you")
print("expected, and re-seed with Lab 03's upsert cell. The bar only goes up.")

# Read the judges' *comments*, not just the scores — a stuck score is a lever you have not pulled.
# `ExperimentResults` is iterable; each item carries its EvaluationResults for that example.
print("\nWhy the stuck rows are stuck (one judge comment per red evaluator):")
_seen: set[str] = set()
for _row in candidate_experiment:
    for _res in (_row.get("evaluation_results") or {}).get("results", []):
        if _res.key in still_failing and _res.key not in _seen and _res.comment:
            _seen.add(_res.key)
            print(f"  {_res.key}: {_res.comment[:160]}")
if not _seen:
    print("  (no comments captured — open the experiment in LangSmith and read the feedback column)")

# %% [markdown]
# ## 12. Recap & next
#
# | Step | What you did |
# |---|---|
# | Hold the bar | Reused Lab 03's dataset + 8 evaluators unchanged |
# | Baseline | Ran a deliberately under-specified agent → real failures |
# | Diagnose | Mapped each failing evaluator to one of four levers |
# | Levers | Saw a targeted change for prompt, tool descriptions, retrieval strategy, output schema |
# | Candidate | Ran v2 on the same dataset + evaluators |
# | Compare | Per-evaluator score deltas, failure-mode changes, trace diffs |
# | Close the loop | Promoted remaining failures toward the dataset |
#
# **Next:** `05_human_review.py` — the annotation queue walkthrough: how reviewed production
# traces become new dataset rows, so the eval set grows from *real* failures, not just
# hand-written ones.
