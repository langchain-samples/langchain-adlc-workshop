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
# # 04 · Observability — Monitoring, Insights + Production Signals
#
# **Workshop:** LangChain ADLC Workshop · **Day 3** · **ADLC stage:** Monitor / Operate
#
# > **Loop Engineering focus: Hill-climbing loop** — the agent is deployed (Lab 01) and evaluated
# > continuously (Lab 03). This lab builds the **operating picture**: traces from deployed runs,
# > monitoring dashboards (latency, errors, cost, tool failures), quality signals, user feedback,
# > and drift — then turns one of those signals into the next improvement cycle.
#
# > Hands-on module · ~30 min
#
# ```mermaid
# graph LR
#     A[Deployed ticket agent<br/>Lab 01] --> B[Production traces<br/>LangSmith project]
#     B --> C1[Latency]
#     B --> C2[Errors]
#     B --> C3[Cost / tokens]
#     B --> C4[Tool failures]
#     B --> C5[Quality signals]
#     B --> C6[User feedback]
#     B --> C7[Drift]
#     C1 --> D[Monitoring dashboard<br/>this lab]
#     C2 --> D
#     C3 --> D
#     C4 --> D
#     C5 --> D
#     C6 --> D
#     C7 --> D
#     D -->|one signal| E[Improvement cycle<br/>prompt / tools / retrieval]
#     E -->|redeploy + re-evaluate| A
# ```
#
# Labs 01–02 put the ticket agent into production and attached online evals. This lab answers:
# *"How is the agent actually doing out there?"* — with numbers, not vibes.
#
# The ticket agent resolves customer support tickets (MFA resets, account lockouts, VPN issues)
# by searching a knowledge base, looking up ticket history, checking user permissions, and —
# for sensitive actions — pausing for human approval. Every run is traced to a LangSmith project.
#
# By the end you can:
# - **Inspect traces** from deployed runs — filter by time, status, metadata
# - **Read monitoring dashboards** — latency percentiles, error rates, token/cost usage, tool
#   failure rates, and drift indicators
# - **Identify one production signal** that should drive the next improvement cycle
# - **Review quality signals** — online eval scores and user feedback patterns
#
# > 🧭 **Builds on Labs 01–02; runs standalone if traces exist.** The cells below query the
# > LangSmith project where deployed runs are traced. If no runs exist yet, the lab prints a
# > diagnostic and suggests running Lab 01 first. All ticket data is **synthetic/fictional**.


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
# Same setup cell as the other labs — loads `.env`, adds the workshop root to the path, prints
# the LangSmith workspace. Observability only needs the **LangSmith SDK** (`langsmith.Client`);
# no model calls happen in this lab.

# %%
import os
import warnings
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# No `override=True`: real environment variables must win over `.env`, otherwise the
# `export LANGSMITH_GATEWAY=true / export LANGSMITH_GATEWAY_API_KEY=...` gateway setup documented in
# the README (Model access section) and .env.example is silently clobbered by whatever `.env` happens to contain.
# `.env` fills the gaps; your shell wins.
load_dotenv(find_dotenv())

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

client = Client()

print("workspace:", os.getenv("LANGSMITH_WORKSPACE_ID"), "| tracing:", os.getenv("LANGSMITH_TRACING"))

# %% [markdown]
# ## 1. Inspect traces from deployed runs
#
# 📖 [Deployments / CLI](https://docs.langchain.com/langsmith/cli)
#
# The deployed ticket agent from Lab 01 sends every run to a LangSmith project. The `Client`
# gives you programmatic access to the same traces you see in the UI — filterable by time,
# status, metadata, and more.
#
# Key `list_runs` filters used below:
#
# | Filter | What it does |
# |---|---|
# | `project_name` | The LangSmith project where deployed runs land |
# | `is_root=True` | Only top-level runs (not child LLM/tool calls) |
# | `start_time` | Only runs after this timestamp |
# | `error=True/False` | Filter by error status |
# | `limit` | Cap results — **max 100**; a larger value is rejected with a 400 |
#
# > 💡 **In the UI:** the same filters are available in the LangSmith project view —
# > *Traces → Filters → Add filter*. The SDK and UI read the same data.

# %%
# Read the *deployment's* project — same key and default as Lab 01 §6 and Lab 03. Deliberately NOT
# falling through to LANGSMITH_PROJECT: that variable is always set for the labs, so a fallback chain
# ending there silently computes "production" latency, error rate and cost over Day 1–2 lab traces.
# Numbers that look real and mean nothing are worse than an empty dashboard.
PROJECT = os.getenv("OBSERVABILITY_PROJECT") or os.getenv(
    "LANGSMITH_DEPLOYMENT_PROJECT", "acme-ticket-deployment"
)


def _has_runs(project: str) -> bool:
    try:
        return bool(next(iter(client.list_runs(project_name=project, is_root=True, limit=1)), None))
    except Exception:
        return False


IS_REAL_DEPLOYMENT = _has_runs(PROJECT)
if not IS_REAL_DEPLOYMENT:
    _fallback = os.getenv("LANGSMITH_PROJECT")
    if _fallback and _has_runs(_fallback):
        print(f"⚠ No runs in deployment project {PROJECT!r} — falling back to {_fallback!r} so the")
        print("  mechanics below still run. These are LAB traces, not production traffic: treat every")
        print("  number as a demonstration of the *method*, not a real signal. Run Lab 01 against a")
        print("  deployment (or set OBSERVABILITY_PROJECT) for figures worth acting on.")
        PROJECT = _fallback
    else:
        print(f"⚠ No runs in {PROJECT!r} — run Lab 01 first to generate deployment traces.")

print("observability project:", PROJECT, "(deployment)" if IS_REAL_DEPLOYMENT else "(lab traces — see warning)")

# ── Recent runs: last 24 hours ──────────────────────────────────────────────
now = datetime.now(timezone.utc)
yesterday = now - timedelta(hours=24)

recent_runs = list(client.list_runs(
    project_name=PROJECT,
    is_root=True,
    start_time=yesterday,
    limit=50,
))

if not recent_runs:
    # Widen the window — maybe the deployment ran earlier
    recent_runs = list(client.list_runs(
        project_name=PROJECT,
        is_root=True,
        limit=50,
    ))

print(f"project: {PROJECT!r}")
print(f"recent root runs: {len(recent_runs)}")

if not recent_runs:
    print(f"\n⚠ No runs found in project {PROJECT!r}.")
    print("  Run Lab 01 (deployments) first to generate production traces,")
    print("  or set LANGSMITH_PROJECT to a project with existing runs.")
else:
    print(f"\n{'run ID':<40} {'status':<10} {'latency':>8} {'time'}")
    print("─" * 90)
    for r in recent_runs[:10]:
        latency = f"{r.latency:.1f}s" if r.latency else "—"
        status = "❌ error" if r.error else "✅ ok"
        ts = r.start_time.strftime("%H:%M:%S") if r.start_time else "—"
        print(f"{str(r.id):<40} {status:<10} {latency:>8} {ts}")
    if len(recent_runs) > 10:
        print(f"  … and {len(recent_runs) - 10} more")

# %% [markdown]
# ### Drill into one trace
#
# Pick the most recent run and inspect its structure — inputs, outputs, child runs (tool calls,
# LLM calls), and metadata. This is the same view as clicking a trace in the LangSmith UI.

# %%
if recent_runs:
    run = recent_runs[0]
    print(f"run: {run.id}")
    print(f"  name:      {run.name}")
    print(f"  status:    {'error' if run.error else 'ok'}")
    print(f"  started:   {run.start_time}")
    print(f"  latency:   {run.latency:.2f}s" if run.latency else "  latency:   —")
    print(f"  tokens:    {run.total_tokens}" if run.total_tokens else "  tokens:    —")

    # Inputs — the user's question
    if isinstance(run.inputs, dict):
        msgs = run.inputs.get("messages", [])
        if msgs and isinstance(msgs[0], dict):
            print(f"\n  input:  {msgs[0].get('content', '')[:120]}")

    # Outputs — the agent's final answer
    if isinstance(run.outputs, dict):
        msgs = run.outputs.get("messages", [])
        if msgs and isinstance(msgs[-1], dict):
            content = msgs[-1].get("content", "")
            if isinstance(content, str):
                print(f"  output: {content[:120]}")
            elif isinstance(content, list):
                # Structured output may be a list of content blocks
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        print(f"  output: {block.get('text', '')[:120]}")
                        break

    # Child runs — tool calls and LLM calls
    child_runs = list(client.list_runs(project_name=PROJECT, trace_id=run.trace_id, is_root=False))
    tools_used = [c.name for c in child_runs if c.run_type == "tool"]
    llm_calls = [c for c in child_runs if c.run_type == "llm"]

    print(f"\n  tool calls ({len(tools_used)}): {tools_used}")
    print(f"  llm calls ({len(llm_calls)})")

    # Trace URL for UI inspection
    if hasattr(run, "url") and run.url:
        print(f"\n  🔗 trace: {run.url}")
else:
    print("No runs to inspect — run Lab 01 first.")

# %% [markdown]
# ## 2. Monitoring dashboard — latency, errors, cost, tool failures
#
# 📖 [Observability](https://docs.langchain.com/langsmith/observability)
#
# The LangSmith UI shows these as charts on the project dashboard. Here we compute the same
# signals from the trace data so you can see *how the sausage is made* — and so you can
# build custom alerts or reports later.
#
# ### Build the dashboard in the UI first
#
# 📖 [Dashboards](https://docs.langchain.com/langsmith/dashboards) ·
# [Observability](https://docs.langchain.com/langsmith/observability)
#
# **1. Open the built-in project dashboard**
# - **Projects** → your project → the **Dashboard** tab
# - LangSmith ships charts for run count, error rate, latency percentiles, token usage and cost
#   out of the box. For most teams this is the whole monitoring story on day one
#
# **2. Add a custom chart for a quality signal**
# - **Dashboards** in the left sidebar → **+ New Dashboard** (or the project dashboard →
#   **+ Chart**)
# - Pick the metric — to chart an online-eval score, chart the **feedback key** the evaluator
#   writes (e.g. `groundedness`). This is why §3's evaluators matter here: their scores become
#   feedback, and feedback is chartable
# - Group by metadata to split a chart by cohort. With `PARTICIPANT` set, `metadata.participant`
#   splits the workshop's traffic per attendee; in production you would group by tenant or plan
#
# **3. Filter the runs table, then keep the filter**
# - The runs-table filter syntax is the same one evaluators and automations use, so a filter you
#   prove out on the table can be pasted into a rule. Useful starting points:
#   `has(feedback_key, "groundedness") and feedback_score < 1`, or filtering on a specific tool call
#
# > **Why this lab still computes the numbers in code.** The UI answers *"what is happening?"*.
# > Computing p50/p95/p99 and error rate yourself answers *"what threshold should alert me?"* — and
# > gives you something to put in CI. Read the cells below as the reasoning behind the chart, not a
# > replacement for it.
#
# | Signal | What it tells you | Healthy looks like |
# |---|---|---|
# | **Latency (p50/p95/p99)** | How long users wait | p95 < 30s for a support agent |
# | **Error rate** | Crashed runs / total runs | < 2% |
# | **Token usage** | Cost per run, cost trend | Stable or declining over time |
# | **Tool failure rate** | Which tools break | < 5% per tool |
# | **Drift** | Are runs getting slower/more expensive? | Flat trend lines |

# %%
# Use a wider window for meaningful stats — last 7 days or all available runs
week_ago = now - timedelta(days=7)

# `limit` is capped at 100 by the runs API — a larger value is a 400, not a bigger page.
RUN_LIMIT = 100

all_runs = list(client.list_runs(
    project_name=PROJECT,
    is_root=True,
    start_time=week_ago,
    limit=RUN_LIMIT,
))
if not all_runs:
    all_runs = list(client.list_runs(project_name=PROJECT, is_root=True, limit=RUN_LIMIT))

if not all_runs:
    print(f"⚠ No runs in project {PROJECT!r} — cannot compute monitoring stats.")
    print("  Run Lab 01 first to generate production traces.")
else:
    # ── Latency percentiles ──────────────────────────────────────────────────
    latencies = sorted(r.latency for r in all_runs if r.latency)
    if latencies:
        n = len(latencies)
        p50 = latencies[n // 2]
        p95 = latencies[min(int(n * 0.95), n - 1)]
        p99 = latencies[min(int(n * 0.99), n - 1)]
        print(f"latency (n={n}):")
        print(f"  p50 = {p50:.1f}s   p95 = {p95:.1f}s   p99 = {p99:.1f}s")
        if p95 > 30:
            print("  ⚠ p95 > 30s — users are waiting too long")

    # ── Error rate ───────────────────────────────────────────────────────────
    errors = [r for r in all_runs if r.error]
    error_rate = len(errors) / len(all_runs) * 100 if all_runs else 0
    print(f"\nerror rate: {error_rate:.1f}%  ({len(errors)}/{len(all_runs)})")
    if error_rate > 2:
        print("  ⚠ error rate > 2% — investigate failing runs")

    # ── Token usage ──────────────────────────────────────────────────────────
    token_runs = [r for r in all_runs if r.total_tokens]
    if token_runs:
        total_tokens = sum(r.total_tokens for r in token_runs)
        prompt_tokens = sum(r.prompt_tokens or 0 for r in token_runs)
        completion_tokens = sum(r.completion_tokens or 0 for r in token_runs)
        avg_tokens = total_tokens / len(token_runs)
        print(f"\ntokens (n={len(token_runs)}):")
        print(f"  avg = {avg_tokens:.0f}   total = {total_tokens:,} "
              f"(in {prompt_tokens:,} / out {completion_tokens:,})")
        # Input and output are priced differently (output costs several times input), so pricing
        # `total_tokens` at the input rate understates the bill. Split them.
        # These rates are PLACEHOLDERS — set them from your provider's current price list for the
        # model you actually run, or read LangSmith's own cost column instead of computing one.
        IN_RATE = float(os.getenv("MODEL_IN_RATE_PER_1M", "0.40"))
        OUT_RATE = float(os.getenv("MODEL_OUT_RATE_PER_1M", "1.60"))
        est_cost = (prompt_tokens * IN_RATE + completion_tokens * OUT_RATE) / 1_000_000
        print(f"  est. cost ≈ ${est_cost:.4f} "
              f"(placeholder rates: in ${IN_RATE}/1M, out ${OUT_RATE}/1M — set MODEL_IN_RATE_PER_1M / "
              f"MODEL_OUT_RATE_PER_1M)")
        print("  ℹ️  LangSmith also reports actual cost per run when provider pricing is configured.")

    # ── Tool failures ────────────────────────────────────────────────────────
    # Fetch child runs to see tool-level errors
    tool_stats = defaultdict(lambda: {"total": 0, "errors": 0})
    for r in all_runs[:20]:  # sample last 20 traces for tool-level detail
        children = list(client.list_runs(project_name=PROJECT, trace_id=r.trace_id, is_root=False))
        for c in children:
            if c.run_type == "tool":
                tool_stats[c.name]["total"] += 1
                if c.error:
                    tool_stats[c.name]["errors"] += 1

    if tool_stats:
        print(f"\ntool call stats (sampled from {min(20, len(all_runs))} traces):")
        for name, stats in sorted(tool_stats.items(), key=lambda x: x[1]["errors"], reverse=True):
            rate = stats["errors"] / stats["total"] * 100 if stats["total"] else 0
            flag = " ⚠" if rate > 5 else ""
            print(f"  {name:<30} {stats['total']:>3} calls  {stats['errors']:>2} errors  ({rate:.0f}%){flag}")

# %% [markdown]
# ### Drift detection — are runs changing over time?
#
# Drift means the production distribution is shifting: latency creeping up, token counts
# growing, new tool patterns emerging. Simple approach: split runs into two halves by time
# and compare.

# %%
if all_runs and len(all_runs) >= 10:
    # Sort by time and split into older/newer halves
    timed = sorted(all_runs, key=lambda r: r.start_time)
    mid = len(timed) // 2
    older, newer = timed[:mid], timed[mid:]

    def _avg_latency(runs):
        lats = [r.latency for r in runs if r.latency]
        return sum(lats) / len(lats) if lats else 0

    def _avg_tokens(runs):
        toks = [r.total_tokens for r in runs if r.total_tokens]
        return sum(toks) / len(toks) if toks else 0

    old_lat, new_lat = _avg_latency(older), _avg_latency(newer)
    old_tok, new_tok = _avg_tokens(older), _avg_tokens(newer)

    print("drift check (older half vs newer half):")
    print(f"  latency:  {old_lat:.1f}s → {new_lat:.1f}s  ({'↑' if new_lat > old_lat else '↓'} {abs(new_lat - old_lat):.1f}s)")
    print(f"  tokens:   {old_tok:.0f} → {new_tok:.0f}  ({'↑' if new_tok > old_tok else '↓'} {abs(new_tok - old_tok):.0f})")

    if new_lat > old_lat * 1.2:
        print("  ⚠ latency increased >20% — possible drift")
    if new_tok > old_tok * 1.2:
        print("  ⚠ token usage increased >20% — possible drift")
    if new_lat <= old_lat * 1.2 and new_tok <= old_tok * 1.2:
        print("  ✅ no significant drift detected")
elif all_runs:
    print(f"Only {len(all_runs)} runs — need ≥10 for meaningful drift comparison.")

# %% [markdown]
# ## 3. Quality signals — online eval scores
#
# 📖 [Online evaluations](https://docs.langchain.com/langsmith/online-evaluations)
#
# Lab 03 attached online evaluators to the deployed agent. Those scores live as **feedback**
# on each run. Here we aggregate them to see the quality trend.
#
# Feedback keys from Lab 03 might include: `groundedness`, `policy_compliance`,
# `escalation_correctness`, `action_quality` — whatever evaluators were attached.

# %%
if all_runs:
    # Collect feedback across recent runs
    feedback_by_key = defaultdict(list)
    runs_with_feedback = 0

    for r in all_runs[:30]:  # sample last 30 runs
        fb_list = list(client.list_feedback(run_ids=[r.id]))
        if fb_list:
            runs_with_feedback += 1
            for fb in fb_list:
                if fb.score is not None:
                    feedback_by_key[fb.key].append(fb.score)

    if feedback_by_key:
        print(f"quality signals (from {runs_with_feedback} runs with feedback):\n")
        for key, scores in sorted(feedback_by_key.items()):
            avg = sum(scores) / len(scores)
            bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
            flag = " ⚠" if avg < 0.7 else ""
            print(f"  {key:<30} {bar} {avg:.2f}  (n={len(scores)}){flag}")
    else:
        print("No feedback found on recent runs.")
        print("  Run Lab 03 (online evals) first to attach evaluators to deployed runs.")
else:
    print("No runs available — run Lab 01 first.")

# %% [markdown]
# ## 4. User feedback patterns
#
# Users can give thumbs-up/thumbs-down on deployed agent responses (via the LangSmith UI,
# a chat interface, or an API call). This feedback is the ground truth for whether the agent
# is actually helping.
#
# Feedback patterns to look for:
# - **Negative feedback clusters** — certain ticket types getting thumbs-down
# - **Feedback volume** — are users bothering to rate? (low volume = low engagement)
# - **Correlation with evals** — do low eval scores match negative user feedback?

# %%
if all_runs:
    # Look for user feedback (thumbs up/down, corrections, comments)
    user_feedback = []
    for r in all_runs[:30]:
        fb_list = list(client.list_feedback(run_ids=[r.id]))
        for fb in fb_list:
            user_feedback.append({
                "run_id": r.id,
                "key": fb.key,
                "score": fb.score,
                "value": fb.value,
                "comment": fb.comment,
                "created_at": fb.created_at,
            })

    if user_feedback:
        # Separate automated eval feedback from user feedback
        eval_keys = {"groundedness", "policy_compliance", "escalation_correctness",
                      "action_quality", "report_schema_valid", "required_fields_present"}
        user_fb = [f for f in user_feedback if f["key"] not in eval_keys]
        eval_fb = [f for f in user_feedback if f["key"] in eval_keys]

        print(f"feedback summary (last {min(30, len(all_runs))} runs):")
        print(f"  eval feedback entries:  {len(eval_fb)}")
        print(f"  user feedback entries:  {len(user_fb)}")

        if user_fb:
            # Show user feedback with comments
            with_comments = [f for f in user_fb if f["comment"]]
            print(f"\n  user comments ({len(with_comments)}):")
            for f in with_comments[:5]:
                print(f"    [{f['key']}] {f['comment'][:100]}")

            # Score distribution
            scores = [f["score"] for f in user_fb if f["score"] is not None]
            if scores:
                avg = sum(scores) / len(scores)
                print(f"\n  avg user score: {avg:.2f} (n={len(scores)})")
                if avg < 0.5:
                    print("  ⚠ avg user score < 0.5 — users are not satisfied")
        else:
            print("\n  No user-specific feedback found.")
            print("  User feedback is captured via the LangSmith UI thumbs-up/down,")
            print("  a chat interface, or client.create_feedback() from your app.")
    else:
        print("No feedback found on any recent runs.")
        print("  Run Lab 03 first, or add user feedback via the LangSmith UI.")
else:
    print("No runs available — run Lab 01 first.")

# %% [markdown]
# ## 5. Insights — let LangSmith find the patterns for you
#
# 📖 [Insights](https://docs.langchain.com/langsmith/insights) ·
# [Model configurations](https://docs.langchain.com/langsmith/model-configurations)
#
# Everything above assumes you already know what to measure — latency, errors, cost, a named
# feedback key. **Insights answers the question you have not thought to ask yet.** It reads a sample
# of traces and groups them into an auto-generated hierarchy of categories, surfacing usage
# patterns, common agent behaviours and failure modes without anyone reading thousands of runs.
#
# For Acme this is the difference between *"our p95 is 30s"* and *"31% of tickets are password
# resets that never needed the agent at all"*. The first is a metric; the second is a decision.
#
# > **Plan gating — check before you promise it in a demo.** Insights requires a LangSmith **Plus or
# > Enterprise** plan, plus a **model configuration** for Insights in workspace settings and
# > permission to create rules. On a free/Developer workspace the tab will not be there.
#
# ### Generate a report in the UI
#
# 1. **Tracing Projects** in the left-hand menu → select your project
# 2. **+ New** (top right) → **New Insights Report**
# 3. Name the job
# 4. Configure a model for Insights in workspace settings, if you have not already
# 5. Answer the **guided questions** to focus the report on what you want to learn, then **Run job**
#    *(toggle to Manual mode to configure the job by hand)*
#
# It runs in the background and **can take up to 30 minutes** — start it before a break, not during
# a live demo.
#
# ### Reading the report
#
# | Part | What it gives you |
# |---|---|
# | **Executive summary** | The dominant patterns, each with the **percentage of traces** it covers, and clickable references to the traces the analysis found most representative |
# | **Top-level categories** | Auto-generated groupings with distribution bars — this is where "happens more often than expected" jumps out |
# | **Subcategories** | Click any category to break it down further (e.g. *Data & Retrieval* → *Vector Stores*, *Data Ingestion*) |
# | **Per-category metrics** | Error rate, latency and cost for that category, plus **your evaluator feedback scores** and any attributes you extracted |
# | **Individual traces** | Click through any category to the traces table, then into the full conversation |
#
# The metrics row is why §3's online evaluators matter here: Insights aggregates *your* quality
# scores per discovered category, so you can see which behaviour is both common **and** scoring
# badly. That intersection is the improvement backlog, ranked.
#
# ### Configuring the job — what you can actually control
#
# The guided flow ("Auto" toggle) turns your answers into a draft config. Switch to **Manual** for
# the full surface:
#
# **Select traces**
#
# | Control | Detail |
# |---|---|
# | **Sample size** | Max traces to analyse — **1,000 limit** |
# | **Time range** | Traces are sampled from this window |
# | **Filters** | Same filter syntax as everywhere else; the UI shows the matching count as you adjust |
#
# **Two models, different jobs** — pick both from the same provider for best results:
#
# | Role | Does | Cost profile |
# |---|---|---|
# | **Thinking model** | The clustering step | More capable, higher cost |
# | **Summarization model** | Per-trace summaries | Faster, cheaper |
#
# **Categories.** By default they are discovered bottom-up. If you already know the buckets you
# care about — for Acme, say *password reset · access request · VPN · hardware · security incident* —
# enumerate them with descriptions and the job sorts into those instead. Subcategories are still
# auto-generated inside them. **When a job finishes, discovered categories are saved back to the
# config** (only if none were predefined), so scheduled runs stay consistent over time.
#
# **Summary prompt — the highest-leverage setting.** Every trace is summarised first, and *only
# what the summary captures can influence the categories*. Edit it to say what matters, and use
# mustache variables to control what gets sent:
#
# | Variable | Contents |
# |---|---|
# | `{{run.inputs}}` / `{{run.outputs}}` | Inputs/outputs of the most recent root run |
# | `{{run.error}}` | Error string, if the run failed |
# | `{{run.feedback}}` | All feedback scores as JSON |
# | `{{run.feedback.<key>}}` | One score, e.g. `{{run.feedback.groundedness}}` |
# | `{{all_thread_messages}}` | Full thread history *(threads-enabled projects only)* |
#
# Dot notation works for nesting (`{{run.inputs.foo.bar}}`). At least one variable is required.
# Trimming a large trace down to the relevant fields makes the job cheaper **and** the categories
# sharper — noise in, noise out.
#
# **Attributes.** Extract string, numeric or boolean fields per trace. They steer clustering
# (similar values group together) and are aggregated per category. E.g. `escalated: boolean` would
# split the Acme data into escalated vs self-served, and show the escalation rate per category.
# Setting `"filter_by": true` on a boolean attribute **pre-filters**: only traces where it evaluates
# true are analysed — the clean way to scope a report to "errors only" or "security tickets only".
#
# ### What to actually look for in the Acme data
#
# The docs' example is the chat.langchain.com bot, where *Data & Retrieval* breaks into
# *Vector Stores* and *Data Ingestion*. Here is the equivalent for this workshop's ticket corpus —
# useful both as **predefined categories** (§ *Categories* above) and as a quick check that the
# report found something real.
#
# The 24 committed tickets carry these `category` values: network 4 · access 5 · software 4 ·
# account 3 · hardware 5 · security 2 · knowledge 1. But **do not just predefine those** — the
# database field is what a human already labelled it. Insights earns its keep by grouping on
# *behaviour*, which cuts across that field:
#
# | Top-level category to predefine | Subcategories you should expect | Why it matters to Acme |
# |---|---|---|
# | **Credential & access recovery** | MFA re-enrolment after device change · password expiry / lockout · group-membership requests (SharePoint, SAP, share drives) | The biggest self-service candidate. Note it spans *both* `access` and `account` — the DB field splits what is operationally one workflow |
# | **Remote connectivity** | VPN drops by location (home · hotel Wi-Fi · headquarters office) · Wi-Fi roaming between floors | Recurrence by *location* is the signal; three near-identical VPN tickets point at infrastructure, not user error |
# | **Endpoint & peripherals** | Printers · docks and monitors after firmware updates · laptop performance post-patch | "After update" is the pattern to watch — it links tickets to a change window |
# | **Software provisioning** | Install requests needing admin rights (Docker, Zoom) · application faults (Outlook sync, Excel crashes) | Splits *entitlement* requests from genuine faults — very different fixes |
# | **Security reporting** | Suspicious email / phishing · credential-re-entry lures | Must never be auto-resolved; check these correlate with escalation |
# | **How-to / knowledge** | Setup guidance answerable from the KB alone | Pure deflection candidate — if the agent resolved these unaided, that is your ROI number |
#
# **The three findings worth hunting for**, in order of value:
#
# 1. **A category that is large *and* scores badly.** Cross-reference the per-category
#    `groundedness` / `escalation_appropriateness` averages from §3. Big + bad = top of the backlog.
# 2. **A category that is large and scores *well*.** That is the automation candidate — the agent
#    already handles it, so the question becomes whether it needs to reach a human at all.
# 3. **A category that splits differently from the `category` field.** Wherever Insights groups
#    things the database separates (or vice versa), your taxonomy is wrong — and taxonomy drives
#    routing, SLAs and staffing.
#
# > **Useful attributes to extract here:** `escalated: boolean` (splits self-served from escalated,
# > and gives an escalation rate per category), `required_admin_rights: boolean`, and
# > `location: string` for the connectivity tickets. Set `filter_by: true` on `escalated` to produce
# > a report about *only* the escalated traffic.
#
# **Schedule + save.** Reports can run **Daily** (08:00 UTC), **Weekly on Monday**, or on a custom
# cron. Time ranges are computed at execution, so "last 24 hours" always means the last 24 hours.
# **Save as** a named config to compare like-for-like reports over time — which is the point: one
# report is a snapshot, a scheduled series shows drift.

# %%
# The SDK path — useful when the conversations you want analysed live OUTSIDE LangSmith (an
# existing ITSM export, chat logs, a CSV of past tickets). `generate_insights` uploads them as
# traces to a new project, runs the report, and returns a link.
#
# Guarded behind an env flag: a report costs roughly $1-2 per 1,000 threads on OpenAI models
# (~$3-4 on Anthropic) and takes up to 30 minutes, so it must never run by accident in a workshop.
# Set RUN_INSIGHTS=1 to actually execute it.
import os

if os.getenv("RUN_INSIGHTS") == "1":
    from langsmith import Client as _LSClient

    from day1.src.models import scoped as _scoped

    _ls = _LSClient()
    # Stand in for an ITSM export: each item is one conversation. Loaded here rather than relying
    # on a module-level name, so this cell is self-contained if you run it on its own.
    import json as _json

    _tickets = _json.loads((WORKSHOP / "day1" / "data" / "tickets.json").read_text())
    _chat_histories = [
        [{"role": "user", "content": _t["subject"]},
         {"role": "assistant", "content": _t.get("resolution") or "unresolved"}]
        for _t in _tickets[:20]
    ]
    # generate_insights parameters (langsmith SDK):
    #   chat_histories   list[list[dict]]  required — each item is one conversation
    #   instructions     str               what to learn; defaults to a generic usage question
    #   name             str | None        report name
    #   model            'openai'|'anthropic'|None   provider for the analysis
    #   openai_api_key / anthropic_api_key           optional if set as a workspace secret
    _report = _ls.generate_insights(
        chat_histories=_chat_histories,
        name=_scoped("acme-ticket-insights"),
        instructions=(
            "What categories of IT support request are these, which ones look automatable, "
            "and which ones required human escalation?"
        ),
        model="openai",
    )
    print("insights report started — poll or open it in the Insights tab")
    # poll_insights(report=, id=, project_id=, rate=30, timeout=1800, verbose=False)
    #   blocks until the report completes; `rate` is the poll interval in seconds
    # _ls.poll_insights(report=_report, verbose=True)
    #
    # get_insights_report(id=, report=, project_id=, include_runs=True)
    #   fetches a finished report; include_runs=False keeps the payload small
    # _done = _ls.get_insights_report(report=_report, include_runs=False)
else:
    print("⏭ Insights report not run (set RUN_INSIGHTS=1 to generate one)")
    print("   Cost ~$1-2 per 1,000 threads (OpenAI) · runs up to 30 min in the background.")
    print("   Requires a LangSmith Plus/Enterprise plan and an Insights model configuration.")

# %% [markdown]
# ### Where Insights sits in the loop
#
# ```mermaid
# graph LR
#     T[production traces] --> I[Insights<br/>auto-categorise]
#     I --> P[pattern + % of traffic]
#     P --> Q{also scoring badly?}
#     Q -->|yes| B[improvement backlog<br/>ranked by volume x quality]
#     Q -->|no| M[monitor only]
#     B --> D[(offline dataset)]
#     D --> E[experiment] --> R[redeploy]
#     R --> T
# ```
#
# Insights tells you *what is happening at scale*; §3's online evals tell you *what is going wrong*.
# You need both to rank work: a failure mode in 0.1% of traffic is a bug report, the same failure in
# 31% is a roadmap item.

# %% [markdown]
# ## 6. Alerts — 👀 **presenter demo, no participant code**
#
# 📖 [Alerts](https://docs.langchain.com/langsmith/alerts)
#
# > **Show, don't build.** Alerts are configured entirely in the UI and need a notification channel
# > (Slack, PagerDuty, a webhook) that a workshop workspace usually will not have wired up. Walk
# > through it on screen; there is deliberately no cell to run here.
#
# Everything so far is *pull* — you open a dashboard and look. Alerts are *push*: the thing that
# tells you at 02:00 that the agent's groundedness has fallen off a cliff.
#
# Alerts are **project-scoped**, so each tracing project is configured separately.
#
# ### The walkthrough
#
# **1.** Open the tracing project → click the **Alerts** icon, top right of the page
#
# **2. Pick a metric** — five threshold-based types:
#
# | Metric | Alerts you to | Acme example |
# |---|---|---|
# | **Run Count** | Volume dropping (or spiking) unexpectedly | The ticket agent stopped receiving traffic — an integration broke |
# | **Cost** | Spend above expectation *(needs cost tracking configured)* | A prompt change quietly tripled token use |
# | **Errors** | Error count, or error **rate** | Tool failures after a dependency upgrade |
# | **Feedback Score** | A named feedback key regressing | **`groundedness` average falling** — this is the one that matters here |
# | **Latency** | Average execution time rising | p95 drifting past the support SLA |
#
# **3. Define the condition** — aggregation (Average / Percentage / Count), operator (`>=`, `<=`),
# threshold, and a **5- or 15-minute window**. Feedback-score alerts also take the **feedback key**.
#
# **4. Preview before saving.** The UI replays the condition over a historical window and shows
# which datapoints *would* have fired, in red. Use it — a threshold that looks sensible in a meeting
# usually fires constantly or never, and this is where you find out which.
#
# **5. Route it** — Slack (native on Cloud), PagerDuty, Dynatrace, or any HTTP endpoint via webhook.
# The webhook tab ships recipes for Microsoft Teams, email, Slack on self-hosted, and Google Chat.
# Use **Send Test Notification** to confirm delivery before you rely on it.
#
# > For **Errors** and **Latency** you can stack filters — **Status**, **Run Type**, **Tag**,
# > **Error** — so an alert can be scoped to, say, `Status = error AND Run Type = llm AND
# > Tag = support_agent AND Error matches RateLimitExceeded`. Scope alerts narrowly or they become
# > noise people mute, which is worse than having none.
#
# > **Self-hosted:** alerts need Helm chart **0.10.3 or later**. Worth checking before promising
# > them in an air-gapped Acme deployment.
#
# ### Why this closes the loop
#
# The `groundedness` alert is only possible because §3 attached an online evaluator that writes that
# feedback key. Evaluator → feedback → chart → **alert** is one chain: each step exists because the
# previous one produced something nameable.

# %% [markdown]
# ## 7. Identify one production signal for improvement
#
# The whole point of monitoring is to find the **one thing** that should drive the next
# improvement cycle. Not five things — one. The signal should be:
#
# 1. **Measurable** — you can quantify it (error rate, latency p95, eval score, feedback avg)
# 2. **Actionable** — you know which lever to pull (prompt, tools, retrieval, schema)
# 3. **High-impact** — fixing it improves the user experience meaningfully
#
# ### Common signals and their levers
#
# | Signal | Threshold | Likely lever |
# |---|---|---|
# | High error rate | > 2% of runs fail | Tool error handling, retry logic |
# | High latency | p95 > 30s | Reduce tool calls, parallelize, cheaper model |
# | Tool failures | > 5% for a specific tool | Fix tool description, add error handling |
# | Low eval score | avg < 0.7 on any key | Prompt, retrieval strategy, output schema |
# | Negative user feedback | avg < 0.5 | Prompt, escalation rules, response quality |
# | Token drift | avg tokens ↑ >20% | Shorter prompts, cheaper model, caching |
# | Low feedback volume | < 10% of runs rated | Add feedback prompt to the UI |
#
# > **Exercise:** Look at the signals computed above. Pick **one** that crosses a threshold
# > (or comes closest). Write down: (a) the signal, (b) its current value, (c) the lever you'd
# > pull, and (d) what you'd expect to change. That's your improvement cycle for the next lab.

# %%
# Summarize all signals in one place for the exercise
if all_runs:
    print("=" * 60)
    print("SIGNAL SUMMARY — pick one for the improvement cycle")
    print("=" * 60)

    # Latency
    if latencies:
        print(f"\n📊 Latency:     p50={p50:.1f}s  p95={p95:.1f}s  p99={p99:.1f}s")

    # Errors
    print(f"📊 Error rate:  {error_rate:.1f}%  ({len(errors)}/{len(all_runs)})")

    # Tokens
    if token_runs:
        print(f"📊 Avg tokens:  {avg_tokens:.0f}  (total: {total_tokens:,})")

    # Tool failures
    if tool_stats:
        worst_tool = max(tool_stats.items(), key=lambda x: x[1]["errors"] / max(x[1]["total"], 1))
        worst_rate = worst_tool[1]["errors"] / max(worst_tool[1]["total"], 1) * 100
        print(f"📊 Worst tool:  {worst_tool[0]}  ({worst_rate:.0f}% failure rate)")

    # Quality
    if feedback_by_key:
        for key, scores in sorted(feedback_by_key.items()):
            avg = sum(scores) / len(scores)
            print(f"📊 Eval [{key}]: {avg:.2f}  (n={len(scores)})")

    # User feedback
    if user_feedback:
        user_scores = [f["score"] for f in user_feedback if f["score"] is not None
                       and f["key"] not in eval_keys]
        if user_scores:
            print(f"📊 User score:  {sum(user_scores)/len(user_scores):.2f}  (n={len(user_scores)})")

    # Drift
    if len(all_runs) >= 10:
        print(f"📊 Drift:       latency {'↑' if new_lat > old_lat else '↓'}  "
              f"tokens {'↑' if new_tok > old_tok else '↓'}")

    print("\n" + "=" * 60)
    print("👉 Pick ONE signal above. Write down:")
    print("   (a) the signal          (b) current value")
    print("   (c) the lever to pull   (d) expected change")
    print("=" * 60)
else:
    print("No runs available — run Lab 01 first to generate production traces.")

# %% [markdown]
# ## 8. Recap & next
#
# | Step | What you did |
# |---|---|
# | Inspect traces | Queried deployed runs with `Client.list_runs()`, drilled into one trace |
# | Monitor | Computed latency percentiles, error rate, token usage, tool failure rates |
# | Drift | Compared older vs newer runs for latency and token drift |
# | Quality | Aggregated online eval feedback scores across runs |
# | User feedback | Reviewed user-submitted feedback patterns |
# | **Signal** | Identified one production signal to drive the next improvement cycle |
#
# **Next:** Advanced exercise 4 — take the signal you identified and implement the improvement:
# update a prompt, add error handling to a tool, adjust the retrieval strategy, or add a new
# monitoring signal to track. Then redeploy and re-measure.
#
# > The **hill-climbing loop** is now complete: deploy → monitor → identify signal → improve →
# > redeploy → re-measure. Each turn makes the agent measurably better.
