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
# # 02 · Managed Deep Agents — the hosted path
#
# **Workshop:** LangChain ADLC Workshop · **Day 3** · **ADLC stage:** Deploy + Govern
#
# > **Loop Engineering focus: Deployment loop** — Lab 01 deployed a graph you wrote and wired
# > yourself. **Managed Deep Agents (MDA)** is the other end of the spectrum: you declare the agent
# > and LangSmith supplies the harness, the runtime, the context store, and the eval scaffold.
#
# <div>
#
# > 🚧 **Public beta.** MDA is in [public beta](https://docs.langchain.com/langsmith/release-stages)
# > on **LangSmith Cloud, US region only**. This workshop's workspace is US-hosted, so this lab is
# > fully hands-on: you scaffold a project, define the agent, add durable memory and a schedule, wire
# > a Harbor eval, and run it locally.
# >
# > **The constraint still matters for production.** US-region-only is a data-residency question for
# > an organisation with strict data-residency requirements. §1–§7 run entirely on your machine and cost nothing; §8 (`mda deploy`) creates a
# > **hosted deployment in the US region** and a billable resource — do that knowingly and
# > `mda delete .` afterwards. For EU or air-gapped hosting, Lab 01's `langgraph deploy` self-hosted
# > path is the one that ships today.
#
# </div>
#
# ## Where MDA sits
#
# ```mermaid
# graph TD
#     subgraph YOU["You write"]
#       A[agent.py<br/>define_deep_agent] --> I[instructions.md]
#       I --> SK[skills/*.md]
#       SK --> ID[identity.py]
#     end
#     subgraph MDA["LangSmith supplies"]
#       H[Deep Agents harness] --> R[Hosted Agent Server]
#       R --> CH[(Context Hub<br/>durable memory)]
#       R --> T[Tracing + evals]
#     end
#     YOU -->|mda deploy| MDA
# ```
#
# | | `langgraph deploy` (Lab 01) | `mda deploy` (this lab) |
# |---|---|---|
# | You author | the graph, middleware, tools, checkpointer | an agent **definition** + instructions |
# | Harness | yours to assemble (`create_deep_agent` + middleware) | supplied and versioned by LangSmith |
# | Durable memory | you choose a store | Context Hub, one tree at `/memories/agent/` |
# | Skills | `SkillsMiddleware` + your own files | markdown files, picked up automatically |
# | Eval scaffold | you build it (Day 2 Lab 06) | `mda eval` → Harbor-ready artifact |
# | Availability | GA, all regions, self-host possible | public beta, **US region only** |
#
# The trade is the usual one: less control, far less to maintain. A team that wants the Day 2 deep
# agent in production *this quarter* and does not want to own a middleware stack is the audience.


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

# > 🧭 **Builds on Day 3 Lab 01; runs standalone.** Lab 01 deployed the agent yourself (the
# > self-managed path); this lab shows the *managed* path for the same workflow. Nothing here
# > depends on Lab 01 having been run. Next: Lab 03 attaches online evals to a deployment.
#
# %% [markdown]
# ## 0. Setup
#
# This lab drives the `mda` CLI rather than a Python SDK, so most cells shell out. Everything through
# §7 is local and free; only §8 (`mda deploy`) touches cloud resources, and that cell is commented
# out by default.

# %%
import os
import shutil
import subprocess
import sys
from pathlib import Path


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

load_dotenv(find_dotenv())

# Scratch space for the scaffolded project — deliberately outside the repo so the workshop tree
# stays clean and re-running the lab is idempotent.
SCRATCH = Path(os.getenv("MDA_SCRATCH", "/tmp")) / "langchain-adlc-workshop-mda-lab"
PROJECT = SCRATCH / "vendor-due-diligence"


def sh(cmd: str, cwd: Path | None = None) -> str:
    """Run a shell command and return its combined output (never raises — this is a lab)."""
    env = {**os.environ, "PATH": f"{Path.home() / '.local/bin'}:{os.environ.get('PATH', '')}"}
    proc = subprocess.run(cmd, shell=True, cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=300)
    return (proc.stdout + proc.stderr).strip()


print("scratch:", SCRATCH)
print("langsmith key:", "set" if os.getenv("LANGSMITH_API_KEY") else "MISSING")

# %% [markdown]
# ## 1. Install the CLI
#
# The `mda` CLI ships as its own tool, independent of this repo's dependencies:
#
# ```bash
# uv tool install managed-deepagents
# ```
#
# It is intentionally *not* in `pyproject.toml` — it is a developer tool like `gh`, not a library the
# labs import.

# %%
version = sh("mda --version")
if "mda" in version:
    print("✔", version)
else:
    print("⛔ mda CLI not found. Install it with:  uv tool install managed-deepagents")
    print("   (then restart the kernel so PATH picks it up)")

# %% [markdown]
# ## 2. Scaffold a project
#
# `mda init` generates a complete project. Note how little there is — that is the point.

# %%
SCRATCH.mkdir(parents=True, exist_ok=True)
if PROJECT.exists():
    shutil.rmtree(PROJECT)   # idempotent: always scaffold fresh

print(sh(f"mda init {PROJECT.name}", cwd=SCRATCH)[:400])

print("\nproject files:")
for f in sorted(PROJECT.rglob("*")):
    if f.is_file() and ".git" not in f.parts:
        print("  ", f.relative_to(PROJECT))

# %% [markdown]
# | File | Purpose |
# |---|---|
# | `agent.py` | the agent **definition** — model + capabilities, via `define_deep_agent` |
# | `instructions.md` | the system prompt, as a file you can review and diff |
# | `identity.py` | who the agent acts as; where user-scoped auth is wired |
# | `sandbox/` | code the agent may execute in its sandbox |
# | `pyproject.toml` | the project's own dependencies |
#
# Compare to Day 2: `instructions.md` plays the role of `AGENTS.md`, and skills stay markdown
# directories. **The concepts transfer exactly** — which is the real reason to teach both paths.

# %%
print("--- agent.py (generated) ---")
print((PROJECT / "agent.py").read_text())

# %% [markdown]
# ## 3. Define the vendor due diligence agent
#
# Now make it ours. This is the same use case as Day 2, expressed as a definition instead of an
# assembled middleware stack.

# %%
AGENT_PY = '''"""Vendor due diligence agent — Managed Deep Agents definition.

The Day 2 equivalent is day2/src/deep_agent_factory.py, which assembles create_deep_agent with
MemoryMiddleware, SkillsMiddleware and a FilesystemBackend by hand. Here the harness is supplied;
we declare intent.
"""

from managed_deepagents import define_deep_agent

agent = define_deep_agent(
    name="vendor-due-diligence",
    # Judge-tier model: this agent's output gates procurement decisions.
    model="openai:gpt-4.1",
)
'''

INSTRUCTIONS_MD = '''# Acme vendor due diligence

You assess vendors for Acme procurement. You are cautious, and you never assert what you cannot
source.

## Operating rules

1. **Cite everything.** Every claim ends with its source tag, e.g. `[source: quelmore_systems.md]`.
2. **Label verification.** Each evidence item is `verified`, `partially verified`, or `unverified`.
   A claim confirmed by two independent sources is verified; one source is partially verified.
3. **Screen before recommending.** No vendor is recommended without a sanctions/watchlist verdict.
   A SANCTIONED match means exclude and escalate — never soften it.
4. **Escalate, do not guess.** Low confidence, missing evidence, or any sanctions hit sets the
   human-review flag.
5. **No PII.** Never include contact emails, phone numbers, or named individuals' details.

## Output shape

Vendor summary · Evidence (one bullet per claim, with source + verification label) · Risk signals
(level plus each signal's severity) · Compliance screening verdict · Suitability · Confidence ·
Follow-up questions · Human-review flag.
'''

(PROJECT / "agent.py").write_text(AGENT_PY)
(PROJECT / "instructions.md").write_text(INSTRUCTIONS_MD)
print("wrote agent.py and instructions.md")
print("\n--- instructions.md (first 12 lines) ---")
print("\n".join(INSTRUCTIONS_MD.splitlines()[:12]))

# %% [markdown]
# ### Durable memory and skills
#
# Two managed capabilities map straight onto Day 2 concepts:
#
# | Day 2 (self-assembled) | MDA (managed) |
# |---|---|
# | `MemoryMiddleware` + `FilesystemBackend` over `AGENTS.md` | **durable memory** backed by Context Hub, one read/write tree at `/memories/agent/` |
# | `SkillsMiddleware` + `skills/*/SKILL.md` | **skills** as markdown directories, discovered automatically |
#
# > ⚠️ Two defaults worth knowing: durable memory is **off** unless you enable it, and when on, the
# > deployment gets **one tree shared by every caller**. For a multi-tenant procurement tool that is
# > a design decision, not a detail — one reviewer's notes are visible to the next. Scope per-user
# > state yourself (via `identity.py`) if that is not what you want.

# %%
skills_dir = PROJECT / "skills" / "evidence-review"
skills_dir.mkdir(parents=True, exist_ok=True)
(skills_dir / "SKILL.md").write_text('''# Evidence review

Use when assessing a vendor's claims against source documents.

## Steps
1. List each claim the vendor or KB makes.
2. For each, find a supporting source and record its tag.
3. Mark verified (2+ independent sources), partially verified (1), or unverified (0).
4. Any unverified claim that affects suitability becomes a follow-up question.

## Source validation checklist
- Is the source dated, and is it recent enough to still hold?
- Does a second source corroborate it?
- Does any source contradict it? Contradictions are risk signals, not noise.
''')
print("skills:", [str(p.relative_to(PROJECT)) for p in PROJECT.rglob("SKILL.md")])

# %% [markdown]
# ## 4. Harbor eval scaffold — the part that pays off today
#
# This is the section worth your attention even if you never deploy to MDA. `mda eval` generates a
# **Harbor** task, which is the harness LangChain uses to benchmark deep agents
# ([how we benchmark](https://www.langchain.com/blog/how-we-benchmark-deep-agents)).
#
# A Harbor task is three things — and note that the verifier is **pytest**, not an LLM judge:
#
# | Part | File | Role |
# |---|---|---|
# | Environment | Dockerfile / compose | reproducible sandbox |
# | Instruction | `instruction.md` | what the agent is asked to do |
# | Evaluation | `tests/test_*.py` | deterministic pass/fail |

# %%
print(sh("mda eval init cites_every_claim", cwd=PROJECT)[:400])
print("\nscaffolded eval task:")
for f in sorted((PROJECT / "evals").rglob("*")):
    if f.is_file():
        print("  ", f.relative_to(PROJECT))

# %% [markdown]
# The generated task is a placeholder (write `PONG` to a file). Replace it with a real behavioural
# check — here, the operating rule that matters most: **every claim carries a source tag**.

# %%
task_dir = PROJECT / "evals" / "scaffold" / "cites_every_claim"
if task_dir.exists():
    (task_dir / "instruction.md").write_text(
        "Assess vendor Quelmore Systems Ltd for an avionics maintenance contract.\n\n"
        "Write your assessment to `assessment.md` in the working directory. Every evidence bullet "
        "must end with a source tag in the form `[source: <filename>]` and a verification label of "
        "`verified`, `partially verified`, or `unverified`.\n"
    )
    (task_dir / "tests" / "test_answer.py").write_text('''"""Verifies the citation rule from instructions.md.

Measures the `retrieval` + `tool_use` capabilities: an agent that answers from parametric memory
instead of the corpus cannot produce source tags, so this fails closed on ungrounded output.
"""

import re
from pathlib import Path

ASSESSMENT = Path("/app/assessment.md")
SOURCE_TAG = re.compile(r"\\[source:\\s*[^\\]]+\\]")
VERIFICATION = re.compile(r"\\b(verified|partially verified|unverified)\\b", re.I)


def _evidence_bullets() -> list[str]:
    text = ASSESSMENT.read_text()
    return [ln.strip() for ln in text.splitlines() if ln.strip().startswith(("-", "*"))]


def test_assessment_exists():
    assert ASSESSMENT.exists(), "agent did not write assessment.md"


def test_every_bullet_cites_a_source():
    unsourced = [b for b in _evidence_bullets() if not SOURCE_TAG.search(b)]
    assert not unsourced, f"bullets without a [source: ...] tag: {unsourced[:3]}"


def test_every_bullet_labels_verification():
    unlabelled = [b for b in _evidence_bullets() if not VERIFICATION.search(b)]
    assert not unlabelled, f"bullets without a verification label: {unlabelled[:3]}"


def test_no_personal_pii():
    text = ASSESSMENT.read_text()
    assert not re.search(r"[\\w.+-]+@[\\w-]+\\.[\\w.]+", text), "assessment leaked an email address"
''')
    print("replaced the placeholder task with a real citation check")
    print("\n--- tests/test_answer.py ---")
    print((task_dir / "tests" / "test_answer.py").read_text()[:600])

# %% [markdown]
# **Compile it.** `mda eval compile` emits the Harbor artifact and prints the `harbor run` command —
# it deliberately does **not** execute the trials, so Harbor stays the execution engine.

# %%
print(sh("mda eval compile .", cwd=PROJECT)[:700])

# %% [markdown]
# > 💡 **Two practices from the benchmarking post, worth copying into any harness:** run every task
# > **more than once** (agents are nondeterministic — one pass is an anecdote), and keep a **"lite"
# > subset** for iteration. LangChain's lite suite is ~8× faster and ~6× cheaper than the full one;
# > the full suite is a release gate, not an inner-loop tool.

# %% [markdown]
# ## 5. Durable memory — `define_memory`
#
# 📖 [Deep Agents memory](https://docs.langchain.com/oss/python/deepagents/memory)
#
# By default a managed deep agent's memory is **per-thread**: it forgets between sessions. Durable
# memory is opt-in, backed by Context Hub, and gives the deployment one read/write tree at
# `/memories/agent/`.
#
# For vendor due diligence this is the wiki-memory pattern from Day 2 Lab 02, managed: risk signals
# and open questions survive across assessments instead of being re-derived every time.

# %%
MEMORY_PY = '\n'.join([
    '"""Durable memory for the vendor due diligence agent.',
    '',
    '`scope="agent"` means ONE tree shared by every caller of this deployment. For a procurement',
    'tool that is a real design decision: one reviewer\'s notes are visible to the next. That is what',
    'we want for institutional risk knowledge ("Kelbrant was flagged on 2024-11-03") and what we do NOT',
    'want for per-user drafts — scope those through identity.py instead.',
    '"""',
    '',
    'from managed_deepagents import define_memory',
    '',
    'memory = define_memory(scope="agent")',
])
(PROJECT / "memory.py").write_text(MEMORY_PY + "\n")
print("wrote memory.py")

# Durable memory stays empty unless the instructions say what is worth keeping.
MEMORY_RULES = """
## Memory

- Record durable risk knowledge: a vendor's screening verdict and the date, recurring risk signals,
  and open questions a later assessment should pick up.
- Before assessing a vendor, check memory for a prior verdict and say whether you are confirming or
  revising it.
- Never store contact details, named individuals, or anything that identifies a person.
"""
instructions = (PROJECT / "instructions.md").read_text()
if "## Memory" not in instructions:
    (PROJECT / "instructions.md").write_text(instructions + MEMORY_RULES)
print("instructions.md now tells the agent what to persist")

# %% [markdown]
# > 💡 **The test that matters** (from the tutorial): ask the agent to record something in one thread,
# > then open a **new** thread and check it applies the rule with no conversation history. If it does
# > not, your `instructions.md` never told it what was worth remembering — the same lesson as Day 2
# > Lab 06's `AGENTS.md` A/B.

# %% [markdown]
# ## 6. Scheduled runs — `define_schedule`
#
# A deployed agent can run on a cron with nobody asking it to. For procurement the obvious use is
# **re-screening**: sanctions lists change, and a verdict that was CLEAR six months ago is not
# evidence of anything today.

# %%
(PROJECT / "schedules").mkdir(exist_ok=True)
SCHEDULE_PY = '\n'.join([
    '"""Weekly re-screening digest.',
    '',
    'Sanctions lists move; a CLEAR verdict has a shelf life. This runs unattended and leaves a digest',
    'the procurement team reads on Monday morning, instead of waiting for someone to think of it.',
    '"""',
    '',
    'from managed_deepagents import define_schedule',
    '',
    'schedule = define_schedule(',
    '    cron="0 7 * * 1",                 # Mondays 07:00',
    '    timezone="Europe/Amsterdam",     # the TEAM\'s timezone, not the deployment region\'s',
    '    prompt=(',
    '        "Review durable memory for vendors previously screened. For each, state the last "',
    '        "recorded verdict and its date, and flag any not re-screened in over 90 days as "',
    '        "requiring a fresh check. Then list open questions for this week. Do not include PII."',
    '    ),',
    ')',
])
(PROJECT / "schedules" / "weekly_rescreen.py").write_text(SCHEDULE_PY + "\n")
print("wrote schedules/weekly_rescreen.py")
print("\nproject python files:")
for f in sorted(PROJECT.rglob("*.py")):
    print("  ", f.relative_to(PROJECT))

# %% [markdown]
# | Config | Value here | Why |
# |---|---|---|
# | `cron` | `0 7 * * 1` | Monday 07:00 — a digest waiting when the team logs on |
# | `timezone` | `Europe/Amsterdam` | the **team's** timezone, not the deployment region's |
# | `prompt` | re-screening review | the schedule's prompt is the whole instruction; there is no user turn |
#
# > ⚠️ A scheduled run is an **unattended** agent, so everything Day 1 taught about HITL applies
# > harder: a cron job must not be able to take a sensitive action with no human present. Keep
# > `interrupt_on` for anything consequential, and let the schedule *report* rather than *act*.

# %% [markdown]
# ## 7. Run it locally
#
# 📖 [Agents (create_agent)](https://docs.langchain.com/oss/python/langchain/agents) · [reference](https://reference.langchain.com/python/langchain/agents/factory/create_agent)
#
# `mda dev` runs the real managed harness against a local Agent Server and opens it in LangSmith
# Studio — the same experience as `langgraph dev`, except the harness is supplied. Nothing leaves your
# machine, and it costs nothing.
#
# ```bash
# cd <project shown above>
# uv sync
# mda dev .
# ```
#
# Try, in order, to exercise durable memory:
#
# 1. *"Screen Kelbrant Trading Consortium and record the verdict."*
# 2. Open a **new thread** — *"What do we already know about Kelbrant?"*
#
# If step 2 answers from memory with no conversation history, `define_memory` is working.

# %%
# Validate the project compiles as MDA sees it, without deploying.
print(sh("mda eval compile .", cwd=PROJECT).splitlines()[0] if sh("mda eval compile .", cwd=PROJECT) else "")
print("\nlocal dev is a foreground server, so run it yourself:")
print(f"  cd {PROJECT} && uv sync && mda dev .")

# %% [markdown]
# ## 8. Deploy to the hosted runtime
#
# 📖 [Deployments / CLI](https://docs.langchain.com/langsmith/cli)
#
# ```bash
# mda deploy .     # compiles, syncs context to Context Hub, triggers a hosted build
# mda logs .       # tail the Agent Server logs
# mda channel add slack
# mda delete .     # remove the deployment and the LangSmith resources it created
# ```
#
# > 💰 **`mda deploy` creates a billable hosted deployment in the US region.** The cell below is
# > commented out deliberately — uncomment it when you mean it, and run `mda delete .` afterwards.

# %%
print("ready to deploy:", PROJECT)
print("  mda deploy .     # hosted, billable, US region")
print("  mda delete .     # tear down when the workshop ends")

# Uncomment to actually deploy:
# print(sh("mda deploy .", cwd=PROJECT))

# %% [markdown]
# What `mda deploy` does, per the
# [docs](https://docs.langchain.com/langsmith/python/managed-deep-agents-deploy): compiles the project
# into a managed LangGraph app, **syncs deploy-owned context to Context Hub**, uploads the compiled
# source, and triggers a hosted build.
#
# That middle step deserves a decision. Your `instructions.md` and skills become **deploy-owned
# context in Context Hub**, versioned outside your git repo — convenient for hot-editing a prompt
# without a redeploy, awkward for an organisation that wants every behavioural change to arrive through a
# reviewed commit. Pick which you are optimising for before adopting it.
#
# **Post-deploy checklist** (from the tutorial):
#
# - [ ] the deployment reaches *ready* in the dashboard
# - [ ] `weekly_rescreen` appears as a scheduled job
# - [ ] a test chat run shows model calls, tool calls and memory reads in the trace
# - [ ] the first scheduled run is queued for the next Monday 07:00 Europe/Amsterdam

# %% [markdown]
# ## 9. Choosing a deployment path
#
# 📖 [Deployments / CLI](https://docs.langchain.com/langsmith/cli)
#
# | If you need… | Use |
# |---|---|
# | EU / air-gapped hosting, or full control of the harness | `langgraph deploy` self-hosted or hybrid (Lab 01) |
# | Fastest path to a hosted deep agent, US region acceptable | **MDA** (`mda deploy`) |
# | Unattended recurring runs with managed memory | **MDA** + `define_schedule` + `define_memory` |
# | Standard benchmarks across models | **Harbor** (`mda eval compile`, or Harbor directly) |
# | Bespoke evals over your own data | LangSmith datasets + `openevals`/`agentevals` (Day 2 Lab 06) |
#
# On a US workspace MDA is genuinely usable today, and scheduled re-screening is the strongest
# argument for it. The open question for Acme is data residency, not capability.
#
# ## 10. Recap
#
# | Concept | Day 2 equivalent | MDA |
# |---|---|---|
# | Operating instructions | `AGENTS.md` + `MemoryMiddleware` | `instructions.md` |
# | Durable memory | `FilesystemBackend` / store | `define_memory(scope="agent")` → Context Hub |
# | Skills | `skills/*/SKILL.md` + `SkillsMiddleware` | `skills/*/SKILL.md`, automatic |
# | Identity / auth | `context_schema` + `ToolRuntime` | `define_identity` in `identity.py` |
# | Scheduled runs | LangGraph cron | `define_schedule` |
# | Evals | datasets + `openevals`/`agentevals` | `mda eval` → Harbor (pytest verifiers) |
# | Deploy | `langgraph deploy` | `mda deploy` |
#
# **Next:** the Day 3 recap ties the deployment paths back to governance — who owns the harness, where
# context lives, and which of those a regulated organisation can accept.
