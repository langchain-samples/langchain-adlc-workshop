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
# # 02 · Wiki Memory + Context Patterns — Knowledge That Persists Across Runs
#
# **Workshop:** LangChain ADLC Workshop · **Day 2** · **ADLC stage:** Build / Improve
#
# > **Loop Engineering focus: Verification loop** — an agent that accumulates evidence, risk signals,
# > and durable notes across runs is easier to verify: you can inspect *what it believes* and check it
# > against the source evidence.
#
# > Self-directed module · ~60 min
#
# ```mermaid
# graph LR
#     A[Run 1: assess Vendor A] -->|writes| W[Wiki memory files]
#     B[Run 2: assess Vendor B] -->|writes| W
#     W -->|read at startup| C[Run 3: reassess Vendor A]
#     C -->|already knows prior findings| D[Faster, consistent verdict]
#     E[Day 1 vendor KB] -->|retrieved evidence| A
#     E -->|retrieved evidence| B
# ```
#
# Day 1's agent was **stateless**: every run started from zero. A due diligence agent that reassesses
# the same vendors month after month needs to **remember** — prior findings, evidence quality, open
# questions, risk signals. This lab builds that with the **wiki memory pattern**:
#
# | Memory file | Holds | Written when |
# |---|---|---|
# | `/wiki/vendors/{vendor_id}.md` | Durable per-vendor notes: evidence summary, verification status | After each assessment |
# | `/wiki/risk_signals.md` | Cross-vendor risk signals with severity | A new signal surfaces |
# | `/wiki/open_questions.md` | Unresolved follow-ups for human review | Confidence is low / evidence gaps |
# | `AGENTS.md` | Operating instructions (read-mostly, curated by humans) | Rarely — a process lesson learned |
#
# By the end you can:
# - Wire **filesystem-backed memory** into a deep agent with `MemoryMiddleware` (`AGENTS.md`)
# - Use a **wiki memory pattern**: durable notes, evidence summaries, and risk signals that persist
#   across runs and threads
# - Show how the agent **accumulates knowledge across runs** (run 2 reads what run 1 wrote)
# - Explain the **Context Hub** (managed context for production) and how it replaces local files
# - Distinguish **context** (instructions + memory the agent carries) from **retrieved knowledge**
#   (evidence pulled per-query from the KB)
#
# > 🧭 **Builds on Lab 01 (`01_deep_agent.py`); runs standalone.** All vendor data is
# > **synthetic/fictional** — created for this workshop, no real Acme data.


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

DAY1_DATA = WORKSHOP / "day1" / "data"          # vendor KB articles + vendors.json
DAY2_DATA = WORKSHOP / "day2" / "data"          # agent/AGENTS.md
AGENT_DIR = DAY2_DATA / "agent"                                # AGENTS.md lives here
WIKI_DIR = DAY2_DATA / "wiki"                                  # durable memory files (created on demand)
WIKI_DIR.mkdir(parents=True, exist_ok=True)

print("workspace:", os.getenv("LANGSMITH_WORKSPACE_ID"), "| tracing:", os.getenv("LANGSMITH_TRACING"))
print("AGENTS.md:", AGENT_DIR / "AGENTS.md", "| exists:", (AGENT_DIR / "AGENTS.md").exists())
print("wiki dir:", WIKI_DIR)

# %% [markdown]
# ## 1. Two kinds of "knowledge": context vs retrieval
#
# Before building, pin down the distinction this whole lab rests on:
#
# | | **Context** (carried) | **Retrieved knowledge** (pulled) |
# |---|---|---|
# | What | Instructions, memory, notes the agent *carries into every run* | Evidence pulled *per query* from an external source |
# | Examples | `AGENTS.md` operating rules, wiki notes on past assessments, risk-signal log | Vendor KB articles, `vendors.json` records, web search results |
# | Lifetime | **Persists across runs** — survives thread boundaries | **Per-run** — fetched fresh, then discarded |
# | Written by | The agent itself (learned) + humans (curated) | Data owners (KB authors, vendor DB) |
# | Failure mode | Stale or self-reinforcing beliefs | Missed or hallucinated evidence |
# | In Deep Agents | `memory=[...]` + filesystem/store backends | Tools: `search_vendor_kb`, `get_vendor` |
#
# > 💡 **Why it matters for due diligence:** an evidence citation ("Quelmore holds AS9100D per
# > `quelmore_systems.md`") is *retrieved knowledge* — re-verify it every run. "We already flagged
# > Kelbrant Trading for sanctions exposure on 2024-11-03" is *context* — carrying it forward stops the
# > agent from re-deriving (or forgetting) a known risk. Confusing the two is how agents either
# > hallucinate citations from memory, or re-litigate settled findings every run.
#
# The **wiki memory pattern** sits in the context column: durable, human-readable, source-grounded
# notes the agent reads at startup and appends to as it works — like a wiki page a research team
# maintains, not a chat log.

# %% [markdown]
# ## 2. Retrieval side — Day 1 vendor KB tool
#
# 📖 [Retrieval](https://docs.langchain.com/oss/python/langchain/retrieval)
#
# The retrieval half is the same pattern as Day 1: chunk the 13 synthetic vendor KB articles, embed
# them in an in-memory store, and return cited snippets. This is **retrieved knowledge** — fresh per
# query, never stored in memory verbatim.

# %%
import json

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from day1.src.models import get_embeddings  # routes to gateway or direct provider — see the README (Model access section)

docs = [
    Document(page_content=p.read_text(), metadata={"source": p.name})
    for p in sorted((DAY1_DATA / "kb").glob("*.md"))
]
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
kb_index = InMemoryVectorStore.from_documents(chunks, get_embeddings())
print(f"indexed {len(chunks)} chunks from {len(docs)} vendor KB articles")

VENDORS = {k: v for k, v in json.loads((DAY1_DATA / "vendors.json").read_text()).items() if not k.startswith("_")}
print(f"loaded {len(VENDORS)} vendor records")

# %%
from langchain_core.tools import tool


@tool
def search_vendor_kb(query: str) -> str:
    """Search vendor profile KB articles for capability and certification evidence.
    Returns cited snippets — always includes the source filename so you can cite it."""
    hits = kb_index.similarity_search(query, k=4)
    if not hits:
        return "No relevant KB articles found. Say so — do not guess."
    return "\n\n---\n\n".join(f"[source: {h.metadata['source']}]\n{h.page_content}" for h in hits)


@tool
def get_vendor(vendor_id: str) -> str:
    """Look up the structured vendor record (certifications, contract history, risk level,
    compliance flags) by vendor_id, e.g. 'VND-001'."""
    v = VENDORS.get(vendor_id.upper())
    if not v:
        return f"No vendor record for {vendor_id}. Known IDs: {sorted(VENDORS)[:5]} ..."
    # Redact contact PII — the AGENTS.md operating rules forbid PII in assessments.
    safe = {k: val for k, val in v.items() if k != "contact_email"}
    return json.dumps(safe, indent=2)


# %%
print(search_vendor_kb.invoke({"query": "AS9100D aerospace certification"})[:400])
print()
print(get_vendor.invoke({"vendor_id": "VND-001"})[:300])

# %% [markdown]
# ### 2b. Retrieval-quality checks — measure the retriever before you blame the agent
#
# When a due diligence report is wrong, the instinct is to fix the prompt. Often the retriever
# never surfaced the right document, and no prompt can recover from that. So measure the retrieval
# step **on its own**, against a small labelled set of query → which article *should* come back.
#
# Two numbers, both `@k` because a retriever returns a ranked list:
#
# | Metric | Question | Fix when it is low |
# |---|---|---|
# | **Recall@k** | Did the right document appear anywhere in the top *k*? | Chunking, embeddings, k, query rewriting |
# | **Precision@k** | What share of the *k* returned were relevant? | Ranking, filters, smaller k |
#
# Recall@k is the one that bounds everything downstream: a document that never enters the context
# cannot be cited, no matter how good the model is. Precision costs you tokens and invites the model
# to cite something adjacent-but-wrong.

# %%
# A labelled set: query → the KB article(s) that genuinely answer it. Small and hand-checked, which
# is the right size to start — an unreviewed large set measures nothing you can trust.
RETRIEVAL_GOLD: list[tuple[str, set[str]]] = [
    ("AS9100D aerospace quality certification", {"quelmore_systems.md", "norwick_aerospace.md"}),
    ("cyber security incident response capability", {"draymast_cyber.md"}),
    ("cold chain logistics and temperature controlled transport", {"skarloft_industries.md"}),
    ("satellite and field radio communications", {"thalveyn_comms.md"}),
    ("autonomous ground vehicles and robotics", {"dravene_robotics.md"}),
    ("environmental remediation and site cleanup", {"yssric_environmental.md"}),
]


def retrieval_quality(k: int = 4) -> dict:
    """Recall@k and precision@k over the labelled set. Pure measurement — no model involved."""
    recalls, precisions, misses = [], [], []
    for query, expected in RETRIEVAL_GOLD:
        hits = kb_index.similarity_search(query, k=k)
        got = {h.metadata["source"] for h in hits}
        found = expected & got
        recalls.append(len(found) / len(expected))
        precisions.append(len(found) / max(len(got), 1))
        if not found:
            misses.append((query, sorted(got)))
    n = len(RETRIEVAL_GOLD)
    return {
        "k": k,
        "recall@k": sum(recalls) / n,
        "precision@k": sum(precisions) / n,
        "complete_misses": misses,
    }


print(f"{'k':>3}  {'recall@k':>9}  {'precision@k':>12}")
for k in (1, 2, 4, 8):
    m = retrieval_quality(k)
    print(f"{k:>3}  {m['recall@k']:>9.２f}  {m['precision@k']:>12.2f}".replace("２", "2"))

m4 = retrieval_quality(4)
print("\ncomplete misses at k=4 (nothing relevant returned at all):")
for q, got in m4["complete_misses"]:
    print(f"  ⛔ {q!r}\n     returned instead: {got}")
if not m4["complete_misses"]:
    print("  none — every labelled query surfaced at least one correct article")

# %% [markdown]
# **Read the table, not the average.** Recall climbs with `k` and precision falls — that trade is
# the whole design decision. Picking `k` is choosing how much irrelevant context you will pay for
# (in tokens, and in the model's temptation to cite it) to stop missing the right document.
#
# A complete miss is the finding that matters. It means no prompt change can fix that query, and it
# sends you to the retriever: chunk size, embedding model, or a query-rewriting step.
#
# > This is the offline, code-based half of eval. Day 2 Lab 03 puts the *agent* under test; this
# > measures the component underneath it, which is where retrieval bugs actually live.

# %% [markdown]
# ### 2c. Source-quality scoring — not all evidence deserves equal weight
#
# Retrieval tells you *what came back*. Source-quality scoring tells you *how much to trust it*, and
# gives the agent a defensible way to rank conflicting evidence — the `source-validation` skill in
# `day2/data/agent/skills/` encodes the same order as a workflow the agent can follow.

# %%
# Deterministic, auditable scoring. A rubric in code beats a judgement in a prompt whenever the
# rule is this crisp: the same source always scores the same, and you can point at the line.
SOURCE_TIERS: dict[str, tuple[int, str]] = {
    "sanctions_screening": (5, "authoritative — live compliance system"),
    "vendor_database":     (4, "structured record, maintained by Acme"),
    "capability_pdf":      (3, "vendor-issued but documentary"),
    "vendor_kb":           (3, "curated profile article"),
    "web_search":          (2, "unvetted, may be marketing"),
    "model_recall":        (0, "NOT a source — must never appear in a report"),
}


def score_source(source_name: str) -> tuple[int, str]:
    """Map a citation to a trust tier. Unknown sources score 1, never 0 — 0 is reserved for
    'the model made this up', which is a different and more serious problem."""
    s = source_name.lower()
    if "screen" in s or "sanction" in s:
        return SOURCE_TIERS["sanctions_screening"]
    if "vendor database" in s or s.startswith("vnd-"):
        return SOURCE_TIERS["vendor_database"]
    if s.endswith(".pdf"):
        return SOURCE_TIERS["capability_pdf"]
    if s.endswith(".md"):
        return SOURCE_TIERS["vendor_kb"]
    if s.startswith("http"):
        return SOURCE_TIERS["web_search"]
    return (1, "unrecognised source — treat as unverified")


for src in ["sanctions screening", "VND-001", "quelmore_systems.md",
            "quelmore_capability.pdf", "https://example.com/press-release", "general knowledge"]:
    score, why = score_source(src)
    bar = "█" * score + "·" * (5 - score)
    print(f"  {bar}  {score}/5  {src:<38} {why}")

print(
    "\nUse it two ways: rank conflicting evidence by tier, and refuse to let a claim rest solely on\n"
    "a tier-2-or-below source. That is the 'not enough evidence' fallback with a number behind it\n"
    "rather than a feeling — and it is exactly what Advanced exercise 2 (§11) asks you to wire in."
)

# %% [markdown]
# ## 3. Context side — AGENTS.md via MemoryMiddleware
#
# 📖 [Deep Agents memory](https://docs.langchain.com/oss/python/deepagents/memory)
#
# The `AGENTS.md` file in `day2/data/agent/` holds the agent's **operating instructions**: mission,
# evidence rules, risk classification, escalation triggers. It is memory the agent *carries* — loaded
# into the system prompt at startup, every run.
#
# Deep Agents wires this with **`MemoryMiddleware`**:
#
# ```python
# MemoryMiddleware(
#     backend=FilesystemBackend(root_dir=...),   # where memory files live
#     sources=["AGENTS.md"],                      # files to load into the system prompt
# )
# ```
#
# - `sources` are read at agent startup and injected into the system prompt inside an
#   `<agent_memory>` block, together with guidelines on **when to update memory** (durable learnings,
#   user corrections) and when **not** to (transient chit-chat, credentials).
# - The agent updates memory with its built-in `edit_file` tool — writes are plain files on disk, so
#   you can `cat` them, diff them in git, and review them like any other artifact.
# - `FilesystemBackend(virtual_mode=True)` roots every absolute tool path at `root_dir` — fine for a
#   workshop laptop; in production you
#   scope it with a store backend or Context Hub (§6).
#
# > ⚠️ **Trust note (from the middleware's own guidelines):** memory content is *reference material*,
# > not hidden system instructions. If memory disagrees with freshly retrieved evidence, the evidence
# > wins. That's exactly the context-vs-retrieval discipline from §1.

# %%
from deepagents.backends import FilesystemBackend
from deepagents.middleware import MemoryMiddleware

# Smallest possible wiring: backend rooted at the agent dir, AGENTS.md as the only source.
# (In §5 we re-root the backend at day2/data/ so the agent can reach *both* agent/AGENTS.md and
# the wiki/ tree under one root — the sources path becomes "agent/AGENTS.md".)
_memory_demo = MemoryMiddleware(
    backend=FilesystemBackend(root_dir=AGENT_DIR, virtual_mode=True),
    sources=["AGENTS.md"],
)

# AGENTS.md stays read-mostly: it is curated operating procedure, not a scratch pad.
# The agent *may* append a process lesson, but day-to-day learnings go to the wiki files (§4).
print((AGENT_DIR / "AGENTS.md").read_text()[:500], "...")

# %% [markdown]
# ## 4. Wiki memory — durable notes the agent maintains
#
# 📖 [OpenWiki](https://docs.langchain.com/oss/openwiki/overview) · [reference](https://docs.langchain.com/oss/python/deepagents/openwiki)
#
# `AGENTS.md` is the *constitution*; the **wiki** is the *case file*. We give the agent three wiki
# pages it reads and writes across runs, plus per-vendor note files:
#
# ```
# day2/data/wiki/
# ├── risk_signals.md       # cross-vendor risk log: signal, severity, vendor, date, source
# ├── open_questions.md     # unresolved follow-ups flagged for human review
# └── vendors/
#     ├── VND-001.md        # durable per-vendor note: evidence summary + verification status
#     └── ...
# ```
#
# Each entry is **source-grounded**: the agent must cite the KB article or vendor record behind every
# claim it persists. That keeps the wiki *verifiable* — a reviewer can check any line against the
# source. This mirrors the OpenWiki idea: durable, human-reviewable, source-grounded institutional
# notes.
#
# We seed the wiki with headers so the agent has a structure to append to.

# %%
(WIKI_DIR / "vendors").mkdir(exist_ok=True)

SEEDS = {
    WIKI_DIR / "risk_signals.md": (
        "# Risk signals — cross-vendor log\n\n"
        "| Date | Vendor | Signal | Severity | Source |\n"
        "|---|---|---|---|---|\n"
    ),
    WIKI_DIR / "open_questions.md": (
        "# Open questions — for human review\n\n"
        "<!-- Append unresolved follow-ups. Include vendor, question, and why evidence was insufficient. -->\n\n"
    ),
}
for path, text in SEEDS.items():
    if not path.exists():
        path.write_text(text)
        print("seeded", path.name)
    else:
        print("exists ", path.name)

# %% [markdown]
# ## 5. Build the agent — memory + filesystem tools
#
# 📖 [Agents (create_agent)](https://docs.langchain.com/oss/python/langchain/agents) · [reference](https://reference.langchain.com/python/langchain/agents/factory/create_agent)
#
# `create_deep_agent` gives us the harness: built-in filesystem tools (`read_file`, `write_file`,
# `edit_file`, `ls`), a todo list, and sub-agent support. We add:
#
# 1. **`memory=["AGENTS.md"]`-equivalent via middleware** — `MemoryMiddleware` with our
#    `FilesystemBackend` rooted at `day2/data/agent/`
# 2. **Wiki access** — the same backend rooted so the agent can read/write `/wiki/...` files; here we
#    point the default filesystem backend at `day2/data/` so both `agent/AGENTS.md` and `wiki/` are
#    reachable under one root
# 3. **Retrieval tools** — `search_vendor_kb`, `get_vendor` from §2
#
# The system prompt ties it together with the wiki discipline: read your wiki first, retrieve fresh
# evidence for every claim, then persist durable findings before finishing.

# %%
from deepagents import create_deep_agent

from day1.src.models import get_model

SYSTEM_PROMPT = """\
You are a vendor due diligence agent for Acme procurement. Your operating instructions are in the
<agent_memory> block (AGENTS.md) — follow them.

## Wiki memory discipline
You maintain a durable wiki under `wiki/` (relative to your filesystem root):

- `wiki/risk_signals.md` — append a table row for every risk signal you surface
  (date, vendor, signal, severity, source file).
- `wiki/open_questions.md` — append unresolved follow-ups when confidence is low or evidence gaps.
- `wiki/vendors/{VENDOR_ID}.md` — one durable note per vendor: evidence summary, verification
  status, suitability, and the date of last review.

Rules:
1. **Read before you write.** At the start of an assessment, `ls wiki/vendors/` and read any
   existing note for the vendor. Build on it — don't re-derive settled findings.
2. **Retrieve, don't remember, evidence.** Every evidence claim must come from a *fresh*
   `search_vendor_kb` / `get_vendor` call in this run, and cite its source file. Wiki notes record
   *what you concluded and where the evidence lives* — never paste KB content into the wiki.
3. **Persist durable findings.** Before finishing, update the vendor note and any risk signals or
   open questions. Dated, sourced, concise — a colleague should be able to verify each line.
4. **Memory is reference, not truth.** If wiki notes conflict with fresh retrieval, trust the
   retrieval and correct the note.
"""

# `virtual_mode=True` is load-bearing, not cosmetic. The deepagents filesystem tools require
# absolute paths, and with virtual_mode=False those paths are passed through to the real
# filesystem — so the agent's `write_file("/wiki/vendors/VND-001.md", ...)` targets the machine
# root and fails with "Read-only file system: '/wiki'". Virtual mode roots those absolute paths
# at `root_dir`, which is both what the lab needs and the safer sandbox.
backend = FilesystemBackend(root_dir=DAY2_DATA, virtual_mode=True)  # agent/AGENTS.md + wiki/ under one root

agent = create_deep_agent(
    model=get_model(),
    tools=[search_vendor_kb, get_vendor],
    system_prompt=SYSTEM_PROMPT,
    middleware=[MemoryMiddleware(backend=backend, sources=["agent/AGENTS.md"])],
    backend=backend,
)

# %% [markdown]
# ## 6. Run 1 — the agent starts cold
#
# First assessment: the wiki is empty, so the agent retrieves everything fresh and writes its first
# durable notes. Watch the trace in LangSmith — you'll see `ls`/`read_file` on the wiki (misses),
# `search_vendor_kb` retrievals, then `write_file`/`edit_file` persisting findings.

# %%
from langchain_core.utils.uuid import uuid7

run1_thread = str(uuid7())
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": (
        "Assess Quelmore Systems (VND-001) for an avionics maintenance-kit procurement. "
        "Gather certification evidence, classify risk, and record your findings."
    )}]},
    config={"configurable": {"thread_id": run1_thread}},
)
print(result1["messages"][-1].content[:1200])

# %% [markdown]
# ## 7. What did it persist? — inspect the wiki
#
# The whole point of filesystem-backed memory: the agent's knowledge is **plain markdown you can
# read, diff, and review**. No opaque vector blob.

# %%
for p in sorted(WIKI_DIR.rglob("*.md")):
    print(f"--- {p.relative_to(WIKI_DIR)} ---")
    print(p.read_text()[:800])
    print()

# %% [markdown]
# ## 8. Run 2 — a *new thread* reads what run 1 wrote
#
# Now the payoff. A fresh thread (no shared conversation state) assesses a **different** vendor, then
# we ask about Quelmore again. The agent's wiki notes survive the thread boundary: run 2 starts from
# accumulated knowledge instead of zero.
#
# This is the "accumulates knowledge across runs" property:
#
# | | Stateless agent (Day 1) | Wiki-memory agent (this lab) |
# |---|---|---|
# | Run 2 starts with | Nothing | Prior notes, risk log, open questions |
# | Re-assessment cost | Full re-retrieval | Read note → verify deltas only |
# | Cross-vendor patterns | Re-derived every time | Carried forward (risk_signals.md) |
# | Auditability | Trace only | Trace **+ diffable markdown files** |

# %%
run2_thread = str(uuid7())
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": (
        "New task. First, assess Thalveyn Comms (look it up in the KB) for a field-communications "
        "procurement. Then tell me: what do you already know about Quelmore Systems, and what — if "
        "anything — would you need to re-verify before relying on it?"
    )}]},
    config={"configurable": {"thread_id": run2_thread}},
)
print(result2["messages"][-1].content[:1500])

# %%
# Wiki state after two runs — notice accumulation: two vendor notes, risk log rows, open questions.
for p in sorted(WIKI_DIR.rglob("*.md")):
    print(f"--- {p.relative_to(WIKI_DIR)} ({len(p.read_text())} chars) ---")

# %% [markdown]
# ## 9. Context Hub — managed context for production
#
# 📖 [Context engineering](https://docs.langchain.com/oss/python/deepagents/context-engineering)
#
# Everything above stores memory as **local files**. That's right for a workshop laptop and wrong for
# production: no sharing across replicas, no access control, no versioning, no audit trail.
#
# **Context Hub** is the managed answer: agent context (AGENTS.md, skills, memory files) lives in a
# **LangSmith Hub agent repo** — versioned (every write is a commit), shareable, and manageable from
# the LangSmith UI/API. Deep Agents reads it through `ContextHubBackend`:
#
# ```python
# from deepagents.backends import ContextHubBackend
#
# backend = ContextHubBackend("my-org/vendor-due-diligence-agent")   # hub repo: "owner/name"
# agent = create_deep_agent(
#     model=...,
#     tools=[...],
#     middleware=[MemoryMiddleware(backend=backend, sources=["AGENTS.md"])],
#     backend=backend,
# )
# ```
#
# | | `FilesystemBackend` (this lab) | `ContextHubBackend` (production) |
# |---|---|---|
# | Storage | Local disk | LangSmith Hub agent repo |
# | Versioning | None (use git yourself) | Every write is a commit |
# | Sharing | One machine | Any deployment / teammate with access |
# | Review | `cat` / git diff | LangSmith UI, commit history |
# | Access control | Filesystem perms | LangSmith workspace permissions |
# | Good for | Dev, workshop, single-node | Deployed agents, teams, audit |
#
# Migration is a one-line backend swap — the wiki pattern, the `AGENTS.md`, and the system-prompt
# discipline are identical. That's the design lesson: **pick the memory *pattern* first; the storage
# backend is interchangeable.**
#
# > 🔗 Docs: [Deep Agents memory](https://docs.langchain.com/oss/python/deepagents/memory) ·
# > [Context engineering](https://docs.langchain.com/oss/python/deepagents/context-engineering) ·
# > [OpenWiki](https://docs.langchain.com/oss/openwiki/overview)
#
# **Scoped memory for multi-user production:** the memory docs describe agent-scoped, user-scoped,
# and org-scoped namespaces via `StoreBackend` — e.g. per-analyst assessment notes that never leak
# between users, plus a read-only org policy file. Same pattern as this lab, different namespace.

# %% [markdown]
# ## 10. Context vs retrieved knowledge — the contrast, demonstrated
#
# Close the loop on §1 with a concrete check. Ask the agent something that only works if it keeps the
# two apart:
#
# - **From context (wiki):** "what did we conclude last time?" → answered from `wiki/vendors/VND-001.md`
# - **From retrieval (KB):** "what certifications does Quelmore hold?" → must come from a fresh
#   `search_vendor_kb` call, cited to `quelmore_systems.md`
#
# If the agent answers the second from memory without a tool call, it has confused context with
# knowledge — the exact failure mode the system-prompt rules and AGENTS.md anti-hallucination rule
# guard against. Check the trace: you should see `read_file` (wiki) **and** `search_vendor_kb` (KB).

# %%
run3_thread = str(uuid7())
result3 = agent.invoke(
    {"messages": [{"role": "user", "content": (
        "Two questions, keep them separate: (1) From your notes — what did we conclude about "
        "Quelmore's suitability and when? (2) From the source evidence — list Quelmore's "
        "certifications with citations. If your notes and the evidence disagree, say so."
    )}]},
    config={"configurable": {"thread_id": run3_thread}},
)
print(result3["messages"][-1].content[:1500])

# %% [markdown]
# ## 11. Advanced exercise 2 — improve retrieval or context
#
# **Format:** ~20 minutes on your own, then a 10-minute walkthrough.
#
# Two facts about the retrieval tool you just used:
#
# 1. Every vendor KB article is written to the same five facets — **Overview, Certifications,
#    Contract history, Capabilities, Risk assessment**. An assessment that cites only the Overview
#    is not wrong, it is *thin*, and nothing in the agent notices the difference.
# 2. `similarity_search(query, k=6)` **always returns six chunks.** Ask about a vendor that is not in
#    the corpus at all and you still get six confident-looking snippets from other vendors. Run
#    `kb_index.similarity_search("Zephyr Quantum Shipbuilding hull tonnage", k=6)` and look at the
#    sources — that is the single most common way a grounded agent still hallucinates.
#
# So a useful retrieval-quality check needs **both** halves: is the evidence *relevant*, and does it
# *cover* what a due diligence answer requires.
#
# **Your task:** finish `assess_retrieval()` below. The relevance floor and the facet-coverage
# computation are written; the **verdict rule is missing**.
#
# | Facets missing (after the relevance floor) | Verdict | What the agent must do |
# |---|---|---|
# | 0 | `sufficient` | answer normally |
# | 1 | `partial` | answer, but name the gap in the assessment |
# | 2 or more (or nothing clears the floor) | `insufficient` | do **not** assess — log an open question |
#
# **Other options** (pick one instead — each is a self-contained ~20 min change):
#
# | Option | What to change |
# |---|---|
# | A. Relevance + coverage gate *(the starter below)* | Refuse to assess on irrelevant or thin evidence |
# | B. Source freshness | Add a `last_verified` field to the vendor note template; re-verify anything older than N days |
# | C. `AGENTS.md` rule | Add "medium-risk vendors always get a supply-chain-continuity open question", then re-run §6–§8 and diff the wiki |
# | D. Tool-description tuning | Rewrite `search_vendor_kb`'s docstring to steer the agent toward facet-specific queries, and compare the trajectories |
#
# **Done when:** the self-check prints all ✅ — including the case where six chunks come back and the
# answer is still "insufficient evidence".

# %%
# --- STARTER -------------------------------------------------------------------------------
# The five facets every vendor KB article is written to. Coverage is measured against these.
DD_FACETS = ("overview", "certifications", "contract history", "capabilities", "risk assessment")

# Cosine similarity below this is noise, not evidence. Calibrated against this corpus: a real vendor
# query tops out around 0.63–0.71, a query for a vendor that does not exist peaks near 0.40.
# Re-calibrate whenever the corpus or the embedding model changes — this number is not universal.
RELEVANCE_FLOOR = 0.45


def assess_retrieval(scored_hits) -> dict:
    """Turn scored retrieval results into a coverage verdict.

    `scored_hits` is what `similarity_search_with_score` returns: a list of (Document, score)
    pairs, highest similarity first.
    """
    relevant = [doc for doc, score in scored_hits if score >= RELEVANCE_FLOOR]
    covered = sorted({
        facet for facet in DD_FACETS
        if any(facet in doc.page_content.lower() for doc in relevant)
    })
    missing = [f for f in DD_FACETS if f not in covered]

    # TODO(exercise): set `verdict` from the table above — using `relevant`, not `scored_hits`.
    #   The whole point is that six retrieved chunks can still mean zero evidence.
    verdict = "sufficient"  # ← replace me

    return {"relevant": len(relevant), "retrieved": len(scored_hits),
            "covered": covered, "missing": missing, "verdict": verdict}


# %%
# --- SELF-CHECK: deterministic, no embedding calls -----------------------------------------
from langchain_core.documents import Document


def _hit(text: str, score: float):
    return (Document(page_content=text, metadata={"source": "test.md"}), score)


COVERAGE_CASES = [
    ("full profile, all relevant",
     [_hit("## Overview\nA supplier.", 0.71), _hit("## Certifications\nAS9100D", 0.68),
      _hit("## Contract history\nAcme 2023", 0.62), _hit("## Capabilities\nAvionics", 0.58),
      _hit("## Risk assessment\nRisk level: Low", 0.55)],
     "sufficient"),
    ("one gap — no contract history",
     [_hit("## Overview\nA supplier.", 0.70), _hit("## Certifications\nAS9100D", 0.66),
      _hit("## Capabilities\nAvionics", 0.60), _hit("## Risk assessment\nRisk level: Low", 0.52)],
     "partial"),
    ("thin — overview and certifications only",
     [_hit("## Overview\nA supplier.", 0.69), _hit("## Certifications\nAS9100D", 0.64)],
     "insufficient"),
    ("nothing retrieved",
     [],
     "insufficient"),
    ("six chunks, none relevant — the vendor is not in the corpus",
     [_hit("## Overview\nSome other supplier.", 0.40), _hit("## Certifications\nISO 9001", 0.36),
      _hit("## Contract history\nEDA 2022", 0.35), _hit("## Capabilities\nRobotics", 0.31),
      _hit("## Risk assessment\nRisk level: Medium", 0.31), _hit("## Overview\nAnother.", 0.30)],
     "insufficient"),
]


def check_coverage(fn) -> bool:
    ok = True
    for label, hits, expected in COVERAGE_CASES:
        got = fn(hits)
        passed = got["verdict"] == expected
        ok = ok and passed
        print(f"{'✅' if passed else '❌'} {label:52} → {got['verdict']:13} "
              f"({got['relevant']}/{got['retrieved']} relevant, missing={len(got['missing'])})")
        if not passed:
            print(f"   expected verdict: {expected}")
    return ok


print("all cases pass:", check_coverage(assess_retrieval))

# %% [markdown]
# ### Solution walkthrough
#
# The verdict is four lines. Three things about it matter more than the arithmetic:
#
# 1. **Relevance first, coverage second.** Filter, *then* count facets. Reverse the order and the
#    last self-check case passes for the wrong reason: six irrelevant chunks happen to mention every
#    facet, so coverage alone declares the evidence complete.
# 2. **The gate belongs in the tool, not the prompt.** "Say so when you don't have evidence" in a
#    system prompt is a request. A tool that returns `INSUFFICIENT EVIDENCE` and no snippets is a
#    constraint — the agent has nothing to hallucinate *from*. Push the rule down the stack.
# 3. **`insufficient` is not an error.** The agent still has work: record an open question in the
#    wiki so the gap becomes a durable, human-reviewable artifact rather than a dropped run. That is
#    the wiki-memory discipline from §4 applied to a failure path.

# %%
def assess_retrieval_solution(scored_hits) -> dict:
    """Reference solution."""
    relevant = [doc for doc, score in scored_hits if score >= RELEVANCE_FLOOR]
    covered = sorted({
        facet for facet in DD_FACETS
        if any(facet in doc.page_content.lower() for doc in relevant)
    })
    missing = [f for f in DD_FACETS if f not in covered]

    if not relevant or len(missing) >= 2:
        verdict = "insufficient"
    elif missing:
        verdict = "partial"
    else:
        verdict = "sufficient"

    return {"relevant": len(relevant), "retrieved": len(scored_hits),
            "covered": covered, "missing": missing, "verdict": verdict}


print("all cases pass:", check_coverage(assess_retrieval_solution))

# %% [markdown]
# **The gated tool.** Same retrieval, but the verdict rides along with the snippets — and on
# `insufficient` the tool withholds them and tells the agent what to do instead.

# %%
@tool
def search_vendor_kb_gated(query: str) -> str:
    """Search vendor profile KB articles for capability and certification evidence.

    Returns cited snippets plus a COVERAGE line naming which due diligence facets the evidence
    actually supports (overview, certifications, contract history, capabilities, risk assessment).
    When the evidence is irrelevant or too thin this tool returns NO snippets: record an open
    question in wiki/open_questions.md and report insufficient evidence instead of assessing.
    """
    scored = kb_index.similarity_search_with_score(query, k=6)
    report = assess_retrieval_solution(scored)

    if report["verdict"] == "insufficient":
        return (
            f"INSUFFICIENT EVIDENCE for {query!r}.\n"
            f"retrieved {report['retrieved']} chunk(s), {report['relevant']} above the relevance floor.\n"
            f"covered facets: {report['covered'] or 'none'} · missing: {report['missing']}\n"
            "Do NOT assess this vendor. Append an entry to wiki/open_questions.md naming the "
            "missing facets, and report insufficient evidence."
        )

    snippets = "\n\n---\n\n".join(
        f"[source: {doc.metadata['source']}]\n{doc.page_content}"
        for doc, _ in scored[:4]
    )
    return (
        f"COVERAGE: {report['verdict']} — covered {report['covered']}, missing {report['missing']} "
        f"({report['relevant']}/{report['retrieved']} chunks above the relevance floor)\n\n"
        f"{snippets}"
    )


# %%
# A vendor the corpus covers, vs one it does not. Both retrieve six chunks; only one is evidence.
print("— Quelmore Systems (in the corpus) —")
print(search_vendor_kb_gated.invoke(
    {"query": "Quelmore Systems certifications contract history risk"})[:280])
print()
print("— Zephyr Quantum Shipbuilding (not in the corpus) —")
print(search_vendor_kb_gated.invoke(
    {"query": "Zephyr Quantum Shipbuilding hull fabrication tonnage"})[:420])

# %% [markdown]
# > 💡 Whichever option you picked, keep the wiki discipline: durable findings in the wiki, evidence
# > by fresh retrieval, every persisted claim source-grounded — and now, every *gap* recorded too.

# %% [markdown]
# ## 12. Recap
#
# - **Wiki memory pattern** — durable, human-readable, source-grounded markdown notes
#   (`risk_signals.md`, `open_questions.md`, `vendors/{id}.md`) that persist across runs; the agent
#   reads them at startup and appends as it works.
# - **Accumulation across runs** — run 2 (new thread) started from run 1's notes: faster
#   re-assessment, carried-forward risk signals, diffable audit trail.
# - **AGENTS.md via `MemoryMiddleware`** — operating instructions loaded into the system prompt from
#   a filesystem-backed memory file; reference material, not hidden truth.
# - **Context Hub** — the production path: same pattern, but memory lives in a versioned LangSmith
#   Hub repo via `ContextHubBackend`. Swap the backend, keep the pattern.
# - **Context ≠ retrieved knowledge** — context is carried (memory, instructions); knowledge is
#   pulled per query (KB, vendor records). Evidence claims always come from fresh retrieval with
#   citations; conclusions get persisted. Mixing them up is how agents hallucinate.
#
# **Next:** `03_evals_datasets.py` — build the eval dataset and graders that verify this agent's
# evidence quality, escalation behavior, and source grounding.
