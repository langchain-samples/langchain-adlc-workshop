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
# # 06 · Knowledge-base and large-data architecture
#
# **Workshop:** LangChain ADLC Workshop · **Day 3** · **ADLC stage:** Design + Govern
#
# > **Agenda slot:** 14:30–15:15 — *Demo + discussion: knowledge-base and large-data best
# > practices*. It sits between the governance demo (`05_governance.py` §1–3) and the
# > deployment/Fleet/Engine demo (`05_governance.py` §4–6), which is why the numbering runs
# > 05 → 06 → back into 05. Run it standalone; it depends on no other Day 3 lab.
#
# > **Demo walkthrough** (presenter-led) · ~45 min · one small model call at the end
#
# The single most common production failure in this space is not a bad embedding model. It is
# choosing **one** retrieval architecture and applying it to data it does not fit. This module is
# the decision procedure, measured against the workshop's own corpus rather than asserted.
#
# ```mermaid
# graph TD
#     Q[A question arrives] --> S{What shape is the data?}
#     S -->|"Structured rows<br/>(tickets, vendors, contracts)"| SQL[SQL / API / MCP server<br/>exact filters · authorization at the query]
#     S -->|"Prose, curated,<br/>fits in a vector store"| RAG[Simple RAG<br/>chunk · embed · top-k]
#     S -->|"Prose, institutional,<br/>human-reviewed"| WIKI[Wiki / LLM Wiki<br/>durable notes · source-grounded]
#     S -->|"Very large, historical,<br/>heterogeneous"| BIG[Object storage + ontology<br/>live vs archival split]
#
#     SQL --> C[Agent context window]
#     RAG --> C
#     WIKI --> C
#     BIG -->|working set only| C
#
#     C --> G[Grounding check:<br/>every claim carries a source]
# ```


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
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# No `override=True`: real environment variables must win over `.env`, otherwise the
# exported gateway setup documented in the README (Model access section) is silently clobbered.
load_dotenv(find_dotenv())


# Put the workshop root on sys.path so `day1.src.*` and `utils.*` import cleanly. Works as a
# script (`__file__` is defined) and in a Jupyter kernel (it is not — fall back to cwd).
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
DAY1_DATA = WORKSHOP / "day1" / "data"
DAY2_DATA = WORKSHOP / "day2" / "data"

from day1.src.models import get_model

print("workspace:", os.getenv("LANGSMITH_WORKSPACE_ID"), "| tracing:", os.getenv("LANGSMITH_TRACING"))

# %% [markdown]
# ## 1. RAG is not one architecture
#
# 📖 [Retrieval](https://docs.langchain.com/oss/python/langchain/retrieval)
#
# "Use RAG" is not an architecture decision, it is a category. Four properties decide which
# member of the category you actually want — and they routinely point at *different* answers for
# different data in the same agent:
#
# | Property | Question to ask | What it rules out |
# |---|---|---|
# | **Scale** | Does the corpus fit in a vector store you can afford to rebuild? | Naive embed-everything, above ~10⁶ chunks |
# | **Freshness** | How stale may an answer be — seconds, hours, or a quarter? | Batch-indexed RAG for live operational state |
# | **Authorization** | Can every user see every document? | Shared indexes, when they cannot |
# | **Data shape** | Rows with exact filters, or prose needing semantic match? | Embeddings for `status = 'open' AND priority = 'P1'` |
#
# The workshop's own two use cases land in different places, which is the point:

# %%
tickets = json.loads((DAY1_DATA / "tickets.json").read_text())
kb_files = sorted((DAY1_DATA / "kb_tickets").glob("*.md"))
vendor_files = sorted((DAY1_DATA / "kb").glob("*.md"))

print(f"{'corpus':<34} {'items':>6}  {'shape':<12} {'right tool'}")
print("-" * 92)
rows = [
    ("day1/data/tickets.json", len(tickets), "structured", "SQL / API / MCP — exact filters, row-level authz"),
    ("day1/data/kb_tickets/*.md", len(kb_files), "prose", "simple RAG — semantic match over curated notes"),
    ("day1/data/kb/*.md (vendors)", len(vendor_files), "prose", "simple RAG + per-vendor scoping"),
    ("Acme production ITSM (illustrative)", 4_000_000, "structured", "SQL + live/archival split"),
]
for name, n, shape, tool in rows:
    print(f"{name:<34} {n:>6,}  {shape:<12} {tool}")

print(
    "\nThe ticket agent needs BOTH: exact structured lookup for ticket state (Day 1 §5 does this\n"
    "over MCP) and semantic retrieval for the how-to knowledge base. One agent, two retrieval\n"
    "architectures, chosen per data source — not one blessed pipeline for everything."
)

# %% [markdown]
# ## 2. Small / medium data — prefer the boring option
#
# For data that fits, the boring options beat a vector store on precision, latency, cost, and —
# most importantly — **authorization**. A SQL `WHERE` clause enforces row-level security exactly;
# a similarity search over a shared index enforces nothing.
#
# | Option | Use when | Authorization story | In this workshop |
# |---|---|---|---|
# | **API access** | The owning system already exposes one | Delegated to the API's own authz | Day 1 mock API tools |
# | **SQL query** | Structured rows, exact filters | `WHERE org_unit = :user_ou` — row-level, enforced by the DB | Pattern shown in `05_governance.py` §3 |
# | **MCP server** | The system is owned by another team | The server enforces; agent holds no credential | **`day1/src/ticket_mcp_server.py`** — real, used in Day 1 §5 |
# | **Simple vector store** | Prose, curated, modest size | Filter *before* retrieval, never after | Day 1 KB, Day 2 vendor KB |
# | **Curated knowledge base** | Answers must be human-approved | Editorial review is the control | Day 2 wiki (`02_wiki_memory.py`) |
#
# > **The filter-before-retrieval rule.** Retrieving top-k and *then* dropping unauthorized hits
# > silently degrades recall: a user entitled to 3 documents may get 0 back because 10 documents
# > they cannot see outranked them. Day 1 Lab 05 applies the entitlement filter to the candidate
# > set *before* scoring, which is why it returns stable results per user.

# %%
# Why "the boring option" is not a slogan — the same question, two architectures.
open_p1 = [t for t in tickets if t["status"] == "open" and t["priority"] == "P1"]
print("Q: 'Which P1 tickets are still open?'\n")
print(f"  structured lookup : exact, complete, auditable → {len(open_p1)} ticket(s): "
      f"{', '.join(t['ticket_id'] for t in open_p1) or 'none'}")
print("  vector search     : approximate, ranked by *similarity to the words* 'P1' and 'open' —")
print("                      returns resolved P1s and open P3s alongside, with no completeness")
print("                      guarantee. There is no k that fixes this; it is the wrong index.")

# %% [markdown]
# ## 3. Large / very large data — separate the working set from the archive
#
# Past a certain size the binding constraint stops being retrieval quality and becomes the
# **context window**. The failure mode is subtle: the agent still answers, but it answers from
# whichever fragments happened to land in context, and its confidence does not drop.
#
# Four techniques, in the order they usually pay off:
#
# 1. **Object storage as the system of record** (S3-style). Keep documents where they are cheap
#    and durable; index *metadata and pointers*, not the full text.
# 2. **Live context vs archival context.** Split the corpus by access pattern, not by topic. The
#    live working set — this quarter's tickets, currently-active vendors — is small, hot, and
#    frequently re-indexed. The archive is large, cold, and queried only on explicit demand.
# 3. **Ontology over the data.** A vocabulary the agent can filter on (vendor → contract →
#    delivery → incident) turns an unbounded semantic search into a bounded structured traversal.
# 4. **Never push the corpus into the window.** Retrieve a working set, cite it, and let the agent
#    request more. The measurement below is why.

# %%
# Measure it, don't assert it: what does "just put the knowledge base in the prompt" actually cost?
model = get_model()


def _tokens(text: str) -> int:
    """Token count via the model's own tokenizer, falling back to a ~4-chars/token estimate."""
    try:
        return model.get_num_tokens(text)
    except Exception:
        return len(text) // 4


ticket_kb_text = "\n\n".join(p.read_text() for p in kb_files)
vendor_kb_text = "\n\n".join(p.read_text() for p in vendor_files)
tickets_text = json.dumps(tickets, indent=2)

corpora = [
    (f"day1 ticket KB ({len(kb_files)} notes)", ticket_kb_text),
    (f"day1 vendor KB ({len(vendor_files)} profiles)", vendor_kb_text),
    (f"day1 tickets.json ({len(tickets)} rows)", tickets_text),
]
total = sum(_tokens(t) for _, t in corpora)

print(f"{'corpus':<38} {'tokens':>9}  {'% of a 128k window':>20}")
print("-" * 72)
for label, text in corpora:
    n = _tokens(text)
    print(f"{label:<38} {n:>9,}  {n / 128_000:>19.1%}")
print("-" * 72)
print(f"{'ALL of it, stuffed into the prompt':<38} {total:>9,}  {total / 128_000:>19.1%}")

# The honest framing: this corpus is TINY and still non-trivial. Scale it to Acme's real volumes.
REAL_TICKETS = 4_000_000
per_ticket = _tokens(tickets_text) / max(len(tickets), 1)
print(
    f"\nThis is a 24-ticket teaching fixture. At Acme scale ({REAL_TICKETS:,} tickets, at the\n"
    f"measured ~{per_ticket:,.0f} tokens/ticket) the same 'just include it' approach needs\n"
    f"~{REAL_TICKETS * per_ticket / 1e9:.1f}B tokens per call — about "
    f"{REAL_TICKETS * per_ticket / 128_000:,.0f}× a 128k window.\n"
    "That is the entire argument for retrieval, and for the live/archival split: you are not\n"
    "choosing retrieval because it is elegant, but because the alternative does not exist."
)

# %% [markdown]
# ### The live / archival split, concretely
#
# ```mermaid
# graph LR
#     subgraph Live["Live context — small, hot, re-indexed often"]
#         L1[Open + in-progress tickets]
#         L2[Active vendors and contracts]
#         L3[Current policies and SLAs]
#     end
#     subgraph Archive["Archival context — large, cold, queried on demand"]
#         A1[Resolved ticket history]
#         A2[Expired contracts]
#         A3[Superseded policy versions]
#     end
#     L1 & L2 & L3 --> AG[Agent working set<br/>always in reach]
#     A1 & A2 & A3 -.explicit lookup only.-> AG
# ```
#
# The workshop fixture already has this shape, which is what makes the Day 1 escalation demo work:

# %%
from collections import Counter

by_status = Counter(t["status"] for t in tickets)
live = [t for t in tickets if t["status"] in ("open", "in_progress", "escalated")]
print("ticket corpus by status:", dict(by_status))
print(f"\nlive working set : {len(live):>3} tickets ({len(live)/len(tickets):.0%}) — always in reach")
print(f"archival         : {by_status['resolved']:>3} tickets ({by_status['resolved']/len(tickets):.0%}) — "
      "retrieved only when a question is about history")
print(
    "\nThe ratio is the point. Even here the archive is ~83% of the corpus; in production it is\n"
    "well over 99%. Keeping it out of the default working set is not an optimisation — it is the\n"
    "difference between an agent that works and one that cannot be built."
)

# %% [markdown]
# ## 4. Wiki / OpenWiki / LLM Wiki — useful, and routinely oversold
#
# > 📖 Official docs: [OpenWiki overview](https://docs.langchain.com/oss/openwiki/overview) ·
# > [Quickstart](https://docs.langchain.com/oss/openwiki/quickstart) ·
# > [Code mode](https://docs.langchain.com/oss/openwiki/code-mode) ·
# > [Personal mode](https://docs.langchain.com/oss/openwiki/personal-mode) ·
# > [Automate updates](https://docs.langchain.com/oss/openwiki/automate-updates) ·
# > [Deep Agents + OpenWiki](https://docs.langchain.com/oss/python/deepagents/openwiki)
#
# **OpenWiki is a real LangChain tool, not a metaphor.** It is a CLI that writes and maintains a
# Markdown wiki so that *agents* — not primarily humans — have durable context and stop
# rediscovering the same architecture on every run:
#
# ```bash
# npm install -g openwiki
# openwiki --init          # code mode (default) → writes ./openwiki/
# openwiki --update        # refresh; only rewrites when content actually changed
# openwiki personal --init # personal mode → ~/.openwiki/wiki
# ```
#
# | Aspect | What OpenWiki does |
# |---|---|
# | **Code mode** (default) | Generates `openwiki/` for a repository, then adds pointers in the repo-root `AGENTS.md` / `CLAUDE.md` so coding agents discover it |
# | **Personal mode** | Builds `~/.openwiki/wiki` from git repos, Gmail, Notion, web search, Hacker News, Slack, X |
# | **Open Knowledge Format (OKF v0.1)** | Markdown bundles with front matter, indexes, and linked concepts — structured for agent consumption |
# | **Automatic updates** | GitHub Actions / GitLab CI / Bitbucket Pipelines run `--update` and open a PR when the wiki changes |
# | **Observability** | Documentation runs trace to LangSmith |
#
# Two ideas there transfer directly to Acme and are worth stealing even if you never run the CLI:
#
# 1. **The wiki is discovered through `AGENTS.md`.** That is exactly the mechanism Day 2 Lab 02
#    uses — `MemoryMiddleware` loads `day2/data/agent/AGENTS.md`, which points at the wiki. Same
#    pattern, different writer.
# 2. **Updates are a reviewed pull request, not a silent write.** A human approves the diff. That is
#    the accountability answer to the question at the end of this section.
#
# Day 2 Lab 02 builds the *hand-rolled* version of this: the agent writes durable, source-grounded
# notes into `day2/data/wiki/` and reads them back on later runs. We build it by hand because the
# lesson is the **pattern** — durable notes, citations, human review — and OpenWiki is the
# productised form of that pattern for repositories.
#
# **Where a wiki earns its place**
# - Durable notes that outlive a single run — risk signals, open questions, decisions and why
# - Source-grounded prose where each claim carries a citation
# - Content a human can review, correct, and be accountable for
# - Scoped, modest volume — hundreds of pages, not millions
#
# **Where it does not**
# - ❌ As an enterprise-scale data store. A wiki is not a database with a nicer interface.
# - ❌ As the system of record for anything transactional. Tickets, contracts and entitlements
#   live in the systems that own them; the wiki holds *notes about* them.
# - ❌ Unreviewed agent writes. An agent that reads its own unverified claims back as fact will
#   compound its errors with total confidence — the notes look like evidence.
#
# > The controlling question: **who is accountable for a wrong page?** If the answer is "nobody,
# > the agent wrote it", the wiki is a cache of the agent's opinions, not a knowledge base.
# > OpenWiki's answer is the PR: a person approves the diff before it lands. Whatever you build,
# > have an equivalent — for Acme that likely means the wiki writes to a branch, not to main.

# %%
wiki_dir = DAY2_DATA / "wiki"
if wiki_dir.is_dir():
    pages = sorted(p for p in wiki_dir.rglob("*.md"))
    print(f"day2 wiki: {len(pages)} page(s)")
    for p in pages[:6]:
        cited = "[source:" in p.read_text()
        print(f"  {'✅ cited  ' if cited else '⚠️  uncited'} {p.relative_to(DAY2_DATA)}")
    if not pages:
        print("  (empty — run day2/src/02_wiki_memory.py first; the agent writes pages there)")
    print(
        "\n⚠️  markers are the review queue. An uncited page is exactly what a human should read\n"
        "    before anyone treats it as institutional knowledge."
    )
else:
    print("⏭ day2/data/wiki not present — run day2/src/02_wiki_memory.py to populate it")

# %% [markdown]
# ## 5. Context management — four kinds of knowledge, four homes
#
# The most consequential design decision on Day 2 was not the retriever. It was deciding *which
# kind* of knowledge each piece of information was, because that determines where it lives, how
# it is updated, and who is accountable for it.
#
# | Kind | Lives in | Changes | Workshop example |
# |---|---|---|---|
# | **Instructions** | System prompt / `AGENTS.md` | On deliberate edit, version-controlled | `day2/data/agent/AGENTS.md` — how to do due diligence |
# | **Skills** | `SKILL.md` files, loaded on demand | On deliberate edit | `agent/skills/evidence-review/SKILL.md` |
# | **Retrieved knowledge** | Vector store / DB / API | Continuously, at the source | Ticket KB, vendor KB, `tickets.json` |
# | **Durable memory** | Wiki / store backend | The agent writes it, a human reviews it | `day2/data/wiki/*.md` |
#
# The common mistake is collapsing two of these. Putting *instructions* in the vector store makes
# the agent's behaviour depend on retrieval luck. Putting *retrieved knowledge* in the system
# prompt makes it stale the moment the source changes, with no way to tell.
#
# ### Freshness and refresh policy
#
# Every retrieved source needs a stated staleness budget, and the agent should say when it is
# outside it:
#
# | Source | Acceptable staleness | Refresh mechanism |
# |---|---|---|
# | Ticket state (open/closed) | Seconds | Live query — never cached (Day 1 §5, over MCP) |
# | SLA and routing policy | Hours | Cached, invalidated on policy change |
# | Vendor capability notes | Weeks | Scheduled re-index |
# | Sanctions / watchlist screening | **Per query, always** | Live — a cached "CLEAR" is a compliance failure |
#
# ### Citations and source validation
#
# Every tool in this workshop returns a `[source: ...]` tag, and the `groundedness` judge in
# Day 2 Lab 03 grades whether claims carry one. That is not decoration — it is the only reason a
# reviewer can check the agent's work in less time than doing the work themselves.

# %%
# The freshness rule that actually bites: a cached compliance verdict.
print("Why 'always live' for screening — the failure is silent:\n")
print("  t=0    screen_vendor('Kelbrant Trading')  → CLEAR      (cached, TTL 24h)")
print("  t=2h   vendor added to a sanctions list")
print("  t=6h   agent recommends the vendor, citing the CLEAR verdict — with a source tag,")
print("         a confident tone, and a citation that was true when it was written.")
print("\n  Nothing in the trace looks wrong. The grounding check passes. The citation is real.")
print("  Only the staleness budget would have caught it — which is why it has to be declared")
print("  per source, and enforced in the tool, not left to the model's judgement.")

# %% [markdown]
# ## 6. Putting it together — one small model call
#
# A decision procedure is only useful if it survives contact with a question. Ask the model to
# route three Acme-shaped questions to an architecture, then check the answers against §1–3.

# %%
QUESTIONS = [
    "How many P1 tickets breached their SLA last quarter?",
    "What has Acme learned about Kelbrant Trading's delivery reliability?",
    "Which of our 4 million archived tickets mention a specific firmware defect?",
]
prompt = (
    "For each question, name the retrieval architecture you would use and give a one-line reason.\n"
    "Choose from: SQL/API/MCP (structured, exact) · simple RAG (prose, curated) · "
    "wiki/durable notes (institutional, human-reviewed) · object storage + ontology with a "
    "live/archival split (very large, historical).\n"
    "Answer as: <n>. <architecture> — <reason>. Be terse.\n\n"
    + "\n".join(f"{i}. {q}" for i, q in enumerate(QUESTIONS, 1))
)
print(model.invoke(prompt).content)

print(
    "\nExpected: 1 → SQL/API/MCP (exact aggregate over structured rows, not a similarity match)\n"
    "          2 → wiki / durable notes (institutional judgement, human-reviewed, source-grounded)\n"
    "          3 → object storage + ontology, archival tier (too large for the working set)\n"
    "\nIf the model routes all three to 'RAG', that is the exact failure this module exists to\n"
    "prevent — and a good prompt to put in an eval dataset."
)

# %% [markdown]
# ## 7. Recap
#
# | Decision | Rule |
# |---|---|
# | Choosing an architecture | Decide per **data source**, not per agent. One agent, several architectures, is normal. |
# | Structured data | SQL / API / MCP. Exact filters and row-level authorization beat similarity search. |
# | Authorization | Filter **before** retrieval. Post-filtering silently destroys recall. |
# | Very large corpora | Split live from archival by access pattern. Index pointers, not full text. |
# | Wikis | Excellent for reviewed institutional notes; not a database, not a system of record. |
# | Context management | Instructions, skills, retrieved knowledge and durable memory are four different things with four different homes. |
# | Freshness | Every source gets a declared staleness budget, enforced in the tool. Compliance screening is always live. |
# | Grounding | Every claim carries a source tag, or a reviewer cannot check the work. |
#
# **Next:** `05_governance.py` §4–6 — deployment paths, Fleet, and Engine (agenda 15:15–16:00).
