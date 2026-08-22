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
# # 01 · Deep Agent — Vendor Due Diligence
#
# **Workshop:** LangChain ADLC Workshop · **Day 2** · **ADLC stage:** Build + Verify
#
# > **Loop Engineering focus: Verification loop** — a `create_deep_agent` supervisor delegates to
# > specialist sub-agents, loads operating instructions via `MemoryMiddleware`, exposes an evidence
# > review workflow via `SkillsMiddleware`, and verifies its own work at runtime with
# > `RubricMiddleware` before it is allowed to finish.
#
# ```mermaid
# graph TD
#     U[Procurement officer] -->|due diligence request| S[Supervisor deep agent]
#     S -->|AGENTS.md always-on instructions| S
#     S -->|SKILL.md evidence review workflow| S
#     S -->|task()| EC[evidence_collector]
#     S -->|task()| RA[risk_assessor]
#     S -->|task()| CS[compliance_screener]
#     EC --> KB[search_vendor_kb · parse_vendor_pdf · tavily_search]
#     RA --> DB[get_vendor · filter_vendors · get_risk_criteria]
#     CS --> SN[screen_vendor]
#     S -->|draft report| R{RubricMiddleware}
#     R -->|needs_revision + feedback| S
#     R -->|satisfied| OUT[Due diligence report]
# ```
#
# | Sub-agent | Tools | Role |
# |---|---|---|
# | `evidence_collector` | `search_vendor_kb`, `parse_vendor_pdf`, `tavily_search` | Gather cited capability/certification evidence from KB, PDFs, and the web |
# | `risk_assessor` | `get_vendor`, `filter_vendors`, `get_risk_criteria` | Classify risk signals against the due diligence criteria |
# | `compliance_screener` | `screen_vendor` | Sanctions / watchlist screening |
#
# | Middleware | Purpose |
# |---|---|
# | `MemoryMiddleware` | Loads `day2/data/agent/AGENTS.md` (DD operating instructions) into every system prompt |
# | `SkillsMiddleware` | Exposes four skills (`day2/data/agent/skills/`) via progressive disclosure: `evidence-review`, `source-validation`, `risk-classification`, `report-drafting` |
# | `RubricMiddleware` | A grader sub-agent checks the report against a rubric and loops the agent until it passes |
#
# By the end you can:
# - build a multi-sub-agent deep agent with `create_deep_agent`
# - load `AGENTS.md` operating instructions with `MemoryMiddleware`
# - expose four `SKILL.md` workflows with `SkillsMiddleware` (evidence review, source validation,
#   risk classification, report drafting) — the four the agenda calls for
# - verify agent output at runtime with `RubricMiddleware` — and compare runs **with and without** it
# - demonstrate **source validation**: every claim traced to a verified source
#
# > 🧭 **Builds on Day 1 Labs 02–05; runs standalone.** All vendor data is **synthetic/fictional** —
# > created for this workshop, no real Acme data.
# >
# > Run `day2/notebooks/00_setup.ipynb` first (or `uv run python day2/verify_setup.py`). It checks
# > today's prerequisites — the agent skills, the eval sets, the vendor corpus — in about ten
# > seconds. §0 below is a different thing: the `sys.path` and `.env` bootstrap every lab needs to
# > run standalone, not an environment check.


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
#
# Same setup cell pattern as Day 1: load `.env` first (so the model layer sees the right keys), then
# put `langchain_adlc_workshop/` on `sys.path` so `day1.src.models` and `utils.trace` import cleanly.

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
load_dotenv(find_dotenv())  # FIRST — before any model/embeddings client
# The model layer (day1/src/models.py) handles gateway vs direct API key routing.
# See the README (Model access section) for gateway setup instructions.

# No gateway key juggling here: `day1/src/models.py` routes both the chat model
# (`get_model`) and the embeddings client (`get_embeddings`) by inspecting the gateway env vars, and
# passes the gateway credential explicitly as `api_key=`. See the README (Model access section).

DAY1_DATA = WORKSHOP / "day1" / "data"       # vendors.json, kb/, pdfs/, sanctions_list.json
DAY2_DATA = WORKSHOP / "day2" / "data"       # agent/AGENTS.md, agent/skills/
AGENTS_MD = DAY2_DATA / "agent" / "AGENTS.md"              # exists — DD operating instructions
SKILLS_DIR = DAY2_DATA / "agent" / "skills"                # four SKILL.md workflows live here

print("workspace:", os.getenv("LANGSMITH_WORKSPACE_ID"), "| tracing:", os.getenv("LANGSMITH_TRACING"))
print("AGENTS.md:", AGENTS_MD.relative_to(WORKSHOP), "exists:", AGENTS_MD.exists())
print("skills dir:", SKILLS_DIR.relative_to(WORKSHOP), "exists:", SKILLS_DIR.exists(),
      "| skills:", sorted(p.name for p in SKILLS_DIR.iterdir() if p.is_dir()))

# %% [markdown]
# ## 1. Tools — evidence, risk, and compliance
#
# 📖 [Tools](https://docs.langchain.com/oss/python/langchain/tools)
#
# The Day 1 vendor tools are re-declared here in the Day 2 notebook style so the lab runs
# standalone. The sub-agents split them by concern:
#
# | Data file | Tool(s) that read it |
# |---|---|
# | `day1/data/kb/*.md` (13 KB articles) | `search_vendor_kb` |
# | `day1/data/pdfs/*.pdf` (13 capability statements) | `parse_vendor_pdf` |
# | `day1/data/vendors.json` (13 vendors) | `get_vendor`, `filter_vendors`, `get_risk_criteria` |
# | `day1/data/sanctions_list.json` | `screen_vendor` |
# | live web *(optional)* | `tavily_search` |
#
# First, the RAG index over the 13 vendor KB articles — same pattern as Lab 02.

# %%
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from day1.src.models import get_embeddings  # routes to gateway or direct provider — see the README (Model access section)

KB_DIR = DAY1_DATA / "kb"

docs = [Document(page_content=p.read_text(), metadata={"source": p.name}) for p in sorted(KB_DIR.glob("*.md"))]
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
vendor_kb_index = InMemoryVectorStore.from_documents(chunks, get_embeddings())
print(f"indexed {len(chunks)} chunks from {len(docs)} vendor KB articles")

# %%
from langchain_core.tools import tool


@tool
def search_vendor_kb(query: str) -> str:
    """Search vendor profile pages by capability, certification, or keyword.
    Returns cited snippets from the vendor knowledge base (13 synthetic vendor profiles)."""
    hits = vendor_kb_index.similarity_search(query, k=4)
    if not hits:
        return "No relevant vendor profiles found."
    return "\n\n---\n\n".join(f"[source: {h.metadata['source']}]\n{h.page_content}" for h in hits)


# %%
print(search_vendor_kb.invoke({"query": "QS-100 avionics certification"})[:400])

# %% [markdown]
# **Vendor database tools** — `vendors.json` powers the structured lookup and the deterministic
# constraint filter. These are the risk assessor's inputs.

# %%
VENDORS: dict = json.loads((DAY1_DATA / "vendors.json").read_text())
VENDOR_LIST = [rec for vid, rec in VENDORS.items() if not vid.startswith("_")]
print(f"loaded {len(VENDOR_LIST)} vendors from vendors.json")


def _format_vendor(rec: dict) -> str:
    certs = ", ".join(rec.get("certifications", []))
    flags = ", ".join(rec.get("compliance_flags", []))
    contracts = "; ".join(
        f"{c['buyer']} ({c['year']}): EUR {c['value_eur']:,} — {c['scope']}"
        for c in rec.get("contract_history", [])
    )
    return (
        f"{rec['name']} ({rec['vendor_id']}) — {rec['category']} · {rec['country']}\n"
        f"  size: {rec['size']} · risk: {rec['risk_level']}\n"
        f"  certifications: {certs}\n"
        f"  compliance: {flags}\n"
        f"  contracts: {contracts}\n"
        f"  highlights: {rec.get('highlights', 'N/A')}\n"
        f"  [source: vendor database]"
    )


@tool
def get_vendor(vendor: str) -> str:
    """Get a single vendor's full structured record by vendor ID (e.g. VND-001) or name."""
    q = vendor.strip().lower()
    for rec in VENDOR_LIST:
        if q == rec["vendor_id"].lower() or q in rec["name"].lower():
            return _format_vendor(rec)
    return f"No vendor found for {vendor!r}."


@tool
def filter_vendors(
    category: str | None = None,
    country: str | None = None,
    certification: str | None = None,
    risk_level: str | None = None,
    size: str | None = None,
) -> str:
    """Filter the vendor database by category, country, certification, risk level, or company size.
    Returns a shortlist ranked by risk level (low first) then by Acme contract history."""
    rows = VENDOR_LIST
    if category:
        rows = [r for r in rows if category.lower() in r["category"].lower()]
    if country:
        rows = [r for r in rows if country.lower() in r["country"].lower()]
    if certification:
        rows = [r for r in rows if any(certification.lower() in c.lower() for c in r.get("certifications", []))]
    if risk_level:
        rows = [r for r in rows if r["risk_level"].lower() == risk_level.lower()]
    if size:
        rows = [r for r in rows if r["size"].lower() == size.lower()]

    if not rows:
        return "No vendors match the given filters."

    _risk_order = {"low": 0, "medium": 1, "high": 2}
    rows.sort(
        key=lambda r: (
            _risk_order.get(r["risk_level"], 3),
            -sum(1 for c in r.get("contract_history", []) if c["buyer"] == "Acme"),
            -sum(c["value_eur"] for c in r.get("contract_history", [])),
        )
    )

    lines = [f"Found {len(rows)} vendor(s):"]
    for i, r in enumerate(rows, 1):
        acme_contracts = sum(1 for c in r.get("contract_history", []) if c["buyer"] == "Acme")
        total_value = sum(c["value_eur"] for c in r.get("contract_history", []))
        lines.append(
            f"  {i}. {r['name']} ({r['vendor_id']}) — {r['category']} · {r['country']} · "
            f"risk: {r['risk_level']} · Acme contracts: {acme_contracts} · total value: EUR {total_value:,}"
        )
    return "\n".join(lines)


# %%
print(get_vendor.invoke({"vendor": "VND-001"}))

# %%
print(filter_vendors.invoke({"category": "Aerospace", "risk_level": "low"}))

# %% [markdown]
# **Risk criteria tool** — the assessor's rubric, derived deterministically from `vendors.json`
# plus the workshop's escalation rules (mirrors `day2/data/agent/AGENTS.md` rule 3).

# %%
@tool
def get_risk_criteria() -> str:
    """Retrieve the vendor risk assessment criteria: how risk levels are assigned, which signals
    to surface, and when to escalate to human review. Use this before classifying vendor risk."""
    risk_counts = {}
    for rec in VENDOR_LIST:
        risk_counts[rec["risk_level"]] = risk_counts.get(rec["risk_level"], 0) + 1
    dist = ", ".join(f"{k}: {v}" for k, v in sorted(risk_counts.items()))
    return (
        "Acme vendor risk assessment criteria (synthetic — for workshop use):\n"
        "\n"
        "Risk classification:\n"
        "  - LOW: Regional/allied-based, current quality certifications (ISO 9001/AS9100D/QS-100), clean\n"
        "    compliance record, established Acme/Tier-1 aerospace integrator contract history.\n"
        "  - MEDIUM: missing or expired certifications, limited contract history, incomplete\n"
        "    documentation, or non-EU supply chain dependencies.\n"
        "  - HIGH: sanctions/watchlist match, export-control violations, no verifiable certifications,\n"
        "    or domiciled in an embargoed jurisdiction.\n"
        "\n"
        "Risk signals to surface (with severity):\n"
        "  - missing_certifications (medium) — required certs absent from the vendor record\n"
        "  - sanctions_proximity (high) — name or country overlap with a listed entity\n"
        "  - thin_contract_history (low/medium) — few or low-value past contracts\n"
        "  - single_source_dependency (medium) — sole supplier for a critical capability\n"
        "  - unverified_claims (high) — evidence the agent could not verify against a source\n"
        "\n"
        f"Current vendor database risk distribution ({len(VENDOR_LIST)} vendors): {dist}.\n"
        "\n"
        "Escalate to human review when: confidence is low, any vendor is medium or high risk,\n"
        "source validation is incomplete, or the need is high priority / high value.\n"
        "[source: vendors.json risk fields + AGENTS.md operating rules]"
    )


# %%
print(get_risk_criteria.invoke({}))

# %% [markdown]
# **Compliance screening** — `screen_vendor` checks the local synthetic sanctions list
# (`sanctions_list.json`) and optionally the OpenSanctions API when `OPENSANCTIONS_API_KEY` is set.
# Same implementation as Day 1's vendor graph.

# %%
SANCTIONS: dict = json.loads((DAY1_DATA / "sanctions_list.json").read_text())


def _screen_local(name: str, country: str) -> dict | None:
    """Check the synthetic local sanctions list for a match by name (fuzzy) and country."""
    name_lower = name.lower()
    for entity in SANCTIONS.get("sanctioned_entities", []):
        ent_name = entity["name"].lower()
        if name_lower in ent_name or ent_name in name_lower:
            return {"matched": True, "list_type": "sanctioned", **entity}
    for entity in SANCTIONS.get("watchlist", []):
        ent_name = entity["name"].lower()
        if name_lower in ent_name or ent_name in name_lower:
            return {"matched": True, "list_type": "watchlist", **entity}
    return None


def _screen_opensanctions(name: str, country: str) -> dict | None:
    """Call the OpenSanctions /match API. Returns a match dict or None on failure/no-key.

    Guarded: only runs when OPENSANCTIONS_API_KEY is set. Any exception returns None
    (the agent falls back to local screening)."""
    api_key = os.getenv("OPENSANCTIONS_API_KEY")
    if not api_key:
        return None
    try:
        import requests

        resp = requests.post(
            "https://api.opensanctions.org/match/default",
            headers={"Authorization": f"ApiKey {api_key}"},
            json={"queries": {"q": {"schema": "Company", "properties": {"name": [name], "country": [country]}}}},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("responses", {}).get("q", {}).get("results", [])
        if not results:
            return None
        best = results[0]
        return {
            "matched": best.get("match", False),
            "list_type": "opensanctions",
            "entity_id": best.get("id"),
            "name": best.get("caption"),
            "score": best.get("score"),
            "topics": best.get("properties", {}).get("topics", []),
            "programs": best.get("properties", {}).get("programId", []),
            "datasets": best.get("datasets", []),
            "source": "OpenSanctions API",
        }
    except Exception:
        return None


@tool
def screen_vendor(vendor_name: str, country: str) -> str:
    """Screen a vendor against sanctions lists and watchlists. Checks a local synthetic list first,
    then optionally calls the OpenSanctions API when OPENSANCTIONS_API_KEY is set. Returns a screening
    result with match status, sanctions programs, and risk topics."""
    # 1. Local synthetic list (always runs — offline-safe, deterministic)
    local = _screen_local(vendor_name, country)
    if local:
        status = "SANCTIONED" if local["list_type"] == "sanctioned" else "WATCHLIST"
        return (
            f"⚠️  {status} MATCH — {local['name']} ({local['entity_id']})\n"
            f"  country: {local['country']} · programs: {', '.join(local.get('programs', []))}\n"
            f"  topics: {', '.join(local.get('topics', []))}\n"
            f"  reason: {local.get('reason', 'N/A')}\n"
            f"  listed: {local.get('listed_date', 'N/A')} · source: {local.get('source', 'N/A')}\n"
            f"  [screening: local synthetic list]"
        )

    # 2. OpenSanctions API (optional — requires OPENSANCTIONS_API_KEY)
    os_result = _screen_opensanctions(vendor_name, country)
    if os_result and os_result["matched"]:
        return (
            f"⚠️  SANCTIONS MATCH — {os_result['name']} ({os_result['entity_id']})\n"
            f"  score: {os_result['score']:.2f} · topics: {', '.join(os_result.get('topics', []))}\n"
            f"  programs: {', '.join(os_result.get('programs', []))}\n"
            f"  datasets: {', '.join(os_result.get('datasets', []))}\n"
            f"  [screening: OpenSanctions API]"
        )

    # 3. Clear
    return (
        f"✅ CLEAR — No sanctions or watchlist matches for {vendor_name} ({country}).\n"
        f"  [screening: local synthetic list" + (" + OpenSanctions API" if os_result is not None else "") + "]"
    )


# %%
# Quick check: Kelbrant Trading Consortium is on the synthetic sanctions list; Quelmore is clear.
print(screen_vendor.invoke({"vendor_name": "Kelbrant Trading Consortium", "country": "Non-EU jurisdiction D"}))
print()
print(screen_vendor.invoke({"vendor_name": "Quelmore Systems Ltd", "country": "Netherlands"}))

# %% [markdown]
# **PDF capability statements** — `parse_vendor_pdf` extracts text from the 13 synthetic PDFs so the
# evidence collector can corroborate KB claims against a second source (key for source validation).

# %%
PDF_DIR = DAY1_DATA / "pdfs"


@tool
def parse_vendor_pdf(vendor_name: str) -> str:
    """Parse a synthetic PDF vendor capability statement. Returns the extracted text content.
    Use this to corroborate KB article claims against the vendor's own capability statement —
    a second, independent source for source validation."""
    safe_name = vendor_name.lower().replace(" ", "_").replace("&", "and")
    pdf_path = PDF_DIR / f"{safe_name}_capability_statement.pdf"

    if not pdf_path.exists():
        matches = list(PDF_DIR.glob(f"*{safe_name}*.pdf"))
        if matches:
            pdf_path = matches[0]
        else:
            available = ", ".join(p.stem.replace("_capability_statement", "") for p in PDF_DIR.glob("*.pdf"))
            return f"No PDF found for {vendor_name!r}. Available: {available}"

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() for page in reader.pages)
        return f"[source: {pdf_path.name}]\n{text[:2000]}"  # truncate for readability
    except Exception as e:
        return f"Error parsing PDF {pdf_path.name}: {e}"


# %%
print(parse_vendor_pdf.invoke({"vendor_name": "quelmore_systems"})[:300])

# %% [markdown]
# **Optional live web** — Tavily, same pattern as Day 1: added only when `TAVILY_API_KEY` is set.
# The evidence collector uses it to validate claims against sources outside the bundled fixtures.

# %%
tavily_tool = None
if os.getenv("TAVILY_API_KEY"):
    from langchain_tavily import TavilySearch

    tavily_tool = TavilySearch(max_results=3)
print(f"Tavily: {'✅ enabled (evidence_collector gets live web search)' if tavily_tool else '❌ not set — evidence_collector runs on bundled data only'}")

# %% [markdown]
# ## 2. Sub-agents — evidence, risk, compliance
#
# 📖 [Multi-agent](https://docs.langchain.com/oss/python/langchain/multi-agent)
#
# `create_deep_agent` gives the supervisor a built-in `task()` tool that delegates work to
# **sub-agents**. Each sub-agent is a `SubAgent` spec with a name, an action-oriented description,
# its own system prompt, and its own tool list. The supervisor never touches the raw tools — it
# decomposes the request and routes it:
#
# ```mermaid
# graph LR
#     S[supervisor] -->|task: gather evidence| EC[evidence_collector]
#     S -->|task: classify risk| RA[risk_assessor]
#     S -->|task: screen compliance| CS[compliance_screener]
#     EC -->|cited snippets + PDF text| S
#     RA -->|risk signals + level| S
#     CS -->|sanctions verdict| S
# ```
#
# > Each sub-agent gets **only** the tools for its concern — the assessor cannot call the screening
# > tool, the screener cannot browse the KB. Narrow tool surfaces make traces readable and keep
# > responsibilities separable for evals.

# %%
from deepagents.middleware.subagents import SubAgent

EVIDENCE_TOOLS = [search_vendor_kb, parse_vendor_pdf] + ([tavily_tool] if tavily_tool else [])
RISK_TOOLS = [get_vendor, filter_vendors, get_risk_criteria]
COMPLIANCE_TOOLS = [screen_vendor]

evidence_collector: SubAgent = {
    "name": "evidence_collector",
    "description": (
        "Gathers cited evidence about a vendor's capabilities, certifications, and track record "
        "from the vendor knowledge base, PDF capability statements, and (when configured) live "
        "web search. Use first whenever an assessment needs evidence."
    ),
    "system_prompt": (
        "You are the evidence collector for an Acme vendor due diligence team.\n"
        "- Search the vendor KB for capability and certification evidence (search_vendor_kb).\n"
        "- Corroborate KB claims against the vendor's PDF capability statement (parse_vendor_pdf) —\n"
        "  a claim confirmed by two independent sources is 'verified'; one source is 'partially verified'.\n"
        "- When available, use tavily_search for external corroboration.\n"
        "- Every claim you return MUST cite its source ([source: ...] tag from the tool output).\n"
        "- Never invent evidence. If you cannot find support for a claim, say 'no evidence found'.\n"
        "Return a compact evidence list: one bullet per claim, with source and verification status."
    ),
    "tools": EVIDENCE_TOOLS,
}

risk_assessor: SubAgent = {
    "name": "risk_assessor",
    "description": (
        "Classifies a vendor's risk level (low/medium/high) and surfaces specific risk signals "
        "with severity, using the vendor database and the published risk criteria. Use after "
        "evidence is gathered, or to shortlist vendors by risk constraints."
    ),
    "system_prompt": (
        "You are the risk assessor for an Acme vendor due diligence team.\n"
        "- Always retrieve the criteria first (get_risk_criteria) before classifying.\n"
        "- Use get_vendor for the full record; filter_vendors to compare against peers.\n"
        "- Classify risk low/medium/high and list each risk signal with a severity.\n"
        "- Ground every signal in the vendor record — cite [source: vendor database].\n"
        "- Flag unverified or missing information explicitly; do not guess.\n"
        "Return: risk level, risk signals (name + severity + evidence), and data gaps."
    ),
    "tools": RISK_TOOLS,
}

compliance_screener: SubAgent = {
    "name": "compliance_screener",
    "description": (
        "Screens a vendor against sanctions lists and watchlists (screen_vendor). Always run "
        "before recommending any vendor; a SANCTIONED or WATCHLIST match is an automatic "
        "high-risk escalation."
    ),
    "system_prompt": (
        "You are the compliance screener for an Acme vendor due diligence team.\n"
        "- Screen every vendor under assessment (screen_vendor) — no exceptions.\n"
        "- Report the verdict verbatim: CLEAR, WATCHLIST, or SANCTIONED, with programs/topics/reason.\n"
        "- A SANCTIONED match means the vendor must be excluded and escalated; say so plainly.\n"
        "- Never soften or reinterpret a screening result.\n"
        "Return: screening verdict, matched entity (if any), and the screening source used."
    ),
    "tools": COMPLIANCE_TOOLS,
}

SUBAGENTS = [evidence_collector, risk_assessor, compliance_screener]
print("sub-agents:", [s["name"] for s in SUBAGENTS])

# %% [markdown]
# ## 3. Memory — AGENTS.md operating instructions
#
# 📖 [Deep Agents memory](https://docs.langchain.com/oss/python/deepagents/memory)
#
# `MemoryMiddleware` implements the [agents.md](https://agents.md/) pattern: file(s) on disk are
# loaded and injected into the system prompt on every run, so the operating instructions are
# **always on** — not something the agent has to remember to read.
#
# Here it loads `day2/data/agent/AGENTS.md` — the due diligence operating rules (evidence-based
# claims, source validation, risk classification, escalation). Because it is memory, the agent can
# also *update* it via `edit_file` as the team learns — but today we use it read-only.
#
# > **Backend:** `FilesystemBackend(root_dir=DAY2_DATA, virtual_mode=True)` — the root is
# > `day2/data/`, not the whole repo. Two reasons, both learned the hard way:
# >
# > 1. **Blast radius.** `virtual_mode=True` stops the agent escaping the root, but the root is
# >    still whatever you point it at. Rooted at the repo, the agent's `ls`/`grep`/`read` tools can
# >    reach every source file — and `.env` sits there too. Scope the root to the data the agent
# >    needs, not to everything it *could* be given.
# > 2. **PDFs are not text.** `deepagents` maps `.pdf` to a multimodal `"file"` content block
# >    (`backends/utils.py`). OpenAI's chat-completions API accepts only `text`, `refusal`,
# >    `image_url` and `input_audio`, so the moment a PDF lands in the message history the *next*
# >    model call fails with `Invalid value: 'file'`. With the repo as root the agent could reach
# >    `day1/data/pdfs/`, and whether it did was down to which tool it picked — the failure was
# >    intermittent, which is the worst kind. Use the `parse_vendor_pdf` tool for PDFs: it extracts
# >    text, so the model gets something it can actually read.

# %%
import warnings

with warnings.catch_warnings():
    # virtual_mode default is changing in deepagents; we set it explicitly and silence the notice.
    warnings.simplefilter("ignore", DeprecationWarning)
    from deepagents.backends.filesystem import FilesystemBackend

    backend = FilesystemBackend(root_dir=DAY2_DATA, virtual_mode=True)

from deepagents.middleware import MemoryMiddleware

memory_middleware = MemoryMiddleware(
    backend=backend,
    sources=[str(AGENTS_MD.relative_to(DAY2_DATA))],  # agent/AGENTS.md, relative to the root
)

# Peek: confirm the memory source resolves and loads through the backend.
resp = backend.download_files([str(AGENTS_MD.relative_to(DAY2_DATA))])
print(f"memory source loaded: {resp[0].path} ({len(resp[0].content)} bytes)")
print(resp[0].content.decode()[:200], "...")

# %% [markdown]
# ## 4. Skills — evidence review workflow
#
# 📖 [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) · [reference](https://github.com/langchain-ai/deepagents/tree/main/libs/deepagents)
#
# `SkillsMiddleware` implements the **agent skills** pattern with *progressive disclosure*: the
# system prompt lists each skill's name + description, and the agent reads the full `SKILL.md`
# only when a task matches. Skills live one directory each under a source root:
#
# ```text
# day2/data/agent/skills/
# └── evidence-review/
#     └── SKILL.md        # ← evidence-review skill
# ```
#
# > ✅ **The evidence-review skill is ready.** The middleware discovers `SKILL.md` from the
# > skills directory and makes it available to the agent via progressive disclosure.

# %%
from deepagents.middleware import SkillsMiddleware

skills_middleware = SkillsMiddleware(
    backend=backend,
    sources=[(str(SKILLS_DIR.relative_to(DAY2_DATA)), "Workshop")],  # agent/skills
)

# The middleware discovers skills by scanning the source dir for */SKILL.md. Print what discovery
# finds so the progressive-disclosure list is explicit.
skills_ls = backend.ls(str(SKILLS_DIR.relative_to(DAY2_DATA)))
entries = getattr(skills_ls, "entries", skills_ls) or []
print("skills discovered:", [e["path"] for e in entries if e.get("is_dir")] or "none found")

# %% [markdown]
# ## 5. Rubric — runtime verification
#
# 📖 [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
#
# `RubricMiddleware` is the **verification loop**. When the agent would otherwise finish (a response
# with no more tool calls), the middleware hands the transcript to a separate **grader sub-agent**
# with the rubric. If the grader says `needs_revision`, its feedback is injected as a human message
# and the agent loop resumes — the agent revises its own report until the rubric passes or
# `max_iterations` is hit.
#
# The rubric is passed **per invocation** as `state["rubric"]` — so the same agent runs with or
# without verification depending on the input (Section 7 compares the two). With no rubric in the
# input, the middleware is a no-op.
#
# ```mermaid
# graph LR
#     A[agent draft] --> G{grader}
#     G -->|satisfied| D[done]
#     G -->|needs_revision + gaps| A
#     G -->|max_iterations| D
# ```

# %%
from deepagents.middleware import RubricMiddleware

from day1.src.models import get_embeddings, get_judge_model, get_model

model = get_model()              # the agent under verification
grader_model = get_judge_model()  # the verifier — a stronger tier, see day1/src/models.py

# The grader runs on the JUDGE tier, not the agent tier: RubricMiddleware is runtime
# LLM-as-judge, and a grader that shares the agent's blind spots will pass the agent's mistakes.
# It judges the transcript against the rubric, so it needs no tools. max_iterations caps the revise loop (hard cap 20; 5 gives the agent enough
# budget to fix a draft, then stop). Track every grading pass in a local list — the
# `_rubric_*` keys are PRIVATE state (not part of the invoke output), so the supported way to
# observe grading from a bare .invoke is the on_evaluation callback.
RUBRIC_LOG: list[dict] = []


def _log_evaluation(ev) -> None:
    """on_evaluation callback — record each grading pass and print a one-line verdict."""
    entry = {
        "iteration": ev.get("iteration"),
        "result": ev.get("result"),
        "explanation": (ev.get("explanation") or "")[:200],
        "failed_criteria": [c.get("name") for c in ev.get("criteria", []) if not c.get("passed")],
    }
    RUBRIC_LOG.append(entry)
    gaps = f" | gaps: {', '.join(entry['failed_criteria'])}" if entry["failed_criteria"] else ""
    print(f"🔎 rubric pass {entry['iteration']} — {entry['result']}{gaps}")


rubric_middleware = RubricMiddleware(
    model=grader_model,
    max_iterations=5,
    on_evaluation=_log_evaluation,
)

# The due diligence rubric — what "done" looks like for a vendor assessment. Passed per-run.
# Kept checkable and concrete so the light grader model can verify it against the transcript:
# every criterion maps to something the agent can literally show in the final message.
DD_RUBRIC = """Grade ONLY the agent's final report message (ignore intermediate chatter). The report
passes when it contains ALL of the following:
1. Sources — each evidence bullet carries an inline source tag, e.g. [source: quelmore_systems.md]
   or [source: vendor database].
2. Verification status — each evidence bullet is explicitly labelled verified / partially verified /
   unverified.
3. Risk — a named risk level (low, medium, or high) AND at least one risk signal with a severity.
4. Screening — a sanctions screening verdict (CLEAR, WATCHLIST, or SANCTIONED).
5. Structure — labelled sections for: vendor summary, evidence, risk signals, suitability,
   confidence, follow-up questions, and a human-review flag.
6. No PII — no email addresses or phone numbers anywhere in the report.
A criterion passes if the required content is present anywhere in the final message — wording and
formatting are up to the agent."""

print("rubric criteria:", len(DD_RUBRIC.splitlines()) - 1, "lines")
print(DD_RUBRIC)

# %% [markdown]
# ## 6. Build the deep agent
#
# 📖 [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
#
# One call: model + sub-agents + middleware stack. The supervisor's system prompt is short — the
# heavy lifting comes from `AGENTS.md` (memory), the skills list, and the rubric.
#
# > The middleware stack composes: memory injects `AGENTS.md`, skills advertises the
# > evidence-review workflow, rubric verifies the final report. Sub-agents each get the deepagents
# > default middleware stack automatically.

# %%
from deepagents import create_deep_agent

SUPERVISOR_PROMPT = (
    "You are the Acme vendor due diligence supervisor.\n"
    "Follow the operating instructions in your agent memory (AGENTS.md) at all times.\n"
    "For each due diligence request:\n"
    "1. Delegate evidence gathering to evidence_collector.\n"
    "2. Delegate risk classification to risk_assessor.\n"
    "3. Delegate sanctions/watchlist screening to compliance_screener.\n"
    "4. Synthesize ONE final due diligence report with these labelled sections, in this order:\n"
    "   **Vendor summary** · **Evidence** (one bullet per claim, each ending with its\n"
    "   [source: ...] tag and a verification label: verified / partially verified / unverified) ·\n"
    "   **Risk signals** (risk level low/medium/high + each signal with a severity) ·\n"
    "   **Compliance screening** (the CLEAR / WATCHLIST / SANCTIONED verdict) · **Suitability**\n"
    "   (high/medium/low) · **Confidence** (high/medium/low) · **Follow-up questions** ·\n"
    "   **Human review flag** (yes/no + why) · **Source validation status**.\n"
    "Return the full report as your final message — do not summarize or truncate it.\n"
    "If a skills list advertises an evidence-review skill, read it and follow its workflow.\n"
    "Never include PII (emails, phone numbers) in the report."
)

agent = create_deep_agent(
    model=model,
    subagents=SUBAGENTS,
    system_prompt=SUPERVISOR_PROMPT,
    middleware=[memory_middleware, skills_middleware, rubric_middleware],
    backend=backend,
    name="vendor_due_diligence_deep_agent",
)

# %% [markdown]
# **The agent's shape** — `create_deep_agent` returns a compiled LangGraph graph; render it to see
# the supervisor, the `task()` delegation edge, and the middleware hooks.

# %%
from IPython.display import Image, display

display(Image(agent.get_graph().draw_mermaid_png()))

# %% [markdown]
# ## 7. Run it — with and without the rubric
#
# 📖 [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
#
# `ask()` (from `utils/trace.py`) invokes the agent and prints a clickable **LangSmith trace** link.
# To pass a rubric into state we invoke with `{"messages": ..., "rubric": DD_RUBRIC}` — so we use
# `invoke_traced`, the full-result sibling of `ask()`.
#
# The request: assess **Quelmore Systems (VND-001)** for an avionics maintenance procurement need.
# Quelmore is a low-risk, QS-100-certified incumbent — a clean case where the report structure
# and source validation are what get tested.

# %%
from utils.trace import invoke_traced

REQUEST = (
    "Assess Quelmore Systems Ltd (VND-001) for an upcoming Acme avionics maintenance kits "
    "procurement (est. EUR 2.5M). Produce the due diligence report."
)

# %% [markdown]
# ### 7a. WITHOUT rubric — verification loop off
#
# No `rubric` key in the input → `RubricMiddleware` is a no-op. The agent answers in one pass:
# whatever structure and citations the prompt and AGENTS.md produce, **unchecked**.

# %%
result_plain = invoke_traced(agent, {"messages": [{"role": "user", "content": REQUEST}]})
report_plain = result_plain["messages"][-1].content
print(report_plain)

# %% [markdown]
# ### 7b. WITH rubric — verification loop on
#
# Same request, now with `state["rubric"]` set. Watch the trace: when the first draft misses a
# criterion (typically source validation statuses or the human-review flag), the grader returns
# `needs_revision` with the gap, and the agent revises before finishing.

# %%
RUBRIC_LOG.clear()  # reset the grading trail so this run's passes are easy to read
result_verified = invoke_traced(agent, {"messages": [{"role": "user", "content": REQUEST}], "rubric": DD_RUBRIC})
report_verified = result_verified["messages"][-1].content
print(report_verified)

# %% [markdown]
# **Compare** — the `_rubric_status` / `_rubric_iterations` keys are **private state** and are not
# part of the `.invoke` output, so the supported way to observe grading without a checkpointer is
# the `on_evaluation` callback (captured into `RUBRIC_LOG`). A `satisfied` verdict on the last
# pass means the report met all six criteria; `needs_revision` on every pass means the agent hit
# the iteration budget and its last draft is returned intact — open the trace to see the gaps the
# grader flagged.

# %%
final = RUBRIC_LOG[-1] if RUBRIC_LOG else None
print("grading passes:", len(RUBRIC_LOG))
print("final verdict:", final["result"] if final else "n/a (no rubric supplied)")
for p in RUBRIC_LOG:
    gaps = f" — gaps: {', '.join(p['failed_criteria'])}" if p["failed_criteria"] else ""
    print(f"  pass {p['iteration']}: {p['result']}{gaps}")


# %%
def _quick_check(report: str) -> dict:
    """Cheap local sanity scan of a report — the same things the rubric grader checks, so we can
    eyeball the with/without-rubric difference without opening the trace."""
    text = report.lower()
    return {
        "cites a source": "source:" in text or "[source" in text,
        "verification status": any(k in text for k in ("verified", "partially verified", "unverified")),
        "risk level stated": any(k in text for k in ("low risk", "medium risk", "high risk", "risk: low", "risk level")),
        "screening included": any(k in text for k in ("clear", "sanction", "watchlist")),
        "no PII (emails)": "@" not in text,
        "follow-ups present": any(k in text for k in ("follow-up", "follow up", "questions")),
    }


print("WITHOUT rubric:", json.dumps(_quick_check(report_plain), indent=2))
print()
print("WITH rubric:   ", json.dumps(_quick_check(report_verified), indent=2))

# %% [markdown]
# ## 8. Demonstrate source validation
#
# The core due diligence behaviour: a claim is only as good as its source, and one source is not
# verification. Ask about a **specific, checkable claim** — Quelmore's QS-100 certification — and
# require the agent to validate it against two independent sources (KB article + PDF capability
# statement) and label its verification status.
#
# This runs **with** the rubric so the grader enforces the source-validation criterion.

# %%
VALIDATION_REQUEST = (
    "Quelmore Systems claims Acme QS-100 certification. Validate this claim: check the vendor "
    "knowledge base AND the vendor's PDF capability statement, mark the claim verified / partially "
    "verified / unverified, and explain your reasoning. Cite every source."
)

validation_result = invoke_traced(
    agent,
    {"messages": [{"role": "user", "content": VALIDATION_REQUEST}], "rubric": DD_RUBRIC},
)
validation_report = validation_result["messages"][-1].content
print(validation_report)

# %% [markdown]
# **Negative case** — a claim that should **not** verify: Quelmore's KB profile says ISO 27001 is
# "not yet achieved". An agent that validates sources should mark this one *unverified / contradicted*
# and flag it — not launder it into the report. This is the anti-hallucination rule (AGENTS.md §7)
# exercised end to end.

# %%
FALSE_CLAIM_REQUEST = (
    "Confirm that Quelmore Systems Ltd (VND-001) holds ISO 27001 information security certification. "
    "Validate against the knowledge base and the PDF capability statement and mark the claim's "
    "verification status."
)

false_claim_result = invoke_traced(
    agent,
    {"messages": [{"role": "user", "content": FALSE_CLAIM_REQUEST}], "rubric": DD_RUBRIC},
)
print(false_claim_result["messages"][-1].content)

# %% [markdown]
# ## 9. Where the agent's files live — and the sandbox option
#
# Section 3 gave this agent a `FilesystemBackend(virtual_mode=True)`: its `read_file` / `write_file`
# tools are rooted under `day2/data/`, and it has **no shell**. For a workshop over committed
# fixtures that is the right default — the tools are pure-Python lookups, so nothing needs to
# execute.
#
# The moment an agent needs to *run* something — install a package, run a test, use a CLI — you want
# a **sandbox backend** instead. Sandboxes implement one method, `execute()`, and Deep Agents builds
# every file operation on top of it; when the harness detects a sandbox it adds an `execute` tool.
#
# | Provider | Import | Credential |
# |---|---|---|
# | **LangSmith** | `deepagents.backends.langsmith.LangSmithSandbox` | `LANGSMITH_API_KEY` (nothing else) |
# | AWS AgentCore | `langchain_agentcore_codeinterpreter.AgentCoreSandbox` | AWS creds |
# | Daytona / E2B / Modal / Runloop / Vercel / NVIDIA OpenShell | `langchain_<provider>` | provider key |
#
# > 🔐 **Why an Acme team should care.** A sandbox is the difference between "the agent can write
# > anywhere the notebook user can" and "the agent acts inside a disposable, isolated VM with no
# > access to the analyst's laptop". That is a governance answer, not a convenience — and it is the
# > same isolation Harbor relies on to score evals reproducibly (Lab 06 §10).
#
# The LangSmith sandbox needs no extra provider account: your existing `LANGSMITH_API_KEY` is enough.

# %%
# Live sandbox: create → execute → clean up. Skips cleanly if sandboxes are unavailable on your plan.
def sandbox_demo() -> None:
    try:
        from langsmith.sandbox import SandboxClient
    except ImportError:
        print("⏭ this langsmith SDK has no sandbox module — upgrade langsmith to try it")
        return

    client = SandboxClient()
    sandbox = None
    try:
        sandbox = client.create_sandbox(name="acme-dd-demo", wait_for_ready=True, timeout=180)
        print("sandbox created:", sandbox.id)

        from deepagents.backends.langsmith import LangSmithSandbox

        sandbox_backend = LangSmithSandbox(sandbox=sandbox)

        # The `execute` tool the agent would get. Note it is a *different machine* — the vendor
        # fixtures are NOT here unless you copy them in, which is exactly the isolation you wanted.
        for command in ("python3 --version", "ls /"):
            result = sandbox_backend.execute(command)
            print(f"  $ {command}\n    {str(getattr(result, 'output', result)).strip()[:120]}")
    except Exception as e:
        print(f"⏭ sandbox unavailable ({type(e).__name__}: {str(e)[:120]})")
        print("   The lab continues on FilesystemBackend — nothing above depends on this cell.")
    finally:
        if sandbox is not None:
            try:
                client.delete_sandbox(sandbox.id)
                print("sandbox deleted")
            except Exception:
                print("⚠ could not delete the sandbox — check Settings → Sandboxes in LangSmith")


sandbox_demo()

# %% [markdown]
# **Which backend for this agent?** Keep `FilesystemBackend` — the due diligence tools read committed
# JSON and Markdown, so a sandbox would add latency and a network dependency for no capability. Reach
# for a sandbox when the agent needs a shell. Lab 06 §12 compares all six backends and what each one
# means for *testability*.
#
# > ⚠️ `LocalShellBackend` also exists and gives the agent a real shell **on your machine**. Never
# > point it at a laptop you care about; that is what sandboxes are for.

# %% [markdown]
# ## 10. Open it in LangGraph Studio
#
# 📖 [Studio quick start](https://docs.langchain.com/langsmith/quick-start-studio)
#
# The deep agent is registered in `day1/langgraph.json` as **`vendor_due_diligence_deep_agent`**, so
# you can drive it in Studio the same way Day 1 §6 drove the ticket agent:
#
# ```bash
# cd day1
# uv run langgraph dev
# ```
#
# Studio: **https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024**
#
# Studio earns its keep more on a deep agent than on a single agent, because the interesting
# behaviour is *structural*:
#
# | In Studio, look at | What it tells you |
# |---|---|
# | The graph shape | the middleware nodes — memory, skills, todo — that wrap every model call |
# | A `task` tool call | which sub-agent the supervisor chose, and the isolated context it was handed |
# | The sub-agent's own steps | whether the specialist did its job, separately from the synthesis |
# | State on each step | what `AGENTS.md` injected, and what the filesystem tools wrote |
#
# | Try in Studio | Watch for |
# |---|---|
# | *"Full due diligence on Kelbrant Trading Consortium (VND-013)."* | three `task` delegations, then a SANCTIONED verdict that is not softened |
# | *"Summarize Quelmore's certifications."* | **one** delegation — a cheap question should not cost full diligence (the efficiency point in Lab 06 §9) |
#
# > 💡 This is the visual version of Lab 06's **graph trajectory** layer: Studio shows you the node
# > sequence by eye; `extract_langgraph_trajectory_from_thread` asserts on it in CI. Same signal, one
# > for debugging and one for regression.

# %% [markdown]
# ## 11. Wrap-up
#
# | Piece | What you saw |
# |---|---|
# | `create_deep_agent` | supervisor + `task()` delegation to 3 narrow sub-agents |
# | `MemoryMiddleware` | `AGENTS.md` operating instructions always in the system prompt |
# | `SkillsMiddleware` | `evidence-review` skill offered by progressive disclosure — read on demand |
# | `RubricMiddleware` | grader loops the agent until the report passes 6 DD criteria |
# | Source validation | claims verified across KB + PDF; a false claim caught and flagged |
#
# **Next:** Lab 03 builds offline evals over a dataset; Lab 05 adds the human review
# gate for medium/high-risk vendors.
#
# > 📝 **When `day2/data/agent/skills/evidence-review/SKILL.md` is created**, rerun Section 4 and the
# > runs above — the evidence-review workflow will appear in the system prompt's skills list and the
# > agent will follow it (progressive disclosure: listed always, read on demand).
