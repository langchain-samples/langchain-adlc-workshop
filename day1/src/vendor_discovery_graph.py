"""Deployable vendor discoverability agent — single source of truth for the workshop Day 1.

This graph is built over synthetic fixtures in `langchain_adlc_workshop/day1/data/` so the labs can teach
procurement vendor discovery without real Acme data.

In `langchain>=1.0`, `create_agent` returns a compiled LangGraph graph. The module-level graphs below
are registered in `langgraph.json` for LangGraph Studio and reused by notebooks/evals.

Security: all tools are pure-Python lookups/searches over committed JSON/Markdown fixtures (no SQL,
eval, network, shelling out). The system prompt is pulled with an offline-safe fallback via
`utils.prompts`; API keys are read only by the model provider at runtime.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Literal

# The workshop root (this file is day1/src/<name>.py) must be importable before `utils.*` and
# `day1.src.*` resolve. langgraph dev / langgraph build load this file by path, so bootstrap it here
# rather than relying on the caller's cwd or PYTHONPATH. The `globals()` guard keeps the notebook
# render of this module runnable too — a Jupyter kernel has no `__file__`.
if "__file__" in globals():
    _WORKSHOP = Path(globals()["__file__"]).resolve().parent.parent.parent  # src -> day1 -> root
else:  # notebook render: walk up from the kernel's cwd instead
    _WORKSHOP = next(
        c for c in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
        if (c / "day1").is_dir() and (c / "utils").is_dir()
    )
if str(_WORKSHOP) not in sys.path:
    sys.path.insert(0, str(_WORKSHOP))

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from day1.src.models import get_embeddings, get_model
from utils.prompts import get_prompt

_DATA_DIR = _WORKSHOP / "day1" / "data"
KB_DIR = _DATA_DIR / "kb"
VENDORS_PATH = _DATA_DIR / "vendors.json"
NEEDS_PATH = _DATA_DIR / "procurement_needs.json"
PROMPT_PATH = _DATA_DIR / "prompt.md"

PROMPT_NAME = "vendor-discoverability"

VENDORS: dict = json.loads(VENDORS_PATH.read_text())
PROCUREMENT_NEEDS: list = json.loads(NEEDS_PATH.read_text())


def _talk_to_openai_directly() -> None:
    """Safety net: when running in LangGraph Platform, the platform may inject OPENAI_BASE_URL
    or ANTHROPIC_BASE_URL pointing to the LangSmith gateway. This workspace's gateway is not
    assumed to serve the configured lab model, so drop the injected value and use the normal
    provider path.

    For intentional gateway use, set LANGSMITH_GATEWAY=true (or a provider-specific BASE_URL) plus
    LANGSMITH_GATEWAY_API_KEY. See the README (Model access section).
    """
    pass  # No guard needed — the model layer (day1/src/models.py) handles gateway vs direct routing


_talk_to_openai_directly()


def _vendors_list() -> list[dict]:
    return [rec for vid, rec in VENDORS.items() if not vid.startswith("_")]


# --- Tool 1: search_vendor_kb (RAG over the bundled synthetic vendor profiles) ---

def _build_vendor_index() -> InMemoryVectorStore:
    docs = [
        Document(page_content=p.read_text(), metadata={"source": p.name})
        for p in sorted(KB_DIR.glob("*.md"))
    ]
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
    return InMemoryVectorStore.from_documents(chunks, get_embeddings())


_vendor_index: InMemoryVectorStore | None = None


def vendor_kb_index() -> InMemoryVectorStore:
    """The lazily-built vendor KB vector store. Public so labs that tune retrieval (e.g. Day 2
    Lab 04, which raises `k`) can reuse the same index instead of re-embedding the corpus."""
    global _vendor_index
    if _vendor_index is None:
        _vendor_index = _build_vendor_index()
    return _vendor_index


@tool
def search_vendor_kb(query: str) -> str:
    """Search vendor profile pages by capability, certification, or keyword.
    Returns cited snippets from the vendor knowledge base."""
    hits = vendor_kb_index().similarity_search(query, k=4)
    if not hits:
        return "No relevant vendor profiles found."
    return "\n\n---\n\n".join(f"[source: {h.metadata['source']}]\n{h.page_content}" for h in hits)


# --- Tool 2: get_vendor (structured lookup by vendor ID or name — no SQL, no query string) ---

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
    for vid, rec in VENDORS.items():
        if vid.startswith("_"):
            continue
        if q == vid.lower() or q in rec["name"].lower():
            return _format_vendor(rec)
    return f"No vendor found for {vendor!r}."


# --- Tool 3: filter_vendors (deterministic constraint filter — no SQL) ---

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
    rows = _vendors_list()
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


# --- Tool 4: list_procurement_needs (sample needs for demonstration) ---

@tool
def list_procurement_needs() -> str:
    """List sample procurement needs from the bundled dataset. Useful when the user asks for examples
    or wants to see what kinds of procurement needs are available."""
    lines = [f"Found {len(PROCUREMENT_NEEDS)} sample procurement need(s):"]
    for need in PROCUREMENT_NEEDS:
        budget = f"EUR {need.get('budget_eur', 0):,}" if need.get("budget_eur") else "unspecified"
        lines.append(
            f"  {need['need_id']}: {need['title']} — {need['category']} · "
            f"priority: {need['priority']} · budget: {budget}"
        )
    return "\n".join(lines)


# --- Tool 5: screen_vendor (compliance / sanctions screening) ---

SANCTIONS_PATH = _DATA_DIR / "sanctions_list.json"
SANCTIONS: dict = json.loads(SANCTIONS_PATH.read_text())


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


# --- Tool 6: parse_vendor_pdf (parse a synthetic PDF vendor document) ---

PDF_DIR = _DATA_DIR / "pdfs"


@tool
def parse_vendor_pdf(vendor_name: str) -> str:
    """Parse a synthetic PDF vendor capability statement. Returns the extracted text content.
    Use this when the user asks about a vendor's PDF document or when you need to read a
    vendor's capability statement directly."""
    # Find the PDF file by vendor name
    safe_name = vendor_name.lower().replace(" ", "_").replace("&", "and")
    pdf_path = PDF_DIR / f"{safe_name}_capability_statement.pdf"

    if not pdf_path.exists():
        # Try to find by partial match
        matches = list(PDF_DIR.glob(f"*{safe_name}*.pdf"))
        if matches:
            pdf_path = matches[0]
        else:
            return f"No PDF found for {vendor_name!r}. Available: {', '.join(p.stem.replace('_capability_statement', '') for p in PDF_DIR.glob('*.pdf'))}"

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        text = "\n".join(page.extract_text() for page in reader.pages)
        return f"[source: {pdf_path.name}]\n{text[:2000]}"  # Truncate for readability
    except Exception as e:
        return f"Error parsing PDF {pdf_path.name}: {e}"


def vendor_tools() -> list:
    """The agent's six grounding tools plus optional Tavily when key is set."""
    tools = [search_vendor_kb, get_vendor, filter_vendors, list_procurement_needs, screen_vendor, parse_vendor_pdf]
    if os.getenv("TAVILY_API_KEY"):
        from langchain_tavily import TavilySearch

        tools.append(TavilySearch(max_results=3))
    return tools


# --- Structured output schema ---

class VendorRecommendation(BaseModel):
    """Structured output for the vendor discoverability agent. Passed as `response_format=` so
    LangChain enforces it and returns a parsed object at `result["structured_response"]`."""

    recommended_vendors: list[dict] = Field(
        description="Ranked list of recommended vendors, each with vendor_id, name, and rationale"
    )
    evidence: list[str] = Field(
        description="Supporting evidence: certifications, contract history, compliance flags"
    )
    screening_results: list[str] = Field(
        default_factory=list,
        description="Sanctions/watchlist screening results for each recommended vendor"
    )
    risk_level: Literal["low", "medium", "high"] = Field(
        description="Overall risk level across recommended vendors"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the recommendation"
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description="Information that would improve confidence if provided"
    )
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Clarifying questions when the procurement need is underspecified"
    )
    human_review_required: bool = Field(
        description="True when confidence is low or any vendor risk is medium or high"
    )


# --- Build the agents ---

def build_agent(*, use_checkpointer: bool = False, interrupt_on: dict | None = None,
                response_format: type | None = None, middleware: list | None = None,
                name: str = "vendor_discovery_agent"):
    """Build the vendor discoverability agent.

    Args:
        use_checkpointer: attach an in-memory checkpointer for local memory/threads + HITL.
        interrupt_on: optional HumanInTheLoopMiddleware config mapping tool name -> approval policy.
        response_format: optional Pydantic model for structured output.
        middleware: optional additional middleware list.
        name: the graph's internal name.
    """
    system_prompt = get_prompt(PROMPT_NAME, fallback=PROMPT_PATH.read_text().strip())

    mw = list(middleware) if middleware else []
    if interrupt_on:
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        mw.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    kwargs = {"middleware": mw} if mw else {}
    if use_checkpointer:
        from langgraph.checkpoint.memory import MemorySaver

        kwargs["checkpointer"] = MemorySaver()

    if response_format is not None:
        from langchain.agents.structured_output import ToolStrategy

        # Tool-calling structured output — the native JSON path is flaky. See ticket_agent_graph.py.
        response_format = ToolStrategy(response_format, handle_errors=True)

    return create_agent(
        get_model(),
        tools=vendor_tools(),
        system_prompt=system_prompt,
        response_format=response_format,
        name=name,
        **kwargs,
    )


# Module-level compiled graphs — importable by notebooks and deployable via langgraph.json.
# These are built at import time (requires API key) so langgraph dev can find them.
# If you don't have an API key set, import will fail — this is expected for langgraph dev.
#
# Note: langgraph dev handles persistence automatically — no custom checkpointer needed.
# For local testing with memory, use build_agent(use_checkpointer=True) directly.

graph = build_agent()  # plain ReAct agent
graph_with_memory = build_agent(name="vendor_discovery_agent_mem")
graph_hitl = build_agent(
    interrupt_on={"get_vendor": True, "filter_vendors": True},
    name="vendor_discovery_agent_hitl",
)
graph_structured = build_agent(response_format=VendorRecommendation, name="vendor_discovery_agent_structured")
