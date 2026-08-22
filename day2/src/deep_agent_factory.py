"""Shared factory for the vendor due diligence Deep Agent.

Lab 01 (`01_deep_agent.py`) builds this agent step by step — that build-up *is* the lesson, so it
stays inline there. This module is the same agent as a single importable call, for the places that
need to *use* it rather than teach it:

- `06_deep_agent_evals.py` evaluates it (and needs to toggle memory on/off for an A/B)
- `03_evals_datasets.py` / `04_experiments.py` can target it instead of the scaffold stub
- Day 3 can deploy it

Keeping one factory means an eval always grades the agent the labs actually build. If you change a
sub-agent prompt in Lab 01, change it here too — or better, import from here in Lab 01 once
participants have seen it assembled.

Security: every tool is a pure-Python lookup over committed synthetic fixtures. The filesystem
backend runs in `virtual_mode=True`, which roots all absolute tool paths under `day2/data/` rather
than the real filesystem.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

_WORKSHOP = Path(__file__).resolve().parent.parent.parent  # src -> day2 -> workshop root
if str(_WORKSHOP) not in sys.path:
    sys.path.insert(0, str(_WORKSHOP))

from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent

from day1.src.models import get_judge_model, get_model
from day1.src.vendor_discovery_graph import (
    filter_vendors,
    get_vendor,
    screen_vendor,
    search_vendor_kb,
)

DAY2_DATA = _WORKSHOP / "day2" / "data"
AGENTS_MD_REL = "agent/AGENTS.md"
SKILLS_DIR_REL = "agent/skills"

SUPERVISOR_PROMPT = (
    "You are the Acme vendor due diligence supervisor.\n"
    "Follow the operating instructions in your agent memory (AGENTS.md) at all times.\n"
    "For each due diligence request:\n"
    "1. Delegate evidence gathering to evidence_collector.\n"
    "2. Delegate risk classification to risk_assessor.\n"
    "3. Delegate sanctions/watchlist screening to compliance_screener.\n"
    "4. Synthesize ONE final report with labelled sections: vendor summary, evidence (each bullet "
    "ending in its [source: ...] tag and a verification label), risk signals (level + severity), "
    "compliance screening verdict, suitability, confidence, follow-up questions, and a "
    "human-review flag.\n"
    "Never include contact emails, phone numbers, or other PII."
)


def _subagents() -> list[SubAgent]:
    """The three specialists, matching Lab 01 §2."""
    evidence_collector: SubAgent = {
        "name": "evidence_collector",
        "description": (
            "Gathers cited evidence about a vendor's capabilities, certifications, and track record "
            "from the vendor knowledge base. Use first whenever an assessment needs evidence."
        ),
        "system_prompt": (
            "You are the evidence collector for an Acme vendor due diligence team.\n"
            "- Search the vendor KB for capability and certification evidence.\n"
            "- Every claim you return MUST cite its source ([source: ...] tag from the tool output).\n"
            "- Never invent evidence. If you cannot find support, say 'no evidence found'.\n"
            "Return one bullet per claim, with source and verification status."
        ),
        "tools": [search_vendor_kb],
    }
    risk_assessor: SubAgent = {
        "name": "risk_assessor",
        "description": (
            "Classifies a vendor's risk level (low/medium/high) and surfaces risk signals with "
            "severity, using the vendor database. Use after evidence is gathered."
        ),
        "system_prompt": (
            "You are the risk assessor for an Acme vendor due diligence team.\n"
            "- Use get_vendor for the full record; filter_vendors to compare against peers.\n"
            "- Classify risk low/medium/high and list each signal with a severity.\n"
            "- Ground every signal in the record — cite [source: vendor database].\n"
            "Return: risk level, risk signals, and data gaps."
        ),
        "tools": [get_vendor, filter_vendors],
    }
    compliance_screener: SubAgent = {
        "name": "compliance_screener",
        "description": (
            "Screens a vendor against sanctions lists and watchlists. Always run before "
            "recommending any vendor; a SANCTIONED or WATCHLIST match is an automatic escalation."
        ),
        "system_prompt": (
            "You are the compliance screener for an Acme vendor due diligence team.\n"
            "- Screen every vendor under assessment — no exceptions.\n"
            "- Report the verdict verbatim: CLEAR, WATCHLIST, or SANCTIONED, with the reason.\n"
            "- A SANCTIONED match means the vendor must be excluded and escalated; say so plainly.\n"
            "Return: screening verdict, matched entity (if any), and the source used."
        ),
        "tools": [screen_vendor],
    }
    return [evidence_collector, risk_assessor, compliance_screener]


def _backend():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from deepagents.backends.filesystem import FilesystemBackend

        # virtual_mode=True roots the agent's absolute tool paths under DAY2_DATA. With it False,
        # a write to "/wiki/x.md" targets the real machine root and fails — see the README (Model access section)'s
        # sibling note in day2/src/02_wiki_memory.py §5.
        return FilesystemBackend(root_dir=DAY2_DATA, virtual_mode=True)


def build_dd_agent(
    *,
    with_memory: bool = True,
    with_skills: bool = True,
    with_rubric: bool = False,
    checkpointer=None,
    name: str = "vendor_due_diligence_deep_agent",
    extra_tools: list | None = None,
    interrupt_on: dict | None = None,
):
    """Build the due diligence deep agent.

    Args:
        with_memory: load `AGENTS.md` via the `memory=` parameter. Set False to A/B whether the
            operating instructions actually change behaviour (see Lab 06 §6).
        with_skills: expose the four `SKILL.md` workflows (evidence review, source validation,
            risk classification, report drafting) via the `skills=` parameter.
        with_rubric: add `RubricMiddleware` runtime verification. Off by default because it only
            activates when the caller passes a `rubric` in state, and because an eval usually wants
            to grade the *unverified* draft — measuring the agent, not the agent plus its grader.
        checkpointer: pass a `MemorySaver()` for multi-turn or graph-trajectory evals.
        name: graph name, as it appears in traces.
        extra_tools: additional supervisor-level tools. Used by Lab 06 §7b to attach a gated
            `submit_recommendation` action so an interrupt-aware simulation has something to pause on.
        interrupt_on: map of tool name -> True to gate behind human approval. `create_deep_agent`
            supports this natively, so there is no need to attach `HumanInTheLoopMiddleware` by
            hand — used by Lab 06 §7b's interrupt-aware simulation.
    """
    backend = _backend()
    middleware = []

    # Skills and memory are passed as FIRST-CLASS parameters, which is what the docs prescribe:
    #   create_deep_agent(..., memory=["/memories/AGENTS.md"], skills=["/skills/"])
    # See https://docs.langchain.com/oss/python/deepagents/skills and .../memory
    #
    # Hand-wiring `SkillsMiddleware` / `MemoryMiddleware` into `middleware=` also works, but places
    # them in the user-middleware slot rather than their canonical positions, and skips the
    # `add_cache_control=True` the library passes for memory.

    if with_rubric:
        from deepagents.middleware import RubricMiddleware

        # The grader runs on the JUDGE tier, not the agent tier — a verifier that shares the
        # agent's blind spots will pass the agent's mistakes. This one IS user middleware: it has
        # no canonical slot, because it is not part of the harness profile.
        middleware.append(RubricMiddleware(model=get_judge_model(), max_iterations=5))

    kwargs = {"checkpointer": checkpointer} if checkpointer is not None else {}
    if extra_tools:
        kwargs["tools"] = list(extra_tools)
    if interrupt_on:
        kwargs["interrupt_on"] = interrupt_on
    if with_memory:
        kwargs["memory"] = [AGENTS_MD_REL]      # -> MemoryMiddleware, placed after prompt caching
    if with_skills:
        kwargs["skills"] = [SKILLS_DIR_REL]     # -> SkillsMiddleware, placed before Filesystem
    return create_deep_agent(
        model=get_model(),
        subagents=_subagents(),
        system_prompt=SUPERVISOR_PROMPT,
        middleware=middleware,
        backend=backend,
        name=name,
        **kwargs,
    )


# Module-level compiled graph — registered in `day1/langgraph.json` so the Day 2 deep agent can be
# opened in LangGraph Studio and deployed, exactly like the Day 1 ticket agents.
#
# Deliberately no checkpointer and no rubric: LangGraph Server supplies persistence (a custom
# checkpointer makes it refuse to start), and the rubric only activates when a caller passes a
# `rubric` in state — so leaving it off keeps the deployed graph honest about what it verifies.
graph = build_dd_agent()
