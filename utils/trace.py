"""Invoke an agent and surface a clickable LangSmith trace link.

This keeps the tracing plumbing out of the lab notebooks: instead of a ~15-line helper per notebook,
a cell just calls ``ask(graph, "...")`` and gets the answer plus a printed trace URL. When
``LANGSMITH_TRACING`` is off, it simply returns the answer (no link, no error).
"""

import os
import re

from langchain_core.tracers.context import tracing_v2_enabled

# The SDK's cb.get_run_url() returns the /r/{run}?poll=true route, which 500s ("Something went wrong")
# for some orgs. The project "peek" route renders the same trace reliably, so we rewrite the URL into
# that form. The run id IS the trace id, so peek/peeked_trace both use it.
_RUN_URL_RE = re.compile(r"^(?P<base>https?://[^/]+/o/[^/]+/projects/p/(?P<proj>[0-9a-fA-F-]+))/r/(?P<run>[0-9a-fA-F-]+)")


def _peek_url(cb) -> str:
    """Clickable trace URL. Prefer the reliable project-peek route; fall back to the raw SDK URL."""
    raw = cb.get_run_url()
    m = _RUN_URL_RE.match(raw)
    if not m:
        return raw
    proj, run = m.group("proj"), m.group("run")
    return f"{m.group('base')}?peek={run}&peek_project={proj}&peeked_trace={run}"


def ask(graph, question: str, config: dict | None = None, context: dict | None = None):
    """Send ``question`` to a compiled agent ``graph`` and return its final answer.

    When LangSmith tracing is on, also prints a clickable trace URL for the run.
    ``config`` passes through to ``.invoke`` (e.g. a ``thread_id`` for memory); ``context``
    passes through as static runtime context for agents built with ``context_schema=``.

    Agents built with ``response_format=`` return their parsed object under
    ``structured_response`` — return that when present (so the caller sees the typed report), else
    fall back to the final message text.
    """
    payload = {"messages": [{"role": "user", "content": question}]}
    kwargs = {"config": config, "context": context}
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        with tracing_v2_enabled() as cb:
            result = graph.invoke(payload, **kwargs)
        try:
            print("🔗 trace:", _peek_url(cb))
        except Exception:
            pass  # a missing link should never break the lab
    else:
        result = graph.invoke(payload, **kwargs)
    if result.get("structured_response") is not None:
        return result["structured_response"]
    return result["messages"][-1].content


def invoke_traced(graph, inputs, config: dict | None = None, context: dict | None = None) -> dict:
    """Invoke ``graph`` and return its FULL result dict, printing a clickable trace URL when tracing is on.

    Use this instead of ``ask()`` when a cell needs the whole result — ``messages``, ``__interrupt__``,
    ``structured_response`` — not just the final answer. ``inputs`` is whatever ``.invoke`` accepts (a
    ``{"messages": …}`` payload or a ``Command(resume=…)``). ``context`` passes through as static
    runtime context for agents built with ``context_schema=``.
    """
    kwargs = {"config": config, "context": context}
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        with tracing_v2_enabled() as cb:
            result = graph.invoke(inputs, **kwargs)
        try:
            print("🔗 trace:", _peek_url(cb))
        except Exception:
            pass  # a missing link should never break the lab
    else:
        result = graph.invoke(inputs, **kwargs)
    return result
