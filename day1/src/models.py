"""Single place to pick the chat model for every workshop lab.

Two model access paths are supported:

1. **Direct API key** (default) — when no gateway base URL is set, `init_chat_model` reads
   `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` directly from the environment. This is the path for
   local development and workshop laptops without gateway access.

2. **LangSmith LLM Gateway** — `init_chat_model` routes through the gateway. The gateway handles
   model routing, rate limiting, spend controls, and sensitive-data redaction. Use this path when
   the gateway is configured and you want centralized governance.

   To enable, either set `LANGSMITH_GATEWAY=true` (the documented convenience switch) or point the
   provider-specific base URL at the gateway, then set `LANGSMITH_GATEWAY_API_KEY` to a
   workspace-scoped LangSmith key with the `gateway:invoke` permission. You do **not** set a
   provider key on this path, and you do not put the gateway key into `OPENAI_API_KEY`.
   See the README (Model access section) for full setup instructions.

API keys are read from the environment (see `langchain_adlc_workshop/.env.example`) — never hardcode them here.
"""

import os

from langchain.chat_models import init_chat_model

# Three tiers, chosen by measurement on this workshop's own workload (2026-08-18) — swap here in
# one place. Override the agent tier with LABS_MODEL, the judge tier with LABS_JUDGE_MODEL.
#
# | Tier | Model | Used by | Why this one |
# |---|---|---|---|
# | agent | `gpt-4.1-mini` | every lab's agent | 3/3 on proposing the HITL-gated `mock_api_action`, correctly declines for unauthorized users, 3/3 on `response_format=TicketResolution`, ~2.5–4.5s |
# | heavy | `gpt-4.1` | Lab 04 "would a better agent fix this?" | same family as the agent tier, so an A/B isolates capability rather than provider quirks |
# | judge | `gpt-5.4` | LLM-as-judge evaluators, `RubricMiddleware` | 18/18 against the documented ground truth vs **15/18** for every cheaper judge tried |
#
# **The judge is deliberately the most expensive tier.** Grading a response is usually harder than
# producing one, and the cheap judges did not fail *randomly* — `gpt-4o-mini`, `gpt-4.1-mini` and
# `gpt-4.1` all passed T-9002 (a sensitive account action taken on an unverified identity at low
# confidence) with the explanation that it "properly uses the mock_api_action tool despite low
# confidence and missing identity check". A judge that is confidently wrong on the safety-critical
# case is worse than no judge, because the improvement loop trusts it. `gpt-5.4` failed that case.
#
# The cost asymmetry points the same way: agent calls scale with production traffic (unbounded),
# while judge calls scale with dataset size × experiments offline, or traffic × sampling rate online
# (bounded, and you choose the bound). Spend on the judge; economize on the agent. And gate on
# deterministic *code* evaluators wherever you can — they cost nothing and never drift.
#
# Rejected: `gpt-5-mini` (0/3 on the HITL gate, 52s median — unusable in a live workshop) and
# `gpt-5.4-mini` (fast at 2.5s but only 2/3 on the gate).
DEFAULT_MODEL = os.getenv("LABS_MODEL", "openai:gpt-4.1-mini")        # agent — most labs
HEAVY_MODEL = "openai:gpt-4.1"                                        # stronger agent — A/B comparisons
JUDGE_MODEL = os.getenv("LABS_JUDGE_MODEL", "openai:gpt-5.4")         # graders and LLM-as-judge


def _timeout() -> float:
    """Per-request timeout in seconds.

    Load-bearing, not defensive boilerplate: `max_retries` only helps when a call *errors*. A request
    that simply never returns hangs forever and the retry logic never fires — during verification a
    lab sat stalled inside `evaluate()` for 59 minutes with no output and no error. A hung cell in
    front of a room is worse than a failed one, because nobody can tell it from slow.

    120s is generous enough for a deep-agent turn with several tool calls, and short enough that a
    genuinely stuck call surfaces as an error the retry budget can then act on. Override with
    LABS_TIMEOUT.
    """
    raw = os.getenv("LABS_TIMEOUT", "")
    try:
        value = float(raw)
        return value if value > 0 else 120.0
    except ValueError:
        return 120.0


def _max_retries() -> int:
    """Client-level retry budget (exponential backoff) so a transient connection drop — e.g. ECONNRESET
    when a laptop naps mid-run — self-heals instead of killing a long agent/eval run. Trusted int env
    override only; anything non-numeric or negative falls back to the default."""
    raw = os.getenv("LABS_MAX_RETRIES", "")
    return int(raw) if raw.isdigit() else 6


_MISSING_GATEWAY_KEY = (
    "The LangSmith LLM Gateway is configured as the model endpoint (via LANGSMITH_GATEWAY, or a "
    "BASE_URL pointing at gateway.smith.langchain.com), but no gateway key is set. Set "
    "LANGSMITH_GATEWAY_API_KEY to a workspace-scoped LangSmith key with the `gateway:invoke` "
    "permission (see the README (Model access section)), or unset the gateway configuration to use "
    "a direct provider key instead."
)

GATEWAY_HOST = "gateway.smith.langchain.com"
# The base URL the gateway serves OpenAI-compatible traffic on, per
# https://docs.langchain.com/langsmith/llm-gateway-quickstart ("GW default").
DEFAULT_GATEWAY_OPENAI_BASE = f"https://{GATEWAY_HOST}/openai/v1"


def _gateway_setting() -> str:
    """The raw `LANGSMITH_GATEWAY` value, or "" when it is unset or explicitly off.

    `LANGSMITH_GATEWAY` is the convenience switch documented on
    https://docs.langchain.com/langsmith/llm-gateway-quickstart — `"true"` routes supported chat
    models through the default gateway, and a URL routes them through a regional one (e.g. the EU
    instance). Anything falsy is treated as "not configured".
    """
    raw = os.environ.get("LANGSMITH_GATEWAY", "").strip()
    return "" if raw.lower() in ("", "false", "0", "no", "off") else raw


def _using_gateway() -> bool:
    """True when the LangSmith gateway is configured as the model endpoint.

    Three configurations count, because all three are things a participant may actually have set:

    1. `LANGSMITH_GATEWAY=true` (or a regional gateway URL) — the **documented** convenience switch.
    2. `OPENAI_BASE_URL` pointing at the gateway — OpenAI-compatible providers.
    3. `ANTHROPIC_BASE_URL` pointing at the gateway — Anthropic.

    Case 1 is the one worth understanding, because LangChain handles it only *partly*.
    `LANGSMITH_GATEWAY` is honoured by the supported **chat** models, so `init_chat_model` routes
    itself with no help from us. `OpenAIEmbeddings` does **not** honour it — with no provider key in
    the environment it raises `OpenAIError: Missing credentials` the moment you construct it. So a
    participant who follows the official quickstart gets a working agent and a hard failure in every
    RAG lab. Recognising case 1 here is what lets `get_embeddings()` route explicitly and keep the
    two in step.
    """
    return bool(_gateway_setting()) or any(
        GATEWAY_HOST in os.environ.get(var, "")
        for var in ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL")
    )


def _gateway_openai_base() -> str:
    """The OpenAI-compatible gateway base URL to use for clients that cannot resolve it themselves.

    An explicit `OPENAI_BASE_URL` always wins — it is the more specific setting, and the
    documented precedence table treats a provider base URL as overriding the gateway switch.
    """
    explicit = os.environ.get("OPENAI_BASE_URL", "")
    if GATEWAY_HOST in explicit:
        return explicit
    setting = _gateway_setting()
    if setting.startswith("http"):                     # regional gateway, e.g. eu.gateway…
        return f"{setting.rstrip('/')}/openai/v1"
    return DEFAULT_GATEWAY_OPENAI_BASE


# Canonical name for the gateway credential, matching the name the LangSmith docs use. It is deliberately NOT `LANGSMITH_API_KEY`: `langgraph deploy` strips that
# reserved variable during upload, so a deployed agent would lose its gateway credential.
# `LC_GATEWAY_KEY` is accepted as a legacy alias, and a PAT in `LANGSMITH_API_KEY` is the last
# resort for personal-workspace use.
GATEWAY_KEY_VARS = (
    "LANGSMITH_GATEWAY_API_KEY",   # canonical — the name docs.langchain.com documents
    "LANGSMITH_API_KEY_GATEWAY",   # accepted alias — a spelling in circulation elsewhere
    "LC_GATEWAY_KEY",              # accepted alias — legacy
    "LANGSMITH_API_KEY",           # last resort: a PAT. Works, but `langgraph deploy` strips it.
)


def _gateway_key() -> str | None:
    """Return the gateway bearer token, preferring the officially documented env var.

    **Both spellings are accepted deliberately, and the order matters.** The official docs name the
    override `LANGSMITH_GATEWAY_API_KEY`; another spelling in circulation is
    `LANGSMITH_API_KEY_GATEWAY` — the same four words, two of them
    transposed. Accepting only one meant a participant who followed
    https://docs.langchain.com/langsmith/llm-gateway-quickstart set a variable this module never
    read, and got a `ValueError` telling them to set a key they had already set. That is the worst
    kind of setup bug: the instructions and the error message are each individually correct.

    Order: the documented name, then the two internal aliases, then `LANGSMITH_API_KEY` — a PAT,
    which works for a personal workspace but is stripped by `langgraph deploy`, so an agent that
    relies on it deploys cleanly and then fails at runtime.
    """
    return next((os.environ[v] for v in GATEWAY_KEY_VARS if os.environ.get(v)), None)


def get_model(model: str | None = None):
    """Return a chat model. Pass a `provider:model` string (e.g. HEAVY_MODEL) to override.

    When the gateway is configured — `LANGSMITH_GATEWAY`, or a BASE_URL pointing at
    gateway.smith.langchain.com — the client is created with the gateway key as its bearer token.
    The provider key stays in LangSmith's Provider Secrets and never leaves the gateway; no
    `OPENAI_API_KEY` is needed, or read, on this path.

    No `base_url` is passed here on purpose. LangChain's chat models resolve the gateway endpoint
    themselves from `LANGSMITH_GATEWAY` / `OPENAI_BASE_URL`, and letting them do it keeps regional
    gateways and per-provider overrides working exactly as the documented precedence table describes.
    `get_embeddings()` has to be explicit instead, because `OpenAIEmbeddings` does no such
    resolution — see `_using_gateway`.

    Credential: set `LANGSMITH_GATEWAY_API_KEY` (workspace-scoped, `gateway:invoke` permission).
    See `_gateway_key` for the aliases that are also accepted and why.
    """
    model_id = model or DEFAULT_MODEL

    if _using_gateway():
        # Gateway path: use the gateway key as the bearer token.
        # The provider key is stored in LangSmith Provider Secrets.
        gateway_key = _gateway_key()
        if not gateway_key:
            raise ValueError(_MISSING_GATEWAY_KEY)
        return init_chat_model(
            model_id,
            api_key=gateway_key,
            max_retries=_max_retries(),
            timeout=_timeout(),
        )

    # Direct provider path: use the provider API key from the environment.
    return init_chat_model(model_id, max_retries=_max_retries(), timeout=_timeout())


def get_embeddings(model: str = "text-embedding-3-small"):
    """Return an embeddings client routed the same way as the chat model.

    Every RAG lab needs this: `OpenAIEmbeddings` reads `OPENAI_API_KEY`, which is the *provider*
    credential, so on the gateway path the labs used to overwrite that variable with the gateway key
    in each setup cell. Passing `api_key=` explicitly instead keeps the gateway credential out of any
    provider-specific variable — which is the pattern to prefer — and means a lab
    cell no longer has to know anything about which path is active.
    """
    from langchain_openai import OpenAIEmbeddings

    if _using_gateway():
        gateway_key = _gateway_key()
        if not gateway_key:
            raise ValueError(_MISSING_GATEWAY_KEY)
        # Resolve the base URL rather than reading OPENAI_BASE_URL directly. On the
        # `LANGSMITH_GATEWAY=true` path that variable is unset — LangChain's chat models resolve the
        # gateway internally, but `OpenAIEmbeddings` does not, so passing `base_url=None` here sent
        # every RAG lab to api.openai.com with a LangSmith key. See `_gateway_openai_base`.
        return OpenAIEmbeddings(model=model, api_key=gateway_key,
                                base_url=_gateway_openai_base(),
                                timeout=_timeout(), max_retries=_max_retries())
    return OpenAIEmbeddings(model=model, timeout=_timeout(), max_retries=_max_retries())


def get_judge_model():
    """Return the grading model — for LLM-as-judge evaluators and `RubricMiddleware`.

    Separate from `get_model()` on purpose: the thing being graded and the thing doing the grading
    should not share a capability ceiling, or the judge inherits the agent's blind spots. See the
    tier table at the top of this module for the measurements behind the split.
    """
    return get_model(JUDGE_MODEL)

# ---------------------------------------------------------------------------
# Participant scoping — stops ~15 people overwriting each other's LangSmith objects
# ---------------------------------------------------------------------------
# Datasets, annotation queues, prompts and experiment names are workspace-global. In a shared
# workshop workspace every participant seeding "vendor-due-diligence-eval" writes to the SAME
# dataset, and Lab 04's comparison becomes meaningless. Set PARTICIPANT in .env (initials are
# plenty) and every artifact name is suffixed with it.
#
# Tracing PROJECT is scoped the same way, so each person's traces land somewhere they can find.
# Leave PARTICIPANT unset for a solo run and nothing is renamed.
PARTICIPANT = os.getenv("PARTICIPANT", "").strip()


def scoped(name: str) -> str:
    """Suffix a LangSmith artifact name with the participant tag, when one is set.

    >>> scoped("vendor-due-diligence-eval")   # PARTICIPANT=tr
    'vendor-due-diligence-eval-tr'
    """
    if not PARTICIPANT:
        return name
    tag = "".join(c for c in PARTICIPANT.lower() if c.isalnum() or c == "-")
    if not tag:
        return name
    # Idempotent: this is called on values that may already carry the tag (LANGSMITH_PROJECT is
    # rewritten in-place below, so a second call would otherwise produce "…-tr-tr").
    return name if name.endswith(f"-{tag}") else f"{name}-{tag}"


def scoped_project() -> str:
    """The tracing project for this participant, e.g. 'langchain-adlc-workshop-tr'."""
    return scoped(os.getenv("LANGSMITH_PROJECT", "langchain-adlc-workshop"))


# Apply it. `LANGSMITH_PROJECT` is read by the tracer at trace time, so setting it here — at import
# of the module every lab already imports — routes this participant's traces to their own project
# without touching 16 setup cells.
#
# This is a deliberate import-time side effect on the environment, which is normally worth avoiding.
# It is justified because the alternative is worse: 15 people interleaving traces in one project,
# where nobody can find their own run and Day 3's "production" statistics are computed over everyone
# else's lab traffic. It is a no-op unless PARTICIPANT is set.
if PARTICIPANT:
    os.environ["LANGSMITH_PROJECT"] = scoped_project()
