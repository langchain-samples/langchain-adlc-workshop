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
# # 05 · Governance — LLM Gateway, LiteLLM, Security Context, Deployment Paths, Fleet & Engine
#
# **Workshop:** LangChain ADLC Workshop · **Day 3** · **ADLC stage:** Govern + Improve
#
# > **Loop Engineering focus: Event-driven loop + Hill-climbing loop** — the agent loop (Day 1)
# > and the verification loop (Day 2) are built and evaluated. This demo covers the loops that
# > *operate* them in production: a governed model-access layer, user-scoped data access,
# > deployment paths for sensitive environments, event-driven operation, and a continuous
# > trace-driven improvement loop.
#
# > **Demo walkthrough** (presenter-led, no participant code) · ~45 min
#
# ```mermaid
# graph TD
#     subgraph GovernedAgents["The two workshop agents"]
#         TA[Day 1 · Ticket resolution agent]
#         DD[Day 2 · Vendor due diligence agent]
#     end
#
#     TA --> GW[1 · LLM Gateway<br/>managed proxy · Provider Secrets]
#     DD --> GW
#     GW -->|policies: spend · rate limits · fallback · redaction| P[Model providers<br/>OpenAI · Anthropic · …]
#     GW -.self-hosted alternative.-> LL[2 · LiteLLM proxy<br/>OpenAI-compatible · config-as-code]
#     LL --> P
#
#     TA --> SC[3 · User security context<br/>identity → tool calls → row/doc/system permissions]
#     DD --> SC
#
#     TA --> DP[4 · Deployment paths<br/>SaaS · BYOC · hybrid · self-hosted · air-gapped]
#     DD --> DP
#
#     DP --> FL[5 · Fleet<br/>channels · schedules · triggers · automations]
#     FL --> TR[Production traces]
#     GW --> TR
#     TR --> EN[6 · Engine<br/>failure-mode discovery · recommended fixes]
#     EN -->|fix → evaluator → dataset → redeploy| DP
# ```
#
# Days 1–2 built two agents and proved they work against a dataset. Day 3 morning deployed them and
# attached online evals. This session answers the question every enterprise asks next: **how do we
# govern this?** Six building blocks, each mapped to the ticket resolution agent and the vendor due
# diligence agent as the systems being governed:
#
# | # | Block | Question it answers |
# |---|---|---|
# | 1 | **LLM Gateway** | Who holds the provider keys, and what policies apply to every model call? |
# | 2 | **LiteLLM** | What if we need the same gateway pattern, self-hosted on our own infrastructure? |
# | 3 | **User security context** | How does the agent know what *this user* is allowed to see and do? |
# | 4 | **Deployment paths** | Where can the platform run — SaaS, our cloud, our data center, air-gapped? |
# | 5 | **Fleet** | How do production events and schedules invoke agents without a human pressing run? |
# | 6 | **Engine** | How do production traces become diagnosed issues, fixes, and regression evaluators? |
#
# > 🧭 **This is a demo, not hands-on.** Code cells below are configuration snippets and small,
# > runnable reference patterns the instructor shows — no participant exercises. Where a control is
# > configured in the LangSmith UI (gateway policies, Fleet channels, Engine), the markdown says
# > exactly where to click. All tickets, users, and vendors are **synthetic/fictional** fixtures
# > created for this workshop.
#
# > 📚 **References:** [LLM Gateway](https://docs.langchain.com/langsmith/llm-gateway) ·
# > [LiteLLM](https://docs.litellm.ai/docs/) ·
# > [Deployment environments](https://docs.langchain.com/langsmith/deployment) ·
# > [Fleet](https://docs.langchain.com/langsmith/fleet) ·
# > [Engine](https://docs.langchain.com/langsmith/engine) ·
# > workshop-local: the README (Model access section) (full gateway setup for these machines)


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
# Same setup cell as the other labs — loads `.env`, prints the LangSmith workspace and which
# model-access path is active. **The demo works with or without a configured gateway**: when
# `LANGSMITH_GATEWAY` is set (or a provider base URL points at `gateway.smith.langchain.com`) the
# model calls below route through the LLM Gateway; otherwise they use the direct provider key and the
# gateway cells become talk-throughs. No new dependencies.

# %%
import json
import os
import sys
from pathlib import Path

from datetime import datetime, timezone

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
DATA = WORKSHOP / "day1" / "data"

from day1.src.models import get_model

# Ask the model layer rather than re-deriving it. This cell used to sniff the two BASE_URL vars
# itself, which quietly disagreed with `models.py` once `LANGSMITH_GATEWAY=true` became a supported
# way to switch the gateway on — the labs would route through the gateway while this banner said
# "direct provider key".
from day1.src.models import _using_gateway

GATEWAY_CONFIGURED = _using_gateway()

print("workspace:", os.getenv("LANGSMITH_WORKSPACE_ID"), "| tracing:", os.getenv("LANGSMITH_TRACING"))
print("model access path:", "LLM Gateway ✅" if GATEWAY_CONFIGURED else "direct provider key (gateway cells are talk-throughs)")

# %% [markdown]
# ## 1. LLM Gateway — one governed front door for every model call
#
# 📖 [LLM Gateway](https://docs.langchain.com/langsmith/llm-gateway)
#
# The **LangSmith LLM Gateway** is a managed proxy that sits between your agents and the model
# providers. Neither the ticket agent nor the due diligence agent ever holds a provider key —
# they authenticate to the gateway with a LangSmith key, and the gateway forwards the call using
# provider credentials stored centrally in **Provider Secrets**.
#
# ```
# ┌───────────────────────────────┐
# │  Ticket agent · DD agent      │   no provider keys on the client —
# │  (init_chat_model, unchanged) │   only a LangSmith key
# └──────────────┬────────────────┘
#                │  Authorization: Bearer <LANGSMITH_GATEWAY_API_KEY>
#                ▼
# ┌───────────────────────────────┐     ┌──────────────────────────┐
# │  gateway.smith.langchain.com  │────▶│  Provider Secrets         │  real OPENAI / ANTHROPIC
# │                               │     │  (stored in LangSmith,    │  keys live here — never on
# │  policies on every call:      │     │   workspace-scoped)       │  a laptop or in a container
# │  · spend limits               │     └──────────────────────────┘
# │  · rate limits                │                │
# │  · model fallback             │                ▼
# │  · PII / secrets redaction    │     ┌──────────────────────────┐
# └──────────────┬────────────────┘     │  OpenAI · Anthropic · …  │
#                │                      └──────────────────────────┘
#                ▼
#      LangSmith trace for every call (gateway project + your project)
# ```
#
# **Why a gateway is the enterprise answer to "who has the API key?":**
#
# | Concern | Without a gateway | With the LLM Gateway |
# |---|---|---|
# | **Provider keys** | Copied into env vars on every laptop / pod | Stored once in Provider Secrets; clients hold only a LangSmith key |
# | **Revoking access** | Find every copy of the provider key | Revoke the LangSmith key — provider key never distributed |
# | **Cost control** | Provider dashboard, after the fact | Spend policies enforced *at request time*, per workspace/user/key |
# | **Rate limiting** | Whatever the provider enforces | Workspace-level policies in front of provider limits |
# | **Outages / limits** | Agent crashes on 429 / 5xx | Model fallback — route to a backup model automatically |
# | **Sensitive data** | Up to each app to redact | PII / secrets redaction policies applied to every request (and to the trace) |
# | **Audit** | Provider logs | Every call auto-traced to LangSmith with policy outcomes in metadata |
#
# > 📌 **Workshop machines are already configured** — see the README (Model access section). The
# > `day1/src/models.py` helper every lab uses detects the gateway env vars and passes the gateway
# > key as the bearer token; the setup paths (workspace PAT vs a service key in
# > `LANGSMITH_GATEWAY_API_KEY`) are documented there. The cell below demonstrates the detection live.

# %%
# DEMO: which path will this kernel use? The same check day1/src/models.py performs.
from day1.src.models import _gateway_key, _using_gateway

print("gateway configured:", _using_gateway())
if _using_gateway():
    key = _gateway_key() or ""
    print(f"gateway key: {key[:9]}…{key[-4:]}  (LANGSMITH_GATEWAY_API_KEY service key, or a PAT)")
    print("provider key location: LangSmith Provider Secrets (Settings → Integrations)")
else:
    print("direct path: OPENAI_API_KEY / ANTHROPIC_API_KEY read from the environment")

# %% [markdown]
# ### 1.1 Switching an agent to the gateway = changing env vars, not code
#
# This is the crucial demo point: **the agent code does not change**. The Day 1 ticket agent and
# the Day 2 DD agent both call `get_model()`; whether that returns a direct-provider client or a
# gateway-routed client is decided entirely by environment variables:
#
# ```bash
# # Direct provider access (what a participant laptop might use)
# OPENAI_API_KEY=sk-...
#
# # LLM Gateway access (what production uses)
# LANGSMITH_GATEWAY=true                    # the documented switch; or a regional gateway URL
# LANGSMITH_GATEWAY_API_KEY=lsv2_sk_...     # service key — the bearer token for the gateway
# # …and NO provider key at all. That absence is the security property.
# ```
#
# > ⚠️ **Do not set `OPENAI_API_KEY=$LANGSMITH_GATEWAY_API_KEY`.** It works, which is exactly why it
# > spreads, and it is still the wrong shape: it puts a LangSmith credential into a variable named
# > for a provider, so every other library in the process that reads `OPENAI_API_KEY` now holds a key
# > that only functions against the gateway — and a key with a completely different blast radius from
# > the one its name implies. `get_model()` and `get_embeddings()` pass `api_key=` explicitly instead,
# > so the gateway credential never occupies a provider variable.
#
# To route **one** provider through the gateway and leave the others alone, set that provider's base
# URL — `OPENAI_BASE_URL=https://gateway.smith.langchain.com/openai`. A provider base URL takes
# precedence over the switch, and the provider namespaces (`/openai`, `/anthropic`, `/gemini`) mean
# one URL pattern covers every configured provider: **switching providers is switching a model ID**,
# not re-credentialing.
#
# > **One gap worth knowing before you rely on the switch.** `LANGSMITH_GATEWAY` is honoured by
# > LangChain's *chat* models, which resolve the gateway endpoint themselves. `OpenAIEmbeddings` is
# > not on that list and does no gateway resolution — construct one with no provider key present and
# > it raises `OpenAIError: Missing credentials` before issuing a request. So a half-migrated app
# > shows a working agent and dead retrieval. This is why every RAG lab here calls `get_embeddings()`
# > rather than building the client directly.
#
# The cell below makes one call through whichever path is active and shows the client config.

# %%
model = get_model()

print("client class :", type(model).__name__)
print("model        :", model.model)
print("base URL     :", getattr(getattr(model, "client", None), "base_url", "(default provider endpoint)"))
print("max_retries  :", model.max_retries)

# One cheap call through the active path. When the gateway is active, this call lands in the
# workspace's `gateway` tracing project as well as the lab project — demo: open both in LangSmith.
reply = model.invoke("Reply with exactly: gateway check ok")
print("reply        :", reply.content)

# %% [markdown]
# ### 1.2 Policies — the governance payload of the gateway
#
# Policies are workspace-level rules enforced on **every** call that passes through the gateway.
# They are configured in the UI — **LangSmith → Settings → Gateway → LLM Gateway** — and require
# `organization:manage` permission. Nothing in the agent code opts in or out; that's the point.
#
# | Policy | What it does | Acme-flavored example |
# |---|---|---|
# | **Spend limits** | Cap $ per workspace / user / API key per period; requests over budget are rejected | Cap the workshop workspace at $25/day so a runaway tool loop can't burn budget |
# | **Rate limits** | Throttle requests (and/or tokens) per key or workspace | 60 req/min per user keeps one script from starving the shared quota |
# | **Model fallback** | If the primary model errors / is rate-limited, retry against a configured backup | `gpt-4.1` → `gpt-4.1-mini` during a provider incident; the DD agent still answers |
# | **Model allow-list** | Restrict which model IDs the workspace may call | Only approved models for procurement data; everything else is rejected |
# | **PII redaction** | Detect emails, phone numbers, SSNs, credit cards… and redact before the provider sees them | A requester's email in a ticket is redacted before leaving the trust boundary |
# | **Secrets redaction** | Detect API keys, tokens, passwords in the request and redact them | A user pastes a connection string into a ticket; it never reaches the provider |
#
# Two properties make redaction special:
# - **Redacted content is also redacted in the LangSmith trace** — sensitive data doesn't persist
#   in observability data either.
# - The gateway can **restore original values in the provider's response**, so redaction doesn't
#   break agent functionality (the model reasons over placeholders; the user sees real values).
#
# When a policy fires (spend cap hit, PII redacted, secret caught), the event is recorded **as
# metadata on the gateway trace** — policy IDs matched / passed / violated — and policy violations
# surface as **issues in LangSmith Engine** (§6). Governance isn't just enforcement; it's
# observable enforcement.
#
# #### What a blocked call actually looks like
#
# 📖 [Spend policies](https://docs.langchain.com/langsmith/llm-gateway-spend-policies)
#
# Worth seeing concretely, because "requests over budget are rejected" understates how it lands in
# your code. A blocked request comes back as **HTTP 402**, and the response header names the reason:
#
# ```http
# HTTP/2 402
# x-langsmith-gateway-metadata: {"outcome":"blocked","reason":"spend_limit","provider":"openai",
#                                "scopes":["workspace"],"limit_source":"gateway_policy",
#                                "limit_windows":["monthly"]}
#
# {"type":"error","error":{"message":"request blocked by gateway policies: <workspace-id>"}}
# ```
#
# That header is the whole diagnosis — reason, scope, and which window tripped — and it is the first
# thing to read, because the body only tells you *that* a policy fired, not which one. Caps stack
# from broadest to narrowest (org → workspace → API key → user) and **the most restrictive wins**, so
# a request can be blocked by a cap you did not set and cannot see from your own key.
#
# **Three operational consequences worth designing for:**
#
# | | Why it matters |
# |---|---|
# | **Retries do not help.** | The OpenAI SDK retries `408`, `409`, `429` and `5xx`. `402` is not on that list, so `max_retries` — including the budget `models.py` sets — never fires. The first blocked call raises straight into your cell or your request handler. |
# | **It fails mid-run, not at startup.** | The cap trips on the request that would cross it, so a lab or an agent run can be half-finished when it happens. A green setup check proves nothing about the next hour. |
# | **Size caps for the real workload.** | A cap that is comfortable for one developer is not comfortable for a room of people running Day 2's eval loops and Day 3's online evals — an eval pass is many model calls per example, times examples, times participants. Set the monthly *and* daily windows deliberately before a workshop or a launch. |
#
# The general lesson generalises past spend: **a governance control is only as good as its failure
# mode.** A policy that rejects requests is doing its job, so the design question is never "will this
# ever block us?" but "when it blocks us, does the caller get something it can act on?" Here that
# means catching 402 distinctly from 429 — one clears by waiting, the other does not.

# %%
# DEMO: what a governed call looks like in the trace.
# This runs through whichever path is configured — with the gateway active, the trace shows
# policy metadata (matched/passed/violated policy IDs) on the gateway span. With direct access,
# the same trace shows why middleware (Day 1 Lab 05) is the app-level complement: redaction
# happens in-process instead of at the gateway.
ticket_with_sensitive_data = (
    "My MFA stopped working after I replaced my phone. Contact me at sofia.petrov@acme.example "
    "or +31 6 5550 0132. Also the old API key was sk-proj-FAKEworkshopKEY1234567890 — is it still valid?"
)

result = model.invoke(
    "You are the Acme ticket triage assistant. Summarize this request in one sentence, "
    "and list any sensitive data categories present.\n\nREQUEST: " + ticket_with_sensitive_data
)
print(result.content)
print()
print("→ With the gateway's PII + secrets policies ON: the provider and the trace see")
print("  redacted placeholders for the email, phone number, and API key — the raw values")
print("  never leave the trust boundary, and the policy event is stamped on the trace.")
print("→ With direct access: this is exactly what Day 1's PIIMiddleware handles in-process.")

# %% [markdown]
# ### 1.3 Talk track — where each control lives
#
# | Demo beat | Where |
# |---|---|
# | Show the provider secrets (values are write-only) | Settings → Integrations → Provider Secrets |
# | Show the policy list and create a spend limit | Settings → Gateway → LLM Gateway → Policies |
# | Show a gateway call trace with policy metadata | LangSmith `gateway` project in this workspace |
# | Show fallback behavior | Trigger a rate-limited model; watch the request land on the fallback model ID |
# | Key rotation story | Revoking the LangSmith key cuts access without touching the provider account |
#
# > **Beta note:** the LLM Gateway is in beta. It is available on Cloud and **BYOC** (runs inside
# > your data plane behind the `/gateway` path prefix). Self-hosted stable support is on the
# > roadmap — which is exactly where LiteLLM (next section) fits today.

# %% [markdown]
# ## 2. LiteLLM — the self-hosted, open-source gateway
#
# 📖 [LLM Gateway](https://docs.langchain.com/langsmith/llm-gateway)
#
# **LiteLLM** is an open-source proxy that provides the same *shape* of capability for teams that
# need the gateway on their own infrastructure: one **OpenAI-compatible endpoint**, multi-provider
# routing, cost tracking, budgets, fallbacks, and request logging — all defined in a config file
# you version in your own repo.
#
# For an organization like Acme, this matters for two reasons:
# 1. **Sovereignty / air-gap** — the proxy runs inside your network; no SaaS dependency.
# 2. **Provider breadth** — 100+ providers behind one `/v1/chat/completions`, including
#    self-hosted models (vLLM, Ollama) — e.g. an on-prem model for highly sensitive workloads.
#
# ```yaml
# # litellm_config.yaml — version-controlled, deployed with the proxy
# model_list:
#   - model_name: ticket-agent-standard              # alias the agents call
#     litellm_params:
#       model: openai/gpt-4.1-mini
#       api_key: os.environ/OPENAI_API_KEY
#       rpm: 60                              # per-deployment rate limit
#   - model_name: ticket-agent-standard              # same alias, second provider → load-balance/fallback pool
#     litellm_params:
#       model: anthropic/claude-haiku-4-5
#       api_key: os.environ/ANTHROPIC_API_KEY
#   - model_name: ticket-agent-strong                # heavier tier for the DD agent's final report
#     litellm_params:
#       model: openai/gpt-4.1
#       api_key: os.environ/OPENAI_API_KEY
#
# router_settings:
#   fallbacks: [{"ticket-agent-strong": ["ticket-agent-standard"]}]   # strong model down → degrade gracefully
#   num_retries: 3
#
# litellm_settings:
#   success_callback: ["langsmith"]          # log every call to LangSmith for observability
#
# general_settings:
#   master_key: os.environ/LITELLM_MASTER_KEY
# ```
#
# ```bash
# # Run the proxy (Docker in production; pip for the demo)
# pip install "litellm[proxy]"
# litellm --config litellm_config.yaml --port 4000
# ```
#
# And the client-side change is — again — **only a base URL**:
#
# ```python
# from langchain.chat_models import init_chat_model
#
# model = init_chat_model(
#     "openai:ticket-agent-standard",                       # the LiteLLM alias, not a provider model ID
#     base_url="http://litellm.internal:4000/v1",   # the self-hosted proxy
#     api_key=os.environ["LITELLM_VIRTUAL_KEY"],    # per-team virtual key with its own budget
# )
# ```

# %%
# DEMO: the client-side shape, constructed but NOT invoked (no LiteLLM server is running in this
# workshop — that's the point of the talk track: the agent code is identical, only the endpoint,
# key, and model alias change).
def build_litellm_model(alias: str = "ticket-agent-standard"):
    """What `get_model()` would look like against a self-hosted LiteLLM proxy.
    Not executed in the demo — shown to prove the call shape is unchanged."""
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        f"openai:{alias}",
        base_url=os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1"),
        api_key=os.environ.get("LITELLM_VIRTUAL_KEY", "sk-demo-not-real"),
        max_retries=3,
    )


print("LiteLLM client shape (not invoked):")
print("  endpoint : http://<litellm-host>:4000/v1   ← OpenAI-compatible")
print("  model    : 'ticket-agent-standard'                 ← alias resolved by the proxy config")
print("  auth     : per-team virtual key with its own budget / rate limit")
print("  logging  : success_callback='langsmith'    ← same traces as the rest of the workshop")

# %% [markdown]
# ### 2.1 LLM Gateway vs LiteLLM — the honest comparison
#
# | Capability | LangSmith LLM Gateway | LiteLLM proxy |
# |---|---|---|
# | Hosting | Managed SaaS (Cloud) or inside your BYOC data plane | Self-hosted anywhere — your VM, VPC, or air-gapped network |
# | Provider keys | LangSmith **Provider Secrets**, managed in the UI | Your secret store; env vars / config file / KMS you operate |
# | Endpoint | OpenAI-compatible (`/v1/chat/completions`), Anthropic Messages, Responses | OpenAI-compatible (`/v1/chat/completions` + pass-through routes) |
# | Spend / rate limits | Workspace policies in the UI, enforced at request time | Per-key / per-team / per-tag budgets in config, enforced at request time |
# | Model fallback | Configured policy | `router_settings.fallbacks` in config |
# | PII / secrets redaction | Built-in data-protection policies (Enterprise), values restorable in responses | Via guardrail integrations / custom callbacks — you assemble it |
# | Logging / audit | Automatic LangSmith traces with policy metadata | Callbacks (LangSmith, Langfuse, S3, …) — you configure it |
# | Access model | LangSmith API keys + roles (`gateway:invoke`) | Virtual keys issued by your admins |
# | Best fit | Teams already on LangSmith wanting zero-ops governance | Sovereign / air-gapped environments, custom infra, self-hosted models |
#
# **They are not either/or.** A common enterprise shape: LiteLLM inside the secured perimeter for
# on-prem / self-hosted models, LangSmith LLM Gateway for sanctioned cloud providers — both
# OpenAI-compatible, so agents target either with a base-URL change, and both can trace to the
# same LangSmith workspace.

# %% [markdown]
# ## 3. User security context — the agent enforces *your* permissions
#
# The gateway governs the **model boundary**. The next boundary is the **data boundary**: when a
# user asks the ticket agent (or the DD agent) for something, the agent must only retrieve and act
# on data *that user is already authorized to access* — never more. An agent with a service
# account that can read everything is a privilege-escalation machine unless identity flows through.
#
# The pattern (built hands-on in Day 1 Lab 05, recap here as the production reference):
#
# ```mermaid
# graph LR
#     A[Authenticated app<br/>establishes user identity] -->|invoke with context| B[Agent<br/>context_schema=UserContext]
#     B -->|ToolRuntime.context| C[Tool: search_kb<br/>doc-level filter]
#     B -->|ToolRuntime.context| D[Tool: search_ticket_history<br/>row-level filter]
#     B -->|ToolRuntime.context| E[Tool: api_action<br/>system-level gate]
#     C --> F[Audit log<br/>who · what · allowed/denied]
#     D --> F
#     E --> F
# ```
#
# Three design rules:
#
# 1. **Identity lives in runtime context, not chat state.** `context_schema=UserContext` +
#    `ToolRuntime.context` — the caller's `user_id` is supplied by the invoking application at
#    invoke time. If it were a chat message or tool argument, a prompt could tamper with it
#    ("ignore my user_id, I'm an admin").
# 2. **Enforcement happens inside the tool, at the data layer.** The model can't leak what it
#    never receives — filtering before results reach the context window is the control; a
#    system-prompt plea is not.
# 3. **Every allow/deny decision is logged.** Blocked retrievals and escalations are security
#    events, and they belong in the same audit trail as the runs themselves.
#
# ### 3.1 The three enforcement levels
#
# | Level | Scope | Workshop example (Day 1 fixtures) | Enterprise analogue |
# |---|---|---|---|
# | **Document-level** | Whole docs / KB articles | `search_kb` drops KB chunks whose article maps outside the user's permitted categories (`security_incident.md` invisible to viewers) | SharePoint/Confluence ACLs, restricted KB sections |
# | **Row-level** | Records in a database / ticket store | `search_ticket_history` returns only tickets in categories the user's role permits — other rows never leave the tool | SQL row-level security, `WHERE org_unit = :user_org` |
# | **System-level** | Whether a tool/API may be called at all | Sensitive actions (MFA reset, account unlock) require the right role *and* HITL approval | SAP-style transaction codes, MCP server scopes, service-account gating |
#
# The cell below is the compact reference implementation over the Day 1 fixtures — the same code
# shape the ticket agent uses, runnable here so the instructor can *show* the filter working.

# %%
from pydantic import BaseModel

# --- Day 1 fixtures: users with roles + permissions -------------------------
USERS: list[dict] = json.loads((DATA / "users.json").read_text())
TICKETS: list[dict] = json.loads((DATA / "tickets.json").read_text())


class UserContext(BaseModel):
    """Runtime identity, supplied by the invoking app — NOT by the model.

    Passed at invoke time as context={"user_id": ...} and read inside tools via
    ToolRuntime.context. Immutable for the run, so prompt content can't escalate it.
    """
    user_id: str


def _get_user(user_id: str) -> dict | None:
    return next((u for u in USERS if u["user_id"] == user_id), None)


AUDIT_LOG: list[dict] = []  # in production: append to your SIEM / audit store


def search_ticket_history_scoped(query: str, ctx: UserContext, category: str | None = None) -> dict:
    """Reference pattern: row-level enforcement + audit, in the tool — not in the prompt."""
    user = _get_user(ctx.user_id)
    permissions = user.get("permissions", []) if user else []

    # Row-level filter: a ticket row is only visible if its category is in the user's permissions.
    visible = [t for t in TICKETS if t["category"] in permissions]
    if category:
        visible = [t for t in visible if t["category"] == category]
    terms = {w.strip(".,?!") for w in query.lower().split() if len(w) > 2}
    hits = [t for t in visible
            if any(term in f"{t['subject']} {t['description']}".lower() for term in terms)][:3]

    # Audit: record the access decision. The decision must reflect what the *permission filter* did,
    # not how many rows happened to match the query — otherwise a run where 19 of 20 rows were
    # suppressed logs as "allow", and the audit trail cannot answer "was anything withheld from this
    # user?", which is the only question it exists to answer.
    suppressed = len([t for t in TICKETS if t["category"] not in permissions])
    if user is None:
        decision = "deny_unknown_user"
    elif not visible:
        decision = "deny_empty_scope"          # nothing in this user's scope at all
    elif suppressed:
        decision = "allow_scoped"              # partial: some rows withheld by permissions
    else:
        decision = "allow_full"                # user may see the whole table

    AUDIT_LOG.append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "tool": "search_ticket_history",
        "user_id": ctx.user_id,
        "role": user["role"] if user else "unknown",
        "query": query,
        "rows_total": len(TICKETS),
        "rows_visible": len(visible),
        "rows_suppressed": suppressed,
        "rows_returned": len(hits),
        "matched": bool(hits),
        "decision": decision,
    })
    return {"user": user["name"] if user else ctx.user_id, "hits": hits}


# %% [markdown]
# **Same query, two users** — USR-004 is a `viewer` (knowledge-only), USR-001 is an `admin`
# (all categories). The row-level filter changes what the model would ever see:

# %%
for uid in ("USR-004", "USR-001"):
    out = search_ticket_history_scoped("vpn connection dropping", UserContext(user_id=uid))
    user = _get_user(uid)
    entry = AUDIT_LOG[-1]
    print(f"{uid} · {user['name']} ({user['role']}, permissions={user['permissions']})")
    print(f"  rows returned: {len(out['hits'])} · visible: {entry['rows_visible']}/{entry['rows_total']}"
          f" · suppressed by permissions: {entry['rows_suppressed']}")
    print(f"  audit decision: {entry['decision']}")
    for t in out["hits"]:
        print(f"    {t['ticket_id']} [{t['category']}] {t['subject'][:60]}")
    print()

print("Audit trail (every decision logged — this is the auditable artifact):")
for entry in AUDIT_LOG:
    print(f"  {entry['user_id']} · {entry['decision']:<18} visible {entry['rows_visible']}/{entry['rows_total']} rows")

# %% [markdown]
# ### 3.2 How it composes with the two workshop agents
#
# - **Ticket resolution agent (Day 1):** `search_kb` (document-level), `search_ticket_history`
#   (row-level), and the action tool (system-level, plus HITL for sensitive actions) all read
#   `ToolRuntime.context.user_id`. A viewer asking about a security incident gets *no results* —
#   the model then honestly reports "no authorized information found" instead of leaking.
# - **Vendor due diligence agent (Day 2):** the same pattern scopes vendor records, internal
#   pricing, and screening detail to authorised procurement roles, and the escalation rules in
#   `AGENTS.md` route sensitive cases to human review. Identity + escalation + audit log =
#   the evidence trail a procurement audit asks for.
#
# | Auditability pattern | Where it lands |
# |---|---|
# | Access decisions (allow/deny per tool call) | Audit log / SIEM (pattern above) |
# | Full run traces — inputs, outputs, tool calls | LangSmith traces (already on for both agents) |
# | Run metadata: `user_id`, sensitivity, `requires_hitl` | `tag_run_metadata()` from Day 1 Lab 05 → trace metadata |
# | HITL approvals / rejections | Interrupt payloads in the trace + reviewer identity |
# | Escalations and blocked retrievals | Online-eval automations → annotation queue (Day 3 Lab 03) |
#
# > 🔑 **Key message:** the LLM Gateway centralizes *model* governance; user security context
# > decentralizes *data* governance into every tool. You need both, and they're independent —
# > swap the gateway for LiteLLM and the security-context pattern is unchanged.

# %% [markdown]
# ## 4. Deployment paths — where the platform itself runs
#
# 📖 [Deployments / CLI](https://docs.langchain.com/langsmith/cli)
#
# Everything so far assumes the workshop's LangSmith **cloud** workspace. Acme's production path
# will likely be different. All options run the same Agent Server runtime and the same agent code
# — what changes is **where the control plane and data plane live**:
#
# | Path | Control plane | Data plane (agents + data) | Traces | When it's the answer |
# |---|---|---|---|---|
# | **SaaS / Cloud** | LangChain | LangChain (AWS/GCP) | LangSmith Cloud | Fastest start; workshop default; non-sensitive workloads |
# | **BYOC** | LangChain | **Your cloud** VPC | Your data plane + LangSmith | Data must stay in your AWS/GCP account; keep managed ops |
# | **Hybrid** | LangChain | Your Kubernetes | Cloud or self-hosted LangSmith | Agents next to internal systems; managed control plane OK |
# | **Self-hosted** | Your Kubernetes | Your Kubernetes | Self-hosted LangSmith | Full data residency, regulated environments |
# | **Air-gapped** | Your infra, no egress | Your infra, no egress | Self-hosted LangSmith, no egress | Regulated / disconnected networks |
# | **Standalone server** | None (you operate it) | Docker / K8s, your Postgres + Redis | Optional, Cloud or self-hosted | Just the agent runtime inside an existing platform |
#
# **How the workshop maps forward:** the `langgraph.json` + graphs deployed this morning
# (Lab 01) are exactly what a standalone/self-hosted Agent Server runs. The moving parts that
# change per path:
#
# | Concern | SaaS | Self-hosted / air-gapped |
# |---|---|---|
# | Model access | LLM Gateway | **LiteLLM** or direct in-VPC endpoints (gateway on roadmap; BYOC today) |
# | Secrets | Provider Secrets in LangSmith | Your vault / KMS / sealed-secrets |
# | Observability | LangSmith Cloud | Self-hosted LangSmith instance behind your auth |
# | LLM models | Provider APIs | Self-hosted models (vLLM/Ollama) via LiteLLM, or approved egress |
# | Upgrades | Managed | You pin and upgrade container versions |
#
# > 💬 **Discussion prompt:** for a ticket agent reading internal KB articles and a DD agent
# > touching procurement data — which path keeps each data class in an acceptable boundary, and
# > does one deployment serve both, or do sensitivity tiers imply separate deployments?

# %% [markdown]
# ## 5. Fleet — agents that start on events, schedules, and triggers
#
# 📖 [Fleet](https://docs.langchain.com/langsmith/fleet/code)
#
# So far every agent run started with a human pressing ▶ in a notebook. **LangSmith Fleet** is the
# event-driven layer: a workspace for creating and operating agents (no-code or exported from
# code) where **channels, schedules, and triggers** start runs, and **automations** route their
# outputs — all traced to the same LangSmith projects you already use.
#
# ```mermaid
# graph LR
#     subgraph Triggers["How a run starts"]
#         C[Channels<br/>Slack · Teams · Gmail]
#         S[Schedules<br/>recurring UTC cron-like]
#         W[Webhooks / API<br/>CI-CD · external systems]
#     end
#     Triggers --> A[Fleet agent<br/>memory · tools · skills · approvals]
#     A --> T[LangSmith traces]
#     A --> AU[Automations<br/>approvals · escalations · follow-on workflows]
#     T --> E[Online evals + Engine]
# ```
#
# | Concept | What it is | Workshop-flavored example |
# |---|---|---|
# | **Channel** | An external event source that starts a run — Slack mention/DM, Teams message, new Gmail | `#it-support` Slack channel: a message mentioning `@ticket-agent` starts the Day 1 agent; the requester's Slack identity maps to `UserContext.user_id` (§3) |
# | **Schedule** | Recurring time-based trigger (UTC), with an optional per-run prompt | Every Monday 07:00 UTC: "Re-screen all active vendors against the updated sanctions list and flag status changes" — the Day 2 DD agent runs unprompted |
# | **Webhook / API trigger** | Programmatic start from CI/CD or another system; agent packages (`AGENTS.md`, tools, skills) portable as files | The vendor DB emits a webhook when a new vendor is registered → DD intake run starts automatically |
# | **Automation / approval** | Human checkpoints and follow-on routing around runs | A high-risk DD assessment pauses for a procurement officer's approval before the report is distributed |
# | **Agent identity** | Each Fleet agent can be its own Slack bot / identity | `@ticket-agent` and `@vendor-dd` are separate identities with separate tools, memories, and access |
#
# **Governance hooks that make this enterprise-safe:**
# - **Admin controls** — who can create/edit agents, which credentials and connected services each
#   agent may use (workspace vs private agents).
# - **Approvals** — important actions pause for a human, mirroring the HITL patterns from Day 1.
# - **Full observability** — every Fleet run is a native LangSmith trace: the audit trail from §3
#   extends to event-driven runs with no extra work.
# - **Extend with code** — agents can be invoked via API or exported as agent files, so the
#   code-first agents built in this workshop and Fleet-managed agents are not separate worlds.
#
# > **Self-hosted note:** Fleet is available in beta for self-hosted LangSmith — relevant to the
# > §4 deployment discussion. Channels pause/resume per agent, which is the kill-switch story for
# > a misbehaving event source.

# %% [markdown]
# ## 6. Engine — production traces become a fix loop
#
# 📖 [Observability](https://docs.langchain.com/langsmith/observability)
#
# The last block closes the **hill-climbing loop**. **LangSmith Engine** analyzes the production
# traces both agents already emit, clusters recurring failures into **issues**, diagnoses root
# cause, and drives the fix across the same lifecycle artifacts this workshop used manually:
#
# ```mermaid
# graph LR
#     A[Production traces<br/>ticket agent + DD agent] --> B[Engine scans + clusters<br/>recurring failure → Issue]
#     B --> C[Root cause diagnosed<br/>+ relevant traces attached]
#     C --> D[Fix proposed<br/>prompt / tool / routing change]
#     D --> E[Custom evaluator generated<br/>to catch this failure mode]
#     E --> F[Ground-truth examples<br/>from production inputs → dataset]
#     F --> G[Redeploy + re-evaluate<br/>Day 2 experiment loop]
#     G --> H{Issue resurfaced?}
#     H -->|no| I[Closed]
#     H -->|yes| B
# ```
#
# **What Engine detects — the failure-mode taxonomy.** Each issue is tagged with a category; the
# ones most relevant to the two workshop agents:
#
# | Category | What it looks like in our agents |
# |---|---|
# | **Silent tool error** | `search_ticket_history` returns "no tickets authorized" and the agent presents it as "no similar tickets exist" — a permission denial laundered into a wrong answer |
# | **Wrong tool** | The ticket agent calls web search for a question the KB covers; the DD agent skips `screen_vendor` |
# | **Hallucination** | A KB citation that doesn't exist in `kb_tickets/`, or evidence the retriever never returned |
# | **Task evasion** | The DD agent summarizes one source and declares the assessment complete instead of gathering required evidence |
# | **Failed error recovery** | One Tavily timeout and the agent gives up instead of falling back to the vendor KB |
# | **Context explosion** | Multi-turn ticket threads replay full history into every call — token counts climb every turn |
# | **Tracing quality** | Missing `thread_id` / model metadata — an instrumentation gap that silently breaks threads view and cost tracking |
#
# **The workflow per issue** (Engine tab in LangSmith):
# 1. **Agent overview** — Engine first generates a doc describing the project's purpose,
#    architecture, and key metrics from traces; a human reviews/edits it, because all later
#    analysis uses it as context.
# 2. **Issue list** — recurring failures clustered, ranked by severity and recency, each with
#    contributing traces attached. Trace scoping ("only the DD agent's runs") keeps analysis
#    precise when a project mixes workloads.
# 3. **Diagnosis + proposed fix** — root cause, plus a concrete change (prompt edit, tool
#    description, routing rule). Engine can carry the fix into a PR workflow.
# 4. **Regression prevention** — a **custom evaluator** for that failure mode, plus **dataset
#    examples built from the production inputs** — the Day 2 flywheel, automated.
# 5. **Closed loop** — if the issue resurfaces after being closed, it **reopens automatically**.
#
# **Where it meets the rest of this lab:** gateway policy violations (§1.2 — spend cap hit, PII
# redacted, secret caught) **surface as Engine issues**, so the governance layer and the
# improvement layer share one inbox. Fleet runs (§5) produce the same traces, so event-driven
# agents are inside the loop too.
#
# > 🔑 **The hill-climbing loop in one line:** production traces → Engine issue → targeted fix →
# > generated evaluator + dataset examples → redeploy → re-evaluate → repeat. Day 2 did this by
# > hand for one experiment; Engine runs it continuously against everything in production.

# %% [markdown]
# ## 7. Recap — the governed operating model
#
# | Block | Control | Workshop anchor |
# |---|---|---|
# | **LLM Gateway** | Provider Secrets, spend/rate policies, model fallback, PII/secrets redaction, auto-traced calls with policy metadata | `models.py` gateway detection; §1 demo call |
# | **LiteLLM** | Self-hosted OpenAI-compatible proxy: aliases, virtual keys, budgets, fallbacks, logging callbacks | §2 config + client shape (talk-through) |
# | **User security context** | Identity in immutable runtime context; document/row/system enforcement inside tools; audit-logged decisions | Day 1 Lab 05 tools; §3.1 runnable pattern |
# | **Deployment paths** | SaaS → BYOC → hybrid → self-hosted → air-gapped; same agent code, different residency | Lab 01 deployment maps to every path |
# | **Fleet** | Channels (Slack/Teams/Gmail), schedules, webhook/API triggers, approvals, admin controls | `@ticket-agent` channel; Monday re-screening schedule |
# | **Engine** | Trace clustering → issues → diagnosis → fix → generated evaluator + dataset → auto-reopen | §6 loop over both agents' production traces |
#
# **The three independence properties worth repeating:**
# 1. **Gateway choice is independent of agent code** — direct key, LLM Gateway, and LiteLLM all
#    differ by env vars / base URL only.
# 2. **Data governance is independent of model governance** — `UserContext` enforcement works the
#    same no matter which gateway (or none) fronts the model.
# 3. **The improvement loop is independent of how runs start** — notebook, deployment, Fleet
#    channel, or schedule: if it traces to LangSmith, Engine can hill-climb on it.
#
# > **Next steps for Acme:** pick the deployment path per sensitivity tier (§4), decide the
# > gateway story per environment (§1 vs §2), and wire the two workshop agents' traces into
# > Engine + a review queue so the hill-climbing loop starts running on real usage.
#
# > 📚 **Docs:** [LLM Gateway](https://docs.langchain.com/langsmith/llm-gateway) ·
# > [Gateway data protection](https://docs.langchain.com/langsmith/llm-gateway-data-protection) ·
# > [LiteLLM proxy](https://docs.litellm.ai/docs/proxy/quick_start) ·
# > [Deployment environments](https://docs.langchain.com/langsmith/deployment) ·
# > [Fleet](https://docs.langchain.com/langsmith/fleet) ·
# > [Engine](https://docs.langchain.com/langsmith/engine)
