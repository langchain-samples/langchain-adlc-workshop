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
# # 00 · Setup check — Day 2
#
# **Workshop:** LangChain ADLC Workshop · **Day 2** · **ADLC stage:** Operate
#
# > **Loop Engineering focus: Operate loop** — verify the Day 2 prerequisites before the deep agent labs — the agent skills, the eval sets, and the vendor corpus the sub-agents retrieve from
#
# **Runs standalone.** This is the first thing to run on Day 2; it depends on nothing from the
# other labs. It is the notebook form of `uv run python day2/verify_setup.py`.
#
# Why re-check on Day 2 when Day 1 already passed: the workshop edits its own fixtures as it goes.
# Agents write to the wiki, evals write datasets, and you may have edited a JSON fixture between
# sessions. "It worked yesterday" is not evidence that today's labs will run, and a fixture problem
# found here costs a minute — found mid-lab it costs the room's attention.

# %% [markdown]
# ## 1. Which model access path are you on?
#
# Both paths are supported: a direct provider key, or the LangSmith LLM Gateway. The labs resolve
# this the same way this cell does, so whatever it prints is what the labs will use.

# %%
import sys
from pathlib import Path

WORKSHOP = Path.cwd()
while WORKSHOP != WORKSHOP.parent and not (WORKSHOP / "verify_setup.py").exists():
    WORKSHOP = WORKSHOP.parent
sys.path.insert(0, str(WORKSHOP))
# `day2/verify_setup.py` is a real file (a thin per-day wrapper around the root one), so the loop
# above stops there instead of walking up to the true workshop root — one level short. Add the
# parent too so `day1.src.*` / `utils.*` resolve regardless of which one the loop actually found.
sys.path.insert(0, str(WORKSHOP.parent))

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())  # no override — exported shell vars win, as in every lab

from day1.src.models import DEFAULT_MODEL, JUDGE_MODEL, _using_gateway

print(f"model access path : {'LangSmith LLM Gateway' if _using_gateway() else 'direct provider key'}")
print(f"model tiers       : agent={DEFAULT_MODEL}  judge={JUDGE_MODEL}")

# %% [markdown]
# ## 2. Run the Day 2 checks
#
# The same checks as `day2/verify_setup.py`, in-process so you can see each result. Each one makes
# a real call or reads a real file — nothing here passes just because a variable is set.

# %%
import verify_setup

verify_setup.CHECKS.clear()
verify_setup.DAY = 2
_rc = verify_setup.main()

# Fail loudly. Printing the exit code and carrying on would let this notebook finish green while
# Day 2 is unrunnable — and the sweep that runs these as scripts would record a pass. A setup
# check that cannot fail is not a check.
if _rc != 0:
    _failed = [name for name, ok, _ in verify_setup.CHECKS if not ok]
    raise RuntimeError(
        f"Day 2 is not ready — {len(_failed)} check(s) failed: {_failed}. "
        "Fix these before starting the labs; the README's *Model access* section covers credentials."
    )
print("\nexit code: 0 — every Day 2 prerequisite answered.")

# %% [markdown]
# ## 3. Can the deep agent actually be built?

# %%
# The Day 2 labs all start from this factory. Building it here fails fast on a bad fixture path or
# a missing skill file, rather than three cells into Lab 01.
from day2.src.deep_agent_factory import build_dd_agent

_agent = build_dd_agent(with_memory=True, with_skills=True)
_tools = [getattr(t, "name", type(t).__name__) for t in getattr(_agent, "tools", [])] or None
print(f"deep agent built  : {_agent.name}")
print(f"subagents wired   : delegation via task() is available")
print("✅ Day 2 agent constructs — skills and memory paths resolved")

# %% [markdown]
# ## 4. Recap
#
# | Checked | Why it matters on Day 2 |
# |---|---|
# | Model + judge tier answer | a wrong key fails at the first `invoke`, not at import |
# | Embeddings answer | retrieval fails silently if this is misconfigured |
# | LangSmith authenticated | tracing, datasets and annotation queues all need it |
# | Shipped fixtures present and consistent | a missing SKILL.md or eval example breaks the labs mid-run |
#
# If everything above is green, open **`day2/notebooks/01_deep_agent.ipynb`**. If not, the failing line names the piece — and
# the README's *Model access* section covers the two credential paths.
