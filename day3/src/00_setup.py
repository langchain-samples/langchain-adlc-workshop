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
# # 00 · Setup check — Day 3
#
# **Workshop:** LangChain ADLC Workshop · **Day 3** · **ADLC stage:** Operate
#
# > **Loop Engineering focus: Operate loop** — verify the Day 3 prerequisites before the deployment labs — `langgraph.json`, the graphs it declares, and the LangSmith SDK surface the observability labs call
#
# **Runs standalone.** This is the first thing to run on Day 3; it depends on nothing from the
# other labs. It is the notebook form of `uv run python day3/verify_setup.py`.
#
# Why re-check on Day 3 when Day 1 already passed: the workshop edits its own fixtures as it goes.
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


def _find_workshop_root(start: Path) -> Path:
    """Walk up from `start` looking for the directory that holds both `day1/` and `utils/` — the
    actual workshop root. Checking for those two directories (rather than a filename like
    `verify_setup.py`) is deliberate: `day3/verify_setup.py` is a real file too, a thin per-day
    wrapper, so a filename-based check stops one level short of the true root and every path built
    from `WORKSHOP` afterward (not just imports) silently points at the wrong place."""
    for cand in (start.resolve(), *start.resolve().parents):
        if (cand / "day1").is_dir() and (cand / "utils").is_dir():
            return cand
    raise FileNotFoundError("Could not locate the workshop root (the directory holding day1/ and utils/).")


WORKSHOP = _find_workshop_root(Path.cwd())
sys.path.insert(0, str(WORKSHOP))

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())  # no override — exported shell vars win, as in every lab

from day1.src.models import DEFAULT_MODEL, JUDGE_MODEL, _using_gateway

print(f"model access path : {'LangSmith LLM Gateway' if _using_gateway() else 'direct provider key'}")
print(f"model tiers       : agent={DEFAULT_MODEL}  judge={JUDGE_MODEL}")

# %% [markdown]
# ## 2. Run the Day 3 checks
#
# The same checks as `day3/verify_setup.py`, in-process so you can see each result. Each one makes
# a real call or reads a real file — nothing here passes just because a variable is set.

# %%
import verify_setup

verify_setup.CHECKS.clear()
verify_setup.DAY = 3
_rc = verify_setup.main()

# Fail loudly. Printing the exit code and carrying on would let this notebook finish green while
# Day 3 is unrunnable — and the sweep that runs these as scripts would record a pass. A setup
# check that cannot fail is not a check.
if _rc != 0:
    _failed = [name for name, ok, _ in verify_setup.CHECKS if not ok]
    raise RuntimeError(
        f"Day 3 is not ready — {len(_failed)} check(s) failed: {_failed}. "
        "Fix these before starting the labs; the README's *Model access* section covers credentials."
    )
print("\nexit code: 0 — every Day 3 prerequisite answered.")

# %% [markdown]
# ## 3. Is the deployment surface ready?

# %%
# Day 3 deploys and then queries what it deployed. Both halves are checked here: the graphs
# `langgraph.json` declares must import, and the LangSmith SDK must authenticate.
import json

from langsmith import Client

cfg = json.loads((WORKSHOP / "day1" / "langgraph.json").read_text())
print(f"graphs declared   : {len(cfg.get('graphs', {}))}")
for name in cfg.get("graphs", {}):
    print(f"  · {name}")

_client = Client()
next(iter(_client.list_datasets(limit=1)), None)  # 401s here if the key is unusable
print("✅ LangSmith SDK authenticated")

# `langgraph dev` is what Lab 01 §2 uses for a local deployment. If this says it is missing, run the
# labs against a cloud deployment by setting DEPLOYMENT_API_URL instead.
import shutil

print(f"langgraph CLI     : {'on PATH' if shutil.which('langgraph') else 'use `uv run langgraph dev`'}")

# %% [markdown]
# ## 4. Recap
#
# | Checked | Why it matters on Day 3 |
# |---|---|
# | Model + judge tier answer | a wrong key fails at the first `invoke`, not at import |
# | Embeddings answer | retrieval fails silently if this is misconfigured |
# | LangSmith authenticated | tracing, datasets and annotation queues all need it |
# | Shipped fixtures present and consistent | Day 3 reads Day 1 and Day 2 data; a missing fixture surfaces as an empty trace |
#
# If everything above is green, open **`day3/notebooks/01_deployments.ipynb`**. If not, the failing line names the piece — and
# the README's *Model access* section covers the two credential paths.
