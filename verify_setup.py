#!/usr/bin/env python3
"""First-run check: can this machine run the workshop labs?

    uv run python verify_setup.py

Resolves the model access path (direct provider key vs LangSmith LLM Gateway) exactly the way the
labs do, then proves each piece actually answers rather than merely being configured. Run it before
Lab 01 — if this passes, the labs run.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

WORKSHOP = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSHOP))

from dotenv import find_dotenv, load_dotenv  # noqa: E402

load_dotenv(find_dotenv())  # no override — exported vars win, see the README (Model access section)

import os  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


DAY = 0  # 0 = every day; set from --day


def record(label: str, fn, days: tuple[int, ...] = (1, 2, 3)) -> None:
    """Run a check and remember the result. `days` limits it to the days that actually need it.

    A three-day workshop edits its own fixtures between days, so "did setup pass on Monday" is not
    the same question as "will Day 3 run on Wednesday". Tagging each check by day means the morning
    check covers that day's prerequisites without failing on data a later day has not touched yet.
    """
    if DAY and DAY not in days:
        return
    try:
        detail = fn() or ""
        CHECKS.append((label, True, str(detail)))
    except Exception as e:  # noqa: BLE001 — a check must never abort the run
        CHECKS.append((label, False, f"{type(e).__name__}: {str(e)[:150]}"))


def main() -> int:
    from day1.src.models import (
        DEFAULT_MODEL,
        HEAVY_MODEL,
        JUDGE_MODEL,
        _using_gateway,
        get_embeddings,
        get_judge_model,
        get_model,
    )

    gateway = _using_gateway()
    print("=" * 72)
    print("workshop — setup check" + (f" (Day {DAY})" if DAY else " (all days)"))
    print("=" * 72)
    print(f"model access path : {'LangSmith LLM Gateway' if gateway else 'direct provider key'}")
    if gateway:
        # Report the URL the labs will actually use, not the raw env var: on the
        # `LANGSMITH_GATEWAY=true` path OPENAI_BASE_URL is unset, and printing `None` there made the
        # gateway look misconfigured when it was fine.
        from day1.src.models import GATEWAY_KEY_VARS, _gateway_openai_base, _gateway_setting

        switch = _gateway_setting()
        print(f"  switched on by  : {'LANGSMITH_GATEWAY=' + switch if switch else 'provider BASE_URL'}")
        print(f"  gateway base URL: {_gateway_openai_base()}")
        which = next((v for v in GATEWAY_KEY_VARS if os.environ.get(v)), None)
        print(f"  gateway key from: {which}")
        if which == "LANGSMITH_API_KEY":
            print("  ⚠ that is a tracing PAT — fine locally, but `langgraph deploy` strips it.")
            print("    Use LANGSMITH_GATEWAY_API_KEY (service key) for anything deployed.")
        if os.getenv("OPENAI_API_KEY"):
            print("  ⚠ OPENAI_API_KEY is also set. The labs will not use it, but leaving it there")
            print("    means a lab that bypasses get_model()/get_embeddings() would silently work")
            print("    here and fail on a gateway-only machine. Comment it out to test honestly.")
    else:
        print(f"  provider key    : {'set' if os.getenv('OPENAI_API_KEY') else 'MISSING'}")
    from day1.src.models import PARTICIPANT, scoped_project

    print(f"participant       : {PARTICIPANT or '(not set — fine for a solo run)'}")
    if PARTICIPANT:
        print(f"  your artifacts  : {scoped_project()}, vendor-due-diligence-eval-…")
    print(f"tracing           : {os.getenv('LANGSMITH_TRACING', 'not set')}"
          f"  →  project {os.getenv('LANGSMITH_PROJECT', '(default)')}")
    print(f"model tiers       : agent={DEFAULT_MODEL}  heavy={HEAVY_MODEL}  judge={JUDGE_MODEL}")
    print("-" * 72)

    record(f"agent model ({DEFAULT_MODEL})", lambda: get_model().invoke("reply with: ok").content.strip()[:20])
    record(f"judge model ({JUDGE_MODEL})", lambda: get_judge_model().invoke("reply with: ok").content.strip()[:20])
    record("embeddings", lambda: f"dim {len(get_embeddings().embed_query('probe'))}")

    def langsmith() -> str:
        """Make a real authenticated call — an unusable key must fail here, not at Lab 03."""
        from langsmith import Client

        client = Client()
        next(iter(client.list_datasets(limit=1)), None)  # 401s if the key is bad
        ws = os.getenv("LANGSMITH_WORKSPACE_ID", "(LANGSMITH_WORKSPACE_ID unset)")
        return f"authenticated · workspace {ws}"

    record("LangSmith API key", langsmith)

    def data() -> str:
        import json

        d = WORKSHOP / "day1" / "data"
        tickets = json.loads((d / "tickets.json").read_text())
        users = json.loads((d / "users.json").read_text())
        kb = list((d / "kb_tickets").glob("*.md"))
        assert tickets and users and kb, "day1 fixtures missing"
        return f"{len(tickets)} tickets, {len(kb)} KB articles, {len(users)} users"

    record("Day 1 fixtures", data, days=(1, 2, 3))

    def shipped_db() -> str:
        """The database ships in the repo — confirm the committed copy is usable and matches the JSON.

        Nothing in the labs needs to build it, so a broken or stale shipped copy would otherwise only
        surface mid-lab as a confusing `no such table` or a row count that disagrees with the fixtures.
        """
        import json
        import sqlite3

        d = WORKSHOP / "day1" / "data"
        db = d / "tickets.db"
        assert db.exists(), "day1/data/tickets.db missing — run: python -c 'from day1.src.ticket_db import build_database; build_database()'"
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = con.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        people = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        orphans = con.execute(
            "SELECT COUNT(*) FROM tickets t LEFT JOIN users u ON t.user_id = u.user_id WHERE u.user_id IS NULL"
        ).fetchone()[0]
        con.close()
        expect_t = len(json.loads((d / "tickets.json").read_text()))
        expect_u = len(json.loads((d / "users.json").read_text()))
        assert rows == expect_t, f"db has {rows} tickets, tickets.json has {expect_t} — rebuild it"
        assert people == expect_u, f"db has {people} users, users.json has {expect_u} — rebuild it"
        assert orphans == 0, f"{orphans} ticket(s) reference a missing user"
        return f"{rows} tickets, {people} users, 0 orphans — matches the JSON"

    record("shipped ticket database", shipped_db, days=(1,))

    def shipped_pdfs() -> str:
        """The vendor PDFs ship too, and no lab regenerates them — so check they are readable here."""
        from pypdf import PdfReader

        pdfs = sorted((WORKSHOP / "day1" / "data" / "pdfs").glob("*.pdf"))
        assert pdfs, "day1/data/pdfs is empty"
        chars = 0
        for p in pdfs:
            text = "".join(pg.extract_text() or "" for pg in PdfReader(str(p)).pages)
            assert len(text.strip()) > 100, f"{p.name} yielded no extractable text"
            chars += len(text)
        return f"{len(pdfs)} PDFs, {chars:,} chars extractable"

    record("shipped vendor PDFs", shipped_pdfs, days=(1, 2))

    def rag_corpus() -> str:
        """The retrieval corpora ship as Markdown — no lab regenerates them, so check them here.

        `kb_tickets/` backs Day 1 support retrieval; `kb/` backs Day 1/Day 2 vendor retrieval. An
        empty or truncated article does not fail loudly at retrieval time, it just quietly stops
        being findable, so a size floor catches more than an existence check.
        """
        d = WORKSHOP / "day1" / "data"
        tickets_kb = sorted((d / "kb_tickets").glob("*.md"))
        vendor_kb = sorted((d / "kb").glob("*.md"))
        assert tickets_kb and vendor_kb, "RAG corpora missing"
        thin = [p.name for p in tickets_kb + vendor_kb if len(p.read_text().strip()) < 200]
        assert not thin, f"suspiciously short article(s): {thin}"
        chars = sum(len(p.read_text()) for p in tickets_kb + vendor_kb)
        return f"{len(tickets_kb)} support + {len(vendor_kb)} vendor articles, {chars:,} chars"

    record("Day 1 RAG corpora", rag_corpus, days=(1, 2))

    def day2_fixtures() -> str:
        """Day 2 ships the deep agent's skills plus its evaluation and risk data."""
        import json

        d = WORKSHOP / "day2" / "data"
        skills = sorted((d / "agent" / "skills").glob("*/SKILL.md"))
        assert (d / "agent" / "SKILL.md").exists(), "day2/data/agent/SKILL.md missing"
        assert len(skills) >= 4, f"expected 4 skill files, found {len(skills)}"
        for f in ("due_diligence_data.json", "due_diligence_eval.json", "risk_criteria.json"):
            assert json.loads((d / f).read_text()), f"day2/data/{f} missing or empty"
        seeds = sorted((d / "wiki").glob("*.md"))
        assert seeds, "day2/data/wiki seeds missing — the wiki-memory lab needs the directory to exist"
        evals = json.loads((d / "due_diligence_eval.json").read_text())
        return f"{len(skills)} skills, {len(evals)} eval examples, {len(seeds)} wiki seeds"

    record("Day 2 fixtures", day2_fixtures, days=(2, 3))

    def cross_refs() -> str:
        """Every pointer between fixtures must resolve.

        These break silently: a vendor whose `page` names a missing article returns an empty KB hit
        rather than an error, and an eval example naming an unknown vendor_id scores a confident zero.
        """
        import json

        d = WORKSHOP / "day1" / "data"
        vendors = {k: v for k, v in json.loads((d / "vendors.json").read_text()).items()
                   if k.startswith("VND-")}
        missing_pages = [v["page"] for v in vendors.values() if not (d / "kb" / v["page"]).exists()]
        assert not missing_pages, f"vendor KB page(s) missing: {missing_pages}"

        d2 = WORKSHOP / "day2" / "data"
        bad = []
        for name in ("due_diligence_data.json", "due_diligence_eval.json"):
            rows = json.loads((d2 / name).read_text())
            for row in rows if isinstance(rows, list) else []:
                vid = row.get("vendor_id") or (row.get("inputs") or {}).get("vendor_id")
                if vid and vid not in vendors:
                    bad.append(f"{name}:{vid}")
        assert not bad, f"eval data references unknown vendor(s): {bad}"
        return f"{len(vendors)} vendors → KB pages, eval vendor_ids all resolve"

    record("fixture cross-references", cross_refs, days=(1, 2, 3))

    def rag() -> str:
        from day1.src.ticket_agent_graph import search_kb

        out = search_kb.invoke({"query": "mfa reset"})
        assert "source:" in out, "KB search returned no cited source"
        return "KB search returns cited snippets"

    record("RAG index (embeddings end-to-end)", rag, days=(1, 2))

    def graphs() -> str:
        import day1.src.ticket_agent_graph as t

        names = [g for g in ("graph", "graph_with_memory", "graph_hitl", "graph_structured") if hasattr(t, g)]
        assert len(names) == 4, names
        return "4 deployable graphs"

    record("deployable graphs", graphs, days=(1, 3))

    def deploy_tooling() -> str:
        """Day 3 deploys — check the pieces that only Day 3 uses, before Day 3 starts.

        `langgraph.json` naming a graph that no longer imports is the classic Day 3 failure: it
        surfaces as an opaque server start-up error rather than as a missing symbol.
        """
        import importlib
        import json
        import shutil

        cfg_path = WORKSHOP / "day1" / "langgraph.json"
        assert cfg_path.exists(), "day1/langgraph.json missing — Day 3 deploys from it"
        cfg = json.loads(cfg_path.read_text())
        graphs_cfg = cfg.get("graphs", {})
        assert graphs_cfg, "day1/langgraph.json declares no graphs"
        broken = []
        for gname, target in graphs_cfg.items():
            mod_path, _, attr = str(target).partition(":")
            # Paths in langgraph.json are relative to the config file, which is how LangGraph Server
            # resolves them — not relative to the repo root. Resolving from the root instead is why
            # this check first reported a false ModuleNotFoundError.
            target_file = (cfg_path.parent / mod_path).resolve()
            mod = ".".join(target_file.relative_to(WORKSHOP).with_suffix("").parts)
            try:
                if not hasattr(importlib.import_module(mod), attr):
                    broken.append(f"{gname} -> {target} (no attribute {attr})")
            except Exception as e:  # noqa: BLE001
                broken.append(f"{gname} -> {target} ({type(e).__name__})")
        assert not broken, f"langgraph.json points at missing graph(s): {broken}"
        cli = "langgraph CLI on PATH" if shutil.which("langgraph") else "langgraph CLI not on PATH (use `uv run langgraph dev`)"
        return f"{len(graphs_cfg)} graphs declared, all importable · {cli}"

    record("Day 3 deployment tooling", deploy_tooling, days=(3,))

    print()
    for label, ok, detail in CHECKS:
        print(f"  {'✅' if ok else '⛔'} {label:38} {detail}")

    failed = [c for c in CHECKS if not c[1]]
    print("-" * 72)
    if not failed:
        nb = {1: "day1/notebooks/01_setup.ipynb", 2: "day2/notebooks/01_deep_agent.ipynb",
              3: "day3/notebooks/01_deployments.ipynb"}.get(DAY, "day1/notebooks/01_setup.ipynb")
        print(f"✅ Ready. Start with {nb} (or its .py twin in {nb.split(chr(47))[0]}/src/).")
        return 0
    print(f"⛔ {len(failed)} check(s) failed.")
    if any("model" in c[0] or "embeddings" in c[0] for c in failed):
        print("   Model/embedding failures are almost always credentials — see the README (Model access section) for the")
        print("   two supported paths and how to tell which one you are on.")
    return 1


def _cli() -> int:
    import argparse

    global DAY
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", type=int, choices=(1, 2, 3), default=None,
                    help="check only the prerequisites for this day (default: all three)")
    DAY = ap.parse_args().day or 0
    return main()


if __name__ == "__main__":
    raise SystemExit(_cli())
