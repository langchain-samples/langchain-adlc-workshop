---
name: report-drafting
description: Draft the final Acme vendor due diligence report with required sections, citations on every material claim, and an explicit human-review flag.
---

# Due Diligence Report Drafting Skill

Use this skill last, to assemble validated evidence and a risk classification into the deliverable a
procurement officer reads. The report is the audit artifact: if a decision is challenged in twelve
months, this is what gets re-read.

## Required sections, in order

1. **Vendor summary** — who they are, what they supply, jurisdiction. Two or three sentences.
2. **Evidence** — one bullet per material claim, each ending in its `[source: ...]` tag and a
   verification label (`verified` / `partially verified` / `unverified`).
3. **Risk signals** — the table from `risk-classification`, with severity per signal.
4. **Compliance screening** — the verdict verbatim: `CLEAR`, `WATCHLIST`, or `SANCTIONED`, with the
   entity matched and the source. Never paraphrase a screening verdict.
5. **Suitability** — recommend / recommend with conditions / do not recommend, and the conditions.
6. **Confidence** — high / medium / low, and what would raise it.
7. **Follow-up questions** — specific and answerable, addressed to a named party.
8. **Human review flag** — required / not required, and which rule triggered it.

## Rules

- **Every material claim in the narrative carries a citation**, not just the bullets in §2. A
  narrative sentence like "an established supplier" is a claim and needs a source or must be cut.
  This is the single most common failure in this report — see Day 2 Lab 04, where the
  `groundedness` judge grades exactly this.
- **No PII.** No contact emails, phone numbers, or personal names of vendor staff. Refer to roles.
- **Do not smooth over gaps.** A section with no evidence says "no evidence found", not nothing.
- **Never restate the risk level more favourably** than `risk-classification` produced it.
- Recommend nothing without a screening verdict present. If screening is missing, the suitability
  section says "cannot recommend — screening not completed".

## Output skeleton

```
# Vendor Due Diligence: {vendor_name}

## Vendor summary
...

## Evidence
- {claim} — {detail} [source: ...] (verified)

## Risk signals
| Signal | Severity | Evidence |
|---|---|---|

## Compliance screening
**Verdict:** CLEAR | WATCHLIST | SANCTIONED — {matched entity or "no match"} [source: ...]

## Suitability
...

## Confidence
...

## Follow-up questions
1. ...

## Human review
Required: yes/no — {rule}
```
