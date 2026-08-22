---
name: source-validation
description: Validate that every citation resolves to a real source, is current, and actually supports the claim it is attached to.
---

# Source Validation Skill

Use this skill before any claim leaves the agent. Evidence review (`evidence-review`) judges whether
evidence is *good*; this skill judges whether the citation is *real*. Both are needed — a fabricated
citation of a high-quality source is the most dangerous failure mode there is, because it survives a
quality check.

## Steps

1. **Resolve the citation.** Every `[source: ...]` tag must name something the agent actually
   retrieved in this run: a vendor KB filename, a vendor ID, a PDF filename, a screening verdict, or
   a URL. If it names nothing retrieved, the claim is **fabricated** — remove it, do not soften it.
2. **Check support.** Re-read the cited passage. Does it support *this* claim, or a neighbouring one?
   A citation that supports a weaker claim than the one made is **overreach**.
3. **Check currency.** Compare the source date against the staleness budget:

   | Source type | Staleness budget | On breach |
   |---|---|---|
   | Sanctions / watchlist screening | Per query, always live | Re-screen; never cite a cached verdict |
   | Vendor database record | 30 days | Cite with an explicit "as of" date |
   | Vendor KB article / capability PDF | 12 months | Flag as dated evidence |
   | Web result | 6 months | Prefer a primary source |

4. **Rank the sources.** When two sources disagree, prefer in this order: audited or official
   documentation → sanctions/screening system → vendor database record → vendor self-reported
   material → web result. State that a conflict existed and which source you preferred, and why.
5. **Decide the verdict** for each claim: `resolved`, `overreach`, `stale`, or `fabricated`.

## Output format

```
## Source Validation: {vendor_name}

| Claim | Citation | Resolves? | Supports? | Current? | Verdict |
|---|---|---|---|---|---|
| ... | [source: ...] | yes/no | yes/partly/no | yes/stale | resolved/overreach/stale/fabricated |

### Conflicts
- {claim} — {source A} says X, {source B} says Y. Preferred {A} because {reason}.

### Removed claims
- {claim} — fabricated citation, no matching retrieval in this run.
```

## Rules

- A claim whose citation does not resolve is **removed**, never rewritten to sound less certain.
- Never invent an "as of" date. If the source carries no date, say `date unknown` and treat it as stale.
- If **every** citation for a material claim fails, return "not enough evidence" for that claim
  rather than a hedged answer. An honest gap is more useful than a confident guess.
