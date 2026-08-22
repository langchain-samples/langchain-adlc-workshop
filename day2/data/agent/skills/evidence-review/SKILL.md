---
name: evidence-review
description: Review evidence gathered during vendor due diligence — validate sources, assess quality, and flag gaps.
---

# Evidence Review Skill

Use this skill when reviewing evidence gathered during vendor due diligence.

## Steps

1. **Source identification** — For each claim, identify the source: vendor KB article, vendor database record, PDF capability statement, web search result, or sanctions screening.
2. **Verification status** — Mark each claim as:
   - **Verified** — claim is directly supported by the cited source
   - **Partially verified** — claim is partially supported; note what's missing
   - **Unverified** — claim has no supporting evidence; flag for follow-up
   - **Contradicted** — evidence contradicts the claim; flag immediately
3. **Quality assessment** — Rate evidence quality:
   - **High** — official documentation, verified certifications, audited records
   - **Medium** — vendor self-reported data, dated but plausible claims
   - **Low** — unattributed claims, marketing language, stale data
4. **Gap analysis** — List what's missing: certifications not confirmed, no financial data, no reference customers, no sanctions check

## Output format

```
## Evidence Review: {vendor_name}

### Claims
| Claim | Source | Status | Quality |
|---|---|---|---|
| ... | ... | Verified/Partially verified/Unverified/Contradicted | High/Medium/Low |

### Gaps
- ...

### Recommendation
- ...
```

## Source validation checklist
- [ ] Source is cited by name (KB article filename, vendor ID, PDF filename, URL)
- [ ] Claim is directly supported by the source content
- [ ] Source is current (not stale or outdated)
- [ ] No PII in the evidence (contact emails, phone numbers redacted)
- [ ] Sanctions screening has been run for the vendor