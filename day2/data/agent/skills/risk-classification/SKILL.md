---
name: risk-classification
description: Classify a vendor's risk level (low/medium/high) from evidence, with a severity per signal and an explicit escalation decision.
---

# Risk Classification Skill

Use this skill after evidence has been gathered and validated, to turn findings into a defensible
risk level. The goal is a classification a procurement officer can act on and an auditor can retrace.

## Risk signal catalogue

| Signal | Severity | Notes |
|---|---|---|
| Sanctions list match | **Critical** | Automatic HIGH + escalate. No offsetting evidence applies. |
| Watchlist / adverse media match | **High** | HIGH unless the match is positively excluded as a false positive |
| Ownership in a restricted jurisdiction | **High** | Check ultimate beneficial ownership, not just the trading entity |
| Required certification missing or expired | **Medium** | Blocking for regulated-sector work; note which certification |
| No verifiable delivery history with Acme | **Medium** | Common for new entrants — a gap, not a fault |
| Financial data unavailable | **Medium** | Cannot assess going-concern risk |
| Single-source dependency for a critical item | **Medium** | Supply-chain concentration risk |
| Vendor self-reported claims only | **Low** | Reduce confidence; request primary evidence |
| Stale evidence (> 12 months) | **Low** | Note the age explicitly |

## Classification rules

Apply in order; the **first** rule that matches decides the level.

1. Any **Critical** signal → **HIGH**, escalate, recommend exclusion pending review.
2. Two or more **High** signals → **HIGH**, escalate.
3. One **High** signal → **MEDIUM**, escalate only if it touches sanctions, ownership, or security.
4. Two or more **Medium** signals → **MEDIUM**.
5. Otherwise → **LOW**.

A missing sanctions screening is **not** a LOW result. If screening has not been run, the
classification is **incomplete** — say so rather than returning a level.

## Output format

```
## Risk Classification: {vendor_name}

**Level:** LOW | MEDIUM | HIGH | INCOMPLETE
**Escalate:** yes/no — {which rule fired}

| Signal | Severity | Evidence | [source] |
|---|---|---|---|
| ... | Critical/High/Medium/Low | ... | ... |

### Data gaps
- {what could not be assessed, and what would close the gap}

### Confidence
{high/medium/low} — {what would raise it}
```

## Rules

- Never average severities into a level. One critical signal is not offset by five clean checks.
- State the rule number that fired. "HIGH because rule 1 (sanctions match)" is auditable;
  "HIGH based on overall assessment" is not.
- Distinguish **absence of evidence** from **evidence of absence** in every gap you list.
