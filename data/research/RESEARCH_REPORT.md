# Mikuru LM Pilot Corpus Research Report

**Research date:** 2026-11-15  
**Corpus target:** 5,000 conversations

## Executive Summary

The pilot collection contains two original synthetic English conversations:
one Japanese culture knowledge example and one Japanese-inspired style example.

No external dataset records have been collected into the pilot JSONL.

The original examples have traceable project provenance, but their license
remains Pending / Project Internal. They are not yet cleared for redistribution
or unrestricted training use.

External dataset candidates remain pending local collection and verification
or have been excluded for documented reasons.

## Verification Status Matrix

| Dataset | Source ID | Raw Text | Reconstruction | License | Privacy | Decision |
|---|---|---|---|---|---|---|
| OASST1 | Yes | Unverified: viewer truncation | Pending local validation | Verified candidate | Filtering required | Pending |
| Taskmaster-1 | Yes | Pending local fetch | Pending | Verified candidate | Review required | Pending |
| SOC-2508 | Yes | Available | Role mapping unresolved | Verified candidate | Review required | Excluded pending review |
| Original Culture | Yes | Yes | Yes | Project Internal, pending | Reviewed | Internal sample only |
| Original Style | Yes | Yes | Yes | Project Internal, pending | Reviewed | Internal sample only |

## Unresolved Items

1. **OASST1 raw text verification**
   - Retrieve the complete, untruncated text for message
     `c8e83833-ecbc-44fe-b6db-735228c25a1c`.
   - Verify `review_result` and `rank` selection logic.
   - Inspect nested labels, including PII and quality indicators.
   - Exclude records that fail the project's privacy and quality thresholds.

2. **Taskmaster-1 raw fetch**
   - Download `self-dialogs.json` and `woz-dialogs.json` from the official
     repository.
   - Verify genuine `conversationId` values and utterance indexes.
   - Preserve source attribution.

3. **SOC-2508 role mapping**
   - Inspect persona identifiers and sender relationships.
   - Do not arbitrarily assign user and assistant roles.
   - Keep excluded until a defensible mapping is established.

4. **Original record licensing**
   - Decide and document the project's internal-use and redistribution policy.
   - Do not treat Project Internal as an open license.

## Category Counts

| Category | Actual Collected | Pilot Target |
|---|---:|---:|
| General Conversation | 0 | 3,500 |
| Japanese Culture Knowledge | 1 | 750 |
| Japanese-Inspired Style | 1 | 750 |
| **Total** | **2** | **5,000** |

Targets are planning allocations, not completed collection counts.

## Cultural Fact-Checking References

The synthetic culture record `culture-genkan-nuance-001` discusses
variation in restroom slipper practices in Japan.

References:
- Web Japan / Kids Web Japan: https://web-japan.org/kidsweb/
- CLAIR Multicultural Life-in-Japan: https://www.clair.or.jp/e/multiculture/index.html

These references were used to avoid presenting regional and household
customs as universal rules.

## Compatibility

The JSONL schema has not yet been fully validated against every Rust
dataset parser field. Run schema validation before treating the pilot
records as approved training input.

## Final Summary

- Actual collected records: 2 original synthetic examples.
- External records collected: 0.
- External candidates: OASST1 and Taskmaster-1 pending local verification.
- SOC-2508: excluded pending role-mapping review.
- Restricted or proprietary sources: excluded.
- Training approval: pending.
