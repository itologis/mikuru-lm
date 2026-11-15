# Attribution and Licenses

## Included Sources: Original Generation

The pilot corpus currently contains two original synthetic generations.
External sources are documented in the manifest for future local collection
but are not included in the JSONL.

Records categorized as `japanese_culture_knowledge` and
`japanese_inspired_style` are original English dialogues generated
specifically for the Mikuru LM pilot.

- **License:** Pending / Project Internal.
- **Redistribution:** Not approved until a formal license decision is made.
- **Factual basis:** Cultural facts were checked against Web Japan and
  CLAIR resources to account for regional and household variations.

## Documented External Sources

### OASST1
- Publisher: OpenAssistant
- License: Apache-2.0
- Source: https://huggingface.co/datasets/OpenAssistant/oasst1
- Status: Pending local raw-text verification.
- Requirements: Fetch Parquet files, verify complete text, inspect review
  labels, and apply privacy and quality filters.

### Taskmaster-1
- Publisher: Google Research
- License: CC BY 4.0
- Source: https://github.com/google-research-datasets/Taskmaster/tree/master/TM-1-2019
- Status: Pending local fetch.
- Requirements: Download the official JSON files and verify conversation
  IDs and utterance indexes.

### SOC-2508
- Publisher: marcodsn
- License: CC BY 4.0
- Source: https://huggingface.co/datasets/marcodsn/SOC-2508
- Status: Excluded pending role-mapping review.
- Reason: Peer-to-peer persona structure requires reliable role mapping.

## Excluded Sources

- **DailyDialog:** CC BY-NC-SA 4.0. NonCommercial and ShareAlike restrictions.
- **EmpatheticDialogues:** CC BY-NC 4.0. NonCommercial restriction.
- **Taskmaster-4:** Self-dialogue format, not genuine two-party dialogue.
- **Marugoto / Irodori:** Proprietary Japan Foundation educational materials.

## Approval Policy

A source must pass provenance, raw-text, reconstruction, license, privacy,
and quality checks before its records can enter an approved training corpus.

A documented source is not automatically an approved source.
A generated example is not automatically cleared for redistribution.
