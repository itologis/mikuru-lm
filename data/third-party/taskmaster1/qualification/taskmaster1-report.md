# Taskmaster-1 inspection report

- Tool: `mikuru-lm inspect-taskmaster1` (`mikuru-core` 0.1.0)
- Report format: 1
- Deterministic: no timestamps, host paths, or conversation text are recorded.

## Dataset identity

| Field | Value |
|---|---|
| Name | Taskmaster-1 |
| Publisher | Google LLC |
| Upstream repository | https://github.com/google-research-datasets/Taskmaster |
| Upstream revision | `d92cb6af3005f1dc09c39e75e7daf4a04905e00b` |
| Subset directory | `TM-1-2019` |
| Paper | https://arxiv.org/abs/1909.05358 |
| License | CC BY 4.0 |
| License evidence | `README.md` (present: yes) |

13,215 task-oriented dialogs: 7,708 written self-dialogs and 5,507 spoken Wizard-of-Oz dialogs across six domains.

## License findings

- **Repository metadata:** No separate LICENSE file or repository-level license metadata was present at the pinned revision.
- **Data notice:** The TM-1-2019 README begins with a COPYRIGHT NOTICE applying the Creative Commons Attribution 4.0 License to the dialog data.
- **Attribution:** Credit Bill Byrne, Karthik Krishnamoorthi, Chinnadhurai Sankar, Arvind Neelakantan, Amit Dubey, Kyu-Young Kim, and Andy Cedilnik (Google LLC), and cite the Taskmaster-1 EMNLP 2019 paper.
- **Redistribution:** CC BY 4.0 permits sharing and redistribution with attribution. This is a summary of the published terms, not legal advice.
- **Modifications:** CC BY 4.0 permits adaptation when the adaptation is indicated and attribution is preserved; adaptations must not imply endorsement by the licensor.
- **Training use:** No additional training-use restriction was found in the official README or paper; the notice applies to the dialog data, not only to source code.

Caveats:

- The license notice is in the dataset README, not in a separate LICENSE file or repository metadata.
- Embedded third-party content cannot be ruled out from the published files.
- No consent, anonymization, or PII-scrubbing statement was found in the README or the Taskmaster-1 paper.

## File inventory

| Path | Kind | Bytes | SHA-256 | Upstream source |
|---|---|---:|---|---|
| `README.md` | license_evidence | 8046 | `05e1bc05724716ca7905f567a2ec30d99c8d2f5f58e926c2ca7d9da116002446` | https://raw.githubusercontent.com/google-research-datasets/Taskmaster/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019/README.md |
| `ontology.json` | ontology | 1426 | `b8103a812799a8c0095800ba9fbf2b4654bccf7f099daa8a4a145c340a69bc52` | https://raw.githubusercontent.com/google-research-datasets/Taskmaster/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019/ontology.json |
| `sample.json` | sample | 7116 | `83fa11504f6e6a8b32fc4f140db82ed624d0e1fdfd0b48f4f40b60c118ac266d` | https://raw.githubusercontent.com/google-research-datasets/Taskmaster/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019/sample.json |
| `self-dialogs.json` | dialog_subset | 65748638 | `1e590ed0ccee279e40c2fb9e083d3b9417477c6bfe35ce5b2277167698dd858d` | https://raw.githubusercontent.com/google-research-datasets/Taskmaster/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019/self-dialogs.json |
| `train-dev-test/dev.csv` | split_list | 32340 | `d83a6f413f45e01b376d611aadf07fd0fc49f6cedffb4ffff7e7b1f8393c13bf` | https://raw.githubusercontent.com/google-research-datasets/Taskmaster/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019/train-dev-test/dev.csv |
| `train-dev-test/test.csv` | split_list | 32340 | `1938698dc98b71c5256c05da092dff0d4f9eff224ec413b3c3eed57bc0bca928` | https://raw.githubusercontent.com/google-research-datasets/Taskmaster/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019/train-dev-test/test.csv |
| `train-dev-test/train.csv` | split_list | 259056 | `88e48f53524ed123dfc57733ccc23f80e176d1e057d7ea46a62a01ddce02626e` | https://raw.githubusercontent.com/google-research-datasets/Taskmaster/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019/train-dev-test/train.csv |
| `woz-dialogs.json` | dialog_subset | 37204053 | `cd3bc4e968487315d412c044d30af2bf0a4b33c3ef8b74c589f1e1fa832bf72f` | https://raw.githubusercontent.com/google-research-datasets/Taskmaster/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019/woz-dialogs.json |

## Totals

| Metric | Value |
|---|---:|
| Records read | 13215 |
| Conversations parsed | 13215 |
| Conversations valid | 7195 |
| Conversations flagged | 6016 |
| Conversations invalid | 4 |
| Malformed records | 0 |
| Messages | 303066 |
| Characters | 12602704 |

## Qualification summary

| Subset | Status |
|---|---|
| `self-dialogs` | **CONDITIONAL** |
| `woz-dialogs` | **CONDITIONAL** |

## Subset: `self-dialogs` (`self-dialogs.json`)

**Qualification: CONDITIONAL**

Reasons:

- License: CC BY 4.0, stated in the TM-1-2019 README copyright notice; the upstream repository has no separate LICENSE file.
- Pinned upstream revision: d92cb6af3005f1dc09c39e75e7daf4a04905e00b.
- 953 of 7708 records are flagged, invalid, or malformed; see the structural findings.
- 953 conversations contain contact-like patterns (email, URL, phone-like digit runs); no consent, anonymization, or PII-scrubbing statement was found upstream.
- No conversation fits the current 8-message / 500-character training budget; the source is only usable at a larger sequence budget.
- Conditional decision: the subset is potentially useful, but the listed issues need explicit repair/exclusion and privacy decisions before training.

### Counts

| Metric | Value |
|---|---:|
| Records read | 7708 |
| Conversations parsed | 7708 |
| Valid / flagged / invalid | 6755 / 953 / 0 |
| Malformed records | 0 |
| Messages | 169469 |
| Characters | 7390510 |
| Annotated utterances | 68228 |
| Non-ASCII characters | 1735 |
| CJK characters | 0 |

### Speaker roles

| Original label | Utterances |
|---|---:|
| `ASSISTANT` | 83776 |
| `USER` | 85693 |

Mapped roles:

- `assistant`: 83776
- `user`: 85693

Unmapped speaker labels:

- none

### Domains

| Domain | Conversations |
|---|---:|
| `auto_repair` | 1161 |
| `coffee_ordering` | 1376 |
| `movie_tickets` | 1305 |
| `pizza_ordering` | 1468 |
| `restaurant_reservation` | 1300 |
| `ride_service` | 1098 |

### Instruction identifiers

| Instruction ID | Conversations |
|---|---:|
| `auto-repair-appt-1` | 1161 |
| `coffee-ordering-1` | 735 |
| `coffee-ordering-2` | 641 |
| `movie-finder` | 54 |
| `movie-ticket-1` | 37 |
| `movie-tickets-1` | 642 |
| `movie-tickets-2` | 377 |
| `movie-tickets-3` | 195 |
| `pizza-ordering-1` | 257 |
| `pizza-ordering-2` | 1211 |
| `restaurant-table-1` | 704 |
| `restaurant-table-2` | 494 |
| `restaurant-table-3` | 102 |
| `uber-lyft-1` | 646 |
| `uber-lyft-2` | 452 |

### Conversation lengths

| Measure | Min | Median | P90 | Max |
|---|---:|---:|---:|---:|
| Messages | 15 | 21 | 26 | 30 |
| Characters | 21 | 897 | 1397 | 6024 |
| Utterance characters | 1 | 36 | 85 | 3619 |

### Structural findings

| Finding | Records affected |
|---|---:|
| `potential_contact_data` | 953 |

Occurrence counts: deleted placeholders 0, same-role adjacencies 0, adjacent duplicate utterances 0, empty conversations 0.

Utterance-level gaps:

- non-object utterances: 0
- missing text: 0
- empty text: 0
- missing index: 0
- non-sequential index: 0
- missing speaker: 0
- unknown speaker: 0

### Privacy signals

| Pattern | Utterances |
|---|---:|
| Email-like | 33 |
| URL-like | 3 |
| Phone-like digit runs | 1191 |
| Long digit runs (13+) | 24 |
| Any pattern | 1217 |

Conversations affected: 953.

### Budget fit (applicability, not a source defect)

| Budget | Conversations |
|---|---:|
| 8 messages / 500 characters | 0 |
| 8 messages / 1000 characters | 0 |
| 16 messages / 2000 characters | 14 |
| 32 messages / 4000 characters | 7706 |

## Subset: `woz-dialogs` (`woz-dialogs.json`)

**Qualification: CONDITIONAL**

Reasons:

- License: CC BY 4.0, stated in the TM-1-2019 README copyright notice; the upstream repository has no separate LICENSE file.
- Pinned upstream revision: d92cb6af3005f1dc09c39e75e7daf4a04905e00b.
- 5067 of 5507 records are flagged, invalid, or malformed; see the structural findings.
- 948 conversations contain contact-like patterns (email, URL, phone-like digit runs); no consent, anonymization, or PII-scrubbing statement was found upstream.
- 6 records repeat a conversation identifier; the identifier is not a unique key in this subset.
- 11 records share an identical ordered speaker/text payload, ignoring annotations and task identifiers; a duplicate policy is required.
- 1038 `(deleted)` placeholder utterances remain in the source; a documented repair or exclusion rule is required.
- 55 utterances use missing or non-USER/ASSISTANT speaker labels.
- Conditional decision: the subset is potentially useful, but the listed issues need explicit repair/exclusion and privacy decisions before training.

### Counts

| Metric | Value |
|---|---:|
| Records read | 5507 |
| Conversations parsed | 5507 |
| Valid / flagged / invalid | 440 / 5063 / 4 |
| Malformed records | 0 |
| Messages | 133597 |
| Characters | 5212194 |
| Annotated utterances | 42924 |
| Non-ASCII characters | 362 |
| CJK characters | 0 |

### Speaker roles

| Original label | Utterances |
|---|---:|
| `<missing>` | 55 |
| `ASSISTANT` | 73195 |
| `USER` | 60347 |

Mapped roles:

- `assistant`: 73195
- `user`: 60347

Unmapped speaker labels:

- `<missing>`: 55

### Domains

| Domain | Conversations |
|---|---:|
| `auto_repair` | 699 |
| `coffee_ordering` | 766 |
| `movie_tickets` | 1037 |
| `pizza_ordering` | 970 |
| `restaurant_reservation` | 1031 |
| `ride_service` | 1004 |

### Instruction identifiers

| Instruction ID | Conversations |
|---|---:|
| `auto-repair-1` | 361 |
| `auto-repair-2` | 338 |
| `coffee-ordering-1` | 766 |
| `movie-tickets-1` | 424 |
| `movie-tickets-2` | 613 |
| `pizza-ordering-1` | 970 |
| `restaurant-table-1` | 632 |
| `restaurant-table-2` | 399 |
| `uber-lyft-1` | 1004 |

### Conversation lengths

| Measure | Min | Median | P90 | Max |
|---|---:|---:|---:|---:|
| Messages | 0 | 23 | 36 | 148 |
| Characters | 0 | 879 | 1464 | 5534 |
| Utterance characters | 1 | 31 | 80 | 747 |

### Structural findings

| Finding | Records affected |
|---|---:|
| `adjacent_duplicate_utterance` | 199 |
| `deleted_placeholder` | 744 |
| `duplicate_conversation_id` | 6 |
| `duplicate_payload` | 11 |
| `empty_conversation` | 4 |
| `missing_speaker` | 13 |
| `potential_contact_data` | 948 |
| `same_role_adjacency` | 4972 |

Occurrence counts: deleted placeholders 1038, same-role adjacencies 23628, adjacent duplicate utterances 1410, empty conversations 4.

Utterance-level gaps:

- non-object utterances: 0
- missing text: 0
- empty text: 0
- missing index: 0
- non-sequential index: 0
- missing speaker: 55
- unknown speaker: 0

### Privacy signals

| Pattern | Utterances |
|---|---:|
| Email-like | 0 |
| URL-like | 1 |
| Phone-like digit runs | 1784 |
| Long digit runs (13+) | 63 |
| Any pattern | 1785 |

Conversations affected: 948.

### Budget fit (applicability, not a source defect)

| Budget | Conversations |
|---|---:|
| 8 messages / 500 characters | 81 |
| 8 messages / 1000 characters | 81 |
| 16 messages / 2000 characters | 967 |
| 32 messages / 4000 characters | 4614 |

## Duplicate records

Payload duplicates compare ordered speaker/text pairs; annotations, utterance indices, and task identifiers are ignored. Identifier duplicates are reported separately.

### Repeated conversation identifiers

| Conversation ID | Occurrences | Locations |
|---|---:|---|
| `dlg-fdd242eb-56be-48c0-a56e-5478472500d0` | 6 | woz-dialogs.json:1499324, woz-dialogs.json:1499595, woz-dialogs.json:1499846, woz-dialogs.json:1500137, woz-dialogs.json:1500418, woz-dialogs.json:1500699 |

### Repeated utterance payloads

| Payload SHA-256 | Occurrences | Locations |
|---|---:|---|
| `0602f0a3155b726c231a78ba10ea5e833c8f609d2a02f4a66f7af5160d02d92e` | 5 | woz-dialogs.json:133609, woz-dialogs.json:167816, woz-dialogs.json:462748, woz-dialogs.json:939864, woz-dialogs.json:1236788 |
| `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` | 4 | woz-dialogs.json:389979, woz-dialogs.json:819413, woz-dialogs.json:1300617, woz-dialogs.json:1412005 |
| `6a9b6606b9c068c6c9660badd8bebb991421c33341a7b818e78fc95062e3ac15` | 2 | woz-dialogs.json:953842, woz-dialogs.json:1111181 |

## Split coverage

| Split | Rows | Unique IDs | Duplicate IDs | Malformed rows |
|---|---:|---:|---:|---:|
| `dev` | 770 | 770 | 0 | 0 |
| `test` | 770 | 770 | 0 | 0 |
| `train` | 6168 | 6168 | 0 | 0 |

Self-dialog IDs: 7708; IDs in split lists: 7708; covered: 7708; missing from splits: 0; extra in splits: 0.

## Privacy and content concerns

- Pattern scans are heuristics; they do not prove the presence or absence of personal data.
- No consent, anonymization, or PII-scrubbing statement was found in the official README or the Taskmaster-1 paper.
- The WOZ subset contains upstream `(deleted)` placeholders and blank speaker labels, which indicate redaction or transcription gaps rather than a project repair.
- Reports and manifests contain counts and identifiers only; conversation text is never written.

## Limitations

- The inspector reads extracted directories only; archive extraction is left to standard tools.
- Contact-data detection is a pattern scan, not a PII classifier.
- The candidate manifest is not a training dataset and does not authorize training.
- The license summary is not legal advice.

## Unresolved questions

- Are contact-like strings actual personal data, and does the project require redaction before training?
- What repair policy should apply to WOZ blank speakers, duplicate identifiers, same-role adjacency, and deleted placeholders?
- Should the sequence budget grow to admit Taskmaster-1 conversations, or should only the few short conversations be used?
- What attribution text and placement will satisfy CC BY 4.0 in model cards and redistributed artifacts?
- Can third-party content embedded in the published dialogs be ruled out?
- Who gives final training approval, and on what evidence?
