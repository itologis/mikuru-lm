# ADR 0005: Deterministic mini-batch training

- **Status:** Accepted
- **Date:** 2026-11-18

## Context

Version 0 trained on one padded batch containing every training record, once
per epoch. The OASST1 v2 runs therefore performed 50 and 77 optimizer updates
respectively (one per epoch), at a learning rate of 1e-3, while validation loss
was still improving. The records were never shuffled and the trainer could not
scale to a larger corpus, a longer context, or a wider model.

The pipeline needs a training loop that performs many updates per pass, visits
every record according to a stated policy, and records enough measurement to
compare against the preserved full-batch baselines. Existing checkpoints and
their metadata must keep loading unchanged.

## Decision

1. **Mini-batching is opt-in.** `TrainerConfig::batch_size = None` keeps the
   legacy path: one padded batch per epoch, dataset order, no shuffle.
   `--batch-size N` enables the new path. No existing command changes meaning.
2. **Shuffle policy is deterministic and documented.** Each epoch uses
   Fisher-Yates over the record indices, driven by SplitMix64 seeded with
   `seed ^ (epoch * 0x9E3779B97F4A7C15)`. Every record appears exactly once per
   epoch; the same seed and epoch reproduce the same order. No new dependency.
3. **One optimizer update per batch.** The final batch may be shorter and is
   still used; batch boundaries are plain contiguous chunks of the epoch order.
4. **Loss is a token-weighted mean.** Each batch contributes its cross-entropy
   scaled by its number of supervised positions (`sequence_len - 1`), so a short
   final batch is not over-weighted. Epoch-level reporting uses the same unit as
   the legacy path.
5. **Initial loss is measured before any update**, over the full training split.
   On the legacy path this equals the old first-epoch forward pass.
6. **Validation is unchanged.** Forward pass only, after each epoch, never
   passed to the optimizer. Early stopping keeps its existing patience and
   minimum improvement.
7. **Metrics are recorded in checkpoint metadata.** New optional fields:
   `batch_size`, `batching_policy`, `steps_per_epoch`, `total_steps`,
   `step_time_ms_mean/max`, `epoch_time_s_mean/max`, `peak_rss_kb`, `swap_kb`,
   `cpu_utilization_percent`. `format_version` stays 1; absent fields mean an
   older or full-batch run.
   Measurement-only additions (2026-11-19): `initial_validation_loss`,
   `total_train_time_s`, `train_loss_history`, `validation_loss_history`, and
   per-update `step_losses` (`len == total_steps`). These add forward passes and
   logging only; they do not alter optimizer updates.
8. **Metric definitions are explicit.** Peak RSS is `VmHWM`. Swap is a
   point-in-time `VmSwap` (the kernel exposes no per-process peak). CPU
   utilisation comes from `/proc/self/stat` utime+stime and assumes Linux
   `USER_HZ = 100`; the assumption is stated in the printed report.
9. **Checkpoint behaviour is unchanged.** Overwrite refusal, best-validation
   saving, and reload verification stay as they are. Continuation runs inherit
   `batch_size` from the source checkpoint metadata.

## Acceptance checks

| Requirement | Where satisfied |
| --- | --- |
| Documented deterministic shuffle/sampling policy | Decision 2, `shuffled_indices` doc comment |
| Batch boundaries and final partial batch tested | `batch_ranges_cover_every_record_once_with_a_short_final_batch` |
| Loss averaged consistently across batches | Decision 4, token-weighted accumulation in the epoch loop |
| Validation without optimizer updates | Decision 6, existing `validation_evaluation_is_read_only` |
| Gradients and optimizer updates per mini-batch | Epoch loop calls `backward` + `optimizer.step` inside the batch loop |
| Reproducibility with a fixed seed | `shuffle_is_deterministic_and_visits_every_record_once` |
| Peak RAM/swap, step/epoch time, CPU utilization recorded | Decision 7–8, verified by the mini-batch integration test |
| Original checkpoints unchanged | No artifact path is written; `verify-checkpoint` regression check |

## Consequences

- Epoch counts are no longer comparable across batching modes. Compare runs by
  **total optimizer steps** or wall time, not by epoch number.
- The legacy path remains the default, so the 50/100-epoch baselines stay
  reproducible in behaviour and their checkpoints remain valid.
- Burn Flex 0.22.1 still ignores `Backend::seed`, so weight initialisation is
  not reproducible. Fixed-seed reproducibility covers split and batch order
  only; this limitation is unchanged and must be stated in experiment notes.
- Metrics are Linux-specific. On other platforms the fields are absent rather
  than fabricated.
- Mini-batching does not by itself create conversational quality. It removes the
  optimizer-step bottleneck and is a prerequisite for corpus, context, and
  capacity experiments.
