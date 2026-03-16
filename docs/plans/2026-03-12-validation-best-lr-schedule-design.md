# Validation-Best Checkpoint And Late LR Decay Design

## Problem

The latest supervised run no longer has the old global amplitude blow-up, but the training loop still picks checkpoints using only training-side `Pi`/`E_int`, and the optimizer uses a fixed learning rate for the full run.

Observed behavior:

- Supervision quality improves materially by the end of training, but not monotonically.
- `drrms` has large mid-run spikes and later recovers.
- Contact penetration keeps drifting upward even when supervision error improves.
- The current "best checkpoint" logic cannot prefer the checkpoint with the best supervised validation behavior because it never measures validation during training.

This means the trainer can keep the numerically lowest training loss while missing the checkpoint that best suppresses speckle on held-out supervised cases.

## Goal

Make the conservative anti-speckle path practical without redesigning the whole trainer:

- periodically evaluate a fixed validation supervision metric during training
- save the best checkpoint by validation `drrms`
- reduce the learning rate late in training when validation stops improving

## Chosen Scope

Implement only these two changes:

1. Validation-driven best checkpoint selection
2. Late learning-rate decay driven by validation stagnation

Do not add multi-case batching, gradient accumulation, or spatial smoothing in this change.

## Design

### Validation Metric

Reuse the existing staged supervision evaluation path already used after training.

For the configured validation split:

- build staged prediction rows with `_build_supervision_eval_rows`
- compute a compact scalar summary:
  - `val_rmse_vec_mm_mean`
  - `val_ratio_median = median(pred_rms_vec_mm / max(true_rms_vec_mm, eps))`
  - `val_drrms_mean = mean(rmse_vec_mm / max(true_rms_vec_mm, eps))`

`val_drrms_mean` is not identical to the training-step pointwise `drrms`, but it is the closest held-out analogue available from the existing evaluation rows because it uses vector-field RMSE normalized by the true displacement RMS. The metric must still be labeled as validation-only in logs.

The best-checkpoint metric should default to validation ratio quality, not training `Pi`.

### Checkpoint Policy

Add support for `save_best_on: val_drrms`.

At each logging interval:

- run validation evaluation if supervision is enabled and validation cases exist
- compute the configured metric
- compare against `best_metric`
- save checkpoint when the validation metric improves

Preserve existing behavior for `Pi` and `E_int`.

### Learning-Rate Decay

Add a simple plateau scheduler stored in trainer runtime state:

- monitor the same validation metric used for checkpointing
- after a configurable warmup
- if the metric fails to improve for `patience` validation checks
- multiply LR by `factor`
- clamp at `min_lr`

This should update the actual optimizer learning rate in-place so no optimizer rebuild is needed.

### Logging

Extend the printed training postfix with compact validation fields when available:

- `vdr=...` for validation mean RMS ratio metric
- `vrat=...` for validation median amplitude ratio
- `vlr=...` for current learning rate

Also print a short note when LR is decayed.

## Why This Option

- Smallest change with the highest leverage on speckle selection
- Reuses existing supervision-eval code instead of inventing a parallel validation pipeline
- Avoids conflating training loss descent with held-out field quality
- Lets longer runs help when they should, but reduces late-run noise chasing

## Rejected Alternatives

### Only increase max steps

This can improve some cases, but current logs already show that longer training also worsens contact-side metrics. Without validation-based selection, a longer run can just overwrite a better earlier checkpoint.

### Add smoothing regularization first

That changes the model objective and can suppress both noise and real gradients. It is a follow-up option, not the first conservative step.

### Implement mini-batching first

Likely useful, but it is a broader training-path change. Validation-best selection and LR decay are lower-risk and easier to verify.

## Implementation Scope

- Extend `TrainerConfig` with validation-checkpoint and LR-decay knobs.
- Parse the new config keys in `main new.py`.
- Add trainer helpers to compute validation supervision summaries during training.
- Extend checkpoint selection in `TrainerRunMixin`.
- Add plateau LR scheduler state/update hooks.
- Add tests for:
  - validation metric aggregation
  - `save_best_on=val_drrms`
  - LR decay after validation stagnation

## Non-Goals

- No changes to loss definitions
- No smoothing-only visualization masking
- No multi-case batch training in this patch
