# Supervision Loss Scaling Design

## Problem

The staged ANSYS supervision path is connected, but its numeric contribution is too small to shape the learned displacement field.

Observed evidence from the current run:

- `E_data` decays to about `1e-6`, while contact and residual terms remain many orders larger.
- For `C073 stage3`, the predicted displacement RMS is about `10x` the FEM RMS.
- For the same case, predicted surface roughness is about `133x` the FEM roughness.
- `u_fn` and `u_fn_pointwise` produce the same rough field on the supervised surface, so the speckle is not introduced by visualization routing.

## Root Cause

The current data supervision term uses raw displacement MSE in `mm^2`:

`mean((u_pred - u_obs)^2)`

The rigid-removed FEM targets are often only `1e-4 mm` to `1e-3 mm` in magnitude. That makes even materially bad predictions produce a very small scalar loss, so `w_data` cannot compete with the contact, stress, and residual terms.

This is a scaling failure, not a missing-loss failure.

## Chosen Fix

Normalize supervision residuals by the observed displacement RMS of the current supervised batch:

`u_ref = sqrt(mean(u_obs^2))`

`E_data = mean(((u_pred - u_obs) / max(u_ref, eps))^2)`

This makes supervision relative to the case amplitude instead of absolute millimeter scale.

## Why This Option

- Minimal behavioral change: only the supervision term changes.
- Directly targets the identified root cause.
- Keeps existing `w_data` semantics intact.
- Works for staged supervision because each stage already carries its own `U_obs`.

## Rejected Alternatives

### Only increase `w_data`

This is fragile because the correct multiplier depends on target amplitude and unit choices. It would likely drift again when the dataset or rigid-removal pipeline changes.

### Add extra smoothing regularization first

That suppresses the visible symptom but does not fix why supervision is numerically drowned out. It should remain optional follow-up work, not the first repair.

## Implementation Scope

- Update `TotalEnergy._compute_data_supervision_terms` to compute relative supervision loss.
- Export additional stats for monitoring:
  - `data_ref_rms`
  - `data_rel_rms`
  - `data_rel_mae`
- Extend train-log formatting to print the relative metrics.
- Add regression tests proving:
  - equal relative error at different amplitudes yields equal `E_data`
  - exact supervision still gives zero loss
  - logs include the new relative metrics when present

## Non-Goals

- No new spatial smoothing regularizer in training.
- No retraining policy change.
- No visualization-only masking to hide speckle.
