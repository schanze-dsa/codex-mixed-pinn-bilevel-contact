# Supervision Log Design (2026-03-09)

## Context
- `E_data` already participates in the loss when `X_obs/U_obs` are present.
- Current training step logs only show physics and uncertainty terms.
- Users cannot confirm from the step log whether supervision is active or how large the supervision error is.

## Goal
Expose supervision information in the per-step training log without making unsupervised runs noisier.

## Design
### Energy summary
- Extend the existing energy summary to include `E_data` with the label `Edata`.
- Reuse the existing per-term weight lookup so the displayed weight reflects adaptive updates.

### Train postfix
- Read `data_rms` and `data_mae` from the step stats map.
- Append `drms=...` and `dmae=...` only when those stats exist.
- Keep unsupervised runs unchanged when the supervision stats are absent.

### Testing
- Add a unit test that verifies `_format_energy_summary()` includes `Edata=...(w=...)`.
- Add a unit test that verifies `_format_train_log_postfix()` appends `drms` and `dmae` when supervision stats are present.
