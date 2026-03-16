# Anti-Speckle Training Design

## Problem

The new same-pipeline FEM/PINN comparison confirms the remaining speckle is not a visualization artifact.

Observed evidence from the latest run:

- The same training case still shows strong PINN speckle even when FEM and PINN go through the exact same `mirror_viz.py` rendering path.
- Spatial analysis shows the PINN field carries a nearly global high-frequency noise floor rather than a single bad angular sector.
- The worst roughness hotspots concentrate near the inner and outer ring boundaries.
- Training logs show the weighted supervision contribution remains much smaller than the weighted stress/contact contributions, even after supervision-loss rescaling.

This points to a training objective problem, not a plotting problem:

1. the optimizer still prioritizes `E_sigma` and `E_ct` over local displacement-field fidelity
2. nothing in the objective explicitly suppresses high-frequency displacement noise on supervised mirror nodes

## Goal

Reduce PINN speckle at the field level without redesigning the training route.

The change should:

- preserve the current supervision amplitude fix
- prevent supervision from being drowned out by stress/contact terms
- add a small explicit penalty against local high-frequency displacement noise
- avoid changing data loaders, graph construction, or visualization code

## Chosen Scope

Implement exactly two training-side changes:

1. a supervision effective-contribution floor after adaptive weighting
2. a light spatial smoothing regularizer on supervised displacement predictions

Do not modify:

- sampling route
- case batching
- visualization smoothing
- graph cache construction

## Design

### 1. Supervision Effective-Contribution Floor

The current issue is not the raw `E_data` definition anymore. It is the *post-weight* contribution.

The floor should therefore be applied after adaptive weights are computed.

Define:

- `C_data = w_data_eff * E_data`
- `C_phys = w_sigma_eff * E_sigma + w_ct_eff * E_ct`

Then enforce:

- `C_data >= rho * C_phys`

by increasing the effective supervision weight used in the current step only.

Important details:

- This should not rewrite the stored adaptive scheduler state.
- It should only affect the weight vector used to combine the current-step losses.
- It should be skipped when supervision is absent or `E_data <= 0`.
- It should expose diagnostics so logs can show whether the floor is active and what effective data weight was used.

Why this form:

- It targets the actual imbalance visible in logs.
- It is robust to scale changes in `E_data`, `E_sigma`, and `E_ct`.
- It avoids hard-coding a global `w_data` that may be wrong for every case.

### 2. Light Spatial Smoothing Regularizer

Add an explicit high-frequency penalty on supervised mirror nodes:

- evaluate `u_pred` on `X_obs`
- build a small kNN neighborhood in the observed mirror-node coordinates
- penalize deviation from the local neighbor mean

Recommended form:

- `E_smooth = mean(||u_i - mean(u_neighbors)||^2) / data_ref_rms^2`

Normalization by `data_ref_rms` is required so the term remains comparable across low- and high-amplitude cases.

Important details:

- Only compute this when supervision is present.
- Keep it separate from visualization smoothing; this is a real training loss term.
- Use the supervised node coordinates already present in `params["X_obs"]`, not assembly-wide nodes.
- Default to a small `k` and a small scalar weight.

Why this form:

- The diagnosed speckle is high-frequency vector-field noise.
- The noise is strongest near boundaries but not restricted to one sector, so a local isotropic smoothing prior is appropriate.
- This can suppress the noise floor without forcing a new data path or mesh preprocessing step.

### 3. Logging and Diagnostics

Add compact stats to confirm the fix is doing what we intend:

- `dsmw` or similar: effective supervision weight after the floor
- `dsmf=1` when the floor is active
- `smrms`: smoothing residual RMS before weighting

This is required because the root problem was only visible after looking at weighted contributions, not raw terms.

## Recommended Hyperparameter Strategy

Start conservative:

- contribution floor ratio `rho`: small but non-trivial
- smoothing weight: small enough that the field shape is still driven by supervision and physics
- kNN size: local only

This patch is meant to reveal whether the objective is missing exactly these two constraints, not to fully retune the trainer in one shot.

## Alternatives Considered

### Raise fixed `w_data` globally

Rejected because the current problem is step-wise post-weight imbalance, not just the base scalar in config.

### Add only smoothing regularization

Rejected because if `E_data` is still too weak, the model can remain locally noisy while the smoothing term competes against dominant stress/contact terms.

### Change mini-batching or gradient accumulation first

Likely useful later, but too broad for the current diagnosis loop.

## Risks

- Too aggressive a floor can overfit supervised displacement at the expense of contact/stress consistency.
- Too aggressive smoothing can erase real local gradients and make the field look artificially clean.
- Because hotspots cluster near boundaries, the smoothing term must remain light; otherwise it may bias edge behavior.

## Success Criteria

The patch is successful only if all of the following improve together:

- same-pipeline roughness ratio PINN/FEM drops materially
- `val_drrms` does not regress materially
- amplitude ratio metrics do not return to large blow-up behavior

## Non-Goals

- No contact-model redesign
- No graph-neighborhood redesign
- No new visualization masking
- No full training-route rewrite
