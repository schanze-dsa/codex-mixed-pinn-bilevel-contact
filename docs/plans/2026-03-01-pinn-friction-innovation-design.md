# PINN Friction Innovation Design (2026-03-01)

## Context
- Project: frictional-contact DFEM/PINN for mirror deformation prediction from CDB geometry and bolt tightening path.
- User goal: prioritize full-field mirror deformation accuracy (`Total Deformation`), while making the method publishable with stronger novelty.
- Constraints:
  - Existing codebase already has residual/energy physics terms, contact ALM, staged preload, graph-based displacement model, and mirror visualization.
  - Ground truth is sparse in number of cases but full-field per case (`txt` exports).

## Target Contribution Set (5 points)
1. Incremental energy-dissipation consistency for frictional loading path.
2. Differentiable friction-contact unified constraints (normal complementarity + tangential friction consistency).
3. Finite-domain spectral encoding for geometry representation.
4. CDB engineering semantic encoding (contact/bc/material tags).
5. Physics-residual calibrated uncertainty for trustworthiness.

## Design Decisions
### A. Optimization Scope in This Iteration
- Implement a safe, testable first milestone that upgrades core model inputs and training diagnostics without breaking existing training:
  - Add finite spectral encoding hooks in `pinn_model`.
  - Add CDB semantic feature injection hooks in `pinn_model` and compute semantic node features in `trainer`.
  - Add uncertainty head hooks and inference API in `pinn_model`.
  - Add lightweight uncertainty calibration utility driven by residual proxies.
- Keep existing contact/training pipeline backward-compatible.
- Defer heavy algorithmic changes (full incremental energy-dissipation loss and bipotential friction loss reformulation) to next milestone after baseline verification.

### B. Data/Feature Flow
1. `trainer.build()` loads CDB and constructs sorted global nodes.
2. Trainer computes per-node semantic features from assembly data:
   - contact flag
   - boundary-condition flag
   - mirror-surface/part flag
   - normalized material id
3. Trainer attaches semantic features to `model.field`.
4. `DisplacementNet.call()` concatenates:
   - node embedding (DFEM mode) or positional encoding
   - finite spectral encoding
   - semantic features (if present and shape matches)
   - preload condition vector `z`

### C. Uncertainty Design
- Add optional aleatoric head for displacement log-variance.
- Add `uvar_fn(X, params)` returning `(u_mean, log_var)`.
- Add calibration utility:
  - inputs: predicted sigma, residual proxy
  - output: calibrated sigma via monotone affine map
- No forced loss coupling in this iteration; hooks are added for controlled activation.

### D. Backward Compatibility
- All new features disabled by default.
- Existing config behavior remains unchanged when new keys are absent.
- No breaking changes to saved checkpoints unless new heads are explicitly enabled.

## Validation Plan
- Smoke checks:
  - model creation with old config (no new keys)
  - model creation with finite spectral + semantic enabled
  - uncertainty head output shapes
- Runtime checks:
  - one build-only path to ensure semantic feature extraction does not crash.
- Regression checks:
  - existing fast scripts (`test_config_read.py`, `test_dfem.py`) continue to run.

## Risks and Mitigations
- Risk: semantic features misaligned with node ordering.
  - Mitigation: compute with trainer/elasticity sorted node ids and strict shape checks.
- Risk: added features destabilize training.
  - Mitigation: default-off flags, small feature dimensions, optional normalization.
- Risk: uncertainty head hurts convergence if coupled too early.
  - Mitigation: provide API only first, keep training loss unchanged until calibrated.

