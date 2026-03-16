# ANSYS Mirror Supervision Design (2026-03-08)

## Context
- Project: staged contact PINN trained from `mir111.cdb` geometry and bolt-tightening conditions.
- New data:
  - case table: `D:\shuangfan\pinn luowen-worktrees\ansys_cases_180_deg2to6_step0p5_pinn.csv`
  - stage labels: `D:\shuangfan\pinn luowen-worktrees\rigid_removed_csv\*.csv`
- User decision:
  - supervise all 3 stages
  - use rigid-body-removed mirror deformation as labels
  - align FE bolt numbering to PINN numbering via the derived `_pinn.csv` case table

## Goal
Keep the existing physics-driven PINN route, but add mirror-node displacement supervision for the 180 ANSYS cases so training becomes:

`physics loss + staged mirror-node data loss`

The ANSYS data are labels, not model inputs.

## Recommended Approach
### Option A: Runtime FE->PINN mapping in trainer
- Pros: keeps original case table untouched.
- Cons: mapping logic leaks into training, export, debugging, and validation.

### Option B: Convert case table once, then train directly on PINN-aligned cases
- Pros: simpler runtime, easier auditing, fewer hidden transformations.
- Cons: one extra derived CSV to maintain.

### Option C: Pure supervised surrogate model
- Pros: simplest optimization.
- Cons: discards current contact and tightening physics structure.

Recommendation: Option B.

## Design
### 1. Supervision Data Model
- Add an optional supervision config block to `TrainerConfig`.
- Load the derived PINN-aligned case table and stage CSV directory.
- Build a fixed-shape tensor per case:
  - `P`: `(3,)`
  - `order`: `(3,)`
  - `X_obs`: `(3, N, 3)` mirror node coordinates from `asm.nodes`
  - `U_obs`: `(3, N, 3)` rigid-removed displacements from ANSYS
- Filter training cases by `split=train` by default.

### 2. Sampling Behavior
- When supervision is enabled, training cases should come from the supervision dataset instead of random/LHS preload sampling.
- Reuse the existing staged preload path so the network still consumes the same conditional features.
- Preserve the old random path when supervision is disabled.

### 3. Stage Alignment
- FE labels contain exactly 3 cumulative stages.
- Current `force_then_lock` logic appends a release/final stage; that must become configurable.
- For supervised ANSYS training:
  - keep `preload_stage_mode=force_then_lock`
  - disable the appended release stage
  - use `stage_count=3`

### 4. Loss Integration
- Add a new scalar loss part `E_data`.
- For each active stage, evaluate `u_fn(X_obs, params_stage)` and compare to `U_obs`.
- Use MSE on `ux, uy, uz` directly:
  - `E_data = mean((u_pred - u_obs)^2)`
- Keep this inside `TotalEnergy` so logging and weight handling stay consistent with existing physics terms.

### 5. Logging and Compatibility
- Add `w_data` to `TotalConfig`.
- Expose per-stage `E_data` stats in the same style as the existing stage stats.
- Preserve backward compatibility:
  - no supervision config -> old behavior
  - no case files -> no data loss

## Risks
- Risk: stage tensors and supervision tensors drift out of sync.
  - Mitigation: keep supervision tensors inside the same staged params structure.
- Risk: current config still assumes 4 stage schedule entries.
  - Mitigation: use the actual inferred stage count and ignore extra schedule entries.
- Risk: mirror node ids from CSV do not exist in CDB.
  - Mitigation: validate all ids at load time and fail early with exact missing ids.

## Validation Plan
- Unit test supervision case loading and FE node-coordinate alignment.
- Unit test stage construction without appended release stage.
- Unit test `E_data` is zero for exact predictions and positive for mismatched predictions.
- Run focused trainer tests to verify staged extraction and SavedModel stage encoding remain stable.
