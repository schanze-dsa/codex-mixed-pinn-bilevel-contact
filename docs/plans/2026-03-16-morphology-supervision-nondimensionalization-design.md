# Morphology-Aware Supervision And Nondimensionalization Design

## Problem

The current supervised route still treats displacement supervision as a global average relative pointwise error:

- `E_data` is a uniform mean of relative displacement errors over all observed points.
- `E_smooth` penalizes local deviation from neighbor means and therefore favors smoother fields.
- staged supervision is currently implicit across multiple snapshots, not explicit on the stage-to-stage displacement deltas.
- physical scaling exists only in partial form (`output_scale`, preload feature scaling, `sigma_ref`), so coordinates, displacements, and physics residuals do not yet live in one fully consistent nondimensional space.

This creates two practical failures:

1. local peaks, edges, and high-contrast morphology can be diluted by the many ordinary regions in the global average loss
2. stage-to-stage behavior can be approximately fit in aggregate while still missing the actual displacement increments that matter visually and physically

The user also confirmed that the current "best" supervision level is still not good enough, so simply preserving the current best checkpoint is not sufficient. The default training route itself must become more morphology-aware and more scale-consistent.

## Goal

Make the default supervised training route better match the observed local morphology and staged evolution while improving scale stability through a complete nondimensionalization path.

The next change should:

- make supervision emphasize important local morphology by default
- add explicit supervision on staged displacement increments
- normalize coordinates and displacements under a single reference-scale system
- keep the current network architecture unchanged
- keep `E_smooth` semantics unchanged
- keep `E_data` out of adaptive focus routing
- become the new default main route rather than a hidden experimental toggle

## Chosen Scope

Implement the following as the new default supervised route:

1. replace uniform `E_data` averaging with default morphology-aware weighting
2. add an explicit `E_stage_delta` term for adjacent staged displacement increments
3. promote `L_ref`, `u_ref`, and `sigma_ref` into one connected nondimensionalization path used by supervision and physics
4. expose the final resolved scales and supervision-weight statistics in logs so correctness is inspectable at runtime

Do not change:

- the PINN architecture
- the supervision CSV schema
- the current `E_smooth` formula
- adaptive focus exclusion of `E_data`
- the two-stage orchestration design already adopted

## Design

### 1. Morphology-Aware Default Supervision

The supervised loss should keep pointwise displacement fitting as the main contract, but it should stop treating every observed point as equally important.

The new default supervision term is:

`E_sup = E_data_weighted + lambda_delta * E_stage_delta + E_smooth`

`E_data_weighted` remains a relative pointwise displacement error, but each observed point receives a default importance weight derived only from the observed supervision data:

- a displacement-magnitude component, so large-amplitude regions are not diluted by broad low-amplitude regions
- a local-contrast component, so points that differ strongly from their spatial neighborhood receive more attention

The weight field is clipped and normalized back to mean approximately `1.0` so it changes emphasis without silently exploding the total supervision scale.

This keeps the main contract simple:

- same observed points
- same pointwise comparison
- stronger default emphasis on peaks, edges, and other locally distinctive regions

### 2. Explicit Stage-Delta Supervision

Current staged supervision fits each stage snapshot, but it does not explicitly say that the model must match the displacement increment between neighboring stages.

The new `E_stage_delta` term compares:

- `Delta U_pred = U_pred(stage t+1) - U_pred(stage t)`
- `Delta U_obs = U_obs(stage t+1) - U_obs(stage t)`

for each adjacent stage pair.

This term directly constrains staged evolution instead of leaving it implicit inside multiple static snapshots. It is intended to improve:

- preload-sequence consistency
- shape evolution between stages
- visible differences that may otherwise be washed out by static pointwise averaging

The first version does not add curvature or higher-order derivative supervision. That was considered, but rejected for the initial implementation because it is more fragile and more likely to fight the existing smoothing term.

### 3. Full Nondimensionalization Path

The code already has partial scale controls, but they are not yet a unified nondimensionalization pipeline.

The default route should normalize:

- coordinates by `L_ref`
- displacements by `u_ref`
- stress-related quantities by `sigma_ref`

The intended path is:

- data loading converts observation coordinates and observed displacements into nondimensional tensors
- network training operates primarily in nondimensional coordinate/displacement space
- supervision losses are computed in nondimensional displacement space
- physics residual scaling aligns with the same `L_ref` / `u_ref` / `sigma_ref` system

`PhysicalScaleConfig` remains the source of truth, but it should be extended from a mostly stress-focused utility into a full reference-scale resolver with stable defaults and explicit runtime logging.

### 4. Data-Derived Feature Construction

The supervision dataset loader should precompute the derived supervision artifacts that are reused throughout training:

- nondimensional `X_obs_nd`
- nondimensional `U_obs_nd`
- per-point morphology weights
- adjacent-stage displacement deltas

These should be computed once during dataset loading instead of being rebuilt inside the loss on every step.

The feature derivation should rely only on existing information already present in the supervision dataset:

- observed coordinates
- observed displacements
- stage ordering already loaded from ANSYS snapshots

No new annotation files or manual ROI masks are required.

### 5. Runtime Observability

This change must not be a black box.

At training startup, logs should print:

- resolved `L_ref`, `u_ref`, and `sigma_ref`
- representative nondimensional coordinate range
- representative nondimensional displacement RMS or similar scale summary
- morphology-weight summary such as min / mean / max

During training, the new supervision stats should remain visible so it is possible to tell whether:

- weighted supervision is dominating too aggressively
- stage-delta supervision is active and decreasing
- the nondimensionalized supervision quantities remain in reasonable `O(1)` ranges

### 6. Acceptance Criteria

This change is only considered correct if all of the following can be demonstrated:

1. unit-level behavior
   - when the observed field is locally smooth, morphology weighting is close to uniform
   - when the same displacement error is placed at a high-contrast local feature, the weighted supervision term penalizes it more strongly than the old uniform mean
   - stage-delta mismatch is detectable even when static snapshots are otherwise similar
2. scale invariance
   - changing units consistently, such as meters to millimeters, does not materially change nondimensional supervision loss values once `L_ref` and `u_ref` are updated consistently
3. runtime observability
   - logs show resolved scales and supervision-weight summaries clearly
4. training-level evidence
   - short runs show both weighted data supervision and stage-delta supervision decreasing
   - local morphology in comparisons improves relative to the old route
   - staged drift is harder to reintroduce than before

## Alternatives Considered

### Keep Uniform `E_data` And Only Add Nondimensionalization

Rejected because it improves scale consistency but does not address the core issue that important local features are diluted inside a uniform global average.

### Add Full Gradient Or Curvature Supervision Immediately

Rejected for the first version because it is more invasive, more sensitive to neighborhood quality, and more likely to conflict with the current smoothing term.

### Hide The New Route Behind Experimental Config Flags

Rejected because the user explicitly asked for the new route to become the default main route rather than an opt-in experiment.

## Risks

- if morphology weights are too sharp, training may overfit sparse local features and destabilize broader field fitting
- if stage-delta supervision is weighted too strongly, it may distort static snapshot fitting
- if nondimensionalization is only partially wired in, the code may appear to normalize while actually leaving one branch in old units
- metric baselines will shift because the semantics of the default supervision path are changing

## Non-Goals

- no network architecture redesign
- no new supervision file format
- no manual ROI annotation workflow
- no higher-order geometry supervision in the first pass
- no visualization-only patch that leaves the loss unchanged
