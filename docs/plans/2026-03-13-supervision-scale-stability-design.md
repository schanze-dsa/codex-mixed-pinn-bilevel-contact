# Supervision Scale Stability Design

## Problem

The latest anti-speckle run improved field smoothness, but the old amplitude problem reappeared in the exported PINN/FEM comparisons.

Observed evidence from the current run `run-20260312-232551`:

- Validation ratio quality is best very early, then degrades badly:
  - `step 100`: `vrat=2.5226`
  - `step 1500`: `vrat=17.089`
- The exported `final` checkpoint is much worse than the saved early `best` checkpoint:
  - `final` test median ratio is about `16.83`
  - `best` test median ratio is about `2.57`
- Training logs show `dsmf=1` through the late run, meaning supervision keeps relying on the contribution floor rather than winning naturally.
- The current floor ratio is only `0.02`, so even when the floor activates, supervision is still allowed to stay much smaller than the weighted stress/contact terms.
- `E_data` is still included in adaptive focus terms, so the generic balancing logic continues to move the supervision weight during training.

This means the current trainer is doing two conflicting things:

1. it uses supervision to fight speckle and amplitude drift
2. it also allows the adaptive balancing logic to keep rescaling supervision relative to the physics terms

That combination is unstable for amplitude.

## Goal

Stabilize displacement amplitude around FEM while keeping the existing anti-speckle smoothing term.

The next change should:

- stop late-run amplitude drift
- preserve the current supervision loss normalization
- preserve the current smoothing regularizer
- keep adaptive balancing for the physics-side terms
- avoid redesigning the training route, data path, or visualization path

## Chosen Scope

Implement exactly two behavioral changes:

1. remove `E_data` from adaptive focus balancing
2. raise the supervision contribution floor to a materially stronger default

Do not change:

- supervision dataset
- smoothing loss definition
- visualization/export logic
- optimizer type
- contact route

## Design

### 1. Stop Adaptive Reweighting Of Supervision

`E_data` should no longer participate in `loss_focus_terms`.

Adaptive balancing is still useful for the physics-side terms such as `E_cn`, `E_ct`, `E_eq`, and the auto-added `E_sigma`, but the supervision term is different:

- it is the only direct amplitude anchor
- it already has a physically meaningful relative normalization
- when it is reweighted by the generic balance logic, the trainer can drift away from the desired displacement scale even while logs still show a nonzero data term

Implementation intent:

- keep the existing base supervision weight `w_data`
- keep using the same relative supervision loss `E_data`
- exclude `E_data` from the configured adaptive focus-term tuple before creating the runtime loss state

This should make supervision weight behavior easier to reason about:

- `w_data` stays at its configured/base level
- the floor can still increase its effective contribution when needed
- the adaptive scheduler remains free to balance the physics-side terms

### 2. Strengthen The Supervision Contribution Floor

The current floor ratio `0.02` is too weak for amplitude stability.

The existing floor logic already works at the correct level: post-weight effective contribution. That mechanism should stay, but the default target needs to be stronger.

The intended policy remains:

- `C_data = w_data_eff * E_data`
- `C_phys = w_sigma_eff * E_sigma + w_ct_eff * E_ct`
- enforce `C_data >= rho * C_phys`

The change is to raise the default `rho` from `0.02` to `0.1` for the supervised ANSYS config.

Why this level:

- it is large enough to matter
- it is still clearly below the physics-side dominant contribution
- it gives supervision a real chance to anchor amplitude without fully taking over the objective

### 3. Keep Smoothing As-Is For This Change

The current `E_smooth` term already did the job it was introduced for: it reduced speckle.

This patch is not the place to retune smoothing. Changing smoothing and amplitude controls simultaneously would make the next run harder to interpret.

So this change intentionally keeps:

- `w_smooth`
- `data_smoothing_k`
- the same supervision-only smoothing definition

### 4. Logging Expectations

No new logging family is required.

The existing metrics are already sufficient to judge whether this patch works:

- `drrms`
- `vdr`
- `vrat`
- `dsmw`
- `dsmf`

Expected post-change behavior:

- `vrat` should stop exploding late in training
- `dsmf` should trigger less often or at least do less rescue work
- `dsmw` should become easier to interpret because adaptive balancing is no longer moving `w_data`

## Alternatives Considered

### Only Increase The Floor

Rejected as the primary fix because the adaptive balancer would still be allowed to keep shrinking or distorting supervision weight behavior. That treats the symptom, not the cause.

### Two-Stage Training

Rejected for this round because it changes training dynamics too broadly. Useful later if this smaller patch is still insufficient.

### Change `output_scale`

Rejected because it would only mask the scale issue at the network-output level, not fix the objective imbalance.

## Risks

- If the stronger floor is too aggressive, the trainer may over-prioritize supervised displacement and hurt contact/stress quality.
- If the current `best` checkpoint still occurs very early after this patch, then scale stability is still not solved and the next step should be training-stage scheduling rather than further loss-weight tweaks.

## Success Criteria

This patch is successful only if all of the following hold after retraining:

- `best` and `final` ratio metrics are materially closer to each other
- `vrat` no longer blows up late in training
- same-scale PINN/FEM comparisons no longer show a globally oversized PINN field
- speckle does not regress materially versus the current anti-speckle run

## Non-Goals

- No batching redesign
- No new validation metric
- No change to exported artifact structure
- No visualization-side smoothing changes
