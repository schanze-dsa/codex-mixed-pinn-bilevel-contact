# P2 Stress Branch Design

## Goal

Implement the first P2 architecture upgrade for mixed PINN by replacing the current late stress head with:

- an earlier split between displacement and stress branches
- an explicit `epsilon(u)`-guided bridge into the stress head

This is the PDF's recommended P2-Phase A. It is meant to make the stress path more physically grounded before adding contact semantics or high-frequency bypasses.

## Scope

In scope:

- early split between `u_branch` and `stress_branch`
- explicit `epsilon(u)` bridge feeding stress prediction
- feature/config switches to enable the new path without breaking old runs
- mixed residual compatibility through existing `u_fn`, `us_fn`, `sigma_fn`

Out of scope:

- contact semantic features such as `n`, `t1`, `t2`
- stress pointwise/high-frequency bypass as the main new path
- full P2 semantic-feature ablations
- full-IFT integration work

## Design Summary

The current model still behaves like a shared trunk with a late stress head hanging off the final feature tensor. That is too coupled for contact-heavy mixed training:

- displacement wants smoother, more global structure
- stress near contact wants more local, more sensitive structure

The proposed change keeps the shared trunk, but stops forcing both heads to share the same late representation.

## Architecture

### Shared Trunk Stays

The shared trunk remains useful for:

- geometry context
- preload/stage conditioning
- coarse global deformation structure

P2-Phase A does not delete the trunk and does not fork the entire model.

### Early Split

After the mid-to-late shared representation, split into:

- `u_branch`
- `stress_branch`

`u_branch` keeps the current displacement-facing role and should preserve the existing `u_fn()` and `u_fn_pointwise()` behavior.

`stress_branch` becomes an actual branch, not just a final dense layer on top of the same terminal `hfeat`.

### Epsilon(u) Bridge

The stress branch should no longer depend only on shared learned features. It should receive an explicit physics bridge:

- compute `u(x)` from the displacement branch
- differentiate `u(x)` with respect to `x`
- assemble engineering strain / strain-like bridge features
- concatenate those bridge features with stress-side branch features

The key objective is to make stress prediction explicitly aware of the strain field rather than relying on constitutive residuals alone to align the head after the fact.

## File-Level Plan

### `src/model/pinn_model.py`

Main implementation site.

Required direction:

- add config-gated early stress split
- add config-gated `epsilon(u)`-guided stress head
- keep existing interfaces stable:
  - `u_fn`
  - `u_fn_pointwise`
  - `us_fn`
  - `us_fn_pointwise`
  - `sigma_fn`

The old behavior should remain the default when the new flags are off.

### `src/physics/elasticity_residual.py`

This module already has the right mixed semantics:

- `constitutive_residual(u_fn, sigma_fn, params)`
- `equilibrium_residual(sigma_fn, params)`

P2-Phase A should preserve these APIs and make sure the new model path remains compatible with them. The residual code should not need a conceptual rewrite here.

### `src/physics/traction_utils.py`

No architectural rewrite in this step. Reuse canonical traction projection exactly as-is so P2-Phase A isolates the stress prediction change.

## Configuration

Add explicit flags so old runs remain stable:

- `stress_branch_early_split`
- `use_eps_guided_stress_head`

Suggested behavior:

- default `False`
- only explicit P2 configs turn them on

This keeps regression risk controlled and makes ablation easy.

## Training / Migration Strategy

Follow the PDF's P2-Phase A ordering:

1. keep the strict mixed / P1 contact path unchanged
2. replace only the stress architecture
3. warm up the new stress path with stronger constitutive alignment pressure
4. do not simultaneously add semantic features or high-frequency bypasses

This isolates the source of any gain or regression.

## Testing Strategy

### Model tests

- with flags off, existing behavior stays unchanged
- with flags on, `us_fn()` still returns valid `(u, sigma)`
- pointwise stress path still works

### Mixed residual tests

- `constitutive_residual()` remains numerically valid
- `equilibrium_residual()` still accepts `sigma_fn`
- output shapes and basic stability remain intact

### Regression tests

- old config path remains green
- new config path is at least numerically well-formed

## Acceptance Criteria

P2-Phase A is successful when:

- old runs remain unchanged with flags off
- the new stress path is trainable with flags on
- constitutive residual behavior is at least as stable as before
- traction matching is not worse than the current late-head baseline

If those hold, the project is ready for P2-Phase B:

- contact semantics
- then later high-frequency / pointwise stress enhancements

## Risks

Main risks:

- stress path becomes too expensive if the `epsilon(u)` bridge is computed everywhere
- mixed training graphs become harder to stabilize if bridge logic is enabled in all code paths

Mitigation:

- enable the bridge only on stress-returning mixed paths
- leave displacement-only paths unchanged
- keep the new path behind explicit config flags
