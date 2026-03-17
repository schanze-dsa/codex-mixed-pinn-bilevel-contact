# PDF Strict Mixed Experimental Design

## Goal

Implement the P0, P1, and P2 items from the user-provided PDF checklist without changing the default locked training route.

The repository should support two parallel routes:

- the existing locked route, still driven by the current default `config.yaml`
- a new experimental route, enabled only through a separate profile such as `strict_mixed_experimental.yaml`

The experimental route is the PDF target route. The default route remains the stable baseline.

## Approved Execution Route

The user approved `option 1`: keep the default locked route intact and drive the full PDF checklist through the `strict_mixed_experimental` route.

This means:

- `config.yaml` remains the stable locked baseline
- `strict_mixed_experimental.yaml` becomes the closure route for the PDF's item 1 through item 9
- legacy contact code may remain as a compatibility path, but the experimental route should treat it as an adapter boundary rather than the main implementation target

## Constraints

- Do not change the effective default behavior of [config.yaml].
- Do not silently widen the default route to include experimental features.
- Do not mix in the larger legacy contact rewrite from the older repository checklist unless explicitly requested later.
- Reuse existing strict mixed infrastructure where it already exists instead of creating parallel trainer or model implementations.

## Scope

In scope:

- P0 inner solver readiness, linearization plumbing, and trainer diagnostics
- P1 `ContactOperator` backend dispatch and typed strict mixed input contract
- P2 stress-head integration using `eps_bridge`, contact-surface semantics, and pointwise stress evaluation
- profile-aware config parsing and locked-route gating
- regression coverage for both the default route and the new experimental route

Out of scope:

- enabling the old checklist's larger `contact_rar`, `stage_resample_contact`, and `lbfgs` feature set in this pass
- changing baseline legacy ALM numerics for the default route
- full tangential IFT or a full strict stick-slip complementarity upgrade

## Architecture

### Dual Route Model

The codebase will keep a single implementation tree with two runtime modes:

- `locked`
  - default, conservative, and still canonicalized from `config.yaml`
  - preserves the current force-then-lock plus incremental route
- `strict_mixed_experimental`
  - selected explicitly via a separate YAML profile
  - enables the PDF target route features and validations

This design avoids a forked trainer while still giving the repository a safe default.

### Config and Profile Resolution

Profile resolution belongs in [main new.py]. The parser should:

- keep the current locked-route canonicalization for the default route
- recognize an experimental profile and relax only the capability gates needed for the PDF tasks
- map profile-level settings into [trainer_config.py] without changing default values for the locked route

The profile boundary must be explicit and testable. A misconfigured experimental profile should fail fast rather than partially falling back to the locked route.

### Training Control Plane

The trainer already contains strict-mixed and continuation plumbing, but much of the runtime is still hard-locked. The design is to reuse:

- [trainer_run_mixin.py]
- [trainer_opt_mixin.py]
- [loss_energy.py]

and convert current "locked skip" branches into profile-guarded branches.

The default route continues to skip those branches. The experimental route is allowed to execute them.

### Contact Control Plane

The existing contact stack already provides:

- `strict_mixed_inputs(...)`
- `solve_strict_inner(...)`
- strict-mixed objective assembly
- inner diagnostics aggregation

The missing work is to turn this into a stable contract:

- introduce a typed `StrictMixedContactInputs` container
- make backend routing explicit
- extend the inner solver result with optional linearization output
- make the trainer consume a uniform diagnostics schema

This keeps P0 and P1 aligned without rewriting the whole legacy operator.

## P0 Design

P0 covers the inner solver plus trainer diagnostics.

### Solver Contract

[contact_inner_solver.py] should expose an IFT-ready interface by supporting:

- the existing geometry-driven inputs
- `return_linearization=False|True`
- a stable result object that carries both the solved state and diagnostics

The initial implementation does not need full tangential IFT. It only needs to return enough linearization structure for the normal-ready route and future extension.

### Diagnostics Contract

The trainer-facing diagnostics should consistently expose:

- `fn_norm`
- `ft_norm`
- `cone_violation`
- `max_penetration`
- `fb_residual_norm`
- `fallback_used`
- `iters`
- `normal_step_norm`
- `tangential_step_norm`

These diagnostics should be emitted by the inner solver, propagated through [contact_operator.py], and surfaced by [loss_energy.py] and [trainer_opt_mixin.py].

### Runtime Policy

The default route may still ignore strict-mixed diagnostics if the route is not active. The experimental route should log them and include them in continuation-freeze decisions.

## P1 Design

P1 formalizes `ContactOperator` as a backend dispatcher and adapter.

### Typed Strict Input Contract

The current `strict_mixed_inputs()` dictionary should be upgraded to a `StrictMixedContactInputs` dataclass with fixed fields:

- `g_n`
- `ds_t`
- `normals`
- `t1`
- `t2`
- `mu`
- `eps_n`
- `k_t`
- `init_state`
- plus geometry and weight fields already consumed by strict mixed assembly

This keeps [loss_energy.py], the trainer, and tests from depending on ad hoc key strings.

### Backend Dispatch

`ContactOperator` should keep a single public surface while explicitly supporting:

- `legacy_alm`
- `inner_solver`

The locked route continues to default to legacy semantics. The experimental route resolves to the strict-mixed backend.

The design goal is not to fully rewrite legacy numerics. It is to make backend semantics explicit and predictable.

## P2 Design

P2 extends the mixed stress path for strict mixed contact training.

### Stress Head Inputs

The repository already has:

- `predict_stress_from_features(..., eps_bridge=...)`
- `semantic_feat_dim`
- pointwise stress evaluation helpers

P2 should build on that instead of replacing it.

The design separates two semantic channels:

- existing node-level engineering semantics used by the broader model
- new contact-surface semantics used only when evaluating stress on contact samples

The contact-surface semantics should include:

- contact-surface flag
- normal `n`
- tangent basis `t1`
- tangent basis `t2`

These features are used only for the contact-point stress path in the experimental route.

### Pointwise Stress Route

Strict mixed contact matching should prefer the pointwise stress path rather than the graph stress branch when evaluating contact samples. This avoids graph-specific coupling during contact-point traction matching and aligns with the PDF requirement for a pointwise stress path.

### Explicit Strict Outer Loss Route

The experimental route should assemble strict mixed training losses through an explicit strict-mixed entry point rather than through the legacy total-energy contact semantics.

That entry point should make the active route legible in code by assembling only the strict mixed outer terms:

- `R_eq`
- `R_const`
- `R_u`
- `R_t`
- `R_tr`
- `E_data`
- `E_smooth`
- `E_unc`
- `E_reg`
- `E_tight`

Legacy `E_int`, `E_cn`, and `E_ct` semantics may remain available for the locked route, but the experimental route should no longer depend on them as its primary outer objective language.

## PDF Checklist Mapping

The implementation should close all nine PDF items on the experimental route:

1. make `return_linearization` in the inner solver real and IFT-ready for the normal-ready route
2. let inner diagnostics actively alter trainer behavior instead of only being logged
3. reduce `ContactOperator` to a thin adapter / dispatcher for strict mixed execution
4. reuse the same contact kernel primitives across legacy and strict mixed contact math where practical
5. fix the strict mixed input contract with a stable typed container
6. make `eps_bridge` the default strict mixed stress path
7. inject `n`, `t1`, `t2`, and contact-surface flags into the strict mixed contact stress data flow
8. prefer the pointwise / high-frequency stress path for contact-surface stress evaluation
9. assemble strict mixed outer loss through an explicit route that is independent from the legacy total-energy contact semantics

## Files

Primary files to modify:

- [main new.py]
- [config.yaml]
- a new experimental profile file in the repository root
- [src/train/trainer_config.py]
- [src/train/trainer_run_mixin.py]
- [src/train/trainer_opt_mixin.py]
- [src/model/loss_energy.py]
- [src/model/pinn_model.py]
- [src/physics/contact/contact_operator.py]
- [src/physics/contact/contact_inner_solver.py]

Primary tests to extend:

- [test_main_new_config_override.py]
- [test_contact_inner_solver.py]
- [test_mixed_bilevel_diagnostics.py]
- [test_mixed_model_outputs.py]
- [test_trainer_optimization_hooks.py]

## Testing Strategy

### Default Route Regression

The existing locked route must keep passing its current configuration-path and trainer-path tests. No experimental setting should leak into the default path.

### Experimental Route Validation

New tests should prove that the experimental profile:

- parses successfully
- resolves to the strict mixed backend
- enables the new stress-head and diagnostics paths
- still leaves the default route unchanged

### Contact Unit Coverage

Strict-mixed contact tests should validate:

- typed strict input construction
- backend dispatch
- warm-start state reuse
- optional linearization return path
- complete diagnostics propagation

### Model Unit Coverage

P2 tests should validate:

- `eps_bridge` remains wired into stress prediction
- contact-surface semantics can be injected without breaking existing semantics
- pointwise stress evaluation is used for strict mixed contact outputs

## Acceptance Criteria

The design is complete when:

- the default [config.yaml] still resolves to the current locked route
- a new experimental profile enables the PDF target route
- strict mixed training uses explicit backend dispatch and typed strict inputs
- the inner solver exposes the required diagnostics and optional linearization output
- the strict mixed stress path uses `eps_bridge` plus contact-surface semantics without breaking existing mixed-model tests
- default-route regression tests and new experimental-route tests both pass

## Risks

### Risk: accidental default-route drift

The biggest risk is changing baseline behavior while introducing the experimental route. This is controlled by profile-gated parsing and explicit regression tests for the locked route.

### Risk: widening scope to the older checklist

The repository contains another, broader checklist that touches `RAR`, resampling, and `LBFGS`. This pass intentionally excludes that work unless separately approved.

### Risk: P2 feature coupling

Stress semantics, graph routing, and pointwise evaluation are easy to entangle. The design keeps the contact-surface semantic path local to strict mixed contact evaluation to minimize collateral regressions.
