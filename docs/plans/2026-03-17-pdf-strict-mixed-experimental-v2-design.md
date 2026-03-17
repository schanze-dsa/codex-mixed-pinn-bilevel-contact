# PDF Strict Mixed Experimental V2 Design

## Goal

Close the `p0_p1_p2_detailed_action_checklist_cn_v2.pdf` items on the `strict_mixed_experimental` route without changing the default locked behavior in `config.yaml`.

The target state is not a new architecture branch. It is a controlled closeout pass that turns the current strict mixed bilevel path from "already runnable" into "trainer-consumable, testable, and ready to compete as the default physical route later".

## Approved Scope

- Keep `config.yaml` on the current locked route.
- Land the V2 closeout work only on `strict_mixed_experimental.yaml` and the strict mixed runtime path.
- Preserve the legacy backend as a compatibility route.
- Do not widen this pass into the older large-scope contact rewrite.

## Design Summary

The repository keeps one implementation tree with two runtime routes:

- `locked`
  - current default route
  - keeps force-then-lock plus incremental semantics intact
- `strict_mixed_experimental`
  - explicit opt-in route
  - receives the V2 closeout work for P0, P1, and P2

The closeout pass is additive and route-gated. No experimental behavior should leak into the locked path.

## P0 Design

### P0-1 Linearization Schema

`src/physics/contact/contact_inner_solver.py` already exposes `return_linearization=True`, but V2 requires the result to be directly consumable by trainer-side normal-only IFT.

The returned `linearization` payload will become a stable schema with these fields:

- `schema_version`
- `route_mode`
- `is_exact`
- `tangential_mode`
- `jac_z`
- `jac_inputs`
- `state_layout`
- `input_layout`
- `flat_z`
- `flat_inputs`
- `residual_at_solution`

The key design choice is to return explicit layout metadata instead of Python callbacks such as `flatten_fn` or `unflatten_fn`. Metadata survives tests and `tf.function`; callable payloads do not.

`state_layout` and `input_layout` define the flatten order and shapes for:

- `lambda_n`
- `lambda_t`
- `g_n`
- `ds_t`

`residual_at_solution` is included so trainer-side code can verify that the linearization point is actually usable rather than assuming convergence from phase flags alone.

The first supported closeout target is still `normal_ready`. Tangential/full IFT remains out of scope for this pass and will be represented explicitly in the schema through `tangential_mode`.

### P0-2 Diagnostics As Control Signals

`src/train/trainer_opt_mixin.py` already aggregates strict mixed diagnostics, but V2 requires the trainer to expose and act on the reason for backoff rather than only toggling booleans.

The trainer closeout adds three explicit diagnostics/control outputs:

- `phase_hold_reason`
- `continuation_backoff_applied`
- `inner_solver_not_stable_count`

These sit on top of the existing policy booleans:

- `strict_phase_hold`
- `strict_continuation_backoff`
- `strict_force_detach`
- `strict_traction_scale`

The runtime policy remains route-gated. Only the experimental route consumes these diagnostics as action signals. The locked route remains unchanged.

### P0-3 Normal-Only IFT Runtime

The experimental profile is updated so the intended V2 normal-only IFT route is explicit:

- `normal_ift_enabled: true`
- `tangential_ift_enabled: false`
- `detach_inner_solution: false`

This keeps the current scope disciplined:

- forward-only remains available
- normal-only IFT becomes the active closeout target
- tangential/full IFT remains disabled

## P1 Design

### P1-1 ContactOperator As Adapter

`src/physics/contact/contact_operator.py` will be rewritten in description and interface emphasis as an adapter / dispatcher, not a unified owner of all contact physics.

Its strict mixed responsibilities are limited to:

- `strict_mixed_inputs(...)`
- `solve_strict_inner(...)`
- `traction_matching_terms(...)`

The contact math stays in:

- `contact_inner_solver.py`
- shared kernel primitives
- legacy ALM modules where route-specific behavior is still required

### P1-2 Unified Strict Input Contract

`StrictMixedContactInputs` becomes the single strict mixed contract shared by operator, loss assembly, trainer plumbing, and tests.

Required fields remain:

- `g_n`
- `ds_t`
- `normals`
- `t1`
- `t2`
- `mu`
- `eps_n`
- `k_t`
- `init_state`

Existing geometry and weighting fields stay because the outer objective already consumes them:

- `weights`
- `xs`
- `xm`

The V2 closeout extends the contract with metadata for debugging and warm-start alignment:

- `batch_meta`
- optional `contact_ids`

The intent is not to increase feature scope. It is to stop passing semi-structured tensor bundles through multiple layers.

### P1-3 Shared Kernel Primitives

The legacy and strict mixed routes should differ in training semantics, not in hidden duplicate implementations of the same low-level contact math.

The shared primitive set remains:

- `fb_normal_residual`
- `project_to_coulomb_disk`
- `compose_contact_traction`
- `check_contact_feasibility`

This pass keeps the legacy route alive while further reducing duplicate math ownership in `ContactOperator`.

## P2 Design

### P2-1 Explicit Strict Mixed Stress Defaults

The current implementation can already force `eps_bridge` on contact-surface stress evaluation, but V2 requires that default to become explicit rather than emergent.

The model config will gain strict mixed route controls such as:

- `strict_mixed_default_eps_bridge`
- `strict_mixed_contact_pointwise_stress`

The experimental profile enables them by default. The locked route remains unchanged.

### P2-2 Contact-Surface Semantics In Real Data Flow

The repository already supports engineering semantics and contact-surface stress context, but V2 requires the strict mixed route to carry those semantics as a real runtime fact.

The closeout does not rebuild the full dataset stack. Instead, it makes the strict mixed contact batch assembly explicitly inject:

- `is_contact_surface`
- `n`
- `t1`
- `t2`

Non-contact samples continue to use zero-filled or fixed placeholder semantics where needed, with one consistent convention.

### P2-3 Default Pointwise Contact Stress Path

Strict mixed contact stress should default to a pointwise/high-frequency path for contact-surface samples while leaving displacement and bulk stress routes free to keep graph or hybrid behavior.

The practical rule is:

- contact-surface `sigma` defaults to pointwise
- bulk `sigma` may still use graph or hybrid
- displacement path remains unchanged unless already configured otherwise

### P2-4 Explicit Strict Mixed Outer Loss Assembly

`src/model/loss_energy.py` will expose an explicit strict mixed outer-loss assembler, for example `assemble_strict_mixed_outer_loss(...)`, and `src/train/trainer_opt_mixin.py` will call that path explicitly on the experimental route.

`strict_mixed_objective()` stays as a thin compatibility wrapper, but the code should make it obvious that strict mixed training is assembled through its own route rather than quietly piggybacking on the old `TotalEnergy` narrative.

The strict mixed outer loss remains limited to:

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

The experimental route should no longer present legacy `E_int`, `E_cn`, or `E_ct` as its primary outer-training language.

## Testing Design

The closeout is verified through route-gated regression coverage, not by optimistic inspection.

New or extended tests will cover:

- linearization schema fields, shape/layout metadata, and finite-difference sanity on a small case
- trainer diagnostics reason fields and visible backoff counters
- experimental profile defaults for normal-only IFT and detach control
- strict mixed adapter contract fields including metadata
- explicit strict mixed stress defaults and pointwise contact stress routing
- explicit strict mixed outer-loss assembly entry point

The locked route regression suite remains mandatory.

## Acceptance Criteria

The V2 design is complete when all of the following are true:

- locked `config.yaml` behavior is unchanged
- the experimental profile is the only route that enables V2 closeout behavior
- trainer-visible IFT metadata is explicit and stable
- trainer logs expose why strict mixed backoff or hold was triggered
- `ContactOperator` reads as an adapter, not as the primary contact solver
- strict mixed contact semantics and stress routing are explicit defaults, not accidental side effects
- the strict mixed outer objective has a named assembly path

## Risks

### Risk: route leakage

The biggest risk remains accidental drift in the locked path. All closeout logic must stay behind experimental route checks.

### Risk: overloading the current contact data path

V2 wants contact semantics to become a real runtime fact. The safe implementation is to inject them at strict mixed contact assembly time instead of rewriting unrelated bulk dataset code.

### Risk: unstable trainer-policy coupling

Adding reason fields and counters should not introduce new policy branches for the locked route. The policy surface stays simple and route-gated.
