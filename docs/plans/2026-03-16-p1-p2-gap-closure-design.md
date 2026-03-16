# P1/P2 Gap Closure Design

## Goal

Close only the remaining gaps from the PDF plan in the current codebase.

Already-landed work stays in place:

- P1 contact backend resolution
- strict mixed inner-solver adapter flow
- P2-A early stress split
- P2-A `epsilon(u)`-guided stress head

This design only covers the unfinished pieces:

- one canonical mixed elasticity residual entry point
- stronger geometry-aware stress semantics
- a contact-side pointwise or hybrid stress path

## Current State

The repository already includes the main P1 backend formalization and the first P2 stress-head refactor:

- `TrainerConfig.contact_backend` exists and strict-mixed routing is tested
- `ContactOperator` already exposes `strict_mixed_inputs()` and `solve_strict_inner()`
- `pinn_model.py` already supports `stress_branch_early_split` and `use_eps_guided_stress_head`

Focused regression confirms these paths are passing.

The remaining gap is that the model and physics stack still do not fully match the PDF's later P2 direction:

- mixed residual usage is still partially assembled downstream instead of consumed through one explicit entry point
- engineering semantics are still coarse and not clearly targeted at stress-side contact reasoning
- contact-side stress prediction does not yet have a dedicated pointwise or hybrid route

## Scope

In scope:

- add `mixed_residual_terms(...)` in `src/physics/elasticity_residual.py`
- route strict mixed residual assembly in `src/model/loss_energy.py` through that entry point
- expand trainer-built semantic features beyond the current coarse flags
- inject geometry-aware semantics into the stress-related path only
- add a config-gated contact-side pointwise or hybrid stress route
- add tests for the new residual contract, stress-side semantics, and contact-side hybrid routing

Out of scope:

- rewriting already-passing P1 backend logic
- changing legacy ALM numerics
- implementing tangential or full IFT
- broad trainer, supervision, or visualization redesign

## Design Decisions

### 1. Preserve completed work

The design treats the current repository state as authoritative for completed PDF items. Existing passing P1 and P2-A code will not be rewritten simply to match the PDF text more literally.

### 2. Standardize mixed residual access

`ElasticityResidual` becomes the canonical home for strict mixed physics residual assembly. Downstream code should ask for:

- `R_eq`
- `R_const`

and optionally consume cached terms such as strain, stress, and divergence if needed for diagnostics or weighting.

This keeps `loss_energy.py` from re-deriving mixed residual semantics ad hoc.

### 3. Keep semantics stress-side

Geometry-aware semantics should primarily support traction matching and stress representation. They should not be injected into the main displacement path by default.

The displacement path keeps the current shared/global behavior.

The stress path receives:

- current learned stress-side features
- `epsilon(u)` bridge information
- geometry-aware semantic features

### 4. Use a contact-side hybrid stress route

The design does not replace the graph path globally. Instead it adds a config-gated route where contact-region stress prediction can use pointwise or hybrid features while non-contact regions continue to use the current graph-dominant path.

This follows the PDF intent:

- preserve global context for displacement and general stress structure
- reduce over-smoothing near contact interfaces

## Architecture

### Mixed Residual Contract

`src/physics/elasticity_residual.py`

Add a unified entry point:

- `mixed_residual_terms(u_fn, sigma_fn, params, *, return_cache=False)`

Expected return shape:

- `R_eq`: `[..., 3]`
- `R_const`: `[..., 6]`

Optional cache contents:

- `eps_u`
- `sigma_pred`
- `sigma_phys`
- `div_sigma`

This wraps the existing constitutive and equilibrium logic rather than duplicating it.

### Geometry-Aware Semantic Features

`src/train/trainer.py`

Expand node semantic feature construction so it can provide a richer stress-side signal, using zero fill outside the supported contact region.

Target feature families:

- contact-surface flag
- boundary class flags
- mirror-region flag
- material or region tags
- contact-direction-related semantics when they can be stably projected to nodes

If full-node continuous `n/t1/t2` projection is not stable across the assembled mesh, use a conservative encoding that is correct on contact nodes and zero elsewhere.

### Stress-Only Semantic Injection

`src/model/pinn_model.py`

Keep existing flags off by default and add config-gated behavior so semantic features are consumed on the stress side only.

That means:

- shared/displacement path remains compatible with current behavior
- stress path can concatenate semantic features with the stress branch representation
- `epsilon(u)` bridge remains available and composes with semantic features

### Contact-Side Pointwise / Hybrid Stress Path

`src/model/pinn_model.py`

Add a config-gated contact stress route:

- non-contact nodes: existing graph stress route
- contact nodes: pointwise or hybrid stress route

The output still uses canonical Voigt ordering so downstream traction and visualization code stays unchanged.

## Error Handling

New features fail fast when enabled incorrectly:

- semantic feature dimensions must match the model config
- contact-side hybrid mode must receive a valid contact mask
- mixed residual output keys and shapes are fixed and validated in tests

Outside supported regions, semantic and contact-side features are zero-filled rather than guessed.

## Testing

Add or extend tests in:

- `test_mixed_elasticity_residuals.py`
- `test_mixed_model_outputs.py`
- `test_model_innovation_hooks.py`
- `test_mixed_contact_matching.py` if needed for contact-side feature plumbing

Test goals:

- unified mixed residual entry point returns stable shapes and finite values
- strict mixed energy assembly consumes the unified residual contract
- stress-side semantics do not break flags-off behavior
- semantics are injected on the stress path only
- contact-side pointwise or hybrid stress path is callable and shape-stable
- existing P1 backend and P2-A tests remain green

## Implementation Order

1. Add failing tests for `mixed_residual_terms(...)`
2. Implement the unified residual entry point
3. Route `loss_energy.py` strict mixed assembly through the new entry point
4. Add failing tests for richer semantic feature wiring
5. Expand trainer-built semantic features and stress-side-only injection
6. Add failing tests for the contact-side pointwise or hybrid stress route
7. Implement the contact-side hybrid path
8. Run focused regression across existing P1 and P2-A coverage

## Risks

- Semantic dimensions may drift between trainer and model if the feature contract is not explicit
- Contact-node projection may be incomplete on some meshes
- Hybrid stress routing could accidentally change flags-off behavior if not tightly gated

Mitigations:

- explicit config dimensions
- zero-fill outside supported contact regions
- focused regression with existing P1 and P2-A tests before completion
