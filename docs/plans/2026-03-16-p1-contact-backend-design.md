# P1 Contact Backend Design

## Goal

Formalize `ContactOperator` backend selection for P1 so the project can explicitly distinguish:

- legacy ALM semantics used by the old training baseline
- inner-solver semantics used by strict mixed bilevel training

The default behavior remains conservative:

- non-strict-mixed routes default to `legacy_alm`
- strict mixed routes default to `inner_solver`

This is a P1 architecture step, not a P2 model redesign and not a full legacy-contact rewrite.

## Scope

In scope:

- add explicit `contact_backend` configuration
- resolve backend at trainer runtime from config plus mixed-bilevel phase flags
- keep route semantics and backend semantics separate
- expose the resolved backend in logs and runtime diagnostics
- make backend selection testable and fail fast on contradictory combinations

Out of scope:

- changing default legacy `energy()/residual()/update_multipliers()` numerics
- implementing full tangential or full IFT
- touching the P2 network architecture work in `pinn_model.py`
- rewriting all legacy contact code to delegate through the inner solver in this step

## Architecture

### Backend Model

`TrainerConfig` gets a new `contact_backend` field with three values:

- `auto`
- `legacy_alm`
- `inner_solver`

`auto` is the only recommended default. Its resolution rule is fixed:

- if the resolved training route is `legacy`, use `legacy_alm`
- if the resolved training route is `forward_only` or `normal_ready`, use `inner_solver`

This keeps the current baseline stable while making strict mixed semantics explicit.

### Route vs Backend

The design separates two decisions that were previously entangled:

- route: how the outer objective is assembled
- backend: which contact semantic shell is active

The trainer remains responsible for route resolution. `loss_energy.py` should not decide which backend to use. It only consumes resolved flags passed in from the trainer.

This gives four desirable properties:

- strict mixed objective stays explicit
- legacy baseline remains available
- logs can report both route and backend
- contradictory user settings can be rejected before training starts

## Runtime Contract

### Trainer

The trainer resolves two runtime values:

- `strict_route_mode`
- `contact_backend`

Recommended mapping:

- `legacy` -> `legacy_alm`
- `forward_only` -> `inner_solver`
- `normal_ready` -> `inner_solver`

If the user explicitly overrides `contact_backend`, the trainer validates the combination. Invalid combinations should fail immediately rather than silently mixing semantics.

Examples of invalid combinations for this P1 step:

- `route=legacy` with `contact_backend=inner_solver`
- any route that would require tangential/full IFT while still marked P0/P1-only

### ContactOperator

`ContactOperator` keeps the current public API, but the semantics become explicit:

- `legacy_alm`
  - `energy()`
  - `residual()`
  - `update_multipliers()`
- `inner_solver`
  - `strict_mixed_inputs()`
  - `solve_strict_inner()`

This is a modeling change, not yet a full internal delegation rewrite. P1 only requires the backend contract to be explicit, configurable, and visible to tests/logging.

## Logging and Diagnostics

Trainer logs should show both route and backend:

- `smode=legacy|forward_only|normal_ready`
- `cback=legacy_alm|inner_solver`

These are reported alongside the existing strict-mixed aggregate diagnostics:

- `iconv`
- `ifb`
- `iskip`
- `cfrz`
- `cfrze`

This makes it possible to understand whether a run is failing because of backend choice, route choice, or inner-solver quality.

## Testing Strategy

Three layers of tests are required.

### Config / Trainer tests

- `contact_backend=auto` resolves to `legacy_alm` for non-strict-mixed routes
- `contact_backend=auto` resolves to `inner_solver` for strict mixed routes
- explicit backend override works for allowed combinations
- contradictory combinations fail fast

### Adapter tests

- `ContactOperator` remains valid in legacy mode
- strict mixed path still consumes `strict_mixed_inputs()` plus `solve_strict_inner()`
- the backend contract is explicit and testable without changing current legacy numerics

### Logging tests

- strict mixed logs include `cback=inner_solver`
- legacy logs include `cback=legacy_alm`
- backend text appears together with route diagnostics

## Acceptance Criteria

P1 backend formalization is complete when:

- default legacy training still uses `legacy_alm`
- strict mixed training defaults to `inner_solver`
- backend choice is explicit in trainer state and logs
- invalid route/backend combinations fail fast
- existing P0 strict-mixed tests still pass
- new backend resolution and logging tests pass

## Risk Control

The design intentionally avoids risky numerical changes in this step:

- no implicit change to legacy contact penalties
- no hidden fallback from contradictory backend choices
- no P2 architecture work mixed into this change

This keeps the change aligned with the PDF recommendation: make the strict mixed line true first, then make the outer model stronger later.
