# 2026-03-19 Inner Solver Effective k_t Backtracking Design

## Context

Tangential residual is aligned with the fixed-point map, and a single accepted tangential update already reduces it. The remaining issue is that the solver often accepts one move and then immediately plateaus, leaving `fallback_trigger_reason = tangential_residual_not_reduced`.

Current damping-only acceptance is not always strong enough because it only blends toward the full tangential proposal:

`lambda_t_next = (1 - alpha) * lambda_t + alpha * Pi(lambda_t + k_t * ds_t, mu * lambda_n)`

When the local tangential region is too stiff, shrinking `alpha` alone may not find a residual-reducing step.

## Decision

Add a second backtracking axis on the proposal driver itself:

`Pi(lambda_t + eta * k_t * ds_t, mu * lambda_n)`, with `eta in [1, 0.5, 0.25, 0.125, 0.0625]`

For each `eta`, reuse the existing damping acceptance schedule. Accept the first candidate whose aligned tangential residual is strictly smaller than the current residual.

## Diagnostics

Keep existing tangential trace fields and add:

- `tangential_backtrack_steps`
- `effective_k_t_scale`

These show whether proposal-level backtracking actually engaged and which tangential scale was accepted.

## Verification

1. Add a failing regression where no damping-only candidate on the full `k_t` proposal reduces residual, but a smaller `effective_k_t_scale` does.
2. Ensure iteration traces expose the new backtracking diagnostics.
3. Re-run focused inner-solver regression tests.
4. Re-check a hard frozen case to confirm the solver now makes a real tangential move with `effective_k_t_scale < 1`.
