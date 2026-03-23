# 2026-03-19 Inner Solver Residual-Driven Tangential Design

## Context

After aligning tangential residuals with the fixed-point gap and adding `effective k_t` backtracking, the solver could usually find a better first tangential step but still tended to plateau early.

The next minimum change is to make the tangential proposal directly follow the aligned residual:

`lambda_t_next = Pi(lambda_t - alpha * F_t(lambda_t), mu * lambda_n)`

where `F_t` is the already-aligned fixed-point residual and `alpha` is chosen from a short backtracking schedule.

## Decision

Replace the tangential proposal search with a residual-driven alpha backtracking loop:

1. Compute `F_t(lambda_t)` at the current post-normal state.
2. Try `alpha in [1.0, 0.5, 0.25, 0.125, 0.0625]`.
3. Form a candidate with `Pi(lambda_t - alpha * F_t, mu * lambda_n)`.
4. Accept the first candidate whose aligned tangential residual strictly decreases.

No Jacobian or semismooth Newton step is introduced in this version.

## Diagnostics

Retain the existing tangential trace fields and add:

- `tangential_step_mode`
- `effective_alpha_scale`

Keep `effective_k_t_scale` for continuity, but set it to the neutral value used by the residual-driven path.

## Fallback Semantics

Fallback classification and iterate retention should use whether the solve has ever reduced residual, not only whether the final iteration reduced it. Otherwise, a useful earlier tangential step can be discarded after a later plateau.

## Verification

1. Red test: residual-driven alpha backtracking can beat the best fixed-point alpha candidate on a hard frozen case.
2. Red test: a multi-iteration hard case continues descending for several residual-driven steps and keeps the improved iterate on fallback.
3. Focused regression on inner-solver, linearization, and mixed-contact matching.
