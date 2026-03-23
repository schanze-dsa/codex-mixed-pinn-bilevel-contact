# 2026-03-19 Inner Solver Residual-Driven Tangential

## Implemented

- Replaced tangential proposal search with residual-driven alpha backtracking.
- Added iteration-trace fields:
  - `tangential_step_mode`
  - `effective_alpha_scale`
- Kept `effective_k_t_scale` in the trace for continuity.
- Changed fallback classification and iterate retention to use whether any reduction happened during the solve, not only the final iteration.

## Verification

Focused regression passed:

`python -m unittest test_contact_inner_kernel_primitives test_contact_inner_solver test_contact_inner_solver_linearization test_mixed_contact_matching -v`

Result: `38/38 OK`

## Diagnostic Outcome

- On a hard frozen case, residual-driven alpha backtracking beats the best fixed-point alpha candidate.
- On a separate multi-step hard case, the residual-driven path keeps reducing `ft_residual` over many iterations instead of stopping after the first accepted move.

See:

- `tmp/strict_mixed_tangential_residual_driven_compare.json`

## Remaining Gap

This step proves the tangential update is now meaningfully stronger at solver level. It does not yet prove smoke-level convergence counts improve on real training batches, so the next validation step should be a fresh eager route comparison once the diagnostic runner is made cheap enough to finish reliably.
