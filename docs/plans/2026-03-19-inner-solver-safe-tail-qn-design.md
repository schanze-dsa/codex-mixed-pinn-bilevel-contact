# 2026-03-19 Inner Solver Safe Tail QN Design

## Context

Real `normal_ready` tail traces reached `residual_driven_tail_qn`, but no accepted step was found because:

- `qn_diag_min_raw = 0`
- `effective_alpha_scale = 0`
- `tail_has_effective_step = false`
- fallback still ended at `iteration_budget_exhausted`

The immediate issue was not route selection or outer budget. It was tail QN diagonal degeneracy.

## Decision

Keep the existing two-stage tangential update:

- regular stage: residual-driven update plus acceptance
- tail stage: diagonal quasi-Newton plus acceptance/backtracking

Strengthen only the tail QN branch with:

1. Safe diagonal regularization
   - use a sign-preserving floor with `floor = 1e-3`
   - keep `gamma = 0` for now
   - record both raw and safe minima

2. Conditional trust-region clipping
   - clip the QN step only when the invalid-diagonal ratio is high
   - keep existing alpha backtracking and acceptance unchanged

3. Better diagnostics
   - `qn_diag_min_raw`
   - `qn_diag_min_safe`
   - `qn_reg_gamma`
   - `qn_invalid_ratio`

## Expected effect

- Degenerate tail-QN cases should stop collapsing into `effective_alpha_scale = 0`
- Tail iterations should regain accepted steps
- Real tail traces should show renewed `ft_residual_after < ft_residual_before`

