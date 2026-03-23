# 2026-03-19 Inner Solver Safe Tail QN Implementation

## Implemented

- Added safe diagonal regularization for tangential tail quasi-Newton in `contact_inner_solver.py`
- Added conditional trust-region clipping for highly degenerate QN diagonals
- Extended iteration trace with:
  - `qn_diag_min_raw`
  - `qn_diag_min_safe`
  - `qn_reg_gamma`
  - `qn_invalid_ratio`

## Verification

Red-green tests added or updated in `test_contact_inner_solver.py`:

- informative tail-QN case still accepted
- degenerate-diagonal tail-QN case now regularizes and accepts
- core iteration trace schema includes the new QN diagnostics

Focused regression target:

- `test_contact_inner_solver`
- `test_contact_inner_solver_linearization`
- `test_mixed_contact_matching`

Real-batch verification target:

- `strict_mixed_tangential_tail_qn_trace.json`
- check `tail_has_effective_step`, `effective_alpha_scale`, and tail `ft_reduction_ratio`

