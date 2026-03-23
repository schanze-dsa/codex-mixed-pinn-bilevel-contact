# 2026-03-19 Inner Solver Effective k_t Backtracking

## Implemented

- Added proposal-level tangential backtracking on `effective k_t` in `solve_contact_inner(...)`.
- Kept damping-based acceptance, but now search over reduced tangential proposal scales first.
- Added iteration-trace diagnostics:
  - `tangential_backtrack_steps`
  - `effective_k_t_scale`

## Test Coverage

- Added a regression proving a case where the full-`k_t` damping schedule cannot reduce residual, but proposal-level `effective k_t` backtracking can.
- Extended the trace-schema coverage to require the new diagnostics.

## Verification Notes

- Focused regression suite passes after the change.
- The new hard-case trace confirms:
  - first tangential iteration reduces residual
  - `effective_k_t_scale < 1`
  - later iterations can still plateau, so this is an improvement to step control, not a complete tangential-solver fix

## Follow-up

If tangential plateau persists on real batches after this step, the next change should target the tangential update rule itself, likely a residual-driven step or stronger proposal acceptance, not the normal block.
