# Inner Solver Tangential Acceptance Plan

Date: 2026-03-18

## Scope

Only change the tangential update path inside `solve_contact_inner(...)`.

Do not modify:

- the normal block
- strict mixed runtime policy
- IFT routing

## Plan

1. Add failing tests for:
   - a backtracking case where full tangential proposal does not reduce the
     aligned residual but a smaller damping step does
   - fallback classification only using
     `tangential_residual_not_reduced` when no accepted tangential step lowers
     the aligned residual

2. Implement a short tangential acceptance loop in the inner solver:
   - base proposal from the current fixed-point target
   - damping schedule using the requested damping and smaller retries
   - accept the first residual-reducing candidate
   - otherwise keep the current tangential state

3. Preserve current trace outputs and update classification bookkeeping.

4. Run focused regression tests.

5. Re-run the existing frozen-batch and eager A/B verification scripts.
