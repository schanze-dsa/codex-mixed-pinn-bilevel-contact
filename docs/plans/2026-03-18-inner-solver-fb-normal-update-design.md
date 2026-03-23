# Inner Solver FB Normal Update Design

## Context

The current strict inner solver can report `converged=1` when the normal step has stalled but the normal Fischer-Burmeister residual is still materially large. The iteration trace from the real smoke batch also shows the normal block plateauing after the first step, while tangential weakening changes fallback behavior without improving the normal residual trajectory.

## Design

Use the existing smooth FB residual as the explicit normal solve target.

1. Extend the normal convergence gate so it requires:
   - small normal step
   - small normal FB residual
   - feasibility
   - tangential residual still remains part of the full solver convergence gate
2. Replace the `target_lambda_n` attraction step with a damped FB-residual correction that uses the diagonal normal block:
   - compute the current FB residual
   - compute the current diagonal derivative with respect to `lambda_n`
   - take a damped, positivity-projected correction step
3. Refine the iteration-trace failure labeling so diagnostics can distinguish:
   - `normal_fb_residual_not_reduced`
   - `tangential_residual_not_reduced`
   - `policy_penetration_gate`
   - `invalid_diag`
   - `nan_or_inf`

## Constraints

- Keep the default execution path stateless.
- Keep the public `solve_contact_inner(...)` interface backward compatible by defaulting `tol_fb` to `tol_n`.
- Reuse the existing `damping` parameter for the first correction-step implementation.
- Preserve the current trace shape and extend it only with higher-signal scalar fields.

## Success Criteria

On the same real batch used in the earlier trace:

- `normal-only` no longer reports `converged=1` while `fb_residual_norm` is still large
- `fn_residual_after` continues to decrease past the first iteration instead of plateauing immediately
- tangential-only failure cases are labeled as `tangential_residual_not_reduced`
- fallback reasons separate solver failure from the runtime policy gate
