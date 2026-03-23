# Inner Solver Tangential Acceptance Design

Date: 2026-03-18

## Context

After aligning the production tangential residual to the fixed-point gap

`F_t(lambda_t) = lambda_t - Pi(lambda_t + k_t ds_t, mu lambda_n)`,

the frozen-batch audit now shows:

- `ft_residual_before` and `fp_gap_norm` are equal
- one tangential update lowers that aligned residual on the real batch

However the eager smoke comparison still ends with:

- `fallback_trigger_reason = tangential_residual_not_reduced`
- `fallback_used_count = 5`
- `converged_count = 0`

The remaining issue is no longer the residual definition. It is the tangential
update path itself, especially its later-iteration plateau behavior.

## Goal

Stabilize the tangential block without changing the normal block, policy, or
IFT path.

## Chosen Approach

Use a minimal fixed-point proposal with acceptance/backtracking.

For each tangential iteration:

1. Build the base proposal

   `target_lambda_t = Pi(lambda_t + k_t ds_t, mu lambda_n_next)`

2. Try accepted candidates along a short damping schedule

   `lambda_t_candidate = (1 - a) lambda_t + a target_lambda_t`

   starting from the requested damping and reducing it geometrically.

3. Accept the first candidate whose aligned tangential residual decreases:

   `||F_t(lambda_t_candidate)|| < ||F_t(lambda_t)||`

4. If no candidate reduces the residual, keep the old state and classify that
   iteration as tangential-not-reduced.

## Diagnostics

Keep the current eager verification path and preserve the existing trace fields.
Add only the minimum internal bookkeeping needed to distinguish:

- a real tangential non-reduction event
- a budget-exhausted case where residual was still decreasing

## Success Criteria

At least some of the existing eager smoke cases should stop ending in
`tangential_residual_not_reduced`, and the frozen-batch trace should no longer
plateau immediately after the first accepted update.
