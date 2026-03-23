# Inner Solver Tangential Residual Alignment Design

## Context

The tangential consistency audit proved that the current production tangential
residual is not the fixed-point gap of the current tangential update map.

On the same real frozen batch:

- the production `ft_residual_norm` stayed around `51`
- the actual fixed-point gap norm was around `0.127`
- after one current update, the production residual still stayed around `51`
- but the fixed-point gap norm dropped to around `0.005`

That means the current tangential update is already reducing its own fixed-point
gap, while the production diagnostics and fallback logic are still looking at a
different equation.

## Goal

Align the production tangential residual definition with the current tangential
update map before changing the tangential update itself.

## Scope

This round should:

- make the production tangential residual equal to the fixed-point gap
  `lambda_t - T(lambda_t)`
- keep the tangential update map unchanged
- make trace, fallback classification, convergence, and linearization use the
  aligned residual

This round should not:

- redesign the tangential update map
- add backtracking or acceptance logic
- revisit normal, policy, or IFT logic

## Chosen Residual

Use:

`F_t(lambda_t) = lambda_t - Π(lambda_t + k_t ds_t, mu * lambda_n)`

This is exactly the fixed-point residual of the current update map.

## Affected Behavior

After the change, the following should all use the aligned residual:

- `friction_fixed_point_residual(...)`
- iteration trace `ft_residual_before/after`
- `ft_reduction_ratio`
- tangential convergence checks
- `fallback_trigger_reason = tangential_residual_not_reduced`
- tangential residual block inside linearization

## Testing

The smallest tests needed are:

1. production residual definition equals fixed-point gap
2. one current update reduces the new residual on a representative frozen batch
3. tangential fallback only triggers when the new residual is not reduced

## Verification

After the code change:

- rerun the focused primitive + solver slice
- rerun the frozen-batch compare
- rerun the eager smoke A/B compare

The expected qualitative change is:

- `ft_residual_after < ft_residual_before` on the frozen batch
- `ft_residual_norm` and `fp_gap_norm` are now the same quantity
- `tangential_residual_not_reduced` should no longer fire for cases where the
  current update actually shrinks the fixed-point gap
