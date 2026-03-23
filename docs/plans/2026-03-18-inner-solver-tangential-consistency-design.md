# Inner Solver Tangential Consistency Audit Design

## Context

The current frozen-batch tangential comparison showed:

- Proposal A/B/C do not reduce the currently reported `ft_residual`.
- cone feasibility is already effectively satisfied.
- the tangential state changes, but the reported residual does not.

That pattern strongly suggests the next question is not about damping or
acceptance. It is whether the current tangential residual is even measuring the
fixed-point equation of the current tangential update map.

## Goal

Audit whether the current tangential residual is consistent with the current
update map before changing solver behavior.

## Scope

This round should:

1. expose the tangential update map explicitly as a reusable primitive
2. expose the fixed-point gap of that update map explicitly as a reusable
   primitive
3. compare, on the same frozen batch:
   - current `ft_residual_norm`
   - `fp_gap_norm = ||lambda_t - T(lambda_t)||`

This round should not:

- change production solver convergence
- change tangential update behavior
- add acceptance/backtracking
- revisit normal or policy behavior

## Chosen Approach

Add two explicit kernel helpers:

- `tangential_update_map(...)`
- `tangential_fixed_point_gap(...)`

Keep the existing `friction_fixed_point_residual(...)` untouched for now. The
audit result should reveal whether the current residual and the update map are
the same equation or not.

## Audit Question

For the same frozen batch, compare:

- `ft_residual_norm`
- `fp_gap_norm`

If they are not the same quantity, or not even the same scale/trend, then the
current residual is not the fixed-point gap of the current update map. In that
case, the next production change should be to align the tangential residual
definition first.

## Expected Outcome

If the suspicion is correct, this audit will show:

- `friction_fixed_point_residual` and `tangential_fixed_point_gap` differ by the
  explicit `k_t * ds_t` term
- the current huge `ft_residual_norm` is therefore not evidence that the update
  map itself failed to reduce its own fixed-point gap

That result is enough to justify the next cut: unify the tangential residual
definition with the update map, then rerun the frozen-batch proposal compare.
