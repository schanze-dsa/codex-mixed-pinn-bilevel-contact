# Inner Solver Tangential Proposal Compare Design

## Context

The current strict-mixed smoke diagnosis already shows:

- `normal_ready` is consumed by the trainer.
- the normal block is no longer the dominant bottleneck.
- fallback now consistently lands on `tangential_residual_not_reduced`.

The latest iteration trace also shows that the tangential state keeps moving
while `ft_residual_after` stays essentially flat. That means the next question
is no longer whether tangential matters. It is which tangential proposal, if
any, actually reduces the current tangential residual.

## Goal

Add the smallest missing diagnostics needed to compare tangential proposals on a
single real frozen batch, without changing the solver update formula yet.

## Scope

This round does only two things:

1. extend the existing iteration trace with:
   - `target_lambda_t_norm`
   - `ft_reduction_ratio`
2. run one frozen-batch proposal comparison on the same real batch.

This round does not:

- change the tangential update formula
- add acceptance/backtracking to production code
- revisit normal, policy, or IFT behavior

## Trace Additions

### `target_lambda_t_norm`

This records the norm of the projected tangential target:

`target_lambda_t = Π(lambda_t + k_t ds_t, mu * next_lambda_n)`

Without it, we cannot tell whether the update is failing because:

- the target itself barely changes, or
- the target changes but the accepted state does not follow it.

### `ft_reduction_ratio`

This records:

`ft_residual_after / (ft_residual_before + eps)`

It is a compact indicator for:

- no reduction
- weak reduction
- unstable oscillation

## Frozen-Batch Comparison

After the two trace additions, run exactly one frozen-batch comparison on the
same real smoke batch. Freeze the batch after the normal substep, because the
tangential proposal uses `next_lambda_n`, not the pre-normal state.

Compare three proposals:

- Proposal A: current implementation
- Proposal B: undamped direct projection
- Proposal C: residual-driven tangential proposal

Because the current codebase defines the tangential residual as
`friction_fixed_point_residual(...)`, Proposal C should be evaluated against
that actual residual definition, not an alternate symbolic residual.

## Success Criteria

This round is successful if it answers:

1. whether `target_lambda_t` is materially different from `lambda_t_after`
2. whether any proposal produces a strictly smaller `ft_residual_after`
3. whether the best proposal preserves low cone violation on the same batch

The output should be enough to justify the next production change: either keep
the current tangential target, or replace it with the best-performing proposal
plus an acceptance rule.
