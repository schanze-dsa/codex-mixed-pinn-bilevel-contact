# Inner Solver Tangential Trace Design

## Context

The strict-mixed smoke A/B run now shows:

- `normal_ready` is genuinely consumed by the trainer.
- The normal block is no longer the main bottleneck.
- Both `forward_only` and `normal_ready` still fallback on
  `tangential_residual_not_reduced`.

The next question is not "how to fix tangential" yet. It is:

1. Is the tangential update actually failing to reduce residual?
2. Or is the residual decreasing but still rejected by the current stop gate?

## Goal

Add the smallest possible tangential diagnostics to the existing
`iteration_trace` so one real batch can answer those two questions without
changing solver behavior.

## Scope

Only extend trace output from `solve_contact_inner(...)`.

Do not:

- change the tangential update formula
- change tangential convergence thresholds
- change strict mixed policy
- change trainer/IFT routing

## Chosen Diagnostics

Keep the existing trace fields and add:

- `lambda_t_before_norm`
- `lambda_t_after_norm`
- `cone_violation_before`
- `cone_violation_after`
- `slip_norm`

These are scalar summaries, not full tensors. The trace stays compact and can
still be dumped to JSON for a single real batch.

## Interpretation

The added fields should let us distinguish:

- true non-convergence:
  `ft_residual_after` does not decrease meaningfully, `delta_lambda_t_norm`
  stays active, and fallback still lands on `tangential_residual_not_reduced`
- over-strict stopping:
  `ft_residual_after` keeps decreasing, `cone_violation_after` improves, and
  only the final threshold prevents convergence

`slip_norm` is defined as the maximum row-wise norm of `ds_t`, so we can tell
whether the batch is actually demanding significant tangential motion.

## Testing

Extend the existing iteration-trace unit test so it fails until the new fields
exist. Then re-run the focused inner-solver slice.

## Verification

After the tests pass, run one eager smoke A/B comparison again and inspect the
last-batch trace for:

- `ft_residual_before/after`
- `lambda_t_before_norm/after_norm`
- `delta_lambda_t_norm`
- `cone_violation_before/after`
- `slip_norm`
- `fallback_trigger_reason`
