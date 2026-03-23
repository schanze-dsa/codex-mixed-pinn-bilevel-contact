# Inner Solver Iteration Trace Design

## Context

The current strict-mixed smoke run reaches the `normal_ready + inner_solver` route, but the inner solve always reports fallback. Existing diagnostics already show:

- `g_n` sign convention is internally consistent.
- `normal_ift_valid_ratio` is healthy on the reconstructed real batch.
- The main unresolved question is whether the solver is numerically failing during the normal step, the tangential step, or the final fallback gate.

The user requested the smallest possible experiment:

1. Add per-iteration logging to `solve_contact_inner(...)`.
2. Run exactly one real batch and one solve.
3. Compare `normal-only` against `normal + weakened tangential`.
4. Focus only on:
   - `fn_residual_before/after`
   - `ft_residual_before/after`
   - `delta_lambda_n_norm`
   - `delta_lambda_t_norm`
   - `fallback_trigger_reason`

## Goals

- Add a default-off structured trace path to the strict inner solver.
- Preserve the current training/runtime behavior when the trace is not requested.
- Make the trace accessible both through direct solver calls and the `ContactOperator` strict-mixed adapter.
- Keep the trace small and machine-readable so a single-batch diagnostic script can dump JSON instead of polluting training logs.

## Chosen Approach

Add an optional `return_iteration_trace: bool = False` flag to `solve_contact_inner(...)` and `ContactOperator.solve_strict_inner(...)`.

When disabled:

- No behavioral change.
- No new diagnostics keys are required by training.

When enabled:

- The solver returns an `iteration_trace` payload inside `ContactInnerResult.diagnostics`.
- Each iteration record captures scalar summary values only.
- Fallback trigger information is derived after the loop from the same convergence/feasibility checks already used by the solver.

## Trace Shape

The trace payload will have this shape:

```python
{
    "iterations": [
        {
            "iter": 1,
            "fn_residual_before": ...,
            "fn_residual_after": ...,
            "ft_residual_before": ...,
            "ft_residual_after": ...,
            "delta_lambda_n_norm": ...,
            "delta_lambda_t_norm": ...,
        },
        ...
    ],
    "fallback_trigger_reason": "...",
}
```

This keeps the new path aligned with the user-requested five core observables.

## Fallback Trigger Reason

`fallback_trigger_reason` will be a short comma-joined string based on the final failed conditions:

- `normal_step`
- `tangential_residual`
- `cone`
- `lambda_negative`
- `max_iters`

This is intentionally solver-local. It should explain why `fallback_used` happened inside `solve_contact_inner(...)`, not why the higher-level strict-mixed policy later froze continuation.

## Testing

Add a targeted unit test that:

- Verifies the default call still does not expose iteration trace data.
- Verifies `return_iteration_trace=True` returns a non-empty structured trace with the requested keys.
- Verifies `ContactOperator.solve_strict_inner(...)` transparently forwards the flag.

## Diagnostic Run

After the test passes, run one real-batch diagnostic using the existing smoke config and checkpoint:

- `normal-only`
- `normal + weakened tangential`

Write the comparison JSON under `tmp/` so the results can be inspected without touching the main training loop.
