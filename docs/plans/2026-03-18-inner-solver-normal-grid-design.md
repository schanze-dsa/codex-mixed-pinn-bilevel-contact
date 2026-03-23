# Inner Solver Normal-Step Grid Design

## Context

The strict inner solver no longer false-converges, but the real smoke batch still exits with `normal_fb_residual_not_reduced`. The current normal-only trace keeps decreasing the normal FB residual through all 8 iterations, which means the local update direction is usable but too slow at the current budget.

## Decision

Keep this round strictly local to the normal block.

- Do not change trainer flow.
- Do not change IFT consumption.
- Do not change tangential logic first.
- Do not change the strict mixed runtime policy.

## Minimal change

Use the existing `damping` knob as the local normal-correction gain for the FB-residual step, and make gains above `1.0` meaningful by removing the current upper clamp and scaling the normal correction cap with the same gain. This preserves clipping while allowing a larger effective normal step.

## Experiment

Re-run the same real smoke batch with a small normal-only grid:

- gain: `1.0`, `1.5`, `2.0`
- `max_inner_iters`: `8`, `12`, `16`

For each run, record:

- final `fn_residual_after`
- `fallback_used`
- `fallback_trigger_reason`
- per-iteration decay ratio `rho_k = F_{k+1} / F_k`
- estimated remaining iterations to reach `tol_fb`, using the recent geometric decay ratio when valid

Then run one `normal + weakened tangential` comparison with the best normal-only setting to verify tangential is not worse than the chosen normal baseline.

## Success criteria

This round is considered useful if at least one of these becomes true on the same real batch:

- `normal_only` no longer stabilizes at `fallback_used=1`
- final normal residual falls materially faster than the current baseline
- the dominant fallback reason is no longer `normal_fb_residual_not_reduced`
- the grid clearly shows whether the bottleneck is step size or iteration budget
