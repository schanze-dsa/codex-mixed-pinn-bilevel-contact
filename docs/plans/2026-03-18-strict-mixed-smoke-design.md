# Strict Mixed Smoke Training Design

## Goal

Create a dedicated smoke-training YAML that exercises the current
`strict_mixed_experimental` normal-ready path with the least runtime needed to
verify the training chain.

## Current Constraint

The base experimental YAML still enables two-stage training with
`phase1.max_steps=300` and `phase2.max_steps=150`. Reducing only
`optimizer_config.epochs` would not shorten the actual run enough.

## Chosen Approach

Create a new `strict_mixed_experimental_smoke.yaml` derived from the existing
experimental YAML, but:

- keep `mixed_bilevel_phase.phase_name=phase1`
- keep `normal_ift_enabled=true`
- keep `tangential_ift_enabled=false`
- keep `detach_inner_solution=false`
- keep `contact_backend=inner_solver`
- disable `two_stage_training`
- reduce step counts, contact points, and output work

## Why This Approach

- It validates the normal-ready strict mixed chain directly.
- It does not mutate the main experiment YAML.
- It avoids an unnecessary phase-2 handoff during a smoke check.

## Non-Goals

- No attempt to measure quality or convergence.
- No change to the main experimental config.
- No long training run.
