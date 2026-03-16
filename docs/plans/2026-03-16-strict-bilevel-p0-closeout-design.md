# Strict Bilevel P0 Closeout Design

## Goal

Close the remaining `P0` items from the mixed bilevel audit checklist without crossing into `P1`.

This pass finishes the semantic boundary that is still missing after the inner-solver foundation landed:

- strict mixed training must use a dedicated outer-objective path
- strict mixed loss must exclude legacy `E_int` / `E_sigma` dominance
- trainer phase flags must map to explicit route modes
- training logs must surface convergence, fallback, skip, and continuation-freeze signals

## Scope Decision

Three implementation shapes were considered:

1. Keep patching the existing `TotalEnergy.energy(...)` path and zero out a few weights.
2. Add a dedicated strict-mixed objective path while leaving legacy `energy(...)` and `residual(...)` intact.
3. Build the whole strict objective directly in `Trainer`, bypassing `TotalEnergy`.

Approved choice: **2**.

Reason:

- It is narrow enough for `P0`.
- It keeps strict mixed semantics explicit.
- It avoids pulling `P1` `ContactOperator` unification into the same change.
- It reuses the existing elasticity / BC / tightening plumbing without duplicating staged-load logic in `Trainer`.

## Functional Requirements

### 1. Dedicated strict mixed training objective

Strict mixed training should no longer behave like "legacy `TotalEnergy.energy(...)` plus a few extra contact stats".

Instead, add a dedicated `TotalEnergy` entry point that computes only the outer terms that are still valid in strict mixed mode:

- strict traction-matching contact residuals
- `E_eq`
- `E_bc`
- `E_tight`
- `E_data`
- `E_smooth`
- `E_unc`
- `E_reg`

It must not make `E_int` or `E_sigma` part of the active optimization profile in this mode.

### 2. Explicit route modes in trainer

Trainer phase flags already exist but are not yet real execution routes.

For `P0`, the route table should be:

- `phase0` -> legacy path
- strict mixed with both IFT flags off -> `forward_only`
- strict mixed with `normal_ift_enabled=True` and tangential IFT off -> `normal_ready`
- strict mixed with tangential IFT requested -> fail fast as unsupported in `P0`

`normal_ready` is still a forward solve in this pass, but it must be a dedicated branch with explicit route labeling so later normal IFT can plug in without changing the outer interface.

### 3. Outer-loss profile in strict mixed mode

The approved outer-loss profile is:

- keep: `E_cn`, `E_ct`, `E_eq`, `E_bc`, `E_tight`, `E_data`, `E_smooth`, `E_unc`, `E_reg`
- disable: `E_int`, `E_sigma`, `E_bi`, `E_ed`, `path_penalty_total`, `fric_path_penalty_total`

`E_bi` may still be computed as a diagnostic or numerical regularizer, but it must not be treated as an active optimization term in strict mixed `P0`.

This change must be visible in both optimization and logging so the reported weights match the real active profile.

### 4. Training-level diagnostics

Single-step inner diagnostics already exist. `P0` still needs trainer-visible aggregate signals:

- `inner_convergence_rate`
- `inner_fallback_rate`
- `inner_skip_rate`
- `strict_route_mode`
- `continuation_frozen`
- `continuation_freeze_events`

These should accumulate across actual strict mixed objective evaluations, not just echo the last batch.

## Architecture

### `TotalEnergy`

Add a dedicated strict-mixed objective path, implemented alongside the existing `energy(...)` and `residual(...)` methods rather than replacing them.

That path will:

- call the strict contact assembly already built around `solve_contact_inner(...)`
- use residual-style elasticity terms for `E_eq` / `E_reg`
- use residual-style BC and tightening terms
- reuse existing data and uncertainty helpers
- intentionally leave `E_int` and `E_sigma` at zero

Legacy `energy(...)` and `update_multipliers(...)` remain stable for baseline ALM behavior.

### `TrainerOptMixin`

Add one route resolver and one active-loss-profile helper:

- route resolver: `legacy`, `forward_only`, `normal_ready`, `full_ift_unsupported`
- active-loss-profile helper: zeroes disallowed terms in strict mixed mode and exposes the effective profile to logging

The actual optimizer step should dispatch to:

- legacy objective for `phase0`
- strict mixed objective for `forward_only`
- strict mixed objective with `normal_ready` route label for `phase2a`
- `NotImplementedError` for tangential IFT / full IFT

### `Trainer` runtime state

Add small runtime counters for strict mixed batches:

- total strict batches
- converged batches
- fallback batches
- skipped batches

Also track contact-hardening freeze transitions so continuation freeze events can be surfaced honestly.

## Error Handling

- If strict mixed mode is requested but tangential IFT is also requested, stop with a clear `NotImplementedError`.
- If strict mixed contact assembly must skip a batch, return zeroed contact contribution for that batch and mark the skip in diagnostics instead of silently falling back to legacy ALM contact enforcement.
- If the stress head is unavailable in strict mixed mode, mark the batch as skipped rather than routing back to legacy contact semantics.

## Tests

`P0` closeout needs new red-green coverage for:

- strict mixed active loss profile excluding `E_int` / `E_sigma`
- route resolution for `forward_only`, `normal_ready`, and `full_ift_unsupported`
- training-level convergence / fallback / skip counters
- continuation freeze diagnostics
- log-format integration for the new strict mixed trainer metrics

## Non-Goals

- Do not make `ContactOperator` a full thin adapter in this pass.
- Do not implement normal IFT linear solves yet.
- Do not implement tangential IFT.
- Do not redesign `pinn_model.py` or graph/stress architecture in this pass.

## Success Criteria

`P0` is complete when:

- strict mixed training uses a dedicated semantic route, not the legacy total-energy path as its main meaning
- active strict mixed optimization excludes `E_int` and `E_sigma`
- trainer flags produce explicit route modes
- convergence, fallback, skip, and continuation-freeze metrics are visible in training stats/logs
- legacy ALM training remains intact
