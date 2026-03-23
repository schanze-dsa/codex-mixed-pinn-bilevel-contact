# Normal-Only IFT Closure Design

## Goal

Add the smallest focused validation needed to prove the `normal_ready` route
both requests inner linearization and turns it into observable trainer stats.

## Current Gap

The repository already proves that:

- strict mixed routing can choose `normal_ready`
- the inner solver can emit linearization payloads
- upper layers surface `ift_linear_residual`

But the image-driven acceptance criteria depend on four explicit stats that do
not exist yet:

- `normal_ift_ready`
- `normal_ift_consumed`
- `normal_ift_condition_metric`
- `normal_ift_valid_ratio`

Runtime gradient split stats (`grad_u_norm`, `grad_sigma_norm`) are also wired
for display/extraction but are not produced during the compiled train step.

## Chosen Approach

1. Add three focused tests matching the image workflow:
   - `test_layout_roundtrip.py`
   - `test_mixed_normal_ready_ift_consumes_linearization.py`
   - `test_mixed_forward_only_vs_normal_ready.py`
2. Plumb the missing `normal_ift_*` stats from strict mixed contact terms.
3. Emit `grad_u_norm` and `grad_sigma_norm` from the compiled trainer step using
   a lightweight variable-name-based split between displacement/shared vars and
   stress-head vars.

## Why This Approach

- It keeps the change local to existing strict mixed statistics code.
- It directly matches the acceptance criteria in the screenshot.
- It avoids introducing a heavy full-training integration test.

## Non-Goals

- No change to route selection logic.
- No tangential/full IFT implementation.
- No full training run as part of this closure check.
