# Physical Scale Resolution Design

## Goal

Restore the `PhysicalScaleConfig` API expected by the current regression suite so
the strict mixed regression run can complete without changing test intent.

## Current Problem

`PhysicalScaleConfig` currently exposes `resolved_sigma_ref()` only. The focused
physical-scale tests still exercise `resolved_L_ref()` and `resolved_u_ref()`.
That interface drift causes `AttributeError` before the regression suite can
finish.

## Chosen Approach

Add back `resolved_L_ref()` and `resolved_u_ref()` in
`src/physics/physical_scales.py`, and make `resolved_sigma_ref()` use those
helpers. Each resolver will clamp invalid or non-positive values to `1.0`.

## Why This Approach

- It is the smallest change that restores the tested contract.
- It keeps normalization behavior consistent in one place.
- It avoids weakening the regression suite by deleting expectations.

## Non-Goals

- No changes to YAML parsing.
- No changes to training behavior beyond scale fallback resolution.
- No test rewrites unless the existing contract proves incorrect.
