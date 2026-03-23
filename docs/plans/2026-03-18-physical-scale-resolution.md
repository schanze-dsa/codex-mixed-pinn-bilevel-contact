# Physical Scale Resolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore the missing `PhysicalScaleConfig` resolution helpers so the strict mixed regression suite passes again.

**Architecture:** Keep the repair local to `PhysicalScaleConfig`. Reintroduce the missing length and displacement resolver methods, clamp invalid values to positive defaults, and reuse those helpers inside stress-scale resolution so all fallback logic stays consistent.

**Tech Stack:** Python 3.8, unittest, dataclasses

---

### Task 1: Repair Physical Scale Resolver API

**Files:**
- Modify: `src/physics/physical_scales.py`
- Test: `test_physical_scales.py`
- Test: `test_physical_scales_config.py`

**Step 1: Write the failing test**

Use the existing focused regression tests:

```python
cfg = PhysicalScaleConfig(L_ref=10.0, u_ref=0.5, sigma_ref=12.0)
assert cfg.resolved_L_ref() == 10.0
assert cfg.resolved_u_ref() == 0.5
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_physical_scales test_physical_scales_config -v`

Expected: `AttributeError` for missing `resolved_L_ref`

**Step 3: Write minimal implementation**

- Add `resolved_L_ref()` with positive fallback.
- Add `resolved_u_ref()` with positive fallback.
- Update `resolved_sigma_ref()` to use the resolved helpers.

**Step 4: Run test to verify it passes**

Run: `python -m unittest test_physical_scales test_physical_scales_config -v`

Expected: all focused physical-scale tests pass.

**Step 5: Run regression verification**

Run:

```bash
python -m unittest test_contact_inner_kernel_primitives test_contact_inner_solver test_contact_inner_solver_linearization test_contact_inner_solver_health_policy test_mixed_bilevel_diagnostics test_mixed_contact_matching test_mixed_elasticity_residuals test_mixed_model_outputs test_mixed_training_phase_controls test_mixed_voigt_traction_utils test_physical_scales test_physical_scales_config
```

Expected: full image-driven regression suite passes.
