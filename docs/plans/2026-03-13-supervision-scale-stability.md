# Supervision Scale Stability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stabilize supervised displacement amplitude by preventing adaptive weighting from rescaling `E_data` and by strengthening the existing supervision contribution floor.

**Architecture:** Keep the current anti-speckle smoothing loss and post-weight contribution-floor logic. Narrow the change to trainer configuration and loss-weight scheduling: exclude `E_data` from adaptive focus terms, raise the supervision floor default, and verify this through focused regression tests and checkpoint-metric inspection.

**Tech Stack:** Python, TensorFlow, unittest, YAML config parsing, trainer mixins, adaptive loss scheduler

---

### Task 1: Lock the config behavior with failing tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_main_new_config_override.py`

**Step 1: Write the failing test**

Update the supervised-defaults test so it asserts:

- `cfg.supervision_contribution_floor_ratio == 1.0e-1`
- `"E_data"` is not present in `cfg.loss_focus_terms`
- physics-side focus terms still remain present

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_main_new_config_override.MainNewConfigOverrideTests.test_default_config_matches_supervised_defaults -v`

Expected: FAIL because the current config still keeps `E_data` in focus terms and uses the weaker floor default.

### Task 2: Lock runtime floor behavior with a focused regression test

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`

**Step 1: Write the failing test**

Add a test that simulates:

- base `w_data` held fixed
- strong `E_sigma` and `E_ct`
- a configured floor ratio of `0.1`

Assert that `_apply_supervision_contribution_floor(...)` raises the effective data contribution to at least the configured fraction of the physics contribution.

**Step 2: Run test to verify it fails if needed**

Run: `python -m unittest test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_weight_vector_enforces_supervision_contribution_floor -v`

Expected: If the current fixture encodes `0.02`-style behavior or assumptions, it should fail until updated to the new target.

### Task 3: Remove `E_data` from adaptive focus parsing

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\main new.py`

**Step 1: Filter parsed focus terms**

When parsing YAML adaptive focus terms, skip `w_data -> E_data` for the supervised-default path used by this trainer change.

**Step 2: Preserve physics-side adaptive behavior**

Do not change the existing auto-add of `E_sigma`. Keep contact/physics focus terms intact.

**Step 3: Keep user intent explicit**

Do not silently drop unrelated terms; only exclude supervision.

### Task 4: Raise the default supervision floor ratio

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\config.yaml`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_main_new_config_override.py`

**Step 1: Update the supervised config**

Change:

- `supervision_contribution_floor_ratio: 2.0e-2`

to:

- `supervision_contribution_floor_ratio: 1.0e-1`

**Step 2: Keep smoothing untouched**

Do not change `w_smooth` or `data_smoothing_k` in this patch.

### Task 5: Verify focused regression tests

**Files:**
- Modify: none

**Step 1: Run the focused tests**

Run: `python -m unittest test_main_new_config_override.MainNewConfigOverrideTests.test_default_config_matches_supervised_defaults test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_weight_vector_enforces_supervision_contribution_floor -v`

Expected: PASS.

**Step 2: Run broader config/trainer regressions**

Run: `python -m unittest test_main_new_config_override test_trainer_optimization_hooks -v`

Expected: PASS.

### Task 6: Re-check the loss-path expectations

**Files:**
- Modify: none

**Step 1: Inspect runtime config parsing**

Confirm from a lightweight config smoke check that:

- `cfg.loss_focus_terms` excludes `E_data`
- `cfg.supervision_contribution_floor_ratio == 0.1`

**Step 2: Record expected training interpretation**

The next run should show:

- `w_data` no longer adaptively drifting
- floor diagnostics still available through `dsmw` / `dsmf`
- amplitude drift judged mainly by `vrat`

### Task 7: Post-change verification on a real run

**Files:**
- Modify: none

**Step 1: Retrain with the updated config**

Run the supervised training entrypoint with the existing ANSYS config.

**Step 2: Compare best vs final outputs**

Check:

- `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\results\ansys_supervised\best\supervision_eval_val.csv`
- `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\results\ansys_supervised\final\supervision_eval_val.csv`

**Step 3: Report residual risk**

If full retraining is not completed in-session, report the code/test verification separately from the unresolved runtime training outcome.
