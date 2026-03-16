# Supervision Loss Scaling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make staged ANSYS supervision numerically comparable across cases by converting raw displacement MSE into amplitude-normalized relative supervision loss.

**Architecture:** Keep the existing staged supervision data flow unchanged and modify only the scalar supervision loss assembly in `TotalEnergy`. The loss will normalize residuals by the current observed displacement RMS and expose relative error metrics to the trainer log.

**Tech Stack:** Python, TensorFlow, unittest, existing trainer and loss modules

---

### Task 1: Lock the regression with loss-scaling tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`

**Step 1: Write the failing test**

Add a test that constructs two supervision batches with the same relative prediction error but different displacement amplitudes, then asserts `E_data` is equal for both.

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_trainer_optimization_hooks.py -k supervision -v`

Expected: FAIL because the current raw-MSE supervision loss is amplitude-dependent.

**Step 3: Write minimal implementation**

Do not implement here. Only add the test.

**Step 4: Run test to verify it still fails for the intended reason**

Run: `python -m pytest test_trainer_optimization_hooks.py -k supervision -v`

Expected: FAIL with mismatched `E_data`.

### Task 2: Normalize supervision loss by observed RMS

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\model\loss_energy.py`

**Step 1: Implement relative supervision loss**

Compute:

- `data_ref_rms = sqrt(mean(U_obs^2))`
- `diff_rel = diff / max(data_ref_rms, eps)`
- `loss = mean(diff_rel^2)`

Keep absolute `data_rms` and `data_mae` stats for readability, and add:

- `data_ref_rms`
- `data_rel_rms`
- `data_rel_mae`

**Step 2: Run tests**

Run: `python -m pytest test_trainer_optimization_hooks.py -k supervision -v`

Expected: PASS for the new scaling regression and the existing exact/inexact supervision tests.

### Task 3: Surface the relative metrics in training logs

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_monitor_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`

**Step 1: Write the failing log-format test**

Extend the train-log formatting test so it expects relative supervision metrics when they are present.

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_trainer_optimization_hooks.py -k relative -v`

Expected: FAIL because the formatter does not yet emit the relative values.

**Step 3: Write minimal implementation**

Append compact fields such as:

- `dref=...`
- `drrms=...`
- `drmae=...`

Only show fields that exist in `stats`.

**Step 4: Run tests to verify it passes**

Run: `python -m pytest test_trainer_optimization_hooks.py -k relative -v`

Expected: PASS.

### Task 4: Run focused verification

**Files:**
- Modify: none

**Step 1: Run the focused test file**

Run: `python -m pytest test_trainer_optimization_hooks.py -v`

Expected: PASS.

**Step 2: Run the supervision dataset and visualization-adjacent regression tests**

Run: `python -m pytest test_ansys_supervision_dataset.py test_viz_supervision_eval_outputs.py -v`

Expected: PASS.

**Step 3: Record any remaining risk**

If training is not rerun in this session, note that the code-path fix is verified by tests only, not by a fresh full training run.
