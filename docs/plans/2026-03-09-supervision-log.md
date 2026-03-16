# Supervision Log Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Show supervision loss and supervision error metrics in step-level training logs.

**Architecture:** Extend the existing training log formatter instead of adding a new logging path. Keep the change local to `TrainerMonitorMixin` so formatting stays centralized and unsupervised runs remain unchanged.

**Tech Stack:** Python, TensorFlow, `unittest`

---

### Task 1: Cover supervision energy summary

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_monitor_mixin.py`

**Step 1: Write the failing test**
- Add a unit test for `_format_energy_summary()` that expects `Edata=...` to appear when `E_data` is present and weighted.

**Step 2: Run test to verify it fails**
- Run `python -m unittest test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_format_energy_summary_includes_supervision_term`

**Step 3: Write minimal implementation**
- Add `E_data` to the energy display list and weight lookup.

**Step 4: Run test to verify it passes**
- Re-run the targeted unittest command.

### Task 2: Cover supervision error metrics in postfix

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_monitor_mixin.py`

**Step 1: Write the failing test**
- Add a unit test for `_format_train_log_postfix()` that expects `drms` and `dmae` when `data_rms` and `data_mae` are present.

**Step 2: Run test to verify it fails**
- Run `python -m unittest test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_format_train_log_postfix_includes_supervision_error_metrics`

**Step 3: Write minimal implementation**
- Append formatted supervision stats only when they are available in `stats`.

**Step 4: Run test to verify it passes**
- Re-run the targeted unittest command.

### Task 3: Verify regression surface

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`

**Step 1: Run focused verification**
- Run `python -m unittest test_trainer_optimization_hooks test_main_new_config_override`

**Step 2: Inspect diff**
- Run `git diff -- src/train/trainer_monitor_mixin.py test_trainer_optimization_hooks.py`
