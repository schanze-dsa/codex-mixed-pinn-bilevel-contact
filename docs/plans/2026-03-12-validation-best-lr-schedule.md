# Validation-Best Checkpoint And Late LR Decay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Save the best supervised checkpoint by validation amplitude quality and reduce the learning rate when that validation metric plateaus.

**Architecture:** Reuse the staged supervision evaluation code path already used for exported reports, but run it periodically during training on the configured validation split. Feed the resulting scalar metric into both checkpoint selection and a simple plateau LR scheduler so the trainer prefers checkpoints that reduce speckle on held-out supervised cases.

**Tech Stack:** Python, TensorFlow, unittest, existing trainer mixins and config plumbing

---

### Task 1: Lock validation metric aggregation with tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`

**Step 1: Write the failing test**

Add a test that feeds a few synthetic supervision-eval rows into the new validation summarizer and asserts:

- median ratio is computed correctly
- mean RMSE is computed correctly
- mean validation `rmse_vec_mm / true_rms_vec_mm` metric is computed correctly

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_supervision_validation_summary_aggregates_rows -v`

Expected: FAIL because the helper does not exist yet.

**Step 3: Write minimal implementation**

Do not implement here. Only add the test.

**Step 4: Run test again to confirm the intended failure**

Run: `python -m unittest test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_supervision_validation_summary_aggregates_rows -v`

Expected: FAIL with missing helper or missing metric fields.

### Task 2: Lock validation-best checkpoint behavior with tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`

**Step 1: Write the failing test**

Add a test that configures `save_best_on='val_drrms'`, stubs validation metrics across two logging steps, and asserts only the improved validation metric updates `best_metric` and triggers checkpoint save.

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_best_metric_uses_validation_drrms_when_configured -v`

Expected: FAIL because the trainer only supports `Pi`/`E_int`.

**Step 3: Write minimal implementation**

Do not implement here. Only add the test.

**Step 4: Run test again to confirm the intended failure**

Run: `python -m unittest test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_best_metric_uses_validation_drrms_when_configured -v`

Expected: FAIL with unsupported metric behavior.

### Task 3: Lock plateau LR decay with tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`

**Step 1: Write the failing test**

Add a test that simulates repeated non-improving validation metrics after warmup and asserts:

- learning rate is multiplied by `factor`
- it does not fall below `min_lr`
- patience resets after a decay

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_validation_plateau_decay_reduces_learning_rate -v`

Expected: FAIL because no scheduler exists yet.

**Step 3: Write minimal implementation**

Do not implement here. Only add the test.

**Step 4: Run test again to confirm the intended failure**

Run: `python -m unittest test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_validation_plateau_decay_reduces_learning_rate -v`

Expected: FAIL with missing scheduler logic.

### Task 4: Implement config and runtime state

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_config.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\main new.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_init_mixin.py`

**Step 1: Add config knobs**

Add fields for:

- validation metric/check interval
- LR plateau enable flag
- LR plateau warmup
- LR plateau patience
- LR plateau factor
- LR minimum

**Step 2: Parse YAML config**

Read the new keys from `optimizer_config` and/or `supervision`.

**Step 3: Initialize runtime state**

Store the latest validation summary, plateau counters, and best validation metric.

**Step 4: Run focused tests**

Run: `python -m unittest test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_supervision_validation_summary_aggregates_rows test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_best_metric_uses_validation_drrms_when_configured test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_validation_plateau_decay_reduces_learning_rate -v`

Expected: still FAIL until the trainer helpers are implemented.

### Task 5: Implement validation summary and checkpoint selection

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_viz_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_run_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_monitor_mixin.py`

**Step 1: Implement validation metric helper**

Build a helper that evaluates the validation split during training and returns a compact summary dict.

**Step 2: Wire best-metric selection**

Support `save_best_on='val_drrms'` and use the validation summary instead of training `Pi`.

**Step 3: Surface compact validation logging**

Append compact validation metrics and current LR to the log postfix when available.

**Step 4: Run focused tests**

Run: `python -m unittest test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_supervision_validation_summary_aggregates_rows test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_best_metric_uses_validation_drrms_when_configured -v`

Expected: PASS.

### Task 6: Implement plateau LR decay

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_run_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_monitor_mixin.py`

**Step 1: Add optimizer LR read/write helpers**

Safely update the live optimizer LR whether it is wrapped by mixed precision or not.

**Step 2: Add plateau scheduler update**

Decay LR after configured stagnation and reset patience after decay or improvement.

**Step 3: Run focused tests**

Run: `python -m unittest test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_validation_plateau_decay_reduces_learning_rate -v`

Expected: PASS.

### Task 7: Run focused verification

**Files:**
- Modify: none

**Step 1: Run trainer hook tests**

Run: `python -m unittest test_trainer_optimization_hooks -v`

Expected: PASS.

**Step 2: Run supervision dataset and eval-output regressions**

Run: `python -m unittest test_ansys_supervision_dataset test_viz_supervision_eval_outputs -v`

Expected: PASS.

**Step 3: Record residual risk**

If a full retrain is not run in this session, state clearly that the change is validated by unit/integration tests only.
