# Anti-Speckle Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce PINN speckle by enforcing a minimum supervised contribution against stress/contact terms and by adding a light spatial smoothing loss on supervised mirror nodes.

**Architecture:** Keep the existing staged supervision route, adaptive loss scheduler, and visualization pipeline. Add two small objective-side controls: a per-step supervision effective-weight floor applied after adaptive weighting, and a local kNN smoothing regularizer computed only on supervised displacement samples.

**Tech Stack:** Python, TensorFlow, unittest, existing trainer mixins, `TotalEnergy`, adaptive loss scheduler

---

### Task 1: Lock the smoothing loss behavior with a failing test

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`

**Step 1: Write the failing test**

Add a test for a small synthetic supervised node set where:

- a constant displacement field gives near-zero smoothing loss
- a checkerboard/high-frequency field gives a larger smoothing loss

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_data_smoothing_loss_penalizes_high_frequency_supervision_noise -v`

Expected: FAIL because no smoothing helper or term exists.

### Task 2: Lock the contribution-floor behavior with a failing test

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`

**Step 1: Write the failing test**

Add a test that simulates weighted parts where `E_data` is too small versus `E_sigma + E_ct`, then asserts the built weight vector raises the effective data contribution to the configured minimum ratio.

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_weight_vector_enforces_supervision_contribution_floor -v`

Expected: FAIL because the floor logic does not exist.

### Task 3: Add config plumbing

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\model\loss_energy.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_config.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\main new.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\config.yaml`

**Step 1: Add new config fields**

Add fields for:

- supervision contribution floor enable flag
- contribution floor ratio
- smoothing loss weight
- smoothing kNN size

**Step 2: Parse YAML**

Read them from config without disturbing existing defaults.

### Task 4: Implement smoothing loss in `TotalEnergy`

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\model\loss_energy.py`

**Step 1: Add a helper for supervision smoothing**

Compute local neighbor-mean vector smoothing on `X_obs` / predicted `u_pred`.

**Step 2: Add a new parts key**

Expose a separate energy part for smoothing and log its RMS-like statistic.

**Step 3: Keep it supervision-only**

Return zero when no supervised observations exist.

### Task 5: Implement the contribution floor in the trainer weighting path

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_opt_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_monitor_mixin.py`

**Step 1: Build the base adaptive weight vector as before**

Do not mutate adaptive scheduler state.

**Step 2: Add a post-processing floor helper**

Using current loss parts, raise the effective `E_data` weight only for the current step when needed.

**Step 3: Log the effective supervision weight and floor-activation flag**

This is needed for diagnosis and regression checks.

### Task 6: Verify focused tests

**Files:**
- Modify: none

**Step 1: Run new targeted tests**

Run: `python -m unittest test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_data_smoothing_loss_penalizes_high_frequency_supervision_noise test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_weight_vector_enforces_supervision_contribution_floor -v`

Expected: PASS.

**Step 2: Run broader trainer/viz regressions**

Run: `python -m unittest test_trainer_optimization_hooks test_viz_supervision_eval_outputs -v`

Expected: PASS.

### Task 7: Real-case verification

**Files:**
- Modify: none

**Step 1: Reuse the current checkpoint or retrain as needed**

Generate same-pipeline debug outputs for the known problematic `deflection_01_231` case.

**Step 2: Compare roughness diagnostics**

Check that:

- PINN/FEM local roughness ratio decreases
- validation metrics do not regress materially

**Step 3: Report residual risk**

If a full retrain is not completed in-session, state clearly that code and diagnostics are verified but final speckle reduction still depends on retraining.
