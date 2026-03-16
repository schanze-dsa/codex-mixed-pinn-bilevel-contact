# Test Visualization Comparison Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export final-report comparison figures that show representative held-out `test` cases as `PINN | FEM | error` triptychs, while preserving full quantitative `test` evaluation outputs.

**Architecture:** Extend the existing supervision evaluation path in `TrainerVizMixin` so it can (1) compute final-stage case metrics for the configured eval split, (2) deterministically select one representative case per `source` category using median-nearest RMSE, and (3) render a three-panel comparison figure using the exact supervision node ordering and final-stage predictions. Keep the existing split-level CSV and heatmap export intact and add a separate summary CSV for the selected comparison cases.

**Tech Stack:** Python, NumPy, Matplotlib, unittest, existing `src/train` visualization and supervision loader code.

---

### Task 1: Lock representative-case selection in tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_viz_supervision_eval_outputs.py`

**Step 1: Write the failing test**

Add a focused test that builds synthetic per-case final-stage metrics and verifies:
- one selected case per source
- selected case is the one nearest the median `rmse_vec_mm` within each source

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_viz_supervision_eval_outputs.py`
Expected: FAIL because no representative-selection helper exists yet.

**Step 3: Write minimal implementation**

Implement a small helper in `TrainerVizMixin` that:
- groups rows by `source`
- keeps only the final stage per `case_id`
- chooses the median-nearest case per requested source

**Step 4: Run test to verify it passes**

Run: `python -m unittest test_viz_supervision_eval_outputs.py`
Expected: PASS for the new selection test.

### Task 2: Lock triptych export in tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_viz_supervision_eval_outputs.py`

**Step 1: Write the failing test**

Add a second test that:
- builds a dummy supervised dataset with `boundary`, `corner`, and `interior` test cases
- runs the comparison export path
- asserts that:
  - three comparison PNGs are created
  - one summary CSV is created
  - file names and summary rows contain the chosen case IDs

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_viz_supervision_eval_outputs.py`
Expected: FAIL because no comparison export path exists yet.

**Step 3: Write minimal implementation**

Add the comparison export logic in `TrainerVizMixin` using:
- final-stage predictions from the same supervision evaluation path
- case coordinates and FEM displacements from `_supervision_dataset`
- Matplotlib subplot rendering for the three panels

**Step 4: Run test to verify it passes**

Run: `python -m unittest test_viz_supervision_eval_outputs.py`
Expected: PASS

### Task 3: Wire config and runtime trigger

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_config.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\main new.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\config_ansys_supervised.yaml`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_viz_mixin.py`

**Step 1: Write the failing test**

If needed, extend `test_main_new_config_override.py` or `test_trainer_optimization_hooks.py` with expectations for new output-config fields such as:
- comparison export enabled flag
- comparison split name
- representative source list

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_main_new_config_override.py test_trainer_optimization_hooks.py`
Expected: FAIL because the new config fields are not yet parsed or stored.

**Step 3: Write minimal implementation**

Add config fields for:
- enabling representative comparison export
- target split, default `test`
- requested source labels, default `boundary,corner,interior`

Parse them in `main new.py` and store them on `TrainerConfig`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest test_main_new_config_override.py test_trainer_optimization_hooks.py`
Expected: PASS

### Task 4: Integrate comparison export into supervision evaluation flow

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_viz_mixin.py`

**Step 1: Write the failing integration test**

Add or extend a test so that after `_write_supervision_eval_outputs()` runs:
- standard eval CSV still exists
- standard eval heatmap still exists
- comparison artifacts also exist when enabled

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_viz_supervision_eval_outputs.py`
Expected: FAIL because the comparison export is not yet called from the supervision export path.

**Step 3: Write minimal implementation**

Refactor `_write_supervision_eval_outputs()` to:
- keep collecting row metrics as before
- retain enough final-stage per-case data for representative export
- call the comparison export helper after split CSV/heatmap generation

**Step 4: Run test to verify it passes**

Run: `python -m unittest test_viz_supervision_eval_outputs.py`
Expected: PASS

### Task 5: Focused regression verification

**Files:**
- Verify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_viz_supervision_eval_outputs.py`
- Verify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_viz_stage_comparison_outputs.py`
- Verify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`
- Verify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_main_new_config_override.py`

**Step 1: Run focused tests**

Run: `python -m unittest test_viz_supervision_eval_outputs.py test_viz_stage_comparison_outputs.py test_trainer_optimization_hooks.py test_main_new_config_override.py`
Expected: PASS

**Step 2: Run a repo-level sanity subset if stable**

Run: `python -m unittest test_ansys_supervision_dataset.py test_viz_supervision_eval_outputs.py test_viz_stage_comparison_outputs.py`
Expected: PASS

**Step 3: Review output naming**

Confirm the final artifact set is small and report-oriented:
- full `test` metrics CSV and heatmap
- three representative comparison PNGs
- one comparison summary CSV
