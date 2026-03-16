# Same-Pipeline FEM Debug Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export matched PINN and FEM mirror deflection figures through the same visualization pipeline so stage-plot speckle can be diagnosed without rendering-path ambiguity.

**Architecture:** Reuse the loaded staged supervision dataset to find the ANSYS case corresponding to each visualization preload case. Build a surface-only visualization context from the staged node IDs, then call the existing `plot_mirror_deflection_by_name(...)` renderer twice: once with the PINN model and once with a FEM lookup adapter.

**Tech Stack:** Python, TensorFlow, unittest, existing trainer visualization mixin and `mirror_viz.py`

---

### Task 1: Lock case matching with a failing test

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_viz_supervision_eval_outputs.py`

**Step 1: Write the failing test**

Add a test asserting that a preload case with a given `P` and `order` matches the expected supervision case across dataset splits.

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_viz_supervision_eval_outputs.VizSupervisionEvalOutputTests.test_matches_supervision_case_by_preload_and_order -v`

Expected: FAIL because the helper does not exist yet.

**Step 3: Write minimal implementation**

Do not implement here. Only add the test.

**Step 4: Run test again to confirm the intended failure**

Run: `python -m unittest test_viz_supervision_eval_outputs.VizSupervisionEvalOutputTests.test_matches_supervision_case_by_preload_and_order -v`

Expected: FAIL with missing helper.

### Task 2: Lock same-pipeline export behavior with a failing test

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_viz_supervision_eval_outputs.py`

**Step 1: Write the failing test**

Add a test that enables the debug export, patches `plot_mirror_deflection_by_name`, runs the export helper, and asserts:

- both PINN and FEM exports are attempted
- final-stage and stage-level files get the expected `_samepipe_pinn` / `_samepipe_fem` names
- the renderer is called with a restricted surface-only assembly node set

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_viz_supervision_eval_outputs.VizSupervisionEvalOutputTests.test_exports_same_pipeline_debug_pairs_for_matching_supervision_case -v`

Expected: FAIL because the helper/config does not exist yet.

**Step 3: Write minimal implementation**

Do not implement here. Only add the test.

**Step 4: Run test again to confirm the intended failure**

Run: `python -m unittest test_viz_supervision_eval_outputs.VizSupervisionEvalOutputTests.test_exports_same_pipeline_debug_pairs_for_matching_supervision_case -v`

Expected: FAIL with missing helper or attribute.

### Task 3: Add config plumbing

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_config.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\main new.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\config.yaml`

**Step 1: Add a gated debug flag**

Add a config field for the same-pipeline debug export and default it to `false`.

**Step 2: Parse YAML**

Read the new flag from `output_config`.

**Step 3: Keep defaults conservative**

Do not enable it automatically in `config.yaml` unless the user wants it on by default.

### Task 4: Implement matching and export helpers

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_viz_mixin.py`

**Step 1: Implement supervision-case matching**

Add a helper that searches the loaded supervision dataset for a case matching a visualization preload vector and tightening order.

**Step 2: Implement a surface-only visualization assembly helper**

Build a lightweight assembly object whose node set is restricted to the supervision case node IDs.

**Step 3: Implement FEM lookup `u_fn`**

Return exact staged FEM vectors for queried surface coordinates.

**Step 4: Implement the debug export writer**

Call `plot_mirror_deflection_by_name(...)` for:

- final-stage PINN
- final-stage FEM
- each plotted stage PINN
- each plotted stage FEM

### Task 5: Wire the helper into post-training visualization

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_viz_mixin.py`

**Step 1: Hook after the normal case/stage plots**

Run the same-pipeline export only when the new flag is enabled and a matching supervision case exists.

**Step 2: Keep failure mode soft**

On missing supervision data or missing match, log and continue without affecting normal outputs.

### Task 6: Run focused verification

**Files:**
- Modify: none

**Step 1: Run new focused tests**

Run: `python -m unittest test_viz_supervision_eval_outputs.VizSupervisionEvalOutputTests.test_matches_supervision_case_by_preload_and_order test_viz_supervision_eval_outputs.VizSupervisionEvalOutputTests.test_exports_same_pipeline_debug_pairs_for_matching_supervision_case -v`

Expected: PASS.

**Step 2: Run broader visualization regressions**

Run: `python -m unittest test_viz_supervision_eval_outputs test_viz_stage_comparison_outputs -v`

Expected: PASS.

**Step 3: Record residual risk**

State clearly that the new helper is verified by unit/integration tests in-session, but real-image judgment still requires running the training visualization with the flag enabled.
