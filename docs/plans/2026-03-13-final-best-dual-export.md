# Final/Best Dual Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export two separate post-training result sets, one from the final checkpoint and one from the best checkpoint.

**Architecture:** Keep the existing visualization/evaluation writers unchanged for the current model state, and add a thin orchestration layer that restores `final` and `best` checkpoints into the trainer before calling the current-state export logic into separate output subdirectories.

**Tech Stack:** Python, TensorFlow checkpoints, trainer mixins, unittest

---

### Task 1: Lock the export-target resolution behavior with a failing test

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_viz_supervision_eval_outputs.py`

**Step 1: Write the failing test**

Add a test that creates a tiny dummy trainer with:

- `viz_export_final_and_best=True`
- `out_dir=<temp>`
- `_final_ckpt_path='.../ckpt-final'`
- `_best_ckpt_path='.../ckpt-best'`

Assert that the resolved export targets are:

- `<temp>/final` for final
- `<temp>/best` for best

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_viz_supervision_eval_outputs.VizSupervisionEvalOutputTests.test_resolves_dual_export_targets_for_final_and_best -v`

Expected: FAIL because the helper does not exist.

### Task 2: Lock the orchestration behavior with a failing test

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_viz_supervision_eval_outputs.py`

**Step 1: Write the failing test**

Add a test that monkeypatches:

- checkpoint restore helper
- current-state export helper

Then call `_visualize_after_training(...)` with dual export enabled and assert:

- export order is `final`, then `best`
- output directories are `.../final` and `.../best`
- the trainer restores final again before returning

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_viz_supervision_eval_outputs.VizSupervisionEvalOutputTests.test_visualize_after_training_exports_final_and_best_sets -v`

Expected: FAIL because the orchestration does not exist.

### Task 3: Add config plumbing

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_config.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\main new.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\config.yaml`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_main_new_config_override.py`

**Step 1: Add the new config field**

Add:

- `viz_export_final_and_best: bool = False`

**Step 2: Parse it from `output_config`**

Wire `output_config.viz_export_final_and_best` into the trainer config.

**Step 3: Enable it in the supervised config**

Turn it on in `config.yaml`.

**Step 4: Add/extend config regression test**

Assert the default supervised config now parses this flag as enabled.

### Task 4: Track best and final checkpoint paths

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_init_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_run_mixin.py`

**Step 1: Initialize runtime fields**

Add runtime fields for:

- `_best_ckpt_path`
- `_final_ckpt_path`

**Step 2: Record best checkpoint path**

When `_maybe_save_best_checkpoint(...)` saves successfully, store the returned path.

**Step 3: Record final checkpoint path**

When the end-of-run checkpoint save succeeds, store the returned path before visualization starts.

### Task 5: Refactor visualization export into orchestration + current-state writer

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_viz_mixin.py`

**Step 1: Extract current-state implementation**

Move the existing body of `_visualize_after_training(...)` into a helper such as:

- `_visualize_current_state_to_out_dir(...)`

This helper should assume the model is already restored and `cfg.out_dir` already points at the target directory.

**Step 2: Add export-target resolution helper**

Add a helper that returns one or two export targets based on:

- config flag
- best/final checkpoint availability

**Step 3: Add checkpoint-restore helper**

Wrap checkpoint restore + partial-restore compatibility in one method used only for export switching.

**Step 4: Rebuild `_visualize_after_training(...)` orchestration**

When dual export is enabled:

- export final to `<out_dir>/final`
- export best to `<out_dir>/best`
- restore final again before returning

When disabled:

- preserve the old single-export behavior

### Task 6: Verify focused tests

**Files:**
- Modify: none

**Step 1: Run the new targeted tests**

Run: `python -m unittest test_viz_supervision_eval_outputs.VizSupervisionEvalOutputTests.test_resolves_dual_export_targets_for_final_and_best test_viz_supervision_eval_outputs.VizSupervisionEvalOutputTests.test_visualize_after_training_exports_final_and_best_sets -v`

Expected: PASS.

**Step 2: Run broader regression tests**

Run: `python -m unittest test_viz_supervision_eval_outputs test_main_new_config_override test_trainer_optimization_hooks -v`

Expected: PASS.

### Task 7: Real-case verification

**Files:**
- Modify: none

**Step 1: Reuse the latest run directory**

Use the existing run with:

- best checkpoint around `ckpt-100`
- final checkpoint `ckpt-1500`

**Step 2: Trigger dual export**

Verify the run produces:

- `results/ansys_supervised/final/...`
- `results/ansys_supervised/best/...`

**Step 3: Confirm the scale difference is visible across directories**

Check that the `best` supervision eval metrics are materially better than `final`, and that the user can inspect both without rerunning training.

**Step 4: Report residual scope**

State clearly that this patch improves export clarity and comparison, but does not itself change model quality.
