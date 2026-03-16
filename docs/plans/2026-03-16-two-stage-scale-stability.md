# Two-Stage Scale Stability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a two-stage training workflow that preserves early supervised scale alignment by handing Phase 1 best checkpoint into a shorter, lower-risk Phase 2 refinement run.

**Architecture:** Keep the current `Trainer` mostly unchanged and orchestrate two phases from `main new.py`. Parse a compact `two_stage_training` config block, derive phase-specific `TrainerConfig` overrides, run Phase 1 normally, resume Phase 2 from the Phase 1 best checkpoint, and preserve per-phase outputs and checkpoints for inspection.

**Tech Stack:** Python, TensorFlow, unittest, YAML config parsing, trainer mixins, checkpoint manager

---

### Task 1: Lock two-stage config parsing with failing tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_main_new_config_override.py`

**Step 1: Write the failing test**

Add a test that feeds `_prepare_config_with_autoguess()` a YAML payload containing:

- `two_stage_training.enabled: true`
- `phase1` overrides for `max_steps`, `learning_rate`, `save_best_on`, and `supervision_contribution_floor_ratio`
- `phase2` overrides for the same keys

Assert that the parsed config:

- enables two-stage mode
- keeps the base single-stage defaults unchanged outside the phase override blocks
- stores Phase 1 and Phase 2 override values in dedicated config objects

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_main_new_config_override.MainNewConfigOverrideTests.test_prepare_config_parses_two_stage_training_controls -v`

Expected: FAIL because `TrainerConfig` and YAML parsing do not yet support `two_stage_training`.

### Task 2: Add two-stage config dataclasses and YAML parsing

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_config.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\main new.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_main_new_config_override.py`

**Step 1: Add minimal config dataclasses**

In `trainer_config.py`, add compact dataclasses for:

- two-stage enable flag
- phase override blocks
- phase-resume checkpoint path if needed by the trainer-side restore hook

Keep the override surface narrow:

- `max_steps`
- `lr`
- `save_best_on`
- validation cadence / plateau settings / early-exit settings
- selected base loss-weight overrides
- `supervision_contribution_floor_ratio`

**Step 2: Parse the YAML block**

In `main new.py`, parse `two_stage_training.phase1` and `two_stage_training.phase2` into those config objects.

Rules:

- omitted override keys inherit from the base config
- unrelated config families are not duplicated per phase
- phase blocks are stored as data, not applied immediately during base config parsing

**Step 3: Run the targeted parsing test**

Run: `python -m unittest test_main_new_config_override.MainNewConfigOverrideTests.test_prepare_config_parses_two_stage_training_controls -v`

Expected: PASS.

### Task 3: Lock phase override application with failing tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_main_new_config_override.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\main new.py`

**Step 1: Write the failing test**

Add a test for a new helper that derives a phase config from the base config.

Assert that:

- Phase 1 override changes only the intended training-control fields
- `total_cfg.w_data`, `total_cfg.w_smooth`, `data_smoothing_k`, `loss_focus_terms`, and dataset paths are inherited unchanged
- phase-specific `out_dir` and `ckpt_dir` are separated into `phase1` / `phase2` subdirectories

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_main_new_config_override.MainNewConfigOverrideTests.test_two_stage_phase_override_keeps_non_phase_fields_stable -v`

Expected: FAIL because the phase-config derivation helper does not yet exist.

### Task 4: Implement phase-config cloning and directory layout

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\main new.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_main_new_config_override.py`

**Step 1: Add a phase-config clone helper**

Implement a helper that:

- deep-copies the base `TrainerConfig`
- applies only supported overrides
- injects phase-specific `out_dir` and `ckpt_dir`
- labels the phase for logging

**Step 2: Keep inheritance explicit**

Do not mutate the original base config in-place.

**Step 3: Re-run the targeted override test**

Run: `python -m unittest test_main_new_config_override.MainNewConfigOverrideTests.test_two_stage_phase_override_keeps_non_phase_fields_stable -v`

Expected: PASS.

### Task 5: Lock checkpoint handoff behavior with failing tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_run_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer.py`

**Step 1: Write the failing test**

Add a focused test that exercises a small resume helper or phase-result helper and asserts:

- the trainer records its best checkpoint path during a run
- the trainer records its final checkpoint path at end-of-run
- a configured resume checkpoint path is restored before training begins

Prefer testing a narrow helper instead of trying to run the whole training loop.

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_trainer_records_phase_checkpoint_paths_for_two_stage_resume -v`

Expected: FAIL because the trainer does not yet expose stable run-result checkpoint paths for orchestration.

### Task 6: Add trainer-side checkpoint restore and run-result plumbing

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_run_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`

**Step 1: Add a small restore hook**

Implement a trainer-side helper that restores a configured checkpoint after build/checkpoint-manager initialization and before the training loop starts.

**Step 2: Record phase outputs**

Store both:

- best checkpoint path chosen during training
- final checkpoint path saved at the end of the run

Expose them in a small run-result structure or stable trainer attributes used by the outer orchestrator.

**Step 3: Re-run the targeted checkpoint test**

Run: `python -m unittest test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_trainer_records_phase_checkpoint_paths_for_two_stage_resume -v`

Expected: PASS.

### Task 7: Lock outer two-stage orchestration with failing tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_main_new_config_override.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\main new.py`

**Step 1: Write the failing test**

Add a test for a new orchestration helper in `main new.py` that patches the single-phase runner and asserts:

- Phase 1 runs first
- Phase 2 receives the Phase 1 best checkpoint path as its resume input
- Phase 2 does not use the Phase 1 final checkpoint when a best checkpoint exists
- both phase results are returned to the caller

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_main_new_config_override.MainNewConfigOverrideTests.test_two_stage_training_resumes_phase2_from_phase1_best_checkpoint -v`

Expected: FAIL because no such orchestration helper exists yet.

### Task 8: Implement the outer two-stage run path

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\main new.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_main_new_config_override.py`

**Step 1: Factor the current single-run path**

Extract the existing training body into a helper that runs exactly one phase and returns:

- trainer instance or a small run result
- best checkpoint path
- final checkpoint path
- phase output directory

**Step 2: Add two-stage orchestration**

When `two_stage_training.enabled` is true:

- derive Phase 1 config
- run Phase 1
- validate that a Phase 1 best checkpoint exists
- derive Phase 2 config with `resume_ckpt_path=phase1_best`
- run Phase 2

When disabled, keep the existing single-stage behavior unchanged.

**Step 3: Keep logging explicit**

Print clear messages for:

- phase start/end
- chosen Phase 1 handoff checkpoint
- Phase 2 resume checkpoint path

**Step 4: Re-run the targeted orchestration test**

Run: `python -m unittest test_main_new_config_override.MainNewConfigOverrideTests.test_two_stage_training_resumes_phase2_from_phase1_best_checkpoint -v`

Expected: PASS.

### Task 9: Verify targeted regressions and broader config/trainer tests

**Files:**
- Modify: none

**Step 1: Run targeted two-stage tests**

Run: `python -m unittest test_main_new_config_override.MainNewConfigOverrideTests.test_prepare_config_parses_two_stage_training_controls test_main_new_config_override.MainNewConfigOverrideTests.test_two_stage_phase_override_keeps_non_phase_fields_stable test_main_new_config_override.MainNewConfigOverrideTests.test_two_stage_training_resumes_phase2_from_phase1_best_checkpoint test_trainer_optimization_hooks.TrainerOptimizationHookTests.test_trainer_records_phase_checkpoint_paths_for_two_stage_resume -v`

Expected: PASS.

**Step 2: Run broader regression suite**

Run: `python -m unittest test_main_new_config_override test_trainer_optimization_hooks -v`

Expected: PASS.

### Task 10: Do a config-level smoke check for the two-stage path

**Files:**
- Modify: none

**Step 1: Run a lightweight smoke script**

Using a patched or synthetic config, confirm that:

- Phase 1 and Phase 2 configs resolve to distinct `out_dir` / `ckpt_dir`
- Phase 2 inherits non-phase settings unchanged
- Phase 2 resume path resolves to the Phase 1 best checkpoint

**Step 2: Record runtime expectations**

The next real run should produce:

- separate `phase1` and `phase2` artifact directories
- a logged Phase 1 best checkpoint handoff
- a Phase 2 resume path that points to Phase 1 best, not Phase 1 final

### Task 11: Real-run follow-up

**Files:**
- Modify: none

**Step 1: Retrain with the two-stage config**

Run the ANSYS supervised training entrypoint with two-stage mode enabled.

**Step 2: Compare phase outputs**

Check:

- Phase 1 best supervision metrics
- Phase 2 best supervision metrics
- Phase 2 final versus Phase 2 best

**Step 3: Report residual risk**

If the full retraining is not completed in-session, report code/test verification separately from unresolved runtime-quality results.
