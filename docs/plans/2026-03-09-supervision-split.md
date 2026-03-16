# Supervision Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace single-shot ratio splitting with the final experiment protocol: fixed grouped test set plus 5-fold grouped cross-validation.

**Architecture:** The supervision loader computes one deterministic protocol from `base_id` groups and `source` labels. It first chooses a fixed test pool with explicit source quotas, then partitions the remaining groups into 5 equal CV folds, and finally materializes `train`, `val`, and `test` for the configured `cv_fold_index`.

**Tech Stack:** Python, pandas, NumPy, unittest, existing `src/train` supervision loader and trainer config/runtime code.

---

### Task 1: Lock the final protocol in tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_ansys_supervision_dataset.py`

**Step 1: Write the failing tests**
- Replace ratio-based expectations with fixed-test-plus-fold expectations.
- Add one test for exact fixed test quotas at group level.
- Add one test for 5 equal folds on the remaining 25 groups.
- Keep the loader smoke test without any CSV `split` column.

**Step 2: Run the tests to verify failure**

Run: `python -m unittest test_ansys_supervision_dataset.py`
Expected: FAIL because the current helper still uses ratio-based splitting.

**Step 3: Implement the minimal protocol support**
- Change the split helper to build fixed test groups and CV folds.
- Keep group integrity at `base_id`.

**Step 4: Run the tests to verify pass**

Run: `python -m unittest test_ansys_supervision_dataset.py`
Expected: PASS

### Task 2: Wire protocol config through runtime

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_config.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\main new.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_build_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\config_ansys_supervised.yaml`

**Step 1: Add failing expectations if needed**
- Extend config expectations to include fixed test quotas, fold count, and active fold index.

**Step 2: Implement config parsing and load wiring**
- Remove obsolete ratio fields from the active supervised config path.
- Pass protocol parameters into the loader.
- Restrict CV evaluation config to `val` only for normal runs.

**Step 3: Run focused verification**

Run: `python -m unittest test_ansys_supervision_dataset.py test_viz_supervision_eval_outputs.py test_trainer_optimization_hooks.py test_main_new_config_override.py`
Expected: PASS

### Task 3: Audit the real dataset under the new protocol

**Files:**
- Verify: `D:\shuangfan\pinn luowen-worktrees\ansys_cases_180_deg2to6_step0p5_pinn.csv`

**Step 1: Run a real-data audit**
- Confirm there are 30 `base_id` groups.
- Confirm `test` contains exactly 5 groups with `1 boundary + 1 corner + 3 interior`.
- Confirm the active `cv_fold_index` gives `train/val/test = 20/5/5` groups.

**Step 2: Re-run verification**

Run: `python -m unittest test_ansys_supervision_dataset.py test_viz_supervision_eval_outputs.py test_trainer_optimization_hooks.py test_main_new_config_override.py`
Expected: PASS
