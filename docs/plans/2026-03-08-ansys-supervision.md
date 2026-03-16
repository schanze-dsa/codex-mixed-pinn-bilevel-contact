# ANSYS Mirror Supervision Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional 3-stage ANSYS mirror-node supervision on top of the existing staged contact PINN training route.

**Architecture:** Load a PINN-aligned case table plus rigid-removed stage labels into a fixed-shape supervision dataset, feed supervised cases through the existing staged preload encoder, and add a new `E_data` term inside `TotalEnergy` so data loss and physics loss are optimized together.

**Tech Stack:** Python, TensorFlow, NumPy, pandas, existing `src/train` and `src/model` modules.

---

### Task 1: Add failing tests for supervision dataset loading

**Files:**
- Create: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_ansys_supervision_dataset.py`
- Create: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\ansys_supervision.py`

**Step 1: Write the failing test**

```python
def test_load_supervision_cases_maps_case_table_and_stage_csvs():
    # synthetic case csv + 3 stage csvs + fake asm.nodes
    # assert P/order/X_obs/U_obs shapes and values
    ...
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_ansys_supervision_dataset.py -q`
Expected: FAIL because the loader module does not exist.

**Step 3: Write minimal implementation**

```python
def load_ansys_supervision_cases(...):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_ansys_supervision_dataset.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add test_ansys_supervision_dataset.py src/train/ansys_supervision.py
git commit -m "train: add ansys supervision dataset loader"
```

### Task 2: Add failing tests for 3-stage alignment without release stage

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_preload_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\saved_model_module.py`

**Step 1: Write the failing test**

```python
def test_build_stage_case_can_disable_release_stage():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_trainer_optimization_hooks.py -q`
Expected: FAIL because release stage is always appended.

**Step 3: Write minimal implementation**

```python
# add preload_append_release_stage config and honor it in staged params/export
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_trainer_optimization_hooks.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add test_trainer_optimization_hooks.py src/train/trainer_preload_mixin.py src/train/saved_model_module.py
git commit -m "train: make release-stage append configurable"
```

### Task 3: Add failing tests for data supervision loss

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\model\loss_energy.py`

**Step 1: Write the failing test**

```python
def test_total_energy_data_loss_is_zero_for_exact_observations():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_trainer_optimization_hooks.py -q`
Expected: FAIL because `E_data` does not exist.

**Step 3: Write minimal implementation**

```python
# add w_data, E_data, and observation MSE in TotalEnergy
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_trainer_optimization_hooks.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add test_trainer_optimization_hooks.py src/model/loss_energy.py
git commit -m "model: add staged mirror supervision loss"
```

### Task 4: Wire supervision config into trainer runtime

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_config.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_init_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_build_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_preload_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\main new.py`

**Step 1: Write the failing test**

```python
def test_supervision_cases_drive_training_sampling_when_enabled():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_trainer_optimization_hooks.py test_ansys_supervision_dataset.py -q`
Expected: FAIL because trainer still samples random/LHS cases only.

**Step 3: Write minimal implementation**

```python
# load supervision cases in build()
# sample train split cases in _sample_preload_case()
# pass X_obs/U_obs through _make_preload_params()
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_trainer_optimization_hooks.py test_ansys_supervision_dataset.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/train/trainer_config.py src/train/trainer_init_mixin.py src/train/trainer_build_mixin.py src/train/trainer_preload_mixin.py \"main new.py\" test_trainer_optimization_hooks.py test_ansys_supervision_dataset.py
git commit -m "train: wire ansys mirror supervision into staged training"
```

### Task 5: Verify focused regression coverage

**Files:**
- Modify: none

**Step 1: Run supervision tests**

Run: `python -m pytest test_ansys_supervision_dataset.py test_trainer_optimization_hooks.py -q`
Expected: PASS

**Step 2: Run existing stage/viz guardrail tests**

Run: `python -m pytest test_trainer_volume_sampling.py test_viz_stage_comparison_outputs.py -q`
Expected: PASS

**Step 3: Run a config import smoke**

Run: `python -c "import os, sys; sys.path.insert(0, r'D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src'); from train.trainer_config import TrainerConfig; print('ok', hasattr(TrainerConfig(), 'supervision'))"`
Expected: prints `ok True`

**Step 4: Summarize config needed for user**

Run: none
Expected: document the `supervision` block and `w_data` usage in the final handoff.

**Step 5: Commit**

```bash
git add .
git commit -m "feat: add optional ansys mirror supervision for staged pinn training"
```
