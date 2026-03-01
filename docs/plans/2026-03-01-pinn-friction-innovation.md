# PINN Friction Innovation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add publishable innovation hooks (finite spectral + CDB semantic encoding + uncertainty calibration scaffolding) while preserving existing training behavior.

**Architecture:** Extend `DisplacementNet` input/heads in a backward-compatible way, compute node-level engineering semantics in `Trainer`, and add calibration utilities without forcing loss changes in baseline runs.

**Tech Stack:** Python, TensorFlow, NumPy, existing DFEM/PINN modules in `src/`.

---

### Task 1: Add failing tests for new model hooks

**Files:**
- Create: `test_model_innovation_hooks.py`
- Test: `test_model_innovation_hooks.py`

**Step 1: Write the failing test**

```python
def test_displacement_model_supports_finite_spectral_semantic_and_uncertainty():
    # Build model with new flags enabled and assert forward outputs shapes.
    ...
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_model_innovation_hooks.py -q`
Expected: FAIL because config fields/APIs do not exist yet.

**Step 3: Write minimal implementation**

```python
# add new FieldConfig flags + forward paths + uvar_fn
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_model_innovation_hooks.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add test_model_innovation_hooks.py src/model/pinn_model.py
git commit -m "model: add spectral-semantic and uncertainty hooks"
```

### Task 2: Add semantic feature extraction in trainer

**Files:**
- Modify: `src/train/trainer.py`
- Test: `test_model_innovation_hooks.py`

**Step 1: Write the failing test**

```python
def test_trainer_semantic_feature_builder_shape_and_flags():
    # Build a tiny mock asm and verify semantic feature matrix shape / value ranges.
    ...
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_model_innovation_hooks.py -q`
Expected: FAIL because builder/helper does not exist.

**Step 3: Write minimal implementation**

```python
# add _build_node_semantic_features and attach to model.field
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_model_innovation_hooks.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/train/trainer.py test_model_innovation_hooks.py
git commit -m "train: derive CDB semantic node features for model input"
```

### Task 3: Add uncertainty calibration utility

**Files:**
- Create: `src/train/uncertainty_calibration.py`
- Test: `test_model_innovation_hooks.py`

**Step 1: Write the failing test**

```python
def test_residual_driven_sigma_calibration_is_monotonic():
    # calibrated sigma should be monotonic wrt residual proxy.
    ...
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_model_innovation_hooks.py -q`
Expected: FAIL because utility module does not exist.

**Step 3: Write minimal implementation**

```python
def calibrate_sigma(...):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_model_innovation_hooks.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/train/uncertainty_calibration.py test_model_innovation_hooks.py
git commit -m "train: add residual-driven uncertainty calibration utility"
```

### Task 4: Wire config loading for innovation flags

**Files:**
- Modify: `main.py`

**Step 1: Write the failing test**

```python
def test_main_reads_innovation_network_flags():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest test_model_innovation_hooks.py -q`
Expected: FAIL because parser ignores new keys.

**Step 3: Write minimal implementation**

```python
# parse finite spectral / semantic / uncertainty flags from network_config
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest test_model_innovation_hooks.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add main.py test_model_innovation_hooks.py
git commit -m "main: parse innovation config flags"
```

### Task 5: Verify end-to-end non-regression

**Files:**
- Modify: none (verification only)

**Step 1: Run smoke tests**

Run: `python test_config_read.py`
Expected: PASS

**Step 2: Run DFEM smoke**

Run: `python test_dfem.py`
Expected: PASS

**Step 3: Run innovation tests**

Run: `python -m pytest test_model_innovation_hooks.py -q`
Expected: PASS

**Step 4: Sanity-check imports**

Run: `python -c "from model.pinn_model import ModelConfig, create_displacement_model; print('ok')"`
Expected: prints `ok`

**Step 5: Commit**

```bash
git add .
git commit -m "feat: innovation hooks for geometry generalization and uncertainty"
```

