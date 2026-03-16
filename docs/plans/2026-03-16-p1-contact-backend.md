# P1 Contact Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Formalize `ContactOperator` backend selection so legacy training defaults to `legacy_alm`, strict mixed training defaults to `inner_solver`, and trainer logs/tests explicitly expose the resolved backend.

**Architecture:** Add an explicit `contact_backend` config field, resolve it in the trainer after route selection, pass the resolved backend into total-energy runtime flags, and surface it in logs. Keep legacy contact numerics unchanged in this step; only formalize backend semantics and validation.

**Tech Stack:** Python, TensorFlow, `unittest`, trainer mixins, contact operator / total energy runtime flags.

---

### Task 1: Add Failing Tests For Backend Resolution

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_training_phase_controls.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`

**Step 1: Write the failing test**

Add tests that assert:

- `contact_backend="auto"` plus `phase0` resolves to `legacy_alm`
- `contact_backend="auto"` plus `phase2a` resolves to `inner_solver`
- explicit `contact_backend="legacy_alm"` remains valid for legacy route
- explicit contradictory combinations raise `ValueError`

Representative test shape:

```python
def test_resolve_contact_backend_auto_uses_legacy_for_phase0(self):
    cfg = TrainerConfig(contact_backend="auto")
    trainer = object.__new__(Trainer)
    trainer.cfg = cfg
    trainer._mixed_phase_flags = {"phase_name": "phase0", "normal_ift_enabled": False, "tangential_ift_enabled": False}
    self.assertEqual(trainer._resolve_contact_backend(), "legacy_alm")
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_mixed_training_phase_controls.py test_trainer_optimization_hooks.py -v
```

Expected:

- new tests fail with missing `_resolve_contact_backend()` or missing `contact_backend` config support

**Step 3: Write minimal implementation**

Add:

- `contact_backend` to `TrainerConfig`
- trainer helpers to resolve and validate backend selection

Do not change legacy contact numerics yet.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_mixed_training_phase_controls.py test_trainer_optimization_hooks.py -v
```

Expected:

- backend resolution tests pass

**Step 5: Commit**

```bash
git add test_mixed_training_phase_controls.py test_trainer_optimization_hooks.py src/train/trainer_config.py src/train/trainer.py src/train/trainer_opt_mixin.py
git commit -m "feat: add contact backend resolution"
```

### Task 2: Add Failing Tests For Backend Visibility In Runtime Flags And Logs

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_training_phase_controls.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`

**Step 1: Write the failing test**

Add tests that assert:

- `_assemble_total()` propagates resolved `contact_backend`
- `_format_train_log_postfix()` includes `cback=legacy_alm` or `cback=inner_solver`

Representative assertions:

```python
self.assertEqual(total.mixed_bilevel_flags["contact_backend"], "inner_solver")
self.assertIn("cback=inner_solver", postfix)
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_mixed_training_phase_controls.py test_trainer_optimization_hooks.py -v
```

Expected:

- missing backend flag in total-energy runtime flags
- missing `cback=` token in log postfix

**Step 3: Write minimal implementation**

Update:

- trainer runtime flag assembly
- trainer monitor logging formatter

Use existing route diagnostics instead of inventing a parallel logging path.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_mixed_training_phase_controls.py test_trainer_optimization_hooks.py -v
```

Expected:

- runtime flag and log tests pass

**Step 5: Commit**

```bash
git add test_mixed_training_phase_controls.py test_trainer_optimization_hooks.py src/train/trainer.py src/train/trainer_monitor_mixin.py
git commit -m "feat: expose contact backend in runtime flags and logs"
```

### Task 3: Add Failing Tests For ContactOperator Backend Contract

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_contact_matching.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_contact_inner_solver.py`

**Step 1: Write the failing test**

Add tests that assert:

- `ContactOperator` exposes an explicit backend selection helper or contract value
- legacy backend does not silently route through strict-mixed-only paths
- strict mixed backend still uses `strict_mixed_inputs()` plus `solve_strict_inner()`

Representative test shape:

```python
def test_contact_operator_backend_contract_defaults_to_legacy(self):
    op = ContactOperator()
    self.assertEqual(op.resolve_backend("legacy_alm"), "legacy_alm")
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_mixed_contact_matching.py test_contact_inner_solver.py -v
```

Expected:

- missing backend contract helper or missing explicit semantics

**Step 3: Write minimal implementation**

Add a thin backend helper in `contact_operator.py` that makes backend semantics explicit without changing default legacy numerics.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_mixed_contact_matching.py test_contact_inner_solver.py -v
```

Expected:

- adapter-level backend tests pass

**Step 5: Commit**

```bash
git add test_mixed_contact_matching.py test_contact_inner_solver.py src/physics/contact/contact_operator.py
git commit -m "feat: formalize contact operator backend contract"
```

### Task 4: Regress P0 And New P1 Backend Tests Together

**Files:**
- No new files

**Step 1: Run focused regression**

Run:

```bash
python -m unittest test_contact_inner_kernel_primitives.py test_contact_inner_solver.py test_mixed_contact_matching.py test_mixed_bilevel_diagnostics.py test_mixed_training_phase_controls.py test_trainer_optimization_hooks.py -v
```

Expected:

- all P0 tests remain green
- new backend formalization tests are green

**Step 2: Inspect runtime/logging assumptions**

Check that:

- non-strict-mixed tests report `legacy_alm`
- strict mixed tests report `inner_solver`
- no test had to change legacy loss semantics

**Step 3: Commit**

```bash
git add src/train/trainer.py src/train/trainer_config.py src/train/trainer_monitor_mixin.py src/train/trainer_opt_mixin.py src/physics/contact/contact_operator.py test_contact_inner_solver.py test_mixed_contact_matching.py test_mixed_training_phase_controls.py test_trainer_optimization_hooks.py
git commit -m "feat: complete p1 contact backend formalization"
```
