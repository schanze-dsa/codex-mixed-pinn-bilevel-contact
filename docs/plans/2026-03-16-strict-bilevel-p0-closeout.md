# Strict Bilevel P0 Closeout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Finish the remaining `P0` work by giving strict mixed training its own semantic objective, explicit trainer route modes, and aggregate diagnostics without starting `P1` kernel unification.

**Architecture:** Add a dedicated strict-mixed objective path in `TotalEnergy`, then let `TrainerOptMixin` choose between legacy, forward-only strict mixed, and normal-ready strict mixed routes. Keep legacy ALM contact behavior stable while masking disallowed strict-mixed outer-loss terms and surfacing convergence, fallback, skip, and continuation-freeze rates in trainer logs.

**Tech Stack:** Python, TensorFlow, `unittest`, existing `TotalEnergy` / `Trainer` mixins under `src/model` and `src/train`

---

### Task 1: Add failing tests for strict mixed route and loss profile

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_training_phase_controls.py`

**Step 1: Write the failing test**

```python
def test_strict_mixed_weight_profile_disables_legacy_energy_terms(self):
    trainer = object.__new__(Trainer)
    trainer.loss_state = None
    trainer._base_weights = {"E_int": 1.0, "E_cn": 2.0, "E_sigma": 3.0, "E_eq": 4.0}
    trainer._loss_keys = list(trainer._base_weights.keys())
    trainer._active_weight_overrides = {}

    total = TotalEnergy()
    total.set_mixed_bilevel_flags(
        {
            "phase_name": "phase2a",
            "normal_ift_enabled": True,
            "tangential_ift_enabled": False,
            "detach_inner_solution": False,
        }
    )

    trainer._configure_mixed_bilevel_route(total)
    weights = trainer._build_weight_vector().numpy()
    self.assertEqual(list(weights), [0.0, 2.0, 0.0, 4.0])
```

Also add a failing test that `tangential_ift_enabled=True` raises a clear unsupported error.

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_trainer_optimization_hooks.py test_mixed_training_phase_controls.py -v`
Expected: `FAIL` because strict mixed route/profile helpers do not exist yet.

**Step 3: Write minimal implementation**

Implement helpers in `TrainerOptMixin` that:

- resolve route mode from `mixed_bilevel_flags`
- set active weight overrides for strict mixed mode
- raise `NotImplementedError` for tangential/full IFT in `P0`

**Step 4: Run test to verify it passes**

Run: `python -m unittest test_trainer_optimization_hooks.py test_mixed_training_phase_controls.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add test_trainer_optimization_hooks.py test_mixed_training_phase_controls.py src/train/trainer_opt_mixin.py
git commit -m "test: add strict mixed route and loss profile coverage"
```

### Task 2: Add failing tests for dedicated strict mixed objective semantics

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_contact_matching.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\model\loss_energy.py`

**Step 1: Write the failing test**

```python
def test_strict_mixed_objective_zeroes_eint_and_esigma(self):
    total = TotalEnergy(TotalConfig(loss_mode="energy"))
    total.set_mixed_bilevel_flags({"phase_name": "phase2a"})
    total.attach(contact=contact)

    _, parts, stats = total.strict_mixed_objective(u_fn, params={}, stress_fn=stress_fn)

    self.assertAlmostEqual(float(parts["E_int"].numpy()), 0.0)
    self.assertAlmostEqual(float(parts["E_sigma"].numpy()), 0.0)
    self.assertGreater(float(parts["E_cn"].numpy()), 0.0)
```

Add a second failing test showing that if strict mixed contact cannot be formed for a batch, the batch is marked skipped instead of silently falling back to legacy ALM contact enforcement.

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_mixed_contact_matching.py test_trainer_optimization_hooks.py -v`
Expected: `FAIL` because `strict_mixed_objective(...)` does not exist yet.

**Step 3: Write minimal implementation**

Add `TotalEnergy.strict_mixed_objective(...)` and a helper that:

- computes strict contact terms
- computes residual-style `E_eq`, `E_reg`, `E_bc`, `E_tight`
- keeps `E_int`, `E_sigma`, `E_bi`, and `E_ed` at zero for optimization semantics
- surfaces `mixed_strict_skipped` when contact must be skipped

**Step 4: Run test to verify it passes**

Run: `python -m unittest test_mixed_contact_matching.py test_trainer_optimization_hooks.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add test_mixed_contact_matching.py test_trainer_optimization_hooks.py src/model/loss_energy.py
git commit -m "feat: add dedicated strict mixed objective path"
```

### Task 3: Add failing tests for aggregate strict mixed diagnostics and log formatting

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_bilevel_diagnostics.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_monitor_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_init_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer.py`

**Step 1: Write the failing test**

```python
def test_monitor_reports_strict_mixed_rates_and_freeze_events(self):
    stats = {
        "inner_convergence_rate": 0.75,
        "inner_fallback_rate": 0.25,
        "inner_skip_rate": 0.10,
        "continuation_freeze_events": 2.0,
    }
    picked = TrainerMonitorMixin.extract_bilevel_diagnostics(stats)
    self.assertEqual(picked["inner_convergence_rate"], 0.75)
```

Add a second failing test for `_format_train_log_postfix(...)` that expects concise strict mixed log tokens such as route mode and cumulative rates.

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_mixed_bilevel_diagnostics.py test_trainer_optimization_hooks.py -v`
Expected: `FAIL` because the aggregate diagnostics and log formatting are not wired yet.

**Step 3: Write minimal implementation**

Add:

- strict mixed runtime counters in `TrainerInitMixin`
- continuation-freeze state in `Trainer`
- monitor extraction and log formatting for rate metrics

**Step 4: Run test to verify it passes**

Run: `python -m unittest test_mixed_bilevel_diagnostics.py test_trainer_optimization_hooks.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add test_mixed_bilevel_diagnostics.py test_trainer_optimization_hooks.py src/train/trainer_monitor_mixin.py src/train/trainer_init_mixin.py src/train/trainer.py
git commit -m "feat: add strict mixed aggregate diagnostics and log output"
```

### Task 4: Route trainer optimization through the dedicated strict mixed objective

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_opt_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\model\loss_energy.py`
- Test: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`

**Step 1: Write the failing test**

```python
def test_strict_mixed_route_uses_dedicated_total_entrypoint(self):
    trainer = object.__new__(Trainer)
    total = SimpleNamespace(
        mixed_bilevel_flags={"phase_name": "phase2a", "normal_ift_enabled": True, "tangential_ift_enabled": False},
        strict_mixed_objective=lambda *args, **kwargs: (tf.constant(0.0), {"E_cn": tf.constant(1.0)}, {"mixed": 1.0}),
        energy=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy energy path should not be used")),
    )
```

Assert that strict mixed routes call `strict_mixed_objective(...)` and legacy routes call `energy(...)`.

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_trainer_optimization_hooks.py -v`
Expected: `FAIL` because trainer still calls `total.energy(...)` everywhere.

**Step 3: Write minimal implementation**

Refactor trainer loss/compiled-step helpers so they dispatch through one route-aware forward helper:

- legacy route -> `total.energy(...)`
- strict mixed route -> `total.strict_mixed_objective(...)`
- tangential/full IFT requested -> `NotImplementedError`

Update strict mixed counters after each strict route evaluation.

**Step 4: Run test to verify it passes**

Run: `python -m unittest test_trainer_optimization_hooks.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add test_trainer_optimization_hooks.py src/train/trainer_opt_mixin.py src/model/loss_energy.py
git commit -m "feat: route strict mixed training through dedicated objective"
```

### Task 5: Run focused P0 verification and record the boundary

**Files:**
- Verify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_contact_inner_kernel_primitives.py`
- Verify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_contact_inner_solver.py`
- Verify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_contact_matching.py`
- Verify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_bilevel_diagnostics.py`
- Verify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_training_phase_controls.py`
- Verify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`

**Step 1: Run the focused suite**

Run:

```bash
python -m unittest test_contact_inner_kernel_primitives.py test_contact_inner_solver.py test_mixed_contact_matching.py test_mixed_bilevel_diagnostics.py test_mixed_training_phase_controls.py test_trainer_optimization_hooks.py -v
```

Expected: `PASS`

**Step 2: Review P0 boundary**

Confirm:

- strict mixed objective is dedicated
- active strict mixed profile excludes `E_int` and `E_sigma`
- route modes are explicit
- aggregate diagnostics are visible
- tangential/full IFT is still clearly unsupported
- legacy ALM path is still intact

**Step 3: Commit**

```bash
git add docs/plans/2026-03-16-strict-bilevel-p0-closeout-design.md docs/plans/2026-03-16-strict-bilevel-p0-closeout.md
git commit -m "docs: record strict bilevel p0 closeout plan"
```
