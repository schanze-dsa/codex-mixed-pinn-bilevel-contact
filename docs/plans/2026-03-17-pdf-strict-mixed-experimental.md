# PDF Strict Mixed Experimental Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a PDF-target strict mixed experimental route alongside the existing locked route, covering P0 inner-solver diagnostics, P1 backend/contract cleanup, and P2 stress-head contact semantics without changing the default `config.yaml` behavior.

**Architecture:** Keep one trainer/model/contact implementation tree with explicit profile gating. The default profile remains canonicalized to the current locked route. A new `strict_mixed_experimental.yaml` profile enables strict mixed backend dispatch, typed strict contact inputs, richer inner diagnostics, and the P2 pointwise stress path with contact-surface semantics.

**Tech Stack:** Python, TensorFlow, YAML config parsing, `unittest`, trainer mixins, contact operator / strict mixed loss assembly.

---

### Task 1: Add Failing Tests For Profile-Gated Config Parsing

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_main_new_config_override.py`

**Step 1: Write the failing test**

Add tests that assert:

- the default config path still resolves to the locked route and keeps `contact_backend` on the legacy-compatible default
- a dedicated experimental YAML can enable the strict mixed route without tripping locked-route validation
- the experimental profile can enable `contact_backend="inner_solver"` plus strict mixed phase flags while the default route remains unchanged

Representative test shape:

```python
def test_prepare_config_keeps_default_locked_route(self):
    cfg, _ = main_new._prepare_config_with_autoguess(config_path="config.yaml")
    self.assertTrue(cfg.preload_use_stages)
    self.assertTrue(cfg.incremental_mode)
    self.assertEqual(cfg.contact_backend, "auto")
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_main_new_config_override.py -v
```

Expected:

- new profile-gating assertions fail because the parser does not yet distinguish the locked route from the experimental route

**Step 3: Write minimal implementation**

Implement profile-aware parsing in:

- `main new.py`
- `src/train/trainer_config.py`

Add the minimum new config fields needed to represent the experimental route, but keep all locked-route defaults intact.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_main_new_config_override.py -v
```

Expected:

- default-route tests still pass
- new experimental-profile parsing tests pass

**Step 5: Commit**

```bash
git add test_main_new_config_override.py main\ new.py src/train/trainer_config.py
git commit -m "feat: add strict mixed experimental profile parsing"
```

### Task 2: Add The Experimental YAML Profile

**Files:**
- Create: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\strict_mixed_experimental.yaml`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\config.yaml`

**Step 1: Write the failing test**

Extend `test_main_new_config_override.py` so a real repository-local profile file is loaded and verified.

Representative assertion:

```python
self.assertEqual(cfg.contact_backend, "inner_solver")
self.assertEqual(cfg.mixed_bilevel_phase.phase_name, "phase1")
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_main_new_config_override.py -v
```

Expected:

- the test fails because `strict_mixed_experimental.yaml` does not yet exist or lacks the expected keys

**Step 3: Write minimal implementation**

Create `strict_mixed_experimental.yaml` with:

- strict mixed phase flags
- `contact_backend: inner_solver`
- P2 stress flags
- only the experimental capabilities approved in the design

Do not change the effective default route in `config.yaml`; only add comments if needed for discoverability.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_main_new_config_override.py -v
```

Expected:

- experimental profile file is discoverable and parses into the expected runtime config

**Step 5: Commit**

```bash
git add strict_mixed_experimental.yaml config.yaml test_main_new_config_override.py
git commit -m "feat: add strict mixed experimental profile"
```

### Task 3: Add Failing Tests For Typed Strict Contact Inputs And Inner Linearization

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_contact_inner_solver.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_contact_matching.py`

**Step 1: Write the failing test**

Add tests that assert:

- `ContactOperator.strict_mixed_inputs(...)` returns a typed `StrictMixedContactInputs` object rather than an unstructured dict
- `solve_contact_inner(..., return_linearization=True)` returns a linearization payload
- warm-start state survives the typed input round-trip

Representative test shape:

```python
inputs = op.strict_mixed_inputs(...)
self.assertIsInstance(inputs, StrictMixedContactInputs)
result = solve_contact_inner(..., return_linearization=True)
self.assertIn("jac_z", result.linearization)
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_contact_inner_solver.py test_mixed_contact_matching.py -v
```

Expected:

- failures because the strict input contract is still dict-based and the solver does not yet expose linearization output

**Step 3: Write minimal implementation**

Implement in:

- `src/physics/contact/contact_operator.py`
- `src/physics/contact/contact_inner_solver.py`

Add:

- `StrictMixedContactInputs`
- optional `linearization` field on the solver result
- pass-through support in `solve_strict_inner(...)`

Do not change the legacy backend behavior.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_contact_inner_solver.py test_mixed_contact_matching.py -v
```

Expected:

- typed-contract and linearization tests pass

**Step 5: Commit**

```bash
git add test_contact_inner_solver.py test_mixed_contact_matching.py src/physics/contact/contact_operator.py src/physics/contact/contact_inner_solver.py
git commit -m "feat: add typed strict mixed inputs and inner linearization"
```

### Task 4: Add Failing Tests For P0 Diagnostics Propagation

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_bilevel_diagnostics.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`

**Step 1: Write the failing test**

Add tests that assert the following keys survive from inner solve to trainer diagnostics:

- `fb_residual_norm`
- `normal_step_norm`
- `tangential_step_norm`
- existing `fallback_used`, `cone_violation`, and `max_penetration`

Representative assertion:

```python
picked = TrainerMonitorMixin.extract_bilevel_diagnostics(stats)
self.assertEqual(picked["inner_fb_residual_norm"], 1.2)
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_mixed_bilevel_diagnostics.py test_trainer_optimization_hooks.py -v
```

Expected:

- new diagnostics are missing from the current trainer aggregation path

**Step 3: Write minimal implementation**

Update:

- `src/model/loss_energy.py`
- `src/train/trainer_opt_mixin.py`
- `src/train/trainer_monitor_mixin.py`

Ensure the diagnostics schema is consistent from solver output through trainer reporting.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_mixed_bilevel_diagnostics.py test_trainer_optimization_hooks.py -v
```

Expected:

- all strict-mixed diagnostics, old and new, are visible in trainer-side extraction

**Step 5: Commit**

```bash
git add test_mixed_bilevel_diagnostics.py test_trainer_optimization_hooks.py src/model/loss_energy.py src/train/trainer_opt_mixin.py src/train/trainer_monitor_mixin.py
git commit -m "feat: propagate strict mixed inner diagnostics"
```

### Task 5: Add Failing Tests For Experimental Runtime Gating

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_training_phase_controls.py`

**Step 1: Write the failing test**

Add tests that assert:

- the locked route still skips experimental execution branches
- the experimental profile allows strict mixed backend execution and logging
- the default route and experimental route produce different resolved runtime flags without changing shared code paths

Representative assertion:

```python
self.assertEqual(trainer_locked._resolve_contact_backend(), "legacy_alm")
self.assertEqual(trainer_exp._resolve_contact_backend(), "inner_solver")
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_mixed_training_phase_controls.py test_trainer_optimization_hooks.py -v
```

Expected:

- failures because trainer execution is still globally locked rather than profile-gated

**Step 3: Write minimal implementation**

Implement profile-aware runtime branching in:

- `src/train/trainer.py`
- `src/train/trainer_run_mixin.py`
- `src/train/trainer_opt_mixin.py`

Keep the default route unchanged while allowing the experimental route to execute strict mixed branches.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_mixed_training_phase_controls.py test_trainer_optimization_hooks.py -v
```

Expected:

- locked-route tests remain green
- experimental-route gating tests pass

**Step 5: Commit**

```bash
git add test_mixed_training_phase_controls.py test_trainer_optimization_hooks.py src/train/trainer.py src/train/trainer_run_mixin.py src/train/trainer_opt_mixin.py
git commit -m "feat: gate strict mixed runtime by profile"
```

### Task 6: Add Failing Tests For P2 Contact Stress Semantics

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_model_outputs.py`

**Step 1: Write the failing test**

Add tests that assert:

- the strict mixed contact stress path can receive contact-surface semantics
- `eps_bridge` remains active when those semantics are present
- the pointwise stress path is used for strict mixed contact evaluation rather than the graph stress branch

Representative test shape:

```python
self.assertEqual(tuple(sigma.shape), (4, 6))
tf.debugging.assert_all_finite(sigma, "strict mixed contact stress output must stay finite")
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_mixed_model_outputs.py -v
```

Expected:

- failures because the model does not yet distinguish the contact-surface semantic path from the existing node semantic path

**Step 3: Write minimal implementation**

Update:

- `src/model/pinn_model.py`
- `src/model/loss_energy.py`

Add:

- contact-surface semantic feature assembly for strict mixed contact samples
- pointwise stress-path usage in the strict mixed route
- `eps_bridge` preservation through the contact stress path

Do not change the default non-contact stress path.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_mixed_model_outputs.py -v
```

Expected:

- P2 stress-path tests pass without breaking existing mixed-model output tests

**Step 5: Commit**

```bash
git add test_mixed_model_outputs.py src/model/pinn_model.py src/model/loss_energy.py
git commit -m "feat: add p2 contact stress semantics path"
```

### Task 7: Run Focused Regression Before Any Larger Training Run

**Files:**
- No new files

**Step 1: Run focused regression**

Run:

```bash
python -m unittest test_main_new_config_override.py test_contact_inner_solver.py test_mixed_contact_matching.py test_mixed_bilevel_diagnostics.py test_mixed_training_phase_controls.py test_trainer_optimization_hooks.py test_mixed_model_outputs.py -v
```

Expected:

- all default-route regression tests pass
- all new experimental-route tests pass

**Step 2: Run a locked-route smoke check**

Run:

```bash
python -m unittest test_main_new_config_override.py test_trainer_optimization_hooks.py -v
```

Expected:

- the default locked route still resolves and logs as before

**Step 3: Run an experimental-profile smoke parse**

Run:

```bash
python - <<'PY'
import importlib.util, pathlib
root = pathlib.Path(r"D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact")
spec = importlib.util.spec_from_file_location("main_new_module", root / "main new.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg, _ = mod._prepare_config_with_autoguess(config_path=str(root / "strict_mixed_experimental.yaml"))
print(cfg.contact_backend)
print(cfg.mixed_bilevel_phase.phase_name)
PY
```

Expected:

- prints `inner_solver`
- prints the non-`phase0` strict mixed phase configured for the experimental route

**Step 4: Commit**

```bash
git add main\ new.py strict_mixed_experimental.yaml src/train/trainer.py src/train/trainer_run_mixin.py src/train/trainer_opt_mixin.py src/train/trainer_config.py src/model/loss_energy.py src/model/pinn_model.py src/physics/contact/contact_operator.py src/physics/contact/contact_inner_solver.py test_main_new_config_override.py test_contact_inner_solver.py test_mixed_contact_matching.py test_mixed_bilevel_diagnostics.py test_mixed_training_phase_controls.py test_trainer_optimization_hooks.py test_mixed_model_outputs.py
git commit -m "feat: add PDF strict mixed experimental route"
```
