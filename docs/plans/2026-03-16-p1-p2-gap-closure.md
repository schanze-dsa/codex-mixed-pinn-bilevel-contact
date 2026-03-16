# P1/P2 Gap Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the remaining PDF gaps by standardizing mixed residual assembly, enriching stress-side geometry semantics, and adding a contact-side pointwise or hybrid stress path without rewriting already-passing P1 and P2-A work.

**Architecture:** Keep the current P1 backend and P2-A stress-branch foundations intact. Add a canonical `mixed_residual_terms(...)` entry point in elasticity physics, make strict mixed total-energy assembly consume it, then extend trainer/model wiring so richer semantic features feed the stress path only and contact-region stress can use a pointwise or hybrid route.

**Tech Stack:** Python, TensorFlow, `unittest`, trainer mixins, mixed PINN model code, elasticity/contact utilities.

---

### Task 1: Canonical Mixed Residual Entry Point

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_elasticity_residuals.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\elasticity_residual.py`

**Step 1: Write the failing test**

Add tests that assert:

- `mixed_residual_terms(...)` returns `R_eq` and `R_const`
- returned shapes match the current equilibrium and constitutive contracts
- results stay finite for the existing mixed-model path

Representative test shape:

```python
terms = residual.mixed_residual_terms(model.u_fn, model.sigma_fn, params)
self.assertIn("R_eq", terms)
self.assertIn("R_const", terms)
self.assertEqual(tuple(terms["R_eq"].shape), (4, 3))
self.assertEqual(tuple(terms["R_const"].shape), (4, 6))
tf.debugging.assert_all_finite(terms["R_eq"], "R_eq must stay finite")
tf.debugging.assert_all_finite(terms["R_const"], "R_const must stay finite")
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_mixed_elasticity_residuals.py -v
```

Expected:

- failure because `mixed_residual_terms(...)` does not exist yet

**Step 3: Write minimal implementation**

Implement `mixed_residual_terms(...)` in `src/physics/elasticity_residual.py` by delegating to the existing equilibrium and constitutive helpers.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_mixed_elasticity_residuals.py -v
```

Expected:

- the new mixed residual entry-point tests pass

**Step 5: Commit**

```bash
git add test_mixed_elasticity_residuals.py src/physics/elasticity_residual.py
git commit -m "feat: add unified mixed residual entry point"
```

### Task 2: Consume Canonical Residual Terms In Strict Mixed Energy Assembly

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_trainer_optimization_hooks.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\model\loss_energy.py`

**Step 1: Write the failing test**

Add tests that assert strict mixed assembly can consume the new residual contract and still produce `E_eq` without changing route behavior.

Representative assertion shape:

```python
Pi, parts, stats = total.strict_mixed_objective(model.u_fn, params=params, stress_fn=model.us_fn)
self.assertIn("E_eq", parts)
tf.debugging.assert_all_finite(parts["E_eq"], "E_eq must stay finite")
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_trainer_optimization_hooks.py -v
```

Expected:

- failure because strict mixed assembly is still assembling residual details ad hoc

**Step 3: Write minimal implementation**

Refactor `src/model/loss_energy.py` strict mixed code to use `elasticity.mixed_residual_terms(...)` instead of reconstructing the mixed residual path inline.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_trainer_optimization_hooks.py test_mixed_elasticity_residuals.py -v
```

Expected:

- strict mixed energy assembly remains green using the canonical residual entry point

**Step 5: Commit**

```bash
git add test_trainer_optimization_hooks.py src/model/loss_energy.py src/physics/elasticity_residual.py
git commit -m "refactor: route strict mixed energy through unified residual terms"
```

### Task 3: Richer Geometry-Aware Semantics For Stress Path

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_model_innovation_hooks.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_model_outputs.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\model\pinn_model.py`

**Step 1: Write the failing test**

Add tests that assert:

- trainer-built semantic features can expose richer contact-aware dimensions
- semantic feature dimensions are validated
- stress-enabled forward paths accept the richer semantic contract
- flags-off behavior remains unchanged

Representative test shape:

```python
cfg.field.use_engineering_semantics = True
cfg.field.semantic_feat_dim = 8
model = DisplacementModel(cfg)
model.set_node_semantic_features(features)
u, sigma = model.us_fn(X, params)
self.assertEqual(tuple(sigma.shape), (n, 6))
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_model_innovation_hooks.py test_mixed_model_outputs.py -v
```

Expected:

- failure because the richer semantic feature contract is not implemented yet

**Step 3: Write minimal implementation**

Expand `build_node_semantic_features(...)` and update stress-side model wiring so the added semantics feed the stress path only.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_model_innovation_hooks.py test_mixed_model_outputs.py -v
```

Expected:

- richer semantic feature tests pass
- old flags-off paths remain green

**Step 5: Commit**

```bash
git add test_model_innovation_hooks.py test_mixed_model_outputs.py src/train/trainer.py src/model/pinn_model.py
git commit -m "feat: add stress-side geometry-aware semantics"
```

### Task 4: Contact-Side Pointwise Or Hybrid Stress Route

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_model_outputs.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_model_innovation_hooks.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\model\pinn_model.py`

**Step 1: Write the failing test**

Add tests that assert:

- enabling the contact-side hybrid route keeps `us_fn()`, `us_fn_pointwise()`, and `sigma_fn()` callable
- contact-region masks can steer the stress path without changing output shape
- flags-off behavior remains the current graph-dominant path

Representative test shape:

```python
cfg.field.enable_contact_stress_hybrid = True
model = DisplacementModel(cfg)
model.set_node_semantic_features(features_with_contact_mask)
u, sigma = model.us_fn(X, params)
self.assertEqual(tuple(u.shape), (n, 3))
self.assertEqual(tuple(sigma.shape), (n, 6))
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_mixed_model_outputs.py test_model_innovation_hooks.py -v
```

Expected:

- failure because no explicit contact-side hybrid stress route exists yet

**Step 3: Write minimal implementation**

Implement a config-gated contact-region pointwise or hybrid stress path in `src/model/pinn_model.py`, using a contact-region mask and preserving canonical Voigt output ordering.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_mixed_model_outputs.py test_model_innovation_hooks.py test_mixed_elasticity_residuals.py -v
```

Expected:

- contact-side hybrid path tests pass
- mixed residual path remains numerically valid

**Step 5: Commit**

```bash
git add test_mixed_model_outputs.py test_model_innovation_hooks.py src/model/pinn_model.py
git commit -m "feat: add contact-side hybrid stress path"
```

### Task 5: Focused Regression Across Existing P1 And P2-A Coverage

**Files:**
- No new files

**Step 1: Run focused regression**

Run:

```bash
python -m unittest test_contact_inner_kernel_primitives.py test_contact_inner_solver.py test_mixed_training_phase_controls.py test_trainer_optimization_hooks.py test_mixed_model_outputs.py test_mixed_elasticity_residuals.py test_model_innovation_hooks.py -v
```

Expected:

- existing P1 backend tests stay green
- existing P2-A stress-branch tests stay green
- new gap-closure tests stay green

**Step 2: Inspect gating assumptions**

Confirm:

- flags-off behavior did not change
- strict mixed still uses the current backend resolution
- semantic and hybrid stress features are only active when explicitly enabled

**Step 3: Commit**

```bash
git add src/physics/elasticity_residual.py src/model/loss_energy.py src/train/trainer.py src/model/pinn_model.py test_mixed_elasticity_residuals.py test_mixed_model_outputs.py test_model_innovation_hooks.py test_trainer_optimization_hooks.py
git commit -m "feat: close remaining p1 p2 plan gaps"
```
