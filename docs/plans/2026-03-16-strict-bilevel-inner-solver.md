# Strict Bilevel Inner Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the placeholder mixed-bilevel inner solver with a real stateless contact solve and wire it only into the strict mixed traction-matching path.

**Architecture:** Introduce a new low-level contact kernel primitives module, rebuild `solve_contact_inner(...)` on top of those stateless primitives, and then connect only the mixed bilevel route to the new solver while leaving the legacy ALM path unchanged. Use TDD throughout so each math primitive, fallback rule, and mixed-path adapter is proven by failing tests before production changes.

**Tech Stack:** Python, TensorFlow, `unittest`, existing contact physics modules under `src/physics/contact`

---

### Task 1: Add primitive-kernel tests

**Files:**
- Create: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_contact_inner_kernel_primitives.py`
- Reference: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\contact\contact_normal_alm.py`
- Reference: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\contact\contact_friction_alm.py`

**Step 1: Write the failing test**

```python
def test_fb_normal_residual_matches_closed_form(self):
    g_n = tf.constant([-0.2, 0.1], dtype=tf.float32)
    lambda_n = tf.constant([0.3, 0.4], dtype=tf.float32)
    eps_n = tf.constant(1.0e-6, dtype=tf.float32)

    got = fb_normal_residual(g_n, lambda_n, eps_n)
    want = tf.sqrt(g_n * g_n + lambda_n * lambda_n + eps_n * eps_n) - g_n - lambda_n

    tf.debugging.assert_near(got, want)
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_contact_inner_kernel_primitives.py -v`
Expected: `FAIL` or `ERROR` because `contact_inner_kernel_primitives.py` and the tested functions do not exist yet.

**Step 3: Write minimal implementation**

```python
def fb_normal_residual(g_n, lambda_n, eps_n):
    return tf.sqrt(g_n * g_n + lambda_n * lambda_n + eps_n * eps_n) - g_n - lambda_n
```

Also add minimal implementations and tests for:

- `fb_normal_jacobian`
- `project_to_coulomb_disk`
- `compose_contact_traction`
- `check_contact_feasibility`

**Step 4: Run test to verify it passes**

Run: `python -m unittest test_contact_inner_kernel_primitives.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add test_contact_inner_kernel_primitives.py src/physics/contact/contact_inner_kernel_primitives.py
git commit -m "test: add inner contact kernel primitive coverage"
```

### Task 2: Rebuild `solve_contact_inner(...)` around geometric inputs

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_contact_inner_solver.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\contact\contact_inner_solver.py`
- Reference: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\contact\contact_inner_kernel_primitives.py`

**Step 1: Write the failing test**

```python
def test_inner_solver_solves_from_gap_and_slip(self):
    result = solve_contact_inner(
        g_n=tf.constant([-0.1], dtype=tf.float32),
        ds_t=tf.constant([[0.2, 0.0]], dtype=tf.float32),
        normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
        t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
        t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
        mu=0.3,
        eps_n=1.0e-6,
        k_t=10.0,
        init_state=None,
    )
    self.assertGreater(float(result.diagnostics["fn_norm"]), 0.0)
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_contact_inner_solver.py -v`
Expected: `FAIL` because the old signature still expects `lambda_n` and `lambda_t`.

**Step 3: Write minimal implementation**

```python
lambda_n = zeros_or_init(...)
lambda_t = zeros_or_init(...)
for _ in range(max_inner_iters):
    r_n = fb_normal_residual(...)
    j_n = fb_normal_jacobian(...)
    lambda_n = tf.maximum(0.0, lambda_n - damping * r_n / safe_j_n)
    tau_trial = lambda_t + k_t * ds_t
    lambda_t = project_to_coulomb_disk(tau_trial, mu * lambda_n, eps_t)
```

Return:

- updated `ContactInnerState`
- `traction_vec`
- `traction_tangent`
- diagnostics for `fn_norm`, `ft_norm`, `cone_violation`, `max_penetration`, `fallback_used`

**Step 4: Run test to verify it passes**

Run: `python -m unittest test_contact_inner_solver.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add test_contact_inner_solver.py src/physics/contact/contact_inner_solver.py
git commit -m "feat: add stateless strict bilevel inner solver"
```

### Task 3: Add explicit fallback and diagnostics coverage

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_contact_inner_solver.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\contact\contact_inner_solver.py`

**Step 1: Write the failing test**

```python
def test_inner_solver_reuses_feasible_init_state_on_failure(self):
    init_state = ContactInnerState(
        lambda_n=tf.constant([0.5], dtype=tf.float32),
        lambda_t=tf.constant([[0.1, 0.0]], dtype=tf.float32),
        converged=True,
    )
    result = solve_contact_inner(
        g_n=tf.constant([-10.0], dtype=tf.float32),
        ds_t=tf.constant([[100.0, 0.0]], dtype=tf.float32),
        normals=...,
        t1=...,
        t2=...,
        mu=0.2,
        eps_n=1.0e-6,
        k_t=1.0,
        init_state=init_state,
        max_inner_iters=1,
        tol_n=1.0e-20,
        tol_t=1.0e-20,
    )
    self.assertTrue(result.state.fallback_used)
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_contact_inner_solver.py -v`
Expected: `FAIL` because fallback is not yet explicit or does not preserve feasible state semantics.

**Step 3: Write minimal implementation**

```python
if not converged:
    if init_state is feasible:
        use init_state
    else:
        use projected tangential fallback
```

Ensure diagnostics include:

- `fallback_used`
- `cone_violation`
- `max_penetration`
- `iters`

**Step 4: Run test to verify it passes**

Run: `python -m unittest test_contact_inner_solver.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add test_contact_inner_solver.py src/physics/contact/contact_inner_solver.py
git commit -m "feat: add explicit inner solver fallback diagnostics"
```

### Task 4: Wire the strict mixed traction path to the new solver

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\model\loss_energy.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\contact\contact_operator.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_contact_matching.py`
- Reference: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\contact\contact_inner_solver.py`

**Step 1: Write the failing test**

```python
def test_mixed_path_consumes_inner_solved_traction_without_legacy_multiplier_update(self):
    inner = solve_contact_inner(
        g_n=tf.constant([-0.1], dtype=tf.float32),
        ds_t=tf.constant([[0.0, 0.0]], dtype=tf.float32),
        normals=n,
        t1=t1,
        t2=t2,
        mu=0.3,
        eps_n=1.0e-6,
        k_t=10.0,
    )
    rs, rm = traction_matching_residual(sigma_s, sigma_m, n, t1, t2, inner)
    self.assertEqual(tuple(rs.shape), (1, 3))
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_mixed_contact_matching.py -v`
Expected: `FAIL` because the mixed path still assumes a manually assembled placeholder result or has no strict inner-solve adapter.

**Step 3: Write minimal implementation**

```python
def solve_inner_state_from_geometry(...):
    return solve_contact_inner(g_n=..., ds_t=..., normals=..., t1=..., t2=..., ...)
```

Only add the minimal strict mixed-path adapter. Do not rewrite legacy residual/update flows.

**Step 4: Run test to verify it passes**

Run: `python -m unittest test_mixed_contact_matching.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add test_mixed_contact_matching.py src/model/loss_energy.py src/physics/contact/contact_operator.py
git commit -m "feat: wire mixed traction matching to strict inner solver"
```

### Task 5: Surface strict bilevel diagnostics in trainer stats

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_opt_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_monitor_mixin.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_bilevel_diagnostics.py`

**Step 1: Write the failing test**

```python
def test_monitor_reports_inner_convergence_and_fallback_metrics(self):
    diagnostics = {
        "fn_norm": 1.0,
        "ft_norm": 2.0,
        "cone_violation": 3.0,
        "max_penetration": 4.0,
        "fallback_used": 1.0,
    }
```

Assert these are injected and extracted with stable trainer keys.

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_mixed_bilevel_diagnostics.py -v`
Expected: `FAIL` because the new diagnostics are not yet mapped through trainer helpers.

**Step 3: Write minimal implementation**

```python
key_map = {
    "inner_fn_norm": "fn_norm",
    "inner_ft_norm": "ft_norm",
    "inner_cone_violation": "cone_violation",
    "inner_max_penetration": "max_penetration",
    "inner_fallback_used": "fallback_used",
}
```

**Step 4: Run test to verify it passes**

Run: `python -m unittest test_mixed_bilevel_diagnostics.py -v`
Expected: `PASS`

**Step 5: Commit**

```bash
git add test_mixed_bilevel_diagnostics.py src/train/trainer_opt_mixin.py src/train/trainer_monitor_mixin.py
git commit -m "feat: expose strict bilevel inner solver diagnostics"
```

### Task 6: Run focused regression checks and record P0 boundary

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\docs\plans\2026-03-16-strict-bilevel-inner-solver.md`
- Verify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_contact_inner_kernel_primitives.py`
- Verify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_contact_inner_solver.py`
- Verify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_contact_matching.py`
- Verify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_bilevel_diagnostics.py`

**Step 1: Write the failing test**

No new production behavior. Instead, define the verification batch and expected P0 boundary before refactoring anything else.

**Step 2: Run test to verify it fails**

Run the focused suite before the last fixes are in:

`python -m unittest test_contact_inner_kernel_primitives.py test_contact_inner_solver.py test_mixed_contact_matching.py test_mixed_bilevel_diagnostics.py -v`

Expected: one or more failures until all previous tasks are complete.

**Step 3: Write minimal implementation**

Complete the missing fixes only. Do not begin P1 `ContactOperator` unification in this task.

**Step 4: Run test to verify it passes**

Run:

`python -m unittest test_contact_inner_kernel_primitives.py test_contact_inner_solver.py test_mixed_contact_matching.py test_mixed_bilevel_diagnostics.py -v`

Expected: `PASS`

**Step 5: Commit**

```bash
git add docs/plans/2026-03-16-strict-bilevel-inner-solver.md
git commit -m "test: verify strict bilevel inner solver p0 boundary"
```
