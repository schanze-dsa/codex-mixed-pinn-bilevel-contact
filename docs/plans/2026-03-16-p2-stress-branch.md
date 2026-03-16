# P2 Stress Branch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a config-gated early stress branch and `epsilon(u)`-guided stress head while preserving existing mixed-PINN interfaces and keeping old behavior unchanged when the new flags are off.

**Architecture:** Keep the shared trunk, split into displacement and stress branches earlier, and feed stress prediction with an explicit strain bridge derived from the displacement branch. Reuse current mixed residual APIs and canonical traction utilities rather than rewriting downstream physics.

**Tech Stack:** Python, TensorFlow, `unittest`, mixed PINN model code, elasticity residual path.

---

### Task 1: Add Failing Config And Model-Surface Tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_model_outputs.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_model_innovation_hooks.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_config.py`

**Step 1: Write the failing test**

Add tests that assert:

- model config exposes `stress_branch_early_split`
- model config exposes `use_eps_guided_stress_head`
- turning these on still allows `us_fn()` to return valid `(u, sigma)`
- turning these off preserves current behavior

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_mixed_model_outputs.py test_model_innovation_hooks.py -v
```

Expected:

- new config fields missing or model path not recognizing them

**Step 3: Write minimal implementation**

Add the config fields in the model-facing config types and thread them through model construction without changing behavior yet.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_mixed_model_outputs.py test_model_innovation_hooks.py -v
```

Expected:

- config surface tests pass

**Step 5: Commit**

```bash
git add test_mixed_model_outputs.py test_model_innovation_hooks.py src/train/trainer_config.py src/model/pinn_model.py
git commit -m "feat: add p2 stress branch config switches"
```

### Task 2: Add Failing Tests For Early Split Stress Path

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_model_outputs.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_model_cache_and_restore.py`

**Step 1: Write the failing test**

Add tests that assert:

- enabling `stress_branch_early_split` preserves output shapes
- `us_fn_pointwise()` still works
- saved/loaded model graphs remain compatible with the new flags

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_mixed_model_outputs.py test_mixed_model_cache_and_restore.py -v
```

Expected:

- new branch path missing or incompatible with current serialization/forward surface

**Step 3: Write minimal implementation**

Modify `src/model/pinn_model.py` so the stress path is no longer only a late dense head on the final shared representation.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_mixed_model_outputs.py test_mixed_model_cache_and_restore.py -v
```

Expected:

- early split tests pass

**Step 5: Commit**

```bash
git add test_mixed_model_outputs.py test_mixed_model_cache_and_restore.py src/model/pinn_model.py
git commit -m "feat: add early split stress branch"
```

### Task 3: Add Failing Tests For Epsilon-Guided Stress Head

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_elasticity_residuals.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_mixed_model_outputs.py`

**Step 1: Write the failing test**

Add tests that assert:

- enabling `use_eps_guided_stress_head` still yields valid `sigma`
- constitutive residual remains computable with the new path
- equilibrium residual path still accepts the resulting `sigma_fn`

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_mixed_elasticity_residuals.py test_mixed_model_outputs.py -v
```

Expected:

- no strain bridge exists yet or mixed residual calls fail

**Step 3: Write minimal implementation**

Implement the `epsilon(u)` bridge in `src/model/pinn_model.py` on stress-returning paths only.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_mixed_elasticity_residuals.py test_mixed_model_outputs.py -v
```

Expected:

- mixed residual tests remain green with the new head enabled

**Step 5: Commit**

```bash
git add test_mixed_elasticity_residuals.py test_mixed_model_outputs.py src/model/pinn_model.py
git commit -m "feat: add epsilon-guided stress head"
```

### Task 4: Run Regression Against Existing Mixed And Stress Paths

**Files:**
- No new files

**Step 1: Run focused regression**

Run:

```bash
python -m unittest test_mixed_model_outputs.py test_mixed_model_cache_and_restore.py test_mixed_elasticity_residuals.py test_mixed_contact_matching.py test_trainer_optimization_hooks.py -v
```

Expected:

- old flags-off behavior remains green
- new flags-on behavior is numerically valid

**Step 2: Check P2-Phase A boundary**

Verify that this change did not also introduce:

- semantic contact features
- high-frequency stress bypass as default
- backend/contact-route changes

**Step 3: Commit**

```bash
git add src/model/pinn_model.py src/train/trainer_config.py test_mixed_model_outputs.py test_mixed_model_cache_and_restore.py test_mixed_elasticity_residuals.py
git commit -m "feat: complete p2 phase-a stress branch upgrade"
```
