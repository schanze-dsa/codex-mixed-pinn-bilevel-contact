# PDF Strict Mixed Experimental V2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the V2 strict mixed bilevel checklist on the `strict_mixed_experimental` route while keeping the locked `config.yaml` route behavior unchanged.

**Architecture:** Keep one runtime tree with route-gated behavior. The implementation will tighten the strict mixed inner-solver schema, make trainer diagnostics produce explicit control reasons, formalize `ContactOperator` as an adapter with one typed contract, and make strict mixed stress and outer-loss defaults explicit instead of implicit. The locked route remains conservative; only the experimental route gets the V2 closeout behavior.

**Tech Stack:** Python, TensorFlow, YAML config parsing, `unittest`, trainer mixins, contact operator / inner solver / loss assembly.

---

### Task 1: Close Out The Linearization Schema And Experimental IFT Defaults

**Files:**
- Create: `test_contact_inner_solver_linearization.py`
- Modify: `test_main_new_config_override.py`
- Modify: `strict_mixed_experimental.yaml`
- Modify: `src/physics/contact/contact_inner_solver.py`

**Step 1: Write the failing test**

Add a new `unittest` module that verifies the linearization payload exposes stable V2 metadata instead of only the raw Jacobian tensors.

```python
class ContactInnerSolverLinearizationTests(unittest.TestCase):
    def test_linearization_exposes_v2_schema_and_layout_metadata(self):
        result = solve_contact_inner(
            g_n=tf.constant([-0.1], dtype=tf.float32),
            ds_t=tf.constant([[0.02, 0.0]], dtype=tf.float32),
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=0.3,
            eps_n=1.0e-6,
            k_t=10.0,
            return_linearization=True,
        )

        lin = result.linearization
        self.assertEqual(lin["schema_version"], "strict_mixed_v2")
        self.assertEqual(lin["route_mode"], "normal_ready")
        self.assertIn("state_layout", lin)
        self.assertIn("input_layout", lin)
        self.assertIn("residual_at_solution", lin)
        self.assertEqual(lin["state_layout"]["order"], ["lambda_n", "lambda_t"])
```

Extend `test_main_new_config_override.py` to verify the experimental profile defaults to normal-only IFT with attached gradients:

```python
def test_prepare_config_sets_v2_normal_only_ift_defaults(self):
    cfg, _ = main_new._prepare_config_with_autoguess(
        config_path="strict_mixed_experimental.yaml"
    )
    self.assertTrue(cfg.mixed_bilevel_phase.normal_ift_enabled)
    self.assertFalse(cfg.mixed_bilevel_phase.tangential_ift_enabled)
    self.assertFalse(cfg.mixed_bilevel_phase.detach_inner_solution)
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_contact_inner_solver_linearization.py test_main_new_config_override.py -v
```

Expected:
- FAIL because `schema_version`, `state_layout`, and `residual_at_solution` do not exist yet
- FAIL because `strict_mixed_experimental.yaml` still keeps `detach_inner_solution: true`

**Step 3: Write minimal implementation**

In `src/physics/contact/contact_inner_solver.py`, extend the `linearization` dict to include:

```python
linearization = {
    "schema_version": "strict_mixed_v2",
    "route_mode": "normal_ready",
    "is_exact": False,
    "tangential_mode": "smooth_not_enabled",
    "jac_z": ...,
    "jac_inputs": ...,
    "state_layout": {
        "order": ["lambda_n", "lambda_t"],
        "lambda_n_shape": [n_contacts],
        "lambda_t_shape": [n_contacts, 2],
    },
    "input_layout": {
        "order": ["g_n", "ds_t"],
        "g_n_shape": [n_contacts],
        "ds_t_shape": [n_contacts, 2],
    },
    "flat_z": flat_state,
    "flat_inputs": flat_inputs,
    "residual_at_solution": flat_residual,
}
```

Update `strict_mixed_experimental.yaml` so the experimental route explicitly reflects V2:

```yaml
mixed_bilevel_phase:
  phase_name: phase1
  normal_ift_enabled: true
  tangential_ift_enabled: false
  detach_inner_solution: false
```

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_contact_inner_solver_linearization.py test_main_new_config_override.py -v
```

Expected:
- PASS
- the profile still parses as `strict_mixed_experimental`
- the linearization payload now carries stable schema metadata

**Step 5: Commit**

```bash
git add test_contact_inner_solver_linearization.py test_main_new_config_override.py strict_mixed_experimental.yaml src/physics/contact/contact_inner_solver.py
git commit -m "feat: close out strict mixed linearization schema"
```

### Task 2: Turn Inner Diagnostics Into Explicit Trainer Control Reasons

**Files:**
- Create: `test_contact_inner_solver_health_policy.py`
- Modify: `test_trainer_optimization_hooks.py`
- Modify: `src/physics/contact/strict_mixed_policy.py`
- Modify: `src/train/trainer_opt_mixin.py`
- Modify: `src/train/trainer_monitor_mixin.py`

**Step 1: Write the failing test**

Create a new health-policy test for reasoned backoff:

```python
class ContactInnerSolverHealthPolicyTests(unittest.TestCase):
    def test_runtime_policy_exposes_reason_string_and_backoff_flag(self):
        policy = resolve_strict_mixed_runtime_policy(
            {
                "fallback_used": 1.0,
                "max_penetration": 2.0e-3,
                "fb_residual_norm": 1.0e-1,
            },
            route_mode="normal_ready",
        )
        stats = policy.as_stats()
        self.assertEqual(stats["strict_phase_hold"], 1.0)
        self.assertEqual(stats["continuation_backoff_applied"], 1.0)
        self.assertIn("fallback", stats["phase_hold_reason"])
```

Extend `test_trainer_optimization_hooks.py`:

```python
def test_strict_bilevel_stats_record_reason_and_instability_count(self):
    trainer = object.__new__(Trainer)
    trainer.cfg = TrainerConfig(training_profile="strict_mixed_experimental")
    trainer._strict_bilevel_stats = {"total": 0, "converged": 0, "fallback": 0, "skipped": 0}
    trainer._inner_solver_not_stable_count = 0

    out = trainer._accumulate_strict_bilevel_stats(
        {
            "inner_converged": 0.0,
            "inner_fallback_used": 1.0,
            "inner_fb_residual_norm": 0.2,
            "inner_max_penetration": 0.01,
        },
        route_mode="normal_ready",
    )

    self.assertEqual(out["continuation_backoff_applied"], 1.0)
    self.assertIn("fallback", out["phase_hold_reason"])
    self.assertEqual(out["inner_solver_not_stable_count"], 1.0)
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_contact_inner_solver_health_policy.py test_trainer_optimization_hooks.py -v
```

Expected:
- FAIL because the policy stats do not expose `phase_hold_reason`
- FAIL because trainer stats do not expose `continuation_backoff_applied` or `inner_solver_not_stable_count`

**Step 3: Write minimal implementation**

In `src/physics/contact/strict_mixed_policy.py`, extend `as_stats()`:

```python
def as_stats(self) -> Dict[str, object]:
    return {
        "strict_phase_hold": float(self.phase_hold),
        "strict_continuation_backoff": float(self.continuation_backoff),
        "continuation_backoff_applied": float(self.continuation_backoff),
        "strict_force_detach": float(self.force_detach),
        "strict_traction_scale": float(self.traction_scale),
        "phase_hold_reason": ",".join(self.reasons),
    }
```

In `src/train/trainer_opt_mixin.py`, maintain a monotonic counter:

```python
if policy.phase_hold:
    self._inner_solver_not_stable_count = int(
        getattr(self, "_inner_solver_not_stable_count", 0)
    ) + 1

out["inner_solver_not_stable_count"] = float(
    getattr(self, "_inner_solver_not_stable_count", 0)
)
```

Add the new keys to the monitor extraction list in `src/train/trainer_monitor_mixin.py`.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_contact_inner_solver_health_policy.py test_trainer_optimization_hooks.py -v
```

Expected:
- PASS
- trainer-facing logs now surface reasoned hold/backoff diagnostics

**Step 5: Commit**

```bash
git add test_contact_inner_solver_health_policy.py test_trainer_optimization_hooks.py src/physics/contact/strict_mixed_policy.py src/train/trainer_opt_mixin.py src/train/trainer_monitor_mixin.py
git commit -m "feat: expose strict mixed trainer hold reasons"
```

### Task 3: Finish The ContactOperator Adapter Contract

**Files:**
- Modify: `test_contact_inner_solver.py`
- Modify: `test_mixed_contact_matching.py`
- Modify: `src/physics/contact/contact_operator.py`
- Modify: `src/physics/contact/contact_inner_solver.py`

**Step 1: Write the failing test**

Extend the existing operator tests so the typed contract includes metadata and no longer reads like an anonymous tensor bundle.

```python
def test_contact_operator_strict_inputs_include_batch_metadata(self):
    inputs = op.strict_mixed_inputs(u_fn, params={})
    self.assertIsInstance(inputs, StrictMixedContactInputs)
    self.assertIn("weights", inputs.batch_meta)
    self.assertIn("xs", inputs.batch_meta)
    self.assertIn("xm", inputs.batch_meta)
```

Also assert the contract can carry optional ids:

```python
self.assertTrue(hasattr(inputs, "contact_ids"))
self.assertTrue(hasattr(inputs, "batch_meta"))
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_contact_inner_solver.py test_mixed_contact_matching.py -v
```

Expected:
- FAIL because `StrictMixedContactInputs` does not yet carry `batch_meta` or `contact_ids`

**Step 3: Write minimal implementation**

Update `StrictMixedContactInputs` in `src/physics/contact/contact_operator.py`:

```python
@dataclass
class StrictMixedContactInputs:
    g_n: tf.Tensor
    ds_t: tf.Tensor
    normals: tf.Tensor
    t1: tf.Tensor
    t2: tf.Tensor
    weights: tf.Tensor
    xs: tf.Tensor
    xm: tf.Tensor
    mu: tf.Tensor
    eps_n: tf.Tensor
    k_t: tf.Tensor
    init_state: Optional[ContactInnerState] = None
    contact_ids: Optional[tf.Tensor] = None
    batch_meta: Optional[Dict[str, tf.Tensor]] = None
```

Populate `batch_meta` from the operator’s current contact frame:

```python
batch_meta = {
    "weights": frame["weights"],
    "xs": frame["xs"],
    "xm": frame["xm"],
}
```

At the top of `src/physics/contact/contact_operator.py`, rewrite the module/class docstrings so they describe adapter / dispatcher behavior instead of a unified contact solver identity.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_contact_inner_solver.py test_mixed_contact_matching.py -v
```

Expected:
- PASS
- the operator contract is explicit and metadata-bearing

**Step 5: Commit**

```bash
git add test_contact_inner_solver.py test_mixed_contact_matching.py src/physics/contact/contact_operator.py src/physics/contact/contact_inner_solver.py
git commit -m "refactor: finalize strict mixed contact adapter contract"
```

### Task 4: Make Strict Mixed Stress Defaults Explicit And Route Contact Semantics Through Forward

**Files:**
- Modify: `test_main_new_config_override.py`
- Modify: `test_mixed_model_outputs.py`
- Modify: `strict_mixed_experimental.yaml`
- Modify: `main new.py`
- Modify: `src/train/trainer_config.py`
- Modify: `src/model/pinn_model.py`

**Step 1: Write the failing test**

Add a config test:

```python
def test_prepare_config_sets_strict_mixed_stress_defaults(self):
    cfg, _ = main_new._prepare_config_with_autoguess(
        config_path="strict_mixed_experimental.yaml"
    )
    self.assertTrue(cfg.model_cfg.field.strict_mixed_default_eps_bridge)
    self.assertTrue(cfg.model_cfg.field.strict_mixed_contact_pointwise_stress)
```

Add a model test that contact-surface stress takes the explicit strict mixed route while bulk stress can remain unchanged:

```python
def test_strict_mixed_contact_surface_defaults_to_pointwise_eps_bridge(self):
    cfg = ModelConfig(
        encoder=EncoderConfig(out_dim=8),
        field=FieldConfig(
            cond_dim=8,
            use_graph=True,
            graph_layers=1,
            graph_width=16,
            graph_k=2,
            stress_out_dim=6,
            strict_mixed_default_eps_bridge=True,
            strict_mixed_contact_pointwise_stress=True,
        ),
    )
    model = create_displacement_model(cfg)
    _, sigma = model.us_fn_pointwise(X, params_with_contact_frame)
    self.assertEqual(tuple(sigma.shape), (4, 6))
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_main_new_config_override.py test_mixed_model_outputs.py -v
```

Expected:
- FAIL because the new config fields do not exist yet
- FAIL because the model cannot distinguish explicit strict mixed defaults from the old implicit behavior

**Step 3: Write minimal implementation**

Add new config fields in `src/train/trainer_config.py` and parse them in `main new.py`:

```python
strict_mixed_default_eps_bridge: bool = False
strict_mixed_contact_pointwise_stress: bool = False
```

Enable them in `strict_mixed_experimental.yaml`:

```yaml
network_config:
  use_eps_guided_stress_head: true
  use_engineering_semantics: true
  semantic_feat_dim: 4
  strict_mixed_default_eps_bridge: true
  strict_mixed_contact_pointwise_stress: true
```

In `src/model/pinn_model.py`, route contact-surface stress through the explicit strict mixed defaults instead of inferring only from frame presence:

```python
contact_surface_active = self._extract_contact_surface_frame(params) is not None
force_pointwise = bool(
    force_pointwise
    or (contact_surface_active and self.field.strict_mixed_contact_pointwise_stress)
)
use_eps_bridge = bool(
    self.field.use_eps_guided_stress_head
    or (contact_surface_active and self.field.strict_mixed_default_eps_bridge)
)
```

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_main_new_config_override.py test_mixed_model_outputs.py -v
```

Expected:
- PASS
- explicit strict mixed stress defaults are now visible in config and in model behavior

**Step 5: Commit**

```bash
git add test_main_new_config_override.py test_mixed_model_outputs.py strict_mixed_experimental.yaml main\ new.py src/train/trainer_config.py src/model/pinn_model.py
git commit -m "feat: make strict mixed stress defaults explicit"
```

### Task 5: Add An Explicit Strict Mixed Outer-Loss Assembler And Use It In Trainer Dispatch

**Files:**
- Modify: `test_mixed_contact_matching.py`
- Modify: `test_trainer_optimization_hooks.py`
- Modify: `src/model/loss_energy.py`
- Modify: `src/train/trainer_opt_mixin.py`

**Step 1: Write the failing test**

Add a loss-assembly test:

```python
def test_total_energy_exposes_explicit_strict_outer_loss_assembler(self):
    total = TotalEnergy(TotalConfig(loss_mode="residual"))
    self.assertTrue(callable(total.assemble_strict_mixed_outer_loss))
```

Add a trainer dispatch test:

```python
def test_evaluate_total_objective_calls_explicit_strict_outer_loss_assembler(self):
    trainer = object.__new__(Trainer)
    trainer.cfg = TrainerConfig(training_profile="strict_mixed_experimental")
    trainer._mixed_phase_flags = {
        "phase_name": "phase1",
        "normal_ift_enabled": True,
        "tangential_ift_enabled": False,
        "detach_inner_solution": False,
    }

    called = {"count": 0}

    class _FakeTotal:
        def assemble_strict_mixed_outer_loss(self, *args, **kwargs):
            called["count"] += 1
            return tf.constant(0.0), {"R_t": tf.constant(0.0)}, {"strict_route_mode": "normal_ready"}

    trainer._evaluate_total_objective(_FakeTotal(), params={})
    self.assertEqual(called["count"], 1)
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m unittest test_mixed_contact_matching.py test_trainer_optimization_hooks.py -v
```

Expected:
- FAIL because `assemble_strict_mixed_outer_loss` does not exist yet
- FAIL because trainer still dispatches through `strict_mixed_objective()` only

**Step 3: Write minimal implementation**

In `src/model/loss_energy.py`, add a named assembler that returns the same triple shape the trainer expects:

```python
def assemble_strict_mixed_outer_loss(self, u_fn, params=None, tape=None, stress_fn=None):
    return self.strict_mixed_objective(u_fn, params=params, tape=tape, stress_fn=stress_fn)
```

In `src/train/trainer_opt_mixin.py`, dispatch explicitly:

```python
if route_mode == "legacy":
    Pi, parts, stats = total.energy(...)
else:
    Pi, parts, stats = total.assemble_strict_mixed_outer_loss(
        self.model.u_fn,
        params=params,
        tape=tape,
        stress_fn=stress_fn,
    )
```

Keep `strict_mixed_objective()` in place as the compatibility wrapper so the refactor remains incremental.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m unittest test_mixed_contact_matching.py test_trainer_optimization_hooks.py -v
```

Expected:
- PASS
- trainer strict-route dispatch is now explicit in code

**Step 5: Commit**

```bash
git add test_mixed_contact_matching.py test_trainer_optimization_hooks.py src/model/loss_energy.py src/train/trainer_opt_mixin.py
git commit -m "refactor: add explicit strict mixed outer loss assembler"
```

### Task 6: Run Focused Regression And Profile Smoke Checks

**Files:**
- No new files

**Step 1: Run the V2 focused regression**

Run:

```bash
python -m unittest test_main_new_config_override.py test_contact_inner_solver.py test_contact_inner_solver_linearization.py test_contact_inner_solver_health_policy.py test_mixed_contact_matching.py test_mixed_bilevel_diagnostics.py test_mixed_training_phase_controls.py test_trainer_optimization_hooks.py test_mixed_model_outputs.py -v
```

Expected:
- all targeted strict mixed regression tests PASS
- the locked route tests still PASS

**Step 2: Run locked-route smoke verification**

Run:

```bash
python -m unittest test_main_new_config_override.py test_trainer_optimization_hooks.py -v
```

Expected:
- PASS
- locked-route parsing and trainer formatting remain unchanged

**Step 3: Run experimental-profile smoke verification**

Run:

```bash
@'
import importlib.util
import pathlib

root = pathlib.Path(r"D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact")
spec = importlib.util.spec_from_file_location("main_new_module", root / "main new.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg, _ = mod._prepare_config_with_autoguess(config_path=str(root / "strict_mixed_experimental.yaml"))
print(cfg.training_profile)
print(cfg.mixed_bilevel_phase.normal_ift_enabled)
print(cfg.mixed_bilevel_phase.tangential_ift_enabled)
print(cfg.mixed_bilevel_phase.detach_inner_solution)
'@ | python -
```

Expected:
- prints `strict_mixed_experimental`
- prints `True`
- prints `False`
- prints `False`

**Step 4: Commit**

```bash
git add test_main_new_config_override.py test_contact_inner_solver.py test_contact_inner_solver_linearization.py test_contact_inner_solver_health_policy.py test_mixed_contact_matching.py test_trainer_optimization_hooks.py test_mixed_model_outputs.py strict_mixed_experimental.yaml main\ new.py src/train/trainer_config.py src/train/trainer_opt_mixin.py src/train/trainer_monitor_mixin.py src/model/pinn_model.py src/model/loss_energy.py src/physics/contact/contact_operator.py src/physics/contact/contact_inner_solver.py src/physics/contact/strict_mixed_policy.py
git commit -m "feat: close out strict mixed experimental v2 route"
```
