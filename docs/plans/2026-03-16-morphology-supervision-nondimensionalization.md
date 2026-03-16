# Morphology-Aware Supervision And Nondimensionalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the default supervised route emphasize local morphology and stage-to-stage displacement changes while moving supervision and physics onto a consistent nondimensional scale path.

**Architecture:** Extend the supervision dataset loader to precompute nondimensional observations, morphology weights, and staged displacement deltas. Then update `TotalEnergy` to consume those tensors through a weighted displacement term plus a new stage-delta term, and wire `PhysicalScaleConfig` plus config parsing so coordinates, displacements, and residual scaling resolve from the same reference-scale system.

**Tech Stack:** Python, TensorFlow, unittest, YAML config parsing, NumPy, pandas

---

### Task 1: Lock supervision dataset feature derivation with failing tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_ansys_supervision_dataset.py`

**Step 1: Write the failing test**

Add a test that loads a tiny staged supervision dataset and asserts that each loaded case now includes:

- nondimensional observation coordinates
- nondimensional observed displacements
- per-point morphology weights with mean close to `1.0`
- adjacent-stage displacement deltas

Assert the shapes explicitly, for example:

```python
self.assertEqual(case["X_obs_nd"].shape, case["X_obs"].shape)
self.assertEqual(case["U_obs_nd"].shape, case["U_obs"].shape)
self.assertEqual(case["obs_morphology_weight"].shape, case["U_obs"].shape[:2] + (1,))
self.assertEqual(case["U_obs_delta"].shape, (2, case["U_obs"].shape[1], case["U_obs"].shape[2]))
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_ansys_supervision_dataset.AnsysSupervisionDatasetTests.test_loader_adds_nondimensional_supervision_features -v`

Expected: FAIL because the loader does not yet populate those derived tensors.

### Task 2: Implement supervision feature derivation in the loader

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\ansys_supervision.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_ansys_supervision_dataset.py`

**Step 1: Add small helper functions**

Add focused helpers for:

- resolving per-case or global reference scales used during dataset normalization
- converting `X_obs` and `U_obs` into nondimensional tensors
- building local-contrast and magnitude-based morphology weights
- computing adjacent-stage displacement deltas

Keep each helper narrow and deterministic.

**Step 2: Extend the loaded case payload**

Populate each case with fields such as:

- `X_obs_nd`
- `U_obs_nd`
- `obs_morphology_weight`
- `U_obs_delta`

Keep the existing `X_obs` and `U_obs` fields unchanged so downstream code can migrate incrementally.

**Step 3: Run the targeted loader test**

Run: `python -m unittest test_ansys_supervision_dataset.AnsysSupervisionDatasetTests.test_loader_adds_nondimensional_supervision_features -v`

Expected: PASS.

### Task 3: Lock physical scale resolution with failing tests

**Files:**
- Create: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_physical_scales.py`

**Step 1: Write the failing tests**

Add tests for `PhysicalScaleConfig` covering:

- explicit `L_ref`, `u_ref`, and `sigma_ref`
- fallback `sigma_ref` resolution from `E_ref * u_ref / L_ref`
- guardrails that clamp or reject degenerate zero / negative scales

Example test skeleton:

```python
def test_resolved_sigma_ref_uses_strain_scale_when_explicit_sigma_missing(self):
    cfg = PhysicalScaleConfig(L_ref=10.0, u_ref=0.5, E_ref=200.0)
    self.assertAlmostEqual(cfg.resolved_sigma_ref(), 10.0)
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_physical_scales.PhysicalScaleConfigTests -v`

Expected: FAIL because the existing class does not yet expose the full normalized-scale behavior needed by the new route.

### Task 4: Implement full physical-scale resolution

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\physical_scales.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_physical_scales.py`

**Step 1: Extend the scale utility**

Add helpers that:

- resolve stable positive `L_ref` and `u_ref`
- resolve `sigma_ref` consistently
- optionally report the final scale triple in a form usable by logging

Keep the API small and easy to assert in tests.

**Step 2: Re-run the physical-scale tests**

Run: `python -m unittest test_physical_scales.PhysicalScaleConfigTests -v`

Expected: PASS.

### Task 5: Lock config parsing and default-route scale wiring with failing tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_main_new_config_override.py`

**Step 1: Write the failing test**

Add a config parsing test that feeds `_prepare_config_with_autoguess()` a YAML payload with:

- `physical_scales.L_ref`
- `physical_scales.u_ref`
- `physical_scales.E_ref`
- default supervision enabled

Assert that the parsed config stores those references, resolves a positive `sigma_ref`, and keeps two-stage training plus existing supervision defaults intact.

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_main_new_config_override.MainNewConfigOverrideTests.test_prepare_config_parses_full_physical_scale_controls -v`

Expected: FAIL because the parser does not yet expose the new scale path end-to-end.

### Task 6: Implement config parsing and runtime scale logging

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\main new.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\train\trainer_config.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_main_new_config_override.py`

**Step 1: Parse full physical-scale controls**

Make `_prepare_config_with_autoguess()` parse and store:

- `L_ref`
- `u_ref`
- `sigma_ref`
- `E_ref`
- any needed derived values for the trainer and loss path

**Step 2: Add startup logging**

Print the final resolved reference scales and the representative nondimensional supervision summaries during config preparation or trainer initialization.

**Step 3: Re-run the targeted config test**

Run: `python -m unittest test_main_new_config_override.MainNewConfigOverrideTests.test_prepare_config_parses_full_physical_scale_controls -v`

Expected: PASS.

### Task 7: Lock weighted supervision and stage-delta behavior with failing tests

**Files:**
- Create: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_loss_energy_supervision.py`

**Step 1: Write the failing tests**

Add focused `TotalEnergy` tests for:

- weighted supervision penalizes a local high-contrast error more than the old uniform average
- stage-delta supervision increases when adjacent-stage increments are wrong even if static snapshots are otherwise close
- morphology weighting becomes nearly uniform for a spatially smooth observed field

Example skeleton:

```python
def test_weighted_data_supervision_emphasizes_local_feature_error(self):
    total = TotalEnergy(TotalConfig(w_data=1.0))
    loss, _, stats = total._compute_data_supervision_terms(u_fn, params)
    self.assertGreater(float(stats["data_weighted_rel_rms"]), float(stats["data_rel_rms"]))
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_loss_energy_supervision.TotalEnergySupervisionTests -v`

Expected: FAIL because `TotalEnergy` does not yet compute weighted supervision or stage-delta supervision.

### Task 8: Implement weighted supervision and stage-delta loss terms

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\model\loss_energy.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_loss_energy_supervision.py`

**Step 1: Extend `TotalConfig` minimally**

Add only the new knobs required by the new default route, such as:

- stage-delta supervision weight
- optional morphology-weight clipping controls if needed

Keep defaults aligned with the new main route.

**Step 2: Update supervision computation**

Change `_compute_data_supervision_terms()` so it:

- prefers nondimensional observations when present
- uses morphology weights by default
- computes and returns a separate `E_stage_delta`
- reports clear stats for weighted supervision and delta supervision

**Step 3: Wire the new term into staged and non-staged parts**

Make sure `E_stage_delta` enters the combined loss path consistently and is visible in stats.

**Step 4: Re-run the targeted supervision tests**

Run: `python -m unittest test_loss_energy_supervision.TotalEnergySupervisionTests -v`

Expected: PASS.

### Task 9: Lock nondimensional invariance with failing tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_loss_energy_supervision.py`

**Step 1: Write the failing invariance test**

Add a test that constructs the same physical sample in two unit systems, for example meters and millimeters, together with consistent `L_ref` and `u_ref`, and asserts that the nondimensionalized supervision losses match within tolerance.

**Step 2: Run test to verify it fails**

Run: `python -m unittest test_loss_energy_supervision.TotalEnergySupervisionTests.test_nondimensional_supervision_is_unit_invariant -v`

Expected: FAIL until the loss path consistently prefers nondimensional tensors and scales.

### Task 10: Implement the remaining nondimensional loss-path integration

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\model\loss_energy.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\main new.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\config.yaml`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\config_ansys_supervised.yaml`

**Step 1: Route losses through nondimensional tensors**

Ensure the supervision path uses nondimensional tensors consistently instead of mixing raw-unit and normalized inputs.

**Step 2: Update default configs**

Set explicit default reference scales and default stage-delta supervision weight in both default configs.

**Step 3: Re-run the invariance test**

Run: `python -m unittest test_loss_energy_supervision.TotalEnergySupervisionTests.test_nondimensional_supervision_is_unit_invariant -v`

Expected: PASS.

### Task 11: Run targeted regressions

**Files:**
- Modify: none

**Step 1: Run the targeted test set**

Run:

`python -m unittest test_ansys_supervision_dataset.AnsysSupervisionDatasetTests.test_loader_adds_nondimensional_supervision_features test_physical_scales.PhysicalScaleConfigTests test_main_new_config_override.MainNewConfigOverrideTests.test_prepare_config_parses_full_physical_scale_controls test_loss_energy_supervision.TotalEnergySupervisionTests -v`

Expected: PASS.

### Task 12: Run broader regressions

**Files:**
- Modify: none

**Step 1: Run the broader suite**

Run:

`python -m unittest test_ansys_supervision_dataset test_main_new_config_override test_loss_energy_supervision test_trainer_optimization_hooks -v`

Expected: PASS.

### Task 13: Do a short runtime smoke check

**Files:**
- Modify: none

**Step 1: Run a short supervised smoke training**

Use the default config with a small step budget and confirm from logs that:

- resolved `L_ref`, `u_ref`, and `sigma_ref` are printed
- nondimensional observation magnitudes are in reasonable `O(1)` ranges
- morphology weights print sane min / mean / max values
- `E_data`, `E_stage_delta`, and `E_smooth` all appear in the stats stream

**Step 2: Check the outputs**

Confirm that the short run does not crash and that the new supervision terms decrease at least directionally.

### Task 14: Report residual risk

**Files:**
- Modify: none

**Step 1: Separate code verification from quality verification**

If the full retraining is not completed in-session, report:

- code/test status from the unit and regression suites
- runtime smoke evidence
- remaining uncertainty around full-quality morphology improvement on the complete ANSYS training run

### Execution Notes

- Follow TDD strictly: no production code before the matching failing test is observed.
- Do not commit from this worktree unless the changes are first isolated from the unrelated pending edits already present in the repository.
