# Normal-Only IFT Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add focused `normal_ready` closure tests and the minimal stats plumbing needed for the image-driven acceptance criteria.

**Architecture:** Keep the implementation inside the current strict mixed stats path. The new tests should exercise linearization layout round-tripping, `normal_ready` linearization consumption, and the contrast with `forward_only`. Runtime changes are limited to adding explicit `normal_ift_*` stats and gradient split norms.

**Tech Stack:** Python 3.8, TensorFlow, unittest, dataclasses

---

### Task 1: Add Failing Focused Closure Tests

**Files:**
- Create: `test_layout_roundtrip.py`
- Create: `test_mixed_normal_ready_ift_consumes_linearization.py`
- Create: `test_mixed_forward_only_vs_normal_ready.py`

**Step 1: Write the failing tests**

- Assert the V2 layout metadata can be round-tripped into flat sizes.
- Assert `normal_ready` returns `normal_ift_ready=1`, `normal_ift_consumed=1`,
  finite `normal_ift_condition_metric`, and `normal_ift_valid_ratio>0`.
- Assert `forward_only` keeps those stats disabled while `normal_ready` keeps
  them finite and surfaces finite gradient split norms.

**Step 2: Run tests to verify they fail**

Run:

```bash
python -m unittest test_layout_roundtrip test_mixed_normal_ready_ift_consumes_linearization test_mixed_forward_only_vs_normal_ready -v
```

Expected: failures for missing `normal_ift_*` keys and missing runtime gradient
 stats.

### Task 2: Add Minimal Stats Plumbing

**Files:**
- Modify: `src/model/loss_energy.py`
- Modify: `src/train/trainer_opt_mixin.py`
- Modify: `src/train/trainer_monitor_mixin.py`

**Step 1: Implement missing strict mixed stats**

- Add explicit `normal_ift_*` stats in strict mixed contact terms.
- Add default zero values on skip paths.
- Add gradient split norm helpers and attach them during compiled train steps.

**Step 2: Run the new tests**

Run:

```bash
python -m unittest test_layout_roundtrip test_mixed_normal_ready_ift_consumes_linearization test_mixed_forward_only_vs_normal_ready -v
```

Expected: all new tests pass.

### Task 3: Run Image-Driven Closure Verification

**Files:**
- Test only

**Step 1: Run the minimal closure suite from the screenshot plus the new tests**

Run:

```bash
python -m unittest test_contact_inner_solver_linearization test_mixed_bilevel_diagnostics test_mixed_training_phase_controls test_layout_roundtrip test_mixed_normal_ready_ift_consumes_linearization test_mixed_forward_only_vs_normal_ready -v
```

Expected: full focused closure suite passes.
