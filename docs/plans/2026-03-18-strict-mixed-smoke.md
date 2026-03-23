# Strict Mixed Smoke Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create and run a dedicated short normal-ready smoke training config for the strict mixed route.

**Architecture:** Derive a new YAML from `strict_mixed_experimental.yaml`, keep the strict mixed normal-ready route intact, disable two-stage execution, and sharply reduce runtime-sensitive settings before launching `main new.py` against the new config.

**Tech Stack:** Python 3.8, YAML config, TensorFlow training entrypoint

---

### Task 1: Create Smoke YAML

**Files:**
- Create: `strict_mixed_experimental_smoke.yaml`

**Step 1: Copy the strict mixed experimental structure**

- Preserve route settings:
  - `training_profile: strict_mixed_experimental`
  - `mixed_bilevel_phase.phase_name: phase1`
  - `mixed_bilevel_phase.normal_ift_enabled: true`
  - `mixed_bilevel_phase.tangential_ift_enabled: false`
  - `mixed_bilevel_phase.detach_inner_solution: false`
  - `contact_backend: inner_solver`

**Step 2: Reduce runtime-sensitive settings**

- `two_stage_training.enabled: false`
- `optimizer_config.epochs: 5`
- `optimizer_config.log_every: 1`
- `optimizer_config.validation_eval_every: 1`
- `stage_schedule_steps: [5, 5, 5]`
- `stage_inner_steps: 1`
- `n_contact_points_per_pair: 64`
- `tightening_n_points_each: 32`
- `contact_mortar_max_points: 256`
- `elasticity_config.n_points_per_step: 512`
- use a smoke-specific output directory

### Task 2: Validate YAML Loading

**Files:**
- Test only

**Step 1: Run config parse through `main new.py` helpers**

Run a small Python snippet that calls `_prepare_config_with_autoguess()` on the
new YAML.

Expected: config loads successfully and still resolves to the strict mixed
normal-ready route.

### Task 3: Run Smoke Training

**Files:**
- Test only

**Step 1: Launch smoke training**

Run:

```bash
python ".\main new.py" --config strict_mixed_experimental_smoke.yaml
```

Expected: the training chain runs end-to-end far enough to prove the route is
wired, without using the full experiment settings.
