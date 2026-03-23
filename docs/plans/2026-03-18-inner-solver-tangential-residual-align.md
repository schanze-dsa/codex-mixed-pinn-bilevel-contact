# Inner Solver Tangential Residual Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align the production tangential residual with the fixed-point gap of the current tangential update map, then verify that diagnostics and fallback logic follow the new residual.

**Architecture:** Keep the current tangential update unchanged and replace only the residual definition and its downstream consumers. Use focused primitive and solver tests to lock the new semantics before re-running frozen-batch and eager-smoke verification.

**Tech Stack:** Python, TensorFlow, `unittest`, JSON diagnostics

---

### Task 1: Add the Failing Tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_contact_inner_kernel_primitives.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_contact_inner_solver.py`

**Step 1: Write the failing primitive test**

- Replace the audit-only expectation that old production residual differs from
  the fixed-point gap.
- Require production `friction_fixed_point_residual(...)` to equal the
  fixed-point gap.

**Step 2: Write the failing solver tests**

- Add a test that one current tangential update reduces the new residual.
- Add a test that tangential fallback follows the new residual, not the old
  `k_t * ds_t`-dominated quantity.

**Step 3: Run the focused tests to confirm red**

Run:

```powershell
python -m unittest test_contact_inner_kernel_primitives test_contact_inner_solver -v
```

Expected: FAIL on the new residual-alignment assertions.

### Task 2: Align the Production Residual

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\contact\contact_inner_kernel_primitives.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\contact\contact_inner_solver.py`

**Step 1: Change the production tangential residual**

- Make `friction_fixed_point_residual(...)` return the fixed-point gap.

**Step 2: Update solver consumers**

- Ensure iteration trace, convergence, fallback reason, and linearization all
  now operate on the aligned residual.

**Step 3: Do not change the update map**

- Keep `target_lambda_t` and `next_lambda_t` logic unchanged in this round.

### Task 3: Verify the Focused Slice

**Files:**
- No additional edits expected

**Step 1: Re-run the focused slice**

```powershell
python -m unittest test_contact_inner_kernel_primitives test_contact_inner_solver test_contact_inner_solver_linearization test_mixed_contact_matching -v
```

Expected: PASS.

### Task 4: Re-run Frozen-Batch Verification

**Files:**
- Output: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_tangential_frozen_batch_compare_aligned.json`

**Step 1: Re-run the same frozen batch**

- Same real batch, same current tangential update.

**Step 2: Verify**

- `ft_residual_after < ft_residual_before`
- `ft_residual_norm` and `fp_gap_norm` are aligned

### Task 5: Re-run Eager Smoke A/B Verification

**Files:**
- Output: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_smoke_route_compare_tangential_aligned_eager.json`

**Step 1: Re-run**

- `forward_only`
- `normal_ready`

**Step 2: Inspect**

- `fallback_trigger_reason`
- rate of `tangential_residual_not_reduced`

### Task 6: Decide Whether to Touch Tangential Update

**Files:**
- No further edits in this step

**Step 1: Interpret**

- If aligned residual now decreases and fallback rate drops, stop here.
- Only if aligned residual still does not decrease should the next round modify
  the tangential update map itself.
