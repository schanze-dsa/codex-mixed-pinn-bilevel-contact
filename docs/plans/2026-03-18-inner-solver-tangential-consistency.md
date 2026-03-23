# Inner Solver Tangential Consistency Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Audit whether the current tangential residual matches the fixed-point equation of the current tangential update map, without changing solver behavior.

**Architecture:** Add explicit tangential update-map and fixed-point-gap primitives, lock them with focused tests, and run a one-off frozen-batch audit that compares the existing residual norm against the fixed-point gap norm on the same real batch.

**Tech Stack:** Python, TensorFlow, `unittest`, JSON diagnostics

---

### Task 1: Add the Failing Primitive Tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_contact_inner_kernel_primitives.py`

**Step 1: Write the failing tests**

- Add a test for `tangential_update_map(...)`.
- Add a test for `tangential_fixed_point_gap(...)`.
- Add a relation test that proves:

```python
friction_fixed_point_residual == tangential_fixed_point_gap + k_t * ds_t
```

**Step 2: Run the tests to confirm red**

Run:

```powershell
python -m unittest test_contact_inner_kernel_primitives.ContactInnerKernelPrimitiveTests.test_tangential_fixed_point_gap_matches_lambda_minus_update_map test_contact_inner_kernel_primitives.ContactInnerKernelPrimitiveTests.test_friction_residual_differs_from_fixed_point_gap_by_kt_ds_t -v
```

Expected: FAIL because the new helpers do not exist yet.

### Task 2: Add Minimal Kernel Helpers

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\contact\contact_inner_kernel_primitives.py`

**Step 1: Add `tangential_update_map(...)`**

- Reuse the existing projection logic.

**Step 2: Add `tangential_fixed_point_gap(...)`**

- Define it explicitly as `lambda_t - T(lambda_t)`.

**Step 3: Keep the current residual untouched**

- `friction_fixed_point_residual(...)` must keep its current behavior in this
  round.

### Task 3: Verify the Primitive Slice

**Files:**
- No additional edits expected

**Step 1: Re-run the new focused tests**

```powershell
python -m unittest test_contact_inner_kernel_primitives.ContactInnerKernelPrimitiveTests.test_tangential_fixed_point_gap_matches_lambda_minus_update_map test_contact_inner_kernel_primitives.ContactInnerKernelPrimitiveTests.test_friction_residual_differs_from_fixed_point_gap_by_kt_ds_t -v
```

Expected: PASS.

**Step 2: Re-run the adjacent primitive + solver slice**

```powershell
python -m unittest test_contact_inner_kernel_primitives test_contact_inner_solver test_contact_inner_solver_linearization test_mixed_contact_matching -v
```

Expected: PASS.

### Task 4: Run the Frozen-Batch Consistency Audit

**Files:**
- Output: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_tangential_consistency_audit.json`

**Step 1: Reuse one real frozen batch**

- Same smoke-style real batch as the prior tangential compare.

**Step 2: Compute both norms**

- `ft_residual_norm`
- `fp_gap_norm`

**Step 3: Record scale and relation**

- Include enough scalar detail to show whether the two norms are the same
  quantity or clearly different.

### Task 5: Recommend the Next Production Change

**Files:**
- No code edits required

**Step 1: Interpret**

- If the norms are inconsistent, the next cut is to align the tangential
  residual definition with `lambda_t - T(lambda_t)`.
- If they are consistent, the next cut is to change the tangential update map
  itself.
