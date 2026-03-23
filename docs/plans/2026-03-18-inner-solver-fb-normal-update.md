# Inner Solver FB Normal Update Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove false convergence from the strict inner solver by gating on the normal FB residual, updating the normal multiplier with an FB-residual-guided correction, and exposing clearer failure reasons in the iteration trace.

**Architecture:** Keep the solver stateless and preserve the current public API shape. Add focused tests first, then update the normal solve loop and fallback labeling, and finally re-run the real-batch trace to confirm the normal residual trajectory changes in the intended direction.

**Tech Stack:** Python, TensorFlow, `unittest`, JSON diagnostics

---

### Task 1: Lock the Regressions with Failing Tests

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_contact_inner_solver.py`

**Step 1: Write the failing tests**

- Add a test showing a negative-gap normal-only case must not report `converged=1` when `fb_residual_norm` is still large.
- Add a test showing the normal-only iteration trace keeps reducing `fn_residual_after` after the first iteration.
- Add a test showing a tangential-only failure is labeled `tangential_residual_not_reduced`.

**Step 2: Run the focused tests and verify they fail**

Run:

```powershell
python -m unittest `
  test_contact_inner_solver.ContactInnerSolverTests.test_inner_solver_does_not_false_converge_when_normal_fb_residual_is_large `
  test_contact_inner_solver.ContactInnerSolverTests.test_normal_only_trace_keeps_reducing_fb_residual `
  test_contact_inner_solver.ContactInnerSolverTests.test_iteration_trace_classifies_tangential_fallback `
  -v
```

Expected: FAIL on the current implementation.

### Task 2: Update the Solver

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\contact\contact_inner_solver.py`

**Step 1: Add the normal FB tolerance gate**

- Add `tol_fb` to `solve_contact_inner(...)`.
- Default it to `tol_n`.
- Require the normal FB residual to be below `tol_fb` before reporting convergence.

**Step 2: Replace the normal update**

- Remove the direct attraction to `target_lambda_n`.
- Compute a damped diagonal correction from the current FB residual and diagonal derivative.
- Clamp the updated `lambda_n` to the nonnegative cone.

**Step 3: Refine failure reasons**

- Emit `normal_fb_residual_not_reduced`, `tangential_residual_not_reduced`, `policy_penetration_gate`, `invalid_diag`, or `nan_or_inf` as appropriate.

### Task 3: Verify the Focused Solver Slice

**Files:**
- No additional code changes expected

**Step 1: Re-run the focused tests**

Run:

```powershell
python -m unittest test_contact_inner_solver -v
```

Expected: PASS.

**Step 2: Re-run the adjacent regression slice**

Run:

```powershell
python -m unittest test_contact_inner_solver test_contact_inner_solver_linearization test_mixed_contact_matching -v
```

Expected: PASS.

### Task 4: Re-run the Real-Batch Comparison

**Files:**
- Output: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_inner_iteration_trace_compare.json`

**Step 1: Re-run the same one-batch two-case diagnostic**

- `normal-only`
- `normal + weakened tangential`

**Step 2: Compare the trace**

- Confirm `normal-only` does not false-converge with a large FB residual.
- Confirm `fn_residual_after` continues decreasing.
- Confirm fallback reasons separate normal and tangential causes.

### Task 5: Final Verification and Hygiene

**Files:**
- No additional source changes expected

**Step 1: Check exact command outputs**

- focused tests
- regression slice
- one-batch diagnostic

**Step 2: Check probe residue**

- repository root
- `tmp/`
