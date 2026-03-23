# Inner Solver Tangential Trace Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the minimum tangential diagnostics needed to explain why the inner solver still falls back on `tangential_residual_not_reduced`.

**Architecture:** Reuse the existing `iteration_trace` path and extend each iteration record with a few scalar tangential summaries. Keep solver behavior unchanged; this round is diagnosis only.

**Tech Stack:** Python, TensorFlow, `unittest`, JSON diagnostics

---

### Task 1: Extend the Failing Trace Test

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_contact_inner_solver.py`

**Step 1: Write the failing test**

- Extend `test_iteration_trace_exposes_requested_core_metrics`.
- Require each iteration record to contain:
  - `lambda_t_before_norm`
  - `lambda_t_after_norm`
  - `cone_violation_before`
  - `cone_violation_after`
  - `slip_norm`

**Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest test_contact_inner_solver.ContactInnerSolverTests.test_iteration_trace_exposes_requested_core_metrics -v
```

Expected: FAIL because the new keys do not exist yet.

### Task 2: Implement Minimal Tangential Trace Instrumentation

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\contact\contact_inner_solver.py`

**Step 1: Add scalar helpers if needed**

- Keep the trace compact.
- Prefer row-wise tangential norms for vector quantities.

**Step 2: Extend per-iteration records**

- Capture:
  - tangential state norm before and after
  - cone violation before and after
  - slip norm from `ds_t`

**Step 3: Keep default behavior unchanged**

- No changes to tangential update.
- No changes to convergence logic.

### Task 3: Verify the Focused Solver Slice

**Files:**
- No additional edits expected

**Step 1: Run the focused trace test**

```powershell
python -m unittest test_contact_inner_solver.ContactInnerSolverTests.test_iteration_trace_exposes_requested_core_metrics -v
```

Expected: PASS.

**Step 2: Run the adjacent solver slice**

```powershell
python -m unittest test_contact_inner_solver test_contact_inner_solver_linearization test_mixed_contact_matching -v
```

Expected: PASS.

### Task 4: Re-run the Eager Smoke A/B Diagnostic

**Files:**
- Output: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_smoke_route_compare_tangential_eager.json`

**Step 1: Re-run**

- `forward_only`
- `normal_ready`

**Step 2: Summarize**

- Compare the last traced batch on:
  - `ft_residual_before/after`
  - `lambda_t_before_norm/after_norm`
  - `delta_lambda_t_norm`
  - `cone_violation_before/after`
  - `slip_norm`
  - `fallback_trigger_reason`

### Task 5: Decide the Next Cut

**Files:**
- No edits required

**Step 1: Classify the result**

- If tangential residual is flat: update formula is the likely issue.
- If tangential residual is clearly descending: threshold/budget is the likely issue.

**Step 2: Report only that conclusion**

- No tangential formula changes in this round.
