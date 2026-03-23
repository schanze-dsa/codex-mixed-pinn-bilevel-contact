# Inner Solver Iteration Trace Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a default-off per-iteration trace to the strict inner solver, validate it with focused tests, and run a one-batch two-case diagnostic.

**Architecture:** Extend the inner solver and strict contact adapter with a trace flag that returns structured scalar summaries without changing the default execution path. Use a focused unit test to lock the interface before implementation, then run a single reconstructed real-batch experiment to compare `normal-only` and `normal + weakened tangential`.

**Tech Stack:** Python, TensorFlow, `unittest`, JSON diagnostics

---

### Task 1: Add the Failing Test

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_contact_inner_solver.py`

**Step 1: Write the failing test**

- Add a test that calls `solve_contact_inner(..., return_iteration_trace=True)`.
- Assert:
  - default path does not expose `iteration_trace`
  - traced path exposes `diagnostics["iteration_trace"]`
  - trace has at least one iteration
  - each iteration contains:
    - `fn_residual_before`
    - `fn_residual_after`
    - `ft_residual_before`
    - `ft_residual_after`
    - `delta_lambda_n_norm`
    - `delta_lambda_t_norm`
  - trace contains `fallback_trigger_reason`
- Extend the contact-operator round-trip test to verify passthrough of `return_iteration_trace=True`.

**Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest test_contact_inner_solver.ContactInnerSolverTests.test_iteration_trace_exposes_requested_core_metrics -v
```

Expected: FAIL because the solver does not yet return `iteration_trace`.

### Task 2: Implement Minimal Trace Support

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\contact\contact_inner_solver.py`
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\contact\contact_operator.py`

**Step 1: Implement inner-solver trace collection**

- Add `return_iteration_trace: bool = False` to `solve_contact_inner(...)`.
- Collect scalar before/after residual summaries per iteration.
- Compute a solver-local `fallback_trigger_reason`.
- Store the trace only when requested.

**Step 2: Add operator passthrough**

- Add the same flag to `ContactOperator.solve_strict_inner(...)`.
- Forward it unchanged to `solve_contact_inner(...)`.

**Step 3: Run the focused test**

Run:

```powershell
python -m unittest test_contact_inner_solver.ContactInnerSolverTests.test_iteration_trace_exposes_requested_core_metrics -v
```

Expected: PASS.

### Task 3: Run the Relevant Regression Slice

**Files:**
- No code changes expected

**Step 1: Run adjacent solver tests**

Run:

```powershell
python -m unittest test_contact_inner_solver test_contact_inner_solver_linearization test_mixed_contact_matching -v
```

Expected: PASS.

### Task 4: Run the One-Batch Diagnostic

**Files:**
- No committed source changes expected
- Output: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_inner_iteration_trace_compare.json`

**Step 1: Reconstruct one real smoke batch**

- Use the smoke config and `ckpt-5`.
- Rebuild the same real-batch stage-0 contact input used in the earlier diagnosis.

**Step 2: Run two cases**

- `normal-only`
- `normal + weakened tangential`

**Step 3: Dump the JSON comparison**

- Include only the requested core metrics plus minimal identifying context.

### Task 5: Verify and Clean Up

**Files:**
- No source changes expected

**Step 1: Check test output and diagnostic artifact**

Run the exact verification commands used above and read the outputs.

**Step 2: Check for probe residue**

- repository root
- `tmp/`

**Step 3: Summarize the measured root cause**

- State whether the tangential weakening changes fallback behavior.
- State whether the normal residual moves at all across iterations.
