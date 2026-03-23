# Inner Solver Tangential Proposal Compare Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the last two tangential diagnostics and run one frozen-batch A/B/C comparison that identifies which proposal actually lowers the current tangential residual.

**Architecture:** Keep the production solver unchanged except for two trace-only scalar diagnostics. Use a one-off frozen-batch script under `tmp/` to compare three tangential proposals against the same real batch after the normal substep has been frozen.

**Tech Stack:** Python, TensorFlow, `unittest`, JSON diagnostics

---

### Task 1: Extend the Failing Trace Test

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_contact_inner_solver.py`

**Step 1: Write the failing test**

- Extend the existing iteration-trace test so each iteration must expose:
  - `target_lambda_t_norm`
  - `ft_reduction_ratio`

**Step 2: Run the focused test to confirm red**

Run:

```powershell
python -m unittest test_contact_inner_solver.ContactInnerSolverTests.test_iteration_trace_exposes_requested_core_metrics -v
```

Expected: FAIL because the new keys are missing.

### Task 2: Add the Minimal Trace Fields

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\contact\contact_inner_solver.py`

**Step 1: Add `target_lambda_t_norm`**

- Compute the norm of `target_lambda_t` before the damped update is applied.

**Step 2: Add `ft_reduction_ratio`**

- Compute `ft_residual_after / (ft_residual_before + eps)` as a scalar trace field.

**Step 3: Keep runtime behavior unchanged**

- No update formula changes.
- No convergence rule changes.

### Task 3: Verify the Focused Solver Slice

**Files:**
- No additional edits expected

**Step 1: Re-run the focused test**

```powershell
python -m unittest test_contact_inner_solver.ContactInnerSolverTests.test_iteration_trace_exposes_requested_core_metrics -v
```

Expected: PASS.

**Step 2: Re-run the adjacent solver slice**

```powershell
python -m unittest test_contact_inner_solver test_contact_inner_solver_linearization test_mixed_contact_matching -v
```

Expected: PASS.

### Task 4: Run the Frozen-Batch A/B/C Comparison

**Files:**
- Output: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_tangential_frozen_batch_compare.json`

**Step 1: Capture one real batch**

- Use the current smoke configuration.
- Save the last strict-mixed batch inputs from a real run.

**Step 2: Freeze the batch after the normal substep**

- Reuse the same `next_lambda_n` for all tangential proposals.

**Step 3: Compare**

- Proposal A: current update
- Proposal B: direct projection
- Proposal C: residual-driven proposal based on the current codebase residual

**Step 4: Record**

- `ft_residual_before`
- `ft_residual_after`
- `lambda_t_after_norm`
- `cone_violation_after`
- `target_lambda_t_norm`

### Task 5: Report the Next Cut

**Files:**
- No code edits required

**Step 1: Identify the best proposal**

- A proposal is better only if it materially reduces `ft_residual_after`.

**Step 2: Recommend the production change**

- If one proposal is clearly better, the next round should write that proposal
  into the tangential update and add an acceptance/backtracking gate.
