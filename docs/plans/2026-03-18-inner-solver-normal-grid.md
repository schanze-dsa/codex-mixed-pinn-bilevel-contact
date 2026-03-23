# Inner Solver Normal-Step Grid Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the normal FB correction gain above `1.0` effective, then run a single-real-batch grid search over gain and iteration budget to measure whether normal-only convergence is limited by step size or by iteration count.

**Architecture:** Keep the change local to `solve_contact_inner(...)`. Reuse the existing `damping` argument as the normal correction gain, scale the clip cap with that gain, and leave tangential, trainer, and policy behavior untouched. Use one focused test to prove the higher gain changes the actual normal-only step, then run the real-batch diagnostic grid and summarize the best setting.

**Tech Stack:** Python, TensorFlow, `unittest`, JSON diagnostics

---

### Task 1: Add the Failing Test

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\test_contact_inner_solver.py`

**Step 1: Write the failing test**

- Add a normal-only trace test that runs the same input twice:
  - once with `damping=1.0`
  - once with `damping=2.0`
- Assert the high-gain run reduces `fn_residual_after` more on the first iteration.

**Step 2: Run the focused test and verify it fails**

Run:

```powershell
python -m unittest test_contact_inner_solver.ContactInnerSolverTests.test_normal_only_higher_damping_reduces_fb_residual_faster -v
```

Expected: FAIL because the current solver clamps `damping` to `<= 1.0`.

### Task 2: Implement the Minimal Solver Change

**Files:**
- Modify: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\contact\contact_inner_solver.py`

**Step 1: Remove the upper clamp on `damping`**

- Keep `damping >= 0.0`.
- Allow values above `1.0`.

**Step 2: Make the gain affect the clip limit**

- Scale the normal correction cap with the same gain so `1.5x` and `2.0x` produce larger effective steps while staying clipped.

### Task 3: Verify the Solver Slice

**Files:**
- No additional code changes expected

**Step 1: Re-run the focused test**

Run:

```powershell
python -m unittest test_contact_inner_solver.ContactInnerSolverTests.test_normal_only_higher_damping_reduces_fb_residual_faster -v
```

Expected: PASS.

**Step 2: Re-run the adjacent solver regression slice**

Run:

```powershell
python -m unittest test_contact_inner_solver test_contact_inner_solver_linearization test_mixed_contact_matching -v
```

Expected: PASS.

### Task 4: Run the Real-Batch Grid

**Files:**
- Output: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_inner_normal_grid.json`

**Step 1: Reconstruct the same real smoke batch**

- use the smoke config
- restore `ckpt-5`
- select the same step-5 train case
- use the same reference solve parameters as the previous diagnosis

**Step 2: Run the normal-only grid**

- gains: `1.0`, `1.5`, `2.0`
- iterations: `8`, `12`, `16`

**Step 3: Record the decay statistics**

- final residual
- `rho_k`
- recent-ratio average
- estimated extra iterations to `tol_fb`

**Step 4: Run one weakened-tangential comparison**

- use the best normal-only setting
- compare against the matching normal-only baseline

### Task 5: Final Verification and Hygiene

**Files:**
- No additional source changes expected

**Step 1: Check exact command outputs**

- focused test
- regression slice
- real-batch grid script

**Step 2: Check temporary probe residue**

- repository root
- `tmp/`
