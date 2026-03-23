# Formal Normal-Ready Step 5 Local Replay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Freeze the already-identified `normal_ready` formal step-5 real case and answer whether increasing `max_inner_iters` alone can drive `ft_residual` below `tol_t` on that same case.

**Architecture:** Do not touch solver/loss/policy/logger mainline behavior. Add a temporary capture-and-replay workflow under `tmp/` that (1) freezes one exact formal case from the current diff-trace route, then (2) replays `solve_contact_inner(...)` locally with a budget sweep. Only after this replay answers the budget question should any formal-route config or solver changes be considered.

**Tech Stack:** Python, TensorFlow eager mode, existing strict-mixed formal smoke runner, temporary JSON/NPZ artifacts, `unittest`.

---

### Task 1: Lock the target case definition

**Files:**
- Read: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_formal_route_diff_trace_compare.json`
- Read: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_smoke_normal_ready_formal_difftrace.json`
- Read: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_smoke_normal_ready_formal_difftrace.log`
- Document only: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\docs\plans\2026-03-20-formal-normal-ready-step5-local-replay.md`

**Step 1: Confirm the exact replay target**

Use the already captured formal diff trace to lock the case identity:
- route: `normal_ready`
- step: `5`
- theta: `[2.5000,2.5000,6.0000]deg`
- order: `3-1-2`
- `P序=[6,2,2]`

**Step 2: Record the baseline local-state facts**

Write down the current step-5 baseline from the diff trace before changing anything:
- `fallback_trigger_reason`
- final `ft_residual_after`
- final `tangential_step_mode`
- final `effective_alpha_scale`
- final `tail_has_effective_step`
- `mu_lambda_n_mean`
- `mu_lambda_n_max`

**Step 3: Do not change scope**

Explicitly keep out of scope for this replay:
- `tol_t`
- normal block
- IFT logic
- policy
- logger format
- full smoke / full training changes

No code is written in this task.

### Task 2: Add a failing test for same-case replay helpers

**Files:**
- Create: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\test_formal_same_case_budget_replay.py`
- Create: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\run_formal_same_case_budget_replay.py`

**Step 1: Write the failing test**

Add a small `unittest` file for the temporary replay script with at least these checks:
- it can identify the locked step-5 case from a route-diff JSON payload
- it can summarize replay results into a budget table keyed by `max_inner_iters`
- it can report whether `ft_residual_after < tol_t`

Suggested test names:
- `test_find_target_case_by_step_theta_and_order`
- `test_summarize_budget_row_reports_tol_crossing`

**Step 2: Run test to verify it fails**

Run:

```powershell
@'
import os, sys, unittest
repo = r'D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact'
sys.path.insert(0, os.path.join(repo, 'tmp'))
suite = unittest.defaultTestLoader.loadTestsFromName('test_formal_same_case_budget_replay')
res = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if res.wasSuccessful() else 1)
'@ | python -
```

Expected: `FAIL` because the replay helper functions do not exist yet.

### Task 3: Implement temp-only case capture for the exact formal step

**Files:**
- Modify/Create: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\run_formal_same_case_budget_replay.py`
- Read: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\run_formal_route_diff_trace.py`
- Read: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\src\physics\contact\contact_operator.py`

**Step 1: Reuse the existing formal normal-ready smoke path**

Do not build a new training path. Reuse the same eager/formal route setup already used in:

`D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\run_formal_route_diff_trace.py`

**Step 2: Capture the exact strict inputs for step 5**

When `ContactOperator.solve_strict_inner(...)` is called on the target step, save enough data to replay it directly:
- `g_n`
- `ds_t`
- `normals`
- `t1`
- `t2`
- `mu`
- `eps_n`
- `k_t`
- `init_state.lambda_n`
- `init_state.lambda_t`
- metadata:
  - `step_index`
  - `theta_label`
  - `order_label`
  - `p_order_label`
  - `route_name`

Prefer a temporary artifact like:

`D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_normal_ready_formal_step5_case.json`

or, if tensor size/layout is easier to preserve, a paired `.npz` + `.json` manifest.

**Step 3: Keep capture logic temp-only**

Do not modify:
- `src/physics/contact/contact_inner_solver.py`
- `src/model/loss_energy.py`
- `src/train/trainer_monitor_mixin.py`
- `src/train/trainer_opt_mixin.py`

The capture must live entirely in the temporary runner.

### Task 4: Verify temp helper tests now pass

**Files:**
- Test: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\test_formal_same_case_budget_replay.py`
- Script: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\run_formal_same_case_budget_replay.py`

**Step 1: Run the temp test suite**

Run:

```powershell
@'
import os, sys, unittest
repo = r'D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact'
sys.path.insert(0, os.path.join(repo, 'tmp'))
suite = unittest.defaultTestLoader.loadTestsFromName('test_formal_same_case_budget_replay')
res = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if res.wasSuccessful() else 1)
'@ | python -
```

Expected: `OK`.

### Task 5: Run the same-case local replay budget sweep

**Files:**
- Script: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\run_formal_same_case_budget_replay.py`
- Create: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_normal_ready_step5_budget_replay.json`

**Step 1: Replay the frozen case with the primary inner-iteration sweep**

Sweep:
- `max_inner_iters = 8`
- `max_inner_iters = 12`
- `max_inner_iters = 16`
- `max_inner_iters = 24`

Keep `max_tail_qn_iters = 4` fixed for the first pass.

**Step 2: Record the required outputs for each row**

For each replay row, store:
- `max_inner_iters`
- `max_tail_qn_iters`
- `final_ft_residual_norm`
- `crosses_tol_t`
- `tol_t`
- `final_tangential_step_mode`
- `final_effective_alpha_scale`
- `final_tail_has_effective_step`
- `tail_reduction_ratio_mean`
- final 3 iterations:
  - `ft_residual_before`
  - `ft_residual_after`
  - `ft_reduction_ratio`
  - `effective_alpha_scale`
  - `tangential_step_mode`

**Step 3: Only if the first sweep is ambiguous, run the tail budget sweep**

Second-pass optional sweep:
- `max_tail_qn_iters = 0`
- `max_tail_qn_iters = 4`
- `max_tail_qn_iters = 8`

Keep this optional. Do it only if `max_inner_iters` alone does not answer the question.

### Task 6: Answer the single decision question

**Files:**
- Read: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_normal_ready_step5_budget_replay.json`

**Step 1: Decide whether budget is enough on this same formal case**

Answer exactly this:

`normal_ready` on this frozen formal step-5 case, as `max_inner_iters` increases, does `ft_residual` keep decreasing and cross `tol_t`?

**Step 2: Use only these two outcomes**

Outcome A:
- if `12` or `16` crosses `tol_t`
- or clearly gets there with stable continuing decline

Then conclude:
- budget is now a meaningful variable on this exact formal case
- next session should do a route-local formal config for `normal_ready`
- after that, rerun formal/full smoke and watch `fallback_used_count` / `converged_count`

Outcome B:
- if `12/16/24` still cannot cross `tol_t`
- or late iterations return to `tail_qn + alpha=0`

Then conclude:
- budget is not the main limiter
- next session should go back to tail-entry / tail-qn / accepted-step mechanics

### Task 7: Produce a short execution handoff note

**Files:**
- Update or create: `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_normal_ready_step5_budget_replay_summary.md`

**Step 1: Write a short summary with only these sections**

- `Frozen Case`
- `Budget Grid`
- `Observed Trend`
- `Decision`

**Step 2: Keep the decision explicit**

End with one of:
- `Proceed with route-local normal_ready budget`
- `Do not pursue budget; return to tail mechanism`

### Task 8: Verification before handoff

**Files:**
- Verify artifacts only; no production code changes expected

**Step 1: Verify the temp tests**

Re-run:

```powershell
@'
import os, sys, unittest
repo = r'D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact'
sys.path.insert(0, os.path.join(repo, 'tmp'))
suite = unittest.defaultTestLoader.loadTestsFromName('test_formal_same_case_budget_replay')
res = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if res.wasSuccessful() else 1)
'@ | python -
```

**Step 2: Verify the replay artifact exists**

Confirm these files exist:
- `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_normal_ready_formal_step5_case.json` or equivalent capture artifact
- `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_normal_ready_step5_budget_replay.json`
- `D:\shuangfan\pinn luowen-worktrees\codex-mixed-pinn-bilevel-contact\tmp\strict_mixed_normal_ready_step5_budget_replay_summary.md`

**Step 3: Do not claim system-level success**

This plan is complete when the local replay budget question is answered.
It is **not** complete when:
- full smoke improves
- formal training is frozen
- `tol_t` is changed

Those are later decisions.
