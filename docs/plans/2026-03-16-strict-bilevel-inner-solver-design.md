# Strict Bilevel Inner Solver Design

## Goal

Implement P0 from the mixed bilevel audit checklist:

- make `solve_contact_inner(...)` a real stateless inner solver
- wire it only into the mixed bilevel / traction matching path
- keep legacy `ContactOperator.residual()` / `update_multipliers()` stable for now

This change is about semantic correctness first. The repository already has normal/friction ALM logic and a placeholder `contact_inner_solver.py`, but the current `solve_contact_inner(...)` still receives precomputed `lambda_n` / `lambda_t` and only packages state, traction, and fallback metadata. That does not satisfy strict bilevel semantics.

## Non-Goals

- Do not fully rewrite legacy `ContactOperator.residual()` or `ContactOperator.update_multipliers()` in this pass.
- Do not merge strict bilevel training semantics back into legacy outer contact penalties.
- Do not redesign the mixed network architecture in this pass.
- Do not add tangential IFT yet; P0 only needs forward-only inner solve plus the normal-IFT-ready interface.

## Current State

The current structure is split across three places:

- `src/physics/contact/contact_normal_alm.py` contains normal FB / projection residual ingredients and legacy multiplier updates.
- `src/physics/contact/contact_friction_alm.py` contains tangential projection-style friction logic and legacy multiplier updates.
- `src/physics/contact/contact_inner_solver.py` currently behaves like a lightweight container. It accepts precomputed `lambda_n` / `lambda_t`, builds traction, and exposes fallback metadata.

This means the current "inner solver" is not solving contact variables from geometry and slip inputs. It only repackages already-computed multipliers.

## Design Summary

### 1. New low-level kernel primitives

Create `src/physics/contact/contact_inner_kernel_primitives.py` and move reusable stateless math there.

Required primitives:

- `fb_normal_residual(g_n, lambda_n, eps_n)`
- `fb_normal_jacobian(g_n, lambda_n, eps_n)`
- `project_to_coulomb_disk(tau_trial, radius, eps)`
- `friction_fixed_point_residual(lambda_t, ds_t, lambda_n, mu, k_t, eps)`
- `compose_contact_traction(lambda_n, lambda_t, normals, t1, t2)`
- `check_contact_feasibility(g_n, lambda_n, lambda_t, mu, tol_n, tol_t)`

Reason:

- P0 needs a real solver now.
- P1 needs `ContactOperator` to become a thin adapter over the same math later.
- Isolating the low-level kernels prevents re-burying the math in one large driver.

### 2. Real stateless `solve_contact_inner(...)`

Replace the current placeholder solver with a real stateless driver in `src/physics/contact/contact_inner_solver.py`.

Target P0 input signature:

```python
solve_contact_inner(
    g_n,
    ds_t,
    normals,
    t1,
    t2,
    *,
    mu,
    eps_n,
    k_t,
    init_state=None,
    tol_n=...,
    tol_t=...,
    max_inner_iters=...,
    damping=...,
)
```

Key rule:

- `warm start` can only come from `init_state`
- no hidden mutable module-level or object-level solver state

### 3. Numerical flow

For P0, use the most stable simple scheme rather than the most ambitious one.

Per iteration:

1. Start from `lambda_n`, `lambda_t` from `init_state` or zeros.
2. Compute smooth normal residual with Fischer-Burmeister:
   - residual: `sqrt(g_n^2 + lambda_n^2 + eps_n^2) - g_n - lambda_n`
   - jacobian w.r.t. `lambda_n`: `lambda_n / sqrt(...) - 1`
3. Take a damped Newton step on `lambda_n`.
4. Clamp `lambda_n >= 0`.
5. Update tangential variable with projection fixed-point:
   - `tau_trial = lambda_t + k_t * ds_t`
   - radius = `mu * lambda_n`
   - `lambda_t_next = project_to_coulomb_disk(tau_trial, radius, eps)`
6. Check:
   - normal residual norm
   - tangential fixed-point residual norm
   - cone violation
   - penetration metric
7. Stop when all convergence checks pass or `max_inner_iters` is reached.

This is enough for P0:

- smooth normal complementarity
- projection-based tangential update
- warm start
- convergence checks

Quasi-Newton for tangential updates can be added later without changing the P0 public interface.

## Result Object

Keep the current dataclass pattern but extend the returned diagnostics so P0 users can distinguish solver success from fallback.

Required observable fields:

- `state.lambda_n`
- `state.lambda_t`
- `state.converged`
- `state.iters`
- `state.res_norm`
- `state.fallback_used`
- `traction_vec`
- `traction_tangent`
- `diagnostics["fn_norm"]`
- `diagnostics["ft_norm"]`
- `diagnostics["cone_violation"]`
- `diagnostics["max_penetration"]`

If useful, add explicit fields to `ContactInnerResult` for `cone_violation` and `max_penetration`, but the minimum requirement is that they are always available from diagnostics.

## Fallback Policy

P0 needs explicit fallback, not silent solver failure.

Fallback order:

1. If the final iterate is feasible enough, return it even if strict tolerances were not fully met, with `converged=False`.
2. If not, and `init_state` is feasible, reuse `init_state` and mark `fallback_used=True`.
3. If not, apply a smooth tangential fallback:
   - keep the best nonnegative `lambda_n`
   - project tangential traction to the Coulomb disk
4. If even that is numerically invalid, return a detach/skip marker in diagnostics so the outer mixed path can log and optionally exclude the batch contribution.

P0 only requires forward execution and observability. It is acceptable for the first implementation to expose the skip marker in diagnostics before a wider trainer-level policy is added.

## Mixed Bilevel Wiring

P0 wiring target is narrow on purpose.

### What changes

- The mixed bilevel / traction matching path should call the new `solve_contact_inner(...)` with geometric inputs and `init_state`.
- The outer mixed residual should consume `inner_result.traction_vec`, not legacy ALM multiplier state.
- The strict bilevel path should expose inner diagnostics directly for logging and debugging.

### What does not change yet

- `ContactOperator.residual()`
- `ContactOperator.update_multipliers()`
- legacy soft-contact / ALM outer semantics

If the strict mixed path cannot be wired without fully rewriting legacy `ContactOperator`, that means the P0 boundary has been violated and the change should be rolled back to keep the new path isolated.

## `ContactOperator` Boundary for P0

Only minimal hooks are allowed in this pass:

- optional helper methods for extracting geometry already owned by `ContactOperator`
- optional adapter methods for building `g_n`, `ds_t`, and warm-start state for the mixed path

The legacy operator must remain a stable baseline while the new strict bilevel route is validated.

## Trainer and Loss Semantics

P0 should keep the mixed bilevel semantics clear:

- outer loss remains mixed residual and data driven
- contact satisfaction in strict bilevel mode comes from inner solve plus traction matching
- any temporary inner residual penalty kept for debugging must be labeled as monitoring or numerical regularization, not the main enforcement term

The first code wiring pass should prefer dedicated mixed-bilevel codepaths over routing everything back through the old monolithic `TotalEnergy.energy(...)` semantics.

## Tests

P0 requires new tests before implementation changes.

### Kernel primitive tests

Add focused tests for:

- FB normal residual values
- FB normal jacobian values
- projection onto Coulomb disk
- traction composition
- feasibility checks

### Inner solver tests

Add tests showing:

- solve from geometric inputs rather than precomputed lambdas
- converged solve for simple easy contact cases
- explicit fallback when convergence is forced to fail or tolerances are impossible
- diagnostics expose `fn_norm`, `ft_norm`, `cone_violation`, `max_penetration`, `iters`, `fallback_used`

### Mixed path tests

Add tests showing:

- mixed traction matching consumes an inner-solved traction result
- the strict mixed path does not depend on legacy multiplier updates
- forward-only mode still works after the new solver is wired in

## Rollout Order

1. Add primitive tests.
2. Implement primitive helpers.
3. Add failing inner-solver tests for geometry-driven inputs.
4. Replace placeholder `solve_contact_inner(...)` with the real stateless solve.
5. Add or update mixed-path tests.
6. Wire only the mixed bilevel / traction matching path to the new solver.
7. Expose convergence and fallback diagnostics in trainer stats.

## Risks

### Risk: implicit assumptions about sign conventions

Normal gap sign, tangential slip orientation, and Coulomb radius calculation must be tested with very small deterministic cases. P0 should avoid silent sign conventions inherited from legacy ALM code.

### Risk: training path is not actually using the strict route

Current code suggests `solve_inner_state()` is not yet a real training-path dependency. Before or during wiring, confirm the exact mixed path entry point so the new solver does not remain dead code.

### Risk: fallback hides divergence

Fallback is needed, but it must be observable. Trainer logging should expose convergence rate and fallback rate so strict bilevel failures are visible immediately.

## Success Criteria

P0 is complete when:

- `solve_contact_inner(...)` solves from geometry/slip inputs instead of accepting precomputed lambdas
- kernel primitives exist and are unit-tested
- mixed bilevel / traction matching consumes inner-solved traction
- legacy ALM path still behaves as before
- solver diagnostics make convergence and fallback visible
