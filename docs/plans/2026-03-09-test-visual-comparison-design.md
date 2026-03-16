# Test Visualization Comparison Design (2026-03-09)

## Context
- The current supervised workflow already exports split-level evaluation metrics and heatmaps, and it can render trained PINN deflection maps after training.
- The supervised dataset includes independent `test` cases with staged finite-element mirror displacement targets, so the project already has the ground truth needed for direct visual comparison.
- The current visualization outputs are useful for debugging, but they do not yet provide a final-report artifact that directly shows `PINN vs FEM vs error` on representative held-out cases.

## Goal
- Keep full quantitative evaluation on the held-out `test` split.
- Add a small number of final-report comparison figures that directly compare PINN predictions against FEM mirror deformation.
- Avoid dumping full detailed figures for every `test` case, which would create noisy outputs and encourage over-reading the test set.

## Final Output Strategy
- Continue to export full `test` evaluation artifacts:
  - `supervision_eval_test.csv`
  - `supervision_eval_test_rmse_vec.png`
- Add three representative comparison figures from the `test` split only.
- Keep the existing training-case deflection map behavior for debugging, but do not treat it as the final showcase artifact.

## Representative Case Selection
- Representative figures are chosen from the `test` split.
- Select one case from each `source` category:
  - `boundary`
  - `corner`
  - `interior`
- For each category, rank candidate cases by final-stage `rmse_vec_mm`.
- Choose the case whose final-stage RMSE is closest to the median RMSE of that category.

### Why this rule
- It avoids cherry-picking unusually good or unusually bad cases.
- It produces one stable "typical" example per source category.
- It stays aligned with the dataset design, which already treats `source` as the meaningful stratification key.

## Comparison Figure Design
- Produce one three-panel comparison figure per selected representative case.
- Use only the final preload stage for this figure.

### Panels
1. `PINN` total displacement magnitude `|u_pred|`
2. `FEM` total displacement magnitude `|u_fem|`
3. Error magnitude `|u_pred - u_fem|`

### Plot rules
- The `PINN` and `FEM` panels share the same color scale.
- The error panel uses its own non-negative error color scale.
- Titles include:
  - `case_id`
  - `source`
  - final `stage`
  - `theta`
  - tightening `order`
  - final-stage `rmse_vec_mm`
- File names should be stable and human-readable.

## Data Handling Rules
- Use the staged supervision data already loaded into `_supervision_dataset`.
- Reuse the final-stage prediction path used for quantitative supervision evaluation so that metrics and figures are based on the same predictions.
- Compare predictions and FEM targets on the exact same node ordering from the loaded supervision case.
- Use the case node coordinates for 2D triangulated plotting in the mirror plane.

## Runtime Rules
- Representative comparison export should be controlled by explicit output config flags.
- Full split metrics should still depend on `supervision.eval_splits`.
- Final comparison figures should have their own split target, defaulting to `test`.
- If a requested source category is missing in the chosen split, skip that category with a clear log message instead of failing the whole training run.

## Expected Artifacts
- `supervision_eval_test.csv`
- `supervision_eval_test_rmse_vec.png`
- `supervision_compare_boundary_<case_id>.png`
- `supervision_compare_corner_<case_id>.png`
- `supervision_compare_interior_<case_id>.png`
- `supervision_compare_selected_cases.csv`

## Testing Requirements
- Unit test representative selection:
  - exactly one case per requested source
  - median-like selection rule is respected
- Unit test comparison export:
  - the three-panel figure is written
  - the summary CSV is written
  - the export uses final-stage metrics and case metadata
- Keep existing supervision evaluation export tests green.

## Non-Goals
- Do not generate three-panel figures for every test case.
- Do not generate per-stage comparison figures for the final report flow.
- Do not change the train/val/test split strategy.
- Do not remove the existing single-case post-training deflection export path.
