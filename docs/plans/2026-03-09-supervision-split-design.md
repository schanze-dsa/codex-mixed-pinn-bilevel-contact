# Supervision Split Strategy Design (2026-03-09)

## Context
- ANSYS mirror supervision uses 180 labeled cases, but the independent units are 30 `base_id` groups.
- Each `base_id` owns 6 tightening-order variants, so any split protocol must keep those 6 rows together.
- A single `train/val/test` split is too sensitive to luck at this scale, especially for model selection and ablation work.

## Final Experiment Protocol
- Split by `base_id`.
- The 6 order variants in one `base_id` never cross split boundaries.
- Reserve a fixed final test set of 5 `base_id` groups.
- The fixed test set should target `1 boundary + 1 corner + 3 interior` groups.
- The remaining 25 `base_id` groups form a 5-fold grouped cross-validation pool.
- All model selection, hyperparameter tuning, and ablations happen only inside those 25 groups.
- The fixed test set is used only once for the final reported result.

## Data Rules
- Group key: `base_id`
- Stratify key: `source`
- Seed source: supervision config seed
- Fixed test quotas:
  - `boundary: 1`
  - `corner: 1`
  - `interior: 3`
- CV folds:
  - count: `5`
  - each fold contains exactly `5` `base_id` groups
  - runtime `cv_fold_index` selects which fold is current validation
- Runtime split materialization:
  - fixed test groups -> `test`
  - current fold groups -> `val`
  - remaining CV groups -> `train`

## Implementation Notes
- Stop using ratio-based split generation.
- Stop using the CSV `split` column as supervision truth.
- Loader should compute a deterministic group protocol from config, then attach computed `train`, `val`, and `test` labels to rows.
- CV fold construction should preserve exact fold size first, then spread `source` labels as evenly as feasible.
- The supervised training config should evaluate `val` during CV runs; `test` should stay reserved for final report workflows.

## Validation
- Unit test that loader works without CSV `split`.
- Unit test that fixed test quotas are met exactly at group level.
- Unit test that remaining groups become 5 folds of exactly 5 groups each.
- Unit test that one chosen `cv_fold_index` materializes `train/val/test = 20/5/5` groups, i.e. `120/30/30` cases.
- Audit the real dataset to confirm no `base_id` leakage and the expected `source` mix in the fixed test set.
