# Final/Best Dual Export Design

## Problem

The current training flow saves both an early best checkpoint and a final checkpoint, but only exports visualization and supervision-eval artifacts from the final in-memory model state.

This makes post-run analysis ambiguous:

- the user cannot directly compare final vs best outputs
- a late-stage regression can overwrite the more useful early-best picture
- the exported figures and CSVs can disagree with the best-validation checkpoint that was actually saved

The latest run shows exactly this failure mode:

- `ckpt-100` is materially better on supervision scale metrics than `ckpt-1500`
- exported figures and CSVs were still generated from `ckpt-1500`

## Goal

After one training run, export two separate result sets:

1. final checkpoint outputs
2. best checkpoint outputs

Both sets should be easy to inspect side-by-side without overwriting each other.

## Scope

Implement dual export for the existing visualization/evaluation artifacts only.

This includes:

- `deflection_*.png/.txt`
- stage comparison outputs
- same-pipeline debug outputs
- supervision evaluation CSVs and plots
- supervision comparison figures

This change does not need to duplicate SavedModel export in the same patch.

## Recommended Output Layout

Keep the existing top-level output directory, but place the two export sets in subdirectories:

- `<out_dir>/final/...`
- `<out_dir>/best/...`

Why this layout:

- no filename collisions
- the directory name itself carries the checkpoint identity
- existing artifact filenames can stay unchanged inside each subdirectory
- comparing two runs remains simple and scriptable

## Design

### 1. Track checkpoint identities explicitly

The trainer should remember:

- the path of the best checkpoint saved during training
- the path of the final checkpoint saved at the end

The best path should be recorded when `_maybe_save_best_checkpoint(...)` succeeds.
The final path should be recorded when the end-of-run save succeeds.

### 2. Split export orchestration from export implementation

The current `_visualize_after_training(...)` mixes two responsibilities:

- orchestrating which model state to export
- writing the actual files for the current model state

Refactor it into:

- one orchestration layer that decides which checkpoint sets to export
- one inner method that writes the current-state outputs into the active `out_dir`

This keeps the existing export logic intact and minimizes risk.

### 3. Restore checkpoints per export target

When dual export is enabled:

1. restore final checkpoint and export into `<out_dir>/final`
2. restore best checkpoint and export into `<out_dir>/best`
3. restore final checkpoint again before returning so downstream post-run behavior remains on final state

Important details:

- if best checkpoint is missing, fall back to final-only export
- if best and final are the same checkpoint, export only once
- the restore helper should reuse the existing checkpoint object and partial-restore compatibility logic

### 4. Make the behavior configurable

Add one output-side config flag:

- `viz_export_final_and_best`

Behavior:

- `false`: preserve current single-export behavior
- `true`: export separate `final` and `best` subdirectories when possible

For the supervised workflow, enable it in `config.yaml`.

### 5. Logging

The run log should say exactly what was exported, for example:

- `[viz] export target=final -> .../results/.../final`
- `[viz] export target=best -> .../results/.../best`

This is necessary because the recent confusion came from not knowing which checkpoint generated the figures.

## Alternatives Considered

### Filename suffixes in one directory

Rejected because it creates too much clutter and still makes bulk inspection awkward.

### Export best only

Rejected because the user explicitly wants to compare final and best.

### Duplicate SavedModel export in the same patch

Deferred. It is not necessary to solve the current result-comparison problem and would increase the implementation surface.

## Risks

- Restoring checkpoints during post-run export could leave the trainer in the wrong state if final is not restored at the end.
- Some helper functions implicitly assume `cfg.out_dir` is the root output directory; using subdirectories may reveal hidden path assumptions.
- Tests must ensure the current single-export behavior remains unchanged when the new flag is off.

## Success Criteria

The patch is successful if:

- one training run produces both `<out_dir>/final` and `<out_dir>/best`
- both contain the expected visualization/eval files
- no artifacts are overwritten
- final and best exports can differ when the checkpoints differ
- existing single-export behavior still works when the new flag is disabled
