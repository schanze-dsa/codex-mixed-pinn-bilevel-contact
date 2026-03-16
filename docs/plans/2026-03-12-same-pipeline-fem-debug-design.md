# Same-Pipeline FEM Debug Export Design

## Problem

The current evidence is strong that `supervision_compare_*.png` uses the same render-state helper for PINN and FEM, but the user still sees speckle in `deflection_*` and `deflection_*_s*.png`.

Those stage deflection plots are produced from the PINN path through `mirror_viz.py`, while the FEM/reference side is mostly consumed through separate alignment and comparison utilities. That leaves one remaining doubt:

- are the speckles in stage plots caused by the PINN field itself
- or by a difference between the stage-plot rendering path and the FEM/reference path

## Goal

Add a diagnostic export that renders both PINN and FEM stage fields through the same visualization pipeline so the resulting images can be compared without any postprocessing-path ambiguity.

## Constraints

- Do not change the existing training loop or default visualization outputs.
- Do not change the current `deflection_*.png` outputs used by prior runs.
- Reuse the staged ANSYS CSV truth already loaded for supervision.
- Keep this as a debug export, gated by config.

## Chosen Scope

Implement a new optional export that:

1. Finds the supervision case matching the visualization preload case by preload values and tightening order.
2. Builds a surface-only visualization context from that supervision case.
3. Exports a matched pair of images for PINN and FEM using the same `plot_mirror_deflection_by_name(...)` pipeline and the same surface-node set.
4. Supports final-stage and per-stage outputs.

This export is diagnostic only. It does not replace the existing `deflection_*.png` outputs.

## Design

### Matching A Visualization Case To A Supervision Case

Use the already loaded `self._supervision_dataset` when available.

Build a lookup key from:

- preload vector `P`
- tightening order

Search all splits because the visualization case may correspond to train, val, or test.

This keeps the logic deterministic and avoids reloading the CSV dataset from disk.

### Why A Surface-Only Visualization Context

`mirror_viz.py` evaluates displacements on the assembly or part node set before slicing to the mirror surface. The staged ANSYS supervision CSVs available in this repo contain the mirror-surface field (`5510` nodes), not the full assembly field.

To reuse the same renderer without inventing a second plotting path, create a lightweight visualization assembly whose node set is restricted to the staged supervision node IDs. Then both PINN and FEM can be evaluated on the same surface-node domain and go through the same rigid-removal, refinement, and smoothing code.

### FEM Forward Adapter

Add a small helper that returns a callable `u_fn(X, params=None)` for a staged FEM field.

The adapter should:

- preserve the exact staged displacement vectors from the supervision CSV
- map queried surface coordinates back to the staged displacement rows
- return a TensorFlow tensor so it satisfies the existing `mirror_viz.py` contract

### Output Naming

Keep the exports clearly diagnostic and separate from the existing outputs:

- `deflection_01_231_samepipe_pinn.png`
- `deflection_01_231_samepipe_fem.png`
- `deflection_01_231_s1_samepipe_pinn.png`
- `deflection_01_231_s1_samepipe_fem.png`

Write matching `.txt` data files using the same `plot_mirror_deflection` export path.

### Logging

When the debug export runs, print:

- which supervision case matched
- where the same-pipeline PINN/FEM files were written

When no match is found, print a short skip message and continue normally.

## Rejected Alternatives

### Reuse `supervision_compare_*` only

That already proves same-render behavior for the supervision compare figures, but it does not answer the user's remaining doubt about the `deflection_*` stage plots.

### Force FEM Through The Existing Full-Assembly Stage Plot Path

Not possible with the currently available truth data because the staged ANSYS CSVs do not contain the full assembly field used by the PINN visualization path.

### Change The Default Stage Plot Path

Too risky for a diagnostic task. The current outputs should remain stable.

## Non-Goals

- No training changes
- No loss changes
- No smoothing changes
- No attempt to fix speckle in this patch
