# Two-Stage Scale Stability Design

## Problem

The latest supervised run `run-20260313-143718` confirms that the previous scale-stability patch changed the intended knobs, but it did not remove the core failure mode.

Observed evidence from the current run:

- The run used `config.yaml`, not a stale alternate config.
- Adaptive focus already excludes `E_data`:
  - log shows `Adaptive focus skips supervision term(s): w_data -> E_data`
- Validation quality is still best very early and degrades later:
  - `step 100`: `vdr=6.9775`, `vrat=6.4982`
  - `step 1500`: `vdr=12.240`, `vrat=11.854`
- The exported `best` checkpoint is still materially better than the exported `final` checkpoint:
  - val ratio median: about `6.50` for `best`, about `11.85` for `final`
  - test ratio median: about `6.63` for `best`, about `11.23` for `final`
- `dsmf=1` remains active through most of the run, so supervision still needs contribution-floor rescue instead of naturally dominating scale.

This means the main problem is no longer "adaptive focus keeps moving `w_data`." The current problem is that late training still lets physics-side optimization pull the model away from the supervised scale that was already aligned earlier.

## Goal

Preserve the early-run supervised scale alignment while still allowing a limited amount of later physics refinement.

The next change should:

- keep the current relative `E_data` normalization
- keep the current `E_smooth` definition and weight
- keep `E_data` out of adaptive focus terms
- avoid another late-run drift from a good early checkpoint to a worse final checkpoint
- require only a small extension to the current trainer entrypoint and config surface

## Chosen Scope

Implement a two-stage training workflow with an explicit phase boundary.

The workflow is:

1. run a supervision-dominant Phase 1 from scratch
2. take the Phase 1 best checkpoint, not the Phase 1 final checkpoint
3. resume from that best checkpoint into a shorter, lower-LR Phase 2
4. keep supervision anchors active during Phase 2 so physics refinement cannot freely rewrite global scale

Do not change:

- supervision dataset loading
- `E_data` loss definition
- `E_smooth` loss definition
- network architecture
- output scaling semantics
- staged preload / incremental route
- visualization/export pipeline

## Design

### 1. Use External Two-Stage Orchestration

Two-stage control should live in the outer training entrypoint, not inside `Trainer.run()` as a complex phase state machine.

Recommended structure:

- `main new.py` detects `two_stage_training.enabled`
- it prepares a base `TrainerConfig`
- it derives Phase 1 and Phase 2 configs from that base config via small override blocks
- it runs Phase 1 as a normal trainer run
- it reads the Phase 1 best checkpoint path
- it runs Phase 2 as a second normal trainer run resumed from that checkpoint

Why this boundary:

- smallest change to the existing trainer
- easiest to debug because each phase has its own config and checkpoint directory
- avoids mixing per-phase control flow into the already busy inner training loop

### 2. Phase 1: Supervision-Dominant Warm Start

Phase 1 exists to lock amplitude and gross field shape early.

It is not pure supervision-only training. Physics terms stay present, but they should be weakened enough that supervised displacement remains the dominant scale anchor.

Phase 1 policy:

- start from random initialization
- keep `E_data` non-adaptive
- keep the supervision contribution floor active
- use a stronger supervision-dominant setting than the current single-stage route
- save best checkpoint using validation supervision quality, not training loss
- stop once validation improvement plateaus instead of pushing through a long late phase

Expected behavior:

- `vrat` and `vdr` should drop quickly
- the best checkpoint should be accepted as the main output of Phase 1
- Phase 1 final checkpoint is only a handoff artifact, not the chosen model

### 3. Phase 2: Physics-Lite Refinement

Phase 2 exists only to recover some physics consistency without destroying the amplitude anchor established in Phase 1.

Phase 2 policy:

- restore from the Phase 1 best checkpoint
- run with a clearly lower learning rate than Phase 1
- allow physics weights to rise relative to Phase 1, but not back to the current aggressive single-stage regime
- keep `E_data`, `E_smooth`, and the supervision contribution floor active
- continue selecting best checkpoint by validation supervision metric
- stop early when validation supervision metrics degrade

Expected behavior:

- Phase 2 may improve contact/stress quality modestly
- Phase 2 must not materially worsen `vrat` / `vdr` versus the Phase 1 best checkpoint
- late-run drift should be bounded by design, not merely hoped away with another scalar weight tweak

### 4. Config Surface

Add a compact config block such as:

- `two_stage_training.enabled`
- `two_stage_training.phase1.*`
- `two_stage_training.phase2.*`

Each phase block is an override layer on top of the already parsed base config.

Phase overrides should be limited to training-control knobs:

- `max_steps`
- `learning_rate`
- validation cadence / plateau settings / early-exit settings
- `save_best_on`
- selected base loss weights
- `supervision_contribution_floor_ratio`

Do not create per-phase copies of unrelated model, dataset, or visualization settings.

### 5. Checkpoint And Artifact Layout

Each phase should have its own output directories to preserve evidence and avoid ambiguity.

Recommended layout:

- `checkpoints/.../phase1`
- `checkpoints/.../phase2`
- `results/.../phase1/...`
- `results/.../phase2/...`

Phase 2 should explicitly resume from the saved Phase 1 best checkpoint path.

This separation is important because the current bug is about late-stage degradation. We need the phase boundary to remain inspectable after the run.

### 6. Logging Expectations

No new loss family is required, but logs should clearly state:

- whether two-stage mode is enabled
- Phase 1 and Phase 2 boundaries
- the checkpoint selected as the Phase 1 handoff
- the Phase 2 resume checkpoint path

Existing metrics remain the primary judgment signals:

- `drrms`
- `vdr`
- `vrat`
- `dsmw`
- `dsmf`

## Alternatives Considered

### Single-Run Internal Phase Switch

Rejected as the default approach because it hides the phase boundary inside one long run. That makes it harder to tell whether failure comes from Phase 1 underfitting or Phase 2 drift.

### Phase 1 Pure Supervision, Phase 2 Full Physics

Rejected because it is too likely to recreate the current problem at the start of Phase 2. The second phase would again be free to rewrite the global scale.

### Phase 2 Parameter Freezing

Rejected for the first version because it is more invasive and harder to tune. It remains a follow-up option if the lighter two-stage route still drifts.

## Risks

- If Phase 1 weakens physics too much, Phase 2 may need to repair too much structure.
- If Phase 2 restores physics too aggressively, the current late-run scale drift will return in a new form.
- If the best checkpoint still appears very early inside Phase 1, that implies the problem is not only stage ordering; additional early-stop tightening or stronger Phase 2 constraints may be required.

## Success Criteria

This change is successful only if all of the following hold after retraining:

- Phase 1 best materially improves `vrat` / `vdr` over the current single-stage final checkpoint
- Phase 2 best does not materially degrade `vrat` / `vdr` versus Phase 1 best
- Phase 2 final stays much closer to Phase 2 best than the current single-stage `final` stays to `best`
- same-pipeline PINN/FEM comparisons no longer show the large late-run amplitude oversizing
- speckle does not regress materially versus the current anti-speckle baseline

## Non-Goals

- No supervision dataset redesign
- No new smoothing term
- No architecture change
- No batching redesign
- No visualization-only masking
