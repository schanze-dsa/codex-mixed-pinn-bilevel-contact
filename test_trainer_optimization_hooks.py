#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for trainer-side optimization hooks."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from train.trainer import Trainer, TrainerConfig, _SavedModelModule
from model.pinn_model import ModelConfig, FieldConfig, EncoderConfig, create_displacement_model
from physics.contact.contact_normal_alm import NormalContactALM
from physics.contact.contact_friction_alm import FrictionContactALM


class _OptWithAggregateArg:
    def apply_gradients(self, grads_and_vars, experimental_aggregate_gradients=True):
        del grads_and_vars, experimental_aggregate_gradients
        return None


class _OptNoAggregateArg:
    def apply_gradients(self, grads_and_vars):
        del grads_and_vars
        return None


class TrainerOptimizationHookTests(unittest.TestCase):
    def test_trainer_includes_init_mixin(self):
        from train.trainer_init_mixin import TrainerInitMixin

        self.assertTrue(issubclass(Trainer, TrainerInitMixin))

    def test_trainer_reexports_config_from_config_module(self):
        from train.trainer_config import TrainerConfig as TrainerConfigFromModule

        self.assertIs(TrainerConfig, TrainerConfigFromModule)

    def test_trainer_includes_opt_mixin(self):
        from train.trainer_opt_mixin import TrainerOptMixin

        self.assertTrue(issubclass(Trainer, TrainerOptMixin))

    def test_trainer_includes_monitor_mixin(self):
        from train.trainer_monitor_mixin import TrainerMonitorMixin

        self.assertTrue(issubclass(Trainer, TrainerMonitorMixin))

    def test_trainer_includes_preload_mixin(self):
        from train.trainer_preload_mixin import TrainerPreloadMixin

        self.assertTrue(issubclass(Trainer, TrainerPreloadMixin))

    def test_train_step_always_uses_incremental_path(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(incremental_mode=False)
        preload_case = {"P": np.asarray([1.0, 2.0, 3.0], dtype=np.float32)}

        called = {"count": 0}

        def _fake_incremental(total, case, *, step=None):
            called["count"] += 1
            self.assertEqual(total, "total")
            self.assertIs(case, preload_case)
            self.assertEqual(step, 7)
            return "ok"

        trainer._train_step_incremental = _fake_incremental  # type: ignore[method-assign]
        out = trainer._train_step(total="total", preload_case=preload_case, step=7)

        self.assertEqual(out, "ok")
        self.assertEqual(called["count"], 1)

    def test_validate_locked_route_rejects_conflicting_flags(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(
            preload_use_stages=False,
            incremental_mode=False,
            stage_resample_contact=True,
            resample_contact_every=10,
            contact_rar_enabled=True,
            volume_rar_enabled=True,
            lbfgs_enabled=True,
            friction_smooth_schedule=True,
            viz_compare_cases=True,
        )

        with self.assertRaises(ValueError):
            trainer._validate_locked_route()  # type: ignore[attr-defined]

    def test_savedmodel_module_run_disables_autograph(self):
        cfg = ModelConfig(
            encoder=EncoderConfig(in_dim=3, out_dim=8, width=8, depth=1),
            field=FieldConfig(cond_dim=8, width=16, depth=2, out_dim=3),
            preload_shift=0.0,
            preload_scale=1.0,
            mixed_precision=None,
        )
        model = create_displacement_model(cfg)
        module = _SavedModelModule(
            model=model,
            use_stages=True,
            append_release_stage=True,
            shift=0.0,
            scale=1.0,
            n_bolts=3,
        )

        self.assertFalse(module.run._autograph)

    def test_savedmodel_stage_features_match_trainer_staged_encoding(self):
        # Build a tiny model/module for SavedModel-side param preparation.
        cfg_model = ModelConfig(
            encoder=EncoderConfig(in_dim=16, out_dim=8, width=8, depth=1),
            field=FieldConfig(cond_dim=8, width=16, depth=2, out_dim=3),
            preload_shift=4.0,
            preload_scale=2.0,
            mixed_precision=None,
        )
        model = create_displacement_model(cfg_model)
        module = _SavedModelModule(
            model=model,
            use_stages=True,
            append_release_stage=True,
            shift=4.0,
            scale=2.0,
            n_bolts=3,
        )

        p = tf.constant([2.0, 5.0, 4.0], dtype=tf.float32)
        order = tf.constant([1, 2, 3], dtype=tf.int32)  # 1-based should normalize to 0-based.
        params_sm = module._prepare_params(p, order)

        # Build the same expected final-stage feature vector via trainer path.
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(
            preload_use_stages=True,
            total_cfg=SimpleNamespace(preload_stage_mode="force_then_lock"),
            model_cfg=SimpleNamespace(preload_shift=4.0, preload_scale=2.0),
        )
        case = {
            "P": np.asarray([2.0, 5.0, 4.0], dtype=np.float32),
            "order": np.asarray([0, 1, 2], dtype=np.int32),
        }
        case.update(trainer._build_stage_case(case["P"], case["order"]))
        params_full = trainer._make_preload_params(case)
        params_ref = trainer._extract_final_stage_params(params_full, keep_context=True)

        self.assertEqual(int(params_sm["P_hat"].shape[0]), 16)
        self.assertEqual(int(params_ref["P_hat"].shape[0]), 16)
        np.testing.assert_allclose(
            params_sm["P_hat"].numpy(),
            params_ref["P_hat"].numpy(),
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_resolve_viz_cases_prefers_fixed_cases_by_default(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(viz_use_last_training_case=False)
        trainer._last_preload_case = {"P": np.asarray([9.0, 9.0, 9.0], dtype=np.float32)}
        trainer._fixed_viz_preload_cases = lambda: [  # type: ignore[method-assign]
            {"P": np.asarray([2.0, 2.0, 6.0], dtype=np.float32)}
        ]
        trainer._sample_preload_case = lambda: {"P": np.asarray([1.0, 1.0, 1.0], dtype=np.float32)}  # type: ignore[method-assign]

        cases = trainer._resolve_viz_cases(n_samples=5)

        self.assertEqual(len(cases), 1)
        np.testing.assert_allclose(cases[0]["P"], np.asarray([2.0, 2.0, 6.0], dtype=np.float32))

    def test_resolve_viz_cases_can_use_last_training_case_when_enabled(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(viz_use_last_training_case=True)
        trainer._last_preload_case = {"P": np.asarray([3.0, 4.0, 5.0], dtype=np.float32)}
        trainer._fixed_viz_preload_cases = lambda: [  # type: ignore[method-assign]
            {"P": np.asarray([2.0, 2.0, 6.0], dtype=np.float32)}
        ]
        trainer._sample_preload_case = lambda: {"P": np.asarray([1.0, 1.0, 1.0], dtype=np.float32)}  # type: ignore[method-assign]

        cases = trainer._resolve_viz_cases(n_samples=5)

        self.assertEqual(len(cases), 1)
        np.testing.assert_allclose(cases[0]["P"], np.asarray([3.0, 4.0, 5.0], dtype=np.float32))

    def test_contact_route_update_interval_gate(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(contact_route_update_every=5)

        self.assertTrue(trainer._should_update_contact_route(1))
        self.assertFalse(trainer._should_update_contact_route(2))
        self.assertFalse(trainer._should_update_contact_route(4))
        self.assertTrue(trainer._should_update_contact_route(5))
        self.assertTrue(trainer._should_update_contact_route(10))

    def test_step_scalar_collection_uses_log_and_early_exit_intervals(self):
        trainer = object.__new__(Trainer)
        trainer._tqdm_enabled = True
        trainer.cfg = SimpleNamespace(
            step_bar_enabled=False,
            log_every=50,
            early_exit_enabled=True,
            early_exit_check_every=25,
        )

        self.assertTrue(trainer._should_collect_step_scalars(1))
        self.assertFalse(trainer._should_collect_step_scalars(2))
        self.assertTrue(trainer._should_collect_step_scalars(25))
        self.assertTrue(trainer._should_collect_step_scalars(50))

    def test_detect_apply_gradients_kwargs_for_supported_optimizer(self):
        kwargs = Trainer._compute_apply_gradients_kwargs(_OptWithAggregateArg())
        self.assertEqual(kwargs, {"experimental_aggregate_gradients": False})

    def test_detect_apply_gradients_kwargs_for_plain_optimizer(self):
        kwargs = Trainer._compute_apply_gradients_kwargs(_OptNoAggregateArg())
        self.assertEqual(kwargs, {})

    def test_static_weight_vector_cache_for_non_adaptive_mode(self):
        trainer = object.__new__(Trainer)
        trainer.loss_state = None
        trainer._loss_keys = ["E_int", "E_cn", "E_ct"]
        trainer._base_weights = {"E_int": 1.5, "E_cn": 0.25, "E_ct": 0.0}
        trainer._static_weight_vector = None

        trainer._refresh_static_weight_vector()
        w0 = trainer._build_weight_vector().numpy()
        np.testing.assert_allclose(w0, np.asarray([1.5, 0.25, 0.0], dtype=np.float32), rtol=0.0, atol=0.0)

        # Cache should keep old values until explicitly refreshed.
        trainer._base_weights["E_int"] = 7.0
        w1 = trainer._build_weight_vector().numpy()
        np.testing.assert_allclose(w1, np.asarray([1.5, 0.25, 0.0], dtype=np.float32), rtol=0.0, atol=0.0)

        trainer._refresh_static_weight_vector()
        w2 = trainer._build_weight_vector().numpy()
        np.testing.assert_allclose(w2, np.asarray([7.0, 0.25, 0.0], dtype=np.float32), rtol=0.0, atol=0.0)

    def test_format_energy_summary_is_skipped_when_step_bar_disabled(self):
        trainer = object.__new__(Trainer)
        trainer._tqdm_enabled = True
        trainer.cfg = SimpleNamespace(step_bar_enabled=False)

        called = {"count": 0}

        def _fake_format(parts):
            del parts
            called["count"] += 1
            return "summary"

        trainer._format_energy_summary = _fake_format
        out = trainer._format_energy_summary_if_needed({"E_int": tf.constant(1.0, tf.float32)})
        self.assertEqual(out, "")
        self.assertEqual(called["count"], 0)

    def test_legacy_route_helpers_are_removed(self):
        legacy_methods = [
            "_maybe_update_friction_smoothing",
            "_update_contact_rar_cache",
            "_maybe_apply_contact_rar",
            "_resample_contact",
            "_update_volume_rar_cache",
            "_maybe_apply_volume_rar",
        ]
        for name in legacy_methods:
            self.assertFalse(hasattr(Trainer, name), msg=f"legacy helper should be removed: {name}")

    def test_early_exit_triggers_after_nonfinite_streak(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(
            early_exit_enabled=True,
            early_exit_warmup_steps=0,
            early_exit_nonfinite_patience=3,
            early_exit_divergence_patience=5,
            early_exit_grad_norm_threshold=1.0e6,
            early_exit_pi_ema_rel_increase=0.25,
        )
        trainer._nonfinite_streak = 0
        trainer._diverge_streak = 0
        trainer._best_pi_ema = None
        trainer._pi_ema = 1.0

        r1 = trainer._check_early_exit(step=1, pi_val=float("nan"), grad_val=1.0)
        r2 = trainer._check_early_exit(step=2, pi_val=float("nan"), grad_val=1.0)
        r3 = trainer._check_early_exit(step=3, pi_val=float("nan"), grad_val=1.0)

        self.assertIsNone(r1)
        self.assertIsNone(r2)
        self.assertIsInstance(r3, str)
        self.assertIn("non-finite", r3)

    def test_early_exit_triggers_on_sustained_divergence(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(
            early_exit_enabled=True,
            early_exit_warmup_steps=0,
            early_exit_nonfinite_patience=3,
            early_exit_divergence_patience=2,
            early_exit_grad_norm_threshold=100.0,
            early_exit_pi_ema_rel_increase=0.10,
        )
        trainer._nonfinite_streak = 0
        trainer._diverge_streak = 0
        trainer._best_pi_ema = None

        trainer._pi_ema = 10.0
        r0 = trainer._check_early_exit(step=1, pi_val=10.0, grad_val=20.0)
        self.assertIsNone(r0)

        trainer._pi_ema = 12.0
        r1 = trainer._check_early_exit(step=2, pi_val=12.0, grad_val=150.0)
        self.assertIsNone(r1)

        trainer._pi_ema = 13.5
        r2 = trainer._check_early_exit(step=3, pi_val=13.5, grad_val=180.0)
        self.assertIsInstance(r2, str)
        self.assertIn("divergence", r2)

    def test_contact_residual_route_metric_and_hint_push(self):
        trainer = object.__new__(Trainer)
        trainer._contact_route_ema = None
        trainer._contact_route_ref = None

        s0 = trainer._update_contact_route_metric({"R_contact_comp": tf.constant(10.0, tf.float32)})
        self.assertGreater(s0, 0.9)
        self.assertLess(s0, 1.1)

        s1 = trainer._update_contact_route_metric({"R_contact_comp": tf.constant(20.0, tf.float32)})
        self.assertGreater(s1, 1.0)

        captured = {"v": None}

        class _Field:
            def __init__(self):
                self.cfg = SimpleNamespace(adaptive_depth_route_source="contact_residual")

            def set_contact_residual_hint(self, value):
                if hasattr(value, "numpy"):
                    value = float(value.numpy())
                captured["v"] = float(value)

        trainer.model = SimpleNamespace(field=_Field())
        trainer._push_contact_route_hint()
        self.assertIsNotNone(captured["v"])

    def test_contact_multiplier_updates_are_plain_python_methods(self):
        normal = NormalContactALM()
        friction = FrictionContactALM()

        self.assertFalse(hasattr(normal.update_multipliers, "python_function"))
        self.assertFalse(hasattr(friction.update_multipliers, "python_function"))

    def test_resolve_viz_reference_path_auto_uses_out_dir_3txt(self):
        trainer = object.__new__(Trainer)
        with tempfile.TemporaryDirectory() as td:
            ref_path = os.path.join(td, "3.txt")
            with open(ref_path, "w", encoding="utf-8") as fp:
                fp.write("Node Number\tTotal Deformation (mm)\n")
                fp.write("1\t1.0e-3\n")

            trainer.cfg = SimpleNamespace(
                out_dir=td,
                viz_reference_truth_path="auto",
            )
            resolved = trainer._resolve_viz_reference_path()
            self.assertEqual(os.path.abspath(ref_path), os.path.abspath(str(resolved)))

    def test_write_viz_reference_alignment_filters_non_node_rows(self):
        trainer = object.__new__(Trainer)
        with tempfile.TemporaryDirectory() as td:
            ref_path = os.path.join(td, "3.txt")
            pred_path = os.path.join(td, "deflection_01_123.txt")

            with open(ref_path, "w", encoding="utf-8") as fp:
                fp.write("Node Number\tTotal Deformation (mm)\n")
                fp.write("1\t1.0e-3\n")
                fp.write("11\t9.9e-1\n")  # non-node id for this tiny assembly
                fp.write("3\t3.0e-3\n")

            with open(pred_path, "w", encoding="utf-8") as fp:
                fp.write("# columns: node_id x y z u_x u_y u_z |u| u_plane v_plane\n")
                fp.write("1 0 0 0 0 0 0 2.0e-3 0 0\n")
                fp.write("3 0 0 0 0 0 0 6.0e-3 0 0\n")

            trainer.cfg = SimpleNamespace(
                out_dir=td,
                viz_write_reference_aligned=True,
                viz_reference_truth_path=ref_path,
            )
            trainer.asm = SimpleNamespace(
                nodes={
                    1: (0.0, 0.0, 0.0),
                    2: (1.0, 0.0, 0.0),
                    3: (2.0, 0.0, 0.0),
                },
                parts={},
            )
            trainer._viz_reference_cache_path = None
            trainer._viz_reference_cache = None
            trainer._asm_node_ids = None

            aligned_path = trainer._write_viz_reference_alignment(pred_path)
            self.assertIsNotNone(aligned_path)
            self.assertTrue(os.path.exists(str(aligned_path)))

            rows = []
            with open(str(aligned_path), "r", encoding="utf-8") as fp:
                for raw in fp:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    rows.append(line.split())

            self.assertEqual(len(rows), 2)
            self.assertEqual([int(r[0]) for r in rows], [1, 3])
            # diff = pred - truth
            self.assertAlmostEqual(float(rows[0][3]), 1.0e-3, places=10)
            self.assertAlmostEqual(float(rows[1][3]), 3.0e-3, places=10)

    def test_resolve_stage_plot_indices_can_skip_release_stage(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(viz_skip_release_stage_plot=True)
        preload_case = {
            "stage_last": np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],  # release stage (force_then_lock)
                ],
                dtype=np.float32,
            )
        }
        indices = trainer._resolve_stage_plot_indices(preload_case, 4)
        self.assertEqual(indices, [0, 1, 2])

        trainer.cfg = SimpleNamespace(viz_skip_release_stage_plot=False)
        indices_all = trainer._resolve_stage_plot_indices(preload_case, 4)
        self.assertEqual(indices_all, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
