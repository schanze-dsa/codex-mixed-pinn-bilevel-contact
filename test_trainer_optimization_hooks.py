#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for trainer-side optimization hooks."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from train.trainer import Trainer, _SavedModelModule
from model.pinn_model import ModelConfig, FieldConfig, EncoderConfig, create_displacement_model


class _OptWithAggregateArg:
    def apply_gradients(self, grads_and_vars, experimental_aggregate_gradients=True):
        del grads_and_vars, experimental_aggregate_gradients
        return None


class _OptNoAggregateArg:
    def apply_gradients(self, grads_and_vars):
        del grads_and_vars
        return None


class TrainerOptimizationHookTests(unittest.TestCase):
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

    def test_volume_sampling_falls_back_to_uniform_before_rar_cache_ready(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(
            volume_rar_enabled=True,
            volume_rar_fraction=0.5,
            volume_rar_uniform_ratio=0.2,
            volume_rar_temperature=1.0,
            volume_rar_floor=1.0e-8,
            seed=123,
        )
        trainer.elasticity = SimpleNamespace(
            n_cells=1000,
            cfg=SimpleNamespace(n_points_per_step=64),
        )
        trainer._volume_rar_cache = None

        indices, note = trainer._maybe_apply_volume_rar(step_index=1)

        self.assertIsNotNone(indices)
        self.assertEqual(len(indices), 64)
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < 1000))
        self.assertGreater(len(np.unique(indices)), 32)
        self.assertIn("64", note)

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


if __name__ == "__main__":
    unittest.main()
