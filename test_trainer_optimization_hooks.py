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
from model.loss_energy import TotalEnergy
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

    def test_build_stage_case_can_disable_release_stage(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(
            preload_append_release_stage=False,
            total_cfg=SimpleNamespace(preload_stage_mode="force_then_lock"),
        )

        case = trainer._build_stage_case(
            np.asarray([2.0, 5.0, 4.0], dtype=np.float32),
            np.asarray([0, 1, 2], dtype=np.int32),
        )

        self.assertEqual(case["stages"].shape, (3, 3))
        np.testing.assert_allclose(
            case["stages"][-1],
            np.asarray([2.0, 5.0, 4.0], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )

    def test_make_preload_params_carries_stage_supervision_tensors(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(
            preload_use_stages=True,
            preload_append_release_stage=False,
            total_cfg=SimpleNamespace(preload_stage_mode="force_then_lock"),
            model_cfg=SimpleNamespace(preload_shift=0.0, preload_scale=1.0),
        )
        case = {
            "P": np.asarray([2.0, 5.0, 4.0], dtype=np.float32),
            "order": np.asarray([0, 1, 2], dtype=np.int32),
            "X_obs": np.asarray(
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
                    [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
                ],
                dtype=np.float32,
            ),
            "U_obs": np.asarray(
                [
                    [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
                    [[0.3, 0.0, 0.0], [0.4, 0.0, 0.0]],
                    [[0.5, 0.0, 0.0], [0.6, 0.0, 0.0]],
                ],
                dtype=np.float32,
            ),
        }
        case.update(trainer._build_stage_case(case["P"], case["order"]))

        params_full = trainer._make_preload_params(case)
        stage2 = trainer._extract_stage_params(params_full, 1, keep_context=True)

        self.assertIn("X_obs", params_full["stages"])
        self.assertIn("U_obs", params_full["stages"])
        self.assertEqual(tuple(params_full["stages"]["X_obs"].shape), (3, 2, 3))
        self.assertEqual(tuple(stage2["X_obs"].shape), (2, 3))
        np.testing.assert_allclose(
            stage2["U_obs"].numpy(),
            np.asarray([[0.3, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )

    def test_sample_preload_case_prefers_supervision_dataset(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(
            preload_use_stages=True,
            preload_append_release_stage=False,
            total_cfg=SimpleNamespace(preload_stage_mode="force_then_lock"),
        )

        class _Dataset:
            def __init__(self):
                self.calls = []

            def next_case(self, split="train"):
                self.calls.append(split)
                return {
                    "P": np.asarray([6.0, 2.0, 4.0], dtype=np.float32),
                    "order": np.asarray([2, 0, 1], dtype=np.int32),
                    "X_obs": np.asarray(
                        [
                            [[0.0, 0.0, 0.0]],
                            [[0.0, 0.0, 0.0]],
                            [[0.0, 0.0, 0.0]],
                        ],
                        dtype=np.float32,
                    ),
                    "U_obs": np.asarray(
                        [
                            [[0.1, 0.0, 0.0]],
                            [[0.2, 0.0, 0.0]],
                            [[0.3, 0.0, 0.0]],
                        ],
                        dtype=np.float32,
                    ),
                }

        trainer._supervision_dataset = _Dataset()

        case = trainer._sample_preload_case()

        self.assertEqual(trainer._supervision_dataset.calls, ["train"])
        self.assertIn("stages", case)
        np.testing.assert_allclose(
            case["stages"],
            np.asarray(
                [
                    [0.0, 0.0, 4.0],
                    [6.0, 0.0, 4.0],
                    [6.0, 2.0, 4.0],
                ],
                dtype=np.float32,
            ),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            case["U_obs"][-1],
            np.asarray([[0.3, 0.0, 0.0]], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )

    def test_total_energy_data_loss_tracks_exact_and_inexact_observations(self):
        cfg = SimpleNamespace(
            w_int=0.0,
            w_cn=0.0,
            w_ct=0.0,
            w_bc=0.0,
            w_tight=0.0,
            w_sigma=0.0,
            w_eq=0.0,
            w_reg=0.0,
            w_bi=0.0,
            w_ed=0.0,
            w_unc=0.0,
            w_data=1.0,
            sigma_ref=1.0,
            path_penalty_weight=0.0,
            fric_path_penalty_weight=0.0,
            ed_enabled=False,
            ed_external_scale=1.0,
            ed_margin=0.0,
            ed_use_relu=True,
            ed_square=True,
            adaptive_scheme="contact_only",
            update_every_steps=1,
            dtype="float32",
        )
        total = TotalEnergy(cfg)
        total.attach()

        X = tf.constant([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=tf.float32)
        U = tf.constant([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=tf.float32)

        def exact_u(_x, params=None):
            del _x
            return tf.cast(params["U_obs"], tf.float32)

        Pi0, parts0, _ = total.energy(exact_u, params={"X_obs": X, "U_obs": U})
        self.assertAlmostEqual(float(parts0["E_data"].numpy()), 0.0, places=7)
        self.assertAlmostEqual(float(Pi0.numpy()), 0.0, places=7)

        def zero_u(x, params=None):
            del x, params
            return tf.zeros_like(U)

        Pi1, parts1, _ = total.energy(zero_u, params={"X_obs": X, "U_obs": U})
        self.assertGreater(float(parts1["E_data"].numpy()), 0.0)
        self.assertGreater(float(Pi1.numpy()), 0.0)

    def test_total_energy_data_loss_is_relative_to_observation_scale(self):
        cfg = SimpleNamespace(
            w_int=0.0,
            w_cn=0.0,
            w_ct=0.0,
            w_bc=0.0,
            w_tight=0.0,
            w_sigma=0.0,
            w_eq=0.0,
            w_reg=0.0,
            w_bi=0.0,
            w_ed=0.0,
            w_unc=0.0,
            w_data=1.0,
            sigma_ref=1.0,
            path_penalty_weight=0.0,
            fric_path_penalty_weight=0.0,
            ed_enabled=False,
            ed_external_scale=1.0,
            ed_margin=0.0,
            ed_use_relu=True,
            ed_square=True,
            adaptive_scheme="contact_only",
            update_every_steps=1,
            dtype="float32",
        )
        total = TotalEnergy(cfg)
        total.attach()

        X = tf.constant([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=tf.float32)
        u_small = tf.constant([[1.0e-3, 0.0, 0.0], [2.0e-3, 0.0, 0.0]], dtype=tf.float32)
        u_large = tf.constant([[1.0e-2, 0.0, 0.0], [2.0e-2, 0.0, 0.0]], dtype=tf.float32)

        def zero_u(x, params=None):
            del x, params
            return tf.zeros((2, 3), dtype=tf.float32)

        _, parts_small, _ = total.energy(zero_u, params={"X_obs": X, "U_obs": u_small})
        _, parts_large, _ = total.energy(zero_u, params={"X_obs": X, "U_obs": u_large})

        self.assertAlmostEqual(float(parts_small["E_data"].numpy()), 1.0, places=6)
        self.assertAlmostEqual(
            float(parts_small["E_data"].numpy()),
            float(parts_large["E_data"].numpy()),
            places=6,
        )

    def test_data_smoothing_loss_penalizes_high_frequency_supervision_noise(self):
        cfg = SimpleNamespace(
            w_int=0.0,
            w_cn=0.0,
            w_ct=0.0,
            w_bc=0.0,
            w_tight=0.0,
            w_sigma=0.0,
            w_eq=0.0,
            w_reg=0.0,
            w_bi=0.0,
            w_ed=0.0,
            w_unc=0.0,
            w_data=0.0,
            w_smooth=1.0,
            sigma_ref=1.0,
            path_penalty_weight=0.0,
            fric_path_penalty_weight=0.0,
            ed_enabled=False,
            ed_external_scale=1.0,
            ed_margin=0.0,
            ed_use_relu=True,
            ed_square=True,
            adaptive_scheme="contact_only",
            update_every_steps=1,
            dtype="float32",
            data_smoothing_k=2,
        )
        total = TotalEnergy(cfg)
        total.attach()

        X = tf.constant(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=tf.float32,
        )
        U_obs = tf.constant(
            [
                [1.0e-3, 0.0, 0.0],
                [1.0e-3, 0.0, 0.0],
                [1.0e-3, 0.0, 0.0],
                [1.0e-3, 0.0, 0.0],
            ],
            dtype=tf.float32,
        )

        def smooth_u(x, params=None):
            del x, params
            return tf.identity(U_obs)

        def noisy_u(x, params=None):
            del x, params
            return tf.constant(
                [
                    [2.0e-3, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [2.0e-3, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=tf.float32,
            )

        _, parts_smooth, stats_smooth = total.energy(smooth_u, params={"X_obs": X, "U_obs": U_obs})
        _, parts_noisy, stats_noisy = total.energy(noisy_u, params={"X_obs": X, "U_obs": U_obs})

        self.assertAlmostEqual(float(parts_smooth["E_smooth"].numpy()), 0.0, places=7)
        self.assertGreater(float(parts_noisy["E_smooth"].numpy()), 0.0)
        self.assertGreater(
            float(parts_noisy["E_smooth"].numpy()),
            float(parts_smooth["E_smooth"].numpy()),
        )
        self.assertGreater(float(stats_noisy["data_smooth_rms"].numpy()), 0.0)

    def test_weight_vector_enforces_supervision_contribution_floor(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(
            supervision_contribution_floor_enabled=True,
            supervision_contribution_floor_ratio=0.1,
        )
        trainer._loss_keys = ["E_sigma", "E_ct", "E_data"]

        weights = tf.constant([0.5, 1.0, 1.0], dtype=tf.float32)
        parts = {
            "E_sigma": tf.constant(100.0, dtype=tf.float32),
            "E_ct": tf.constant(20.0, dtype=tf.float32),
            "E_data": tf.constant(1.0, dtype=tf.float32),
        }

        adjusted, diag = trainer._apply_supervision_contribution_floor(parts, weights)

        adjusted_np = adjusted.numpy()
        self.assertAlmostEqual(adjusted_np[0], 0.5, places=6)
        self.assertAlmostEqual(adjusted_np[1], 1.0, places=6)
        self.assertAlmostEqual(adjusted_np[2], 7.0, places=6)
        self.assertAlmostEqual(float(diag["data_eff_w"].numpy()), 7.0, places=6)
        self.assertAlmostEqual(float(diag["data_floor_target"].numpy()), 7.0, places=6)
        self.assertEqual(float(diag["data_floor_active"].numpy()), 1.0)

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

    def test_supervision_load_splits_includes_compare_split_when_enabled(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(
            supervision=SimpleNamespace(
                train_splits=("train",),
                eval_splits=("val",),
            ),
            viz_supervision_compare_enabled=True,
            viz_supervision_compare_split="test",
        )

        splits = trainer._supervision_load_splits()

        self.assertEqual(splits, ("train", "val", "test"))

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

    def test_strict_mixed_weight_profile_zeros_legacy_outer_terms(self):
        trainer = object.__new__(Trainer)
        trainer.loss_state = None
        trainer._loss_keys = ["E_int", "E_cn", "E_ct", "E_sigma", "E_eq", "E_bi", "E_data", "E_ed"]
        trainer._base_weights = {
            "E_int": 2.0,
            "E_cn": 3.0,
            "E_ct": 4.0,
            "E_sigma": 5.0,
            "E_eq": 6.0,
            "E_bi": 7.0,
            "E_data": 8.0,
            "E_ed": 9.0,
        }
        trainer._static_weight_vector = None
        trainer._active_weight_overrides = {}
        trainer._mixed_phase_flags = {
            "phase_name": "phase2a",
            "normal_ift_enabled": False,
            "tangential_ift_enabled": False,
            "detach_inner_solution": True,
        }

        route_mode = trainer._resolve_bilevel_objective_route()
        trainer._apply_route_weight_overrides(route_mode)
        weights = trainer._build_weight_vector().numpy()

        np.testing.assert_allclose(
            weights,
            np.asarray([0.0, 3.0, 4.0, 0.0, 6.0, 0.0, 8.0, 0.0], dtype=np.float32),
            rtol=0.0,
            atol=0.0,
        )

    def test_strict_mixed_route_rejects_tangential_ift_in_p0(self):
        trainer = object.__new__(Trainer)
        trainer._mixed_phase_flags = {
            "phase_name": "phase2b",
            "normal_ift_enabled": True,
            "tangential_ift_enabled": True,
            "detach_inner_solution": True,
        }

        with self.assertRaises(NotImplementedError):
            trainer._resolve_bilevel_objective_route()

    def test_evaluate_total_objective_dispatches_to_strict_mixed_path(self):
        trainer = object.__new__(Trainer)
        trainer.model = SimpleNamespace(u_fn=lambda X, params=None: X)
        trainer.loss_state = None
        trainer._base_weights = {}
        trainer._loss_keys = []
        trainer._static_weight_vector = None
        trainer._active_weight_overrides = {}
        trainer._mixed_phase_flags = {
            "phase_name": "phase2a",
            "normal_ift_enabled": True,
            "tangential_ift_enabled": False,
            "detach_inner_solution": True,
        }

        calls = {"energy": 0, "strict": 0}

        class _FakeTotal:
            def energy(self, *args, **kwargs):
                del args, kwargs
                calls["energy"] += 1
                raise AssertionError("legacy total.energy() should not be used in strict mixed route")

            def strict_mixed_objective(self, *args, **kwargs):
                del args, kwargs
                calls["strict"] += 1
                return (
                    tf.constant(0.0, dtype=tf.float32),
                    {"E_cn": tf.constant(1.0, dtype=tf.float32)},
                    {},
                )

        _, parts, stats = trainer._evaluate_total_objective(_FakeTotal(), params={}, stress_fn=None, tape=None)

        self.assertEqual(calls["energy"], 0)
        self.assertEqual(calls["strict"], 1)
        self.assertAlmostEqual(float(parts["E_cn"].numpy()), 1.0)
        self.assertEqual(stats["strict_route_mode"], "normal_ready")

    def test_strict_mixed_objective_consumes_unified_mixed_residual_terms(self):
        cfg = SimpleNamespace(
            w_int=0.0,
            w_cn=0.0,
            w_ct=0.0,
            w_bc=0.0,
            w_tight=0.0,
            w_sigma=0.0,
            w_eq=1.0,
            w_reg=0.0,
            w_bi=0.0,
            w_ed=0.0,
            w_unc=0.0,
            w_data=0.0,
            w_smooth=0.0,
            sigma_ref=1.0,
            path_penalty_weight=0.0,
            fric_path_penalty_weight=0.0,
            ed_enabled=False,
            ed_external_scale=1.0,
            ed_margin=0.0,
            ed_use_relu=True,
            ed_square=True,
            adaptive_scheme="contact_only",
            update_every_steps=1,
            dtype="float32",
        )
        total = TotalEnergy(cfg)

        calls = {"mixed": 0}
        case = self

        class _FakeElasticity:
            cfg = SimpleNamespace(stress_loss_weight=0.0)

            @staticmethod
            def _eval_u_on_nodes(u_fn, params):
                del u_fn, params
                return tf.zeros((0, 3), dtype=tf.float32)

            def mixed_residual_terms(self, u_fn, sigma_fn, params, *, return_cache=False):
                del u_fn, sigma_fn, params
                case.assertTrue(return_cache)
                calls["mixed"] += 1
                r_eq = tf.ones((2, 3), dtype=tf.float32)
                return {
                    "R_const": tf.zeros((2, 6), dtype=tf.float32),
                    "R_eq": r_eq,
                    "cache": {
                        "eps_vec": tf.zeros((2, 6), dtype=tf.float32),
                        "sigma_pred": tf.zeros((2, 6), dtype=tf.float32),
                        "sigma_phys": tf.zeros((2, 6), dtype=tf.float32),
                        "div_sigma": r_eq,
                        "w_sel": tf.ones((2,), dtype=tf.float32),
                    },
                }

        total.attach(elasticity=_FakeElasticity())
        total.set_mixed_bilevel_flags({"phase_name": "phase2a"})

        def zero_u(X, params=None):
            del X, params
            return tf.zeros((2, 3), dtype=tf.float32)

        def zero_us(X, params=None):
            del X, params
            return (
                tf.zeros((2, 3), dtype=tf.float32),
                tf.zeros((2, 6), dtype=tf.float32),
            )

        Pi, parts, _ = total.strict_mixed_objective(zero_u, params={}, stress_fn=zero_us)

        self.assertEqual(calls["mixed"], 1)
        self.assertIn("E_eq", parts)
        self.assertAlmostEqual(float(parts["E_eq"].numpy()), 3.0, places=6)
        self.assertAlmostEqual(float(Pi.numpy()), 3.0, places=6)

    def test_accumulate_strict_bilevel_rates_and_freeze_request(self):
        trainer = object.__new__(Trainer)
        trainer._strict_bilevel_stats = {"total": 0, "converged": 0, "fallback": 0, "skipped": 0}
        trainer._strict_bilevel_freeze_requested = False
        trainer._contact_hardening_frozen = False
        trainer._continuation_freeze_events = 0

        stats1 = trainer._accumulate_strict_bilevel_stats(
            {
                "inner_converged": 1.0,
                "inner_fallback_used": 0.0,
                "inner_skip_batch": 0.0,
            },
            route_mode="forward_only",
        )
        stats2 = trainer._accumulate_strict_bilevel_stats(
            {
                "inner_converged": 0.0,
                "inner_fallback_used": 1.0,
                "inner_skip_batch": 1.0,
            },
            route_mode="normal_ready",
        )

        self.assertAlmostEqual(float(stats1["inner_convergence_rate"]), 1.0)
        self.assertAlmostEqual(float(stats2["inner_convergence_rate"]), 0.5)
        self.assertAlmostEqual(float(stats2["inner_fallback_rate"]), 0.5)
        self.assertAlmostEqual(float(stats2["inner_skip_rate"]), 0.5)
        self.assertEqual(stats2["strict_route_mode"], "normal_ready")
        self.assertTrue(trainer._strict_bilevel_freeze_requested)

    def test_resolve_contact_backend_accepts_explicit_legacy_backend_for_legacy_route(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(contact_backend="legacy_alm")
        trainer._mixed_phase_flags = {
            "phase_name": "phase0",
            "normal_ift_enabled": False,
            "tangential_ift_enabled": False,
            "detach_inner_solution": True,
        }

        self.assertEqual(trainer._resolve_contact_backend(), "legacy_alm")

    def test_resolve_contact_backend_rejects_contradictory_route_backend_pairs(self):
        cases = [
            (
                "phase0",
                "inner_solver",
                {
                    "phase_name": "phase0",
                    "normal_ift_enabled": False,
                    "tangential_ift_enabled": False,
                    "detach_inner_solution": True,
                },
            ),
            (
                "phase2a",
                "legacy_alm",
                {
                    "phase_name": "phase2a",
                    "normal_ift_enabled": False,
                    "tangential_ift_enabled": False,
                    "detach_inner_solution": True,
                },
            ),
        ]

        for phase_name, backend, flags in cases:
            trainer = object.__new__(Trainer)
            trainer.cfg = SimpleNamespace(contact_backend=backend)
            trainer._mixed_phase_flags = flags
            with self.subTest(phase_name=phase_name, backend=backend):
                with self.assertRaises(ValueError):
                    trainer._resolve_contact_backend()

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

    def test_format_energy_summary_includes_supervision_term(self):
        trainer = object.__new__(Trainer)
        trainer.loss_state = None
        trainer.cfg = SimpleNamespace(
            total_cfg=SimpleNamespace(
                w_int=0.0,
                w_cn=0.0,
                w_ct=0.0,
                w_bc=0.0,
                w_tight=0.0,
                w_sigma=0.0,
                w_eq=0.0,
                w_reg=0.0,
                w_bi=0.0,
                w_ed=0.0,
                w_data=2.5,
            ),
            uncertainty_loss_weight=0.0,
        )

        summary = trainer._format_energy_summary({"E_data": tf.constant(0.125, tf.float32)})

        self.assertIn("Edata=1.250000e-01(w=2.5)", summary)

    def test_format_train_log_postfix_includes_supervision_error_metrics(self):
        trainer = object.__new__(Trainer)
        trainer.loss_state = None
        trainer.contact = None
        trainer.cfg = SimpleNamespace(
            total_cfg=SimpleNamespace(
                w_int=0.0,
                w_cn=0.0,
                w_ct=0.0,
                w_bc=0.0,
                w_tight=0.0,
                w_sigma=0.0,
                w_eq=0.0,
                w_reg=0.0,
                w_bi=0.0,
                w_ed=0.0,
                w_data=1.0,
            ),
            uncertainty_loss_weight=0.0,
            tightening_cfg=SimpleNamespace(angle_unit="deg"),
            yield_strength=None,
        )

        postfix, note = trainer._format_train_log_postfix(
            P_np=np.asarray([2.0, 4.0, 6.0], dtype=np.float32),
            Pi=tf.constant(1.0, dtype=tf.float32),
            parts={"E_data": tf.constant(0.5, dtype=tf.float32)},
            stats={
                "data_rms": tf.constant(0.25, dtype=tf.float32),
                "data_mae": tf.constant(0.125, dtype=tf.float32),
            },
            grad_val=0.75,
            rel_pi=0.1,
            rel_delta=None,
            order=np.asarray([0, 1, 2], dtype=np.int32),
        )

        self.assertEqual(note, "已记录")
        self.assertIsNotNone(postfix)
        self.assertIn("drms=2.5000e-01", postfix)
        self.assertIn("dmae=1.2500e-01", postfix)

    def test_format_train_log_postfix_includes_relative_supervision_metrics(self):
        trainer = object.__new__(Trainer)
        trainer.loss_state = None
        trainer.contact = None
        trainer.cfg = SimpleNamespace(
            total_cfg=SimpleNamespace(
                w_int=0.0,
                w_cn=0.0,
                w_ct=0.0,
                w_bc=0.0,
                w_tight=0.0,
                w_sigma=0.0,
                w_eq=0.0,
                w_reg=0.0,
                w_bi=0.0,
                w_ed=0.0,
                w_data=1.0,
            ),
            uncertainty_loss_weight=0.0,
            tightening_cfg=SimpleNamespace(angle_unit="deg"),
            yield_strength=None,
        )

        postfix, _ = trainer._format_train_log_postfix(
            P_np=np.asarray([2.0, 4.0, 6.0], dtype=np.float32),
            Pi=tf.constant(1.0, dtype=tf.float32),
            parts={"E_data": tf.constant(0.5, dtype=tf.float32)},
            stats={
                "data_rms": tf.constant(0.25, dtype=tf.float32),
                "data_mae": tf.constant(0.125, dtype=tf.float32),
                "data_ref_rms": tf.constant(0.05, dtype=tf.float32),
                "data_rel_rms": tf.constant(5.0, dtype=tf.float32),
                "data_rel_mae": tf.constant(2.5, dtype=tf.float32),
            },
            grad_val=0.75,
            rel_pi=0.1,
            rel_delta=None,
            order=np.asarray([0, 1, 2], dtype=np.int32),
        )

        self.assertIsNotNone(postfix)
        self.assertIn("dref=5.0000e-02", postfix)
        self.assertIn("drrms=5.0000e+00", postfix)
        self.assertIn("drmae=2.5000e+00", postfix)

    def test_format_train_log_postfix_includes_validation_metrics_and_lr(self):
        trainer = object.__new__(Trainer)
        trainer.loss_state = None
        trainer.contact = None
        trainer.optimizer = tf.keras.optimizers.Adam(1.0e-4)
        trainer.cfg = SimpleNamespace(
            total_cfg=SimpleNamespace(
                w_int=0.0,
                w_cn=0.0,
                w_ct=0.0,
                w_bc=0.0,
                w_tight=0.0,
                w_sigma=0.0,
                w_eq=0.0,
                w_reg=0.0,
                w_bi=0.0,
                w_ed=0.0,
                w_data=1.0,
            ),
            uncertainty_loss_weight=0.0,
            tightening_cfg=SimpleNamespace(angle_unit="deg"),
            yield_strength=None,
        )

        postfix, _ = trainer._format_train_log_postfix(
            P_np=np.asarray([2.0, 4.0, 6.0], dtype=np.float32),
            Pi=tf.constant(1.0, dtype=tf.float32),
            parts={"E_data": tf.constant(0.5, dtype=tf.float32)},
            stats={
                "data_rms": tf.constant(0.25, dtype=tf.float32),
                "data_mae": tf.constant(0.125, dtype=tf.float32),
            },
            grad_val=0.75,
            rel_pi=0.1,
            rel_delta=None,
            order=np.asarray([0, 1, 2], dtype=np.int32),
            val_summary={
                "val_drrms_mean": 1.25,
                "val_ratio_median": 1.10,
            },
        )

        self.assertIsNotNone(postfix)
        self.assertIn("vdr=1.2500e+00", postfix)
        self.assertIn("vrat=1.1000e+00", postfix)
        self.assertIn("vlr=1.0000e-04", postfix)

    def test_format_train_log_postfix_includes_strict_bilevel_aggregate_metrics(self):
        trainer = object.__new__(Trainer)
        trainer.loss_state = None
        trainer.contact = None
        trainer.optimizer = tf.keras.optimizers.Adam(1.0e-4)
        trainer._mixed_phase_flags = {
            "phase_name": "phase2a",
            "normal_ift_enabled": True,
            "tangential_ift_enabled": False,
            "detach_inner_solution": True,
        }
        trainer.cfg = SimpleNamespace(
            contact_backend="auto",
            total_cfg=SimpleNamespace(
                w_int=0.0,
                w_cn=1.0,
                w_ct=1.0,
                w_bc=0.0,
                w_tight=0.0,
                w_sigma=0.0,
                w_eq=1.0,
                w_reg=1.0,
                w_bi=0.0,
                w_ed=0.0,
                w_data=1.0,
                w_smooth=0.0,
            ),
            uncertainty_loss_weight=0.0,
            tightening_cfg=SimpleNamespace(angle_unit="deg"),
            yield_strength=None,
        )

        postfix, _ = trainer._format_train_log_postfix(
            P_np=np.asarray([2.0, 4.0, 6.0], dtype=np.float32),
            Pi=tf.constant(1.0, dtype=tf.float32),
            parts={
                "E_cn": tf.constant(0.5, dtype=tf.float32),
                "E_ct": tf.constant(0.25, dtype=tf.float32),
                "E_eq": tf.constant(0.125, dtype=tf.float32),
                "E_reg": tf.constant(0.0625, dtype=tf.float32),
                "E_data": tf.constant(0.03125, dtype=tf.float32),
            },
            stats={
                "inner_convergence_rate": 0.75,
                "inner_fallback_rate": 0.25,
                "inner_skip_rate": 0.125,
                "strict_route_mode": "normal_ready",
                "continuation_frozen": 1.0,
                "continuation_freeze_events": 2.0,
            },
            grad_val=0.75,
            rel_pi=0.1,
            rel_delta=None,
            order=np.asarray([0, 1, 2], dtype=np.int32),
        )

        self.assertIsNotNone(postfix)
        self.assertIn("smode=normal_ready", postfix)
        self.assertIn("iconv=7.5000e-01", postfix)
        self.assertIn("ifb=2.5000e-01", postfix)
        self.assertIn("iskip=1.2500e-01", postfix)
        self.assertIn("cback=inner_solver", postfix)
        self.assertIn("cfrz=1", postfix)
        self.assertIn("cfrze=2", postfix)

    def test_format_train_log_postfix_includes_legacy_contact_backend_token(self):
        trainer = object.__new__(Trainer)
        trainer.loss_state = None
        trainer.contact = None
        trainer._mixed_phase_flags = {
            "phase_name": "phase0",
            "normal_ift_enabled": False,
            "tangential_ift_enabled": False,
            "detach_inner_solution": True,
        }
        trainer.cfg = SimpleNamespace(
            contact_backend="auto",
            total_cfg=SimpleNamespace(
                w_int=0.0,
                w_cn=0.0,
                w_ct=0.0,
                w_bc=0.0,
                w_tight=0.0,
                w_sigma=0.0,
                w_eq=0.0,
                w_reg=0.0,
                w_bi=0.0,
                w_ed=0.0,
                w_data=1.0,
                w_smooth=0.0,
            ),
            uncertainty_loss_weight=0.0,
            tightening_cfg=SimpleNamespace(angle_unit="deg"),
            yield_strength=None,
        )

        postfix, _ = trainer._format_train_log_postfix(
            P_np=np.asarray([2.0, 4.0, 6.0], dtype=np.float32),
            Pi=tf.constant(1.0, dtype=tf.float32),
            parts={"E_data": tf.constant(0.5, dtype=tf.float32)},
            stats={"strict_route_mode": "legacy"},
            grad_val=0.75,
            rel_pi=0.1,
            rel_delta=None,
            order=np.asarray([0, 1, 2], dtype=np.int32),
        )

        self.assertIsNotNone(postfix)
        self.assertIn("cback=legacy_alm", postfix)

    def test_supervision_validation_summary_aggregates_rows(self):
        trainer = object.__new__(Trainer)

        summary = trainer._summarize_supervision_eval_rows(
            [
                {
                    "rmse_vec_mm": 0.20,
                    "pred_rms_vec_mm": 0.15,
                    "true_rms_vec_mm": 0.10,
                },
                {
                    "rmse_vec_mm": 0.30,
                    "pred_rms_vec_mm": 0.20,
                    "true_rms_vec_mm": 0.40,
                },
            ]
        )

        self.assertAlmostEqual(summary["val_rmse_vec_mm_mean"], 0.25)
        self.assertAlmostEqual(summary["val_ratio_median"], 1.0)
        self.assertAlmostEqual(summary["val_drrms_mean"], 1.375)
        self.assertEqual(summary["val_rows"], 2)

    def test_best_metric_uses_validation_drrms_when_configured(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(save_best_on="val_drrms")
        trainer.best_metric = 2.0
        saved = []

        def _fake_save(step):
            saved.append(step)
            return f"ckpt-{step}"

        trainer._save_checkpoint_best_effort = _fake_save  # type: ignore[method-assign]

        note1 = trainer._maybe_save_best_checkpoint(
            step=50,
            pi_val=10.0,
            parts={"E_int": tf.constant(7.0, dtype=tf.float32)},
            val_summary={"val_drrms_mean": 1.5},
        )
        note2 = trainer._maybe_save_best_checkpoint(
            step=100,
            pi_val=1.0,
            parts={"E_int": tf.constant(0.1, dtype=tf.float32)},
            val_summary={"val_drrms_mean": 1.8},
        )

        self.assertEqual(saved, [50])
        self.assertAlmostEqual(trainer.best_metric, 1.5)
        self.assertIn("ckpt-50", note1)
        self.assertEqual(note2, "")

    def test_trainer_records_phase_checkpoint_paths_for_two_stage_resume(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(save_best_on="val_drrms", resume_ckpt_path="phase1-best")
        trainer.best_metric = float("inf")
        trainer._best_ckpt_path = None
        trainer._final_ckpt_path = None
        trainer._resumed_ckpt_path = None

        class _Status:
            def __init__(self):
                self.expect_partial_called = False

            def expect_partial(self):
                self.expect_partial_called = True

        class _Checkpoint:
            def __init__(self):
                self.paths = []
                self.status = _Status()

            def restore(self, path):
                self.paths.append(path)
                return self.status

        trainer.ckpt = _Checkpoint()
        trainer._restore_resume_checkpoint_if_needed()

        self.assertEqual(trainer.ckpt.paths, ["phase1-best"])
        self.assertEqual(trainer._resumed_ckpt_path, "phase1-best")
        self.assertTrue(trainer.ckpt.status.expect_partial_called)

        trainer._save_checkpoint_best_effort = lambda step: f"ckpt-{step}"  # type: ignore[method-assign]

        note = trainer._maybe_save_best_checkpoint(
            step=50,
            pi_val=10.0,
            parts={"E_int": tf.constant(7.0, dtype=tf.float32)},
            val_summary={"val_drrms_mean": 1.5},
        )
        final_ckpt = trainer._save_final_checkpoint(150)

        self.assertIn("ckpt-50", note)
        self.assertEqual(trainer._best_ckpt_path, "ckpt-50")
        self.assertEqual(final_ckpt, "ckpt-150")
        self.assertEqual(trainer._final_ckpt_path, "ckpt-150")

    def test_validation_plateau_decay_reduces_learning_rate(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(
            val_plateau_lr_decay_enabled=True,
            val_plateau_lr_decay_metric="val_drrms",
            val_plateau_lr_decay_warmup=100,
            val_plateau_lr_decay_patience=2,
            val_plateau_lr_decay_factor=0.5,
            val_plateau_lr_decay_min_lr=2.5e-5,
        )
        trainer.optimizer = tf.keras.optimizers.Adam(1.0e-4)
        trainer._val_plateau_best = None
        trainer._val_plateau_bad_count = 0

        msg0 = trainer._maybe_apply_val_plateau_lr_decay(100, {"val_drrms_mean": 1.0})
        msg1 = trainer._maybe_apply_val_plateau_lr_decay(150, {"val_drrms_mean": 1.1})
        msg2 = trainer._maybe_apply_val_plateau_lr_decay(200, {"val_drrms_mean": 1.2})
        lr_after_first = trainer._get_optimizer_learning_rate()
        msg3 = trainer._maybe_apply_val_plateau_lr_decay(250, {"val_drrms_mean": 0.9})
        msg4 = trainer._maybe_apply_val_plateau_lr_decay(300, {"val_drrms_mean": 1.0})
        msg5 = trainer._maybe_apply_val_plateau_lr_decay(350, {"val_drrms_mean": 1.1})
        lr_after_second = trainer._get_optimizer_learning_rate()
        msg6 = trainer._maybe_apply_val_plateau_lr_decay(400, {"val_drrms_mean": 1.2})
        msg7 = trainer._maybe_apply_val_plateau_lr_decay(450, {"val_drrms_mean": 1.3})
        lr_after_third = trainer._get_optimizer_learning_rate()

        self.assertIsNone(msg0)
        self.assertIsNone(msg1)
        self.assertIsInstance(msg2, str)
        self.assertIn("lr_decay", msg2)
        self.assertAlmostEqual(lr_after_first, 5.0e-5)
        self.assertIsNone(msg3)
        self.assertEqual(trainer._val_plateau_bad_count, 0)
        self.assertIsNone(msg4)
        self.assertIsInstance(msg5, str)
        self.assertIn("lr_decay", msg5)
        self.assertAlmostEqual(lr_after_second, 2.5e-5)
        self.assertIsNone(msg6)
        self.assertIsInstance(msg7, str)
        self.assertIn("lr_decay", msg7)
        self.assertAlmostEqual(lr_after_third, 2.5e-5)

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
