#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contrast checks for forward-only versus normal-ready strict mixed routes."""

from __future__ import annotations

import os
import sys
import unittest
from types import MethodType, SimpleNamespace

import numpy as np
import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from model.loss_energy import TotalConfig, TotalEnergy
from model.pinn_model import EncoderConfig, FieldConfig, ModelConfig, create_displacement_model
from train.trainer import Trainer


class _RouteAwareFakeContact:
    def __init__(self):
        self.calls = []
        self.kwargs = []

    def strict_mixed_inputs(self, u_fn, params=None, *, u_nodes=None):
        del u_fn, params, u_nodes
        return {
            "ds_t": tf.constant([[0.01, 0.0]], dtype=tf.float32),
            "xs": tf.zeros((1, 3), dtype=tf.float32),
            "xm": tf.zeros((1, 3), dtype=tf.float32),
            "t1": tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            "t2": tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            "normals": tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            "weights": tf.ones((1,), dtype=tf.float32),
            "mu": tf.constant(0.3, dtype=tf.float32),
        }

    def solve_strict_inner(
        self,
        u_fn,
        params=None,
        *,
        u_nodes=None,
        strict_inputs=None,
        return_linearization=False,
        **kwargs,
    ):
        del u_fn, params, u_nodes, strict_inputs
        self.calls.append(bool(return_linearization))
        self.kwargs.append(dict(kwargs))
        state = SimpleNamespace(
            lambda_n=tf.constant([0.2], dtype=tf.float32),
            lambda_t=tf.constant([[0.01, 0.0]], dtype=tf.float32),
        )
        linearization = None
        if return_linearization:
            linearization = {
                "schema_version": "strict_mixed_v2",
                "route_mode": "normal_ready",
                "state_layout": {
                    "order": ["lambda_n", "lambda_t"],
                    "lambda_n_shape": [1],
                    "lambda_t_shape": [1, 2],
                },
                "input_layout": {
                    "order": ["g_n", "ds_t"],
                    "g_n_shape": [1],
                    "ds_t_shape": [1, 2],
                },
                "residual": tf.constant([0.1, 0.2, 0.3], dtype=tf.float32),
                "residual_at_solution": tf.constant([0.1, 0.2, 0.3], dtype=tf.float32),
                "jac_z": tf.constant(
                    [
                        [4.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=tf.float32,
                ),
                "jac_inputs": tf.constant(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=tf.float32,
                ),
            }
        return SimpleNamespace(
            state=state,
            traction_vec=tf.constant([[0.0, 0.1, 0.0]], dtype=tf.float32),
            diagnostics={
                "fn_norm": tf.constant(1.0, dtype=tf.float32),
                "ft_norm": tf.constant(0.5, dtype=tf.float32),
                "cone_violation": tf.constant(0.0, dtype=tf.float32),
                "max_penetration": tf.constant(0.0, dtype=tf.float32),
                "fb_residual_norm": tf.constant(0.1, dtype=tf.float32),
                "normal_step_norm": tf.constant(0.2, dtype=tf.float32),
                "tangential_step_norm": tf.constant(0.3, dtype=tf.float32),
                "fallback_used": tf.constant(0.0, dtype=tf.float32),
                "converged": tf.constant(1.0, dtype=tf.float32),
                "iteration_trace": {
                    "fallback_trigger_reason": "policy_penetration_gate",
                    "iterations": [
                        {
                            "tangential_step_mode": "residual_driven_tail_qn",
                            "effective_alpha_scale": 0.25,
                            "tail_has_effective_step": True,
                            "ft_residual_after": 0.125,
                        }
                    ],
                },
            },
            linearization=linearization,
        )


def _run_total_energy(
    normal_ift_enabled: bool,
    *,
    max_tail_qn_iters: int = 0,
    normal_ready_max_inner_iters: int = 0,
):
    total = TotalEnergy(TotalConfig(loss_mode="energy", w_cn=1.0, w_ct=1.0))
    contact = _RouteAwareFakeContact()
    total.attach(contact=contact)
    total.set_mixed_bilevel_flags(
        {
            "phase_name": "phase1",
            "normal_ift_enabled": normal_ift_enabled,
            "tangential_ift_enabled": False,
            "detach_inner_solution": True,
            "max_tail_qn_iters": max(0, int(max_tail_qn_iters)),
            "normal_ready_max_inner_iters": max(0, int(normal_ready_max_inner_iters)),
        }
    )

    def zero_u(X, params=None):
        del X, params
        return tf.zeros((1, 3), dtype=tf.float32)

    def zero_us(X, params=None):
        del X, params
        return (
            tf.zeros((1, 3), dtype=tf.float32),
            tf.zeros((1, 6), dtype=tf.float32),
        )

    _, _, stats = total.energy(zero_u, params={}, stress_fn=zero_us)
    return contact, stats


class MixedForwardOnlyVsNormalReadyTests(unittest.TestCase):
    def test_forward_only_disables_normal_ift_stats_while_normal_ready_enables_them(self):
        forward_contact, forward_stats = _run_total_energy(normal_ift_enabled=False)
        normal_contact, normal_stats = _run_total_energy(normal_ift_enabled=True)

        self.assertEqual(forward_contact.calls, [False])
        self.assertEqual(normal_contact.calls, [True])
        self.assertEqual(float(forward_stats["normal_ift_ready"].numpy()), 0.0)
        self.assertEqual(float(forward_stats["normal_ift_consumed"].numpy()), 0.0)
        self.assertEqual(float(normal_stats["normal_ift_ready"].numpy()), 1.0)
        self.assertEqual(float(normal_stats["normal_ift_consumed"].numpy()), 1.0)
        self.assertGreater(float(normal_stats["normal_ift_valid_ratio"].numpy()), 0.0)
        self.assertTrue(np.isfinite(float(normal_stats["normal_ift_condition_metric"].numpy())))

    def test_strict_mixed_route_forwards_max_tail_qn_iters(self):
        contact, _ = _run_total_energy(normal_ift_enabled=True, max_tail_qn_iters=4)

        self.assertEqual(contact.kwargs, [{"max_tail_qn_iters": 4}])

    def test_normal_ready_route_forwards_route_local_max_inner_iters_only(self):
        normal_contact, _ = _run_total_energy(
            normal_ift_enabled=True,
            max_tail_qn_iters=4,
            normal_ready_max_inner_iters=16,
        )
        forward_contact, _ = _run_total_energy(
            normal_ift_enabled=False,
            max_tail_qn_iters=4,
            normal_ready_max_inner_iters=16,
        )

        self.assertEqual(normal_contact.kwargs, [{"max_tail_qn_iters": 4, "max_inner_iters": 16}])
        self.assertEqual(forward_contact.kwargs, [{"max_tail_qn_iters": 4}])

    def test_strict_mixed_route_surfaces_inner_trace_debug_fields(self):
        _, stats = _run_total_energy(normal_ift_enabled=True, max_tail_qn_iters=4)

        self.assertEqual(stats["fallback_trigger_reason"].numpy().decode("utf-8"), "policy_penetration_gate")
        self.assertEqual(stats["tangential_step_mode"].numpy().decode("utf-8"), "residual_driven_tail_qn")
        self.assertAlmostEqual(float(stats["effective_alpha_scale"].numpy()), 0.25, places=7)
        self.assertAlmostEqual(float(stats["ft_residual_norm"].numpy()), 0.125, places=7)
        self.assertEqual(float(stats["tail_has_effective_step"].numpy()), 1.0)

    def test_compiled_step_keeps_gradient_split_norms_finite_for_route_contrast(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(
            supervision_contribution_floor_enabled=False,
            supervision_contribution_floor_ratio=0.0,
        )
        trainer.loss_state = None
        trainer._active_weight_overrides = {}
        trainer._static_weight_vector = None
        trainer._loss_keys = ["R_u", "R_eq"]
        trainer._base_weights = {"R_u": 1.0, "R_eq": 1.0}
        trainer._uncertainty_enabled = lambda: False  # type: ignore[method-assign]
        trainer._total_ref = object()
        trainer.optimizer = tf.keras.optimizers.Adam(learning_rate=1.0e-3)

        cfg = ModelConfig(
            encoder=EncoderConfig(out_dim=8),
            field=FieldConfig(
                cond_dim=8,
                use_graph=False,
                stress_out_dim=6,
                stress_branch_early_split=True,
                use_eps_guided_stress_head=True,
            ),
        )
        trainer.model = create_displacement_model(cfg)
        X = tf.constant(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=tf.float32,
        )
        model_params = {"P_hat": tf.zeros((1, 3), dtype=tf.float32)}
        trainer.model.us_fn(X, model_params)
        trainer._train_vars = trainer._collect_trainable_variables()

        def _fake_eval(self, total, params, *, stress_fn=None, tape=None):
            del total, stress_fn, tape
            u, sigma = self.model.us_fn(params["X"], {"P_hat": params["P_hat"]})
            parts = {
                "R_u": tf.reduce_mean(tf.square(u)) + tf.constant(1.0e-6, dtype=tf.float32),
                "R_eq": tf.reduce_mean(tf.square(sigma)) + tf.constant(1.0e-6, dtype=tf.float32),
            }
            pi = tf.add_n(list(parts.values()))
            return pi, parts, {"strict_route_mode": params["route_mode"]}

        trainer._evaluate_total_objective = MethodType(_fake_eval, trainer)

        weights = tf.constant([1.0, 1.0], dtype=tf.float32)
        _, _, _, normal_stats, _ = trainer._compiled_step(
            {"X": X, "P_hat": model_params["P_hat"], "route_mode": tf.constant("normal_ready")},
            weights,
        )
        _, _, _, forward_stats, _ = trainer._compiled_step(
            {"X": X, "P_hat": model_params["P_hat"], "route_mode": tf.constant("forward_only")},
            weights,
        )

        for stats in (normal_stats, forward_stats):
            self.assertIn("grad_u_norm", stats)
            self.assertIn("grad_sigma_norm", stats)
            self.assertTrue(np.isfinite(float(stats["grad_u_norm"].numpy())))
            self.assertTrue(np.isfinite(float(stats["grad_sigma_norm"].numpy())))


if __name__ == "__main__":
    unittest.main()
