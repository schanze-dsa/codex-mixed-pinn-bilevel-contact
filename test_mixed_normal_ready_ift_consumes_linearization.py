#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused checks for normal-ready linearization consumption stats."""

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

from model.loss_energy import TotalConfig, TotalEnergy


class _FakeContact:
    def __init__(self):
        self.return_linearization_requested = False

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

    def solve_strict_inner(self, u_fn, params=None, *, u_nodes=None, strict_inputs=None, return_linearization=False):
        del u_fn, params, u_nodes, strict_inputs
        self.return_linearization_requested = bool(return_linearization)
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
            },
            linearization=linearization,
        )


class MixedNormalReadyIFTConsumptionTests(unittest.TestCase):
    def test_normal_ready_route_marks_and_consumes_linearization(self):
        total = TotalEnergy(TotalConfig(loss_mode="energy", w_cn=1.0, w_ct=1.0))
        contact = _FakeContact()
        total.attach(contact=contact)
        total.set_mixed_bilevel_flags(
            {
                "phase_name": "phase1",
                "normal_ift_enabled": True,
                "tangential_ift_enabled": False,
                "detach_inner_solution": True,
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

        self.assertTrue(contact.return_linearization_requested)
        self.assertEqual(float(stats["normal_ift_ready"].numpy()), 1.0)
        self.assertEqual(float(stats["normal_ift_consumed"].numpy()), 1.0)
        self.assertGreater(float(stats["normal_ift_valid_ratio"].numpy()), 0.0)
        self.assertTrue(np.isfinite(float(stats["normal_ift_condition_metric"].numpy())))
        self.assertTrue(np.isfinite(float(stats["ift_linear_residual"].numpy())))


if __name__ == "__main__":
    unittest.main()
