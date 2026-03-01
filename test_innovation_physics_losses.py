#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for innovation physics loss components."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from model.loss_energy import compute_incremental_ed_penalty
from physics.contact.contact_friction_alm import FrictionALMConfig, FrictionContactALM
from physics.contact.contact_normal_alm import NormalALMConfig, NormalContactALM
from train.trainer import compute_uncertainty_proxy_sigma


class InnovationPhysicsLossTests(unittest.TestCase):
    def test_incremental_ed_penalty_positive_when_violated(self):
        val = compute_incremental_ed_penalty(
            tf.constant(2.0, tf.float32),
            tf.constant(1.0, tf.float32),
            tf.constant(0.5, tf.float32),
            margin=tf.constant(0.0, tf.float32),
            use_relu=True,
            squared=True,
        )
        self.assertAlmostEqual(float(val.numpy()), 6.25, places=6)

    def test_incremental_ed_penalty_zero_when_satisfied(self):
        val = compute_incremental_ed_penalty(
            tf.constant(0.2, tf.float32),
            tf.constant(0.1, tf.float32),
            tf.constant(1.0, tf.float32),
            margin=tf.constant(0.0, tf.float32),
            use_relu=True,
            squared=True,
        )
        self.assertAlmostEqual(float(val.numpy()), 0.0, places=8)

    def test_friction_bipotential_term_exposed_in_stats(self):
        xs = np.array([[0.2, 0.0, -0.1], [0.1, 0.0, -0.05]], dtype=np.float32)
        xm = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        t1 = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
        t2 = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        w = np.array([1.0, 1.0], dtype=np.float32)

        normal = NormalContactALM(NormalALMConfig(mode="alm", beta=40.0, mu_n=100.0))
        normal.build_from_numpy(xs, xm, n, w)

        fric_cfg = FrictionALMConfig(
            enabled=True,
            mu_f=0.3,
            k_t=200.0,
            mu_t=300.0,
            use_bipotential_residual=True,
            bipotential_weight=1.0,
        )
        fric = FrictionContactALM(fric_cfg)
        fric.link_normal(normal)
        fric.build_from_numpy(xs, xm, t1, t2, w)

        def u_fn(X, params=None):
            X = tf.convert_to_tensor(X, dtype=tf.float32)
            return tf.zeros_like(X)

        Et, stats = fric.energy(u_fn, params=None)
        self.assertGreaterEqual(float(Et.numpy()), 0.0)
        self.assertIn("E_bi", stats)
        self.assertIn("R_bi_comp", stats)
        self.assertGreaterEqual(float(stats["E_bi"].numpy()), 0.0)

    def test_uncertainty_proxy_sigma_shape_and_positive(self):
        u = tf.constant([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]], dtype=tf.float32)
        sigma = compute_uncertainty_proxy_sigma(u, tf.constant(0.5, tf.float32), proxy_scale=1.0)
        self.assertEqual(tuple(sigma.shape), (2, 1))
        self.assertTrue(np.all(sigma.numpy() > 0.0))


if __name__ == "__main__":
    unittest.main()

