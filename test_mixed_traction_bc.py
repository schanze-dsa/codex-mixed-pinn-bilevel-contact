#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for traction boundary residual path in mixed mode."""

from __future__ import annotations

import os
import sys
import unittest

import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from physics.boundary_conditions import traction_bc_residual
from model.loss_energy import traction_bc_residual_from_model


class MixedTractionBoundaryTests(unittest.TestCase):
    def test_traction_bc_residual_matches_sigma_times_normal_minus_target(self):
        sigma = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=tf.float32)
        normals = tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32)
        target_t = tf.constant([[1.0, 1.0, 1.0]], dtype=tf.float32)

        residual = traction_bc_residual(sigma, normals, target_t)
        tf.debugging.assert_near(residual, tf.constant([[3.0, 1.0, 4.0]], dtype=tf.float32))

    def test_traction_bc_uses_sigma_head_not_u_head(self):
        calls = {"sigma": 0, "u": 0}

        class _ModelStub:
            def sigma_fn(self, X, params=None):
                del params
                calls["sigma"] += 1
                sigma = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=tf.float32)
                return tf.tile(sigma, [tf.shape(X)[0], 1])

            def u_fn(self, X, params=None):
                del X, params
                calls["u"] += 1
                raise RuntimeError("u_fn should not be called for traction BC in mixed mode")

        X = tf.zeros((2, 3), dtype=tf.float32)
        normals = tf.tile(tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32), [2, 1])
        target_t = tf.zeros((2, 3), dtype=tf.float32)

        residual = traction_bc_residual_from_model(_ModelStub(), X, {}, normals, target_t)
        self.assertEqual(calls["sigma"], 1)
        self.assertEqual(calls["u"], 0)
        self.assertEqual(tuple(residual.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
