#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for mixed elasticity residual APIs."""

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

from physics.elasticity_config import ElasticityConfig
from physics.elasticity_residual import ElasticityResidual
from physics.material_lib import MaterialLibrary


class MixedElasticityResidualTests(unittest.TestCase):
    def _make_residual(self):
        asm = SimpleNamespace(nodes={1: (0.0, 0.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (0.0, 1.0, 0.0)})
        X_vol = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        w_vol = np.ones((4,), dtype=np.float32)
        mat_id = np.zeros((4,), dtype=np.int64)
        materials = {"mat": (1000.0, 0.25)}
        matlib = MaterialLibrary(materials)
        cfg = ElasticityConfig(coord_scale=1.0, use_forward_mode=False)
        return ElasticityResidual(asm, X_vol, w_vol, mat_id, matlib, materials, cfg=cfg)

    def test_constitutive_residual_is_zero_when_sigma_matches_linear_elasticity(self):
        residual = self._make_residual()
        params = {}

        def u_fn(X, p):
            del p
            return tf.zeros((tf.shape(X)[0], 3), dtype=tf.float32)

        def sigma_fn(X, p):
            del p
            sigma = tf.zeros((6,), dtype=tf.float32)
            return tf.tile(tf.expand_dims(sigma, axis=0), [tf.shape(X)[0], 1])

        r = residual.constitutive_residual(u_fn, sigma_fn, params)
        tf.debugging.assert_near(r, tf.zeros_like(r), atol=1.0e-5)

    def test_equilibrium_residual_is_zero_for_constant_sigma_field(self):
        residual = self._make_residual()

        def sigma_fn(X, params):
            del params
            sigma = tf.constant([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=tf.float32)
            return tf.tile(tf.expand_dims(sigma, axis=0), [tf.shape(X)[0], 1])

        div_sigma = residual.equilibrium_residual(sigma_fn, params={})
        tf.debugging.assert_near(div_sigma, tf.zeros_like(div_sigma), atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
