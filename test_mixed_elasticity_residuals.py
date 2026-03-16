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
from model.pinn_model import ModelConfig, EncoderConfig, FieldConfig, create_displacement_model


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

    def test_eps_guided_stress_head_keeps_mixed_residual_paths_computable(self):
        residual = self._make_residual()
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
        model = create_displacement_model(cfg)
        params = {"P_hat": tf.zeros((1, 3), dtype=tf.float32)}

        constitutive = residual.constitutive_residual(model.u_fn, model.sigma_fn, params)
        equilibrium = residual.equilibrium_residual(model.sigma_fn, params)

        self.assertIsNotNone(model.field.stress_out_eps_mlp)
        self.assertEqual(tuple(constitutive.shape), (4, 6))
        self.assertEqual(tuple(equilibrium.shape), (4, 3))
        tf.debugging.assert_all_finite(constitutive, "constitutive residual must stay finite")
        tf.debugging.assert_all_finite(equilibrium, "equilibrium residual must stay finite")


if __name__ == "__main__":
    unittest.main()
