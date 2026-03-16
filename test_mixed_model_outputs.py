#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for mixed displacement+stress model outputs."""

from __future__ import annotations

import os
import sys
import unittest

import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from model.pinn_model import ModelConfig, EncoderConfig, FieldConfig, create_displacement_model


class MixedModelOutputsTests(unittest.TestCase):
    def test_eps_guided_stress_head_exposes_bridge_head_and_preserves_sigma_shape(self):
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
        X = tf.zeros((4, 3), dtype=tf.float32)
        params = {"P_hat": tf.zeros((1, 3), dtype=tf.float32)}

        u, sigma = model.us_fn(X, params)

        self.assertIsNotNone(model.field.stress_out_eps_mlp)
        self.assertEqual(tuple(u.shape), (4, 3))
        self.assertEqual(tuple(sigma.shape), (4, 6))

    def test_early_split_stress_branch_exposes_dedicated_layers_and_preserves_shapes(self):
        cfg = ModelConfig(
            encoder=EncoderConfig(out_dim=8),
            field=FieldConfig(
                cond_dim=8,
                use_graph=False,
                stress_out_dim=6,
                stress_branch_early_split=True,
            ),
        )
        model = create_displacement_model(cfg)
        X = tf.zeros((4, 3), dtype=tf.float32)
        params = {"P_hat": tf.zeros((1, 3), dtype=tf.float32)}

        u, sigma = model.us_fn(X, params)

        self.assertGreater(len(model.field.stress_branch_mlp_layers), 0)
        self.assertEqual(tuple(u.shape), (4, 3))
        self.assertEqual(tuple(sigma.shape), (4, 6))

    def test_early_split_stress_branch_keeps_us_fn_pointwise_available(self):
        cfg = ModelConfig(
            encoder=EncoderConfig(out_dim=8),
            field=FieldConfig(
                cond_dim=8,
                use_graph=True,
                stress_out_dim=6,
                graph_layers=1,
                graph_width=16,
                graph_k=2,
                stress_branch_early_split=True,
            ),
        )
        model = create_displacement_model(cfg)
        X = tf.zeros((4, 3), dtype=tf.float32)
        params = {"P_hat": tf.zeros((1, 3), dtype=tf.float32)}

        u, sigma = model.us_fn_pointwise(X, params)

        self.assertGreater(len(model.field.stress_branch_mlp_layers), 0)
        self.assertEqual(tuple(u.shape), (4, 3))
        self.assertEqual(tuple(sigma.shape), (4, 6))

    def test_p2_stress_branch_flags_accept_enabled_path_without_breaking_us_fn(self):
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
        X = tf.zeros((4, 3), dtype=tf.float32)
        params = {"P_hat": tf.zeros((1, 3), dtype=tf.float32)}

        u, sigma = model.us_fn(X, params)

        self.assertEqual(tuple(u.shape), (4, 3))
        self.assertEqual(tuple(sigma.shape), (4, 6))

    def test_p2_stress_branch_flags_accept_disabled_path_without_breaking_us_fn(self):
        cfg = ModelConfig(
            encoder=EncoderConfig(out_dim=8),
            field=FieldConfig(
                cond_dim=8,
                use_graph=False,
                stress_out_dim=6,
                stress_branch_early_split=False,
                use_eps_guided_stress_head=False,
            ),
        )
        model = create_displacement_model(cfg)
        X = tf.zeros((4, 3), dtype=tf.float32)
        params = {"P_hat": tf.zeros((1, 3), dtype=tf.float32)}

        u, sigma = model.us_fn(X, params)

        self.assertEqual(tuple(u.shape), (4, 3))
        self.assertEqual(tuple(sigma.shape), (4, 6))

    def test_mixed_model_exposes_sigma_fn_and_us_fn(self):
        cfg = ModelConfig(
            encoder=EncoderConfig(out_dim=8),
            field=FieldConfig(cond_dim=8, use_graph=False, stress_out_dim=6),
        )
        model = create_displacement_model(cfg)
        X = tf.zeros((4, 3), dtype=tf.float32)
        params = {"P_hat": tf.zeros((1, 3), dtype=tf.float32)}

        u, sigma = model.us_fn(X, params)
        sigma_only = model.sigma_fn(X, params)

        self.assertEqual(tuple(u.shape), (4, 3))
        self.assertEqual(tuple(sigma.shape), (4, 6))
        tf.debugging.assert_near(sigma_only, sigma)


if __name__ == "__main__":
    unittest.main()
