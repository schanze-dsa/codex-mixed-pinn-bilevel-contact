#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for mixed displacement+stress model outputs."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

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

    def test_engineering_semantics_affect_stress_path_but_not_displacement_path(self):
        cfg = ModelConfig(
            encoder=EncoderConfig(out_dim=8),
            field=FieldConfig(
                cond_dim=8,
                use_graph=False,
                stress_out_dim=6,
                stress_branch_early_split=True,
                use_engineering_semantics=True,
                semantic_feat_dim=8,
            ),
        )
        model = create_displacement_model(cfg)
        X = tf.constant(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=tf.float32,
        )
        z = model.encoder(tf.zeros((1, 3), dtype=tf.float32))

        model.field.set_node_semantic_features(tf.zeros((4, 8), dtype=tf.float32))
        u0, sigma0 = model.field(X, z, return_stress=True)

        model.field.set_node_semantic_features(tf.ones((4, 8), dtype=tf.float32) * 5.0)
        u1, sigma1 = model.field(X, z, return_stress=True)

        tf.debugging.assert_near(u0, u1, atol=1.0e-6)
        self.assertEqual(tuple(sigma0.shape), (4, 6))
        self.assertEqual(tuple(sigma1.shape), (4, 6))
        tf.debugging.assert_all_finite(sigma0, "sigma must stay finite with zero semantics")
        tf.debugging.assert_all_finite(sigma1, "sigma must stay finite with rich semantics")

    def test_contact_stress_hybrid_bypasses_stress_graph_branch_for_contact_nodes(self):
        cfg = ModelConfig(
            encoder=EncoderConfig(out_dim=8),
            field=FieldConfig(
                cond_dim=8,
                use_graph=True,
                graph_layers=1,
                graph_width=16,
                graph_k=2,
                stress_out_dim=6,
                stress_branch_early_split=True,
                use_engineering_semantics=True,
                semantic_feat_dim=8,
                contact_stress_hybrid_enabled=True,
            ),
        )
        model = create_displacement_model(cfg)
        X = tf.constant(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=tf.float32,
        )
        z = model.encoder(tf.zeros((1, 3), dtype=tf.float32))
        sem = tf.constant(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=tf.float32,
        )
        model.field.set_node_semantic_features(sem)

        with patch.object(
            model.field.stress_branch_graph_layers[0],
            "call",
            side_effect=RuntimeError("stress graph branch should be bypassed"),
        ):
            u, sigma = model.field(X, z, return_stress=True)

        self.assertEqual(tuple(u.shape), (4, 3))
        self.assertEqual(tuple(sigma.shape), (4, 6))

    def test_adaptive_contact_hybrid_feeds_blended_stress_features_into_semantic_fusion(self):
        cfg = ModelConfig(
            encoder=EncoderConfig(out_dim=8),
            field=FieldConfig(
                cond_dim=8,
                use_graph=True,
                graph_layers=1,
                graph_width=16,
                graph_k=2,
                stress_out_dim=6,
                stress_branch_early_split=True,
                use_engineering_semantics=True,
                semantic_feat_dim=8,
                contact_stress_hybrid_enabled=True,
                adaptive_depth_enabled=True,
                adaptive_depth_mode="hard",
                adaptive_depth_shallow_layers=1,
                adaptive_depth_threshold=0.5,
                adaptive_depth_temperature=0.1,
                adaptive_depth_route_source="contact_residual",
            ),
        )
        model = create_displacement_model(cfg)
        X = tf.constant(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=tf.float32,
        )
        z = model.encoder(tf.zeros((1, 3), dtype=tf.float32))
        sem = tf.constant(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=tf.float32,
        )
        model.field.set_node_semantic_features(sem)
        model.field.set_contact_residual_hint(1.0)

        sentinel = tf.fill((4, 16), 7.0)

        def _fake_blend(stress_feat, local_stress_feat, semantic_feat):
            del stress_feat, local_stress_feat, semantic_feat
            return sentinel

        def _fake_fuse(stress_feat, semantic_feat):
            del semantic_feat
            tf.debugging.assert_near(stress_feat, sentinel, atol=0.0)
            return stress_feat

        with patch.object(model.field, "_blend_contact_stress_features", side_effect=_fake_blend):
            with patch.object(model.field, "_fuse_stress_semantics", side_effect=_fake_fuse):
                u, sigma = model.field(X, z, return_stress=True)

        self.assertEqual(tuple(u.shape), (4, 3))
        self.assertEqual(tuple(sigma.shape), (4, 6))


if __name__ == "__main__":
    unittest.main()
