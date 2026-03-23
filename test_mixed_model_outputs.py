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

from model.loss_energy import TotalConfig, TotalEnergy
from model.pinn_model import ModelConfig, EncoderConfig, FieldConfig, create_displacement_model
from physics.contact.contact_inner_solver import ContactInnerResult, ContactInnerState
from physics.contact.contact_operator import StrictMixedContactInputs


CONTACT_SURFACE_NORMALS_KEY = "__contact_surface_normals__"
CONTACT_SURFACE_T1_KEY = "__contact_surface_t1__"
CONTACT_SURFACE_T2_KEY = "__contact_surface_t2__"


class MixedModelOutputsTests(unittest.TestCase):
    def _contact_surface_params(
        self,
        *,
        P_hat: tf.Tensor,
        normals: tf.Tensor,
        t1: tf.Tensor,
        t2: tf.Tensor,
    ):
        return {
            "P_hat": P_hat,
            CONTACT_SURFACE_NORMALS_KEY: normals,
            CONTACT_SURFACE_T1_KEY: t1,
            CONTACT_SURFACE_T2_KEY: t2,
        }

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

    def test_contact_surface_semantics_shift_pointwise_stress_without_moving_displacement(self):
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
                use_eps_guided_stress_head=True,
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
        P_hat = tf.zeros((1, 3), dtype=tf.float32)
        normals_a = tf.tile(tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32), [4, 1])
        t1_a = tf.tile(tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32), [4, 1])
        t2_a = tf.tile(tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32), [4, 1])
        normals_b = tf.tile(tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32), [4, 1])
        t1_b = tf.tile(tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32), [4, 1])
        t2_b = tf.tile(tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32), [4, 1])
        model.field.set_node_semantic_features(tf.ones((4, 8), dtype=tf.float32))

        params_a = self._contact_surface_params(
            P_hat=P_hat,
            normals=normals_a,
            t1=t1_a,
            t2=t2_a,
        )
        params_b = self._contact_surface_params(
            P_hat=P_hat,
            normals=normals_b,
            t1=t1_b,
            t2=t2_b,
        )
        captured_eps = []
        original_predict = model.field.predict_stress_from_features

        def _capture_eps(stress_feat, eps_bridge=None):
            captured_eps.append(None if eps_bridge is None else tuple(eps_bridge.shape.as_list()))
            return original_predict(stress_feat, eps_bridge=eps_bridge)

        with patch.object(
            model.field.graph_layers[0],
            "call",
            side_effect=RuntimeError("graph path should stay bypassed on pointwise contact stress"),
        ):
            with patch.object(model.field, "predict_stress_from_features", side_effect=_capture_eps):
                u_a, sigma_a = model.us_fn_pointwise(X, params_a)
                u_b, sigma_b = model.us_fn_pointwise(X, params_b)

        self.assertEqual(tuple(u_a.shape), (4, 3))
        self.assertEqual(tuple(sigma_a.shape), (4, 6))
        self.assertEqual(tuple(u_b.shape), (4, 3))
        self.assertEqual(tuple(sigma_b.shape), (4, 6))
        tf.debugging.assert_near(u_a, u_b, atol=1.0e-6)
        tf.debugging.assert_all_finite(sigma_a, "contact-surface stress output must stay finite")
        tf.debugging.assert_all_finite(sigma_b, "contact-surface stress output must stay finite")
        self.assertEqual(len(captured_eps), 2)
        self.assertIsNotNone(captured_eps[0])
        self.assertIsNotNone(captured_eps[1])
        self.assertEqual(captured_eps[0][-1], 6)
        self.assertEqual(captured_eps[1][-1], 6)
        self.assertGreater(
            float(tf.reduce_max(tf.abs(sigma_a - sigma_b)).numpy()),
            1.0e-6,
            "contact-surface semantics should change the pointwise stress output",
        )

    def test_strict_mixed_contact_surface_defaults_to_pointwise_eps_bridge(self):
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
                use_eps_guided_stress_head=False,
                use_engineering_semantics=True,
                semantic_feat_dim=8,
                strict_mixed_default_eps_bridge=True,
                strict_mixed_contact_pointwise_stress=True,
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
        params = self._contact_surface_params(
            P_hat=tf.zeros((1, 3), dtype=tf.float32),
            normals=tf.tile(tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32), [4, 1]),
            t1=tf.tile(tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32), [4, 1]),
            t2=tf.tile(tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32), [4, 1]),
        )
        model.field.set_node_semantic_features(tf.zeros((4, 8), dtype=tf.float32))

        captured_eps = []
        original_predict = model.field.predict_stress_from_features

        def _capture_eps(stress_feat, eps_bridge=None):
            captured_eps.append(None if eps_bridge is None else tuple(eps_bridge.shape.as_list()))
            return original_predict(stress_feat, eps_bridge=eps_bridge)

        with patch.object(
            model.field.graph_layers[0],
            "call",
            side_effect=RuntimeError("graph path should stay bypassed on strict-mixed contact stress"),
        ):
            with patch.object(model.field, "predict_stress_from_features", side_effect=_capture_eps):
                u, sigma = model.us_fn(X, params)

        self.assertEqual(tuple(u.shape), (4, 3))
        self.assertEqual(tuple(sigma.shape), (4, 6))
        self.assertEqual(len(captured_eps), 1)
        self.assertIsNotNone(captured_eps[0])
        self.assertEqual(captured_eps[0][-1], 6)

    def test_contact_surface_frame_respects_disabled_strict_mixed_defaults(self):
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
                use_eps_guided_stress_head=False,
                use_engineering_semantics=True,
                semantic_feat_dim=8,
                strict_mixed_default_eps_bridge=False,
                strict_mixed_contact_pointwise_stress=False,
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
        params = self._contact_surface_params(
            P_hat=tf.zeros((1, 3), dtype=tf.float32),
            normals=tf.tile(tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32), [4, 1]),
            t1=tf.tile(tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32), [4, 1]),
            t2=tf.tile(tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32), [4, 1]),
        )
        model.field.set_node_semantic_features(tf.zeros((4, 8), dtype=tf.float32))

        with patch.object(
            model.field.mlp_layers[0],
            "call",
            side_effect=RuntimeError("pointwise route should stay disabled when strict defaults are off"),
        ):
            u, sigma = model.us_fn(X, params)

        self.assertEqual(tuple(u.shape), (4, 3))
        self.assertEqual(tuple(sigma.shape), (4, 6))

    def test_strict_mixed_contact_route_passes_contact_surface_frame_to_pointwise_stress(self):
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
                use_eps_guided_stress_head=True,
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
        params = {"P_hat": tf.zeros((1, 3), dtype=tf.float32)}
        normals = tf.tile(tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32), [4, 1])
        t1 = tf.tile(tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32), [4, 1])
        t2 = tf.tile(tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32), [4, 1])
        model.field.set_node_semantic_features(tf.zeros((4, 8), dtype=tf.float32))

        strict_inputs = StrictMixedContactInputs(
            g_n=tf.zeros((4,), dtype=tf.float32),
            ds_t=tf.zeros((4, 2), dtype=tf.float32),
            normals=normals,
            t1=t1,
            t2=t2,
            weights=tf.ones((4,), dtype=tf.float32),
            xs=X,
            xm=X + 0.1,
            mu=tf.constant(0.0, dtype=tf.float32),
            eps_n=tf.constant(1.0e-8, dtype=tf.float32),
            k_t=tf.constant(0.0, dtype=tf.float32),
        )

        class _FakeContact:
            def strict_mixed_inputs(self, u_fn, params=None, u_nodes=None):
                del u_fn, params, u_nodes
                return strict_inputs

            def solve_strict_inner(self, u_fn, params=None, u_nodes=None, strict_inputs=None):
                del u_fn, params, u_nodes, strict_inputs
                state = ContactInnerState(
                    lambda_n=tf.zeros((4,), dtype=tf.float32),
                    lambda_t=tf.zeros((4, 2), dtype=tf.float32),
                    converged=True,
                    iters=1,
                    res_norm=0.0,
                    fallback_used=False,
                )
                return ContactInnerResult(
                    state=state,
                    traction_vec=tf.zeros((4, 3), dtype=tf.float32),
                    traction_tangent=tf.zeros((4, 2), dtype=tf.float32),
                    diagnostics={
                        "fn_norm": tf.constant(0.0, dtype=tf.float32),
                        "ft_norm": tf.constant(0.0, dtype=tf.float32),
                        "cone_violation": tf.constant(0.0, dtype=tf.float32),
                        "max_penetration": tf.constant(0.0, dtype=tf.float32),
                        "fb_residual_norm": tf.constant(0.0, dtype=tf.float32),
                        "normal_step_norm": tf.constant(0.0, dtype=tf.float32),
                        "tangential_step_norm": tf.constant(0.0, dtype=tf.float32),
                        "fallback_used": tf.constant(0.0, dtype=tf.float32),
                        "converged": tf.constant(1.0, dtype=tf.float32),
                    },
                )

        total = TotalEnergy(TotalConfig(loss_mode="energy", w_cn=1.0, w_ct=1.0))
        total.attach(contact=_FakeContact())
        total.set_mixed_bilevel_flags(
            {
                "phase_name": "phase2a",
                "normal_ift_enabled": True,
                "tangential_ift_enabled": False,
                "detach_inner_solution": True,
            }
        )

        pointwise_params = []
        original_pointwise = model.us_fn_pointwise

        def _capture_pointwise(X_arg, params_arg=None):
            pointwise_params.append(dict(params_arg or {}))
            return original_pointwise(X_arg, params_arg)

        with patch.object(model, "us_fn_pointwise", side_effect=_capture_pointwise):
            active, parts, stats = total._strict_mixed_contact_terms(
                model.u_fn,
                params,
                stress_fn=model.us_fn,
            )

        self.assertTrue(active)
        self.assertIn("E_cn", parts)
        self.assertEqual(float(stats["mixed_strict_active"].numpy()), 1.0)
        self.assertEqual(len(pointwise_params), 2)
        self.assertNotIn(CONTACT_SURFACE_NORMALS_KEY, params)
        self.assertNotIn(CONTACT_SURFACE_T1_KEY, params)
        self.assertNotIn(CONTACT_SURFACE_T2_KEY, params)
        for captured in pointwise_params:
            self.assertIn(CONTACT_SURFACE_NORMALS_KEY, captured)
            self.assertIn(CONTACT_SURFACE_T1_KEY, captured)
            self.assertIn(CONTACT_SURFACE_T2_KEY, captured)
            self.assertEqual(tuple(captured[CONTACT_SURFACE_NORMALS_KEY].shape), (4, 3))
            self.assertEqual(tuple(captured[CONTACT_SURFACE_T1_KEY].shape), (4, 3))
            self.assertEqual(tuple(captured[CONTACT_SURFACE_T2_KEY].shape), (4, 3))


if __name__ == "__main__":
    unittest.main()
