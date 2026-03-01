#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke tests for innovation hooks (spectral + semantic + uncertainty)."""

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

from inp_io.inp_parser import AssemblyModel, BoundaryEntry, ElementBlock, PartMesh
from model.pinn_model import ModelConfig, create_displacement_model
from train.trainer import build_node_semantic_features
from train.uncertainty_calibration import calibrate_sigma_by_residual


def _make_minimal_asm() -> AssemblyModel:
    asm = AssemblyModel()
    asm.nodes = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (0.0, 1.0, 0.0),
        4: (0.0, 0.0, 1.0),
    }
    mirror = PartMesh(name="MIRROR")
    mirror.node_ids = [1, 2, 3, 4]
    mirror.nodes_xyz = {nid: asm.nodes[nid] for nid in mirror.node_ids}
    mirror.element_blocks = [
        ElementBlock(elem_type="C3D4", elem_ids=[1], connectivity=[[1, 2, 3, 4]], raw_params={})
    ]
    contact = PartMesh(name="__CONTACT__")
    contact.node_ids = [2, 3]
    contact.nodes_xyz = {nid: asm.nodes[nid] for nid in contact.node_ids}
    asm.parts = {"MIRROR": mirror, "__CONTACT__": contact}
    asm.boundaries = [BoundaryEntry(raw="D,1,UX,0.0"), BoundaryEntry(raw="D,1,UY,0.0")]
    return asm


class InnovationHookTests(unittest.TestCase):
    def test_displacement_model_supports_finite_spectral_semantic_and_uncertainty(self):
        cfg = ModelConfig()
        cfg.field.dfem_mode = True
        cfg.field.n_nodes = 4
        cfg.field.use_graph = False
        cfg.field.use_finite_spectral = True
        cfg.field.finite_spectral_modes = 2
        cfg.field.use_engineering_semantics = True
        cfg.field.semantic_feat_dim = 4
        cfg.field.uncertainty_out_dim = 3

        model = create_displacement_model(cfg)
        sem = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.1],
                [0.0, 0.0, 1.0, 0.2],
                [0.0, 0.0, 0.0, 0.3],
            ],
            dtype=np.float32,
        )
        model.field.set_node_semantic_features(sem)

        X = tf.convert_to_tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=tf.float32,
        )
        params = {"P": tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)}
        u = model.u_fn(X, params)
        self.assertEqual(tuple(u.shape), (4, 3))

        u_mean, log_var = model.uvar_fn(X, params)
        self.assertEqual(tuple(u_mean.shape), (4, 3))
        self.assertEqual(tuple(log_var.shape), (4, 3))

    def test_build_node_semantic_features_shape_and_value_range(self):
        asm = _make_minimal_asm()
        node_ids = np.asarray([1, 2, 3, 4], dtype=np.int64)
        part2mat = {"MIRROR": "jingmian", "__CONTACT__": "jingmian"}
        feats = build_node_semantic_features(
            asm,
            sorted_node_ids=node_ids,
            part2mat=part2mat,
            mirror_surface_name="MIRROR UP",
        )
        self.assertEqual(feats.shape, (4, 4))
        self.assertGreater(float(feats[1, 0]), 0.5)
        self.assertGreater(float(feats[2, 0]), 0.5)
        self.assertGreater(float(feats[0, 1]), 0.5)
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_residual_driven_sigma_calibration_is_monotonic(self):
        sigma = np.asarray([0.05, 0.05, 0.05, 0.05], dtype=np.float64)
        residual = np.asarray([0.1, 0.2, 0.4, 0.8], dtype=np.float64)
        sigma_cal = calibrate_sigma_by_residual(
            sigma,
            residual,
            alpha=0.6,
            beta=0.4,
        )
        self.assertEqual(sigma_cal.shape, sigma.shape)
        self.assertTrue(np.all(np.diff(sigma_cal) >= -1e-12))
        self.assertTrue(np.all(sigma_cal > 0.0))

    def test_graph_mode_without_prebuilt_graph_uses_dynamic_knn(self):
        # Keep the network small to make this a cheap behavior check.
        cfg = ModelConfig()
        cfg.field.dfem_mode = False
        cfg.field.use_graph = True
        cfg.field.graph_layers = 1
        cfg.field.graph_width = 16
        cfg.field.graph_k = 2
        cfg.field.fourier.num = 0
        cfg.field.use_finite_spectral = False
        cfg.field.use_engineering_semantics = False

        model = create_displacement_model(cfg)
        X = tf.convert_to_tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=tf.float32,
        )
        params = {"P": tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)}

        import model.pinn_model as pinn_model_mod

        original_knn = pinn_model_mod._build_knn_graph
        calls = {"count": 0}

        def _fake_knn(coords, k, chunk_size):
            del chunk_size
            calls["count"] += 1
            n = tf.shape(coords)[0]
            row = tf.reshape(tf.range(n, dtype=tf.int32), (-1, 1))
            return tf.tile(row, [1, k])

        pinn_model_mod._build_knn_graph = _fake_knn
        try:
            _ = model.u_fn(X, params)
        finally:
            pinn_model_mod._build_knn_graph = original_knn

        self.assertGreaterEqual(calls["count"], 1)


if __name__ == "__main__":
    unittest.main()
