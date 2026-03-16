#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for mixed forward cache and restore compatibility helpers."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from model.pinn_model import (
    ModelConfig,
    EncoderConfig,
    FieldConfig,
    create_displacement_model,
    MixedForwardCache,
)
from train.saved_model_module import ensure_partial_restore_compat


class MixedModelCacheAndRestoreTests(unittest.TestCase):
    def test_early_split_stress_branch_checkpoint_restore_keeps_us_fn_compatible(self):
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

        u_ref, sigma_ref = model.us_fn(X, params)
        self.assertGreater(len(model.field.stress_branch_mlp_layers), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = tf.train.Checkpoint(encoder=model.encoder, field=model.field)
            ckpt_path = ckpt.save(os.path.join(tmpdir, "mixed_model"))

            restored = create_displacement_model(cfg)
            _ = restored.us_fn(X, params)
            status = tf.train.Checkpoint(
                encoder=restored.encoder,
                field=restored.field,
            ).restore(ckpt_path)
            ensure_partial_restore_compat(status)
            u_restored, sigma_restored = restored.us_fn(X, params)

        tf.debugging.assert_near(u_restored, u_ref)
        tf.debugging.assert_near(sigma_restored, sigma_ref)

    def test_us_fn_reuses_single_forward_cache_for_same_inputs(self):
        cfg = ModelConfig(
            encoder=EncoderConfig(out_dim=8),
            field=FieldConfig(cond_dim=8, use_graph=False, stress_out_dim=6),
        )
        model = create_displacement_model(cfg)

        calls = {"count": 0}
        orig = model._us_fn_compiled

        def _wrapped(X, P_hat):
            calls["count"] += 1
            return orig(X, P_hat)

        model._us_fn_compiled = _wrapped

        X = tf.zeros((4, 3), dtype=tf.float32)
        params = {"P_hat": tf.zeros((1, 3), dtype=tf.float32)}
        cache = MixedForwardCache()

        batch1 = model.forward_mixed(X, params, cache=cache)
        batch2 = model.forward_mixed(X, params, cache=cache)

        self.assertEqual(calls["count"], 1)
        self.assertIs(batch1, batch2)

    def test_partial_restore_helper_calls_expect_partial_when_available(self):
        class _Status:
            def __init__(self):
                self.called = False

            def expect_partial(self):
                self.called = True
                return self

        status = _Status()
        out = ensure_partial_restore_compat(status)
        self.assertIs(out, status)
        self.assertTrue(status.called)


if __name__ == "__main__":
    unittest.main()
