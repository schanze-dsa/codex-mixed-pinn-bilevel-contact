#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for forward-mode compatibility in input normalization."""

from __future__ import annotations

import os
import sys
import unittest

import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from model.pinn_model import DisplacementModel, ModelConfig


class ForwardModeNormalizeInputsTests(unittest.TestCase):
    def test_normalize_inputs_supports_forward_accumulator_in_tf_function(self):
        model = DisplacementModel(ModelConfig())
        params = {"P_hat": tf.zeros((1, 3), dtype=tf.float32)}

        @tf.function
        def _probe(x: tf.Tensor) -> tf.Tensor:
            n = tf.shape(x)[0]
            ones = tf.ones((n, 1), dtype=x.dtype)
            zeros = tf.zeros((n, 1), dtype=x.dtype)
            tangent_x = tf.concat([ones, zeros, zeros], axis=1)
            with tf.autodiff.ForwardAccumulator(primals=x, tangents=tangent_x) as acc:
                x_norm, _ = model._normalize_inputs(x, params)
            return acc.jvp(x_norm)

        x = tf.ones((5, 3), dtype=tf.float32)
        jvp = _probe(x)
        self.assertEqual(tuple(jvp.shape), (5, 3))
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(jvp)).numpy()))


if __name__ == "__main__":
    unittest.main()
