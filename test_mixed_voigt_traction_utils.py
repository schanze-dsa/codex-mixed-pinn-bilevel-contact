#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for canonical Voigt and traction utilities."""

from __future__ import annotations

import os
import sys
import unittest

import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from model.voigt_utils import voigt6_to_tensor, tensor_to_voigt6
from physics.traction_utils import traction_from_sigma_voigt, normal_tangential_components


class MixedVoigtTractionUtilsTests(unittest.TestCase):
    def test_voigt_round_trip_uses_canonical_order(self):
        sigma = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=tf.float32)
        tensor = voigt6_to_tensor(sigma)
        back = tensor_to_voigt6(tensor)
        tf.debugging.assert_near(back, sigma)

    def test_traction_matches_sigma_n_for_canonical_order(self):
        sigma = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=tf.float32)
        n = tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32)
        t = traction_from_sigma_voigt(sigma, n)
        tf.debugging.assert_near(t, tf.constant([[4.0, 2.0, 5.0]], dtype=tf.float32))

    def test_normal_tangential_components_split_traction_vector(self):
        t = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
        n = tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32)
        t1 = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
        t2 = tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32)

        tn, tt = normal_tangential_components(t, n, tf.stack([t1, t2], axis=1))
        tf.debugging.assert_near(tn, tf.constant([[2.0]], dtype=tf.float32))
        tf.debugging.assert_near(tt, tf.constant([[1.0, 3.0]], dtype=tf.float32))

if __name__ == "__main__":
    unittest.main()
