#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for mixed traction matching using inner contact solver outputs."""

from __future__ import annotations

import os
import sys
import unittest

import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from model.loss_energy import traction_matching_residual
from physics.contact.contact_inner_solver import ContactInnerResult, ContactInnerState
from physics.contact.contact_operator import traction_matching_terms


class MixedContactMatchingTests(unittest.TestCase):
    def test_outer_mixed_loss_consumes_inner_traction_result(self):
        sigma_s = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=tf.float32)
        sigma_m = tf.constant([[3.0, 2.0, 1.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
        n = tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32)
        t1 = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
        t2 = tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32)

        inner = ContactInnerResult(
            state=ContactInnerState(
                lambda_n=tf.constant([1.0], dtype=tf.float32),
                lambda_t=tf.constant([[0.0, 0.0]], dtype=tf.float32),
                converged=True,
                iters=1,
                res_norm=0.0,
                fallback_used=False,
            ),
            traction_vec=tf.constant([[0.1, 0.2, 0.3]], dtype=tf.float32),
            traction_tangent=tf.constant([[0.0, 0.0]], dtype=tf.float32),
            diagnostics={},
        )

        rs, rm = traction_matching_residual(sigma_s, sigma_m, n, t1, t2, inner)
        rs2, rm2 = traction_matching_terms(sigma_s, sigma_m, n, t1, t2, inner)

        tf.debugging.assert_near(rs, rs2)
        tf.debugging.assert_near(rm, rm2)
        self.assertEqual(tuple(rs.shape), (1, 3))
        self.assertEqual(tuple(rm.shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
