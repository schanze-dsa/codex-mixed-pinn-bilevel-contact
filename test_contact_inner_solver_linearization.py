#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the V2 strict-mixed linearization schema."""

from __future__ import annotations

import os
import sys
import unittest

import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from physics.contact.contact_inner_solver import solve_contact_inner


class ContactInnerSolverLinearizationTests(unittest.TestCase):
    def test_linearization_exposes_v2_schema_and_layout_metadata(self):
        result = solve_contact_inner(
            g_n=tf.constant([-0.1], dtype=tf.float32),
            ds_t=tf.constant([[0.02, 0.0]], dtype=tf.float32),
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=0.3,
            eps_n=1.0e-6,
            k_t=10.0,
            return_linearization=True,
        )

        self.assertIsNotNone(result.linearization)
        lin = result.linearization
        self.assertEqual(lin["schema_version"], "strict_mixed_v2")
        self.assertEqual(lin["route_mode"], "normal_ready")
        self.assertIn("state_layout", lin)
        self.assertIn("input_layout", lin)
        self.assertIn("residual_at_solution", lin)
        self.assertEqual(lin["state_layout"]["order"], ["lambda_n", "lambda_t"])
        self.assertEqual(lin["input_layout"]["order"], ["g_n", "ds_t"])

        self.assertEqual(lin["state_layout"]["lambda_n_shape"], [1])
        self.assertEqual(lin["state_layout"]["lambda_t_shape"], [1, 2])
        self.assertEqual(lin["input_layout"]["g_n_shape"], [1])
        self.assertEqual(lin["input_layout"]["ds_t_shape"], [1, 2])

        self.assertEqual(tuple(lin["residual_at_solution"].shape), (3,))
        tf.debugging.assert_near(
            lin["residual_at_solution"],
            lin["residual"],
            atol=1.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
