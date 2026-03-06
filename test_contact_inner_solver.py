#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for contact inner solver state/result and fallback behavior."""

from __future__ import annotations

import os
import sys
import unittest

import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from physics.contact.contact_inner_solver import ContactInnerState
from physics.contact.contact_operator import ContactOperator


class ContactInnerSolverTests(unittest.TestCase):
    def test_inner_solver_returns_state_and_tractions(self):
        op = ContactOperator()
        lam_n = tf.constant([2.0], dtype=tf.float32)
        lam_t = tf.constant([[0.5, -0.25]], dtype=tf.float32)
        n = tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32)
        t1 = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
        t2 = tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32)

        result = op.solve_inner_state(lam_n, lam_t, n, t1, t2)

        self.assertIsInstance(result.state, ContactInnerState)
        self.assertEqual(tuple(result.traction_vec.shape), (1, 3))
        tf.debugging.assert_near(result.traction_tangent, lam_t)

    def test_inner_solver_fallback_reuses_last_feasible_state(self):
        op = ContactOperator()
        n = tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32)
        t1 = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
        t2 = tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32)

        first = op.solve_inner_state(
            tf.constant([1.0], dtype=tf.float32),
            tf.constant([[0.1, 0.2]], dtype=tf.float32),
            n,
            t1,
            t2,
        )
        second = op.solve_inner_state(
            tf.constant([5.0], dtype=tf.float32),
            tf.constant([[3.0, 4.0]], dtype=tf.float32),
            n,
            t1,
            t2,
            force_fail=True,
        )

        self.assertFalse(second.state.converged)
        self.assertTrue(second.state.fallback_used)
        tf.debugging.assert_near(second.state.lambda_n, first.state.lambda_n)
        tf.debugging.assert_near(second.state.lambda_t, first.state.lambda_t)


if __name__ == "__main__":
    unittest.main()
