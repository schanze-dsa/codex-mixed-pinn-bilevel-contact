#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for geometry-driven strict-bilevel inner solver behavior."""

from __future__ import annotations

import os
import sys
import unittest

import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np

from physics.contact.contact_operator import ContactOperator, StrictMixedContactInputs
from physics.contact.contact_inner_solver import ContactInnerState, solve_contact_inner


class ContactInnerSolverTests(unittest.TestCase):
    def test_contact_operator_inner_solver_backend_marks_strict_mixed_path(self):
        op = ContactOperator()

        self.assertEqual(op.resolve_backend("inner_solver"), "inner_solver")
        self.assertTrue(op.uses_inner_solver_backend("inner_solver"))
        self.assertTrue(callable(op.strict_mixed_inputs))
        self.assertTrue(callable(op.solve_strict_inner))

    def test_inner_solver_returns_geometry_driven_state_and_tractions(self):
        result = solve_contact_inner(
            g_n=tf.constant([0.2], dtype=tf.float32),
            ds_t=tf.constant([[0.0, 0.0]], dtype=tf.float32),
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=0.3,
            eps_n=1.0e-6,
            k_t=10.0,
            init_state=None,
        )
        self.assertIsInstance(result.state, ContactInnerState)
        self.assertTrue(result.state.converged)
        self.assertFalse(result.state.fallback_used)
        self.assertEqual(tuple(result.traction_vec.shape), (1, 3))
        tf.debugging.assert_near(
            result.traction_tangent,
            tf.constant([[0.0, 0.0]], dtype=tf.float32),
        )

    def test_inner_solver_reuses_feasible_init_state_when_iters_are_disabled(self):
        init_state = ContactInnerState(
            lambda_n=tf.constant([0.5], dtype=tf.float32),
            lambda_t=tf.constant([[0.1, 0.0]], dtype=tf.float32),
            converged=True,
            iters=3,
            res_norm=0.0,
            fallback_used=False,
        )

        result = solve_contact_inner(
            g_n=tf.constant([-0.5], dtype=tf.float32),
            ds_t=tf.constant([[1.0, 0.0]], dtype=tf.float32),
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=0.3,
            eps_n=1.0e-6,
            k_t=10.0,
            init_state=init_state,
            max_inner_iters=0,
        )

        self.assertFalse(result.state.converged)
        self.assertTrue(result.state.fallback_used)
        tf.debugging.assert_near(result.state.lambda_n, init_state.lambda_n)
        tf.debugging.assert_near(result.state.lambda_t, init_state.lambda_t)
        self.assertIn("cone_violation", result.diagnostics)
        self.assertIn("max_penetration", result.diagnostics)

    def test_inner_solver_can_return_linearization_payload(self):
        result = solve_contact_inner(
            g_n=tf.constant([-0.1], dtype=tf.float32),
            ds_t=tf.constant([[0.05, 0.0]], dtype=tf.float32),
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=0.3,
            eps_n=1.0e-6,
            k_t=10.0,
            init_state=None,
            return_linearization=True,
        )

        self.assertIsNotNone(result.linearization)
        self.assertIn("jac_z", result.linearization)

    def test_contact_operator_strict_mixed_inputs_round_trip_warm_start(self):
        cat = {
            "xs": np.asarray([[0.05, 0.0, 0.0]], dtype=np.float32),
            "xm": np.asarray([[0.0, 0.1, 0.0]], dtype=np.float32),
            "n": np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
            "t1": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            "t2": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            "w_area": np.asarray([1.0], dtype=np.float32),
        }
        op = ContactOperator()
        op.build_from_cat(cat, extra_weights=None, auto_orient=False)

        def u_fn(X, params=None):
            del params
            X = tf.convert_to_tensor(X, dtype=tf.float32)
            return tf.zeros_like(X)

        first = op.solve_strict_inner(u_fn, params={})
        inputs = op.strict_mixed_inputs(u_fn, params={})

        self.assertIsInstance(inputs, StrictMixedContactInputs)
        self.assertIsNotNone(inputs.init_state)
        tf.debugging.assert_near(inputs.init_state.lambda_n, first.state.lambda_n)
        tf.debugging.assert_near(inputs.init_state.lambda_t, first.state.lambda_t)

    def test_inner_solver_runs_inside_tf_function(self):
        @tf.function
        def run():
            result = solve_contact_inner(
                g_n=tf.constant([-0.1], dtype=tf.float32),
                ds_t=tf.constant([[0.05, 0.0]], dtype=tf.float32),
                normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
                t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
                t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
                mu=0.3,
                eps_n=1.0e-6,
                k_t=10.0,
                init_state=None,
            )
            return (
                result.traction_vec,
                result.traction_tangent,
                result.diagnostics["fn_norm"],
                result.diagnostics["ft_norm"],
            )

        traction_vec, traction_tangent, fn_norm, ft_norm = run()
        self.assertEqual(tuple(traction_vec.shape), (1, 3))
        self.assertEqual(tuple(traction_tangent.shape), (1, 2))
        self.assertGreater(float(fn_norm.numpy()), 0.0)
        self.assertGreater(float(ft_norm.numpy()), 0.0)


if __name__ == "__main__":
    unittest.main()
