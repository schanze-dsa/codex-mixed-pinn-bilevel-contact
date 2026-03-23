#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for geometry-driven strict-bilevel inner solver behavior."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import numpy as np

from physics.contact.contact_operator import ContactOperator, StrictMixedContactInputs
from physics.contact.contact_inner_solver import ContactInnerState, solve_contact_inner
from physics.contact.contact_inner_kernel_primitives import (
    friction_fixed_point_residual,
    inner_normal_residual,
    project_to_coulomb_disk,
    smooth_penetration_target,
)


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
        g_n = tf.constant([-0.1], dtype=tf.float32)
        ds_t = tf.constant([[0.05, 0.0]], dtype=tf.float32)
        mu = tf.constant(0.3, dtype=tf.float32)
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)
        k_t = tf.constant(10.0, dtype=tf.float32)
        result = solve_contact_inner(
            g_n=g_n,
            ds_t=ds_t,
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=mu,
            eps_n=eps_n,
            k_t=k_t,
            init_state=None,
            return_linearization=True,
        )

        self.assertIsNotNone(result.linearization)
        self.assertIn("jac_z", result.linearization)
        self.assertIn("jac_inputs", result.linearization)
        self.assertIn("z_splits", result.linearization)
        self.assertIn("input_splits", result.linearization)

        lambda_n = tf.identity(result.state.lambda_n)
        lambda_t = tf.identity(result.state.lambda_t)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(lambda_n)
            tape.watch(lambda_t)
            tape.watch(g_n)
            tape.watch(ds_t)
            flat_residual = tf.concat(
                [
                    tf.reshape(inner_normal_residual(g_n, lambda_n, eps_n), (-1,)),
                    tf.reshape(
                        friction_fixed_point_residual(
                            lambda_t,
                            ds_t,
                            lambda_n,
                            mu,
                            k_t,
                            eps=eps_n,
                        ),
                        (-1,),
                    ),
                ],
                axis=0,
            )

        jac_lambda_n = tape.jacobian(flat_residual, lambda_n)
        jac_lambda_t = tape.jacobian(flat_residual, lambda_t)
        jac_g_n = tape.jacobian(flat_residual, g_n)
        jac_ds_t = tape.jacobian(flat_residual, ds_t)
        del tape

        expected_jac_z = tf.concat(
            [
                tf.reshape(jac_lambda_n, (tf.shape(flat_residual)[0], -1)),
                tf.reshape(jac_lambda_t, (tf.shape(flat_residual)[0], -1)),
            ],
            axis=1,
        )
        expected_jac_inputs = tf.concat(
            [
                tf.reshape(jac_g_n, (tf.shape(flat_residual)[0], -1)),
                tf.reshape(jac_ds_t, (tf.shape(flat_residual)[0], -1)),
            ],
            axis=1,
        )

        tf.debugging.assert_near(result.linearization["residual"], flat_residual, atol=1.0e-6)
        tf.debugging.assert_near(result.linearization["jac_z"], expected_jac_z, atol=1.0e-5)
        tf.debugging.assert_near(result.linearization["jac_inputs"], expected_jac_inputs, atol=1.0e-5)
        self.assertEqual(result.linearization["z_splits"], {"lambda_n": 1, "lambda_t": 2})
        self.assertEqual(result.linearization["input_splits"], {"g_n": 1, "ds_t": 2})

    def test_iteration_trace_exposes_requested_core_metrics(self):
        kwargs = dict(
            g_n=tf.constant([-0.5], dtype=tf.float32),
            ds_t=tf.constant([[0.25, 0.05]], dtype=tf.float32),
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=0.3,
            eps_n=1.0e-6,
            k_t=10.0,
            init_state=None,
            max_inner_iters=1,
        )

        result_default = solve_contact_inner(**kwargs)
        self.assertNotIn("iteration_trace", result_default.diagnostics)

        traced = solve_contact_inner(**kwargs, return_iteration_trace=True)
        self.assertIn("iteration_trace", traced.diagnostics)
        trace = traced.diagnostics["iteration_trace"]
        self.assertIn("iterations", trace)
        self.assertIn("fallback_trigger_reason", trace)
        self.assertGreaterEqual(len(trace["iterations"]), 1)

        first = trace["iterations"][0]
        for key in (
            "fn_residual_before",
            "fn_residual_after",
            "ft_residual_before",
            "ft_residual_after",
            "delta_lambda_n_norm",
            "delta_lambda_t_norm",
            "lambda_t_before_norm",
            "lambda_t_after_norm",
            "cone_violation_before",
            "cone_violation_after",
            "slip_norm",
            "target_lambda_t_norm",
            "ft_reduction_ratio",
            "tangential_backtrack_steps",
            "effective_k_t_scale",
            "tangential_step_mode",
            "effective_alpha_scale",
            "qn_diag_min_raw",
            "qn_diag_min_safe",
            "qn_reg_gamma",
            "qn_invalid_ratio",
            "qn_diag_min",
            "qn_diag_max",
            "qn_step_norm",
            "tail_has_effective_step",
        ):
            self.assertIn(key, first)

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

        traced_op = op.solve_strict_inner(
            u_fn,
            params={},
            return_iteration_trace=True,
            max_inner_iters=1,
        )
        self.assertIn("iteration_trace", traced_op.diagnostics)

    def test_iteration_trace_reports_tangential_target_norm_and_reduction_ratio(self):
        g_n = tf.constant([-0.5], dtype=tf.float32)
        ds_t = tf.constant([[0.25, 0.05]], dtype=tf.float32)
        mu = tf.constant(0.3, dtype=tf.float32)
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)
        k_t = tf.constant(10.0, dtype=tf.float32)

        traced = solve_contact_inner(
            g_n=g_n,
            ds_t=ds_t,
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=mu,
            eps_n=eps_n,
            k_t=k_t,
            init_state=None,
            max_inner_iters=1,
            return_iteration_trace=True,
        )

        first = traced.diagnostics["iteration_trace"]["iterations"][0]
        expected_lambda_n = smooth_penetration_target(g_n, eps_n)
        expected_target = project_to_coulomb_disk(
            k_t * ds_t,
            mu * tf.maximum(expected_lambda_n, 0.0),
            eps=eps_n,
        )
        expected_target_norm = tf.sqrt(
            tf.reduce_sum(tf.square(expected_target), axis=1) + 1.0e-12
        )
        expected_target_norm = float(tf.reduce_max(expected_target_norm).numpy())
        expected_ratio = first["ft_residual_after"] / (first["ft_residual_before"] + 1.0e-12)

        self.assertAlmostEqual(first["target_lambda_t_norm"], expected_target_norm, places=6)
        self.assertAlmostEqual(first["ft_reduction_ratio"], expected_ratio, places=6)

    def test_residual_driven_tangential_step_reduces_aligned_residual_on_first_iteration(self):
        g_n = tf.constant([-0.5], dtype=tf.float32)
        ds_t = tf.constant([[0.25, 0.05]], dtype=tf.float32)
        mu = tf.constant(0.3, dtype=tf.float32)
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)
        k_t = tf.constant(10.0, dtype=tf.float32)
        lambda_n_frozen = smooth_penetration_target(g_n, eps_n)
        lambda_t_before = tf.zeros_like(ds_t)
        residual_before = friction_fixed_point_residual(
            lambda_t_before,
            ds_t,
            lambda_n_frozen,
            mu,
            k_t,
            eps=eps_n,
        )
        lambda_t_after = project_to_coulomb_disk(
            lambda_t_before - residual_before,
            mu * tf.maximum(lambda_n_frozen, 0.0),
        )
        ft_before = tf.reduce_max(
            tf.abs(
                residual_before
            )
        )
        ft_after = tf.reduce_max(
            tf.abs(
                friction_fixed_point_residual(
                    lambda_t_after,
                    ds_t,
                    lambda_n_frozen,
                    mu,
                    k_t,
                    eps=eps_n,
                )
            )
        )

        self.assertLess(float(ft_after.numpy()), float(ft_before.numpy()))

    def test_residual_driven_tangential_step_backtracks_on_alpha_schedule(self):
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)
        lambda_n_target = tf.constant([1.179692], dtype=tf.float32)
        g_n = (tf.square(eps_n) / (4.0 * lambda_n_target)) - lambda_n_target
        lambda_t_init = tf.constant([[-0.8052086, -0.5997116]], dtype=tf.float32)
        ds_t = tf.constant([[0.34474048, -0.35214382]], dtype=tf.float32)
        mu = tf.constant(0.82680154, dtype=tf.float32)
        k_t = tf.constant(0.7622093, dtype=tf.float32)
        init_state = ContactInnerState(
            lambda_n=lambda_n_target,
            lambda_t=lambda_t_init,
            converged=False,
            iters=0,
            res_norm=0.0,
            fallback_used=False,
        )

        full_residual = tf.reduce_max(
            tf.abs(
                friction_fixed_point_residual(
                    project_to_coulomb_disk(
                        lambda_t_init
                        - friction_fixed_point_residual(
                            lambda_t_init,
                            ds_t,
                            lambda_n_target,
                            mu,
                            k_t,
                            eps=eps_n,
                        ),
                        mu * lambda_n_target,
                    ),
                    ds_t,
                    lambda_n_target,
                    mu,
                    k_t,
                    eps=eps_n,
                )
            )
        )
        quarter_candidate = project_to_coulomb_disk(
            lambda_t_init
            - tf.cast(0.25, tf.float32)
            * friction_fixed_point_residual(
                lambda_t_init,
                ds_t,
                lambda_n_target,
                mu,
                k_t,
                eps=eps_n,
            ),
            mu * lambda_n_target,
        )
        quarter_residual = tf.reduce_max(
            tf.abs(
                friction_fixed_point_residual(
                    quarter_candidate,
                    ds_t,
                    lambda_n_target,
                    mu,
                    k_t,
                    eps=eps_n,
                )
            )
        )

        traced = solve_contact_inner(
            g_n=g_n,
            ds_t=ds_t,
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=mu,
            eps_n=eps_n,
            k_t=k_t,
            init_state=init_state,
            max_inner_iters=1,
            return_iteration_trace=True,
        )

        first = traced.diagnostics["iteration_trace"]["iterations"][0]
        self.assertEqual(first["tangential_step_mode"], "residual_driven")
        self.assertAlmostEqual(first["effective_k_t_scale"], 1.0, places=6)
        self.assertAlmostEqual(first["effective_alpha_scale"], 0.25, places=6)
        self.assertEqual(first["tangential_backtrack_steps"], 2)
        self.assertGreaterEqual(float(full_residual.numpy()), first["ft_residual_before"])
        self.assertLess(float(quarter_residual.numpy()), first["ft_residual_before"])
        self.assertAlmostEqual(first["ft_residual_after"], float(quarter_residual.numpy()), places=6)
        tf.debugging.assert_near(traced.state.lambda_t, quarter_candidate, atol=1.0e-6)

    def test_residual_driven_step_beats_best_fixed_point_candidate_on_hard_case(self):
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)
        lambda_n_target = tf.constant([1.179692], dtype=tf.float32)
        g_n = (tf.square(eps_n) / (4.0 * lambda_n_target)) - lambda_n_target
        lambda_t_init = tf.constant([[-0.8052086, -0.5997116]], dtype=tf.float32)
        ds_t = tf.constant([[0.34474048, -0.35214382]], dtype=tf.float32)
        mu = tf.constant(0.82680154, dtype=tf.float32)
        k_t = tf.constant(0.7622093, dtype=tf.float32)
        init_state = ContactInnerState(
            lambda_n=lambda_n_target,
            lambda_t=lambda_t_init,
            converged=False,
            iters=0,
            res_norm=0.0,
            fallback_used=False,
        )

        full_target = project_to_coulomb_disk(
            lambda_t_init + k_t * ds_t,
            mu * lambda_n_target,
            eps=eps_n,
        )
        before_residual = tf.reduce_max(
            tf.abs(
                friction_fixed_point_residual(
                    lambda_t_init,
                    ds_t,
                    lambda_n_target,
                    mu,
                    k_t,
                    eps=eps_n,
                )
            )
        )
        best_alpha_residual = before_residual
        full_target = project_to_coulomb_disk(
            lambda_t_init + k_t * ds_t,
            mu * lambda_n_target,
            eps=eps_n,
        )
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            alpha_target = (1.0 - alpha) * lambda_t_init + alpha * full_target
            alpha_residual = tf.reduce_max(
                tf.abs(
                    friction_fixed_point_residual(
                        alpha_target,
                        ds_t,
                        lambda_n_target,
                        mu,
                        k_t,
                        eps=eps_n,
                    )
                )
            )
            best_alpha_residual = tf.minimum(best_alpha_residual, alpha_residual)

        traced = solve_contact_inner(
            g_n=g_n,
            ds_t=ds_t,
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=mu,
            eps_n=eps_n,
            k_t=k_t,
            init_state=init_state,
            max_inner_iters=1,
            return_iteration_trace=True,
        )

        first = traced.diagnostics["iteration_trace"]["iterations"][0]
        accepted_alpha = tf.constant(first["effective_alpha_scale"], dtype=tf.float32)
        residual_driven_target = project_to_coulomb_disk(
            lambda_t_init
            - accepted_alpha
            * friction_fixed_point_residual(
                lambda_t_init,
                ds_t,
                lambda_n_target,
                mu,
                k_t,
                eps=eps_n,
            ),
            mu * lambda_n_target,
        )
        residual_driven_residual = tf.reduce_max(
            tf.abs(
                friction_fixed_point_residual(
                    residual_driven_target,
                    ds_t,
                    lambda_n_target,
                    mu,
                    k_t,
                    eps=eps_n,
                )
            )
        )
        self.assertGreaterEqual(float(best_alpha_residual.numpy()), float(before_residual.numpy()))
        self.assertEqual(first["tangential_step_mode"], "residual_driven")
        self.assertAlmostEqual(first["effective_k_t_scale"], 1.0, places=6)
        self.assertGreater(first["tangential_backtrack_steps"], 0)
        self.assertLess(float(residual_driven_residual.numpy()), float(before_residual.numpy()))
        self.assertLess(first["ft_residual_after"], float(best_alpha_residual.numpy()))
        self.assertAlmostEqual(first["ft_residual_after"], float(residual_driven_residual.numpy()), places=6)
        tf.debugging.assert_near(traced.state.lambda_t, residual_driven_target, atol=1.0e-6)

    def test_residual_driven_tail_alpha_search_finds_smaller_effective_step(self):
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)
        lambda_n_target = tf.constant([5.352939], dtype=tf.float32)
        g_n = (tf.square(eps_n) / (4.0 * lambda_n_target)) - lambda_n_target
        lambda_t_init = tf.constant([[0.96729338, 0.09507308]], dtype=tf.float32)
        ds_t = tf.constant([[-1.0906749, 1.1028475]], dtype=tf.float32)
        mu = tf.constant(0.7226375, dtype=tf.float32)
        k_t = tf.constant(0.1632299, dtype=tf.float32)
        init_state = ContactInnerState(
            lambda_n=lambda_n_target,
            lambda_t=lambda_t_init,
            converged=False,
            iters=0,
            res_norm=0.0,
            fallback_used=False,
        )

        residual_before = friction_fixed_point_residual(
            lambda_t_init,
            ds_t,
            lambda_n_target,
            mu,
            k_t,
            eps=eps_n,
        )
        before_norm = tf.reduce_max(tf.abs(residual_before))
        base_after = []
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate = project_to_coulomb_disk(
                lambda_t_init - tf.cast(alpha, tf.float32) * residual_before,
                mu * lambda_n_target,
                eps=eps_n,
            )
            candidate_norm = tf.reduce_max(
                tf.abs(
                    friction_fixed_point_residual(
                        candidate,
                        ds_t,
                        lambda_n_target,
                        mu,
                        k_t,
                        eps=eps_n,
                    )
                )
            )
            base_after.append(float(candidate_norm.numpy()))
        tail_candidate = project_to_coulomb_disk(
            lambda_t_init - tf.cast(0.03125, tf.float32) * residual_before,
            mu * lambda_n_target,
            eps=eps_n,
        )
        tail_norm = tf.reduce_max(
            tf.abs(
                friction_fixed_point_residual(
                    tail_candidate,
                    ds_t,
                    lambda_n_target,
                    mu,
                    k_t,
                    eps=eps_n,
                )
            )
        )

        self.assertTrue(all(value >= float(before_norm.numpy()) - 1.0e-12 for value in base_after))
        self.assertLess(float(tail_norm.numpy()), float(before_norm.numpy()))

        traced = solve_contact_inner(
            g_n=g_n,
            ds_t=ds_t,
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=mu,
            eps_n=eps_n,
            k_t=k_t,
            init_state=init_state,
            max_inner_iters=1,
            return_iteration_trace=True,
        )

        first = traced.diagnostics["iteration_trace"]["iterations"][0]
        self.assertEqual(first["tangential_step_mode"], "residual_driven_tail_qn")
        self.assertAlmostEqual(first["effective_alpha_scale"], 0.03125, places=6)
        self.assertGreaterEqual(first["tangential_backtrack_steps"], 5)
        self.assertGreater(first["qn_invalid_ratio"], 0.0)
        self.assertLess(first["ft_residual_after"], first["ft_residual_before"])
        self.assertAlmostEqual(first["ft_residual_after"], float(tail_norm.numpy()), places=6)
        tf.debugging.assert_near(traced.state.lambda_t, tail_candidate, atol=1.0e-6)

    def test_tail_quasi_newton_step_activates_when_diagonal_is_informative(self):
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)
        lambda_n_target = tf.constant([5.1470437], dtype=tf.float32)
        g_n = (tf.square(eps_n) / (4.0 * lambda_n_target)) - lambda_n_target
        lambda_t_init = tf.constant([[0.14286442, -1.3513631]], dtype=tf.float32)
        ds_t = tf.constant([[2.9528809, 0.5218249]], dtype=tf.float32)
        mu = tf.constant(0.25654152, dtype=tf.float32)
        k_t = tf.constant(0.020505147, dtype=tf.float32)
        init_state = ContactInnerState(
            lambda_n=lambda_n_target,
            lambda_t=lambda_t_init,
            converged=False,
            iters=0,
            res_norm=0.0,
            fallback_used=False,
        )

        residual_before = friction_fixed_point_residual(
            lambda_t_init,
            ds_t,
            lambda_n_target,
            mu,
            k_t,
            eps=eps_n,
        )
        before_norm = tf.reduce_max(tf.abs(residual_before))
        residual_driven_after = []
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125):
            candidate = project_to_coulomb_disk(
                lambda_t_init - tf.cast(alpha, tf.float32) * residual_before,
                mu * lambda_n_target,
                eps=eps_n,
            )
            candidate_norm = tf.reduce_max(
                tf.abs(
                    friction_fixed_point_residual(
                        candidate,
                        ds_t,
                        lambda_n_target,
                        mu,
                        k_t,
                        eps=eps_n,
                    )
                )
            )
            residual_driven_after.append(float(candidate_norm.numpy()))

        self.assertTrue(all(value >= float(before_norm.numpy()) - 1.0e-12 for value in residual_driven_after))

        traced = solve_contact_inner(
            g_n=g_n,
            ds_t=ds_t,
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=mu,
            eps_n=eps_n,
            k_t=k_t,
            init_state=init_state,
            max_inner_iters=1,
            return_iteration_trace=True,
        )

        first = traced.diagnostics["iteration_trace"]["iterations"][0]
        self.assertEqual(first["tangential_step_mode"], "residual_driven_tail_qn")
        self.assertGreater(first["qn_diag_max"], first["qn_diag_min"])
        self.assertGreater(first["qn_step_norm"], 0.0)
        self.assertTrue(first["tail_has_effective_step"])
        self.assertGreater(first["effective_alpha_scale"], 0.0)
        self.assertLess(first["ft_residual_after"], first["ft_residual_before"])

    def test_tail_quasi_newton_regularizes_degenerate_diagonal_and_accepts_step(self):
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)
        lambda_n_target = tf.constant([9.444596], dtype=tf.float32)
        g_n = (tf.square(eps_n) / (4.0 * lambda_n_target)) - lambda_n_target
        lambda_t_init = tf.constant([[0.48013484, 0.12382472]], dtype=tf.float32)
        ds_t = tf.constant([[-1.1246238, -3.971846]], dtype=tf.float32)
        mu = tf.constant(0.32480294, dtype=tf.float32)
        k_t = tf.constant(0.032429367, dtype=tf.float32)
        init_state = ContactInnerState(
            lambda_n=lambda_n_target,
            lambda_t=lambda_t_init,
            converged=False,
            iters=0,
            res_norm=0.0,
            fallback_used=False,
        )

        residual_before = friction_fixed_point_residual(
            lambda_t_init,
            ds_t,
            lambda_n_target,
            mu,
            k_t,
            eps=eps_n,
        )
        before_norm = tf.reduce_max(tf.abs(residual_before))
        residual_driven_after = []
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125):
            candidate = project_to_coulomb_disk(
                lambda_t_init - tf.cast(alpha, tf.float32) * residual_before,
                mu * lambda_n_target,
                eps=eps_n,
            )
            candidate_norm = tf.reduce_max(
                tf.abs(
                    friction_fixed_point_residual(
                        candidate,
                        ds_t,
                        lambda_n_target,
                        mu,
                        k_t,
                        eps=eps_n,
                    )
                )
            )
            residual_driven_after.append(float(candidate_norm.numpy()))

        self.assertTrue(all(value >= float(before_norm.numpy()) - 1.0e-12 for value in residual_driven_after))

        traced = solve_contact_inner(
            g_n=g_n,
            ds_t=ds_t,
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=mu,
            eps_n=eps_n,
            k_t=k_t,
            init_state=init_state,
            max_inner_iters=1,
            return_iteration_trace=True,
        )

        first = traced.diagnostics["iteration_trace"]["iterations"][0]
        self.assertEqual(first["tangential_step_mode"], "residual_driven_tail_qn")
        self.assertAlmostEqual(first["qn_diag_min_raw"], 0.0, places=6)
        self.assertGreater(first["qn_diag_min_safe"], 0.0)
        self.assertGreaterEqual(first["qn_reg_gamma"], 0.0)
        self.assertGreater(first["qn_invalid_ratio"], 0.0)
        self.assertGreater(first["qn_diag_min_safe"], first["qn_diag_min_raw"])
        self.assertTrue(first["tail_has_effective_step"])
        self.assertGreater(first["effective_alpha_scale"], 0.0)
        self.assertLess(first["ft_residual_after"], first["ft_residual_before"])

    def test_tail_only_budget_extends_after_effective_tail_qn_step(self):
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)
        lambda_n_target = tf.constant([5.352939], dtype=tf.float32)
        g_n = (tf.square(eps_n) / (4.0 * lambda_n_target)) - lambda_n_target
        lambda_t_init = tf.constant([[0.96729338, 0.09507308]], dtype=tf.float32)
        ds_t = tf.constant([[-1.0906749, 1.1028475]], dtype=tf.float32)
        mu = tf.constant(0.7226375, dtype=tf.float32)
        k_t = tf.constant(0.1632299, dtype=tf.float32)
        init_state = ContactInnerState(
            lambda_n=lambda_n_target,
            lambda_t=lambda_t_init,
            converged=False,
            iters=0,
            res_norm=0.0,
            fallback_used=False,
        )

        traced = solve_contact_inner(
            g_n=g_n,
            ds_t=ds_t,
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=mu,
            eps_n=eps_n,
            k_t=k_t,
            init_state=init_state,
            max_inner_iters=1,
            max_tail_qn_iters=2,
            return_iteration_trace=True,
        )

        iterations = traced.diagnostics["iteration_trace"]["iterations"]
        self.assertEqual(len(iterations), 2)
        self.assertTrue(traced.diagnostics["tail_budget_activated"])
        self.assertEqual(traced.diagnostics["tail_extra_iters_granted"], 1.0)
        self.assertEqual(iterations[0]["tangential_step_mode"], "residual_driven_tail_qn")
        self.assertAlmostEqual(
            iterations[1]["ft_residual_before"],
            iterations[0]["ft_residual_after"],
            places=6,
        )

    def test_tail_only_budget_does_not_extend_non_tail_case(self):
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)
        lambda_n_target = tf.constant([1.179692], dtype=tf.float32)
        g_n = (tf.square(eps_n) / (4.0 * lambda_n_target)) - lambda_n_target
        lambda_t_init = tf.constant([[-0.8052086, -0.5997116]], dtype=tf.float32)
        ds_t = tf.constant([[0.34474048, -0.35214382]], dtype=tf.float32)
        mu = tf.constant(0.82680154, dtype=tf.float32)
        k_t = tf.constant(0.7622093, dtype=tf.float32)
        init_state = ContactInnerState(
            lambda_n=lambda_n_target,
            lambda_t=lambda_t_init,
            converged=False,
            iters=0,
            res_norm=0.0,
            fallback_used=False,
        )

        traced = solve_contact_inner(
            g_n=g_n,
            ds_t=ds_t,
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=mu,
            eps_n=eps_n,
            k_t=k_t,
            init_state=init_state,
            max_inner_iters=1,
            max_tail_qn_iters=2,
            return_iteration_trace=True,
        )

        iterations = traced.diagnostics["iteration_trace"]["iterations"]
        self.assertEqual(len(iterations), 1)
        self.assertFalse(traced.diagnostics["tail_budget_activated"])
        self.assertEqual(traced.diagnostics["tail_extra_iters_granted"], 0.0)
        self.assertEqual(iterations[0]["tangential_step_mode"], "residual_driven")

    def test_tail_bb_step_reduces_second_iteration_when_fixed_alphas_stall(self):
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)
        lambda_n_target = tf.constant([0.76670563], dtype=tf.float32)
        g_n = (tf.square(eps_n) / (4.0 * lambda_n_target)) - lambda_n_target
        lambda_t_init = tf.constant([[-0.20179142, -0.78080326]], dtype=tf.float32)
        ds_t = tf.constant([[-0.80477804, 2.2646985]], dtype=tf.float32)
        mu = tf.constant(0.68090093, dtype=tf.float32)
        k_t = tf.constant(0.09651577, dtype=tf.float32)
        init_state = ContactInnerState(
            lambda_n=lambda_n_target,
            lambda_t=lambda_t_init,
            converged=False,
            iters=0,
            res_norm=0.0,
            fallback_used=False,
        )

        residual0 = friction_fixed_point_residual(
            lambda_t_init,
            ds_t,
            lambda_n_target,
            mu,
            k_t,
            eps=eps_n,
        )
        lambda_t_1 = None
        residual1 = None
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125):
            candidate = project_to_coulomb_disk(
                lambda_t_init - tf.cast(alpha, tf.float32) * residual0,
                mu * lambda_n_target,
                eps=eps_n,
            )
            candidate_residual = friction_fixed_point_residual(
                candidate,
                ds_t,
                lambda_n_target,
                mu,
                k_t,
                eps=eps_n,
            )
            if float(tf.reduce_max(tf.abs(candidate_residual)).numpy()) < float(tf.reduce_max(tf.abs(residual0)).numpy()) - 1.0e-12:
                lambda_t_1 = candidate
                residual1 = candidate_residual
                break

        self.assertIsNotNone(lambda_t_1)
        self.assertIsNotNone(residual1)

        second_iter_schedule_residuals = []
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125):
            candidate = project_to_coulomb_disk(
                lambda_t_1 - tf.cast(alpha, tf.float32) * residual1,
                mu * lambda_n_target,
                eps=eps_n,
            )
            candidate_residual = friction_fixed_point_residual(
                candidate,
                ds_t,
                lambda_n_target,
                mu,
                k_t,
                eps=eps_n,
            )
            second_iter_schedule_residuals.append(float(tf.reduce_max(tf.abs(candidate_residual)).numpy()))

        residual1_norm = float(tf.reduce_max(tf.abs(residual1)).numpy())
        self.assertTrue(all(value >= residual1_norm - 1.0e-12 for value in second_iter_schedule_residuals))

        s = lambda_t_1 - lambda_t_init
        y = residual1 - residual0
        bb_alpha = float((tf.reduce_sum(s * s) / tf.reduce_sum(s * y)).numpy())
        bb_candidate = project_to_coulomb_disk(
            lambda_t_1 - tf.cast(bb_alpha, tf.float32) * residual1,
            mu * lambda_n_target,
            eps=eps_n,
        )
        bb_residual = friction_fixed_point_residual(
            bb_candidate,
            ds_t,
            lambda_n_target,
            mu,
            k_t,
            eps=eps_n,
        )
        bb_residual_norm = float(tf.reduce_max(tf.abs(bb_residual)).numpy())
        self.assertLess(bb_residual_norm, residual1_norm)

        traced = solve_contact_inner(
            g_n=g_n,
            ds_t=ds_t,
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=mu,
            eps_n=eps_n,
            k_t=k_t,
            init_state=init_state,
            max_inner_iters=2,
            return_iteration_trace=True,
        )

        second = traced.diagnostics["iteration_trace"]["iterations"][1]
        self.assertEqual(second["tangential_step_mode"], "residual_driven_tail_bb")
        self.assertGreater(second["effective_alpha_scale"], 1.0)
        self.assertLess(second["ft_residual_after"], second["ft_residual_before"])
        self.assertAlmostEqual(second["effective_alpha_scale"], bb_alpha, places=5)
        self.assertAlmostEqual(second["ft_residual_after"], bb_residual_norm, places=6)
        tf.debugging.assert_near(traced.state.lambda_t, bb_candidate, atol=1.0e-6)

    def test_inactive_tail_budget_does_not_change_final_residual_metrics(self):
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)
        lambda_n_target = tf.constant([1.179692], dtype=tf.float32)
        g_n = (tf.square(eps_n) / (4.0 * lambda_n_target)) - lambda_n_target
        lambda_t_init = tf.constant([[-0.8052086, -0.5997116]], dtype=tf.float32)
        ds_t = tf.constant([[0.34474048, -0.35214382]], dtype=tf.float32)
        mu = tf.constant(0.82680154, dtype=tf.float32)
        k_t = tf.constant(0.7622093, dtype=tf.float32)
        init_state = ContactInnerState(
            lambda_n=lambda_n_target,
            lambda_t=lambda_t_init,
            converged=False,
            iters=0,
            res_norm=0.0,
            fallback_used=False,
        )

        base = solve_contact_inner(
            g_n=g_n,
            ds_t=ds_t,
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=mu,
            eps_n=eps_n,
            k_t=k_t,
            init_state=init_state,
            max_inner_iters=1,
            return_iteration_trace=True,
        )
        with_inactive_extra = solve_contact_inner(
            g_n=g_n,
            ds_t=ds_t,
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=mu,
            eps_n=eps_n,
            k_t=k_t,
            init_state=init_state,
            max_inner_iters=1,
            max_tail_qn_iters=2,
            return_iteration_trace=True,
        )

        tf.debugging.assert_near(base.state.lambda_t, with_inactive_extra.state.lambda_t, atol=1.0e-6)
        self.assertEqual(base.state.iters, with_inactive_extra.state.iters)
        self.assertAlmostEqual(
            float(base.diagnostics["ft_residual_norm"].numpy()),
            float(with_inactive_extra.diagnostics["ft_residual_norm"].numpy()),
            places=6,
        )

    def test_residual_driven_multi_iter_keeps_progress_and_reports_budget_exhaustion(self):
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)
        lambda_n_target = tf.constant([0.96238804], dtype=tf.float32)
        g_n = (tf.square(eps_n) / (4.0 * lambda_n_target)) - lambda_n_target
        lambda_t_init = tf.constant([[-0.7151356, -1.4986448]], dtype=tf.float32)
        ds_t = tf.constant([[0.65127426, -0.11898297]], dtype=tf.float32)
        mu = tf.constant(0.8308563, dtype=tf.float32)
        k_t = tf.constant(0.8712414, dtype=tf.float32)
        init_state = ContactInnerState(
            lambda_n=lambda_n_target,
            lambda_t=lambda_t_init,
            converged=False,
            iters=0,
            res_norm=0.0,
            fallback_used=False,
        )

        traced = solve_contact_inner(
            g_n=g_n,
            ds_t=ds_t,
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=mu,
            eps_n=eps_n,
            k_t=k_t,
            init_state=init_state,
            max_inner_iters=8,
            return_iteration_trace=True,
        )

        iterations = traced.diagnostics["iteration_trace"]["iterations"]
        first = iterations[0]
        second = iterations[1]
        expected_lambda_t = tf.identity(lambda_t_init)
        for _ in range(8):
            residual = friction_fixed_point_residual(
                expected_lambda_t,
                ds_t,
                lambda_n_target,
                mu,
                k_t,
                eps=eps_n,
            )
            before = tf.reduce_max(tf.abs(residual))
            accepted = False
            for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = project_to_coulomb_disk(
                    expected_lambda_t - tf.cast(alpha, tf.float32) * residual,
                    mu * lambda_n_target,
                )
                after = tf.reduce_max(
                    tf.abs(
                        friction_fixed_point_residual(
                            candidate,
                            ds_t,
                            lambda_n_target,
                            mu,
                            k_t,
                            eps=eps_n,
                        )
                    )
                )
                if float(after.numpy()) < float(before.numpy()) - 1.0e-12:
                    expected_lambda_t = candidate
                    accepted = True
                    break
            if not accepted:
                break
        self.assertLess(first["ft_residual_after"], first["ft_residual_before"])
        self.assertLess(second["ft_residual_after"], second["ft_residual_before"])
        self.assertEqual(first["tangential_step_mode"], "residual_driven")
        self.assertEqual(second["tangential_step_mode"], "residual_driven")
        self.assertAlmostEqual(first["effective_alpha_scale"], 1.0, places=6)
        self.assertAlmostEqual(second["effective_alpha_scale"], 1.0, places=6)
        self.assertGreaterEqual(
            sum(1 for item in iterations if item["ft_residual_after"] < item["ft_residual_before"]),
            3,
        )
        self.assertTrue(traced.state.fallback_used)
        self.assertEqual(
            traced.diagnostics["iteration_trace"]["fallback_trigger_reason"],
            "iteration_budget_exhausted",
        )
        tf.debugging.assert_near(traced.state.lambda_t, expected_lambda_t, atol=1.0e-6)

    def test_inner_solver_does_not_false_converge_when_normal_fb_residual_is_large(self):
        traced = solve_contact_inner(
            g_n=tf.constant([-0.5], dtype=tf.float32),
            ds_t=tf.constant([[0.0, 0.0]], dtype=tf.float32),
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=0.0,
            eps_n=1.0e-6,
            k_t=0.0,
            init_state=None,
            max_inner_iters=2,
            damping=0.0,
            return_iteration_trace=True,
        )

        self.assertFalse(traced.state.converged)
        self.assertTrue(traced.state.fallback_used)
        self.assertGreater(
            traced.diagnostics["iteration_trace"]["iterations"][0]["fn_residual_after"],
            1.0e-3,
        )
        self.assertEqual(
            traced.diagnostics["iteration_trace"]["fallback_trigger_reason"],
            "normal_fb_residual_not_reduced",
        )

    def test_normal_only_trace_hits_target_residual_quickly(self):
        traced = solve_contact_inner(
            g_n=tf.constant([-0.84724426], dtype=tf.float32),
            ds_t=tf.constant([[0.0, 0.0]], dtype=tf.float32),
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=0.0,
            eps_n=1.6e-7,
            k_t=0.0,
            init_state=None,
            max_inner_iters=2,
            return_iteration_trace=True,
        )

        iterations = traced.diagnostics["iteration_trace"]["iterations"]
        self.assertGreaterEqual(len(iterations), 1)
        self.assertLess(iterations[0]["fn_residual_after"], iterations[0]["fn_residual_before"])
        self.assertTrue(traced.state.converged)
        self.assertFalse(traced.state.fallback_used)
        self.assertLess(float(traced.diagnostics["fb_residual_norm"]), 1.0e-5)

    def test_tangential_residual_no_longer_false_triggers_on_open_contact_case(self):
        traced = solve_contact_inner(
            g_n=tf.constant([0.2], dtype=tf.float32),
            ds_t=tf.constant([[1.0, 0.0]], dtype=tf.float32),
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=0.1,
            eps_n=1.0e-6,
            k_t=10.0,
            init_state=None,
            max_inner_iters=2,
            return_iteration_trace=True,
        )

        self.assertFalse(traced.state.fallback_used)
        self.assertEqual(traced.diagnostics["iteration_trace"]["fallback_trigger_reason"], "converged")

    def test_normal_only_boosted_gain_clips_to_target_without_overshoot(self):
        kwargs = dict(
            g_n=tf.constant([-0.84724426], dtype=tf.float32),
            ds_t=tf.constant([[0.0, 0.0]], dtype=tf.float32),
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=0.0,
            eps_n=1.6e-7,
            k_t=0.0,
            init_state=None,
            max_inner_iters=2,
            return_iteration_trace=True,
        )

        boosted = solve_contact_inner(**kwargs, damping=2.0)

        boosted_after = boosted.diagnostics["iteration_trace"]["iterations"][0]["fn_residual_after"]
        self.assertLess(boosted_after, 1.0e-5)
        self.assertTrue(boosted.state.converged)

    def test_normal_only_larger_cap_scale_still_clips_to_target(self):
        kwargs = dict(
            g_n=tf.constant([-0.84724426], dtype=tf.float32),
            ds_t=tf.constant([[0.0, 0.0]], dtype=tf.float32),
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=0.0,
            eps_n=1.6e-7,
            k_t=0.0,
            init_state=None,
            max_inner_iters=2,
            return_iteration_trace=True,
            damping=2.0,
        )

        widened = solve_contact_inner(**kwargs, normal_correction_cap_scale=2.0)

        widened_after = widened.diagnostics["iteration_trace"]["iterations"][0]["fn_residual_after"]
        self.assertLess(widened_after, 1.0e-5)
        self.assertTrue(widened.state.converged)

    def test_contact_operator_strict_mixed_inputs_round_trip_warm_start(self):
        cat = {
            "xs": np.asarray([[0.05, 0.0, 0.0]], dtype=np.float32),
            "xm": np.asarray([[0.0, 0.1, 0.0]], dtype=np.float32),
            "n": np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
            "t1": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            "t2": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            "w_area": np.asarray([1.0], dtype=np.float32),
            "pair_id": np.asarray([17], dtype=np.int32),
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
        self.assertIn("weights", inputs.batch_meta)
        self.assertIn("xs", inputs.batch_meta)
        self.assertIn("xm", inputs.batch_meta)
        self.assertTrue(hasattr(inputs, "contact_ids"))
        tf.debugging.assert_equal(inputs.contact_ids, tf.constant([17], dtype=tf.int32))
        self.assertIsNotNone(inputs.init_state)
        tf.debugging.assert_near(inputs.init_state.lambda_n, first.state.lambda_n)
        tf.debugging.assert_near(inputs.init_state.lambda_t, first.state.lambda_t)

    def test_contact_operator_forwards_max_tail_qn_iters(self):
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

        mocked_result = SimpleNamespace(
            state=ContactInnerState(
                lambda_n=tf.constant([0.1], dtype=tf.float32),
                lambda_t=tf.constant([[0.0, 0.0]], dtype=tf.float32),
                converged=True,
                iters=1,
                res_norm=0.0,
                fallback_used=False,
            ),
            traction_vec=tf.constant([[0.0, 0.1, 0.0]], dtype=tf.float32),
            traction_tangent=tf.constant([[0.0, 0.0]], dtype=tf.float32),
            diagnostics={},
            linearization=None,
        )

        with patch("physics.contact.contact_operator.solve_contact_inner", return_value=mocked_result) as mock_solve:
            result = op.solve_strict_inner(u_fn, params={}, max_tail_qn_iters=4)

        self.assertIs(result, mocked_result)
        self.assertEqual(mock_solve.call_args.kwargs["max_tail_qn_iters"], 4)

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

    def test_contact_operator_solve_strict_inner_runs_inside_tf_function(self):
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

        @tf.function
        def run():
            result = op.solve_strict_inner(u_fn, params={}, return_linearization=True)
            return (
                result.traction_vec,
                result.diagnostics["fn_norm"],
                result.diagnostics["ft_norm"],
                result.linearization["jac_z"],
            )

        first = run()
        second = run()
        for traction_vec, fn_norm, ft_norm, jac_z in (first, second):
            self.assertEqual(tuple(traction_vec.shape), (1, 3))
            self.assertGreaterEqual(float(fn_norm.numpy()), 0.0)
            self.assertGreaterEqual(float(ft_norm.numpy()), 0.0)
            self.assertEqual(tuple(jac_z.shape), (3, 3))
        self.assertIsNone(
            op._last_inner_state,
            "graph-mode strict inner solve must not retain symbolic warm-start state in Python cache",
        )


if __name__ == "__main__":
    unittest.main()
