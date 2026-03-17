#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for mixed traction matching using geometry-driven inner contact solver outputs."""

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

from model.loss_energy import TotalConfig, TotalEnergy, traction_matching_residual
from physics.contact.contact_inner_solver import ContactInnerState, solve_contact_inner
from physics.contact.contact_operator import ContactOperator, StrictMixedContactInputs, traction_matching_terms


class MixedContactMatchingTests(unittest.TestCase):
    def test_contact_operator_backend_contract_defaults_to_legacy(self):
        op = ContactOperator()

        self.assertEqual(op.resolve_backend("legacy_alm"), "legacy_alm")
        self.assertFalse(op.uses_inner_solver_backend("legacy_alm"))

    def test_outer_mixed_loss_consumes_inner_traction_result(self):
        sigma_s = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=tf.float32)
        sigma_m = tf.constant([[3.0, 2.0, 1.0, 0.0, 0.0, 0.0]], dtype=tf.float32)
        n = tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32)
        t1 = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
        t2 = tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32)

        inner = solve_contact_inner(
            g_n=tf.constant([0.1], dtype=tf.float32),
            ds_t=tf.constant([[0.0, 0.0]], dtype=tf.float32),
            normals=n,
            t1=t1,
            t2=t2,
            mu=0.3,
            eps_n=1.0e-6,
            k_t=10.0,
            init_state=None,
        )

        rs, rm = traction_matching_residual(sigma_s, sigma_m, n, t1, t2, inner)
        rs2, rm2 = traction_matching_terms(sigma_s, sigma_m, n, t1, t2, inner)

        tf.debugging.assert_near(rs, rs2)
        tf.debugging.assert_near(rm, rm2)
        self.assertEqual(tuple(rs.shape), (1, 3))
        self.assertEqual(tuple(rm.shape), (1, 3))

    def test_total_energy_strict_mixed_mode_bypasses_legacy_contact_ops(self):
        cat = {
            "xs": np.asarray([[0.05, 0.0, 0.0]], dtype=np.float32),
            "xm": np.asarray([[0.0, 0.1, 0.0]], dtype=np.float32),
            "n": np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
            "t1": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            "t2": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            "w_area": np.asarray([1.0], dtype=np.float32),
        }
        contact = ContactOperator()
        contact.build_from_cat(cat, extra_weights=None, auto_orient=False)

        def _legacy_fail(*args, **kwargs):
            raise AssertionError("legacy ALM contact path should not be called in strict mixed mode")

        contact.energy = _legacy_fail  # type: ignore[method-assign]
        contact.residual = _legacy_fail  # type: ignore[method-assign]

        total = TotalEnergy(TotalConfig(loss_mode="energy"))
        total.attach(contact=contact)
        total.set_mixed_bilevel_flags(
            {
                "phase_name": "phase2a",
                "normal_ift_enabled": True,
                "tangential_ift_enabled": False,
                "detach_inner_solution": True,
            }
        )

        def u_fn(X, params=None):
            X = tf.convert_to_tensor(X, dtype=tf.float32)
            return tf.zeros_like(X)

        def stress_fn(X, params=None):
            X = tf.convert_to_tensor(X, dtype=tf.float32)
            sigma = tf.zeros((tf.shape(X)[0], 6), dtype=tf.float32)
            return tf.zeros_like(X), sigma

        _, parts, stats = total.energy(u_fn, params={}, stress_fn=stress_fn)

        self.assertGreater(float(parts["E_cn"].numpy()), 0.0)
        self.assertGreater(float(parts["E_ct"].numpy()), 0.0)
        self.assertAlmostEqual(float(stats["mixed_strict_active"].numpy()), 1.0)
        self.assertAlmostEqual(float(stats["mixed_strict_skipped"].numpy()), 0.0)
        self.assertIn("fn_norm", stats)
        self.assertIn("ft_norm", stats)

    def test_total_energy_strict_mixed_mode_falls_back_without_stress_fn(self):
        class LegacyContact:
            def energy(self, u_fn, params=None, *, u_nodes=None):
                del u_fn, params, u_nodes
                return (
                    tf.constant(7.0, dtype=tf.float32),
                    {
                        "E_n": tf.constant(3.0, dtype=tf.float32),
                        "E_t": tf.constant(4.0, dtype=tf.float32),
                    },
                    {"legacy_cn": tf.constant(1.0, dtype=tf.float32)},
                    {"legacy_ct": tf.constant(2.0, dtype=tf.float32)},
                )

        total = TotalEnergy(TotalConfig(loss_mode="energy"))
        total.attach(contact=LegacyContact())
        total.set_mixed_bilevel_flags(
            {
                "phase_name": "phase2a",
                "normal_ift_enabled": True,
                "tangential_ift_enabled": False,
                "detach_inner_solution": True,
            }
        )

        def u_fn(X, params=None):
            X = tf.convert_to_tensor(X, dtype=tf.float32)
            return tf.zeros_like(X)

        _, parts, stats = total.energy(u_fn, params={}, stress_fn=None)
        self.assertAlmostEqual(float(parts["E_cn"].numpy()), 3.0)
        self.assertAlmostEqual(float(parts["E_ct"].numpy()), 4.0)
        self.assertAlmostEqual(float(stats["mixed_strict_active"].numpy()), 0.0)
        self.assertAlmostEqual(float(stats["mixed_strict_skipped"].numpy()), 1.0)

    def test_contact_operator_solve_strict_inner_accepts_typed_inputs(self):
        cat = {
            "xs": np.asarray([[0.05, 0.0, 0.0]], dtype=np.float32),
            "xm": np.asarray([[0.0, 0.1, 0.0]], dtype=np.float32),
            "n": np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
            "t1": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            "t2": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            "w_area": np.asarray([1.0], dtype=np.float32),
        }
        contact = ContactOperator()
        contact.build_from_cat(cat, extra_weights=None, auto_orient=False)

        def u_fn(X, params=None):
            del params
            X = tf.convert_to_tensor(X, dtype=tf.float32)
            return tf.zeros_like(X)

        inputs = contact.strict_mixed_inputs(u_fn, params={})
        self.assertIsInstance(inputs, StrictMixedContactInputs)
        warm_start = ContactInnerState(
            lambda_n=tf.constant([0.25], dtype=tf.float32),
            lambda_t=tf.constant([[0.05, 0.0]], dtype=tf.float32),
            converged=True,
            iters=2,
            res_norm=0.0,
            fallback_used=False,
        )
        inputs.init_state = warm_start

        result = contact.solve_strict_inner(
            u_fn,
            params={},
            strict_inputs=inputs,
            max_inner_iters=0,
            return_linearization=True,
        )

        tf.debugging.assert_near(inputs.init_state.lambda_n, warm_start.lambda_n)
        tf.debugging.assert_near(inputs.init_state.lambda_t, warm_start.lambda_t)
        self.assertIn("jac_z", result.linearization)

    def test_strict_mixed_objective_skips_instead_of_falling_back_to_legacy_contact(self):
        class LegacyContact:
            def energy(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("strict mixed objective must not call legacy contact.energy()")

            def residual(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("strict mixed objective must not call legacy contact.residual()")

        total = TotalEnergy(TotalConfig(loss_mode="energy", w_cn=1.0, w_ct=1.0))
        total.attach(contact=LegacyContact())
        total.set_mixed_bilevel_flags(
            {
                "phase_name": "phase2a",
                "normal_ift_enabled": True,
                "tangential_ift_enabled": False,
                "detach_inner_solution": True,
            }
        )

        def u_fn(X, params=None):
            X = tf.convert_to_tensor(X, dtype=tf.float32)
            return tf.zeros_like(X)

        pi, parts, stats = total.strict_mixed_objective(u_fn, params={}, stress_fn=None)

        self.assertAlmostEqual(float(pi.numpy()), 0.0)
        self.assertAlmostEqual(float(parts["E_cn"].numpy()), 0.0)
        self.assertAlmostEqual(float(parts["E_ct"].numpy()), 0.0)
        self.assertAlmostEqual(float(stats["mixed_strict_skipped"].numpy()), 1.0)
        self.assertAlmostEqual(float(stats["inner_skip_batch"].numpy()), 1.0)


if __name__ == "__main__":
    unittest.main()
