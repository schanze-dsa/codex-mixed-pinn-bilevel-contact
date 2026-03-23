#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused tests for strict-bilevel inner-contact kernel primitives."""

from __future__ import annotations

import os
import sys
import unittest

import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from physics.contact import contact_friction_alm, contact_normal_alm
from physics.contact.contact_inner_kernel_primitives import (
    check_contact_feasibility,
    compose_contact_traction,
    fb_normal_jacobian,
    fb_normal_residual,
    friction_fixed_point_residual,
    inner_normal_jacobian,
    inner_normal_residual,
    project_to_coulomb_disk,
    smooth_penetration_target,
    tangential_fixed_point_gap,
    tangential_update_map,
)


class ContactInnerKernelPrimitiveTests(unittest.TestCase):
    def test_fb_normal_residual_matches_closed_form(self):
        g_n = tf.constant([0.2, 0.0], dtype=tf.float32)
        lambda_n = tf.constant([0.0, 0.4], dtype=tf.float32)
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)

        got = fb_normal_residual(g_n, lambda_n, eps_n)
        want = tf.sqrt(g_n * g_n + lambda_n * lambda_n + eps_n * eps_n) - g_n - lambda_n

        tf.debugging.assert_near(got, want)

    def test_fb_normal_jacobian_matches_closed_form(self):
        g_n = tf.constant([0.2, 0.0], dtype=tf.float32)
        lambda_n = tf.constant([0.1, 0.4], dtype=tf.float32)
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)

        got = fb_normal_jacobian(g_n, lambda_n, eps_n)
        want = lambda_n / tf.sqrt(g_n * g_n + lambda_n * lambda_n + eps_n * eps_n) - 1.0

        tf.debugging.assert_near(got, want)

    def test_smooth_penetration_target_matches_closed_form(self):
        g_n = tf.constant([-0.2, 0.1], dtype=tf.float32)
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)

        got = smooth_penetration_target(g_n, eps_n)
        want = 0.5 * (tf.sqrt(g_n * g_n + eps_n * eps_n) - g_n)

        tf.debugging.assert_near(got, want)

    def test_inner_normal_residual_zeroes_at_smooth_target(self):
        g_n = tf.constant([-0.2, 0.1], dtype=tf.float32)
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)
        lambda_n = smooth_penetration_target(g_n, eps_n)

        got = inner_normal_residual(g_n, lambda_n, eps_n)

        tf.debugging.assert_near(got, tf.zeros_like(lambda_n), atol=1.0e-6)

    def test_inner_normal_jacobian_is_identity(self):
        g_n = tf.constant([-0.2, 0.1], dtype=tf.float32)
        lambda_n = tf.constant([0.3, 0.4], dtype=tf.float32)
        eps_n = tf.constant(1.0e-6, dtype=tf.float32)

        got = inner_normal_jacobian(g_n, lambda_n, eps_n)

        tf.debugging.assert_near(got, tf.ones_like(lambda_n))

    def test_project_to_coulomb_disk_clamps_norm(self):
        tau_trial = tf.constant([[3.0, 4.0]], dtype=tf.float32)
        radius = tf.constant([2.0], dtype=tf.float32)

        projected = project_to_coulomb_disk(tau_trial, radius, eps=1.0e-6)
        proj_norm = tf.sqrt(tf.reduce_sum(tf.square(projected), axis=1))

        tf.debugging.assert_near(proj_norm, radius, atol=1.0e-5)

    def test_tangential_fixed_point_gap_matches_lambda_minus_update_map(self):
        lambda_t = tf.constant([[0.3, -0.1]], dtype=tf.float32)
        ds_t = tf.constant([[0.2, 0.4]], dtype=tf.float32)
        lambda_n = tf.constant([0.5], dtype=tf.float32)
        mu = tf.constant(0.6, dtype=tf.float32)
        k_t = tf.constant(3.0, dtype=tf.float32)
        eps = tf.constant(1.0e-6, dtype=tf.float32)

        update = tangential_update_map(lambda_t, ds_t, lambda_n, mu, k_t, eps=eps)
        gap = tangential_fixed_point_gap(lambda_t, ds_t, lambda_n, mu, k_t, eps=eps)

        tf.debugging.assert_near(gap, lambda_t - update)

    def test_friction_residual_matches_fixed_point_gap(self):
        lambda_t = tf.constant([[0.3, -0.1]], dtype=tf.float32)
        ds_t = tf.constant([[0.2, 0.4]], dtype=tf.float32)
        lambda_n = tf.constant([0.5], dtype=tf.float32)
        mu = tf.constant(0.6, dtype=tf.float32)
        k_t = tf.constant(3.0, dtype=tf.float32)
        eps = tf.constant(1.0e-6, dtype=tf.float32)

        friction_residual = friction_fixed_point_residual(
            lambda_t,
            ds_t,
            lambda_n,
            mu,
            k_t,
            eps=eps,
        )
        fp_gap = tangential_fixed_point_gap(
            lambda_t,
            ds_t,
            lambda_n,
            mu,
            k_t,
            eps=eps,
        )

        tf.debugging.assert_near(friction_residual, fp_gap)

    def test_compose_contact_traction_combines_normal_and_tangential_components(self):
        traction = compose_contact_traction(
            lambda_n=tf.constant([2.0], dtype=tf.float32),
            lambda_t=tf.constant([[0.5, -0.25]], dtype=tf.float32),
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
        )

        tf.debugging.assert_near(
            traction,
            tf.constant([[0.5, 2.0, -0.25]], dtype=tf.float32),
        )

    def test_check_contact_feasibility_reports_cone_violation_and_penetration(self):
        info = check_contact_feasibility(
            g_n=tf.constant([-0.2, 0.1], dtype=tf.float32),
            lambda_n=tf.constant([1.0, 0.0], dtype=tf.float32),
            lambda_t=tf.constant([[0.3, 0.4], [0.0, 0.0]], dtype=tf.float32),
            mu=tf.constant(0.4, dtype=tf.float32),
            tol_n=1.0e-6,
            tol_t=1.0e-6,
        )

        self.assertIn("feasible", info)
        self.assertIn("cone_violation", info)
        self.assertIn("max_penetration", info)
        self.assertGreaterEqual(float(info["max_penetration"]), 0.2)

    def test_legacy_contact_modules_reuse_shared_kernel_primitives(self):
        self.assertIs(contact_normal_alm.normal_fb_residual, fb_normal_residual)
        self.assertIs(contact_friction_alm.project_to_coulomb_disk, project_to_coulomb_disk)


if __name__ == "__main__":
    unittest.main()
