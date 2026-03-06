#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for stateless contact inner-kernel primitives."""

from __future__ import annotations

import os
import sys
import unittest

import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from physics.contact.contact_normal_alm import fb_residual, normal_fb_residual
from physics.contact.contact_friction_alm import delta_slip_from_pair


class ContactInnerKernelPrimitiveTests(unittest.TestCase):
    def test_normal_fb_kernel_matches_existing_formula(self):
        g = tf.constant([0.2, -0.1], dtype=tf.float32)
        lam = tf.constant([0.3, 0.4], dtype=tf.float32)
        eps = tf.constant(1.0e-3, dtype=tf.float32)

        expect = fb_residual(g, lam, eps)
        got = normal_fb_residual(g, lam, eps)
        tf.debugging.assert_near(got, expect)

    def test_delta_slip_uses_incremental_stage_difference(self):
        us_now = tf.constant([[2.0, 1.0, 0.0]], dtype=tf.float32)
        um_now = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
        us_prev = tf.constant([[1.0, 1.0, 0.0]], dtype=tf.float32)
        um_prev = tf.constant([[0.5, 0.0, 0.0]], dtype=tf.float32)
        t1 = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
        t2 = tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32)

        delta_st = delta_slip_from_pair(us_now, um_now, us_prev, um_prev, t1, t2)
        tf.debugging.assert_near(delta_st, tf.constant([[0.5, 0.0]], dtype=tf.float32))


if __name__ == "__main__":
    unittest.main()
