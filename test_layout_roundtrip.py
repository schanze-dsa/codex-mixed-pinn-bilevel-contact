#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused round-trip checks for strict-mixed linearization layout metadata."""

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


def _numel(shape):
    size = 1
    for dim in shape:
        size *= int(dim)
    return size


class LayoutRoundTripTests(unittest.TestCase):
    def test_strict_mixed_v2_layout_metadata_roundtrips_flat_dimensions(self):
        result = solve_contact_inner(
            g_n=tf.constant([-0.1], dtype=tf.float32),
            ds_t=tf.constant([[0.02, 0.01]], dtype=tf.float32),
            normals=tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32),
            t1=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32),
            t2=tf.constant([[0.0, 0.0, 1.0]], dtype=tf.float32),
            mu=0.3,
            eps_n=1.0e-6,
            k_t=10.0,
            return_linearization=True,
        )

        lin = result.linearization
        self.assertIsNotNone(lin)
        self.assertEqual(lin["schema_version"], "strict_mixed_v2")

        state_layout = lin["state_layout"]
        input_layout = lin["input_layout"]
        state_size = sum(_numel(state_layout[f"{name}_shape"]) for name in state_layout["order"])
        input_size = sum(_numel(input_layout[f"{name}_shape"]) for name in input_layout["order"])

        self.assertEqual(tuple(lin["jac_z"].shape), (3, state_size))
        self.assertEqual(tuple(lin["jac_inputs"].shape), (3, input_size))
        self.assertEqual(tuple(lin["residual"].shape), (3,))
        self.assertEqual(tuple(lin["residual_at_solution"].shape), (3,))


if __name__ == "__main__":
    unittest.main()
