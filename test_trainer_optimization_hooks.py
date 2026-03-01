#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for trainer-side optimization hooks."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from train.trainer import Trainer


class _OptWithAggregateArg:
    def apply_gradients(self, grads_and_vars, experimental_aggregate_gradients=True):
        del grads_and_vars, experimental_aggregate_gradients
        return None


class _OptNoAggregateArg:
    def apply_gradients(self, grads_and_vars):
        del grads_and_vars
        return None


class TrainerOptimizationHookTests(unittest.TestCase):
    def test_detect_apply_gradients_kwargs_for_supported_optimizer(self):
        kwargs = Trainer._compute_apply_gradients_kwargs(_OptWithAggregateArg())
        self.assertEqual(kwargs, {"experimental_aggregate_gradients": False})

    def test_detect_apply_gradients_kwargs_for_plain_optimizer(self):
        kwargs = Trainer._compute_apply_gradients_kwargs(_OptNoAggregateArg())
        self.assertEqual(kwargs, {})

    def test_static_weight_vector_cache_for_non_adaptive_mode(self):
        trainer = object.__new__(Trainer)
        trainer.loss_state = None
        trainer._loss_keys = ["E_int", "E_cn", "E_ct"]
        trainer._base_weights = {"E_int": 1.5, "E_cn": 0.25, "E_ct": 0.0}
        trainer._static_weight_vector = None

        trainer._refresh_static_weight_vector()
        w0 = trainer._build_weight_vector().numpy()
        np.testing.assert_allclose(w0, np.asarray([1.5, 0.25, 0.0], dtype=np.float32), rtol=0.0, atol=0.0)

        # Cache should keep old values until explicitly refreshed.
        trainer._base_weights["E_int"] = 7.0
        w1 = trainer._build_weight_vector().numpy()
        np.testing.assert_allclose(w1, np.asarray([1.5, 0.25, 0.0], dtype=np.float32), rtol=0.0, atol=0.0)

        trainer._refresh_static_weight_vector()
        w2 = trainer._build_weight_vector().numpy()
        np.testing.assert_allclose(w2, np.asarray([7.0, 0.25, 0.0], dtype=np.float32), rtol=0.0, atol=0.0)

    def test_format_energy_summary_is_skipped_when_step_bar_disabled(self):
        trainer = object.__new__(Trainer)
        trainer._tqdm_enabled = True
        trainer.cfg = SimpleNamespace(step_bar_enabled=False)

        called = {"count": 0}

        def _fake_format(parts):
            del parts
            called["count"] += 1
            return "summary"

        trainer._format_energy_summary = _fake_format
        out = trainer._format_energy_summary_if_needed({"E_int": tf.constant(1.0, tf.float32)})
        self.assertEqual(out, "")
        self.assertEqual(called["count"], 0)


if __name__ == "__main__":
    unittest.main()
