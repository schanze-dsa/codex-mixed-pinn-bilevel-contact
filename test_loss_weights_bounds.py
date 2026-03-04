#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for adaptive loss-weight bound interactions."""

from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def _load_loss_weights_module():
    fake_tf = types.SimpleNamespace(Tensor=type("Tensor", (), {}))
    with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
        return importlib.import_module("train.loss_weights")


loss_weights = _load_loss_weights_module()


class LossWeightBoundsTests(unittest.TestCase):
    def test_balance_mode_respects_absolute_min_weight_even_with_large_min_factor(self):
        state = loss_weights.LossWeightState.from_config(
            base_weights={"E_sigma": 1.0, "E_cn": 1.0},
            adaptive_scheme="balance",
            ema_decay=0.95,
            min_factor=0.25,
            max_factor=4.0,
            min_weight=1.0e-6,
            max_weight=10.0,
            gamma=1.0,
            focus_terms=("E_sigma", "E_cn"),
            update_every=1,
        )

        # Make E_sigma dominate by orders of magnitude.
        loss_weights.update_loss_weights(
            state,
            parts={"E_sigma": 1.0e8, "E_cn": 1.0},
            stats=None,
        )

        self.assertLessEqual(state.current["E_sigma"], 1.0e-4)
        self.assertGreaterEqual(state.current["E_sigma"], 1.0e-6)


if __name__ == "__main__":
    unittest.main()
