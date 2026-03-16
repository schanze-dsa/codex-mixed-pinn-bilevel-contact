#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for per-step elasticity volume sampling guardrails."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from train.trainer_run_mixin import TrainerRunMixin


class _ElasticityStub:
    def __init__(self, n_cells: int):
        self.n_cells = int(n_cells)
        self.calls = []

    def set_sample_indices(self, indices):
        self.calls.append(indices)


class _RunnerStub(TrainerRunMixin):
    def __init__(self, n_points_per_step: int, n_cells: int):
        self.cfg = SimpleNamespace(
            elas_cfg=SimpleNamespace(n_points_per_step=n_points_per_step),
        )
        self.elasticity = _ElasticityStub(n_cells)


class TrainerVolumeSamplingTests(unittest.TestCase):
    def test_configure_volume_sampling_uses_n_points_per_step_cap(self):
        runner = _RunnerStub(n_points_per_step=8, n_cells=32)

        note = runner._configure_volume_sampling_for_step()

        self.assertEqual(len(runner.elasticity.calls), 1)
        sampled = runner.elasticity.calls[-1]
        self.assertIsInstance(sampled, np.ndarray)
        self.assertEqual(sampled.dtype, np.int64)
        self.assertEqual(sampled.shape, (8,))
        self.assertTrue(np.all(sampled >= 0))
        self.assertTrue(np.all(sampled < 32))
        self.assertEqual(note, "vol=8/32")


if __name__ == "__main__":
    unittest.main()
