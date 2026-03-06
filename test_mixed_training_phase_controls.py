#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for mixed bilevel phase controls and continuation caps."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from train.trainer import resolve_mixed_phase_flags
from train.trainer_config import MixedBilevelPhaseConfig, TrainerConfig
from train.trainer_opt_mixin import capped_continuation_update


class MixedTrainingPhaseControlTests(unittest.TestCase):
    def test_phase_2a_enables_normal_ift_only(self):
        cfg = TrainerConfig(
            mixed_bilevel_phase=MixedBilevelPhaseConfig(
                phase_name="phase2a",
                normal_ift_enabled=True,
                tangential_ift_enabled=False,
                detach_inner_solution=False,
            )
        )
        flags = resolve_mixed_phase_flags(cfg)
        self.assertEqual(flags["phase_name"], "phase2a")
        self.assertTrue(flags["normal_ift_enabled"])
        self.assertFalse(flags["tangential_ift_enabled"])
        self.assertFalse(flags["detach_inner_solution"])

    def test_continuation_does_not_shrink_eps_and_expand_kt_too_aggressively(self):
        eps_new, kt_new = capped_continuation_update(1.0, 1.0, eps_factor=0.1, k_t_factor=2.0)
        self.assertAlmostEqual(eps_new, 0.7)
        self.assertAlmostEqual(kt_new, 1.3)


if __name__ == "__main__":
    unittest.main()
