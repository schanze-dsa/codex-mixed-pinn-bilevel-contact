#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for shared physical scale configuration."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from physics.physical_scales import PhysicalScaleConfig
from train.trainer_config import TrainerConfig


class PhysicalScalesConfigTests(unittest.TestCase):
    def test_sigma_ref_prefers_physical_force_or_strain_scale(self):
        cfg = PhysicalScaleConfig(
            L_ref=10.0,
            u_ref=0.1,
            sigma_ref=0.0,
            E_ref=1000.0,
            F_ref=20.0,
            A_ref=2.0,
        )
        self.assertAlmostEqual(cfg.resolved_sigma_ref(), 10.0)

    def test_trainer_config_exposes_shared_physical_scales(self):
        cfg = TrainerConfig()
        self.assertTrue(hasattr(cfg, "physical_scales"))
        self.assertGreater(cfg.physical_scales.resolved_sigma_ref(), 0.0)


if __name__ == "__main__":
    unittest.main()
