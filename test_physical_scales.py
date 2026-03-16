#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused tests for shared physical scale resolution helpers."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from physics.physical_scales import PhysicalScaleConfig


class PhysicalScaleConfigTests(unittest.TestCase):
    def test_explicit_scales_are_preserved(self):
        cfg = PhysicalScaleConfig(L_ref=10.0, u_ref=0.5, sigma_ref=12.0)

        self.assertAlmostEqual(cfg.resolved_L_ref(), 10.0)
        self.assertAlmostEqual(cfg.resolved_u_ref(), 0.5)
        self.assertAlmostEqual(cfg.resolved_sigma_ref(), 12.0)

    def test_resolved_sigma_ref_uses_strain_scale_when_explicit_sigma_missing(self):
        cfg = PhysicalScaleConfig(L_ref=10.0, u_ref=0.5, E_ref=200.0)

        self.assertAlmostEqual(cfg.resolved_sigma_ref(), 10.0)

    def test_degenerate_scales_are_clamped_to_positive_defaults(self):
        cfg = PhysicalScaleConfig(L_ref=0.0, u_ref=-1.0, sigma_ref=-5.0, E_ref=0.0)

        self.assertGreater(cfg.resolved_L_ref(), 0.0)
        self.assertGreater(cfg.resolved_u_ref(), 0.0)
        self.assertGreater(cfg.resolved_sigma_ref(), 0.0)


if __name__ == "__main__":
    unittest.main()
