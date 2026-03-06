#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for strict bilevel diagnostics wiring."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from train.trainer_monitor_mixin import TrainerMonitorMixin
from train.trainer_opt_mixin import inject_bilevel_diagnostics


class MixedBilevelDiagnosticsTests(unittest.TestCase):
    def test_monitor_reports_inner_and_ift_residuals(self):
        stats = {}
        diagnostics = {
            "fn_norm": 1.1,
            "ft_norm": 2.2,
            "ift_linear_residual": 3.3,
            "grad_u_norm": 4.4,
            "grad_sigma_norm": 5.5,
        }

        inject_bilevel_diagnostics(stats, diagnostics)
        picked = TrainerMonitorMixin.extract_bilevel_diagnostics(stats)

        self.assertEqual(picked["inner_fn_norm"], 1.1)
        self.assertEqual(picked["inner_ft_norm"], 2.2)
        self.assertEqual(picked["ift_linear_residual"], 3.3)
        self.assertEqual(picked["grad_u_norm"], 4.4)
        self.assertEqual(picked["grad_sigma_norm"], 5.5)


if __name__ == "__main__":
    unittest.main()
