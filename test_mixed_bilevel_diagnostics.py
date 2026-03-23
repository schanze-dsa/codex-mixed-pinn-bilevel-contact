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
            "cone_violation": 2.5,
            "max_penetration": 2.6,
            "fb_residual_norm": 1.2,
            "normal_step_norm": 0.7,
            "tangential_step_norm": 0.8,
            "fallback_used": 1.0,
            "ift_linear_residual": 3.3,
            "grad_u_norm": 4.4,
            "grad_sigma_norm": 5.5,
            "strict_phase_hold": 1.0,
            "strict_continuation_backoff": 1.0,
            "continuation_backoff_applied": 1.0,
            "strict_force_detach": 1.0,
            "strict_traction_scale": 0.25,
            "phase_hold_reason": "fallback,penetration",
            "inner_solver_not_stable_count": 3.0,
        }

        inject_bilevel_diagnostics(stats, diagnostics)
        picked = TrainerMonitorMixin.extract_bilevel_diagnostics(stats)

        self.assertEqual(picked["inner_fn_norm"], 1.1)
        self.assertEqual(picked["inner_ft_norm"], 2.2)
        self.assertEqual(picked["inner_cone_violation"], 2.5)
        self.assertEqual(picked["inner_max_penetration"], 2.6)
        self.assertEqual(picked["inner_fb_residual_norm"], 1.2)
        self.assertEqual(picked["inner_normal_step_norm"], 0.7)
        self.assertEqual(picked["inner_tangential_step_norm"], 0.8)
        self.assertEqual(picked["inner_fallback_used"], 1.0)
        self.assertEqual(picked["ift_linear_residual"], 3.3)
        self.assertEqual(picked["grad_u_norm"], 4.4)
        self.assertEqual(picked["grad_sigma_norm"], 5.5)
        self.assertEqual(picked["strict_phase_hold"], 1.0)
        self.assertEqual(picked["strict_continuation_backoff"], 1.0)
        self.assertEqual(picked["continuation_backoff_applied"], 1.0)
        self.assertEqual(picked["strict_force_detach"], 1.0)
        self.assertEqual(picked["strict_traction_scale"], 0.25)
        self.assertEqual(picked["phase_hold_reason"], "fallback,penetration")
        self.assertEqual(picked["inner_solver_not_stable_count"], 3.0)

    def test_monitor_reports_aggregate_strict_bilevel_rates(self):
        picked = TrainerMonitorMixin.extract_bilevel_diagnostics(
            {
                "inner_convergence_rate": 0.75,
                "inner_fallback_rate": 0.25,
                "inner_skip_rate": 0.125,
                "continuation_frozen": 1.0,
                "continuation_freeze_events": 2.0,
            }
        )

        self.assertEqual(picked["inner_convergence_rate"], 0.75)
        self.assertEqual(picked["inner_fallback_rate"], 0.25)
        self.assertEqual(picked["inner_skip_rate"], 0.125)
        self.assertEqual(picked["continuation_frozen"], 1.0)
        self.assertEqual(picked["continuation_freeze_events"], 2.0)


if __name__ == "__main__":
    unittest.main()
