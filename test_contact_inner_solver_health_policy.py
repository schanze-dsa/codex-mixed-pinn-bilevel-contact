#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for strict-mixed runtime health policy diagnostics."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from physics.contact.strict_mixed_policy import resolve_strict_mixed_runtime_policy


class ContactInnerSolverHealthPolicyTests(unittest.TestCase):
    def test_runtime_policy_exposes_reason_string_and_backoff_flag(self):
        policy = resolve_strict_mixed_runtime_policy(
            {
                "fallback_used": 1.0,
                "max_penetration": 2.0e-3,
                "fb_residual_norm": 1.0e-1,
            },
            route_mode="normal_ready",
        )

        stats = policy.as_stats()
        self.assertEqual(stats["strict_phase_hold"], 1.0)
        self.assertEqual(stats["continuation_backoff_applied"], 1.0)
        self.assertIn("fallback", stats["phase_hold_reason"])


if __name__ == "__main__":
    unittest.main()
