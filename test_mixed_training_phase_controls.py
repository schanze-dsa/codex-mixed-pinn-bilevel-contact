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
from train.trainer import Trainer
from train.trainer_opt_mixin import capped_continuation_update


class MixedTrainingPhaseControlTests(unittest.TestCase):
    def test_phase_2a_enables_normal_ift_only(self):
        cfg = TrainerConfig(
            training_profile="strict_mixed_experimental",
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

    def test_locked_profile_forces_phase0_runtime_flags(self):
        cfg = TrainerConfig(
            training_profile="locked",
            mixed_bilevel_phase=MixedBilevelPhaseConfig(
                phase_name="phase2a",
                normal_ift_enabled=True,
                tangential_ift_enabled=False,
                detach_inner_solution=False,
            ),
        )
        flags = resolve_mixed_phase_flags(cfg)

        self.assertEqual(flags["phase_name"], "phase0")
        self.assertFalse(flags["normal_ift_enabled"])
        self.assertFalse(flags["tangential_ift_enabled"])
        self.assertTrue(flags["detach_inner_solution"])

    def test_resolve_contact_backend_auto_uses_legacy_for_phase0(self):
        cfg = TrainerConfig(contact_backend="auto")
        trainer = object.__new__(Trainer)
        trainer.cfg = cfg
        trainer._mixed_phase_flags = resolve_mixed_phase_flags(cfg)

        self.assertEqual(trainer._resolve_contact_backend(), "legacy_alm")

    def test_resolve_contact_backend_auto_uses_inner_solver_for_phase2a(self):
        cfg = TrainerConfig(
            training_profile="strict_mixed_experimental",
            contact_backend="auto",
            mixed_bilevel_phase=MixedBilevelPhaseConfig(
                phase_name="phase2a",
                normal_ift_enabled=False,
                tangential_ift_enabled=False,
                detach_inner_solution=True,
            ),
        )
        trainer = object.__new__(Trainer)
        trainer.cfg = cfg
        trainer._mixed_phase_flags = resolve_mixed_phase_flags(cfg)

        self.assertEqual(trainer._resolve_contact_backend(), "inner_solver")

    def test_locked_profile_keeps_legacy_backend_even_with_strict_flags(self):
        cfg = TrainerConfig(
            training_profile="locked",
            contact_backend="auto",
            mixed_bilevel_phase=MixedBilevelPhaseConfig(
                phase_name="phase2a",
                normal_ift_enabled=True,
                tangential_ift_enabled=False,
                detach_inner_solution=False,
            ),
        )
        trainer = object.__new__(Trainer)
        trainer.cfg = cfg
        trainer._mixed_phase_flags = {
            "phase_name": "phase2a",
            "normal_ift_enabled": True,
            "tangential_ift_enabled": False,
            "detach_inner_solution": False,
        }

        self.assertEqual(trainer._resolve_contact_backend(), "legacy_alm")

    def test_continuation_does_not_shrink_eps_and_expand_kt_too_aggressively(self):
        eps_new, kt_new = capped_continuation_update(1.0, 1.0, eps_factor=0.1, k_t_factor=2.0)
        self.assertAlmostEqual(eps_new, 0.7)
        self.assertAlmostEqual(kt_new, 1.3)

    def test_assemble_total_propagates_mixed_bilevel_phase_flags(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = TrainerConfig(
            training_profile="strict_mixed_experimental",
            mixed_bilevel_phase=MixedBilevelPhaseConfig(
                phase_name="phase2a",
                normal_ift_enabled=True,
                tangential_ift_enabled=False,
                detach_inner_solution=False,
            )
        )
        trainer.elasticity = None
        trainer.contact = None
        trainer.tightening = None
        trainer.bcs_ops = []
        trainer._mixed_phase_flags = resolve_mixed_phase_flags(trainer.cfg)

        total = trainer._assemble_total()

        self.assertEqual(total.mixed_bilevel_flags["phase_name"], "phase2a")
        self.assertTrue(total.mixed_bilevel_flags["normal_ift_enabled"])
        self.assertFalse(total.mixed_bilevel_flags["tangential_ift_enabled"])
        self.assertFalse(total.mixed_bilevel_flags["detach_inner_solution"])

    def test_assemble_total_propagates_resolved_contact_backend(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = TrainerConfig(
            training_profile="strict_mixed_experimental",
            contact_backend="auto",
            mixed_bilevel_phase=MixedBilevelPhaseConfig(
                phase_name="phase2a",
                normal_ift_enabled=False,
                tangential_ift_enabled=False,
                detach_inner_solution=True,
            ),
        )
        trainer.elasticity = None
        trainer.contact = None
        trainer.tightening = None
        trainer.bcs_ops = []
        trainer._mixed_phase_flags = resolve_mixed_phase_flags(trainer.cfg)

        total = trainer._assemble_total()

        self.assertEqual(total.mixed_bilevel_flags["contact_backend"], "inner_solver")

    def test_assemble_total_propagates_b_signature_gated_inner_budget_controls(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = TrainerConfig(
            training_profile="strict_mixed_experimental",
            max_inner_iters_signature_gate="b",
            signature_gated_max_inner_iters=16,
            mixed_bilevel_phase=MixedBilevelPhaseConfig(
                phase_name="phase2a",
                normal_ift_enabled=True,
                tangential_ift_enabled=False,
                detach_inner_solution=False,
            ),
        )
        trainer.elasticity = None
        trainer.contact = None
        trainer.tightening = None
        trainer.bcs_ops = []
        trainer._mixed_phase_flags = resolve_mixed_phase_flags(trainer.cfg)

        total = trainer._assemble_total()

        self.assertEqual(total.mixed_bilevel_flags["max_inner_iters_signature_gate"], "b")
        self.assertEqual(total.mixed_bilevel_flags["signature_gated_max_inner_iters"], 16)

if __name__ == "__main__":
    unittest.main()
