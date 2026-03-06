#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for CPU<->GPU synchronization guardrails."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import main
from physics.elasticity_config import ElasticityConfig
from physics.elasticity_residual import ElasticityResidual


class _MatLib:
    tags = ["mat"]


class PerformanceSyncGuardTests(unittest.TestCase):
    def test_prepare_config_enforces_locked_route_even_with_legacy_flags(self):
        fake_yaml = {
            "inp_path": "dummy.cdb",
            "mirror_surface_name": "MIRROR UP",
            "material_properties": {"mat": {"E": 1.0e5, "nu": 0.3}},
            "part2mat": {"P1": "mat"},
            "tighten_angle_min": 0.0,
            "tighten_angle_max": 1.0,
            "preload_use_stages": False,
            "incremental_mode": False,
            "preload_staging": {"mode": "all_at_once", "enabled": False},
            "stage_resample_contact": True,
            "resample_contact_every": 20,
            "contact_rar_enabled": True,
            "volume_rar_enabled": True,
            "optimizer_config": {"lbfgs": {"enabled": True}},
            "friction_config": {"smooth_to_strict": True},
            "output_config": {"viz_compare_cases": True},
            "contact_route_update_every": 9,
            "early_exit": {"enabled": True, "check_every": 7},
            "contact_pairs": [],
            "nuts": [],
        }
        fake_asm = SimpleNamespace(surfaces={}, parts={}, nodes={1: (0.0, 0.0, 0.0)})
        with patch.object(main, "_load_yaml_config", return_value=fake_yaml), patch.object(
            main, "load_cdb", return_value=fake_asm
        ), patch.object(main.os.path, "exists", return_value=True):
            cfg, _ = main._prepare_config_with_autoguess()

        self.assertTrue(cfg.preload_use_stages)
        self.assertTrue(cfg.incremental_mode)
        self.assertEqual(str(cfg.total_cfg.preload_stage_mode).strip().lower(), "force_then_lock")
        for legacy_key in (
            "stage_resample_contact",
            "resample_contact_every",
            "contact_rar_enabled",
            "volume_rar_enabled",
            "lbfgs_enabled",
            "friction_smooth_schedule",
            "viz_compare_cases",
        ):
            self.assertFalse(hasattr(cfg, legacy_key), msg=f"legacy key should be removed: {legacy_key}")
        self.assertEqual(cfg.contact_route_update_every, 9)
        self.assertEqual(cfg.early_exit_check_every, 7)

    def test_elasticity_metrics_cache_can_be_disabled(self):
        asm = SimpleNamespace(nodes={1: (0.0, 0.0, 0.0)})
        cfg = ElasticityConfig()
        cfg.cache_sample_metrics = False
        cfg.use_forward_mode = False

        op = ElasticityResidual(
            asm=asm,
            X_vol=np.zeros((4, 3), dtype=np.float32),
            w_vol=np.ones((4,), dtype=np.float32),
            mat_id=np.zeros((4,), dtype=np.int64),
            matlib=_MatLib(),
            materials={"mat": (2.1e5, 0.3)},
            cfg=cfg,
        )

        def _u_fn(X, params):
            del params
            return tf.cast(X, tf.float32) * 0.0

        _E, _stats = op.energy(_u_fn, params={})
        self.assertIsNone(op.last_sample_metrics())


if __name__ == "__main__":
    unittest.main()
