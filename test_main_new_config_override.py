#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for config-path overrides in `main new.py`."""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
MODULE_PATH = ROOT / "main new.py"


def _load_main_new_module():
    spec = importlib.util.spec_from_file_location("main_new_module_for_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MainNewConfigOverrideTests(unittest.TestCase):
    def test_prepare_config_accepts_explicit_config_path(self):
        main_new = _load_main_new_module()
        fake_yaml = {
            "inp_path": "dummy.cdb",
            "mirror_surface_name": "MIRROR UP",
            "material_properties": {"mat": {"E": 1.0e5, "nu": 0.3}},
            "part2mat": {"P1": "mat"},
            "tighten_angle_min": 0.0,
            "tighten_angle_max": 1.0,
            "preload_use_stages": True,
            "incremental_mode": True,
            "preload_staging": {"mode": "force_then_lock", "enabled": True},
            "optimizer_config": {},
            "friction_config": {},
            "output_config": {},
            "contact_pairs": [],
            "nuts": [],
        }
        fake_asm = SimpleNamespace(surfaces={}, parts={}, nodes={1: (0.0, 0.0, 0.0)})
        with patch.object(main_new, "_load_yaml_config", return_value=fake_yaml) as mock_loader, patch.object(
            main_new, "load_cdb", return_value=fake_asm
        ), patch.object(main_new.os.path, "exists", return_value=True):
            cfg, asm = main_new._prepare_config_with_autoguess(config_path="custom_supervision.yaml")

        expected_path = os.path.abspath(str(ROOT / "custom_supervision.yaml"))
        mock_loader.assert_called_once_with(expected_path)
        self.assertIs(asm, fake_asm)
        self.assertTrue(cfg.preload_use_stages)

    def test_prepare_config_parses_grouped_cv_supervision_controls(self):
        main_new = _load_main_new_module()
        fake_yaml = {
            "inp_path": "dummy.cdb",
            "mirror_surface_name": "MIRROR UP",
            "material_properties": {"mat": {"E": 1.0e5, "nu": 0.3}},
            "part2mat": {"P1": "mat"},
            "tighten_angle_min": 0.0,
            "tighten_angle_max": 1.0,
            "preload_use_stages": True,
            "incremental_mode": True,
            "preload_staging": {"mode": "force_then_lock", "enabled": True},
            "optimizer_config": {},
            "friction_config": {},
            "output_config": {},
            "contact_pairs": [],
            "nuts": [],
            "supervision": {
                "enabled": True,
                "case_table_path": "cases.csv",
                "stage_dir": "stages",
                "split_group_key": "base_id",
                "split_stratify_key": "source",
                "test_group_quotas": {"boundary": 1, "corner": 1, "interior": 3},
                "cv_n_folds": 5,
                "cv_fold_index": 3,
                "train_splits": ["train"],
                "eval_splits": ["val"],
            },
        }
        fake_asm = SimpleNamespace(surfaces={}, parts={}, nodes={1: (0.0, 0.0, 0.0)})
        with patch.object(main_new, "_load_yaml_config", return_value=fake_yaml), patch.object(
            main_new, "load_cdb", return_value=fake_asm
        ), patch.object(main_new.os.path, "exists", return_value=True):
            cfg, _ = main_new._prepare_config_with_autoguess(config_path="custom_supervision.yaml")

        self.assertTrue(cfg.supervision.enabled)
        self.assertEqual(cfg.supervision.split_group_key, "base_id")
        self.assertEqual(cfg.supervision.split_stratify_key, "source")
        self.assertEqual(cfg.supervision.test_group_quotas, {"boundary": 1, "corner": 1, "interior": 3})
        self.assertEqual(cfg.supervision.cv_n_folds, 5)
        self.assertEqual(cfg.supervision.cv_fold_index, 3)
        self.assertEqual(cfg.supervision.train_splits, ("train",))
        self.assertEqual(cfg.supervision.eval_splits, ("val",))

    def test_prepare_config_parses_supervision_comparison_controls(self):
        main_new = _load_main_new_module()
        fake_yaml = {
            "inp_path": "dummy.cdb",
            "mirror_surface_name": "MIRROR UP",
            "material_properties": {"mat": {"E": 1.0e5, "nu": 0.3}},
            "part2mat": {"P1": "mat"},
            "tighten_angle_min": 0.0,
            "tighten_angle_max": 1.0,
            "preload_use_stages": True,
            "incremental_mode": True,
            "preload_staging": {"mode": "force_then_lock", "enabled": True},
            "optimizer_config": {},
            "friction_config": {},
            "contact_pairs": [],
            "nuts": [],
            "output_config": {
                "viz_supervision_compare_enabled": True,
                "viz_supervision_compare_split": "test",
                "viz_supervision_compare_sources": ["boundary", "corner", "interior"],
            },
        }
        fake_asm = SimpleNamespace(surfaces={}, parts={}, nodes={1: (0.0, 0.0, 0.0)})
        with patch.object(main_new, "_load_yaml_config", return_value=fake_yaml), patch.object(
            main_new, "load_cdb", return_value=fake_asm
        ), patch.object(main_new.os.path, "exists", return_value=True):
            cfg, _ = main_new._prepare_config_with_autoguess(config_path="custom_supervision.yaml")

        self.assertTrue(cfg.viz_supervision_compare_enabled)
        self.assertEqual(cfg.viz_supervision_compare_split, "test")
        self.assertEqual(cfg.viz_supervision_compare_sources, ("boundary", "corner", "interior"))

    def test_prepare_config_parses_two_stage_training_controls(self):
        main_new = _load_main_new_module()
        fake_yaml = {
            "inp_path": "dummy.cdb",
            "mirror_surface_name": "MIRROR UP",
            "material_properties": {"mat": {"E": 1.0e5, "nu": 0.3}},
            "part2mat": {"P1": "mat"},
            "tighten_angle_min": 0.0,
            "tighten_angle_max": 1.0,
            "preload_use_stages": True,
            "incremental_mode": True,
            "preload_staging": {"mode": "force_then_lock", "enabled": True},
            "optimizer_config": {
                "epochs": 1200,
                "learning_rate": 1.0e-5,
                "validation_eval_every": 100,
            },
            "loss_config": {
                "supervision_contribution_floor_enabled": True,
                "supervision_contribution_floor_ratio": 1.0e-1,
                "base_weights": {
                    "w_data": 1.0,
                    "w_smooth": 1.0e-2,
                },
            },
            "output_config": {
                "save_best_on": "val_drrms",
            },
            "friction_config": {},
            "contact_pairs": [],
            "nuts": [],
            "two_stage_training": {
                "enabled": True,
                "phase1": {
                    "max_steps": 300,
                    "learning_rate": 2.0e-5,
                    "save_best_on": "val_drrms",
                    "supervision_contribution_floor_ratio": 0.2,
                },
                "phase2": {
                    "max_steps": 150,
                    "learning_rate": 2.5e-6,
                    "save_best_on": "val_ratio",
                    "supervision_contribution_floor_ratio": 0.1,
                },
            },
        }
        fake_asm = SimpleNamespace(surfaces={}, parts={}, nodes={1: (0.0, 0.0, 0.0)})
        with patch.object(main_new, "_load_yaml_config", return_value=fake_yaml), patch.object(
            main_new, "load_cdb", return_value=fake_asm
        ), patch.object(main_new.os.path, "exists", return_value=True):
            cfg, _ = main_new._prepare_config_with_autoguess(config_path="custom_two_stage.yaml")

        self.assertTrue(cfg.two_stage_training.enabled)
        self.assertEqual(cfg.max_steps, 1200)
        self.assertEqual(cfg.lr, 1.0e-5)
        self.assertEqual(cfg.save_best_on, "val_drrms")
        self.assertEqual(cfg.supervision_contribution_floor_ratio, 1.0e-1)
        self.assertEqual(cfg.two_stage_training.phase1.max_steps, 300)
        self.assertEqual(cfg.two_stage_training.phase1.lr, 2.0e-5)
        self.assertEqual(cfg.two_stage_training.phase1.save_best_on, "val_drrms")
        self.assertEqual(cfg.two_stage_training.phase1.supervision_contribution_floor_ratio, 0.2)
        self.assertEqual(cfg.two_stage_training.phase2.max_steps, 150)
        self.assertEqual(cfg.two_stage_training.phase2.lr, 2.5e-6)
        self.assertEqual(cfg.two_stage_training.phase2.save_best_on, "val_ratio")
        self.assertEqual(cfg.two_stage_training.phase2.supervision_contribution_floor_ratio, 0.1)

    def test_two_stage_phase_override_keeps_non_phase_fields_stable(self):
        main_new = _load_main_new_module()
        fake_yaml = {
            "inp_path": "dummy.cdb",
            "mirror_surface_name": "MIRROR UP",
            "material_properties": {"mat": {"E": 1.0e5, "nu": 0.3}},
            "part2mat": {"P1": "mat"},
            "tighten_angle_min": 0.0,
            "tighten_angle_max": 1.0,
            "preload_use_stages": True,
            "incremental_mode": True,
            "preload_staging": {"mode": "force_then_lock", "enabled": True},
            "optimizer_config": {
                "epochs": 1200,
                "learning_rate": 1.0e-5,
                "validation_eval_every": 100,
            },
            "loss_config": {
                "supervision_contribution_floor_enabled": True,
                "supervision_contribution_floor_ratio": 1.0e-1,
                "data_smoothing_k": 8,
                "base_weights": {
                    "w_data": 1.0,
                    "w_smooth": 1.0e-2,
                    "w_sigma": 1.0,
                },
                "adaptive": {
                    "enabled": True,
                    "focus_terms": ["w_eq", "w_cn", "w_ct"],
                },
            },
            "output_config": {
                "save_path": "./results/ansys_supervised",
                "save_best_on": "val_drrms",
            },
            "friction_config": {},
            "contact_pairs": [],
            "nuts": [],
            "supervision": {
                "enabled": True,
                "case_table_path": "cases.csv",
                "stage_dir": "stages",
            },
            "two_stage_training": {
                "enabled": True,
                "phase1": {
                    "max_steps": 300,
                    "learning_rate": 2.0e-5,
                    "save_best_on": "val_drrms",
                    "supervision_contribution_floor_ratio": 0.2,
                },
                "phase2": {
                    "max_steps": 150,
                    "learning_rate": 2.5e-6,
                    "save_best_on": "val_ratio",
                    "supervision_contribution_floor_ratio": 0.1,
                },
            },
        }
        fake_asm = SimpleNamespace(surfaces={}, parts={}, nodes={1: (0.0, 0.0, 0.0)})
        with patch.object(main_new, "_load_yaml_config", return_value=fake_yaml), patch.object(
            main_new, "load_cdb", return_value=fake_asm
        ), patch.object(main_new.os.path, "exists", return_value=True):
            cfg, _ = main_new._prepare_config_with_autoguess(config_path="custom_two_stage.yaml")

        phase1_cfg = main_new._derive_phase_config(cfg, "phase1")

        self.assertEqual(phase1_cfg.max_steps, 300)
        self.assertEqual(phase1_cfg.lr, 2.0e-5)
        self.assertEqual(phase1_cfg.save_best_on, "val_drrms")
        self.assertEqual(phase1_cfg.supervision_contribution_floor_ratio, 0.2)
        self.assertEqual(phase1_cfg.total_cfg.w_data, cfg.total_cfg.w_data)
        self.assertEqual(phase1_cfg.total_cfg.w_smooth, cfg.total_cfg.w_smooth)
        self.assertEqual(phase1_cfg.total_cfg.data_smoothing_k, cfg.total_cfg.data_smoothing_k)
        self.assertEqual(phase1_cfg.loss_focus_terms, cfg.loss_focus_terms)
        self.assertEqual(phase1_cfg.supervision.case_table_path, cfg.supervision.case_table_path)
        self.assertEqual(phase1_cfg.supervision.stage_dir, cfg.supervision.stage_dir)
        self.assertEqual(os.path.normpath(phase1_cfg.out_dir), os.path.normpath("./results/ansys_supervised/phase1"))
        self.assertEqual(os.path.normpath(phase1_cfg.ckpt_dir), os.path.normpath("checkpoints/phase1"))
        self.assertEqual(cfg.max_steps, 1200)
        self.assertEqual(cfg.lr, 1.0e-5)
        self.assertEqual(cfg.save_best_on, "val_drrms")
        self.assertEqual(cfg.supervision_contribution_floor_ratio, 1.0e-1)

    def test_two_stage_training_resumes_phase2_from_phase1_best_checkpoint(self):
        main_new = _load_main_new_module()
        asm = object()
        base_cfg = SimpleNamespace(two_stage_training=SimpleNamespace(enabled=True))
        phase1_cfg = SimpleNamespace(run_phase_name="phase1", resume_ckpt_path=None)
        phase2_cfg = SimpleNamespace(run_phase_name="phase2", resume_ckpt_path=None)
        phase1_result = SimpleNamespace(
            best_ckpt_path="checkpoints/phase1/best/ckpt-100",
            final_ckpt_path="checkpoints/phase1/final/ckpt-300",
        )
        phase2_result = SimpleNamespace(
            best_ckpt_path="checkpoints/phase2/best/ckpt-50",
            final_ckpt_path="checkpoints/phase2/final/ckpt-150",
        )

        with patch.object(main_new, "_derive_phase_config", side_effect=[phase1_cfg, phase2_cfg]) as mock_derive, patch.object(
            main_new,
            "_run_single_training_phase",
            side_effect=[phase1_result, phase2_result],
        ) as mock_run:
            result = main_new._run_two_stage_training(base_cfg, asm, export_saved_model="exports/final")

        self.assertEqual(mock_derive.call_args_list[0].args, (base_cfg, "phase1"))
        self.assertEqual(mock_derive.call_args_list[1].args, (base_cfg, "phase2"))
        self.assertEqual(mock_run.call_args_list[0].args, (phase1_cfg, asm))
        self.assertIsNone(mock_run.call_args_list[0].kwargs["export_saved_model"])
        self.assertEqual(mock_run.call_args_list[1].args, (phase2_cfg, asm))
        self.assertEqual(mock_run.call_args_list[1].kwargs["export_saved_model"], "exports/final")
        self.assertEqual(phase2_cfg.resume_ckpt_path, phase1_result.best_ckpt_path)
        self.assertNotEqual(phase2_cfg.resume_ckpt_path, phase1_result.final_ckpt_path)
        self.assertIs(result["phase1"], phase1_result)
        self.assertIs(result["phase2"], phase2_result)

    def test_default_config_matches_supervised_defaults(self):
        main_new = _load_main_new_module()
        fake_asm = SimpleNamespace(surfaces={}, parts={}, nodes={1: (0.0, 0.0, 0.0)})

        with patch.object(main_new, "load_cdb", return_value=fake_asm):
            cfg, _ = main_new._prepare_config_with_autoguess()

        self.assertTrue(cfg.supervision.enabled)
        self.assertEqual(cfg.supervision.stage_count, 3)
        self.assertEqual(cfg.supervision.train_splits, ("train",))
        self.assertEqual(cfg.supervision.eval_splits, ("val", "test"))
        self.assertEqual(
            cfg.supervision.test_group_quotas,
            {"boundary": 1, "corner": 1, "interior": 3},
        )
        self.assertFalse(cfg.preload_append_release_stage)
        self.assertEqual(cfg.stage_schedule_steps, [400, 400, 400])
        self.assertEqual(cfg.total_cfg.w_data, 1.0)
        self.assertEqual(cfg.total_cfg.w_smooth, 1.0e-2)
        self.assertEqual(cfg.total_cfg.data_smoothing_k, 8)
        self.assertTrue(cfg.supervision_contribution_floor_enabled)
        self.assertEqual(cfg.supervision_contribution_floor_ratio, 1.0e-1)
        self.assertNotIn("E_data", cfg.loss_focus_terms)
        self.assertTrue({"E_eq", "E_cn", "E_bc", "E_ct", "E_sigma"}.issubset(set(cfg.loss_focus_terms)))
        self.assertEqual(cfg.out_dir, "./results/ansys_supervised")
        self.assertEqual(cfg.viz_eval_scope, "assembly")
        self.assertFalse(cfg.viz_write_reference_aligned)
        self.assertTrue(cfg.viz_supervision_compare_enabled)
        self.assertEqual(cfg.viz_supervision_compare_split, "test")
        self.assertEqual(cfg.viz_supervision_compare_sources, ("boundary", "corner", "interior"))
        self.assertTrue(cfg.viz_export_final_and_best)
        self.assertTrue(cfg.two_stage_training.enabled)
        self.assertEqual(cfg.two_stage_training.phase1.max_steps, 300)
        self.assertEqual(cfg.two_stage_training.phase1.lr, 2.0e-5)
        self.assertEqual(cfg.two_stage_training.phase1.save_best_on, "val_drrms")
        self.assertEqual(cfg.two_stage_training.phase1.validation_eval_every, 50)
        self.assertEqual(cfg.two_stage_training.phase1.supervision_contribution_floor_ratio, 0.2)
        self.assertEqual(
            cfg.two_stage_training.phase1.base_weights,
            {
                "w_int": 0.25,
                "w_cn": 0.25,
                "w_ct": 0.25,
                "w_bc": 0.25,
                "w_tight": 0.25,
                "w_sigma": 0.25,
                "w_eq": 0.25,
            },
        )
        self.assertEqual(cfg.two_stage_training.phase2.max_steps, 150)
        self.assertEqual(cfg.two_stage_training.phase2.lr, 2.5e-6)
        self.assertEqual(cfg.two_stage_training.phase2.save_best_on, "val_ratio")
        self.assertEqual(cfg.two_stage_training.phase2.validation_eval_every, 50)
        self.assertEqual(cfg.two_stage_training.phase2.supervision_contribution_floor_ratio, 0.1)
        self.assertEqual(
            cfg.two_stage_training.phase2.base_weights,
            {
                "w_int": 0.5,
                "w_cn": 0.5,
                "w_ct": 0.5,
                "w_bc": 0.5,
                "w_tight": 0.5,
                "w_sigma": 0.5,
                "w_eq": 0.5,
            },
        )


if __name__ == "__main__":
    unittest.main()
