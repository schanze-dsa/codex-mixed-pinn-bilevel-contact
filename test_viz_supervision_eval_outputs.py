#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for supervision evaluation exports in visualization mixin."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from train.trainer_preload_mixin import TrainerPreloadMixin
from train.trainer_viz_mixin import TrainerVizMixin


class _DummyTrainer(TrainerVizMixin, TrainerPreloadMixin):
    def __init__(self, out_dir: str):
        self.cfg = SimpleNamespace(
            out_dir=out_dir,
            preload_use_stages=True,
            preload_append_release_stage=False,
            total_cfg=SimpleNamespace(preload_stage_mode="force_then_lock"),
            model_cfg=SimpleNamespace(preload_shift=0.0, preload_scale=1.0),
            supervision=SimpleNamespace(
                enabled=True,
                eval_splits=("val",),
                export_eval_reports=True,
                export_eval_plots=True,
            ),
            viz_force_pointwise=False,
            viz_surface_enabled=True,
            viz_title_prefix="Total Deformation (trained PINN)",
            viz_style="smooth",
            viz_colormap="turbo",
            viz_levels=64,
            viz_symmetric=False,
            viz_draw_wireframe=False,
            viz_units="mm",
            viz_write_data=True,
            viz_write_surface_mesh=False,
            viz_plot_full_structure=False,
            viz_full_structure_part=None,
            viz_write_full_structure_data=False,
            viz_refine_subdivisions=0,
            viz_refine_max_points=None,
            viz_use_shape_function_interp=True,
            viz_remove_rigid=True,
            viz_smooth_vector_iters=0,
            viz_smooth_vector_lambda=0.35,
            viz_smooth_scalar_iters=0,
            viz_smooth_scalar_lambda=0.6,
            viz_retriangulate_2d=False,
            viz_eval_batch_size=1024,
            viz_eval_scope="assembly",
            viz_diagnose_blanks=False,
            viz_auto_fill_blanks=False,
            viz_surface_source="part_top",
            mirror_surface_name="MIRROR UP",
            viz_plot_stages=True,
            viz_skip_release_stage_plot=False,
            viz_same_pipeline_supervision_debug=False,
            viz_export_final_and_best=False,
            viz_supervision_compare_enabled=False,
            viz_supervision_compare_split="test",
            viz_supervision_compare_sources=("boundary", "corner", "interior"),
        )
        self.asm = None
        self.model = SimpleNamespace(
            u_fn=lambda x, params=None: tf.cast(params["U_obs"], tf.float32)
            + tf.constant([0.1, 0.0, 0.0], dtype=tf.float32)
        )


def _make_case(case_id: str, offset: float) -> dict:
    x_obs = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]],
        ],
        dtype=np.float32,
    )
    u_obs = np.asarray(
        [
            [[offset + 0.0, 0.0, 0.0], [offset + 0.1, 0.0, 0.0]],
            [[offset + 0.2, 0.0, 0.0], [offset + 0.3, 0.0, 0.0]],
            [[offset + 0.4, 0.0, 0.0], [offset + 0.5, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    return {
        "case_id": case_id,
        "base_id": "B01",
        "split": "val",
        "source": "corner",
        "job_name": f"job_{case_id}",
        "P": np.asarray([2.0, 4.0, 6.0], dtype=np.float32),
        "order": np.asarray([2, 0, 1], dtype=np.int32),
        "X_obs": x_obs,
        "U_obs": u_obs,
    }


def _make_surface_case(case_id: str, source: str, split: str = "test") -> dict:
    xy = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    x_obs = np.stack([xy, xy, xy], axis=0)
    u0 = np.asarray(
        [
            [0.00, 0.00, 0.00],
            [0.05, 0.00, 0.00],
            [0.05, 0.05, 0.00],
            [0.00, 0.05, 0.00],
        ],
        dtype=np.float32,
    )
    u1 = 1.5 * u0
    u2 = 2.0 * u0
    return {
        "case_id": case_id,
        "base_id": f"B_{case_id}",
        "split": split,
        "source": source,
        "job_name": f"job_{case_id}",
        "P": np.asarray([2.0, 4.0, 6.0], dtype=np.float32),
        "order": np.asarray([2, 0, 1], dtype=np.int32),
        "node_ids": np.asarray([11, 12, 13, 14], dtype=np.int64),
        "X_obs": x_obs,
        "U_obs": np.stack([u0, u1, u2], axis=0),
    }


class VizSupervisionEvalOutputTests(unittest.TestCase):
    def test_supervision_render_state_prefers_surface_topology_for_ring_mesh(self):
        trainer = _DummyTrainer(out_dir=".")
        trainer.cfg.viz_remove_rigid = False
        trainer.asm = SimpleNamespace(
            surfaces={"asm::mirror": SimpleNamespace(name="MIRROR UP")},
            parts={},
            nodes={
                1: (-2.0, -2.0, 0.0),
                2: (2.0, -2.0, 0.0),
                3: (2.0, 2.0, 0.0),
                4: (-2.0, 2.0, 0.0),
                5: (-1.0, -1.0, 0.0),
                6: (1.0, -1.0, 0.0),
                7: (1.0, 1.0, 0.0),
                8: (-1.0, 1.0, 0.0),
            },
        )
        tri_surface = SimpleNamespace(
            part_name="_ASM_",
            tri_node_ids=np.asarray(
                [
                    [1, 2, 6],
                    [1, 6, 5],
                    [2, 3, 7],
                    [2, 7, 6],
                    [3, 4, 8],
                    [3, 8, 7],
                    [4, 1, 5],
                    [4, 5, 8],
                ],
                dtype=np.int64,
            ),
            tri_elem_ids=np.arange(8, dtype=np.int64),
            tri_face_labels=["f"] * 8,
            name="mirror",
        )
        xyz = np.asarray([trainer.asm.nodes[i] for i in range(1, 9)], dtype=np.float64)
        u_vec = np.zeros((8, 3), dtype=np.float64)

        from unittest.mock import patch

        with patch("train.trainer_viz_mixin.resolve_surface_to_tris", return_value=tri_surface, create=True):
            render = trainer._build_supervision_compare_render_state(
                np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64),
                xyz,
                u_vec,
            )

        self.assertIsNotNone(render)
        tri, u_plot, d_plot = render
        self.assertEqual(tuple(tri.triangles.shape), (8, 3))
        self.assertEqual(tuple(u_plot.shape), (8, 3))
        self.assertEqual(tuple(d_plot.shape), (8,))

    def test_supervision_render_state_removes_rigid_translation(self):
        trainer = _DummyTrainer(out_dir=".")
        trainer.cfg.viz_remove_rigid = True
        trainer.asm = SimpleNamespace(
            surfaces={"asm::mirror": SimpleNamespace(name="MIRROR UP")},
            parts={},
            nodes={
                10: (0.0, 0.0, 0.0),
                20: (1.0, 0.0, 0.0),
                30: (0.0, 1.0, 0.0),
            },
        )
        tri_surface = SimpleNamespace(
            part_name="_ASM_",
            tri_node_ids=np.asarray([[10, 20, 30]], dtype=np.int64),
            tri_elem_ids=np.asarray([1], dtype=np.int64),
            tri_face_labels=["f1"],
            name="mirror",
        )
        xyz = np.asarray([trainer.asm.nodes[i] for i in (10, 20, 30)], dtype=np.float64)
        u_vec = np.tile(np.asarray([[0.5, -0.25, 0.1]], dtype=np.float64), (3, 1))

        from unittest.mock import patch

        with patch("train.trainer_viz_mixin.resolve_surface_to_tris", return_value=tri_surface, create=True):
            render = trainer._build_supervision_compare_render_state(
                np.asarray([10, 20, 30], dtype=np.int64),
                xyz,
                u_vec,
            )

        self.assertIsNotNone(render)
        _, _, d_plot = render
        self.assertLess(float(np.max(np.abs(d_plot))), 1.0e-10)

    def test_exports_case_stage_metrics_csv_and_heatmap(self):
        with tempfile.TemporaryDirectory() as td:
            trainer = _DummyTrainer(out_dir=td)
            trainer._supervision_dataset = SimpleNamespace(
                cases_by_split={
                    "val": [
                        _make_case("C001", 0.0),
                        _make_case("C002", 1.0),
                    ]
                }
            )

            outputs = trainer._write_supervision_eval_outputs()

            csv_path = os.path.join(td, "supervision_eval_val.csv")
            heatmap_path = os.path.join(td, "supervision_eval_val_rmse_vec.png")
            self.assertIn(csv_path, outputs)
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(heatmap_path))

            with open(csv_path, "r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))

            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["case_id"], "C001")
            self.assertEqual(rows[0]["stage_rank"], "1")
            self.assertAlmostEqual(float(rows[0]["rmse_x_mm"]), 0.1, places=6)
            self.assertAlmostEqual(float(rows[0]["rmse_vec_mm"]), 0.1, places=6)
            self.assertEqual(rows[-1]["case_id"], "C002")
            self.assertEqual(rows[-1]["stage_rank"], "3")

    def test_selects_median_nearest_final_stage_case_per_source(self):
        trainer = _DummyTrainer(out_dir=".")
        rows = [
            {"case_id": "CB1", "source": "boundary", "stage_rank": 1, "rmse_vec_mm": 9.0},
            {"case_id": "CB1", "source": "boundary", "stage_rank": 3, "rmse_vec_mm": 0.4},
            {"case_id": "CB2", "source": "boundary", "stage_rank": 3, "rmse_vec_mm": 0.6},
            {"case_id": "CB3", "source": "boundary", "stage_rank": 3, "rmse_vec_mm": 0.8},
            {"case_id": "CC1", "source": "corner", "stage_rank": 3, "rmse_vec_mm": 0.2},
            {"case_id": "CC2", "source": "corner", "stage_rank": 3, "rmse_vec_mm": 0.5},
            {"case_id": "CC3", "source": "corner", "stage_rank": 3, "rmse_vec_mm": 0.7},
            {"case_id": "CI1", "source": "interior", "stage_rank": 3, "rmse_vec_mm": 1.0},
            {"case_id": "CI2", "source": "interior", "stage_rank": 3, "rmse_vec_mm": 1.2},
            {"case_id": "CI3", "source": "interior", "stage_rank": 3, "rmse_vec_mm": 3.0},
        ]

        selected = trainer._select_representative_supervision_rows(
            rows,
            sources=("boundary", "corner", "interior"),
        )

        self.assertEqual(
            [(row["source"], row["case_id"]) for row in selected],
            [("boundary", "CB2"), ("corner", "CC2"), ("interior", "CI2")],
        )

    def test_exports_representative_triptych_figures_for_test_split(self):
        with tempfile.TemporaryDirectory() as td:
            trainer = _DummyTrainer(out_dir=td)
            trainer.cfg.supervision.eval_splits = ("test",)
            trainer.cfg.viz_supervision_compare_enabled = True
            trainer.cfg.viz_supervision_compare_split = "test"
            trainer._supervision_dataset = SimpleNamespace(
                cases_by_split={
                    "test": [
                        _make_surface_case("C301", "boundary"),
                        _make_surface_case("C302", "corner"),
                        _make_surface_case("C303", "interior"),
                    ]
                }
            )

            outputs = trainer._write_supervision_eval_outputs()

            csv_path = os.path.join(td, "supervision_eval_test.csv")
            heatmap_path = os.path.join(td, "supervision_eval_test_rmse_vec.png")
            summary_path = os.path.join(td, "supervision_compare_selected_cases.csv")
            self.assertIn(csv_path, outputs)
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(heatmap_path))
            self.assertTrue(os.path.exists(summary_path))
            self.assertTrue(os.path.exists(os.path.join(td, "supervision_compare_boundary_C301.png")))
            self.assertTrue(os.path.exists(os.path.join(td, "supervision_compare_corner_C302.png")))
            self.assertTrue(os.path.exists(os.path.join(td, "supervision_compare_interior_C303.png")))

            with open(summary_path, "r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))

            self.assertEqual(len(rows), 3)
            self.assertEqual({row["case_id"] for row in rows}, {"C301", "C302", "C303"})
            self.assertEqual({row["source"] for row in rows}, {"boundary", "corner", "interior"})

    def test_matches_supervision_case_by_preload_and_order(self):
        trainer = _DummyTrainer(out_dir=".")
        trainer._supervision_dataset = SimpleNamespace(
            cases_by_split={
                "train": [
                    _make_surface_case("C401", "boundary", split="train"),
                ],
                "test": [
                    {
                        **_make_surface_case("C402", "corner", split="test"),
                        "P": np.asarray([2.0, 4.0, 6.0], dtype=np.float32),
                        "order": np.asarray([1, 2, 0], dtype=np.int32),
                    },
                ],
            }
        )
        preload_case = {
            "P": np.asarray([2.0, 4.0, 6.0], dtype=np.float32),
            "order": np.asarray([1, 2, 0], dtype=np.int32),
        }

        matched = trainer._match_supervision_case_for_preload(preload_case)

        self.assertIsNotNone(matched)
        self.assertEqual(matched["case_id"], "C402")

    def test_exports_same_pipeline_debug_pairs_for_matching_supervision_case(self):
        with tempfile.TemporaryDirectory() as td:
            trainer = _DummyTrainer(out_dir=td)
            trainer.cfg.viz_same_pipeline_supervision_debug = True
            trainer.asm = SimpleNamespace(
                surfaces={"asm::mirror": SimpleNamespace(name="MIRROR UP")},
                parts={
                    "MIRROR": SimpleNamespace(
                        nodes_xyz={
                            11: (0.0, 0.0, 0.0),
                            12: (1.0, 0.0, 0.0),
                            13: (1.0, 1.0, 0.0),
                            14: (0.0, 1.0, 0.0),
                        },
                        node_ids=[11, 12, 13, 14],
                    )
                },
                nodes={
                    11: (0.0, 0.0, 0.0),
                    12: (1.0, 0.0, 0.0),
                    13: (1.0, 1.0, 0.0),
                    14: (0.0, 1.0, 0.0),
                },
            )
            sup_case = {
                **_make_surface_case("C501", "interior", split="test"),
                "P": np.asarray([2.0, 4.0, 6.0], dtype=np.float32),
                "order": np.asarray([1, 2, 0], dtype=np.int32),
            }
            trainer._supervision_dataset = SimpleNamespace(cases_by_split={"test": [sup_case]})
            preload_case = {
                "P": np.asarray([2.0, 4.0, 6.0], dtype=np.float32),
                "order": np.asarray([1, 2, 0], dtype=np.int32),
            }
            preload_case.update(trainer._build_stage_case(preload_case["P"], preload_case["order"]))
            params_full = trainer._make_preload_params(preload_case)

            calls = []

            def _fake_plot(asm, mirror_surface_bare_name, u_fn, params, P_values=None, out_path=None, data_out_path=None, **kwargs):
                calls.append(
                    {
                        "out_path": out_path,
                        "n_nodes": len(getattr(asm, "nodes", {})),
                        "mirror_surface_bare_name": mirror_surface_bare_name,
                    }
                )
                if out_path:
                    open(out_path, "wb").close()
                data_path = None
                if out_path:
                    data_path = os.path.splitext(out_path)[0] + ".txt"
                    with open(data_path, "w", encoding="utf-8") as fp:
                        fp.write("# same-pipeline debug\n")
                return None, None, data_path

            with patch("train.trainer_viz_mixin.plot_mirror_deflection_by_name", side_effect=_fake_plot):
                exported = trainer._write_same_pipeline_supervision_debug_exports(
                    case_index=1,
                    preload_case=preload_case,
                    params_full=params_full,
                    suffix="_231",
                    title_prefix="debug title",
                )

            self.assertEqual(len(calls), 8)
            self.assertTrue(all(call["n_nodes"] == 4 for call in calls))
            self.assertEqual({call["mirror_surface_bare_name"] for call in calls}, {"MIRROR UP"})
            self.assertIn(os.path.join(td, "deflection_01_231_samepipe_pinn.png"), exported)
            self.assertIn(os.path.join(td, "deflection_01_231_samepipe_fem.png"), exported)
            self.assertIn(os.path.join(td, "deflection_01_231_s1_samepipe_pinn.png"), exported)
            self.assertIn(os.path.join(td, "deflection_01_231_s1_samepipe_fem.png"), exported)
            self.assertTrue(os.path.exists(os.path.join(td, "deflection_01_231_samepipe_pinn.png")))
            self.assertTrue(os.path.exists(os.path.join(td, "deflection_01_231_samepipe_fem.png")))
            self.assertTrue(os.path.exists(os.path.join(td, "deflection_01_231_s3_samepipe_pinn.png")))
            self.assertTrue(os.path.exists(os.path.join(td, "deflection_01_231_s3_samepipe_fem.png")))

    def test_resolves_dual_export_targets_for_final_and_best(self):
        with tempfile.TemporaryDirectory() as td:
            trainer = _DummyTrainer(out_dir=td)
            trainer.cfg.viz_export_final_and_best = True
            trainer._final_ckpt_path = os.path.join("checkpoints", "run", "ckpt-1500")
            trainer._best_ckpt_path = os.path.join("checkpoints", "run", "ckpt-100")

            targets = trainer._resolve_visual_export_targets()

            self.assertEqual(len(targets), 2)
            self.assertEqual(targets[0]["tag"], "final")
            self.assertEqual(targets[0]["ckpt_path"], trainer._final_ckpt_path)
            self.assertEqual(targets[0]["out_dir"], os.path.join(td, "final"))
            self.assertEqual(targets[1]["tag"], "best")
            self.assertEqual(targets[1]["ckpt_path"], trainer._best_ckpt_path)
            self.assertEqual(targets[1]["out_dir"], os.path.join(td, "best"))

    def test_visualize_after_training_exports_final_and_best_sets(self):
        with tempfile.TemporaryDirectory() as td:
            trainer = _DummyTrainer(out_dir=td)
            trainer.cfg.viz_export_final_and_best = True
            trainer._final_ckpt_path = os.path.join("checkpoints", "run", "ckpt-1500")
            trainer._best_ckpt_path = os.path.join("checkpoints", "run", "ckpt-100")

            restores = []
            exports = []

            trainer._restore_checkpoint_for_export = restores.append  # type: ignore[method-assign]
            trainer._visualize_current_state_to_out_dir = (  # type: ignore[attr-defined]
                lambda n_samples=5: exports.append((trainer.cfg.out_dir, n_samples))
            )

            trainer._visualize_after_training(n_samples=2)

            self.assertEqual(
                restores,
                [
                    trainer._final_ckpt_path,
                    trainer._best_ckpt_path,
                    trainer._final_ckpt_path,
                ],
            )
            self.assertEqual(
                exports,
                [
                    (os.path.join(td, "final"), 2),
                    (os.path.join(td, "best"), 2),
                ],
            )
            self.assertEqual(trainer.cfg.out_dir, td)


if __name__ == "__main__":
    unittest.main()
