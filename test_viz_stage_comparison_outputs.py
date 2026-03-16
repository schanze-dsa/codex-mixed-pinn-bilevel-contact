#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for stage comparison outputs in visualization mixin."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from train.trainer_viz_mixin import TrainerVizMixin


def _write_viz_txt(path: str, umag: np.ndarray) -> None:
    # columns: node_id x y z u_x u_y u_z |u| u_plane v_plane
    nodes = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    x = np.array([-1.0, 1.0, 1.0, -1.0, 0.5, -0.5], dtype=np.float64)
    y = np.array([-1.0, -1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    z = np.zeros_like(x)
    ux = np.zeros_like(x)
    uy = np.zeros_like(x)
    uz = umag.astype(np.float64)
    u_plane = x.copy()
    v_plane = y.copy()

    with open(path, "w", encoding="utf-8") as f:
        f.write("# synthetic viz data\n")
        f.write("# columns: node_id x y z u_x u_y u_z |u| u_plane v_plane\n")
        for i in range(nodes.shape[0]):
            f.write(
                f"{int(nodes[i]):10d} "
                f"{x[i]: .8f} {y[i]: .8f} {z[i]: .8f} "
                f"{ux[i]: .8f} {uy[i]: .8f} {uz[i]: .8f} "
                f"{umag[i]: .8f} {u_plane[i]: .8f} {v_plane[i]: .8f}\n"
            )


class _DummyTrainer(TrainerVizMixin):
    def __init__(self, out_dir: str):
        self.cfg = SimpleNamespace(
            out_dir=out_dir,
            viz_units="mm",
            viz_colormap="turbo",
            viz_compare_cmap="coolwarm",
            viz_refine_subdivisions=0,
            viz_refine_max_points=None,
            viz_use_shape_function_interp=True,
            viz_smooth_scalar_iters=0,
            viz_smooth_scalar_lambda=0.6,
        )


class VizStageComparisonOutputTests(unittest.TestCase):
    def test_build_stage_comparison_mesh_prefers_surface_topology_when_available(self):
        trainer = _DummyTrainer(out_dir=".")
        trainer.cfg.mirror_surface_name = "MIRROR UP"
        trainer.cfg.viz_surface_source = "part_top"
        trainer.asm = SimpleNamespace(
            surfaces={"asm::mirror": SimpleNamespace(name="MIRROR UP")},
            parts={
                "MIRROR": SimpleNamespace(
                    nodes_xyz={
                        10: (-1.0, 0.0, 0.0),
                        20: (1.0, 0.0, 0.0),
                        30: (0.0, 1.0, 0.0),
                    }
                )
            },
            nodes={
                10: (-1.0, 0.0, 0.0),
                20: (1.0, 0.0, 0.0),
                30: (0.0, 1.0, 0.0),
            },
        )

        tri_surface = SimpleNamespace(
            part_name="MIRROR",
            tri_node_ids=np.asarray([[10, 20, 30]], dtype=np.int64),
            tri_elem_ids=np.asarray([1], dtype=np.int64),
            tri_face_labels=["f1"],
            name="mirror",
        )

        mesh_builder = getattr(trainer, "_build_stage_comparison_mesh", None)
        self.assertTrue(callable(mesh_builder))

        with patch("train.trainer_viz_mixin.resolve_surface_to_tris", return_value=tri_surface, create=True), patch(
            "train.trainer_viz_mixin.triangulate_part_boundary",
            return_value=tri_surface,
            create=True,
        ), patch(
            "train.trainer_viz_mixin.compute_tri_geometry",
            return_value=(
                np.asarray([1.0], dtype=np.float64),
                np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
                None,
            ),
            create=True,
        ):
            mesh = mesh_builder(np.asarray([10, 20, 30], dtype=np.int64))

        self.assertIsNotNone(mesh)
        uv, tri_idx = mesh
        self.assertEqual(tuple(uv.shape), (3, 2))
        np.testing.assert_array_equal(tri_idx, np.asarray([[0, 1, 2]], dtype=np.int64))

    def test_build_stage_comparison_mesh_supports_assembly_level_surface(self):
        trainer = _DummyTrainer(out_dir=".")
        trainer.cfg.mirror_surface_name = "MIRROR UP"
        trainer.cfg.viz_surface_source = "part_top"
        trainer.asm = SimpleNamespace(
            surfaces={"asm::mirror": SimpleNamespace(name="MIRROR UP")},
            parts={},
            nodes={
                10: (-1.0, 0.0, 0.0),
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

        with patch("train.trainer_viz_mixin.resolve_surface_to_tris", return_value=tri_surface, create=True):
            mesh = trainer._build_stage_comparison_mesh(np.asarray([10, 20, 30], dtype=np.int64))

        self.assertIsNotNone(mesh)
        uv, tri_idx = mesh
        self.assertEqual(tuple(uv.shape), (3, 2))
        np.testing.assert_array_equal(tri_idx, np.asarray([[0, 1, 2]], dtype=np.int64))

    def test_stage_comparison_render_state_refines_exported_stage_field(self):
        trainer = _DummyTrainer(out_dir=".")
        trainer.cfg.mirror_surface_name = "MIRROR UP"
        trainer.cfg.viz_surface_source = "part_top"
        trainer.cfg.viz_refine_subdivisions = 2
        trainer.asm = SimpleNamespace(
            surfaces={"asm::mirror": SimpleNamespace(name="MIRROR UP")},
            parts={
                "MIRROR": SimpleNamespace(
                    nodes_xyz={
                        10: (-1.0, 0.0, 0.0),
                        20: (1.0, 0.0, 0.0),
                        30: (0.0, 1.0, 0.0),
                    }
                )
            },
            nodes={
                10: (-1.0, 0.0, 0.0),
                20: (1.0, 0.0, 0.0),
                30: (0.0, 1.0, 0.0),
            },
        )

        aligned_sample = {
            "node_id": np.asarray([10, 20, 30], dtype=np.int64),
            "x": np.asarray([-1.0, 1.0, 0.0], dtype=np.float64),
            "y": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            "z": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            "ux": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            "uy": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            "uz": np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            "umag": np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            "u_plane": np.asarray([-1.0, 1.0, 0.0], dtype=np.float64),
            "v_plane": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        }
        tri_surface = SimpleNamespace(
            part_name="MIRROR",
            tri_node_ids=np.asarray([[10, 20, 30]], dtype=np.int64),
            tri_elem_ids=np.asarray([1], dtype=np.int64),
            tri_face_labels=["f1"],
            name="mirror",
        )

        with patch("train.trainer_viz_mixin.resolve_surface_to_tris", return_value=tri_surface, create=True), patch(
            "train.trainer_viz_mixin.triangulate_part_boundary",
            return_value=tri_surface,
            create=True,
        ), patch(
            "train.trainer_viz_mixin.compute_tri_geometry",
            return_value=(
                np.asarray([1.0], dtype=np.float64),
                np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
                None,
            ),
            create=True,
        ):
            render = trainer._build_stage_comparison_render_state(
                np.asarray([10, 20, 30], dtype=np.int64),
                aligned_sample,
            )

        self.assertIsNotNone(render)
        tri, d_plot = render
        self.assertGreater(tri.x.shape[0], 3)
        self.assertEqual(int(tri.triangles.shape[0]), 4)
        self.assertEqual(int(d_plot.shape[0]), int(tri.x.shape[0]))

    def test_generates_common_scale_and_delta_outputs_for_stages(self):
        with tempfile.TemporaryDirectory() as td:
            trainer = _DummyTrainer(out_dir=td)
            base_png = os.path.join(td, "deflection_01_321.png")
            open(base_png, "wb").close()

            s1 = os.path.join(td, "deflection_01_321_s1.txt")
            s2 = os.path.join(td, "deflection_01_321_s2.txt")
            s3 = os.path.join(td, "deflection_01_321_s3.txt")
            _write_viz_txt(s1, np.array([1.0, 1.1, 1.2, 1.3, 1.15, 1.05]) * 1.0e-3)
            _write_viz_txt(s2, np.array([1.1, 1.2, 1.25, 1.35, 1.20, 1.10]) * 1.0e-3)
            _write_viz_txt(s3, np.array([1.2, 1.25, 1.30, 1.40, 1.25, 1.15]) * 1.0e-3)

            stage_records = [
                {"stage_rank": 1, "data_path": s1, "png_path": os.path.join(td, "deflection_01_321_s1.png")},
                {"stage_rank": 2, "data_path": s2, "png_path": os.path.join(td, "deflection_01_321_s2.png")},
                {"stage_rank": 3, "data_path": s3, "png_path": os.path.join(td, "deflection_01_321_s3.png")},
            ]

            report = trainer._write_stage_comparison_for_case(base_png, stage_records)

            self.assertTrue(isinstance(report, str) and os.path.exists(report))
            self.assertTrue(os.path.exists(os.path.join(td, "deflection_01_321_s1_common.png")))
            self.assertTrue(os.path.exists(os.path.join(td, "deflection_01_321_s2_common.png")))
            self.assertTrue(os.path.exists(os.path.join(td, "deflection_01_321_s3_common.png")))
            self.assertTrue(os.path.exists(os.path.join(td, "deflection_01_321_s2_minus_s1.png")))
            self.assertTrue(os.path.exists(os.path.join(td, "deflection_01_321_s3_minus_s2.png")))

            with open(report, "r", encoding="utf-8") as f:
                txt = f.read()
            self.assertIn("s2-s1", txt)
            self.assertIn("s3-s2", txt)


if __name__ == "__main__":
    unittest.main()
