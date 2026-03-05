#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for vector-field denoising on triangulated surfaces."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from viz.mirror_viz import _build_vertex_adjacency, _smooth_vector_on_tri_mesh


def _build_grid_mesh(nx: int, ny: int):
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    def vid(i: int, j: int) -> int:
        return i * ny + j

    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            tris.append([v00, v10, v01])
            tris.append([v10, v11, v01])
    return pts, np.asarray(tris, dtype=np.int32)


def _roughness(values: np.ndarray, triangles: np.ndarray) -> float:
    n = int(values.shape[0])
    adj = _build_vertex_adjacency(triangles, n)
    diff_sum = 0.0
    count = 0
    for i, nbrs in enumerate(adj):
        if not nbrs:
            continue
        mean_nbr = np.mean(values[list(nbrs)], axis=0)
        diff_sum += float(np.linalg.norm(values[i] - mean_nbr))
        count += 1
    return diff_sum / max(count, 1)


class VizVectorSmoothingTests(unittest.TestCase):
    def test_vector_smoothing_reduces_roughness_and_error(self):
        rng = np.random.default_rng(20260305)
        pts, tri = _build_grid_mesh(22, 18)
        x = pts[:, 0]
        y = pts[:, 1]

        clean = np.stack(
            [
                0.0015 * np.sin(2.0 * np.pi * x) * np.cos(1.2 * np.pi * y),
                0.0012 * np.cos(1.6 * np.pi * x + 0.2) * np.sin(2.1 * np.pi * y),
                0.0008 * np.sin(1.4 * np.pi * x + 0.4) * np.sin(1.8 * np.pi * y),
            ],
            axis=1,
        )
        noise = 4.0e-4 * rng.standard_normal(size=clean.shape)
        noisy = clean + noise

        rough_before = _roughness(noisy, tri)
        mae_before = float(np.mean(np.abs(noisy - clean)))

        denoised = _smooth_vector_on_tri_mesh(noisy, tri, iterations=6, lam=0.45, preserve_mean=True)

        rough_after = _roughness(denoised, tri)
        mae_after = float(np.mean(np.abs(denoised - clean)))

        self.assertLess(rough_after, 0.65 * rough_before)
        self.assertLess(mae_after, mae_before)
        self.assertTrue(np.all(np.isfinite(denoised)))

    def test_zero_iterations_keeps_input(self):
        _, tri = _build_grid_mesh(8, 7)
        vals = np.arange(8 * 7 * 3, dtype=np.float64).reshape(-1, 3)
        out = _smooth_vector_on_tri_mesh(vals, tri, iterations=0, lam=0.5, preserve_mean=True)
        self.assertTrue(np.array_equal(out, vals))


if __name__ == "__main__":
    unittest.main()
