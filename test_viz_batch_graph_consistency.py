#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for graph-consistent batching behavior in mirror visualization."""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from viz.mirror_viz import _eval_displacement_batched


class _DummyField:
    def __init__(self, n_nodes: int | None):
        self._global_knn_n = n_nodes
        if n_nodes is None:
            self._global_knn_idx = None
        else:
            self._global_knn_idx = np.zeros((n_nodes, 1), dtype=np.int32)


class _FakeTensor:
    def __init__(self, array: np.ndarray):
        self._array = np.asarray(array)
        self.shape = self._array.shape

    def numpy(self) -> np.ndarray:
        return self._array


class _FakeTF:
    float32 = np.float32

    @staticmethod
    def convert_to_tensor(value, dtype=None):
        arr = np.asarray(value, dtype=dtype or np.float32)
        return _FakeTensor(arr)

    @staticmethod
    def zeros(shape, dtype=None):
        arr = np.zeros(shape, dtype=dtype or np.float32)
        return _FakeTensor(arr)


class _DummyModel:
    def __init__(self, n_nodes: int | None):
        self.field = _DummyField(n_nodes)
        self.calls: list[int] = []

    def u_fn(self, x, params):
        del params
        self.calls.append(int(x.shape[0]))
        return _FakeTF.zeros((x.shape[0], 3), dtype=np.float32)


class VizBatchGraphConsistencyTests(unittest.TestCase):
    def test_prefers_full_graph_eval_when_cached_graph_matches_node_count(self):
        model = _DummyModel(n_nodes=10)
        points = np.random.randn(10, 3).astype(np.float64)

        with mock.patch.dict(sys.modules, {"tensorflow": _FakeTF}):
            out = _eval_displacement_batched(
                model.u_fn,
                {"P_hat": np.zeros((1, 3), dtype=np.float32)},
                points,
                batch_size=3,
            )

        self.assertEqual(out.shape, (10, 3))
        self.assertEqual(model.calls, [10])

    def test_uses_chunked_eval_when_cached_graph_not_available(self):
        model = _DummyModel(n_nodes=None)
        points = np.random.randn(10, 3).astype(np.float64)

        with mock.patch.dict(sys.modules, {"tensorflow": _FakeTF}):
            out = _eval_displacement_batched(
                model.u_fn,
                {"P_hat": np.zeros((1, 3), dtype=np.float32)},
                points,
                batch_size=3,
            )

        self.assertEqual(out.shape, (10, 3))
        self.assertEqual(model.calls, [3, 3, 3, 1])


if __name__ == "__main__":
    unittest.main()
