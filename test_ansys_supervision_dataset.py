#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for loading staged ANSYS mirror supervision data."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from train.ansys_supervision import assign_group_splits, load_ansys_supervision_dataset
from train.trainer import Trainer


def _build_grouped_case_rows():
    rows = []
    case_idx = 1
    for source, n_groups in (("corner", 8), ("interior", 18), ("boundary", 4)):
        prefix = source[0].upper()
        for group_idx in range(1, n_groups + 1):
            base_id = f"{prefix}G{group_idx:02d}"
            for order_variant in range(1, 7):
                rows.append(
                    {
                        "case_id": f"C{case_idx:03d}",
                        "base_id": base_id,
                        "source": source,
                        "theta_1_deg": 2.0 + float(order_variant),
                        "theta_2_deg": 3.0 + float(group_idx % 3),
                        "theta_3_deg": 4.0 + float(group_idx % 5),
                        "order_1": 1,
                        "order_2": 2,
                        "order_3": 3,
                    }
                )
                case_idx += 1
    return pd.DataFrame(rows)


class AnsysSupervisionDatasetTests(unittest.TestCase):
    def test_loader_maps_case_table_and_stage_csvs_without_csv_split_column(self):
        with tempfile.TemporaryDirectory() as td:
            case_table = os.path.join(td, "cases.csv")
            stage_dir = os.path.join(td, "stages")
            os.makedirs(stage_dir, exist_ok=True)

            rows = []
            for case_idx, (base_id, source) in enumerate(
                (("CG01", "corner"), ("IG01", "interior"), ("IG02", "interior")),
                start=1,
            ):
                rows.append(
                    {
                        "case_id": f"C{case_idx:03d}",
                        "base_id": base_id,
                        "source": source,
                        "theta_1_deg": 6.0,
                        "theta_2_deg": 2.0,
                        "theta_3_deg": 4.0,
                        "order_id": case_idx,
                        "order_1": 2,
                        "order_2": 3,
                        "order_3": 1,
                        "order_str": "2/3/1",
                        "job_name": f"ansys_train_{base_id}",
                    }
                )
                for stage, ux in enumerate((0.1, 0.2, 0.3), start=1):
                    pd.DataFrame(
                        [
                            {"node_id": 101, "dx_mm": ux, "dy_mm": 0.0, "dz_mm": 0.0, "total_mm": abs(ux)},
                            {
                                "node_id": 102,
                                "dx_mm": ux + 1.0,
                                "dy_mm": 0.5,
                                "dz_mm": -0.5,
                                "total_mm": abs(ux + 1.0),
                            },
                        ]
                    ).to_csv(os.path.join(stage_dir, f"{case_idx}_stage{stage}.csv"), index=False)

            pd.DataFrame(rows).to_csv(case_table, index=False)
            asm = SimpleNamespace(
                nodes={
                    101: (10.0, 0.0, 1.0),
                    102: (20.0, 5.0, 2.0),
                }
            )

            dataset = load_ansys_supervision_dataset(
                case_table_path=case_table,
                stage_dir=stage_dir,
                asm=asm,
                splits=("train", "val", "test"),
                stage_count=3,
                shuffle=False,
                seed=7,
                test_group_quotas={"corner": 1},
                cv_n_folds=2,
                cv_fold_index=0,
            )
            case = next(
                case
                for cases in dataset.cases_by_split.values()
                for case in cases
                if case["case_id"] == "C001"
            )

            self.assertEqual(case["case_id"], "C001")
            np.testing.assert_allclose(case["P"], np.asarray([6.0, 2.0, 4.0], dtype=np.float32))
            np.testing.assert_array_equal(case["order"], np.asarray([1, 2, 0], dtype=np.int32))
            self.assertEqual(case["X_obs"].shape, (3, 2, 3))
            self.assertEqual(case["U_obs"].shape, (3, 2, 3))
            np.testing.assert_allclose(
                case["X_obs"][0],
                np.asarray([[10.0, 0.0, 1.0], [20.0, 5.0, 2.0]], dtype=np.float32),
                rtol=0.0,
                atol=0.0,
            )
            np.testing.assert_allclose(
                case["U_obs"][2],
                np.asarray([[0.3, 0.0, 0.0], [1.3, 0.5, -0.5]], dtype=np.float32),
                rtol=0.0,
                atol=0.0,
            )
            self.assertIn(case["split"], ("train", "val", "test"))
            self.assertEqual(sum(dataset.counts().values()), 3)

    def test_assign_group_splits_uses_fixed_test_quota_and_grouped_cv(self):
        df = _build_grouped_case_rows()

        split_map = assign_group_splits(
            df,
            seed=42,
            test_group_quotas={"boundary": 1, "corner": 1, "interior": 3},
            cv_n_folds=5,
            cv_fold_index=2,
        )

        self.assertEqual(len(split_map), 30)
        counts = Counter(split_map.values())
        self.assertEqual(counts["train"], 20)
        self.assertEqual(counts["val"], 5)
        self.assertEqual(counts["test"], 5)

        group_source = (
            df[["base_id", "source"]]
            .drop_duplicates(subset=["base_id"])
            .set_index("base_id")["source"]
            .to_dict()
        )
        test_source_counts = Counter(group_source[group_id] for group_id, split in split_map.items() if split == "test")
        self.assertEqual(test_source_counts["boundary"], 1)
        self.assertEqual(test_source_counts["corner"], 1)
        self.assertEqual(test_source_counts["interior"], 3)

        row_split_counts = Counter(split_map[row["base_id"]] for row in df.to_dict(orient="records"))
        self.assertEqual(row_split_counts["train"], 120)
        self.assertEqual(row_split_counts["val"], 30)
        self.assertEqual(row_split_counts["test"], 30)

    def test_assign_group_splits_partitions_remaining_groups_into_five_equal_folds(self):
        df = _build_grouped_case_rows()

        fold_to_val_groups = []
        fixed_test_groups = None
        for fold_idx in range(5):
            split_map = assign_group_splits(
                df,
                seed=42,
                test_group_quotas={"boundary": 1, "corner": 1, "interior": 3},
                cv_n_folds=5,
                cv_fold_index=fold_idx,
            )
            val_groups = {group_id for group_id, split in split_map.items() if split == "val"}
            test_groups = {group_id for group_id, split in split_map.items() if split == "test"}
            self.assertEqual(len(val_groups), 5)
            self.assertEqual(len(test_groups), 5)
            if fixed_test_groups is None:
                fixed_test_groups = test_groups
            else:
                self.assertEqual(test_groups, fixed_test_groups)
            fold_to_val_groups.append(val_groups)

        union_val_groups = set().union(*fold_to_val_groups)
        self.assertEqual(len(union_val_groups), 25)
        self.assertTrue(all(a.isdisjoint(b) for i, a in enumerate(fold_to_val_groups) for b in fold_to_val_groups[i + 1 :]))
        self.assertTrue(union_val_groups.isdisjoint(fixed_test_groups))

    def test_loader_adds_nondimensional_supervision_features(self):
        with tempfile.TemporaryDirectory() as td:
            case_table = os.path.join(td, "cases.csv")
            stage_dir = os.path.join(td, "stages")
            os.makedirs(stage_dir, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "case_id": "C001",
                        "base_id": "CG01",
                        "source": "corner",
                        "theta_1_deg": 6.0,
                        "theta_2_deg": 2.0,
                        "theta_3_deg": 4.0,
                        "order_1": 1,
                        "order_2": 2,
                        "order_3": 3,
                    }
                ]
            ).to_csv(case_table, index=False)

            stage_displacements = (
                ((0.1, 0.0, 0.0), (0.3, 0.0, 0.0), (1.2, 0.0, 0.0)),
                ((0.2, 0.0, 0.0), (0.6, 0.0, 0.0), (2.1, 0.0, 0.0)),
                ((0.4, 0.0, 0.0), (0.9, 0.0, 0.0), (3.6, 0.0, 0.0)),
            )
            for stage_rank, displacements in enumerate(stage_displacements, start=1):
                pd.DataFrame(
                    [
                        {
                            "node_id": node_id,
                            "dx_mm": disp[0],
                            "dy_mm": disp[1],
                            "dz_mm": disp[2],
                            "total_mm": float(np.linalg.norm(disp)),
                        }
                        for node_id, disp in zip((101, 102, 103), displacements)
                    ]
                ).to_csv(os.path.join(stage_dir, f"1_stage{stage_rank}.csv"), index=False)

            asm = SimpleNamespace(
                nodes={
                    101: (10.0, 0.0, 1.0),
                    102: (20.0, 1.0, 1.5),
                    103: (35.0, 2.0, 2.0),
                }
            )

            dataset = load_ansys_supervision_dataset(
                case_table_path=case_table,
                stage_dir=stage_dir,
                asm=asm,
                stage_count=3,
                shuffle=False,
                seed=7,
                test_group_quotas={},
                cv_n_folds=1,
                cv_fold_index=0,
            )
            case = dataset.cases_by_split["val"][0]

            self.assertEqual(case["X_obs_nd"].shape, case["X_obs"].shape)
            self.assertEqual(case["U_obs_nd"].shape, case["U_obs"].shape)
            self.assertEqual(case["obs_morphology_weight"].shape, case["U_obs"].shape[:2] + (1,))
            self.assertEqual(case["U_obs_delta"].shape, (2, case["U_obs"].shape[1], case["U_obs"].shape[2]))
            self.assertAlmostEqual(float(np.mean(case["obs_morphology_weight"])), 1.0, places=5)

    def test_supervision_load_splits_include_train_and_eval_without_duplicates(self):
        trainer = object.__new__(Trainer)
        trainer.cfg = SimpleNamespace(
            supervision=SimpleNamespace(
                train_splits=("train",),
                eval_splits=("val", "test", "val"),
            )
        )

        splits = trainer._supervision_load_splits()

        self.assertEqual(splits, ("train", "val", "test"))


if __name__ == "__main__":
    unittest.main()
