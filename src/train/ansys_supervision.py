#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load staged ANSYS mirror supervision cases for PINN training."""

from __future__ import annotations

import copy
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

_DEFAULT_SPLIT_NAMES: Tuple[str, str, str] = ("train", "val", "test")
_DEFAULT_TEST_GROUP_QUOTAS: Dict[str, int] = {
    "boundary": 1,
    "corner": 1,
    "interior": 3,
}


def _extract_case_index(case_id: Any) -> int:
    text = str(case_id or "").strip()
    match = re.search(r"(\d+)", text)
    if match is None:
        raise ValueError(f"Cannot extract numeric case index from case_id={case_id!r}")
    return int(match.group(1))


def _resolve_stage_csv_path(stage_dir: str, case_id: Any, stage_rank: int) -> str:
    case_idx = _extract_case_index(case_id)
    candidates = [
        os.path.join(stage_dir, f"{case_idx}_stage{stage_rank}.csv"),
        os.path.join(stage_dir, f"{str(case_id).strip()}_stage{stage_rank}.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Stage CSV not found for case_id={case_id!r}, stage={stage_rank}. Tried: {candidates}"
    )


def _copy_case(case: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in case.items():
        if isinstance(value, np.ndarray):
            out[key] = value.copy()
        else:
            out[key] = copy.deepcopy(value)
    return out


def _stable_positive_scale(value: Any, *, fallback: float = 1.0) -> float:
    scale = float(value)
    if not np.isfinite(scale) or scale <= 1.0e-12:
        return float(fallback)
    return scale


def _resolve_case_reference_scales(
    row: Mapping[str, Any],
    *,
    X_obs: np.ndarray,
    U_obs: np.ndarray,
) -> Tuple[float, float]:
    x_flat = np.asarray(X_obs, dtype=np.float32).reshape(-1, X_obs.shape[-1])
    u_flat = np.asarray(U_obs, dtype=np.float32).reshape(-1, U_obs.shape[-1])

    x_extent = float(np.max(np.ptp(x_flat, axis=0))) if x_flat.size else 0.0
    x_norm = float(np.max(np.linalg.norm(x_flat, axis=1))) if x_flat.size else 0.0
    u_norm = float(np.max(np.linalg.norm(u_flat, axis=1))) if u_flat.size else 0.0

    L_ref = _stable_positive_scale(
        row.get("L_ref", 0.0),
        fallback=_stable_positive_scale(x_extent, fallback=_stable_positive_scale(x_norm)),
    )
    u_ref = _stable_positive_scale(row.get("u_ref", 0.0), fallback=_stable_positive_scale(u_norm))
    return (L_ref, u_ref)


def _to_nondimensional_tensor(values: np.ndarray, scale: float) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32) / _stable_positive_scale(scale)).astype(np.float32)


def _build_obs_morphology_weights(U_obs_nd: np.ndarray) -> np.ndarray:
    magnitudes = np.linalg.norm(np.asarray(U_obs_nd, dtype=np.float32), axis=-1, keepdims=True)
    contrast = np.zeros_like(magnitudes, dtype=np.float32)
    neighbor_count = np.zeros_like(magnitudes, dtype=np.float32)
    if magnitudes.shape[1] > 1:
        forward_diff = np.abs(magnitudes[:, 1:, :] - magnitudes[:, :-1, :])
        contrast[:, :-1, :] += forward_diff
        contrast[:, 1:, :] += forward_diff
        neighbor_count[:, :-1, :] += 1.0
        neighbor_count[:, 1:, :] += 1.0
    contrast = contrast / np.maximum(neighbor_count, 1.0)

    raw = 1.0 + magnitudes + contrast
    mean_weight = float(np.mean(raw))
    return (raw / _stable_positive_scale(mean_weight)).astype(np.float32)


def _compute_stage_displacement_deltas(U_obs_nd: np.ndarray) -> np.ndarray:
    if int(U_obs_nd.shape[0]) <= 1:
        return np.zeros((0,) + tuple(U_obs_nd.shape[1:]), dtype=np.float32)
    return np.diff(np.asarray(U_obs_nd, dtype=np.float32), axis=0).astype(np.float32)


def _validated_split_names(split_names: Sequence[str]) -> Tuple[str, str, str]:
    if tuple(split_names) != _DEFAULT_SPLIT_NAMES:
        raise ValueError(f"Unsupported split_names={tuple(split_names)!r}; expected {_DEFAULT_SPLIT_NAMES!r}")
    return tuple(str(x).strip() for x in split_names)


def _prepare_group_metadata(
    case_table: pd.DataFrame,
    *,
    group_key: str,
    stratify_key: Optional[str],
) -> pd.DataFrame:
    group_cols = [group_key]
    if stratify_key:
        group_cols.append(stratify_key)
    group_meta = case_table[group_cols].copy()
    group_meta[group_key] = group_meta[group_key].astype(str).str.strip()
    if stratify_key:
        group_meta[stratify_key] = group_meta[stratify_key].astype(str).str.strip()
        nunique = group_meta.groupby(group_key, dropna=False)[stratify_key].nunique(dropna=False)
        bad = nunique[nunique > 1]
        if not bad.empty:
            bad_groups = [str(x) for x in bad.index.tolist()[:8]]
            raise ValueError(
                f"Each {group_key!r} must map to exactly one {stratify_key!r}; bad groups={bad_groups}"
            )
    return group_meta.drop_duplicates(subset=[group_key], keep="first")


def _normalized_test_group_quotas(test_group_quotas: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    raw = _DEFAULT_TEST_GROUP_QUOTAS if test_group_quotas is None else test_group_quotas
    if not isinstance(raw, Mapping):
        raise TypeError(f"test_group_quotas must be a mapping, got {type(raw)!r}")
    out: Dict[str, int] = {}
    for key, value in raw.items():
        name = str(key).strip()
        quota = int(value)
        if quota < 0:
            raise ValueError(f"test_group_quotas entries must be non-negative, got {key!r}={value!r}")
        if name:
            out[name] = quota
    return out


def _shuffle_groups(groups: Sequence[str], rng: np.random.Generator) -> List[str]:
    items = np.asarray(sorted(str(x).strip() for x in groups), dtype=object)
    if items.size > 1:
        items = items[rng.permutation(items.size)]
    return [str(x) for x in items.tolist()]


def _select_fixed_test_groups(
    group_meta: pd.DataFrame,
    *,
    group_key: str,
    stratify_key: Optional[str],
    test_group_quotas: Mapping[str, int],
    rng: np.random.Generator,
) -> Set[str]:
    if not test_group_quotas:
        return set()

    if stratify_key is None:
        total_quota = int(sum(int(x) for x in test_group_quotas.values()))
        groups = _shuffle_groups(group_meta[group_key].tolist(), rng)
        if total_quota > len(groups):
            raise ValueError(
                f"Requested {total_quota} fixed test groups, but only {len(groups)} groups are available"
            )
        return set(groups[:total_quota])

    selected: Set[str] = set()
    for stratum, quota in sorted(test_group_quotas.items()):
        need = int(quota)
        if need <= 0:
            continue
        strata_rows = group_meta[group_meta[stratify_key] == str(stratum)]
        groups = _shuffle_groups(strata_rows[group_key].tolist(), rng)
        if need > len(groups):
            raise ValueError(
                f"Requested {need} fixed test groups for {stratify_key}={stratum!r}, "
                f"but only {len(groups)} are available"
            )
        selected.update(groups[:need])
    return selected


def _build_grouped_cv_folds(
    group_meta: pd.DataFrame,
    *,
    group_key: str,
    stratify_key: Optional[str],
    cv_n_folds: int,
    rng: np.random.Generator,
) -> List[Tuple[str, ...]]:
    n_folds = int(cv_n_folds)
    if n_folds <= 0:
        raise ValueError(f"cv_n_folds must be positive, got {cv_n_folds!r}")

    total_groups = int(group_meta.shape[0])
    if total_groups == 0:
        return [tuple() for _ in range(n_folds)]
    if total_groups % n_folds != 0:
        raise ValueError(
            f"Remaining grouped CV pool has {total_groups} groups, which cannot be split "
            f"evenly across {n_folds} folds"
        )

    target_size = total_groups // n_folds
    folds = [{"groups": [], "source_counts": Counter()} for _ in range(n_folds)]

    if stratify_key:
        grouped: Dict[str, List[str]] = {}
        for stratum in sorted(group_meta[stratify_key].drop_duplicates().tolist()):
            strata_rows = group_meta[group_meta[stratify_key] == str(stratum)]
            grouped[str(stratum)] = _shuffle_groups(strata_rows[group_key].tolist(), rng)
        strata_order = sorted(grouped.keys(), key=lambda name: (-len(grouped[name]), name))
    else:
        grouped = {"": _shuffle_groups(group_meta[group_key].tolist(), rng)}
        strata_order = [""]

    for stratum in strata_order:
        for group_id in grouped[stratum]:
            candidates = [idx for idx, fold in enumerate(folds) if len(fold["groups"]) < target_size]
            if not candidates:
                raise AssertionError("No fold has remaining capacity during grouped CV assignment")
            best_idx = min(
                candidates,
                key=lambda idx: (
                    int(folds[idx]["source_counts"][stratum]),
                    len(folds[idx]["groups"]),
                    idx,
                ),
            )
            folds[best_idx]["groups"].append(str(group_id))
            folds[best_idx]["source_counts"][stratum] += 1

    if any(len(fold["groups"]) != target_size for fold in folds):
        actual = [len(fold["groups"]) for fold in folds]
        raise AssertionError(f"Grouped CV folds do not have equal size: {actual}")

    return [tuple(str(x) for x in fold["groups"]) for fold in folds]


def assign_group_splits(
    case_table: pd.DataFrame,
    *,
    seed: int = 42,
    group_key: str = "base_id",
    stratify_key: Optional[str] = "source",
    split_names: Sequence[str] = _DEFAULT_SPLIT_NAMES,
    test_group_quotas: Optional[Mapping[str, Any]] = None,
    cv_n_folds: int = 5,
    cv_fold_index: int = 0,
) -> Dict[str, str]:
    if case_table is None or case_table.empty:
        return {}
    if group_key not in case_table.columns:
        raise KeyError(f"Case table is missing required group column: {group_key!r}")
    if stratify_key and stratify_key not in case_table.columns:
        raise KeyError(f"Case table is missing required stratify column: {stratify_key!r}")

    split_names = _validated_split_names(split_names)
    group_meta = _prepare_group_metadata(
        case_table,
        group_key=str(group_key),
        stratify_key=stratify_key,
    )
    quotas = _normalized_test_group_quotas(test_group_quotas)
    rng = np.random.default_rng(int(seed))
    fixed_test_groups = _select_fixed_test_groups(
        group_meta,
        group_key=str(group_key),
        stratify_key=stratify_key,
        test_group_quotas=quotas,
        rng=rng,
    )
    remaining_meta = group_meta[~group_meta[group_key].isin(sorted(fixed_test_groups))].copy()
    folds = _build_grouped_cv_folds(
        remaining_meta,
        group_key=str(group_key),
        stratify_key=stratify_key,
        cv_n_folds=int(cv_n_folds),
        rng=rng,
    )
    fold_index = int(cv_fold_index)
    if fold_index < 0 or fold_index >= len(folds):
        raise IndexError(f"cv_fold_index={cv_fold_index!r} is out of range for {len(folds)} folds")
    val_groups = set(folds[fold_index])

    split_map: Dict[str, str] = {}
    train_name, val_name, test_name = split_names
    for group_id in group_meta[group_key].tolist():
        gid = str(group_id).strip()
        if gid in fixed_test_groups:
            split_map[gid] = test_name
        elif gid in val_groups:
            split_map[gid] = val_name
        else:
            split_map[gid] = train_name
    return split_map


@dataclass
class AnsysSupervisionDataset:
    cases_by_split: Dict[str, List[Dict[str, Any]]]
    shuffle: bool = True
    seed: int = 42
    _orders: MutableMapping[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _indices: MutableMapping[str, int] = field(default_factory=dict, init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(int(self.seed))
        for split, cases in self.cases_by_split.items():
            order = np.arange(len(cases), dtype=np.int32)
            if self.shuffle and len(order) > 1:
                self._rng.shuffle(order)
            self._orders[split] = order
            self._indices[split] = 0

    def counts(self) -> Dict[str, int]:
        return {split: len(cases) for split, cases in self.cases_by_split.items()}

    def next_case(self, split: str = "train") -> Dict[str, Any]:
        if split not in self.cases_by_split or not self.cases_by_split[split]:
            raise KeyError(f"No supervision cases available for split={split!r}")

        order = self._orders[split]
        pos = self._indices[split]
        if pos >= len(order):
            order = np.arange(len(self.cases_by_split[split]), dtype=np.int32)
            if self.shuffle and len(order) > 1:
                self._rng.shuffle(order)
            self._orders[split] = order
            self._indices[split] = 0
            pos = 0

        case = self.cases_by_split[split][int(order[pos])]
        self._indices[split] = pos + 1
        return _copy_case(case)


def load_ansys_supervision_dataset(
    *,
    case_table_path: str,
    stage_dir: str,
    asm,
    splits: Optional[Sequence[str]] = None,
    stage_count: int = 3,
    shuffle: bool = True,
    seed: int = 42,
    split_group_key: str = "base_id",
    split_stratify_key: Optional[str] = "source",
    test_group_quotas: Optional[Mapping[str, Any]] = None,
    cv_n_folds: int = 5,
    cv_fold_index: int = 0,
) -> AnsysSupervisionDataset:
    if not case_table_path or not os.path.exists(case_table_path):
        raise FileNotFoundError(f"Case table not found: {case_table_path!r}")
    if not stage_dir or not os.path.isdir(stage_dir):
        raise FileNotFoundError(f"Stage directory not found: {stage_dir!r}")
    if asm is None or not hasattr(asm, "nodes"):
        raise ValueError("asm with a .nodes mapping is required for supervision loading")

    df = pd.read_csv(case_table_path)
    wanted_splits = None if not splits else {str(s).strip() for s in splits}
    nodes_xyz = getattr(asm, "nodes", {})
    cases_by_split: Dict[str, List[Dict[str, Any]]] = {}
    split_map = assign_group_splits(
        df,
        seed=int(seed),
        group_key=str(split_group_key or "base_id"),
        stratify_key=(None if split_stratify_key is None else str(split_stratify_key)),
        test_group_quotas=test_group_quotas,
        cv_n_folds=int(cv_n_folds),
        cv_fold_index=int(cv_fold_index),
    )

    required_case_cols = [
        "case_id",
        str(split_group_key or "base_id"),
        "theta_1_deg",
        "theta_2_deg",
        "theta_3_deg",
        "order_1",
        "order_2",
        "order_3",
    ]
    missing_case_cols = [c for c in required_case_cols if c not in df.columns]
    if missing_case_cols:
        raise KeyError(f"Case table is missing required columns: {missing_case_cols}")

    for row in df.to_dict(orient="records"):
        group_val = str(row.get(str(split_group_key or "base_id"), "")).strip()
        split = str(split_map.get(group_val, "")).strip()
        if wanted_splits is not None and split not in wanted_splits:
            continue

        P = np.asarray(
            [row["theta_1_deg"], row["theta_2_deg"], row["theta_3_deg"]],
            dtype=np.float32,
        )
        order = np.asarray(
            [row["order_1"], row["order_2"], row["order_3"]],
            dtype=np.int32,
        ).reshape(-1)
        if np.all(order >= 1):
            order = order - 1

        node_ids_ref: Optional[np.ndarray] = None
        xyz_ref: Optional[np.ndarray] = None
        X_stages: List[np.ndarray] = []
        U_stages: List[np.ndarray] = []

        for stage_rank in range(1, int(stage_count) + 1):
            stage_path = _resolve_stage_csv_path(stage_dir, row["case_id"], stage_rank)
            stage_df = pd.read_csv(stage_path)
            required_stage_cols = ["node_id", "dx_mm", "dy_mm", "dz_mm"]
            missing_stage_cols = [c for c in required_stage_cols if c not in stage_df.columns]
            if missing_stage_cols:
                raise KeyError(
                    f"Stage CSV {stage_path} is missing required columns: {missing_stage_cols}"
                )

            node_ids = stage_df["node_id"].to_numpy(dtype=np.int64)
            if node_ids_ref is None:
                node_ids_ref = node_ids
                missing_node_ids = [int(nid) for nid in node_ids if int(nid) not in nodes_xyz]
                if missing_node_ids:
                    raise KeyError(
                        f"Node ids from {stage_path} are missing in asm.nodes, sample={missing_node_ids[:8]}"
                    )
                xyz_ref = np.asarray([nodes_xyz[int(nid)] for nid in node_ids], dtype=np.float32)
            elif not np.array_equal(node_ids_ref, node_ids):
                raise ValueError(
                    f"Stage node_id ordering mismatch for case_id={row['case_id']!r} "
                    f"between stage 1 and stage {stage_rank}"
                )

            disp = stage_df[["dx_mm", "dy_mm", "dz_mm"]].to_numpy(dtype=np.float32)
            X_stages.append(xyz_ref.copy())
            U_stages.append(disp)

        case = {
            "case_id": str(row["case_id"]),
            "base_id": str(row.get("base_id", "")),
            "split": split,
            "source": str(row.get("source", "")),
            "job_name": str(row.get("job_name", "")),
            "P": P,
            "order": order.astype(np.int32),
            "node_ids": node_ids_ref.astype(np.int64),
            "X_obs": np.stack(X_stages, axis=0).astype(np.float32),
            "U_obs": np.stack(U_stages, axis=0).astype(np.float32),
        }
        L_ref, u_ref = _resolve_case_reference_scales(
            row,
            X_obs=case["X_obs"],
            U_obs=case["U_obs"],
        )
        case["X_obs_nd"] = _to_nondimensional_tensor(case["X_obs"], L_ref)
        case["U_obs_nd"] = _to_nondimensional_tensor(case["U_obs"], u_ref)
        case["obs_morphology_weight"] = _build_obs_morphology_weights(case["U_obs_nd"])
        case["U_obs_delta"] = _compute_stage_displacement_deltas(case["U_obs_nd"])
        cases_by_split.setdefault(split, []).append(case)

    return AnsysSupervisionDataset(
        cases_by_split=cases_by_split,
        shuffle=bool(shuffle),
        seed=int(seed),
    )
