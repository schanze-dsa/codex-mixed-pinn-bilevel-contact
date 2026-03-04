#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Total energy assembly for PINN with contact + tightening."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import tensorflow as tf

from physics.elasticity_residual import ElasticityResidual
from physics.elasticity_config import ElasticityConfig
from physics.contact.contact_operator import ContactOperator, ContactOperatorConfig
from physics.boundary_conditions import BoundaryPenalty, BoundaryConfig
from physics.tightening_model import NutTighteningPenalty, TighteningConfig


def compute_incremental_ed_penalty(
    delta_elastic: tf.Tensor,
    friction_dissipation: tf.Tensor,
    external_work_proxy: tf.Tensor,
    *,
    margin: tf.Tensor,
    use_relu: bool = True,
    squared: bool = True,
) -> tf.Tensor:
    """Incremental energy-dissipation mismatch penalty.

    raw = delta_elastic + friction_dissipation - external_work_proxy - margin
    penalty = relu(raw)^2 (default)
    """

    raw = delta_elastic + friction_dissipation - external_work_proxy - margin
    if use_relu:
        pen = tf.nn.relu(raw)
    else:
        pen = tf.abs(raw)
    if squared:
        pen = pen * pen
    return pen


@dataclass
class TotalConfig:
    loss_mode: str = "energy"  # "energy" | "residual"
    w_int: float = 1.0
    w_cn: float = 1.0
    w_ct: float = 1.0
    w_bc: float = 1.0
    w_tight: float = 1.0
    w_sigma: float = 1.0
    w_eq: float = 0.0
    w_reg: float = 0.0
    w_bi: float = 0.0
    w_ed: float = 0.0
    w_unc: float = 0.0
    sigma_ref: float = 1.0
    path_penalty_weight: float = 1.0
    fric_path_penalty_weight: float = 1.0
    ed_enabled: bool = False
    ed_external_scale: float = 1.0
    ed_margin: float = 0.0
    ed_use_relu: bool = True
    ed_square: bool = True
    adaptive_scheme: str = "contact_only"
    update_every_steps: int = 150
    dtype: str = "float32"


class TotalEnergy:
    def __init__(self, cfg: Optional[TotalConfig] = None):
        self.cfg = cfg or TotalConfig()
        self.dtype = tf.float32 if self.cfg.dtype == "float32" else tf.float64
        self.elasticity: Optional[ElasticityResidual] = None
        self.contact: Optional[ContactOperator] = None
        self.bcs: List[BoundaryPenalty] = []
        self.tightening: Optional[NutTighteningPenalty] = None
        self._ensure_weight_vars()
        self._built = False

    def _ensure_weight_vars(self):
        if not hasattr(self, "w_int"):
            self.w_int = tf.Variable(self.cfg.w_int, dtype=self.dtype, trainable=False, name="w_int")
        if not hasattr(self, "w_cn"):
            self.w_cn = tf.Variable(self.cfg.w_cn, dtype=self.dtype, trainable=False, name="w_cn")
        if not hasattr(self, "w_ct"):
            self.w_ct = tf.Variable(self.cfg.w_ct, dtype=self.dtype, trainable=False, name="w_ct")
        if not hasattr(self, "w_bc"):
            self.w_bc = tf.Variable(self.cfg.w_bc, dtype=self.dtype, trainable=False, name="w_bc")
        if not hasattr(self, "w_tight"):
            self.w_tight = tf.Variable(self.cfg.w_tight, dtype=self.dtype, trainable=False, name="w_tight")
        if not hasattr(self, "w_sigma"):
            self.w_sigma = tf.Variable(self.cfg.w_sigma, dtype=self.dtype, trainable=False, name="w_sigma")
        if not hasattr(self, "w_eq"):
            self.w_eq = tf.Variable(self.cfg.w_eq, dtype=self.dtype, trainable=False, name="w_eq")
        if not hasattr(self, "w_reg"):
            self.w_reg = tf.Variable(self.cfg.w_reg, dtype=self.dtype, trainable=False, name="w_reg")
        if not hasattr(self, "w_bi"):
            self.w_bi = tf.Variable(getattr(self.cfg, "w_bi", 0.0), dtype=self.dtype, trainable=False, name="w_bi")
        if not hasattr(self, "w_ed"):
            self.w_ed = tf.Variable(getattr(self.cfg, "w_ed", 0.0), dtype=self.dtype, trainable=False, name="w_ed")
        if not hasattr(self, "w_unc"):
            self.w_unc = tf.Variable(getattr(self.cfg, "w_unc", 0.0), dtype=self.dtype, trainable=False, name="w_unc")

    def _loss_mode(self) -> str:
        mode = str(getattr(self.cfg, "loss_mode", "energy") or "energy").strip().lower()
        if mode in {"residual", "residual_only", "res"}:
            return "residual"
        return "energy"

    @staticmethod
    def _resolve_bound_variant(fn, method_name: str):
        """Use an alternate bound method when available (e.g. pointwise forward)."""

        if fn is None:
            return None
        owner = getattr(fn, "__self__", None)
        if owner is None:
            return fn
        alt = getattr(owner, method_name, None)
        if callable(alt):
            return alt
        return fn

    def attach(self, elasticity: Optional[ElasticityResidual] = None,
               contact: Optional[ContactOperator] = None,
               tightening: Optional[NutTighteningPenalty] = None,
               bcs: Optional[List[BoundaryPenalty]] = None):
        if elasticity is not None:
            self.elasticity = elasticity
        if contact is not None:
            self.contact = contact
        if tightening is not None:
            self.tightening = tightening
        if bcs is not None:
            self.bcs = list(bcs)
        self._built = True

    def reset(self):
        self.elasticity = None
        self.contact = None
        self.tightening = None
        self.bcs = []
        self._built = False

    def scale_volume_weights(self, factor: float):
        if getattr(self.elasticity, "w_vol_tf", None) is None:
            return
        try:
            self.elasticity.w_vol_tf = self.elasticity.w_vol_tf * tf.cast(factor, self.dtype)
        except Exception:
            pass

    def energy(self, u_fn, params=None, tape=None, stress_fn=None):
        self._ensure_weight_vars()
        if not self._built:
            raise RuntimeError("[TotalEnergy] attach(...) must be called before energy().")
        if self._loss_mode() == "residual":
            if isinstance(params, dict) and params.get("stages"):
                Pi, parts, stats = self._residual_staged(u_fn, params["stages"], params, tape, stress_fn=stress_fn)
            else:
                parts, stats = self._compute_parts_residual(u_fn, params or {}, tape, stress_fn=stress_fn)
                Pi = self._combine_parts(parts)
            return Pi, parts, stats
        if isinstance(params, dict) and params.get("stages"):
            Pi, parts, stats = self._energy_staged(u_fn, params["stages"], params, tape, stress_fn=stress_fn)
            return Pi, parts, stats
        parts, stats = self._compute_parts(u_fn, params or {}, tape, stress_fn=stress_fn)
        Pi = self._combine_parts(parts)
        return Pi, parts, stats
    def _compute_parts(self, u_fn, params, tape=None, stress_fn=None):
        dtype = self.dtype
        zero = tf.cast(0.0, dtype)
        parts: Dict[str, tf.Tensor] = {
            "E_int": zero,
            "E_cn": zero,
            "E_ct": zero,
            "E_bc": zero,
            "E_tight": zero,
            "E_sigma": zero,
            "E_eq": zero,
            "E_reg": zero,
            "E_bi": zero,
            "E_ed": zero,
            "E_unc": zero,
        }
        stats: Dict[str, tf.Tensor] = {}

        u_nodes = None
        elastic_cache = None
        u_fn_elastic = self._resolve_bound_variant(u_fn, "u_fn_pointwise")
        stress_fn_elastic = self._resolve_bound_variant(stress_fn, "us_fn_pointwise")
        if self.elasticity is not None:
            u_nodes = self.elasticity._eval_u_on_nodes(u_fn, params)
            E_int_res = self.elasticity.energy(
                u_fn_elastic,
                params,
                tape=tape,
                return_cache=bool(stress_fn_elastic),
                u_nodes=u_nodes,
            )
            if bool(stress_fn_elastic):
                E_int, estates, elastic_cache = E_int_res  # type: ignore[misc]
            else:
                E_int, estates = E_int_res  # type: ignore[misc]
            parts["E_int"] = tf.cast(E_int, dtype)
            stats.update({f"el_{k}": v for k, v in estates.items()})

        if self.contact is not None:
            _, cparts, stats_cn, stats_ct = self.contact.energy(u_fn, params, u_nodes=u_nodes)
            if "E_cn" in cparts:
                parts["E_cn"] = tf.cast(cparts["E_cn"], dtype)
            elif "E_n" in cparts:
                parts["E_cn"] = tf.cast(cparts["E_n"], dtype)
            if "E_ct" in cparts:
                parts["E_ct"] = tf.cast(cparts["E_ct"], dtype)
            elif "E_t" in cparts:
                parts["E_ct"] = tf.cast(cparts["E_t"], dtype)
            stats.update(stats_cn)
            stats.update(stats_ct)
            if "R_fric_comp" in stats_ct:
                parts["R_fric_comp"] = tf.cast(stats_ct["R_fric_comp"], dtype)
            if "R_contact_comp" in stats_cn:
                parts["R_contact_comp"] = tf.cast(stats_cn["R_contact_comp"], dtype)
            if "E_bi" in stats_ct:
                parts["E_bi"] = tf.cast(stats_ct["E_bi"], dtype)

        if self.bcs:
            bc_terms = []
            for i, b in enumerate(self.bcs):
                E_bc_i, si = b.energy(u_fn, params)
                bc_terms.append(tf.cast(E_bc_i, dtype))
                stats.update({f"bc{i+1}_{k}": v for k, v in si.items()})
            if bc_terms:
                parts["E_bc"] = tf.add_n(bc_terms)

        if self.tightening is not None:
            E_tight, tstats = self.tightening.energy(u_fn, params, u_nodes=u_nodes)
            parts["E_tight"] = tf.cast(E_tight, dtype)
            stats.update(tstats)

        w_sigma = float(getattr(self.cfg, "w_sigma", 0.0))
        w_eq = float(getattr(self.cfg, "w_eq", 0.0))
        use_stress = stress_fn_elastic is not None and elastic_cache is not None
        use_sigma = use_stress and w_sigma > 1e-15 and getattr(self.elasticity.cfg, "stress_loss_weight", 0.0) > 0.0
        use_eq = use_stress and w_eq > 1e-15

        if use_sigma or use_eq:
            eps_vec = tf.cast(elastic_cache["eps_vec"], dtype)
            lam = tf.cast(elastic_cache.get("lam", 0.0), dtype)
            mu = tf.cast(elastic_cache.get("mu", 0.0), dtype)
            dof_idx = tf.cast(elastic_cache.get("dof_idx", 0), tf.int32)

            sigma_phys = elastic_cache.get("sigma_phys")
            if sigma_phys is not None:
                sigma_phys = tf.cast(sigma_phys, dtype)
            else:
                eps_tensor = elastic_cache.get("eps_tensor")
                if eps_tensor is None:
                    eps_tensor = tf.stack(
                        [
                            eps_vec[:, 0],
                            eps_vec[:, 1],
                            eps_vec[:, 2],
                            0.5 * eps_vec[:, 3],
                            0.5 * eps_vec[:, 4],
                            0.5 * eps_vec[:, 5],
                        ],
                        axis=1,
                    )
                else:
                    eps_tensor = tf.cast(eps_tensor, dtype)
                tr_eps = eps_tensor[:, 0] + eps_tensor[:, 1] + eps_tensor[:, 2]
                eye_vec = tf.constant([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=dtype)
                sigma_phys = lam[:, None] * tr_eps[:, None] * eye_vec + 2.0 * mu[:, None] * eps_tensor

            sigma_phys = sigma_phys[:, :6]
            sigma_phys = tf.stack(
                [
                    sigma_phys[:, 0],
                    sigma_phys[:, 1],
                    sigma_phys[:, 2],
                    sigma_phys[:, 5],
                    sigma_phys[:, 3],
                    sigma_phys[:, 4],
                ],
                axis=1,
            )

            node_ids = tf.reshape(dof_idx // 3, (-1,))
            unique_nodes, rev = tf.unique(node_ids)
            X_nodes = tf.cast(tf.gather(self.elasticity.X_nodes_tf, unique_nodes), dtype)
            _, sigma_pred_nodes = stress_fn_elastic(X_nodes, params)
            sigma_pred_nodes = tf.cast(sigma_pred_nodes, dtype)

            sigma_nodes_full = tf.gather(sigma_pred_nodes, rev)
            sigma_cells = tf.reshape(sigma_nodes_full, (tf.shape(dof_idx)[0], 4, -1))
            sigma_cells = tf.reduce_mean(sigma_cells, axis=1)
            sigma_cells = sigma_cells[:, : tf.shape(sigma_phys)[1]]

            sigma_ref = tf.cast(getattr(self.cfg, "sigma_ref", 1.0), dtype)
            sigma_ref = tf.maximum(sigma_ref, tf.cast(1e-12, dtype))

            if use_sigma:
                diff = sigma_cells - sigma_phys
                diff_n = diff / sigma_ref
                loss_sigma = tf.reduce_mean(diff_n * diff_n)
                parts["E_sigma"] = loss_sigma * tf.cast(
                    getattr(self.elasticity.cfg, "stress_loss_weight", 1.0), dtype
                )
                stats["stress_rms"] = tf.sqrt(tf.reduce_mean(sigma_cells * sigma_cells) + 1e-20)

            if use_eq:
                div_sigma = elastic_cache.get("div_sigma")
                if div_sigma is not None:
                    div_sigma = tf.cast(div_sigma, dtype) / sigma_ref
                    res = tf.reduce_sum(div_sigma * div_sigma, axis=1)
                    parts["E_eq"] = tf.reduce_mean(res)
                    stats["eq_rms"] = tf.sqrt(tf.reduce_mean(res) + 1e-20)

        return parts, stats

    def _compute_parts_residual(self, u_fn, params, tape=None, stress_fn=None):
        dtype = self.dtype
        zero = tf.cast(0.0, dtype)
        parts: Dict[str, tf.Tensor] = {
            "E_int": zero,
            "E_cn": zero,
            "E_ct": zero,
            "E_bc": zero,
            "E_tight": zero,
            "E_sigma": zero,
            "E_eq": zero,
            "E_reg": zero,
            "E_bi": zero,
            "E_ed": zero,
            "E_unc": zero,
        }
        stats: Dict[str, tf.Tensor] = {}

        w_sigma = float(getattr(self.cfg, "w_sigma", 0.0))
        w_eq = float(getattr(self.cfg, "w_eq", 0.0))
        w_reg = float(getattr(self.cfg, "w_reg", 0.0))
        stress_weight = float(getattr(getattr(self.elasticity, "cfg", None), "stress_loss_weight", 0.0))
        u_fn_elastic = self._resolve_bound_variant(u_fn, "u_fn_pointwise")
        stress_fn_elastic = self._resolve_bound_variant(stress_fn, "us_fn_pointwise")
        need_sigma = stress_fn_elastic is not None and w_sigma > 1e-15 and stress_weight > 0.0
        need_eq = w_eq > 1e-15
        need_reg = w_reg > 1e-15

        u_nodes = None
        elastic_cache = None
        if self.elasticity is not None:
            u_nodes = self.elasticity._eval_u_on_nodes(u_fn, params)
            estates, elastic_cache = self.elasticity.residual_cache(
                u_fn_elastic,
                params,
                stress_fn=stress_fn_elastic,
                need_sigma=need_sigma,
                need_eq=need_eq,
            )
            stats.update({f"el_{k}": v for k, v in estates.items()})

        if self.contact is not None:
            _, cparts, stats_cn, stats_ct = self.contact.residual(u_fn, params, u_nodes=u_nodes)
            if "E_cn" in cparts:
                parts["E_cn"] = tf.cast(cparts["E_cn"], dtype)
            elif "E_n" in cparts:
                parts["E_cn"] = tf.cast(cparts["E_n"], dtype)
            if "E_ct" in cparts:
                parts["E_ct"] = tf.cast(cparts["E_ct"], dtype)
            elif "E_t" in cparts:
                parts["E_ct"] = tf.cast(cparts["E_t"], dtype)
            stats.update(stats_cn)
            stats.update(stats_ct)
            if "R_fric_comp" in stats_ct:
                parts["R_fric_comp"] = tf.cast(stats_ct["R_fric_comp"], dtype)
            if "R_contact_comp" in stats_cn:
                parts["R_contact_comp"] = tf.cast(stats_cn["R_contact_comp"], dtype)
            if "E_bi" in stats_ct:
                parts["E_bi"] = tf.cast(stats_ct["E_bi"], dtype)

        if self.bcs:
            bc_terms = []
            for i, b in enumerate(self.bcs):
                L_bc_i, si = b.residual(u_fn, params)
                bc_terms.append(tf.cast(L_bc_i, dtype))
                stats.update({f"bc{i+1}_{k}": v for k, v in si.items()})
            if bc_terms:
                parts["E_bc"] = tf.add_n(bc_terms)

        if self.tightening is not None:
            E_tight, tstats = self.tightening.residual(u_fn, params, u_nodes=u_nodes)
            parts["E_tight"] = tf.cast(E_tight, dtype)
            stats.update(tstats)

        use_stress = stress_fn_elastic is not None and elastic_cache is not None
        use_sigma = use_stress and w_sigma > 1e-15 and stress_weight > 0.0
        use_eq = elastic_cache is not None and w_eq > 1e-15
        use_reg = elastic_cache is not None and w_reg > 1e-15

        if use_sigma or use_eq or use_reg:
            eps_vec = elastic_cache.get("eps_vec") if isinstance(elastic_cache, dict) else None
            sigma_phys_head = elastic_cache.get("sigma_phys") if isinstance(elastic_cache, dict) else None
            sigma_pred = elastic_cache.get("sigma_pred") if isinstance(elastic_cache, dict) else None
            div_sigma = elastic_cache.get("div_sigma") if isinstance(elastic_cache, dict) else None
            w_sel = elastic_cache.get("w_sel") if isinstance(elastic_cache, dict) else None

            sigma_ref = tf.cast(getattr(self.cfg, "sigma_ref", 1.0), dtype)
            sigma_ref = tf.maximum(sigma_ref, tf.cast(1e-12, dtype))

            if use_sigma and sigma_pred is not None and sigma_phys_head is not None:
                sigma_pred = tf.cast(sigma_pred, dtype)
                sigma_phys_head = tf.cast(sigma_phys_head, dtype)
                diff = sigma_pred - sigma_phys_head
                diff_n = diff / sigma_ref
                res = tf.reduce_sum(diff_n * diff_n, axis=1)
                if w_sel is not None:
                    w_sel = tf.cast(w_sel, dtype)
                    res = res * w_sel
                    denom = tf.reduce_sum(w_sel) + tf.cast(1e-12, dtype)
                    loss_sigma = tf.reduce_sum(res) / denom
                else:
                    loss_sigma = tf.reduce_mean(res)
                parts["E_sigma"] = loss_sigma * tf.cast(stress_weight, dtype)
                stats["stress_rms"] = tf.sqrt(tf.reduce_mean(sigma_pred * sigma_pred) + tf.cast(1e-20, dtype))

            if use_eq and div_sigma is not None:
                div_sigma = tf.cast(div_sigma, dtype) / sigma_ref
                res = tf.reduce_sum(div_sigma * div_sigma, axis=1)
                if w_sel is not None:
                    w_sel = tf.cast(w_sel, dtype)
                    res = res * w_sel
                    denom = tf.reduce_sum(w_sel) + tf.cast(1e-12, dtype)
                else:
                    denom = tf.cast(tf.shape(res)[0], dtype) + tf.cast(1e-12, dtype)
                parts["E_eq"] = tf.reduce_sum(res) / denom
                stats["eq_rms"] = tf.sqrt(tf.reduce_mean(res) + tf.cast(1e-20, dtype))

            if use_reg and eps_vec is not None:
                eps_vec = tf.cast(eps_vec, dtype)
                res = tf.reduce_sum(eps_vec * eps_vec, axis=1)
                if w_sel is not None:
                    w_sel = tf.cast(w_sel, dtype)
                    res = res * w_sel
                    denom = tf.reduce_sum(w_sel) + tf.cast(1e-12, dtype)
                else:
                    denom = tf.cast(tf.shape(res)[0], dtype) + tf.cast(1e-12, dtype)
                parts["E_reg"] = tf.reduce_sum(res) / denom
                stats["reg_rms"] = tf.sqrt(tf.reduce_mean(res) + tf.cast(1e-20, dtype))

        return parts, stats

    def _combine_parts(self, parts: Dict[str, tf.Tensor]) -> tf.Tensor:
        return (
            self.w_int * parts.get("E_int", tf.cast(0.0, self.dtype))
            + self.w_cn * parts.get("E_cn", tf.cast(0.0, self.dtype))
            + self.w_ct * parts.get("E_ct", tf.cast(0.0, self.dtype))
            + self.w_bc * parts.get("E_bc", tf.cast(0.0, self.dtype))
            + self.w_tight * parts.get("E_tight", tf.cast(0.0, self.dtype))
            + self.w_sigma * parts.get("E_sigma", tf.cast(0.0, self.dtype))
            + self.w_eq * parts.get("E_eq", tf.cast(0.0, self.dtype))
            + self.w_reg * parts.get("E_reg", tf.cast(0.0, self.dtype))
            + self.w_bi * parts.get("E_bi", tf.cast(0.0, self.dtype))
            + self.w_ed * parts.get("E_ed", tf.cast(0.0, self.dtype))
            + self.w_unc * parts.get("E_unc", tf.cast(0.0, self.dtype))
        )

    def _energy_staged(self, u_fn, stages, root_params, tape=None, stress_fn=None):
        dtype = self.dtype
        keys = ["E_int", "E_cn", "E_ct", "E_bc", "E_tight", "E_sigma", "E_eq", "E_reg", "E_bi", "E_ed", "E_unc"]
        totals: Dict[str, tf.Tensor] = {k: tf.cast(0.0, dtype) for k in keys}
        stats_all: Dict[str, tf.Tensor] = {}
        path_penalty_total = tf.cast(0.0, dtype)
        fric_path_penalty_total = tf.cast(0.0, dtype)
        Pi_accum = tf.cast(0.0, dtype)

        if isinstance(stages, dict):
            stage_tensor_P = stages.get("P")
            stage_tensor_feat = stages.get("P_hat")
            stage_tensor_rank = stages.get("stage_rank")
            stage_tensor_mask = stages.get("stage_mask")
            stage_tensor_last = stages.get("stage_last")
            if stage_tensor_P is None or stage_tensor_feat is None:
                stage_seq: List[Dict[str, tf.Tensor]] = []
            else:
                stacked_rank = None
                if stage_tensor_rank is not None:
                    stacked_rank = tf.convert_to_tensor(stage_tensor_rank)
                stacked_mask = None
                if stage_tensor_mask is not None:
                    stacked_mask = tf.convert_to_tensor(stage_tensor_mask)
                stacked_last = None
                if stage_tensor_last is not None:
                    stacked_last = tf.convert_to_tensor(stage_tensor_last)
                stage_seq = []
                for idx, (p, z) in enumerate(zip(tf.unstack(stage_tensor_P, axis=0), tf.unstack(stage_tensor_feat, axis=0))):
                    entry = {"P": p, "P_hat": z}
                    if stacked_rank is not None:
                        entry["stage_rank"] = stacked_rank[idx] if stacked_rank.shape.rank == 2 else stacked_rank
                    if stacked_mask is not None:
                        entry["stage_mask"] = stacked_mask[idx]
                    if stacked_last is not None:
                        entry["stage_last"] = stacked_last[idx]
                    stage_seq.append(entry)
        else:
            stage_seq = []
            for item in stages:
                if isinstance(item, dict):
                    stage_seq.append(item)
                else:
                    p_val, z_val = item
                    stage_seq.append({"P": p_val, "P_hat": z_val})

        if not stage_seq:
            return self._combine_parts(totals), totals, stats_all

        prev_P: Optional[tf.Tensor] = None
        prev_slip: Optional[tf.Tensor] = None
        prev_E_int: Optional[tf.Tensor] = None
        stage_count = len(stage_seq)

        for idx, stage_params in enumerate(stage_seq):
            stage_idx = tf.cast(idx, tf.int32)
            stage_frac = tf.cast(0.0 if stage_count <= 1 else idx / max(stage_count - 1, 1), dtype)
            stage_params = dict(stage_params)
            stage_params.setdefault("stage_index", stage_idx)
            stage_params.setdefault("stage_fraction", stage_frac)

            stage_parts, stage_stats = self._compute_parts(u_fn, stage_params, tape, stress_fn=stress_fn)
            stage_parts = dict(stage_parts)
            for k, v in stage_stats.items():
                stats_all[f"s{idx+1}_{k}"] = v

            for key in keys:
                cur = tf.cast(stage_parts.get(key, tf.cast(0.0, dtype)), dtype)
                totals[key] = totals[key] + cur
                stats_all[f"s{idx+1}_{key}"] = cur
                stats_all[f"s{idx+1}_cum{key}"] = totals[key]

            P_vec = tf.cast(tf.convert_to_tensor(stage_params.get("P", [])), dtype)
            slip_t = None
            if self.contact is not None and hasattr(self.contact, "last_friction_slip"):
                slip_t = self.contact.last_friction_slip()

            w_path = tf.cast(getattr(self.cfg, "path_penalty_weight", 1.0), dtype)
            w_fric_path = tf.cast(getattr(self.cfg, "fric_path_penalty_weight", 1.0), dtype)

            stage_path = tf.cast(0.0, dtype)
            stage_fric_path = tf.cast(0.0, dtype)
            if idx > 0:
                load_jump = tf.reduce_sum(tf.abs(P_vec - prev_P)) if prev_P is not None else tf.cast(0.0, dtype)
                if slip_t is not None and prev_slip is not None:
                    slip_jump = tf.reduce_sum(tf.abs(slip_t - prev_slip))
                    stage_fric_path = slip_jump * load_jump
                    fric_path_penalty_total = fric_path_penalty_total + stage_fric_path
                    stats_all[f"s{idx+1}_fric_path_penalty"] = stage_fric_path
                    stats_all[f"s{idx+1}_fric_path_penalty_w"] = w_fric_path
                if bool(getattr(self.cfg, "ed_enabled", False)):
                    cur_e_int = tf.cast(stage_parts.get("E_int", tf.cast(0.0, dtype)), dtype)
                    d_el = cur_e_int - (prev_E_int if prev_E_int is not None else tf.cast(0.0, dtype))
                    d_fric = tf.abs(tf.cast(stage_parts.get("E_ct", tf.cast(0.0, dtype)), dtype))
                    w_ext = tf.cast(getattr(self.cfg, "ed_external_scale", 1.0), dtype) * load_jump
                    ed_pen = compute_incremental_ed_penalty(
                        d_el,
                        d_fric,
                        w_ext,
                        margin=tf.cast(getattr(self.cfg, "ed_margin", 0.0), dtype),
                        use_relu=bool(getattr(self.cfg, "ed_use_relu", True)),
                        squared=bool(getattr(self.cfg, "ed_square", True)),
                    )
                    stage_parts["E_ed"] = tf.cast(ed_pen, dtype)
                    totals["E_ed"] = totals["E_ed"] + tf.cast(ed_pen, dtype)
                    stats_all[f"s{idx+1}_E_ed"] = tf.cast(ed_pen, dtype)
                    stats_all[f"s{idx+1}_ed_delta_el"] = tf.cast(d_el, dtype)
                    stats_all[f"s{idx+1}_ed_d_fric"] = tf.cast(d_fric, dtype)
                    stats_all[f"s{idx+1}_ed_w_ext"] = tf.cast(w_ext, dtype)
            stage_path_penalty = w_path * stage_path + w_fric_path * stage_fric_path
            if stage_path != 0.0:
                path_penalty_total = path_penalty_total + stage_path
                stats_all[f"s{idx+1}_path_penalty"] = stage_path
                stats_all[f"s{idx+1}_path_penalty_w"] = w_path

            stage_mech = self._combine_parts(stage_parts)
            stage_pi_step = stage_mech + stage_path_penalty
            stats_all[f"s{idx+1}_Pi_step"] = stage_pi_step
            stats_all[f"s{idx+1}_Pi_mech"] = stage_mech

            Pi_accum = Pi_accum + stage_pi_step

            if tf.size(P_vec) > 0:
                prev_P = P_vec
            if slip_t is not None:
                prev_slip = slip_t
            prev_E_int = tf.cast(stage_parts.get("E_int", tf.cast(0.0, dtype)), dtype)
            if self.contact is not None:
                try:
                    stage_params_detached = {k: tf.stop_gradient(v) if isinstance(v, tf.Tensor) else v for k, v in stage_params.items()}
                    self.contact.update_multipliers(u_fn, stage_params_detached)
                except Exception:
                    pass
            if self.bcs:
                for bc in self.bcs:
                    try:
                        bc.update_multipliers(u_fn, stage_params)
                    except Exception:
                        pass

        if isinstance(root_params, dict):
            if "stage_order" in root_params:
                stats_all["stage_order"] = root_params["stage_order"]
            if "stage_rank" in root_params:
                stats_all["stage_rank"] = root_params["stage_rank"]
            if "stage_count" in root_params:
                stats_all["stage_count"] = root_params["stage_count"]

        stats_all["path_penalty_total"] = path_penalty_total
        stats_all["fric_path_penalty_total"] = fric_path_penalty_total
        totals["path_penalty_total"] = path_penalty_total
        totals["fric_path_penalty_total"] = fric_path_penalty_total

        Pi = Pi_accum
        return Pi, totals, stats_all

    def _residual_staged(self, u_fn, stages, root_params, tape=None, stress_fn=None):
        dtype = self.dtype
        keys = ["E_int", "E_cn", "E_ct", "E_bc", "E_tight", "E_sigma", "E_eq", "E_reg", "E_bi", "E_ed", "E_unc"]
        totals: Dict[str, tf.Tensor] = {k: tf.cast(0.0, dtype) for k in keys}
        stats_all: Dict[str, tf.Tensor] = {}
        path_penalty_total = tf.cast(0.0, dtype)
        fric_path_penalty_total = tf.cast(0.0, dtype)
        Pi_accum = tf.cast(0.0, dtype)

        if isinstance(stages, dict):
            stage_tensor_P = stages.get("P")
            stage_tensor_feat = stages.get("P_hat")
            stage_tensor_rank = stages.get("stage_rank")
            stage_tensor_mask = stages.get("stage_mask")
            stage_tensor_last = stages.get("stage_last")
            if stage_tensor_P is None or stage_tensor_feat is None:
                stage_seq: List[Dict[str, tf.Tensor]] = []
            else:
                stacked_rank = None
                if stage_tensor_rank is not None:
                    stacked_rank = tf.convert_to_tensor(stage_tensor_rank)
                stacked_mask = None
                if stage_tensor_mask is not None:
                    stacked_mask = tf.convert_to_tensor(stage_tensor_mask)
                stacked_last = None
                if stage_tensor_last is not None:
                    stacked_last = tf.convert_to_tensor(stage_tensor_last)
                stage_seq = []
                for idx, (p, z) in enumerate(zip(tf.unstack(stage_tensor_P, axis=0), tf.unstack(stage_tensor_feat, axis=0))):
                    entry = {"P": p, "P_hat": z}
                    if stacked_rank is not None:
                        entry["stage_rank"] = stacked_rank[idx] if stacked_rank.shape.rank == 2 else stacked_rank
                    if stacked_mask is not None:
                        entry["stage_mask"] = stacked_mask[idx]
                    if stacked_last is not None:
                        entry["stage_last"] = stacked_last[idx]
                    stage_seq.append(entry)
        else:
            stage_seq = []
            for item in stages:
                if isinstance(item, dict):
                    stage_seq.append(item)
                else:
                    p_val, z_val = item
                    stage_seq.append({"P": p_val, "P_hat": z_val})

        if not stage_seq:
            return self._combine_parts(totals), totals, stats_all

        prev_P: Optional[tf.Tensor] = None
        prev_slip: Optional[tf.Tensor] = None
        prev_E_int: Optional[tf.Tensor] = None
        stage_count = len(stage_seq)

        for idx, stage_params in enumerate(stage_seq):
            stage_idx = tf.cast(idx, tf.int32)
            stage_frac = tf.cast(0.0 if stage_count <= 1 else idx / max(stage_count - 1, 1), dtype)
            stage_params = dict(stage_params)
            stage_params.setdefault("stage_index", stage_idx)
            stage_params.setdefault("stage_fraction", stage_frac)

            stage_parts, stage_stats = self._compute_parts_residual(u_fn, stage_params, tape, stress_fn=stress_fn)
            stage_parts = dict(stage_parts)
            for k, v in stage_stats.items():
                stats_all[f"s{idx+1}_{k}"] = v

            for key in keys:
                cur = tf.cast(stage_parts.get(key, tf.cast(0.0, dtype)), dtype)
                totals[key] = totals[key] + cur
                stats_all[f"s{idx+1}_{key}"] = cur
                stats_all[f"s{idx+1}_cum{key}"] = totals[key]

            P_vec = tf.cast(tf.convert_to_tensor(stage_params.get("P", [])), dtype)
            slip_t = None
            if self.contact is not None and hasattr(self.contact, "last_friction_slip"):
                slip_t = self.contact.last_friction_slip()

            w_path = tf.cast(getattr(self.cfg, "path_penalty_weight", 1.0), dtype)
            w_fric_path = tf.cast(getattr(self.cfg, "fric_path_penalty_weight", 1.0), dtype)

            stage_path = tf.cast(0.0, dtype)
            stage_fric_path = tf.cast(0.0, dtype)
            if idx > 0:
                load_jump = tf.reduce_sum(tf.abs(P_vec - prev_P)) if prev_P is not None else tf.cast(0.0, dtype)
                if slip_t is not None and prev_slip is not None:
                    slip_jump = tf.reduce_sum(tf.abs(slip_t - prev_slip))
                    stage_fric_path = slip_jump * load_jump
                    fric_path_penalty_total = fric_path_penalty_total + stage_fric_path
                    stats_all[f"s{idx+1}_fric_path_penalty"] = stage_fric_path
                    stats_all[f"s{idx+1}_fric_path_penalty_w"] = w_fric_path
                if bool(getattr(self.cfg, "ed_enabled", False)):
                    cur_e_int = tf.cast(stage_parts.get("E_int", tf.cast(0.0, dtype)), dtype)
                    d_el = cur_e_int - (prev_E_int if prev_E_int is not None else tf.cast(0.0, dtype))
                    d_fric = tf.abs(tf.cast(stage_parts.get("E_ct", tf.cast(0.0, dtype)), dtype))
                    w_ext = tf.cast(getattr(self.cfg, "ed_external_scale", 1.0), dtype) * load_jump
                    ed_pen = compute_incremental_ed_penalty(
                        d_el,
                        d_fric,
                        w_ext,
                        margin=tf.cast(getattr(self.cfg, "ed_margin", 0.0), dtype),
                        use_relu=bool(getattr(self.cfg, "ed_use_relu", True)),
                        squared=bool(getattr(self.cfg, "ed_square", True)),
                    )
                    stage_parts["E_ed"] = tf.cast(ed_pen, dtype)
                    totals["E_ed"] = totals["E_ed"] + tf.cast(ed_pen, dtype)
                    stats_all[f"s{idx+1}_E_ed"] = tf.cast(ed_pen, dtype)
                    stats_all[f"s{idx+1}_ed_delta_el"] = tf.cast(d_el, dtype)
                    stats_all[f"s{idx+1}_ed_d_fric"] = tf.cast(d_fric, dtype)
                    stats_all[f"s{idx+1}_ed_w_ext"] = tf.cast(w_ext, dtype)
            stage_path_penalty = w_path * stage_path + w_fric_path * stage_fric_path
            if stage_path != 0.0:
                path_penalty_total = path_penalty_total + stage_path
                stats_all[f"s{idx+1}_path_penalty"] = stage_path
                stats_all[f"s{idx+1}_path_penalty_w"] = w_path

            stage_pi_step = self._combine_parts(stage_parts) + stage_path_penalty
            stats_all[f"s{idx+1}_Pi_step"] = stage_pi_step

            Pi_accum = Pi_accum + stage_pi_step

            if tf.size(P_vec) > 0:
                prev_P = P_vec
            if slip_t is not None:
                prev_slip = slip_t
            prev_E_int = tf.cast(stage_parts.get("E_int", tf.cast(0.0, dtype)), dtype)

        stats_all["path_penalty_total"] = path_penalty_total
        stats_all["fric_path_penalty_total"] = fric_path_penalty_total
        totals["path_penalty_total"] = path_penalty_total
        totals["fric_path_penalty_total"] = fric_path_penalty_total

        Pi = Pi_accum
        return Pi, totals, stats_all

    def update_multipliers(self, u_fn, params=None):
        target_params = params
        staged_updates: List[Dict[str, tf.Tensor]] = []
        if isinstance(params, dict) and params.get("stages"):
            stages = params["stages"]
            if isinstance(stages, dict):
                stage_tensor_P = stages.get("P")
                stage_tensor_feat = stages.get("P_hat")
                stage_tensor_rank = stages.get("stage_rank")
                if stage_tensor_P is not None and stage_tensor_feat is not None:
                    for idx, (p, z) in enumerate(zip(tf.unstack(stage_tensor_P, axis=0), tf.unstack(stage_tensor_feat, axis=0))):
                        entry: Dict[str, tf.Tensor] = {"P": p, "P_hat": z}
                        if stage_tensor_rank is not None:
                            entry["stage_rank"] = stage_tensor_rank[idx] if stage_tensor_rank.shape.rank == 2 else stage_tensor_rank
                        staged_updates.append(entry)
                        target_params = entry
            elif isinstance(stages, (list, tuple)) and stages:
                for stage in stages:
                    if isinstance(stage, dict):
                        staged_updates.append(stage)
                        target_params = stage
                    else:
                        p_val, z_val = stage
                        entry = {"P": p_val, "P_hat": z_val}
                        staged_updates.append(entry)
                        target_params = entry

        if self.contact is not None:
            if staged_updates:
                for st_params in staged_updates:
                    u_nodes = None
                    if self.elasticity is not None:
                        u_nodes = self.elasticity._eval_u_on_nodes(u_fn, st_params)
                    self.contact.update_multipliers(u_fn, st_params, u_nodes=u_nodes)
            else:
                u_nodes = None
                if self.elasticity is not None:
                    u_nodes = self.elasticity._eval_u_on_nodes(u_fn, target_params)
                self.contact.update_multipliers(u_fn, target_params, u_nodes=u_nodes)
        if self.bcs:
            if staged_updates:
                for st_params in staged_updates:
                    for bc in self.bcs:
                        bc.update_multipliers(u_fn, st_params)
            else:
                for bc in self.bcs:
                    bc.update_multipliers(u_fn, target_params)

    def set_coeffs(self, w_int: Optional[float] = None,
                   w_cn: Optional[float] = None,
                   w_ct: Optional[float] = None):
        if w_int is not None:
            self.w_int.assign(tf.cast(w_int, self.dtype))
        if w_cn is not None:
            self.w_cn.assign(tf.cast(w_cn, self.dtype))
        if w_ct is not None:
            self.w_ct.assign(tf.cast(w_ct, self.dtype))
