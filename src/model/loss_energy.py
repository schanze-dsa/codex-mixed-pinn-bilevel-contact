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
from physics.contact.contact_operator import (
    ContactOperator,
    ContactOperatorConfig,
    traction_matching_terms,
)
from physics.boundary_conditions import BoundaryPenalty, BoundaryConfig, traction_bc_residual
from physics.tightening_model import NutTighteningPenalty, TighteningConfig
from physics.traction_utils import normal_tangential_components


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


def traction_bc_residual_from_model(model, X, params, normals, target_t):
    """Compute mixed traction BC residual using the model stress head only."""

    sigma_vec = model.sigma_fn(X, params)
    return traction_bc_residual(sigma_vec, normals, target_t)


def traction_matching_residual(sigma_s, sigma_m, n, t1, t2, inner_result):
    """Mixed contact residual via traction matching against inner solve result."""

    return traction_matching_terms(sigma_s, sigma_m, n, t1, t2, inner_result)


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
    w_data: float = 0.0
    w_smooth: float = 0.0
    sigma_ref: float = 1.0
    data_smoothing_k: int = 0
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
    STRICT_MIXED_ACTIVE_KEYS = (
        "E_cn",
        "E_ct",
        "E_eq",
        "E_bc",
        "E_tight",
        "E_data",
        "E_smooth",
        "E_unc",
        "E_reg",
    )
    STRICT_MIXED_ZERO_KEYS = (
        "E_int",
        "E_sigma",
        "E_bi",
        "E_ed",
        "path_penalty_total",
        "fric_path_penalty_total",
    )

    def __init__(self, cfg: Optional[TotalConfig] = None):
        self.cfg = cfg or TotalConfig()
        self.dtype = tf.float32 if self.cfg.dtype == "float32" else tf.float64
        self.elasticity: Optional[ElasticityResidual] = None
        self.contact: Optional[ContactOperator] = None
        self.bcs: List[BoundaryPenalty] = []
        self.tightening: Optional[NutTighteningPenalty] = None
        self._ensure_weight_vars()
        self._built = False
        self.mixed_bilevel_flags = {
            "phase_name": "phase0",
            "normal_ift_enabled": False,
            "tangential_ift_enabled": False,
            "detach_inner_solution": True,
        }
        self._strict_mixed_last_active = False

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
        if not hasattr(self, "w_data"):
            self.w_data = tf.Variable(getattr(self.cfg, "w_data", 0.0), dtype=self.dtype, trainable=False, name="w_data")
        if not hasattr(self, "w_smooth"):
            self.w_smooth = tf.Variable(getattr(self.cfg, "w_smooth", 0.0), dtype=self.dtype, trainable=False, name="w_smooth")

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

    def set_mixed_bilevel_flags(self, flags: Optional[Dict[str, object]] = None):
        """Attach trainer-resolved mixed-bilevel phase flags to this total-energy assembly."""

        merged = {
            "phase_name": "phase0",
            "normal_ift_enabled": False,
            "tangential_ift_enabled": False,
            "detach_inner_solution": True,
        }
        if isinstance(flags, dict):
            merged.update(flags)
        self.mixed_bilevel_flags = merged
        self._strict_mixed_last_active = False

    def _strict_mixed_requested(self) -> bool:
        phase_name = str(self.mixed_bilevel_flags.get("phase_name", "phase0") or "phase0").strip().lower()
        return phase_name not in {"", "phase0"}

    def _strict_mixed_skip_stats(self, reason: str) -> Dict[str, tf.Tensor]:
        return {
            "mixed_strict_active": tf.cast(0.0, self.dtype),
            "mixed_strict_skipped": tf.cast(1.0, self.dtype),
            "inner_converged": tf.cast(0.0, self.dtype),
            "inner_skip_batch": tf.cast(1.0, self.dtype),
            "skip_batch": tf.cast(1.0, self.dtype),
            "inner_fallback_used": tf.cast(0.0, self.dtype),
            "mixed_strict_skip_reason": tf.constant(str(reason), dtype=tf.string),
        }

    def _strict_mixed_contact_terms(
        self,
        u_fn,
        params,
        *,
        u_nodes: Optional[tf.Tensor] = None,
        stress_fn=None,
    ) -> Tuple[bool, Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        dtype = self.dtype
        zero = tf.cast(0.0, dtype)

        if self.contact is None or not self._strict_mixed_requested():
            self._strict_mixed_last_active = False
            return False, {}, {}

        stress_fn_contact = self._resolve_bound_variant(stress_fn, "us_fn_pointwise")
        if stress_fn_contact is None:
            self._strict_mixed_last_active = False
            return False, {}, self._strict_mixed_skip_stats("stress_fn_missing")
        if not hasattr(self.contact, "strict_mixed_inputs") or not hasattr(self.contact, "solve_strict_inner"):
            self._strict_mixed_last_active = False
            return False, {}, self._strict_mixed_skip_stats("strict_contact_adapter_missing")

        strict_inputs = self.contact.strict_mixed_inputs(u_fn, params, u_nodes=u_nodes)
        inner_result = self.contact.solve_strict_inner(
            u_fn,
            params,
            u_nodes=u_nodes,
            strict_inputs=strict_inputs,
        )
        detach_inner = bool(self.mixed_bilevel_flags.get("detach_inner_solution", True))

        lambda_n = tf.cast(inner_result.state.lambda_n, dtype)
        lambda_t = tf.cast(inner_result.state.lambda_t, dtype)
        traction_vec = tf.cast(inner_result.traction_vec, dtype)
        ds_t = tf.cast(strict_inputs["ds_t"], dtype)
        if detach_inner:
            lambda_n = tf.stop_gradient(lambda_n)
            lambda_t = tf.stop_gradient(lambda_t)
            traction_vec = tf.stop_gradient(traction_vec)
            ds_t = tf.stop_gradient(ds_t)

        _, sigma_s = stress_fn_contact(strict_inputs["xs"], params)
        _, sigma_m = stress_fn_contact(strict_inputs["xm"], params)
        sigma_s = tf.cast(sigma_s, dtype)
        sigma_m = tf.cast(sigma_m, dtype)

        basis = tf.stack([strict_inputs["t1"], strict_inputs["t2"]], axis=1)
        inner_for_match = type("StrictInnerMatch", (), {"traction_vec": traction_vec})()
        rs, rm = traction_matching_residual(
            sigma_s,
            sigma_m,
            strict_inputs["normals"],
            strict_inputs["t1"],
            strict_inputs["t2"],
            inner_for_match,
        )
        rs = tf.cast(rs, dtype)
        rm = tf.cast(rm, dtype)

        rs_n, rs_t = normal_tangential_components(rs, strict_inputs["normals"], basis)
        rm_n, rm_t = normal_tangential_components(rm, strict_inputs["normals"], basis)
        rs_n = tf.squeeze(tf.cast(rs_n, dtype), axis=-1)
        rm_n = tf.squeeze(tf.cast(rm_n, dtype), axis=-1)
        rs_t = tf.cast(rs_t, dtype)
        rm_t = tf.cast(rm_t, dtype)

        weights = tf.cast(strict_inputs["weights"], dtype)
        denom = tf.reduce_sum(weights) + tf.cast(1.0e-12, dtype)
        rn_sq = tf.square(rs_n) + tf.square(rm_n)
        rt_sq = tf.reduce_sum(tf.square(rs_t), axis=1) + tf.reduce_sum(tf.square(rm_t), axis=1)
        e_cn = tf.reduce_sum(weights * rn_sq) / (2.0 * denom)
        e_ct = tf.reduce_sum(weights * rt_sq) / (2.0 * denom)

        rt_norm = tf.sqrt(tf.reduce_sum(tf.square(rs_t), axis=1) + tf.cast(1.0e-12, dtype))
        rm_t_norm = tf.sqrt(tf.reduce_sum(tf.square(rm_t), axis=1) + tf.cast(1.0e-12, dtype))
        r_contact = tf.reduce_sum(weights * 0.5 * (tf.abs(rs_n) + tf.abs(rm_n)))
        r_fric = tf.reduce_sum(weights * 0.5 * (rt_norm + rm_t_norm))

        mu = tf.cast(strict_inputs["mu"], dtype)
        eps_bi = tf.cast(1.0e-8, dtype)
        st_norm = tf.sqrt(tf.reduce_sum(tf.square(ds_t), axis=1) + eps_bi)
        bi_raw = mu * tf.maximum(lambda_n, tf.cast(0.0, dtype)) * st_norm - tf.reduce_sum(lambda_t * ds_t, axis=1)
        bi_pos = tf.nn.relu(bi_raw)
        e_bi = tf.reduce_sum(weights * bi_pos * bi_pos) / denom

        diagnostics = dict(inner_result.diagnostics)
        fn_norm = tf.cast(diagnostics.get("fn_norm", zero), dtype)
        ft_norm = tf.cast(diagnostics.get("ft_norm", zero), dtype)
        cone_violation = tf.cast(diagnostics.get("cone_violation", zero), dtype)
        max_penetration = tf.cast(diagnostics.get("max_penetration", zero), dtype)
        fallback_used = tf.cast(diagnostics.get("fallback_used", zero), dtype)
        converged = tf.cast(diagnostics.get("converged", tf.cast(1.0, dtype) - fallback_used), dtype)

        parts = {
            "E_cn": e_cn,
            "E_ct": e_ct,
            "E_bi": e_bi,
            "R_contact_comp": r_contact,
            "R_fric_comp": r_fric,
        }
        stats = {
            "mixed_strict_active": tf.cast(1.0, dtype),
            "mixed_strict_skipped": tf.cast(0.0, dtype),
            "fn_norm": fn_norm,
            "ft_norm": ft_norm,
            "cone_violation": cone_violation,
            "max_penetration": max_penetration,
            "fallback_used": fallback_used,
            "converged": converged,
            "skip_batch": tf.cast(0.0, dtype),
            "inner_converged": converged,
            "inner_skip_batch": tf.cast(0.0, dtype),
            "inner_fn_norm": fn_norm,
            "inner_ft_norm": ft_norm,
            "inner_cone_violation": cone_violation,
            "inner_max_penetration": max_penetration,
            "inner_fallback_used": fallback_used,
            "R_contact_comp": r_contact,
            "R_fric_comp": r_fric,
            "traction_match_n_rms": tf.sqrt(e_cn + tf.cast(1.0e-20, dtype)),
            "traction_match_t_rms": tf.sqrt(e_ct + tf.cast(1.0e-20, dtype)),
        }
        self._strict_mixed_last_active = True
        return True, parts, stats

    def _compute_data_smoothing_terms(
        self,
        X_obs: tf.Tensor,
        U_pred: tf.Tensor,
        data_ref_rms: tf.Tensor,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        dtype = self.dtype
        zero = tf.cast(0.0, dtype)
        k = int(getattr(self.cfg, "data_smoothing_k", 0) or 0)
        if k <= 0 or float(getattr(self.cfg, "w_smooth", 0.0) or 0.0) <= 0.0:
            return zero, {}

        X_obs = tf.cast(tf.reshape(X_obs, (-1, tf.shape(X_obs)[-1])), dtype)
        U_pred = tf.cast(tf.reshape(U_pred, (-1, tf.shape(U_pred)[-1])), dtype)
        n_static = X_obs.shape[0]
        if n_static is not None and int(n_static) <= 1:
            return zero, {}
        if n_static is not None:
            k = min(k, max(int(n_static) - 1, 1))
        if k <= 0:
            return zero, {}

        x2 = tf.reduce_sum(tf.square(X_obs), axis=1, keepdims=True)
        dist2 = x2 - 2.0 * tf.matmul(X_obs, X_obs, transpose_b=True) + tf.transpose(x2)
        dist2 = tf.maximum(dist2, tf.cast(0.0, dtype))
        n = tf.shape(dist2)[0]
        dist2 = dist2 + tf.eye(n, dtype=dtype) * tf.cast(1.0e30, dtype)
        _, nbr_idx = tf.math.top_k(-dist2, k=k)
        nbr_u = tf.gather(U_pred, nbr_idx)
        nbr_mean = tf.reduce_mean(nbr_u, axis=1)
        smooth_res = U_pred - nbr_mean
        smooth_rel = smooth_res / tf.maximum(data_ref_rms, tf.cast(1.0e-12, dtype))
        loss = tf.reduce_mean(tf.square(smooth_rel))
        stats = {
            "data_smooth_rms": tf.sqrt(tf.reduce_mean(tf.square(smooth_res)) + tf.cast(1.0e-20, dtype)),
            "data_smooth_rel_rms": tf.sqrt(tf.reduce_mean(tf.square(smooth_rel)) + tf.cast(1.0e-20, dtype)),
        }
        return loss, stats

    def _compute_data_supervision_terms(self, u_fn, params) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
        dtype = self.dtype
        zero = tf.cast(0.0, dtype)
        if not isinstance(params, dict):
            return zero, zero, {}

        X_obs = params.get("X_obs")
        U_obs = params.get("U_obs")
        if X_obs is None or U_obs is None:
            return zero, zero, {}

        X_obs = tf.cast(tf.convert_to_tensor(X_obs), dtype)
        U_obs = tf.cast(tf.convert_to_tensor(U_obs), dtype)

        U_pred = tf.cast(u_fn(X_obs, params), dtype)
        diff = U_pred - U_obs
        data_ref_rms = tf.sqrt(tf.reduce_mean(tf.square(U_obs)) + tf.cast(1.0e-20, dtype))
        data_ref_rms = tf.maximum(data_ref_rms, tf.cast(1.0e-12, dtype))
        diff_rel = diff / data_ref_rms
        loss = tf.reduce_mean(tf.square(diff_rel))
        loss_smooth, smooth_stats = self._compute_data_smoothing_terms(X_obs, U_pred, data_ref_rms)
        stats = {
            "data_rms": tf.sqrt(tf.reduce_mean(tf.square(diff)) + tf.cast(1.0e-20, dtype)),
            "data_mae": tf.reduce_mean(tf.abs(diff)),
            "data_ref_rms": data_ref_rms,
            "data_rel_rms": tf.sqrt(tf.reduce_mean(tf.square(diff_rel)) + tf.cast(1.0e-20, dtype)),
            "data_rel_mae": tf.reduce_mean(tf.abs(diff_rel)),
            "data_n_obs": tf.cast(tf.shape(X_obs)[0], dtype),
        }
        stats.update(smooth_stats)
        return loss, loss_smooth, stats

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

    def strict_mixed_objective(self, u_fn, params=None, tape=None, stress_fn=None):
        self._ensure_weight_vars()
        if not self._built:
            raise RuntimeError("[TotalEnergy] attach(...) must be called before strict_mixed_objective().")
        params = params or {}
        if isinstance(params, dict) and params.get("stages"):
            Pi, parts, stats = self._residual_staged(u_fn, params["stages"], params, tape, stress_fn=stress_fn)
        else:
            parts, stats = self._compute_parts_residual(
                u_fn,
                params,
                tape,
                stress_fn=stress_fn,
                allow_legacy_contact_fallback=False,
            )
            Pi = self._combine_parts_with_keys(parts, self.STRICT_MIXED_ACTIVE_KEYS)
        for key in self.STRICT_MIXED_ZERO_KEYS:
            if key in parts:
                parts[key] = tf.cast(0.0, self.dtype)
        return Pi, parts, stats

    def _compute_parts(self, u_fn, params, tape=None, stress_fn=None, *, allow_legacy_contact_fallback: bool = True):
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
            "E_data": zero,
            "E_smooth": zero,
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
            strict_requested = self._strict_mixed_requested()
            strict_active, strict_parts, strict_stats = self._strict_mixed_contact_terms(
                u_fn,
                params,
                u_nodes=u_nodes,
                stress_fn=stress_fn,
            )
            stats.update(strict_stats)
            if strict_active:
                for key in ("E_cn", "E_ct", "E_bi", "R_fric_comp", "R_contact_comp"):
                    if key in strict_parts:
                        parts[key] = tf.cast(strict_parts[key], dtype)
            elif (not strict_requested) or allow_legacy_contact_fallback:
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

        E_data, E_smooth, data_stats = self._compute_data_supervision_terms(u_fn, params)
        parts["E_data"] = tf.cast(E_data, dtype)
        parts["E_smooth"] = tf.cast(E_smooth, dtype)
        stats.update(data_stats)

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

    def _compute_parts_residual(
        self,
        u_fn,
        params,
        tape=None,
        stress_fn=None,
        *,
        allow_legacy_contact_fallback: bool = True,
    ):
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
            "E_data": zero,
            "E_smooth": zero,
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
            strict_requested = self._strict_mixed_requested()
            strict_active, strict_parts, strict_stats = self._strict_mixed_contact_terms(
                u_fn,
                params,
                u_nodes=u_nodes,
                stress_fn=stress_fn,
            )
            stats.update(strict_stats)
            if strict_active:
                for key in ("E_cn", "E_ct", "E_bi", "R_fric_comp", "R_contact_comp"):
                    if key in strict_parts:
                        parts[key] = tf.cast(strict_parts[key], dtype)
            elif (not strict_requested) or allow_legacy_contact_fallback:
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

        E_data, E_smooth, data_stats = self._compute_data_supervision_terms(u_fn, params)
        parts["E_data"] = tf.cast(E_data, dtype)
        parts["E_smooth"] = tf.cast(E_smooth, dtype)
        stats.update(data_stats)

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
                # Align residual-path physics stress ordering to stress-head convention:
                # phys: [xx,yy,zz,yz,xz,xy] -> head: [xx,yy,zz,xy,yz,xz].
                n_sigma_comp = sigma_phys_head.shape[-1]
                if n_sigma_comp is None:
                    sigma_cols = tf.shape(sigma_phys_head)[1]

                    def _reorder_sigma():
                        core = tf.stack(
                            [
                                sigma_phys_head[:, 0],
                                sigma_phys_head[:, 1],
                                sigma_phys_head[:, 2],
                                sigma_phys_head[:, 5],
                                sigma_phys_head[:, 3],
                                sigma_phys_head[:, 4],
                            ],
                            axis=1,
                        )
                        return tf.concat([core, sigma_phys_head[:, 6:]], axis=1)

                    sigma_phys_head = tf.cond(
                        sigma_cols >= 6, _reorder_sigma, lambda: sigma_phys_head
                    )
                elif n_sigma_comp >= 6:
                    sigma_phys_core = tf.stack(
                        [
                            sigma_phys_head[:, 0],
                            sigma_phys_head[:, 1],
                            sigma_phys_head[:, 2],
                            sigma_phys_head[:, 5],
                            sigma_phys_head[:, 3],
                            sigma_phys_head[:, 4],
                        ],
                        axis=1,
                    )
                    sigma_phys_head = (
                        sigma_phys_core
                        if n_sigma_comp == 6
                        else tf.concat([sigma_phys_core, sigma_phys_head[:, 6:]], axis=1)
                    )
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
            + self.w_data * parts.get("E_data", tf.cast(0.0, self.dtype))
            + self.w_smooth * parts.get("E_smooth", tf.cast(0.0, self.dtype))
        )

    def _combine_parts_with_keys(self, parts: Dict[str, tf.Tensor], active_keys) -> tf.Tensor:
        active = set(active_keys or ())
        combined = tf.cast(0.0, self.dtype)
        for key, weight in (
            ("E_int", self.w_int),
            ("E_cn", self.w_cn),
            ("E_ct", self.w_ct),
            ("E_bc", self.w_bc),
            ("E_tight", self.w_tight),
            ("E_sigma", self.w_sigma),
            ("E_eq", self.w_eq),
            ("E_reg", self.w_reg),
            ("E_bi", self.w_bi),
            ("E_ed", self.w_ed),
            ("E_unc", self.w_unc),
            ("E_data", self.w_data),
            ("E_smooth", self.w_smooth),
        ):
            if key not in active:
                continue
            combined = combined + weight * parts.get(key, tf.cast(0.0, self.dtype))
        return combined

    def _energy_staged(self, u_fn, stages, root_params, tape=None, stress_fn=None):
        dtype = self.dtype
        keys = ["E_int", "E_cn", "E_ct", "E_bc", "E_tight", "E_sigma", "E_eq", "E_reg", "E_bi", "E_ed", "E_unc", "E_data", "E_smooth"]
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
            stage_tensor_x_obs = stages.get("X_obs")
            stage_tensor_u_obs = stages.get("U_obs")
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
                stacked_x_obs = None
                if stage_tensor_x_obs is not None:
                    stacked_x_obs = tf.convert_to_tensor(stage_tensor_x_obs)
                stacked_u_obs = None
                if stage_tensor_u_obs is not None:
                    stacked_u_obs = tf.convert_to_tensor(stage_tensor_u_obs)
                stage_seq = []
                for idx, (p, z) in enumerate(zip(tf.unstack(stage_tensor_P, axis=0), tf.unstack(stage_tensor_feat, axis=0))):
                    entry = {"P": p, "P_hat": z}
                    if stacked_rank is not None:
                        entry["stage_rank"] = stacked_rank[idx] if stacked_rank.shape.rank == 2 else stacked_rank
                    if stacked_mask is not None:
                        entry["stage_mask"] = stacked_mask[idx]
                    if stacked_last is not None:
                        entry["stage_last"] = stacked_last[idx]
                    if stacked_x_obs is not None:
                        entry["X_obs"] = stacked_x_obs[idx]
                    if stacked_u_obs is not None:
                        entry["U_obs"] = stacked_u_obs[idx]
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
        keys = ["E_int", "E_cn", "E_ct", "E_bc", "E_tight", "E_sigma", "E_eq", "E_reg", "E_bi", "E_ed", "E_unc", "E_data", "E_smooth"]
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
            stage_tensor_x_obs = stages.get("X_obs")
            stage_tensor_u_obs = stages.get("U_obs")
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
                stacked_x_obs = None
                if stage_tensor_x_obs is not None:
                    stacked_x_obs = tf.convert_to_tensor(stage_tensor_x_obs)
                stacked_u_obs = None
                if stage_tensor_u_obs is not None:
                    stacked_u_obs = tf.convert_to_tensor(stage_tensor_u_obs)
                stage_seq = []
                for idx, (p, z) in enumerate(zip(tf.unstack(stage_tensor_P, axis=0), tf.unstack(stage_tensor_feat, axis=0))):
                    entry = {"P": p, "P_hat": z}
                    if stacked_rank is not None:
                        entry["stage_rank"] = stacked_rank[idx] if stacked_rank.shape.rank == 2 else stacked_rank
                    if stacked_mask is not None:
                        entry["stage_mask"] = stacked_mask[idx]
                    if stacked_last is not None:
                        entry["stage_last"] = stacked_last[idx]
                    if stacked_x_obs is not None:
                        entry["X_obs"] = stacked_x_obs[idx]
                    if stacked_u_obs is not None:
                        entry["U_obs"] = stacked_u_obs[idx]
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

        if self.contact is not None and not self._strict_mixed_last_active:
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
