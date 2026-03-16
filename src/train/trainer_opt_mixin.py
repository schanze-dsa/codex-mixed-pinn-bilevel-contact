# -*- coding: utf-8 -*-
"""Optimization/loss-step mixin extracted from Trainer to reduce trainer.py size."""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from model.loss_energy import TotalEnergy
from train.loss_weights import combine_loss, update_loss_weights


def _compute_uncertainty_proxy_sigma(
    u_pred: tf.Tensor,
    residual_scalar: tf.Tensor,
    *,
    proxy_scale: float = 1.0,
    eps: float = 1.0e-6,
) -> tf.Tensor:
    """Build residual-driven sigma proxy from predicted displacement magnitude."""

    u_pred = tf.cast(u_pred, tf.float32)
    residual_scalar = tf.cast(residual_scalar, tf.float32)
    umag = tf.sqrt(tf.reduce_sum(tf.square(u_pred), axis=1, keepdims=True) + tf.cast(eps, tf.float32))
    umag_mean = tf.reduce_mean(umag) + tf.cast(eps, tf.float32)
    sigma = tf.cast(proxy_scale, tf.float32) * residual_scalar * (umag / umag_mean) + tf.cast(eps, tf.float32)
    return sigma


def capped_continuation_update(
    eps_n: float,
    k_t: float,
    *,
    eps_factor: float = 0.7,
    k_t_factor: float = 1.3,
):
    """Apply continuation update with hard per-step caps."""

    eps_scale = max(0.7, float(eps_factor))
    kt_scale = max(0.0, min(float(k_t_factor), 1.3))
    return float(eps_n) * eps_scale, float(k_t) * kt_scale


def inject_bilevel_diagnostics(stats: Dict[str, Any], diagnostics: Mapping[str, Any]) -> Dict[str, Any]:
    """Inject strict-bilevel diagnostics into trainer stats with canonical keys."""

    if stats is None:
        stats = {}
    diagnostics = diagnostics or {}
    key_map = {
        "inner_fn_norm": "fn_norm",
        "inner_ft_norm": "ft_norm",
        "inner_cone_violation": "cone_violation",
        "inner_max_penetration": "max_penetration",
        "inner_fallback_used": "fallback_used",
        "inner_converged": "converged",
        "inner_skip_batch": "skip_batch",
        "inner_convergence_rate": "inner_convergence_rate",
        "inner_fallback_rate": "inner_fallback_rate",
        "inner_skip_rate": "inner_skip_rate",
        "continuation_frozen": "continuation_frozen",
        "continuation_freeze_events": "continuation_freeze_events",
        "ift_linear_residual": "ift_linear_residual",
        "grad_u_norm": "grad_u_norm",
        "grad_sigma_norm": "grad_sigma_norm",
    }
    for out_key, in_key in key_map.items():
        if in_key not in diagnostics:
            continue
        value = diagnostics[in_key]
        try:
            if isinstance(value, tf.Tensor):
                if value.shape.rank == 0:
                    stats[out_key] = float(tf.cast(value, tf.float32).numpy())
                else:
                    stats[out_key] = value
            else:
                stats[out_key] = float(value)
        except Exception:
            stats[out_key] = value
    return stats


class TrainerOptMixin:
    _STRICT_MIXED_ALLOWED_KEYS = frozenset(
        {
            "E_cn",
            "E_ct",
            "E_eq",
            "E_bc",
            "E_tight",
            "E_data",
            "E_smooth",
            "E_unc",
            "E_reg",
        }
    )
    _STRICT_MIXED_DISABLED_KEYS = frozenset(
        {
            "E_int",
            "E_sigma",
            "E_bi",
            "E_ed",
            "path_penalty_total",
            "fric_path_penalty_total",
            "R_contact_comp",
            "R_fric_comp",
        }
    )
    _CONTACT_BACKEND_AUTO = "auto"
    _CONTACT_BACKEND_LEGACY = "legacy_alm"
    _CONTACT_BACKEND_INNER = "inner_solver"
    _VALID_CONTACT_BACKENDS = frozenset(
        {
            _CONTACT_BACKEND_AUTO,
            _CONTACT_BACKEND_LEGACY,
            _CONTACT_BACKEND_INNER,
        }
    )

    @staticmethod
    def _stat_as_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        try:
            if isinstance(value, tf.Tensor):
                if value.dtype == tf.string:
                    value = value.numpy().decode("utf-8")
                elif value.shape.rank == 0:
                    value = value.numpy()
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _stat_as_text(value: Any, default: str = "") -> str:
        if value is None:
            return default
        try:
            if isinstance(value, tf.Tensor):
                if value.dtype == tf.string:
                    return value.numpy().decode("utf-8")
                if value.shape.rank == 0:
                    return str(value.numpy())
            return str(value)
        except Exception:
            return default

    def _resolve_bilevel_objective_route(self) -> str:
        flags = getattr(self, "_mixed_phase_flags", {}) or {}
        phase_name = str(flags.get("phase_name", "phase0") or "phase0").strip().lower()
        normal_ift = bool(flags.get("normal_ift_enabled", False))
        tangential_ift = bool(flags.get("tangential_ift_enabled", False))
        if phase_name in {"", "phase0"}:
            return "legacy"
        if tangential_ift:
            raise NotImplementedError(
                "P0 only supports forward-only or normal-ready strict mixed routes; tangential/full IFT remains disabled."
            )
        if normal_ift:
            return "normal_ready"
        return "forward_only"

    def _resolve_contact_backend(self) -> str:
        route_mode = self._resolve_bilevel_objective_route()
        requested = str(getattr(getattr(self, "cfg", None), "contact_backend", self._CONTACT_BACKEND_AUTO) or "")
        requested = requested.strip().lower() or self._CONTACT_BACKEND_AUTO
        if requested not in self._VALID_CONTACT_BACKENDS:
            valid = ", ".join(sorted(self._VALID_CONTACT_BACKENDS))
            raise ValueError(f"Unsupported contact_backend '{requested}'. Expected one of: {valid}.")

        default_backend = (
            self._CONTACT_BACKEND_LEGACY
            if route_mode == "legacy"
            else self._CONTACT_BACKEND_INNER
        )
        if requested == self._CONTACT_BACKEND_AUTO:
            return default_backend
        if requested != default_backend:
            raise ValueError(
                f"contact_backend='{requested}' is incompatible with strict_route_mode='{route_mode}' "
                f"(expected '{default_backend}')."
            )
        return requested

    def _apply_route_weight_overrides(self, route_mode: str) -> Dict[str, float]:
        if route_mode == "legacy":
            self._active_weight_overrides = {}
            return self._active_weight_overrides
        overrides = {key: 0.0 for key in self._STRICT_MIXED_DISABLED_KEYS}
        self._active_weight_overrides = overrides
        return overrides

    def _evaluate_total_objective(
        self,
        total: TotalEnergy,
        params: Dict[str, Any],
        *,
        stress_fn=None,
        tape=None,
    ):
        route_mode = self._resolve_bilevel_objective_route()
        self._apply_route_weight_overrides(route_mode)
        if route_mode == "legacy":
            Pi, parts, stats = total.energy(self.model.u_fn, params=params, tape=tape, stress_fn=stress_fn)
        else:
            if not hasattr(total, "strict_mixed_objective"):
                raise RuntimeError("TotalEnergy.strict_mixed_objective() is required for strict mixed bilevel mode.")
            Pi, parts, stats = total.strict_mixed_objective(
                self.model.u_fn,
                params=params,
                tape=tape,
                stress_fn=stress_fn,
            )
        if stats is None:
            stats = {}
        else:
            stats = dict(stats)
        if tf.inside_function():
            stats["strict_route_mode"] = tf.constant(route_mode, dtype=tf.string)
        else:
            stats["strict_route_mode"] = route_mode
        return Pi, parts, stats

    def _accumulate_strict_bilevel_stats(
        self,
        stats: Optional[Mapping[str, Any]],
        *,
        route_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {} if stats is None else dict(stats)
        if route_mode is None:
            route_mode = self._stat_as_text(out.get("strict_route_mode"), "legacy")
        if route_mode == "legacy":
            out["strict_route_mode"] = route_mode
            out["continuation_frozen"] = float(bool(getattr(self, "_contact_hardening_frozen", False)))
            out["continuation_freeze_events"] = float(int(getattr(self, "_continuation_freeze_events", 0) or 0))
            return out

        counters = getattr(self, "_strict_bilevel_stats", None)
        if not isinstance(counters, dict):
            counters = {"total": 0, "converged": 0, "fallback": 0, "skipped": 0}
            self._strict_bilevel_stats = counters

        converged = self._stat_as_float(
            out.get("inner_converged", out.get("converged", 0.0)),
            0.0,
        ) > 0.5
        fallback = self._stat_as_float(
            out.get("inner_fallback_used", out.get("fallback_used", 0.0)),
            0.0,
        ) > 0.5
        skipped = self._stat_as_float(
            out.get("inner_skip_batch", out.get("mixed_strict_skipped", 0.0)),
            0.0,
        ) > 0.5

        counters["total"] = int(counters.get("total", 0)) + 1
        counters["converged"] = int(counters.get("converged", 0)) + int(converged)
        counters["fallback"] = int(counters.get("fallback", 0)) + int(fallback)
        counters["skipped"] = int(counters.get("skipped", 0)) + int(skipped)
        total_count = max(1, int(counters["total"]))

        if skipped or fallback or (not converged):
            self._strict_bilevel_freeze_requested = True

        out["inner_convergence_rate"] = float(counters["converged"]) / float(total_count)
        out["inner_fallback_rate"] = float(counters["fallback"]) / float(total_count)
        out["inner_skip_rate"] = float(counters["skipped"]) / float(total_count)
        out["strict_route_mode"] = route_mode
        out["continuation_frozen"] = float(bool(getattr(self, "_contact_hardening_frozen", False)))
        out["continuation_freeze_events"] = float(int(getattr(self, "_continuation_freeze_events", 0) or 0))
        return out

    def _collect_trainable_variables(self):
        m = self.model

        if hasattr(m, "trainable_variables") and m.trainable_variables:
            return m.trainable_variables

        vars_list = []
        common_attrs = [
            "field",
            "net",
            "model",
            "encoder",
            "cond_encoder",
            "cond_enc",
            "embed",
            "embedding",
            "backbone",
            "trunk",
            "head",
            "blocks",
            "layers",
        ]
        for name in common_attrs:
            sub = getattr(m, name, None)
            if sub is None:
                continue
            if hasattr(sub, "trainable_variables"):
                vars_list += list(sub.trainable_variables)
            elif isinstance(sub, (list, tuple)):
                for layer in sub:
                    if hasattr(layer, "trainable_variables"):
                        vars_list += list(layer.trainable_variables)

        seen, out = set(), []
        for v in vars_list:
            if v is None:
                continue
            vid = id(v)
            if vid in seen:
                continue
            seen.add(vid)
            out.append(v)

        if not out:
            try:
                out = list(tf.compat.v1.trainable_variables())
            except Exception:
                out = []
        if not out:
            raise RuntimeError(
                "[trainer] Could not find trainable variables. Ensure model submodules are built."
            )
        return out

    def _uncertainty_enabled(self) -> bool:
        if float(getattr(self.cfg, "uncertainty_loss_weight", 0.0) or 0.0) <= 0.0:
            return False
        if int(getattr(self.cfg, "uncertainty_sample_points", 0) or 0) <= 0:
            return False
        try:
            out_dim = int(getattr(self.model.field.cfg, "uncertainty_out_dim", 0) or 0)
        except Exception:
            out_dim = 0
        return out_dim > 0 and hasattr(self.model, "uvar_fn") and self.elasticity is not None

    def _compute_uncertainty_proxy_loss_tf(
        self,
        params: Dict[str, Any],
        parts: Dict[str, tf.Tensor],
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        if not self._uncertainty_enabled():
            z = tf.cast(0.0, tf.float32)
            return z, {"unc_sigma_mean": z, "unc_proxy_mean": z}

        n_req = int(getattr(self.cfg, "uncertainty_sample_points", 0) or 0)
        X_all = tf.cast(self.elasticity.X_vol_tf, tf.float32)
        n = tf.minimum(tf.shape(X_all)[0], tf.cast(n_req, tf.int32))
        X = X_all[:n]

        u_mean, log_var = self.model.uvar_fn(X, params)
        lv_min = float(getattr(self.cfg, "uncertainty_logvar_min", -8.0))
        lv_max = float(getattr(self.cfg, "uncertainty_logvar_max", 6.0))
        log_var = tf.clip_by_value(tf.cast(log_var, tf.float32), lv_min, lv_max)
        sigma_pred = tf.exp(0.5 * log_var)

        e_cn = tf.cast(parts.get("E_cn", tf.cast(0.0, tf.float32)), tf.float32)
        e_ct = tf.cast(parts.get("E_ct", tf.cast(0.0, tf.float32)), tf.float32)
        e_eq = tf.cast(parts.get("E_eq", tf.cast(0.0, tf.float32)), tf.float32)
        residual_scalar = tf.sqrt(tf.maximum(e_cn + e_ct + e_eq, 0.0) + 1.0e-12)
        residual_scalar = residual_scalar / (1.0 + residual_scalar)

        sigma_proxy = _compute_uncertainty_proxy_sigma(
            tf.cast(u_mean, tf.float32),
            residual_scalar,
            proxy_scale=float(getattr(self.cfg, "uncertainty_proxy_scale", 1.0)),
        )
        sigma_proxy = tf.broadcast_to(sigma_proxy, tf.shape(sigma_pred))
        loss_main = tf.reduce_mean(tf.square(sigma_pred - sigma_proxy))
        loss_reg = tf.cast(1.0e-3, tf.float32) * tf.reduce_mean(tf.square(log_var))
        loss_unc = loss_main + loss_reg
        stats = {
            "unc_sigma_mean": tf.reduce_mean(sigma_pred),
            "unc_proxy_mean": tf.reduce_mean(sigma_proxy),
            "unc_residual_scalar": residual_scalar,
        }
        return loss_unc, stats

    def _compute_total_loss(
        self,
        total: TotalEnergy,
        params: Dict[str, Any],
        *,
        adaptive: bool = True,
    ):
        stress_head_enabled = False
        try:
            stress_head_enabled = getattr(self.model.field.cfg, "stress_out_dim", 0) > 0
        except Exception:
            stress_head_enabled = False

        stress_fn = self.model.us_fn if stress_head_enabled and hasattr(self.model, "us_fn") else None

        Pi_raw, parts, stats = self._evaluate_total_objective(total, params, stress_fn=stress_fn, tape=None)
        route_mode = self._stat_as_text(stats.get("strict_route_mode"), "legacy")
        stats = self._accumulate_strict_bilevel_stats(stats, route_mode=route_mode)
        if self._uncertainty_enabled():
            E_unc, unc_stats = self._compute_uncertainty_proxy_loss_tf(params, parts)
            parts["E_unc"] = tf.cast(E_unc, tf.float32)
            stats.update(unc_stats)
        Pi = Pi_raw
        if self.loss_state is not None and adaptive:
            update_loss_weights(self.loss_state, parts, stats)
        weights = self._build_weight_vector()
        weights, floor_diag = self._apply_supervision_contribution_floor(parts, weights)
        stats.update(floor_diag)
        Pi = self._loss_from_parts_and_weights(parts, weights)
        reg = tf.add_n(self.model.losses) if getattr(self.model, "losses", None) else 0.0
        loss = Pi + reg
        return loss, Pi, parts, stats

    def _compute_total_loss_incremental(
        self,
        total: TotalEnergy,
        params: Dict[str, Any],
        *,
        locked_deltas: Optional[tf.Tensor] = None,
        force_then_lock: bool = False,
        adaptive: bool = True,
    ):
        """Compute loss for a single stage with optional lock penalty."""

        del locked_deltas, force_then_lock
        stress_head_enabled = False
        try:
            stress_head_enabled = getattr(self.model.field.cfg, "stress_out_dim", 0) > 0
        except Exception:
            stress_head_enabled = False

        stress_fn = self.model.us_fn if stress_head_enabled and hasattr(self.model, "us_fn") else None

        _, parts, stats = self._evaluate_total_objective(total, params, stress_fn=stress_fn, tape=None)
        route_mode = self._stat_as_text(stats.get("strict_route_mode"), "legacy")
        stats = self._accumulate_strict_bilevel_stats(stats, route_mode=route_mode)
        if self._uncertainty_enabled():
            E_unc, unc_stats = self._compute_uncertainty_proxy_loss_tf(params, parts)
            parts["E_unc"] = tf.cast(E_unc, tf.float32)
            stats.update(unc_stats)

        if self.loss_state is not None and adaptive:
            update_loss_weights(self.loss_state, parts, stats)
        weights = self._build_weight_vector()
        weights, floor_diag = self._apply_supervision_contribution_floor(parts, weights)
        stats.update(floor_diag)
        Pi = self._loss_from_parts_and_weights(parts, weights)
        reg = tf.add_n(self.model.losses) if getattr(self.model, "losses", None) else 0.0
        loss = Pi + reg
        return loss, Pi, parts, stats

    def _loss_from_parts_and_weights(self, parts: Dict[str, tf.Tensor], weights: tf.Tensor) -> tf.Tensor:
        """Combine scalar parts with a fixed weight vector (order follows self._loss_keys)."""

        loss = tf.constant(0.0, dtype=tf.float32)
        for idx, key in enumerate(getattr(self, "_loss_keys", [])):
            if key not in parts:
                continue
            val = parts[key]
            if not isinstance(val, tf.Tensor):
                continue
            if val.shape.rank != 0:
                continue
            loss = loss + tf.cast(weights[idx], tf.float32) * tf.cast(val, tf.float32)
        return loss

    @staticmethod
    def _compute_apply_gradients_kwargs(optimizer: Any) -> Dict[str, Any]:
        """Detect optimizer kwargs once to avoid per-step reflection overhead."""

        if optimizer is None:
            return {}
        try:
            sig = inspect.signature(optimizer.apply_gradients)
        except (TypeError, ValueError, AttributeError):
            return {}
        if "experimental_aggregate_gradients" in sig.parameters:
            return {"experimental_aggregate_gradients": False}
        return {}

    def _build_weight_vector_from_maps(
        self,
        weight_map: Mapping[str, Any],
        sign_map: Optional[Mapping[str, Any]] = None,
    ) -> tf.Tensor:
        keys = getattr(self, "_loss_keys", [])
        if not keys:
            return tf.zeros((0,), dtype=tf.float32)
        sign_map = sign_map or {}
        overrides = getattr(self, "_active_weight_overrides", {}) or {}
        weights = []
        for key in keys:
            if key in overrides:
                w = float(overrides.get(key, 0.0) or 0.0)
            else:
                w = float(weight_map.get(key, 0.0) or 0.0)
            sign = float(sign_map.get(key, 1.0))
            weights.append(w * sign)
        return tf.convert_to_tensor(weights, dtype=tf.float32)

    def _refresh_static_weight_vector(self):
        """Refresh cached weight vector for non-adaptive training."""

        if self.loss_state is not None:
            self._static_weight_vector = None
            return
        self._static_weight_vector = self._build_weight_vector_from_maps(getattr(self, "_base_weights", {}), {})

    def _build_weight_vector(self) -> tf.Tensor:
        """Build a weight vector aligned with self._loss_keys (sign applied)."""

        keys = getattr(self, "_loss_keys", [])
        if not keys:
            return tf.zeros((0,), dtype=tf.float32)

        route_mode = self._resolve_bilevel_objective_route()
        self._apply_route_weight_overrides(route_mode)

        if self.loss_state is not None:
            return self._build_weight_vector_from_maps(
                self.loss_state.current,
                self.loss_state.sign_overrides,
            )

        if getattr(self, "_active_weight_overrides", None):
            return self._build_weight_vector_from_maps(getattr(self, "_base_weights", {}), {})

        cached = getattr(self, "_static_weight_vector", None)
        if cached is None:
            self._refresh_static_weight_vector()
            cached = getattr(self, "_static_weight_vector", None)
        if cached is None:
            return tf.zeros((0,), dtype=tf.float32)
        return cached

    def _loss_key_index(self, key: str) -> Optional[int]:
        try:
            return list(getattr(self, "_loss_keys", [])).index(key)
        except ValueError:
            return None

    def _apply_supervision_contribution_floor(
        self,
        parts: Mapping[str, tf.Tensor],
        weights: tf.Tensor,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        dtype = tf.float32
        zero = tf.cast(0.0, dtype)
        idx_data = self._loss_key_index("E_data")
        if idx_data is None:
            return weights, {}

        data_weight = tf.cast(weights[idx_data], dtype)
        data_loss = tf.cast(parts.get("E_data", zero), dtype)
        ratio = tf.cast(
            float(getattr(self.cfg, "supervision_contribution_floor_ratio", 0.0) or 0.0),
            dtype,
        )
        enabled = bool(getattr(self.cfg, "supervision_contribution_floor_enabled", False))

        diag = {
            "data_eff_w": data_weight,
            "data_floor_active": zero,
            "data_floor_target": zero,
            "data_phys_contrib": zero,
            "data_eff_contrib": data_weight * data_loss,
        }
        if (not enabled) or float(getattr(self.cfg, "supervision_contribution_floor_ratio", 0.0) or 0.0) <= 0.0:
            return weights, diag

        phys_contrib = zero
        for key in ("E_sigma", "E_ct"):
            idx = self._loss_key_index(key)
            if idx is None:
                continue
            phys_contrib = phys_contrib + tf.abs(tf.cast(weights[idx], dtype)) * tf.abs(
                tf.cast(parts.get(key, zero), dtype)
            )

        target = ratio * phys_contrib
        safe_data_loss = tf.maximum(tf.abs(data_loss), tf.cast(1.0e-12, dtype))
        has_data = tf.abs(data_loss) > tf.cast(1.0e-12, dtype)
        required_weight = tf.where(has_data, target / safe_data_loss, data_weight)
        eff_weight = tf.where(has_data, tf.maximum(data_weight, required_weight), data_weight)
        floor_active = tf.cast(
            tf.logical_and(
                has_data,
                eff_weight > data_weight + tf.cast(1.0e-12, dtype),
            ),
            dtype,
        )
        weights = tf.tensor_scatter_nd_update(
            tf.cast(weights, dtype),
            indices=tf.constant([[idx_data]], dtype=tf.int32),
            updates=tf.reshape(eff_weight, (1,)),
        )
        diag = {
            "data_eff_w": eff_weight,
            "data_floor_active": floor_active,
            "data_floor_target": target,
            "data_phys_contrib": phys_contrib,
            "data_eff_contrib": eff_weight * data_loss,
        }
        return weights, diag

    @tf.function(reduce_retracing=True)
    def _compiled_step(self, params: Dict[str, Any], weights: tf.Tensor):
        """Compiled forward+backward for the standard (non-incremental) path."""

        train_vars = self._train_vars
        opt = self.optimizer
        total = self._total_ref

        stress_head_enabled = False
        try:
            stress_head_enabled = getattr(self.model.field.cfg, "stress_out_dim", 0) > 0
        except Exception:
            stress_head_enabled = False
        stress_fn = self.model.us_fn if stress_head_enabled and hasattr(self.model, "us_fn") else None

        with tf.GradientTape() as tape:
            _, parts, stats = self._evaluate_total_objective(total, params, stress_fn=stress_fn, tape=None)
            if self._uncertainty_enabled():
                E_unc, unc_stats = self._compute_uncertainty_proxy_loss_tf(params, parts)
                parts["E_unc"] = tf.cast(E_unc, tf.float32)
                stats.update(unc_stats)
            eff_weights, floor_diag = self._apply_supervision_contribution_floor(parts, weights)
            stats.update(floor_diag)
            loss_no_reg = self._loss_from_parts_and_weights(parts, eff_weights)
            reg = tf.add_n(self.model.losses) if getattr(self.model, "losses", None) else 0.0
            loss_total = loss_no_reg + reg

            use_loss_scale = hasattr(opt, "get_scaled_loss")
            if use_loss_scale:
                scaled_loss = opt.get_scaled_loss(loss_total)

        if use_loss_scale:
            scaled_grads = tape.gradient(scaled_loss, train_vars)
            grads = opt.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss_total, train_vars)

        return loss_total, loss_no_reg, parts, stats, grads

    @tf.function(reduce_retracing=True)
    def _compiled_stage_step(
        self,
        params: Dict[str, Any],
        weights: tf.Tensor,
    ):
        """Compiled forward+backward for one incremental stage."""

        train_vars = self._train_vars
        opt = self.optimizer
        total = self._total_ref

        stress_head_enabled = False
        try:
            stress_head_enabled = getattr(self.model.field.cfg, "stress_out_dim", 0) > 0
        except Exception:
            stress_head_enabled = False
        stress_fn = self.model.us_fn if stress_head_enabled and hasattr(self.model, "us_fn") else None

        with tf.GradientTape() as tape:
            _, parts, stats = self._evaluate_total_objective(total, params, stress_fn=stress_fn, tape=None)
            if self._uncertainty_enabled():
                E_unc, unc_stats = self._compute_uncertainty_proxy_loss_tf(params, parts)
                parts["E_unc"] = tf.cast(E_unc, tf.float32)
                stats.update(unc_stats)

            eff_weights, floor_diag = self._apply_supervision_contribution_floor(parts, weights)
            stats.update(floor_diag)
            loss_no_reg = self._loss_from_parts_and_weights(parts, eff_weights)
            reg = tf.add_n(self.model.losses) if getattr(self.model, "losses", None) else 0.0
            loss_total = loss_no_reg + reg

            use_loss_scale = hasattr(opt, "get_scaled_loss")
            if use_loss_scale:
                scaled_loss = opt.get_scaled_loss(loss_total)

        if use_loss_scale:
            scaled_grads = tape.gradient(scaled_loss, train_vars)
            grads = opt.get_unscaled_gradients(scaled_grads)
        else:
            grads = tape.gradient(loss_total, train_vars)

        return loss_total, loss_no_reg, parts, stats, grads

    def _train_step(
        self,
        total,
        preload_case: Dict[str, np.ndarray],
        *,
        step: Optional[int] = None,
    ):
        return self._train_step_incremental(total, preload_case, step=step)

    def _train_step_incremental(
        self,
        total,
        preload_case: Dict[str, np.ndarray],
        *,
        step: Optional[int] = None,
    ):
        """Incremental Mode A: solve stages sequentially with per-stage updates."""

        opt = self.optimizer
        train_vars = self._train_vars or self._collect_trainable_variables()
        if self._total_ref is None:
            self._total_ref = total

        params_full = self._make_preload_params(preload_case)
        stage_count = self._get_stage_count(params_full)
        active_count = self._active_stage_count(step, stage_count)

        stage_mode = str(getattr(self.cfg.total_cfg, "preload_stage_mode", "") or "")
        stage_mode = stage_mode.strip().lower().replace("-", "_")
        force_then_lock = stage_mode == "force_then_lock"

        if self.contact is not None:
            self.contact.reset_multipliers(reset_reference=True)

        stage_inner_steps = max(1, int(getattr(self.cfg, "stage_inner_steps", 1)))
        stage_alm_every = max(1, int(getattr(self.cfg, "stage_alm_every", 1)))
        use_delta_st = bool(getattr(self.contact.friction.cfg, "use_delta_st", False)) if self.contact else False

        Pi = tf.constant(0.0, dtype=tf.float32)
        parts: Dict[str, tf.Tensor] = {}
        stats: Dict[str, Any] = {}
        grad_norm = tf.constant(0.0, dtype=tf.float32)

        for stage_idx in range(active_count):
            stage_params = self._extract_stage_params(params_full, stage_idx, keep_context=True)
            if force_then_lock:
                stage_last = stage_params.get("stage_last")
                if stage_last is not None and "P" in stage_params:
                    P_cum = tf.convert_to_tensor(stage_params["P"], dtype=tf.float32)
                    stage_params = dict(stage_params)
                    stage_params["P_cumulative"] = P_cum
                    stage_params["P"] = P_cum * tf.cast(stage_last, P_cum.dtype)

            prev_params = None
            if self.contact is not None and use_delta_st and stage_idx > 0:
                prev_params = self._extract_stage_params(params_full, stage_idx - 1, keep_context=True)
                if force_then_lock:
                    prev_last = prev_params.get("stage_last")
                    if prev_last is not None and "P" in prev_params:
                        P_cum_prev = tf.convert_to_tensor(prev_params["P"], dtype=tf.float32)
                        prev_params = dict(prev_params)
                        prev_params["P_cumulative"] = P_cum_prev
                        prev_params["P"] = P_cum_prev * tf.cast(prev_last, P_cum_prev.dtype)

            if prev_params is not None:
                u_nodes = None
                if self.elasticity is not None:
                    u_nodes = self.elasticity._eval_u_on_nodes(self.model.u_fn, prev_params)
                self.contact.friction.capture_reference(self.model.u_fn, prev_params, u_nodes=u_nodes)

            for _ in range(stage_inner_steps):
                weight_vec = self._build_weight_vector()
                loss, loss_no_reg, parts, stats, grads = self._compiled_stage_step(stage_params, weight_vec)
                route_mode = self._stat_as_text(
                    stats.get("strict_route_mode") if isinstance(stats, Mapping) else None,
                    "legacy",
                )
                stats = self._accumulate_strict_bilevel_stats(stats, route_mode=route_mode)

                if self.loss_state is not None:
                    update_loss_weights(self.loss_state, parts, stats)
                    Pi = combine_loss(parts, self.loss_state)
                else:
                    Pi = loss_no_reg

                if not any(g is not None for g in grads):
                    raise RuntimeError("[trainer] All gradients are None in incremental step.")

                non_none = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
                g_list, v_list = zip(*non_none)
                grad_norm = self._safe_global_norm(g_list)
                grad_norm_val = float(grad_norm.numpy())
                grad_norm_finite = bool(np.isfinite(grad_norm_val))

                clip_norm = (
                    getattr(self, "clip_grad_norm", None)
                    or getattr(self, "grad_clip_norm", None)
                    or getattr(self.cfg, "clip_grad_norm", None)
                    or getattr(self.cfg, "grad_clip_norm", None)
                )
                if clip_norm is not None and float(clip_norm) > 0.0 and grad_norm_finite:
                    g_list = self._safe_clip_by_global_norm(g_list, clip_norm, grad_norm)

                loss_val = float(tf.cast(loss, tf.float32).numpy())
                if not (np.isfinite(loss_val) and grad_norm_finite):
                    continue

                opt.apply_gradients(zip(g_list, v_list), **getattr(self, "_apply_gradients_kwargs", {}))

            if stage_alm_every > 0 and ((stage_idx + 1) % stage_alm_every == 0):
                total.update_multipliers(self.model.u_fn, params=stage_params)

            if use_delta_st and self.contact is not None:
                self.contact.friction.commit_reference()

        return Pi, parts, stats, grad_norm

    def _flatten_tensor_list(
        self, tensors: Sequence[Optional[tf.Tensor]], sizes: Sequence[int]
    ) -> tf.Tensor:
        flats: List[tf.Tensor] = []
        for tensor, size in zip(tensors, sizes):
            if tensor is None:
                flats.append(tf.zeros((size,), dtype=tf.float32))
            else:
                flats.append(tf.reshape(tf.cast(tensor, tf.float32), (-1,)))
        if not flats:
            return tf.zeros((0,), dtype=tf.float32)
        return tf.concat(flats, axis=0)

    def _safe_global_norm(self, grads: Sequence[tf.Tensor]) -> tf.Tensor:
        """Compute global norm without densifying IndexedSlices."""

        def _squared_norm(g: tf.Tensor) -> tf.Tensor:
            if isinstance(g, tf.IndexedSlices):
                values = tf.cast(g.values, tf.float32)
                return tf.reduce_sum(tf.square(values))
            values = tf.cast(g, tf.float32)
            return tf.reduce_sum(tf.square(values))

        squared = [_squared_norm(g) for g in grads]
        if not squared:
            return tf.constant(0.0, dtype=tf.float32)
        return tf.sqrt(tf.add_n(squared))

    def _safe_clip_by_global_norm(
        self, grads: Sequence[tf.Tensor], clip_norm: float, global_norm: tf.Tensor
    ) -> List[tf.Tensor]:
        """Clip gradients using a precomputed global norm while keeping IndexedSlices sparse."""

        clip_norm = tf.cast(clip_norm, tf.float32)
        global_norm = tf.cast(global_norm, tf.float32)
        safe_norm = tf.maximum(global_norm, tf.constant(1e-12, dtype=tf.float32))
        scale = tf.minimum(1.0, clip_norm / safe_norm)

        clipped: List[tf.Tensor] = []
        for g in grads:
            if isinstance(g, tf.IndexedSlices):
                clipped.append(tf.IndexedSlices(g.values * scale, g.indices, g.dense_shape))
            else:
                clipped.append(g * scale)
        return clipped

    def _assign_from_flat(
        self, flat_tensor: tf.Tensor, variables: Sequence[tf.Variable], sizes: Sequence[int]
    ):
        offset = 0
        for var, size in zip(variables, sizes):
            next_offset = offset + size
            slice_tensor = tf.reshape(flat_tensor[offset:next_offset], var.shape)
            var.assign(tf.cast(slice_tensor, var.dtype))
            offset = next_offset
