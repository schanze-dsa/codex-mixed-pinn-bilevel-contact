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


class TrainerOptMixin:
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

        Pi_raw, parts, stats = total.energy(self.model.u_fn, params=params, tape=None, stress_fn=stress_fn)
        if self._uncertainty_enabled():
            E_unc, unc_stats = self._compute_uncertainty_proxy_loss_tf(params, parts)
            parts["E_unc"] = tf.cast(E_unc, tf.float32)
            stats.update(unc_stats)
        Pi = Pi_raw
        if self.loss_state is not None:
            if adaptive:
                update_loss_weights(self.loss_state, parts, stats)
            Pi = combine_loss(parts, self.loss_state)
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

        _, parts, stats = total.energy(self.model.u_fn, params=params, tape=None, stress_fn=stress_fn)
        if self._uncertainty_enabled():
            E_unc, unc_stats = self._compute_uncertainty_proxy_loss_tf(params, parts)
            parts["E_unc"] = tf.cast(E_unc, tf.float32)
            stats.update(unc_stats)

        Pi = total._combine_parts(parts)
        if self.loss_state is not None:
            if adaptive:
                update_loss_weights(self.loss_state, parts, stats)
            Pi = combine_loss(parts, self.loss_state)
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
        weights = []
        for key in keys:
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

        if self.loss_state is not None:
            return self._build_weight_vector_from_maps(
                self.loss_state.current,
                self.loss_state.sign_overrides,
            )

        cached = getattr(self, "_static_weight_vector", None)
        if cached is None:
            self._refresh_static_weight_vector()
            cached = getattr(self, "_static_weight_vector", None)
        if cached is None:
            return tf.zeros((0,), dtype=tf.float32)
        return cached

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
            _, parts, stats = total.energy(self.model.u_fn, params=params, tape=None, stress_fn=stress_fn)
            if self._uncertainty_enabled():
                E_unc, unc_stats = self._compute_uncertainty_proxy_loss_tf(params, parts)
                parts["E_unc"] = tf.cast(E_unc, tf.float32)
                stats.update(unc_stats)
            loss_no_reg = self._loss_from_parts_and_weights(parts, weights)
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
            _, parts, stats = total.energy(self.model.u_fn, params=params, tape=None, stress_fn=stress_fn)
            if self._uncertainty_enabled():
                E_unc, unc_stats = self._compute_uncertainty_proxy_loss_tf(params, parts)
                parts["E_unc"] = tf.cast(E_unc, tf.float32)
                stats.update(unc_stats)

            loss_no_reg = self._loss_from_parts_and_weights(parts, weights)
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
