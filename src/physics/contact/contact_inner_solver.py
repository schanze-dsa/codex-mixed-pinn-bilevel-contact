#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stateless strict-bilevel inner-contact solver with explicit state/result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import tensorflow as tf

from .contact_inner_kernel_primitives import (
    check_contact_feasibility,
    compose_contact_traction,
    fb_normal_residual,
    friction_fixed_point_residual,
    project_to_coulomb_disk,
)


@dataclass
class ContactInnerState:
    lambda_n: tf.Tensor
    lambda_t: tf.Tensor
    converged: bool = False
    iters: int = 0
    res_norm: float = 0.0
    fallback_used: bool = False


@dataclass
class ContactInnerResult:
    state: ContactInnerState
    traction_vec: tf.Tensor
    traction_tangent: tf.Tensor
    diagnostics: Dict[str, tf.Tensor]
    linearization: Optional[Dict[str, object]] = None


def _to_float_tensor(x) -> tf.Tensor:
    if isinstance(x, tf.Tensor):
        return tf.cast(x, tf.float32)
    return tf.convert_to_tensor(x, dtype=tf.float32)


def _smooth_penetration_target(g_n: tf.Tensor, eps_n: tf.Tensor) -> tf.Tensor:
    """Smooth approximation of max(0, -g_n) derived from the FB kernel at lambda_n=0."""

    return 0.5 * fb_normal_residual(g_n, tf.zeros_like(g_n), eps_n)


def _max_abs(x: tf.Tensor) -> tf.Tensor:
    x = _to_float_tensor(x)
    return tf.reduce_max(tf.abs(x))


def _python_scalar(value, cast_fn):
    if isinstance(value, tf.Tensor) and tf.executing_eagerly():
        return cast_fn(value.numpy())
    return value


def flatten_contact_state(lambda_n: tf.Tensor, lambda_t: tf.Tensor) -> tf.Tensor:
    """Flatten `[lambda_n, lambda_t]` using a fixed `[N, N*2]` ordering."""

    return tf.concat(
        [
            tf.reshape(tf.cast(lambda_n, tf.float32), (-1,)),
            tf.reshape(tf.cast(lambda_t, tf.float32), (-1,)),
        ],
        axis=0,
    )


def flatten_contact_inputs(g_n: tf.Tensor, ds_t: tf.Tensor) -> tf.Tensor:
    """Flatten `[g_n, ds_t]` using a fixed `[N, N*2]` ordering."""

    return tf.concat(
        [
            tf.reshape(tf.cast(g_n, tf.float32), (-1,)),
            tf.reshape(tf.cast(ds_t, tf.float32), (-1,)),
        ],
        axis=0,
    )


def _flatten_jacobian_block(jacobian: tf.Tensor, output_size: tf.Tensor) -> tf.Tensor:
    return tf.reshape(tf.cast(jacobian, tf.float32), (output_size, -1))


def solve_contact_inner(
    g_n: tf.Tensor,
    ds_t: tf.Tensor,
    normals: tf.Tensor,
    t1: tf.Tensor,
    t2: tf.Tensor,
    *,
    mu,
    eps_n,
    k_t,
    init_state: Optional[ContactInnerState] = None,
    return_linearization: bool = False,
    tol_n: float = 1.0e-5,
    tol_t: float = 1.0e-5,
    max_inner_iters: int = 8,
    damping: float = 1.0,
) -> ContactInnerResult:
    """Solve a geometry-driven strict-bilevel inner contact state without hidden mutable state."""

    g_n = _to_float_tensor(g_n)
    ds_t = _to_float_tensor(ds_t)
    normals = _to_float_tensor(normals)
    t1 = _to_float_tensor(t1)
    t2 = _to_float_tensor(t2)
    mu = _to_float_tensor(mu)
    eps_n = _to_float_tensor(eps_n)
    k_t = _to_float_tensor(k_t)
    damping = float(max(0.0, min(1.0, damping)))

    target_lambda_n = tf.maximum(_smooth_penetration_target(g_n, eps_n), 0.0)
    if init_state is not None:
        init_lambda_n = _to_float_tensor(init_state.lambda_n)
        init_lambda_t = _to_float_tensor(init_state.lambda_t)
    else:
        init_lambda_n = tf.zeros_like(target_lambda_n)
        init_lambda_t = tf.zeros_like(ds_t)

    lambda_n = init_lambda_n
    lambda_t = init_lambda_t
    normal_step = tf.zeros_like(lambda_n)
    tangential_step = tf.zeros_like(lambda_t)
    converged = tf.constant(False)
    iters = tf.constant(0, dtype=tf.int32)
    res_norm = tf.constant(float("inf"), dtype=tf.float32)
    done = tf.constant(False)
    for it in range(int(max_inner_iters)):
        next_lambda_n = tf.maximum(
            (1.0 - damping) * lambda_n + damping * target_lambda_n,
            0.0,
        )
        target_lambda_t = project_to_coulomb_disk(
            lambda_t + k_t * ds_t,
            mu * next_lambda_n,
            eps=eps_n,
        )
        next_lambda_t = (1.0 - damping) * lambda_t + damping * target_lambda_t

        normal_step = next_lambda_n - lambda_n
        tangential_residual = friction_fixed_point_residual(
            next_lambda_t,
            ds_t,
            next_lambda_n,
            mu,
            k_t,
            eps=eps_n,
        )
        tangential_step = next_lambda_t - lambda_t
        feasibility = check_contact_feasibility(
            g_n,
            next_lambda_n,
            next_lambda_t,
            mu,
            tol_n=tol_n,
            tol_t=tol_t,
        )
        normal_step_norm = _max_abs(normal_step)
        tangential_residual_norm = _max_abs(tangential_residual)
        tangential_step_norm = _max_abs(tangential_step)
        res_norm = tf.maximum(
            normal_step_norm,
            tf.maximum(tangential_step_norm, tangential_residual_norm),
        )
        this_converged = tf.logical_and(
            normal_step_norm <= tf.cast(tol_n, tf.float32),
            tf.logical_and(
                tangential_residual_norm <= tf.cast(tol_t, tf.float32),
                tf.cast(feasibility["feasible"], tf.bool),
            ),
        )
        update = tf.cast(tf.logical_not(done), tf.float32)
        lambda_n = update * next_lambda_n + (1.0 - update) * lambda_n
        lambda_t = update * next_lambda_t + (1.0 - update) * lambda_t
        iters = tf.where(tf.logical_not(done), tf.cast(it + 1, tf.int32), iters)
        converged = tf.logical_or(converged, tf.logical_and(tf.logical_not(done), this_converged))
        done = tf.logical_or(done, this_converged)

    fallback_used = tf.logical_not(converged)
    reused_init = tf.constant(False)
    if init_state is not None:
        init_feasibility = check_contact_feasibility(
            g_n,
            init_lambda_n,
            init_lambda_t,
            mu,
            tol_n=tol_n,
            tol_t=tol_t,
        )
        reused_init = tf.logical_and(fallback_used, tf.cast(init_feasibility["feasible"], tf.bool))

    projected_lambda_t = project_to_coulomb_disk(k_t * ds_t, mu * target_lambda_n, eps=eps_n)
    fallback_lambda_n = tf.cond(reused_init, lambda: init_lambda_n, lambda: target_lambda_n)
    fallback_lambda_t = tf.cond(reused_init, lambda: init_lambda_t, lambda: projected_lambda_t)
    fallback_res_norm = tf.maximum(
        _max_abs(fallback_lambda_n - target_lambda_n),
        _max_abs(
            friction_fixed_point_residual(
                fallback_lambda_t,
                ds_t,
                fallback_lambda_n,
                mu,
                k_t,
                eps=eps_n,
            )
        ),
    )
    lambda_n = tf.cond(fallback_used, lambda: fallback_lambda_n, lambda: lambda_n)
    lambda_t = tf.cond(fallback_used, lambda: fallback_lambda_t, lambda: lambda_t)
    res_norm = tf.where(
        tf.logical_and(fallback_used, tf.logical_not(reused_init)),
        fallback_res_norm,
        res_norm,
    )

    state = ContactInnerState(
        lambda_n=lambda_n,
        lambda_t=lambda_t,
        converged=_python_scalar(converged, bool),
        iters=_python_scalar(iters, int),
        res_norm=_python_scalar(tf.cast(res_norm, tf.float32), float),
        fallback_used=_python_scalar(fallback_used, bool),
    )

    traction_vec = compose_contact_traction(state.lambda_n, state.lambda_t, normals, t1, t2)
    feasibility = check_contact_feasibility(
        g_n,
        state.lambda_n,
        state.lambda_t,
        mu,
        tol_n=tol_n,
        tol_t=tol_t,
    )
    diagnostics = {
        "fn_norm": tf.sqrt(tf.reduce_sum(tf.square(state.lambda_n))),
        "ft_norm": tf.sqrt(tf.reduce_sum(tf.square(state.lambda_t))),
        "cone_violation": tf.cast(feasibility["cone_violation"], tf.float32),
        "max_penetration": tf.cast(feasibility["max_penetration"], tf.float32),
        "converged": tf.cast(converged, tf.float32),
        "skip_batch": tf.cast(0.0, tf.float32),
        "fb_residual_norm": tf.sqrt(
            tf.reduce_mean(tf.square(fb_normal_residual(g_n, state.lambda_n, eps_n)))
            + 1.0e-20
        ),
        "normal_step_norm": _max_abs(normal_step),
        "tangential_step_norm": _max_abs(tangential_step),
        "fallback_used": tf.cast(fallback_used, tf.float32),
        "iters": tf.cast(iters, tf.float32),
    }
    linearization = None
    if return_linearization:
        lambda_n_lin = tf.identity(tf.cast(state.lambda_n, tf.float32))
        lambda_t_lin = tf.identity(tf.cast(state.lambda_t, tf.float32))
        g_n_lin = tf.identity(tf.cast(g_n, tf.float32))
        ds_t_lin = tf.identity(tf.cast(ds_t, tf.float32))
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(lambda_n_lin)
            tape.watch(lambda_t_lin)
            tape.watch(g_n_lin)
            tape.watch(ds_t_lin)
            flat_residual = tf.concat(
                [
                    tf.reshape(fb_normal_residual(g_n_lin, lambda_n_lin, eps_n), (-1,)),
                    tf.reshape(
                        friction_fixed_point_residual(
                            lambda_t_lin,
                            ds_t_lin,
                            lambda_n_lin,
                            mu,
                            k_t,
                            eps=eps_n,
                        ),
                        (-1,),
                    ),
                ],
                axis=0,
            )
        output_size = tf.shape(flat_residual)[0]
        jac_lambda_n = _flatten_jacobian_block(tape.jacobian(flat_residual, lambda_n_lin), output_size)
        jac_lambda_t = _flatten_jacobian_block(tape.jacobian(flat_residual, lambda_t_lin), output_size)
        jac_g_n = _flatten_jacobian_block(tape.jacobian(flat_residual, g_n_lin), output_size)
        jac_ds_t = _flatten_jacobian_block(tape.jacobian(flat_residual, ds_t_lin), output_size)
        del tape
        flat_state = flatten_contact_state(lambda_n_lin, lambda_t_lin)
        flat_inputs = flatten_contact_inputs(g_n_lin, ds_t_lin)
        lambda_n_shape = list(lambda_n_lin.shape.as_list() or [])
        lambda_t_shape = list(lambda_t_lin.shape.as_list() or [])
        g_n_shape = list(g_n_lin.shape.as_list() or [])
        ds_t_shape = list(ds_t_lin.shape.as_list() or [])
        linearization = {
            "schema_version": "strict_mixed_v2",
            "route_mode": "normal_ready",
            "is_exact": False,
            "tangential_mode": "smooth_not_enabled",
            "jac_z": tf.concat([jac_lambda_n, jac_lambda_t], axis=1),
            "jac_inputs": tf.concat([jac_g_n, jac_ds_t], axis=1),
            "state_layout": {
                "order": ["lambda_n", "lambda_t"],
                "lambda_n_shape": lambda_n_shape,
                "lambda_t_shape": lambda_t_shape,
            },
            "input_layout": {
                "order": ["g_n", "ds_t"],
                "g_n_shape": g_n_shape,
                "ds_t_shape": ds_t_shape,
            },
            "flat_z": flat_state,
            "flat_inputs": flat_inputs,
            "z_splits": {"lambda_n": 1, "lambda_t": 2},
            "input_splits": {"g_n": 1, "ds_t": 2},
            "residual": flat_residual,
            "residual_at_solution": flat_residual,
            "normal_step": tf.reshape(tf.cast(normal_step, tf.float32), (-1,)),
            "tangential_step": tf.reshape(tf.cast(tangential_step, tf.float32), (-1,)),
        }
    return ContactInnerResult(
        state=state,
        traction_vec=traction_vec,
        traction_tangent=state.lambda_t,
        diagnostics=diagnostics,
        linearization=linearization,
    )
