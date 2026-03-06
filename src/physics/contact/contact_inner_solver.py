#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stateless inner-contact solver primitives with explicit state/result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import tensorflow as tf


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


def _compose_traction(
    lambda_n: tf.Tensor,
    lambda_t: tf.Tensor,
    normals: tf.Tensor,
    t1: tf.Tensor,
    t2: tf.Tensor,
) -> tf.Tensor:
    return (
        lambda_n[:, None] * normals
        + lambda_t[:, 0:1] * t1
        + lambda_t[:, 1:2] * t2
    )


def solve_contact_inner(
    lambda_n: tf.Tensor,
    lambda_t: tf.Tensor,
    normals: tf.Tensor,
    t1: tf.Tensor,
    t2: tf.Tensor,
    *,
    force_fail: bool = False,
    last_feasible_state: Optional[ContactInnerState] = None,
) -> ContactInnerResult:
    """Build one inner-solver result; optionally fallback to last feasible state."""

    lambda_n = tf.cast(lambda_n, tf.float32)
    lambda_t = tf.cast(lambda_t, tf.float32)
    normals = tf.cast(normals, tf.float32)
    t1 = tf.cast(t1, tf.float32)
    t2 = tf.cast(t2, tf.float32)

    if force_fail and last_feasible_state is not None:
        state = ContactInnerState(
            lambda_n=tf.cast(last_feasible_state.lambda_n, tf.float32),
            lambda_t=tf.cast(last_feasible_state.lambda_t, tf.float32),
            converged=False,
            iters=0,
            res_norm=float(getattr(last_feasible_state, "res_norm", 0.0) or 0.0),
            fallback_used=True,
        )
    else:
        state = ContactInnerState(
            lambda_n=lambda_n,
            lambda_t=lambda_t,
            converged=not force_fail,
            iters=1,
            res_norm=0.0,
            fallback_used=False,
        )

    traction_vec = _compose_traction(state.lambda_n, state.lambda_t, normals, t1, t2)
    diagnostics = {
        "fn_norm": tf.sqrt(tf.reduce_sum(tf.square(state.lambda_n))),
        "ft_norm": tf.sqrt(tf.reduce_sum(tf.square(state.lambda_t))),
        "fallback_used": tf.cast(1.0 if state.fallback_used else 0.0, tf.float32),
    }
    return ContactInnerResult(
        state=state,
        traction_vec=traction_vec,
        traction_tangent=state.lambda_t,
        diagnostics=diagnostics,
    )
