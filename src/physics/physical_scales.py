#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared physical scale configuration for mixed residual normalization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhysicalScaleConfig:
    L_ref: float = 1.0
    u_ref: float = 1.0
    sigma_ref: float = 0.0
    E_ref: float = 0.0
    F_ref: float = 0.0
    A_ref: float = 0.0

    def resolved_sigma_ref(self) -> float:
        """Resolve sigma_ref from explicit value or physics-derived estimates."""
        sigma_ref = float(self.sigma_ref)
        if sigma_ref > 0.0:
            return sigma_ref

        stress_force = float(self.F_ref) / max(float(self.A_ref), 1.0e-12)
        stress_strain = float(self.E_ref) * float(self.u_ref) / max(float(self.L_ref), 1.0e-12)
        return max(stress_force, stress_strain, 1.0)
