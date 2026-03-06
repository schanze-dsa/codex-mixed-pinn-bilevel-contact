# -*- coding: utf-8 -*-
"""TrainerConfig extracted from trainer.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from model.pinn_model import ModelConfig
from physics.elasticity_config import ElasticityConfig
from physics.contact.contact_operator import ContactOperatorConfig
from physics.tightening_model import TighteningConfig
from model.loss_energy import TotalConfig


@dataclass
class TrainerConfig:
    inp_path: str = "data/shuangfan.inp"
    mirror_surface_name: str = "MIRROR up"
    mirror_surface_asm_key: Optional[str] = None

    materials: Dict[str, Any] = field(
        default_factory=lambda: {
            "mirror": (70000.0, 0.33),
            "steel": (210000.0, 0.30),
        }
    )
    part2mat: Dict[str, str] = field(
        default_factory=lambda: {
            "MIRROR": "mirror",
            "BOLT1": "steel",
            "BOLT2": "steel",
            "BOLT3": "steel",
        }
    )

    contact_pairs: List[Dict[str, str]] = field(default_factory=list)
    n_contact_points_per_pair: int = 6000
    contact_seed: int = 1234
    contact_two_pass: bool = False
    contact_mode: str = "sample"
    contact_mortar_gauss: int = 3
    contact_mortar_max_points: int = 0

    contact_hardening_enabled: bool = True
    contact_hardening_fraction: float = 0.4
    contact_beta_start: Optional[float] = None
    contact_mu_n_start: Optional[float] = None
    friction_k_t_start: Optional[float] = None
    friction_mu_t_start: Optional[float] = None

    preload_specs: List[Dict[str, str]] = field(default_factory=list)
    preload_n_points_each: int = 800

    bc_mode: str = "alm"
    bc_mu: float = 1.0e3
    bc_alpha: float = 1.0e4

    preload_min: float = 0.0
    preload_max: float = 2000.0
    preload_sequence: List[Any] = field(default_factory=list)
    preload_sequence_repeat: int = 1
    preload_sequence_shuffle: bool = False
    preload_sequence_jitter: float = 0.0

    preload_sampling: str = "lhs"
    preload_lhs_size: int = 64

    preload_use_stages: bool = True
    preload_randomize_order: bool = False

    incremental_mode: bool = True
    stage_inner_steps: int = 1
    stage_alm_every: int = 1
    reset_contact_state_per_case: bool = True
    stage_schedule_steps: List[int] = field(default_factory=list)

    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    elas_cfg: ElasticityConfig = field(
        default_factory=lambda: ElasticityConfig(coord_scale=1.0, chunk_size=0, use_pfor=False)
    )
    contact_cfg: ContactOperatorConfig = field(default_factory=ContactOperatorConfig)
    tightening_cfg: TighteningConfig = field(default_factory=TighteningConfig)
    total_cfg: TotalConfig = field(
        default_factory=lambda: TotalConfig(
            w_int=1.0,
            w_cn=1.0,
            w_ct=1.0,
            w_bc=1.0,
            w_tight=1.0,
            w_sigma=1.0,
            w_eq=1.0,
            w_reg=1.0e-4,
        )
    )

    loss_adaptive_enabled: bool = True
    loss_update_every: int = 1
    loss_ema_decay: float = 0.95
    loss_min_factor: float = 0.25
    loss_max_factor: float = 4.0
    loss_min_weight: Optional[float] = None
    loss_max_weight: Optional[float] = None
    loss_gamma: float = 2.0
    loss_focus_terms: Tuple[str, ...] = field(default_factory=tuple)

    max_steps: int = 1000
    adam_steps: Optional[int] = None
    lr: float = 1e-3
    grad_clip_norm: Optional[float] = 1.0
    log_every: int = 1
    alm_update_every: int = 0
    early_exit_enabled: bool = True
    early_exit_warmup_steps: int = 200
    early_exit_nonfinite_patience: int = 8
    early_exit_divergence_patience: int = 30
    early_exit_grad_norm_threshold: float = 1.0e6
    early_exit_pi_ema_rel_increase: float = 0.5
    early_exit_check_every: int = 1
    contact_route_update_every: int = 1
    uncertainty_loss_weight: float = 0.0
    uncertainty_sample_points: int = 0
    uncertainty_proxy_scale: float = 1.0
    uncertainty_logvar_min: float = -8.0
    uncertainty_logvar_max: float = 6.0

    build_bar_color: Optional[str] = "cyan"
    train_bar_color: Optional[str] = "cyan"
    step_bar_color: Optional[str] = "green"
    build_bar_enabled: bool = True
    train_bar_enabled: bool = True
    step_bar_enabled: bool = False
    tqdm_disable: bool = False
    tqdm_disable_if_not_tty: bool = True

    mixed_precision: Optional[str] = "mixed_float16"
    seed: int = 42

    out_dir: str = "outputs"
    ckpt_dir: str = "checkpoints"
    ckpt_max_to_keep: int = 3
    ckpt_save_retries: int = 3
    ckpt_save_retry_delay_s: float = 1.0
    ckpt_save_retry_backoff: float = 2.0
    graph_cache_enabled: bool = True
    graph_cache_dir: Optional[str] = None
    graph_cache_name: Optional[str] = None
    viz_samples_after_train: int = 6
    viz_title_prefix: str = "Total Deformation (trained PINN)"
    viz_style: str = "smooth"
    viz_colormap: str = "turbo"
    viz_diagnose_blanks: bool = False
    viz_auto_fill_blanks: bool = False
    viz_levels: int = 64
    viz_symmetric: bool = False
    viz_units: str = "mm"
    viz_draw_wireframe: bool = False
    viz_surface_enabled: bool = True
    viz_surface_source: str = "part_top"
    viz_write_data: bool = True
    viz_write_surface_mesh: bool = False
    viz_plot_full_structure: bool = False
    viz_full_structure_part: Optional[str] = "mirror1"
    viz_write_full_structure_data: bool = False
    viz_retriangulate_2d: bool = False
    viz_refine_subdivisions: int = 3
    viz_refine_max_points: int = 180_000
    viz_use_shape_function_interp: bool = False
    viz_smooth_vector_iters: int = 0
    viz_smooth_vector_lambda: float = 0.35
    viz_smooth_scalar_iters: int = 0
    viz_smooth_scalar_lambda: float = 0.6
    viz_eval_batch_size: int = 65_536
    viz_eval_scope: str = "assembly"
    viz_force_pointwise: bool = False
    viz_remove_rigid: bool = True
    viz_use_last_training_case: bool = False
    viz_write_reference_aligned: bool = True
    viz_reference_truth_path: Optional[str] = "auto"
    viz_plot_stages: bool = False
    viz_skip_release_stage_plot: bool = False
    viz_compare_cmap: str = "coolwarm"
    viz_compare_common_scale: bool = True
    save_best_on: str = "Pi"

    yield_strength: Optional[float] = None
