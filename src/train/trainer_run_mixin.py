# -*- coding: utf-8 -*-
"""Training-loop mixin extracted from Trainer to reduce trainer.py size."""

from __future__ import annotations

import copy
import os
import time
from typing import Dict

from tqdm.auto import tqdm

from train.attach_ties_bcs import attach_bcs_from_asm
from train.loss_weights import LossWeightState


class TrainerRunMixin:
    # ----------------- 训练 -----------------
    def run(self):
        self.build()
        print(f"[trainer] 当前训练设备：{self.device_summary}")
        total = self._assemble_total()
        self.bcs_ops = attach_bcs_from_asm(
            total=total,
            asm=self.asm,
            cfg=self.cfg,
        )
        if self.bcs_ops:
            print(f"[bc] 已挂载 {len(self.bcs_ops)} 组边界约束")
        else:
            print("[bc] 未发现边界约束，跳过挂载")
        self._total_ref = total

        # ---- 初始化自适应损失权重状态 ----
        # 以 TotalConfig 里的 w_int / w_cn / ... 作为基准权重
        base_weights = {
            "E_int": self.cfg.total_cfg.w_int,
            "E_cn": self.cfg.total_cfg.w_cn,
            "E_ct": self.cfg.total_cfg.w_ct,
            "E_bc": self.cfg.total_cfg.w_bc,
            "E_tight": self.cfg.total_cfg.w_tight,
            "E_sigma": self.cfg.total_cfg.w_sigma,
            "E_eq": getattr(self.cfg.total_cfg, "w_eq", 0.0),
            "E_reg": getattr(self.cfg.total_cfg, "w_reg", 0.0),
            "E_bi": getattr(self.cfg.total_cfg, "w_bi", 0.0),
            "E_ed": getattr(self.cfg.total_cfg, "w_ed", 0.0),
            "E_unc": getattr(self.cfg, "uncertainty_loss_weight", 0.0),
            "path_penalty_total": getattr(self.cfg.total_cfg, "path_penalty_weight", 0.0),
            "fric_path_penalty_total": getattr(self.cfg.total_cfg, "fric_path_penalty_weight", 0.0),
            "R_fric_comp": 0.0,
            "R_contact_comp": 0.0,
        }
        self._base_weights = base_weights
        self._loss_keys = list(base_weights.keys())

        adaptive_enabled = bool(getattr(self.cfg, "loss_adaptive_enabled", False))
        sign_overrides = {}
        if adaptive_enabled:
            scheme = getattr(self.cfg.total_cfg, "adaptive_scheme", "contact_only")
            focus_terms = getattr(self.cfg, "loss_focus_terms", tuple())
            self.loss_state = LossWeightState.from_config(
                base_weights=base_weights,
                adaptive_scheme=scheme,
                ema_decay=getattr(self.cfg, "loss_ema_decay", 0.95),
                min_factor=getattr(self.cfg, "loss_min_factor", 0.25),
                max_factor=getattr(self.cfg, "loss_max_factor", 4.0),
                min_weight=getattr(self.cfg, "loss_min_weight", None),
                max_weight=getattr(self.cfg, "loss_max_weight", None),
                gamma=getattr(self.cfg, "loss_gamma", 2.0),
                focus_terms=focus_terms,
                update_every=getattr(self.cfg, "loss_update_every", 1),
                sign_overrides=sign_overrides,
            )
        else:
            self.loss_state = None
        self._refresh_static_weight_vector()
        train_desc = "训练"
        train_pb_kwargs = dict(
            total=self.cfg.max_steps,
            desc=train_desc,
            leave=True,
            disable=not (self._tqdm_enabled and self.cfg.train_bar_enabled),
        )
        step_detail_enabled = self._step_detail_enabled()
        last_step = 0
        stop_reason = None
        if self.cfg.train_bar_color:
            train_pb_kwargs["colour"] = self.cfg.train_bar_color
        with tqdm(**train_pb_kwargs) as p_train:
            for step in range(1, self.cfg.max_steps + 1):
                stop_this_step = False
                # 子进度条：本 step 的 4 个动作
                step_pb_kwargs = dict(
                    total=4,
                    leave=False,
                    disable=not step_detail_enabled,
                )
                if self.cfg.step_bar_color:
                    step_pb_kwargs["colour"] = self.cfg.step_bar_color
                with tqdm(**step_pb_kwargs) as p_step:
                    # 1) 接触重采样
                    if step_detail_enabled:
                        self._set_pbar_desc(p_step, f"step {step}: 接触重采样")
                    t0 = time.perf_counter()
                    contact_note = "跳过"
                    if self.contact is None:
                        contact_note = "跳过 (无接触体)"
                    else:
                        contact_note = "跳过 (路线锁定: 沿用构建采样)"
                    elapsed = time.perf_counter() - t0
                    self._step_stage_times.append(("resample", elapsed))
                    if step_detail_enabled:
                        self._set_pbar_postfix(
                            p_step,
                            f"{contact_note} | {self._format_seconds(elapsed)}"
                        )
                    p_step.update(1)

                    # 2) 前向 + 反传（随机采样三螺栓预紧力）
                    if step_detail_enabled:
                        self._set_pbar_desc(p_step, f"step {step}: 前向/反传")
                    t0 = time.perf_counter()
                    preload_case = self._sample_preload_case()
                    # 动态提升接触惩罚/ALM 参数（软→硬）
                    self._maybe_update_contact_hardening(step)
                    vol_note = ""
                    if self.elasticity is not None and hasattr(self.elasticity, "set_sample_indices"):
                        self.elasticity.set_sample_indices(None)
                    self._push_contact_route_hint()
                    Pi, parts, stats, grad_norm = self._train_step(total, preload_case, step=step)
                    P_np = preload_case["P"]
                    order_np = preload_case.get("order")
                    self._last_preload_case = copy.deepcopy(preload_case)
                    if self._should_update_contact_route(step):
                        route_score = self._update_contact_route_metric(parts)
                    else:
                        route_score = self._contact_route_score()

                    should_collect_scalars = self._should_collect_step_scalars(step)
                    pi_val = float("nan")
                    grad_val = float("nan")
                    rel_pi = None
                    rel_delta = None
                    if should_collect_scalars:
                        pi_val = float(Pi.numpy())
                        if self._pi_baseline is None:
                            self._pi_baseline = pi_val if pi_val != 0.0 else 1.0
                        if self._pi_ema is None:
                            self._pi_ema = pi_val
                        else:
                            ema_alpha = 0.1
                            self._pi_ema = (1 - ema_alpha) * self._pi_ema + ema_alpha * pi_val
                        rel_pi = pi_val / (self._pi_baseline or pi_val or 1.0)
                        if self._prev_pi is not None and self._prev_pi != 0.0:
                            rel_delta = (self._prev_pi - pi_val) / abs(self._prev_pi)
                        self._prev_pi = pi_val
                        grad_val = float(grad_norm.numpy()) if hasattr(grad_norm, "numpy") else float(grad_norm)
                    elif self._prev_pi is not None:
                        pi_val = float(self._prev_pi)
                    elapsed = time.perf_counter() - t0
                    self._step_stage_times.append(("train", elapsed))
                    device = self._short_device_name(getattr(Pi, "device", None))
                    if step_detail_enabled:
                        rel_pct = rel_pi * 100.0 if rel_pi is not None else None
                        rel_txt = (
                            f"Πrel={rel_pct:.2f}%" if rel_pct is not None else "Πrel=--"
                        )
                        d_txt = (
                            f"ΔΠ={rel_delta * 100:+.1f}%"
                            if rel_delta is not None
                            else "ΔΠ=--"
                        )
                        ema_txt = f"Πema={self._pi_ema:.2e}" if self._pi_ema is not None else "Πema=--"
                        order_txt = ""
                        if order_np is not None:
                            order_txt = " order=" + "-".join(str(int(x) + 1) for x in order_np)
                        energy_summary = self._format_energy_summary_if_needed(parts)
                        energy_txt = f" | {energy_summary}" if energy_summary else ""
                        if vol_note:
                            energy_txt += f" | {vol_note}"
                        train_note = (
                            f"P=[{int(P_np[0])},{int(P_np[1])},{int(P_np[2])}]"
                            f"{order_txt}{energy_txt} | Π={pi_val:.2e} {rel_txt} {d_txt} "
                            f"grad={grad_val:.2e} {ema_txt} route={route_score:.2f}"
                        )
                        if step == 1:
                            train_note += " | 首轮包含图追踪/缓存构建"
                        self._set_pbar_postfix(
                            p_step,
                            f"{train_note} | {self._format_seconds(elapsed)} | dev={device}"
                        )
                    p_step.update(1)

                    stop_reason = None
                    if self._should_check_early_exit(step):
                        stop_reason = self._check_early_exit(step, pi_val, grad_val)
                    if stop_reason:
                        stop_this_step = True
                        print(
                            f"[trainer] Early exit at step {step}: {stop_reason}",
                            flush=True,
                        )
                        if step_detail_enabled:
                            self._set_pbar_postfix(
                                p_step,
                                f"触发 early-exit | {self._format_seconds(elapsed)}"
                            )

                    # 3) ALM 更新
                    if step_detail_enabled:
                        self._set_pbar_desc(p_step, f"step {step}: ALM 更新")
                    t0 = time.perf_counter()
                    alm_note = "跳过"
                    if stop_this_step:
                        alm_note = "跳过 (early-exit)"
                    else:
                        alm_note = "跳过 (路线锁定: 阶段内更新)"
                    elapsed = time.perf_counter() - t0
                    self._step_stage_times.append(("alm", elapsed))
                    if step_detail_enabled:
                        self._set_pbar_postfix(
                            p_step,
                            f"{alm_note} | {self._format_seconds(elapsed)}"
                        )
                    p_step.update(1)

                    # 4) 日志/检查点
                    if step_detail_enabled:
                        self._set_pbar_desc(p_step, f"step {step}: 日志/检查点")
                    t0 = time.perf_counter()
                    log_note = "跳过"
                    if stop_this_step:
                        log_note = "跳过 (early-exit)"
                    elif self.cfg.log_every <= 0:
                        log_note = "跳过 (已禁用)"
                    else:
                        should_log = step == 1 or step % self.cfg.log_every == 0
                        if should_log:
                            postfix, log_note = self._format_train_log_postfix(
                                P_np,
                                Pi,
                                parts,
                                stats,
                                grad_val,
                                rel_pi,
                                rel_delta,
                                order_np,
                            )
                            if postfix:
                                p_train.set_postfix_str(postfix)
                                # 额外打印到终端（确保不被进度条覆盖）
                                print(f"\n[Step {step}] {postfix}", flush=True)

                            metric_name = self.cfg.save_best_on.lower()
                            metric_val = (
                                pi_val
                                if metric_name == "pi"
                                else float(parts["E_int"].numpy())
                            )
                            if metric_val < self.best_metric:
                                ckpt_path = self._save_checkpoint_best_effort(step)
                                if ckpt_path:
                                    self.best_metric = metric_val
                                    log_note += f" | 已保存 {os.path.basename(ckpt_path)}"
                                else:
                                    log_note += " | checkpoint 保存失败(已跳过)"

                    if (
                        not stop_this_step
                        and self.cfg.log_every > 0
                        and not (step == 1 or step % self.cfg.log_every == 0)
                    ):
                        remaining = self.cfg.log_every - (step % self.cfg.log_every)
                        log_note = f"跳过 (距下次还有 {remaining} 步)"
                    elapsed = time.perf_counter() - t0
                    self._step_stage_times.append(("log", elapsed))
                    if step_detail_enabled:
                        self._set_pbar_postfix(
                            p_step,
                            f"{log_note} | {self._format_seconds(elapsed)}"
                        )
                    p_step.update(1)

                p_train.update(1)
                last_step = step

                if step % max(1, self.cfg.log_every) == 0:
                    total_spent = sum(t for _, t in self._step_stage_times)
                    if total_spent > 0:
                        label_map = {
                            "resample": "采样",
                            "train": "前向/反传",
                            "alm": "ALM",
                            "log": "日志",
                        }
                        stage_totals: Dict[str, float] = {}
                        for name, t in self._step_stage_times:
                            stage_totals[name] = stage_totals.get(name, 0.0) + float(t)
                        n_steps = max(1, int(round(len(self._step_stage_times) / 4.0)))
                        avg_step = total_spent / n_steps
                        ordered = ["resample", "train", "alm", "log"]
                        parts_txt = ", ".join(
                            f"{label_map.get(name, name)}:{stage_totals.get(name, 0.0) / total_spent * 100:.0f}%"
                            for name in ordered
                        )
                        summary_note = (
                            f"step{step}平均耗时 {self._format_seconds(avg_step)} ({parts_txt})"
                        )
                        if step == 1:
                            summary_note += " | 首轮额外包括图追踪/初次缓存"
                        self._set_pbar_postfix(p_train, summary_note)
                    self._step_stage_times.clear()

                if stop_this_step:
                    break

        # 训练结束：再存一次
        if self.ckpt_manager is not None:
            final_step = last_step if last_step > 0 else self.cfg.max_steps
            final_ckpt = self._save_checkpoint_best_effort(final_step)
            if final_ckpt:
                print(f"[trainer] 训练结束已保存 checkpoint -> {final_ckpt}")
            else:
                print("[trainer] WARNING: 训练结束 checkpoint 保存失败(已跳过)")

        self._visualize_after_training(n_samples=self.cfg.viz_samples_after_train)

