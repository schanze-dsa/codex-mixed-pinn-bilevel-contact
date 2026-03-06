# -*- coding: utf-8 -*-
"""Visualization mixin extracted from Trainer to reduce trainer.py size."""

from __future__ import annotations

import copy
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from viz.mirror_viz import plot_mirror_deflection_by_name


class TrainerVizMixin:
    def _call_viz(self, P: np.ndarray, params: Dict[str, tf.Tensor], out_path: str, title: str):
        bare = self.cfg.mirror_surface_name
        data_path = None
        if self.cfg.viz_write_data and out_path:
            data_path = os.path.splitext(out_path)[0] + ".txt"

        mesh_path = None
        if self.cfg.viz_write_surface_mesh and out_path:
            mesh_path = "auto"

        full_plot_enabled = bool(self.cfg.viz_plot_full_structure)
        full_struct_out = "auto" if (full_plot_enabled and out_path) else None
        full_struct_data = (
            "auto" if (full_plot_enabled and self.cfg.viz_write_full_structure_data and out_path) else None
        )

        diag_out: Dict[str, Any] = {} if self.cfg.viz_diagnose_blanks else None
        u_eval_fn = self.model.u_fn
        if bool(getattr(self.cfg, "viz_force_pointwise", False)) and hasattr(self.model, "u_fn_pointwise"):
            u_eval_fn = self.model.u_fn_pointwise

        result = plot_mirror_deflection_by_name(
            self.asm,
            bare,
            u_eval_fn,
            params,
            P_values=tuple(float(x) for x in P.reshape(-1)),
            out_path=out_path,
            render_surface=self.cfg.viz_surface_enabled,
            surface_source=self.cfg.viz_surface_source,
            title_prefix=title,
            units=self.cfg.viz_units,
            levels=self.cfg.viz_levels,
            symmetric=self.cfg.viz_symmetric,
            data_out_path=data_path,
            surface_mesh_out_path=mesh_path,
            plot_full_structure=full_plot_enabled,
            full_structure_out_path=full_struct_out,
            full_structure_data_out_path=full_struct_data,
            full_structure_part=self.cfg.viz_full_structure_part,
            style=self.cfg.viz_style,
            cmap=self.cfg.viz_colormap,
            draw_wireframe=self.cfg.viz_draw_wireframe,
            refine_subdivisions=self.cfg.viz_refine_subdivisions,
            refine_max_points=self.cfg.viz_refine_max_points,
            use_shape_function_interp=self.cfg.viz_use_shape_function_interp,
            smooth_vector_iters=self.cfg.viz_smooth_vector_iters,
            smooth_vector_lambda=self.cfg.viz_smooth_vector_lambda,
            smooth_scalar_iters=self.cfg.viz_smooth_scalar_iters,
            smooth_scalar_lambda=self.cfg.viz_smooth_scalar_lambda,
            retriangulate_2d=self.cfg.viz_retriangulate_2d,
            eval_batch_size=self.cfg.viz_eval_batch_size,
            eval_scope=self.cfg.viz_eval_scope,
            diagnose_blanks=self.cfg.viz_diagnose_blanks,
            auto_fill_blanks=self.cfg.viz_auto_fill_blanks,
            remove_rigid=self.cfg.viz_remove_rigid,
            diag_out=diag_out,
        )
        return result

    def _fixed_viz_preload_cases(self) -> List[Dict[str, np.ndarray]]:
        """生成固定拧紧角案例以避免可视化阶段的随机性."""

        nb = int(getattr(self, "_preload_dim", 0) or len(self.cfg.preload_specs) or 1)
        lo = float(self.cfg.preload_min)
        hi = float(self.cfg.preload_max)
        mid = 0.5 * (lo + hi)

        def _make_case(P_list: Sequence[float], order: Sequence[int]) -> Dict[str, np.ndarray]:
            P_arr = np.asarray(P_list, dtype=np.float32).reshape(-1)
            if P_arr.size != nb:
                raise ValueError(f"固定可视化需要 {nb} 维角度输入，收到 {P_arr.size} 维。")
            case: Dict[str, np.ndarray] = {"P": P_arr}
            if not self.cfg.preload_use_stages:
                return case
            order_norm = self._normalize_order(order, nb)
            if order_norm is None:
                return case
            case["order"] = order_norm
            case.update(self._build_stage_case(P_arr, order_norm))
            return case

        cases: List[Dict[str, np.ndarray]] = []

        # 单螺母: 仅一个达到 hi，其余为 lo
        for i in range(nb):
            arr = [lo] * nb
            arr[i] = hi
            cases.append(_make_case(arr, order=list(range(nb))))

        # 等幅: 全部为 mid，并给出两种顺序（若 nb>=2）
        cases.append(_make_case([mid] * nb, order=list(range(nb))))
        if nb >= 2:
            cases.append(_make_case([mid] * nb, order=list(reversed(range(nb)))))

        return cases

    def _resolve_viz_cases(self, n_samples: int) -> List[Dict[str, np.ndarray]]:
        """Resolve visualization cases with deterministic defaults.

        By default we use fixed, reproducible cases so exported results can be
        compared against reference datasets across runs. The legacy behavior of
        using the last sampled training case is available via
        ``viz_use_last_training_case=True``.
        """

        use_last = bool(getattr(self.cfg, "viz_use_last_training_case", False))
        if use_last and self._last_preload_case is not None:
            print("[viz] Using last training tightening case for visualization.")
            return [copy.deepcopy(self._last_preload_case)]

        fixed_cases = self._fixed_viz_preload_cases()
        if fixed_cases:
            print("[viz] Using fixed tightening cases for reproducible visualization.")
            return fixed_cases

        if self._last_preload_case is not None:
            print("[viz] Fixed cases unavailable, fallback to last training case.")
            return [copy.deepcopy(self._last_preload_case)]

        return [self._sample_preload_case() for _ in range(n_samples)]

    def _visualize_after_training(self, n_samples: int = 5):
        if self.asm is None or self.model is None:
            return
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        cases = self._resolve_viz_cases(n_samples)
        n_total = len(cases) if cases else n_samples
        print(
            f"[trainer] Generating {n_total} deflection maps for '{self.cfg.mirror_surface_name}' ..."
        )
        iter_cases = cases if cases else [self._sample_preload_case() for _ in range(n_samples)]
        viz_records: List[Dict[str, Any]] = []
        for i, preload_case in enumerate(iter_cases):
            P = preload_case["P"]
            order_display = None
            if self.cfg.preload_use_stages and "order" in preload_case:
                order_display = "-".join(
                    str(int(o) + 1) for o in preload_case["order"].tolist()
                )
            unit = str(getattr(self.cfg.tightening_cfg, "angle_unit", "deg") or "deg")
            angle_txt = ",".join(f"{float(x):.2f}" for x in P.tolist())
            title = f"{self.cfg.viz_title_prefix}  theta=[{angle_txt}]{unit}"
            if order_display:
                title += f"  (order={order_display})"
            suffix = f"_{order_display.replace('-', '')}" if order_display else ""
            save_path = os.path.join(
                self.cfg.out_dir, f"deflection_{i+1:02d}{suffix}.png"
            )
            params_full = self._make_preload_params(preload_case)
            params_eval = self._extract_final_stage_params(params_full, keep_context=True)

            # Write a compact tightening report next to the figure.
            if self.tightening is not None and save_path:
                try:
                    report_path = os.path.splitext(save_path)[0] + "_tightening.txt"
                    stage_rows = []
                    if (
                        self.cfg.preload_use_stages
                        and isinstance(preload_case, dict)
                        and "stages" in preload_case
                    ):
                        stages_np = np.asarray(preload_case.get("stages"), dtype=np.float32)
                        if stages_np.ndim == 2 and stages_np.shape[0] > 0:
                            for s in range(int(stages_np.shape[0])):
                                params_s = self._extract_stage_params(params_full, s, keep_context=True)
                                _, st = self.tightening.energy(self.model.u_fn, params_s)
                                stage_rows.append(
                                    np.asarray(st.get("tightening", {}).get("rms", []))
                                )
                    _, st_final = self.tightening.energy(self.model.u_fn, params_eval)
                    final_row = np.asarray(st_final.get("tightening", {}).get("rms", []))

                    with open(report_path, "w", encoding="utf-8") as fp:
                        fp.write(f"theta = {P.tolist()}  [{unit}]\n")
                        if self.cfg.preload_use_stages and "order" in preload_case:
                            fp.write(f"order = {preload_case['order'].tolist()}  (0-based)\n")
                        fp.write("rms = [r1, r2, ...]\n")
                        for s, row in enumerate(stage_rows, start=1):
                            fp.write(f"stage_{s}: {row.tolist()}\n")
                        fp.write(f"final: {final_row.tolist()}\n")
                except Exception as exc:
                    print(f"[viz] tightening report skipped: {exc}")
            try:
                _, _, data_path = self._call_viz(P, params_eval, save_path, title)
                if self.cfg.viz_surface_enabled:
                    if not os.path.exists(save_path):
                        try:
                            import matplotlib.pyplot as plt
                            plt.savefig(save_path, dpi=200, bbox_inches="tight")
                            plt.close()
                        except Exception:
                            pass
                    if order_display:
                        print(f"[viz] saved -> {save_path}  (order={order_display})")
                    else:
                        print(f"[viz] saved -> {save_path}")
                    if data_path:
                        print(f"[viz] displacement data -> {data_path}")
                aligned_path = None
                if data_path:
                    try:
                        aligned_path = self._write_viz_reference_alignment(str(data_path))
                    except Exception as exc:
                        print(f"[viz] reference alignment skipped: {exc}")
                viz_records.append(
                    {
                        "index": i + 1,
                        "P": np.asarray(P, dtype=np.float64).reshape(-1),
                        "order": None if "order" not in preload_case else preload_case.get("order"),
                        "order_display": order_display,
                        "png_path": save_path,
                        "data_path": data_path,
                        "mesh_path": (
                            os.path.splitext(save_path)[0] + "_surface.ply"
                            if self.cfg.viz_write_surface_mesh and save_path
                            else None
                        ),
                        "aligned_path": aligned_path,
                    }
                )
            except TypeError as e:
                print("[viz] signature mismatch:", e)
            except Exception as e:
                print("[viz] error:", e)

            # Optional: plot each preload stage to make tightening order visible.
            if (
                self.cfg.viz_plot_stages
                and self.cfg.preload_use_stages
                and isinstance(preload_case, dict)
                and "stages" in preload_case
            ):
                try:
                    stages_np = np.asarray(preload_case.get("stages"), dtype=np.float32)
                    if stages_np.ndim == 2 and stages_np.shape[0] > 1:
                        stage_indices = self._resolve_stage_plot_indices(preload_case, int(stages_np.shape[0]))
                        n_plot = int(len(stage_indices))
                        for rank, s in enumerate(stage_indices, start=1):
                            P_stage = stages_np[s]
                            title_s = f"{self.cfg.viz_title_prefix}  P=[{int(P_stage[0])},{int(P_stage[1])},{int(P_stage[2])}]N"
                            if order_display:
                                title_s += f"  (order={order_display})"
                            title_s += f"  (stage={rank}/{n_plot})"
                            save_path_s = os.path.join(
                                self.cfg.out_dir, f"deflection_{i+1:02d}{suffix}_s{s+1}.png"
                            )
                            params_s = self._extract_stage_params(params_full, s, keep_context=True)
                            self._call_viz(P_stage, params_s, save_path_s, title_s)
                except Exception as exc:
                    print(f"[viz] stage plots skipped: {exc}")

        # Additional comparison outputs: common-scale maps and delta maps between cases.
        if cases and viz_records and len(viz_records) > 1 and bool(getattr(self.cfg, "viz_compare_cases", False)):
            try:
                self._write_viz_comparison(viz_records)
            except Exception as exc:
                print(f"[viz] comparison skipped: {exc}")

    def _resolve_stage_plot_indices(self, preload_case: Dict[str, Any], stage_count: int) -> List[int]:
        if stage_count <= 0:
            return []
        indices = list(range(int(stage_count)))
        if not bool(getattr(self.cfg, "viz_skip_release_stage_plot", False)):
            return indices
        stage_last = preload_case.get("stage_last")
        if stage_last is None:
            return indices
        try:
            stage_last_np = np.asarray(stage_last, dtype=np.float32)
        except Exception:
            return indices
        if stage_last_np.ndim != 2 or stage_last_np.shape[0] != stage_count:
            return indices
        keep = [
            i
            for i in range(stage_count)
            if bool(np.any(np.abs(stage_last_np[i]) > 1.0e-8))
        ]
        return keep if keep else indices

    def _resolve_viz_reference_path(self) -> Optional[str]:
        raw = str(getattr(self.cfg, "viz_reference_truth_path", "auto") or "").strip()
        if not raw:
            return None

        low = raw.lower()
        if low in {"none", "off", "false", "0", "disable", "disabled"}:
            return None

        candidates: List[str] = []
        out_dir = str(getattr(self.cfg, "out_dir", "") or "").strip()
        if low == "auto":
            if out_dir:
                candidates.append(os.path.join(out_dir, "3.txt"))
            candidates.append(os.path.join(os.getcwd(), "results", "3.txt"))
        else:
            candidates.append(raw)
            if out_dir and not os.path.isabs(raw):
                candidates.append(os.path.join(out_dir, raw))
            if not os.path.isabs(raw):
                candidates.append(os.path.join(os.getcwd(), raw))

        for cand in candidates:
            path = os.path.abspath(os.path.expanduser(str(cand)))
            if os.path.exists(path):
                return path
        return None

    @staticmethod
    def _read_reference_truth_samples(path: str) -> Optional[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            return None

        node_ids: List[int] = []
        umag: List[float] = []
        parsed_rows = 0

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                cols = re.split(r"[,\s\t]+", line)
                if len(cols) < 2:
                    continue
                try:
                    nid = int(float(cols[0]))
                    u = float(cols[1])
                except Exception:
                    continue
                parsed_rows += 1
                node_ids.append(nid)
                umag.append(u)

        if not node_ids:
            return None

        return {
            "node_id": np.asarray(node_ids, dtype=np.int64),
            "umag": np.asarray(umag, dtype=np.float64),
            "parsed_rows": int(parsed_rows),
            "path": path,
        }

    def _get_asm_node_id_set(self) -> set[int]:
        if self._asm_node_ids is not None:
            return self._asm_node_ids

        ids: set[int] = set()
        asm = getattr(self, "asm", None)
        nodes = getattr(asm, "nodes", None) if asm is not None else None
        if isinstance(nodes, dict):
            for nid in nodes.keys():
                try:
                    ids.add(int(nid))
                except Exception:
                    continue

        if not ids and asm is not None:
            for part in getattr(asm, "parts", {}).values():
                for nid in getattr(part, "node_ids", []) or []:
                    try:
                        ids.add(int(nid))
                    except Exception:
                        continue

        self._asm_node_ids = ids
        return ids

    def _load_viz_reference_truth(self) -> Optional[Dict[str, Any]]:
        path = self._resolve_viz_reference_path()
        if path is None:
            return None
        if self._viz_reference_cache is not None and self._viz_reference_cache_path == path:
            return self._viz_reference_cache

        loaded = self._read_reference_truth_samples(path)
        if loaded is None:
            self._viz_reference_cache_path = None
            self._viz_reference_cache = None
            return None

        self._viz_reference_cache_path = path
        self._viz_reference_cache = loaded
        return loaded

    def _write_viz_reference_alignment(self, pred_data_path: str) -> Optional[str]:
        if not bool(getattr(self.cfg, "viz_write_reference_aligned", False)):
            return None

        pred = self._read_viz_samples(str(pred_data_path))
        if pred is None:
            return None
        ref = self._load_viz_reference_truth()
        if ref is None:
            return None

        valid_node_ids = self._get_asm_node_id_set()
        if not valid_node_ids:
            print("[viz] reference alignment skipped: assembly node ids unavailable.")
            return None

        ref_ids = np.asarray(ref["node_id"], dtype=np.int64).reshape(-1)
        ref_u = np.asarray(ref["umag"], dtype=np.float64).reshape(-1)
        node_mask = np.asarray([int(nid) in valid_node_ids for nid in ref_ids], dtype=bool)

        ref_node_ids = ref_ids[node_mask]
        ref_node_u = ref_u[node_mask]
        nonnode_count = int(ref_ids.size - ref_node_ids.size)
        if ref_node_ids.size == 0:
            print("[viz] reference alignment skipped: no valid node rows in reference.")
            return None

        ref_map: Dict[int, float] = {}
        for nid, val in zip(ref_node_ids.tolist(), ref_node_u.tolist()):
            ref_map[int(nid)] = float(val)

        pred_ids = np.asarray(pred["node_id"], dtype=np.int64).reshape(-1)
        pred_u = np.asarray(pred["umag"], dtype=np.float64).reshape(-1)
        common_mask = np.asarray([int(nid) in ref_map for nid in pred_ids], dtype=bool)
        common_ids = pred_ids[common_mask]
        if common_ids.size == 0:
            print("[viz] reference alignment skipped: no overlapping node ids.")
            return None

        ref_common = np.asarray([ref_map[int(nid)] for nid in common_ids.tolist()], dtype=np.float64)
        pred_common = pred_u[common_mask]
        diff = pred_common - ref_common

        denom = np.where(np.abs(ref_common) > 1.0e-30, ref_common, np.nan)
        ratio = np.divide(pred_common, denom)
        ratio_abs = np.abs(np.divide(pred_common, np.where(np.abs(ref_common) > 1.0e-30, ref_common, np.nan)))
        ratio_med = float(np.nanmedian(ratio_abs)) if ratio_abs.size else float("nan")

        out_path = os.path.splitext(str(pred_data_path))[0] + "_aligned.txt"
        with open(out_path, "w", encoding="utf-8") as fp:
            fp.write("# Aligned mirror displacement: prediction vs reference\n")
            fp.write(f"# reference_path={ref.get('path', '')}\n")
            fp.write(f"# reference_rows_total={int(ref_ids.size)}\n")
            fp.write(f"# reference_rows_node_only={int(ref_node_ids.size)}\n")
            fp.write(f"# reference_rows_nonnode={nonnode_count}\n")
            fp.write(f"# predicted_rows={int(pred_ids.size)}\n")
            fp.write(f"# common_rows={int(common_ids.size)}\n")
            fp.write("# columns: node_id u_ref u_pred diff(pred-ref) ratio(pred/ref)\n")
            for nid, u_ref, u_pred, du, rt in zip(
                common_ids.tolist(),
                ref_common.tolist(),
                pred_common.tolist(),
                diff.tolist(),
                ratio.tolist(),
            ):
                fp.write(
                    f"{int(nid):10d} {float(u_ref): .8e} {float(u_pred): .8e} "
                    f"{float(du): .8e} {float(rt): .8e}\n"
                )

        print(
            "[viz] aligned displacement -> "
            f"{out_path} (common={int(common_ids.size)}, nonnode_ref={nonnode_count}, "
            f"median|pred/ref|={ratio_med:.3e})"
        )
        return out_path

    @staticmethod
    def _read_viz_samples(path: str) -> Optional[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            return None

        node_ids: List[int] = []
        ux: List[float] = []
        uy: List[float] = []
        uz: List[float] = []
        umag: List[float] = []
        u_plane: List[float] = []
        v_plane: List[float] = []
        rigid_line: Optional[str] = None

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    if "rigid_body_removed" in line:
                        rigid_line = line
                    continue
                cols = line.split()
                if len(cols) < 10:
                    continue
                try:
                    node_ids.append(int(cols[0]))
                    ux.append(float(cols[4]))
                    uy.append(float(cols[5]))
                    uz.append(float(cols[6]))
                    umag.append(float(cols[7]))
                    u_plane.append(float(cols[8]))
                    v_plane.append(float(cols[9]))
                except Exception:
                    continue

        if not node_ids:
            return None

        node_arr = np.asarray(node_ids, dtype=np.int64)
        order = np.argsort(node_arr)
        return {
            "node_id": node_arr[order],
            "ux": np.asarray(ux, dtype=np.float64)[order],
            "uy": np.asarray(uy, dtype=np.float64)[order],
            "uz": np.asarray(uz, dtype=np.float64)[order],
            "umag": np.asarray(umag, dtype=np.float64)[order],
            "u_plane": np.asarray(u_plane, dtype=np.float64)[order],
            "v_plane": np.asarray(v_plane, dtype=np.float64)[order],
            "rigid_line": rigid_line,
        }

    def _write_viz_comparison(self, records: List[Dict[str, Any]]) -> None:
        """
        Generate:
        - common-scale |u| maps to make amplitude comparable
        - delta maps (vector displacement difference) to highlight subtle differences
        - a text report with quantitative metrics
        """
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation
        from matplotlib import colors

        def _read_surface_ply_mesh(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            if not path or not os.path.exists(path):
                return None
            n_vert = None
            n_face = None
            header_done = False
            node_ids: List[int] = []
            tris: List[Tuple[int, int, int]] = []
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    if not header_done:
                        if s.startswith("element vertex"):
                            parts = s.split()
                            if len(parts) >= 3:
                                n_vert = int(parts[2])
                        elif s.startswith("element face"):
                            parts = s.split()
                            if len(parts) >= 3:
                                n_face = int(parts[2])
                        elif s == "end_header":
                            header_done = True
                            break
                if not header_done or n_vert is None or n_face is None:
                    return None

                for _ in range(int(n_vert)):
                    row = f.readline()
                    if not row:
                        return None
                    cols = row.strip().split()
                    if len(cols) < 4:
                        return None
                    node_ids.append(int(cols[3]))

                for _ in range(int(n_face)):
                    row = f.readline()
                    if not row:
                        break
                    cols = row.strip().split()
                    if len(cols) < 4:
                        continue
                    try:
                        n = int(cols[0])
                    except Exception:
                        continue
                    if n < 3:
                        continue
                    # Expect triangles; if not, take the first three vertices as a fallback.
                    i0, i1, i2 = int(cols[1]), int(cols[2]), int(cols[3])
                    tris.append((i0, i1, i2))

            if not node_ids or not tris:
                return None
            return (
                np.asarray(node_ids, dtype=np.int64),
                np.asarray(tris, dtype=np.int32),
            )

        samples: List[Dict[str, Any]] = []
        for rec in records:
            data_path = rec.get("data_path")
            if not data_path:
                continue
            s = self._read_viz_samples(str(data_path))
            if s is None:
                continue
            s["record"] = rec
            samples.append(s)

        if len(samples) < 2:
            return

        # Use the first sample as the geometric base for triangulation/mapping.
        geom_base = samples[0]
        geom_base_rec = geom_base["record"]
        base_nodes = geom_base["node_id"]

        # Common scale across all cases (for |u| maps)
        global_umax = 0.0
        for s in samples:
            global_umax = max(global_umax, float(np.nanmax(s["umag"])))
        global_umax = float(global_umax) + 1e-16

        # Triangulation in (u,v) plane for diff plots: prefer FE connectivity from the surface PLY.
        u = np.asarray(geom_base["u_plane"], dtype=np.float64)
        v = np.asarray(geom_base["v_plane"], dtype=np.float64)
        tri = None
        vertex_pos: Optional[np.ndarray] = None
        mesh_info = _read_surface_ply_mesh(str(geom_base_rec.get("mesh_path") or ""))
        if mesh_info is not None:
            mesh_nodes, mesh_tris = mesh_info
            pos = np.searchsorted(base_nodes, mesh_nodes)
            ok = (
                (pos >= 0)
                & (pos < base_nodes.shape[0])
                & (base_nodes[pos] == mesh_nodes)
            )
            if np.all(ok):
                u_vert = u[pos]
                v_vert = v[pos]
                tri = Triangulation(u_vert, v_vert, triangles=mesh_tris)
                vertex_pos = pos
        if tri is None:
            tri = Triangulation(u, v)
            cu, cv = float(np.mean(u)), float(np.mean(v))
            r = np.sqrt((u - cu) ** 2 + (v - cv) ** 2)
            r_inner = float(np.nanmin(r)) * 1.02
            r_outer = float(np.nanmax(r)) * 0.98
            tris = np.asarray(tri.triangles, dtype=np.int64)
            uc = u[tris].mean(axis=1)
            vc = v[tris].mean(axis=1)
            rc = np.sqrt((uc - cu) ** 2 + (vc - cv) ** 2)
            tri.set_mask((rc < r_inner) | (rc > r_outer))

        # Report
        report_path = os.path.join(self.cfg.out_dir, "deflection_compare.txt")
        with open(report_path, "w", encoding="utf-8") as fp:
            fp.write("Deflection comparison report (PINN)\n")
            fp.write(f"triangulation_base = deflection_{geom_base_rec.get('index', 1):02d}\n\n")
            fp.write("Cases:\n")
            for s in samples:
                rec = s["record"]
                idx = int(rec.get("index", 0))
                P = rec.get("P")
                order_disp = rec.get("order_display") or "-"
                fp.write(
                    f"- {idx:02d} P={P.tolist() if hasattr(P, 'tolist') else P} order={order_disp}"
                )
                if s.get("rigid_line"):
                    fp.write(f" | {s['rigid_line'].lstrip('#').strip()}")
                fp.write("\n")
            fp.write("\nDiffs (grouped by identical P):\n")

            # Common-scale maps (optional)
            if self.cfg.viz_compare_common_scale:
                for s in samples:
                    rec = s["record"]
                    idx = int(rec.get("index", 0))
                    out_name = f"deflection_{idx:02d}_common.png"
                    out_path = os.path.join(self.cfg.out_dir, out_name)
                    umag_plot = (
                        s["umag"] if vertex_pos is None else s["umag"][vertex_pos]
                    )
                    fig, ax = plt.subplots(figsize=(7.8, 6.8), constrained_layout=True)
                    sc = ax.tripcolor(
                        tri,
                        umag_plot,
                        shading="gouraud",
                        cmap=str(self.cfg.viz_colormap or "turbo"),
                        norm=colors.Normalize(vmin=0.0, vmax=global_umax),
                        edgecolors="none",
                    )
                    cbar = fig.colorbar(sc, ax=ax, shrink=0.92, pad=0.02)
                    cbar.set_label(f"Total displacement magnitude [{self.cfg.viz_units}] (common scale)")
                    ax.set_aspect("equal", adjustable="box")
                    ax.set_xlabel("u (best-fit plane)")
                    ax.set_ylabel("v (best-fit plane)")
                    title = f"{self.cfg.viz_title_prefix} | common scale"
                    P = rec.get("P")
                    if P is not None and len(P) >= 3:
                        title += f"  P=[{int(P[0])},{int(P[1])},{int(P[2])}]N"
                    od = rec.get("order_display")
                    if od:
                        title += f" (order={od})"
                    ax.set_title(title)
                    fig.savefig(out_path, dpi=200, bbox_inches="tight")
                    plt.close(fig)

            # Delta plots/metrics: compare within each identical-P group so tightening order is directly visible.
            def _key_from_P(rec: Dict[str, Any]) -> Tuple[int, ...]:
                P = rec.get("P")
                if P is None:
                    return tuple()
                arr = np.asarray(P, dtype=np.float64).reshape(-1)
                return tuple(int(round(float(x))) for x in arr.tolist())

            groups: Dict[Tuple[int, ...], List[Dict[str, Any]]] = {}
            for s in samples:
                rec = s["record"]
                key = _key_from_P(rec)
                groups.setdefault(key, []).append(s)

            for key, group in sorted(groups.items(), key=lambda kv: kv[0]):
                if len(group) < 2:
                    continue
                group = sorted(group, key=lambda s: int(s["record"].get("index", 0)))
                base = group[0]
                base_rec = base["record"]
                base_idx = int(base_rec.get("index", 0))
                fp.write(f"\nP={list(key)} base={base_idx:02d}:\n")

                for s in group[1:]:
                    rec = s["record"]
                    idx = int(rec.get("index", 0))
                    nodes = s["node_id"]
                    if nodes.shape != base_nodes.shape or not np.all(nodes == base_nodes):
                        fp.write(f"- {idx:02d}: node mismatch, skipped\n")
                        continue

                    dux = s["ux"] - base["ux"]
                    duy = s["uy"] - base["uy"]
                    duz = s["uz"] - base["uz"]
                    du = np.sqrt(dux * dux + duy * duy + duz * duz)
                    rms = float(np.sqrt(np.mean(du * du)))
                    maxv = float(np.max(du))
                    dmag = s["umag"] - base["umag"]
                    max_abs_dmag = float(np.max(np.abs(dmag)))
                    arg = int(np.argmax(du))
                    node_max = int(nodes[arg])
                    u_max = float(u[arg])
                    v_max = float(v[arg])
                    fp.write(
                        f"- {idx:02d}: rms|du|={rms:.3e} max|du|={maxv:.3e} "
                        f"max|Δ|u||={max_abs_dmag:.3e} @node={node_max} (u,v)=({u_max:.3f},{v_max:.3f})\n"
                    )

                    dmag_plot = dmag if vertex_pos is None else dmag[vertex_pos]
                    vlim = float(np.max(np.abs(dmag_plot))) + 1e-16
                    out_name = f"deflection_diff_{idx:02d}_minus_{base_idx:02d}.png"
                    out_path = os.path.join(self.cfg.out_dir, out_name)
                    fig, ax = plt.subplots(figsize=(7.8, 6.8), constrained_layout=True)
                    norm = colors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
                    sc = ax.tripcolor(
                        tri,
                        dmag_plot,
                        shading="gouraud",
                        cmap=str(self.cfg.viz_compare_cmap or "coolwarm"),
                        norm=norm,
                        edgecolors="none",
                    )
                    cbar = fig.colorbar(sc, ax=ax, shrink=0.92, pad=0.02)
                    cbar.set_label(f"Δ|u| [{self.cfg.viz_units}]")
                    ax.set_aspect("equal", adjustable="box")
                    ax.set_xlabel("u (best-fit plane)")
                    ax.set_ylabel("v (best-fit plane)")
                    title = f"Δ|u| vs base ({base_idx:02d})"
                    P = rec.get("P")
                    if P is not None and len(P) >= 3:
                        title += f"  P=[{int(P[0])},{int(P[1])},{int(P[2])}]N"
                    od = rec.get("order_display")
                    if od:
                        title += f" (order={od})"
                    ax.set_title(title)
                    fig.savefig(out_path, dpi=200, bbox_inches="tight")
                    plt.close(fig)

        print(f"[viz] comparison report -> {report_path}")


