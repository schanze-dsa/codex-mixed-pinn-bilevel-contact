#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pinn_model.py
-------------
Displacement field network for DFEM/PINN with preload conditioning.

Components:
- ParamEncoder: encodes normalized preload features -> condition z
  - GaussianFourierFeatures: optional positional encoding for coordinates
  - DisplacementNet: Graph neural network (GCN) backbone; inputs [x_feat, z] -> u(x; P)

Public factory:
    model = create_displacement_model(cfg)      # returns DisplacementModel
    u = model.u_fn(X, params)                   # X: (N,3) mm (normalized outside if needed)
                                               # params: dict; must contain either:
                                               #   "P_hat": preload feature vector; staged 鎯呭喌涓?
                                               #           鍖呭惈 [P_hat, mask, last, rank]锛岄暱搴?
                                               #           涓?4*n_bolts
                                               # or "P": (3,) with "preload_shift/scale" in cfg

Notes:
- This file鍙叧娉ㄢ€滅綉缁滃墠鍚戔€濓紝涓嶅仛鐗╃悊瑁呴厤锛涜缁冨惊鐜皢鎶婃湰妯″瀷涓庤兘閲?鎺ヨЕ绠楀瓙缁勫悎銆?
- 婵€娲婚粯璁?SiLU锛涘彲閫?GELU/RELU/Tanh銆?
- 娣峰悎绮惧害鍙€夛紙'float16' 鎴?'bfloat16'锛夛紱鏉冮噸淇濇寔 float32锛屾暟鍊肩ǔ瀹氥€?

Author: you
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict

import numpy as np
import tensorflow as tf


CONTACT_SURFACE_NORMALS_KEY = "__contact_surface_normals__"
CONTACT_SURFACE_T1_KEY = "__contact_surface_t1__"
CONTACT_SURFACE_T2_KEY = "__contact_surface_t2__"
CONTACT_SURFACE_SEMANTIC_DIM = 10


# -----------------------------
# Config dataclasses
# -----------------------------

@dataclass
class FourierConfig:
    num: int = 8              # number of Gaussian frequencies per axis; 0 -> disable
    sigma: float = 3.0        # std for frequency sampling (larger -> higher freq coverage)
    sigmas: Optional[Tuple[float, ...]] = (1.0, 10.0, 50.0)  # multi-scale sigmas; if set, overrides sigma
    trainable: bool = False    # whether to learn B instead of keeping it frozen

@dataclass
class EncoderConfig:
    in_dim: int = 3           # (P1,P2,P3) normalized
    width: int = 64
    depth: int = 2
    act: str = "silu"         # silu|gelu|relu|tanh
    out_dim: int = 64         # condition vector size

@dataclass
class FieldConfig:
    in_dim_coord: int = 3     # xyz (normalized outside if闇€瑕?
    fourier: FourierConfig = FourierConfig()
    cond_dim: int = 64
    # 浠ヤ笅 legacy MLP 鍙傛暟浠呬繚鐣欏吋瀹规€э紱褰撳墠瀹炵幇濮嬬粓璧?GCN 涓诲共
    width: int = 256
    depth: int = 7
    act: str = "silu"
    residual_skips: Tuple[int, int] = (3, 6)
    out_dim: int = 3          # displacement ux,uy,uz
    stress_out_dim: int = 6   # 搴斿姏鍒嗛噺杈撳嚭缁村害锛堥粯璁?6: 蟽xx,蟽yy,蟽zz,蟽xy,蟽yz,蟽xz锛夛紱<=0 鍏抽棴搴斿姏澶?
    use_graph: bool = True    # 鏄惁鍚敤 GCN 涓诲共锛涜嫢涓?False 灏嗘姤閿?
    graph_k: int = 12         # kNN 鍥句腑鐨勯偦灞呮暟閲?
    graph_knn_chunk: int = 1024  # 鏋勫缓 kNN/鍥惧嵎绉椂姣忔壒澶勭悊鐨勮妭鐐规暟閲?
    graph_precompute: bool = False  # 鏄惁鍦ㄦ瀯寤洪樁娈甸璁＄畻鍏ㄥ眬閭绘帴骞剁紦瀛?
    graph_layers: int = 4     # 鍥惧嵎绉眰鏁?
    graph_width: int = 192    # 姣忓眰鐨勯殣钘忕壒寰佺淮搴?
    graph_dropout: float = 0.0
    # 鍩轰簬鏉′欢鍚戦噺鐨?FiLM 璋冨埗
    use_film: bool = True
    # 浠呬负鍏煎鏃х増娈嬪樊寮€鍏筹紙宸茬Щ闄ゆ畫宸疄鐜帮級锛涗繚鐣欏瓧娈甸伩鍏嶅姞杞芥棫閰嶇疆鏃舵姤閿?
    graph_residual: bool = False
    # 绠€鍗曠‖绾︽潫鎺╃爜锛氫互鍦嗗瓟涓轰緥锛屽崐寰勫唴寮哄埗浣嶇Щ涓?0锛屽彲閫夊紑鍚?
    hard_bc_radius: Optional[float] = None
    hard_bc_center: Tuple[float, float] = (0.0, 0.0)
    hard_bc_dims: Tuple[bool, bool, bool] = (True, True, True)
    # 杈撳嚭缂╂斁锛氱綉缁滈娴嬫棤閲忕翰浣嶇Щ鍚庡啀涔樹互璇ュ昂搴︼紝渚夸簬寰皬閲忕骇鐨勬暟鍊肩ǔ瀹?
    output_scale: float = 1.0e-2
    output_scale_trainable: bool = False
    
    # DFEM mode: use learnable node embeddings instead of spatial coordinates
    dfem_mode: bool = False           # Enable pure DFEM mode
    n_nodes: Optional[int] = None     # Total number of mesh nodes (required if dfem_mode=True)
    node_emb_dim: int = 64            # Dimension of learnable node embeddings
    # Finite-domain spectral encoding (deterministic and geometry-aware)
    use_finite_spectral: bool = False
    finite_spectral_modes: int = 0
    finite_spectral_with_distance: bool = True
    # Engineering semantics from CDB tags (contact/bc/material/mirror)
    use_engineering_semantics: bool = False
    semantic_feat_dim: int = 0
    # Aleatoric uncertainty head (log-variance for displacement components)
    uncertainty_out_dim: int = 0
    # Sample-level adaptive depth routing:
    # easy samples use shallow head, hard samples use deep head.
    adaptive_depth_enabled: bool = False
    adaptive_depth_mode: str = "hard"  # hard | soft
    adaptive_depth_shallow_layers: int = 2
    adaptive_depth_threshold: float = 0.5
    adaptive_depth_temperature: float = 1.0
    adaptive_depth_route_source: str = "z_norm"  # z_norm | contact_residual
    stress_branch_early_split: bool = False
    use_eps_guided_stress_head: bool = False
    strict_mixed_default_eps_bridge: bool = False
    strict_mixed_contact_pointwise_stress: bool = False
    contact_stress_hybrid_enabled: bool = False

@dataclass
class ModelConfig:
    encoder: EncoderConfig = EncoderConfig()
    field: FieldConfig = FieldConfig()
    mixed_precision: Optional[str] = None      # None|'float16'|'bfloat16'
    preload_shift: float = 500.0               # for P normalization if only "P" is given
    preload_scale: float = 1500.0              # P_hat = (P - shift)/scale


@dataclass
class MixedFieldBatch:
    u: tf.Tensor
    sigma_vec: tf.Tensor
    cache_key: Tuple[Any, Any]


@dataclass
class MixedForwardCache:
    key: Optional[Tuple[Any, Any]] = None
    batch: Optional[MixedFieldBatch] = None


# -----------------------------
# Utilities
# -----------------------------

def _get_activation(name: str):
    name = (name or "silu").lower()
    if name == "silu":
        return tf.nn.silu
    if name == "gelu":
        return tf.nn.gelu
    if name == "relu":
        return tf.nn.relu
    if name == "tanh":
        return tf.nn.tanh
    raise ValueError(f"Unknown activation '{name}'")

def _maybe_mixed_precision(policy: Optional[str]):
    if policy:
        try:
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"[pinn_model] Mixed precision policy set to: {policy}")
        except Exception as e:
            print(f"[pinn_model] Failed to set mixed precision '{policy}': {e}")


def _stress_split_index(total_layers: int) -> int:
    total_layers = int(max(total_layers, 0))
    if total_layers <= 0:
        return 0
    return max(1, total_layers - 2)


def _engineering_strain_from_tape(
    tape: tf.GradientTape,
    coords: tf.Tensor,
    u: tf.Tensor,
) -> tf.Tensor:
    du_dx = tape.gradient(
        tf.reduce_sum(u[:, 0]),
        coords,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )
    dv_dx = tape.gradient(
        tf.reduce_sum(u[:, 1]),
        coords,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )
    dw_dx = tape.gradient(
        tf.reduce_sum(u[:, 2]),
        coords,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )

    eps_xx = du_dx[:, 0]
    eps_yy = dv_dx[:, 1]
    eps_zz = dw_dx[:, 2]
    gamma_yz = dv_dx[:, 2] + dw_dx[:, 1]
    gamma_xz = du_dx[:, 2] + dw_dx[:, 0]
    gamma_xy = du_dx[:, 1] + dv_dx[:, 0]
    return tf.stack([eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy], axis=1)


# -----------------------------
# Layers
# -----------------------------

class GaussianFourierFeatures(tf.keras.layers.Layer):
    """
    Map 3D coordinates x -> concat_k [sin(B_k x), cos(B_k x)] with B_k ~ N(0, sigma_k^2).
    - 鏀寔澶氬昂搴?sigma_k锛堜緥濡?[1,10,50]锛夛紝姣忎釜灏哄害閲囨牱 num 涓鐜囧悗鎷兼帴銆?
    - 鍙€夎 B_k 鍙樹负 trainable锛屼互渚跨綉缁滆嚜閫傚簲棰戞銆傞粯璁や繚鎸佸喕缁撱€?
    Mixed precision 鍏煎绛栫暐锛?
    - 缁熶竴鍦?float32 涓繘琛?matmul/sin/cos/concat锛屽啀 cast 鍥炶緭鍏?dtype锛堥€氬父鏄?float16锛夈€?
    """

    def __init__(
        self,
        in_dim: int,
        num: int,
        sigma: float,
        sigmas: Optional[Tuple[float, ...]] = None,
        trainable: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.num = int(num)
        self.sigma = float(sigma)
        self.sigmas = tuple(sigmas) if sigmas is not None else None
        self.trainable_B = bool(trainable)
        self.B_list: list[tf.Variable] = []

    def build(self, input_shape):
        if self.num <= 0:
            return
        rng = tf.random.Generator.from_non_deterministic_state()
        sigmas = self.sigmas if self.sigmas else (self.sigma,)
        for idx, sig in enumerate(sigmas):
            B_np = rng.normal(shape=(self.in_dim, self.num), dtype=tf.float32) * float(sig)
            self.B_list.append(
                tf.Variable(B_np, trainable=self.trainable_B, name=f"B_fourier_{idx}")
            )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        if self.num <= 0 or not self.B_list:
            return x
        # ---- 淇 dtype 涓嶅尮閰嶏細鍦?float32 閲岃绠楋紝鏈€鍚庡啀 cast 鍥炴潵 ----
        x32 = tf.cast(x, tf.float32)      # (N, in_dim)
        feat_bands = []
        for B in self.B_list:
            B32 = tf.cast(B, tf.float32)  # (in_dim, num)
            xb32 = tf.matmul(x32, B32)    # (N, num) float32
            feat_bands.append(tf.sin(xb32))
            feat_bands.append(tf.cos(xb32))
        feat32 = tf.concat(feat_bands + [x32], axis=-1)
        return tf.cast(feat32, x.dtype)   # 鍥炲埌涓庤緭鍏ヤ竴鑷寸殑 dtype锛坢ixed_float16 涓嬩负 float16锛?

    @property
    def out_dim(self) -> int:
        if self.num <= 0:
            return self.in_dim
        n_bands = len(self.sigmas) if self.sigmas else 1
        return n_bands * self.num * 2 + self.in_dim


class FiniteSpectralFeatures(tf.keras.layers.Layer):
    """Deterministic bounded-domain spectral features for geometry generalization."""

    def __init__(
        self,
        in_dim: int,
        modes: int,
        with_distance: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_dim = int(in_dim)
        self.modes = int(max(0, modes))
        self.with_distance = bool(with_distance)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, tf.float32)
        if self.modes <= 0:
            return x

        # Normalize to [0,1] in current batch for bounded-domain basis.
        xmin = tf.reduce_min(x, axis=0, keepdims=True)
        xmax = tf.reduce_max(x, axis=0, keepdims=True)
        span = tf.maximum(xmax - xmin, 1e-8)
        xn = (x - xmin) / span

        bands = []
        pi = tf.constant(np.pi, dtype=tf.float32)
        for k in range(1, self.modes + 1):
            arg = pi * float(k) * xn
            bands.append(tf.sin(arg))
            bands.append(tf.cos(arg))
        if self.with_distance:
            # Boundary distance in unit box as a cheap finite-domain proxy.
            d = tf.minimum(xn, 1.0 - xn)
            bands.append(d)
        bands.append(xn)
        return tf.concat(bands, axis=-1)

    @property
    def out_dim(self) -> int:
        if self.modes <= 0:
            return self.in_dim
        base = self.in_dim * self.modes * 2
        if self.with_distance:
            base += self.in_dim
        return base + self.in_dim


class MLP(tf.keras.layers.Layer):
    """Simple MLP block with configurable depth/width/activation."""

    def __init__(
        self,
        width: int,
        depth: int,
        act: str,
        final_dim: Optional[int] = None,
        dtype: Optional[tf.dtypes.DType] = None,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        self.act = _get_activation(act)
        self.final_dim = final_dim
        self._dense_dtype = dtype

        self.layers_dense = []
        for i in range(depth):
            dense_kwargs = {
                "units": width,
                "kernel_initializer": "he_uniform",
            }
            if self._dense_dtype is not None:
                dense_kwargs["dtype"] = self._dense_dtype
            self.layers_dense.append(tf.keras.layers.Dense(**dense_kwargs))
        if final_dim is not None:
            final_kwargs = {
                "units": final_dim,
                "kernel_initializer": "glorot_uniform",
            }
            if self._dense_dtype is not None:
                final_kwargs["dtype"] = self._dense_dtype
            self.final_dense = tf.keras.layers.Dense(**final_kwargs)
        else:
            self.final_dense = None

    def call(self, x: tf.Tensor) -> tf.Tensor:
        y = x
        for i in range(self.depth):
            y = self.layers_dense[i](y)
            y = self.act(y)
        if self.final_dense is not None:
            y = self.final_dense(y)
        return y


class GraphConvLayer(tf.keras.layers.Layer):
    """Simple graph message-passing layer over kNN neighborhoods."""

    def __init__(
        self,
        hidden_dim: int,
        k: int,
        act: str,
        dropout: float = 0.0,
        chunk_size: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k = max(int(k), 1)
        self.act = _get_activation(act)
        self.dropout = float(max(dropout, 0.0))
        # chunk_size 鍙傛暟浠呬负鍚戝悗鍏煎淇濈暀锛涙柊瀹炵幇涓轰竴娆℃€у苟琛屾眰瑙?
        self._unused_chunk = chunk_size
        self.lin = tf.keras.layers.Dense(
            hidden_dim,
            kernel_initializer="he_uniform",
        )

    def call(
        self,
        feat: tf.Tensor,
        coords: tf.Tensor,
        knn_idx: tf.Tensor,
        adj: tf.sparse.SparseTensor | None = None,
        training: bool | None = False,
    ) -> tf.Tensor:
        """
        feat   : (N, C)
        coords : (N, 3)
        knn_idx: (N, K)
        adj    : (N, N) Normalized SparseTensor (Optional, preferred for memory efficiency)
        """
        input_dtype = feat.dtype
        feat = tf.ensure_shape(feat, (None, self.hidden_dim))
        coords = tf.cast(coords, input_dtype)
        coords = tf.ensure_shape(coords, (None, 3))
        knn_idx = tf.ensure_shape(knn_idx, (None, self.k))

        # --- Optimization: Use Sparse MatMul if adj provided ---
        if adj is not None:
             # adj scale is 1/k, so matmul performs the mean aggregation
             # sparse_dense_matmul requires matching dtypes. adj is typically float32.
             # If feat is float16 (mixed precision), we must cast to float32 temporarily.
             if adj.values.dtype != feat.dtype:
                 agg = tf.sparse.sparse_dense_matmul(adj, tf.cast(feat, adj.values.dtype))
                 agg = tf.cast(agg, feat.dtype)
             else:
                 agg = tf.sparse.sparse_dense_matmul(adj, feat)  # (N, C)
        else:
             neighbors = tf.gather(feat, knn_idx)  # (N, K, C)
             neighbors.set_shape([None, self.k, self.hidden_dim])
             agg = tf.reduce_mean(neighbors, axis=1)  # (N, C)
        
        agg.set_shape([None, self.hidden_dim])

        if adj is not None:
            # Compute rel_mean and rel_std using sparse ops to avoid gather(coords)
            # which produces sparse gradients (IndexedSlices) and triggers warnings.
            # rel_mean = mean(x_j) - x_i
            # rel_std = std(x_j - x_i) = std(x_j)
            
            c_dtype = coords.dtype
            # Ensure float32 for matmul if mixed precision
            if adj.values.dtype != c_dtype:
                 coords_32 = tf.cast(coords, adj.values.dtype)
                 mean_x = tf.sparse.sparse_dense_matmul(adj, coords_32)
                 mean_x = tf.cast(mean_x, c_dtype)
                 
                 # E[x^2]
                 mean_sq_x = tf.sparse.sparse_dense_matmul(adj, tf.square(coords_32))
                 mean_sq_x = tf.cast(mean_sq_x, c_dtype)
            else:
                 mean_x = tf.sparse.sparse_dense_matmul(adj, coords)
                 # E[x^2]
                 mean_sq_x = tf.sparse.sparse_dense_matmul(adj, tf.square(coords))

            rel_mean = mean_x - coords
            # Var = E[x^2] - E[x]^2. Use relu for numerical stability.
            var_x = tf.nn.relu(mean_sq_x - tf.square(mean_x))
            rel_std = tf.sqrt(var_x)
        else:
            nbr_coords = tf.gather(coords, knn_idx)  # (N, K, 3)
            nbr_coords.set_shape([None, self.k, 3])
            rel = nbr_coords - tf.expand_dims(coords, axis=1)
            rel_mean = tf.reduce_mean(rel, axis=1)
            rel_std = tf.math.reduce_std(rel, axis=1)
        rel_feat = tf.concat([rel_mean, rel_std], axis=-1)  # (N, 6)
        rel_feat.set_shape([None, 6])

        mix = tf.concat([feat, agg, rel_feat], axis=-1)
        out = self.lin(mix)
        out = self.act(out)
        if self.dropout > 0.0:
            if training is None:
                training = False
            train_flag = tf.cast(training, tf.bool)
            out = tf.cond(
                train_flag,
                lambda: tf.nn.dropout(out, rate=self.dropout),
                lambda: out,
            )
        return out


def _build_knn_graph(x: tf.Tensor, k: int, chunk_size: int) -> tf.Tensor:
    """
    杩斿洖姣忎釜鐐圭殑 k 涓偦灞呯储寮?(N, k)銆?

    鏃╂湡瀹炵幇鍗充究鍋氫簡鎸夎鍒嗗潡锛屼緷鏃ч渶瑕佷负姣忎釜琛屽潡涓€娆℃€ф瀯閫犲ぇ灏忎负
    (chunk 脳 N) 鐨勮窛绂荤煩闃碉紝N 鍔ㄨ緞涓婁竾鏃朵細浜х敓鏁扮櫨 MB 鐨勭灛鏃跺垎閰嶏紝浠庤€?
    瑙﹀彂 GPU OOM銆傝繖閲屾敼涓?*鍙屽眰鍒嗗潡*锛氬浜庢瘡涓鍧楋紝鍐嶆寜鍒楀潡閬嶅巻鍏ㄩ泦锛?
    浠呬繚鐣欏綋鍓嶈鍧楃殑 top-k 涓棿缁撴灉锛屼娇寰椾换涓€鏃跺埢鍙渶淇濆瓨
    (chunk 脳 chunk) 鐨勮窛绂荤煩闃碉紝鍐呭瓨闇€姹傞檷鍒扮嚎鎬х骇鍒€?
    """

    x = tf.cast(x, tf.float32)
    n = tf.shape(x)[0]
    k = max(int(k), 1)
    chunk = max(int(chunk_size), 1)
    chunk = min(chunk, 1024)
    k_const = tf.constant(k, dtype=tf.int32)
    chunk_const = tf.constant(chunk, dtype=tf.int32)
    large_val = tf.constant(1e30, dtype=tf.float32)

    def _empty():
        return tf.zeros((0, k), dtype=tf.int32)

    def _build():
        with tf.device("/CPU:0"):
            x_sq = tf.reduce_sum(tf.square(x), axis=1)  # (N,)
            ta = tf.TensorArray(
                dtype=tf.int32,
                size=0,
                dynamic_size=True,
                clear_after_read=False,
                element_shape=None,  # Allow variable-sized chunks
                infer_shape=False,    # Disable shape inference in while_loop
            )

            def _cond(start, *_):
                return tf.less(start, n)

            def _body(start, ta_handle, write_idx):
                end = tf.minimum(n, start + chunk_const)
                rows = tf.range(start, end)
                chunk_len = tf.shape(rows)[0]
                x_chunk = tf.gather(x, rows)
                chunk_sq = tf.gather(x_sq, rows)
                best_shape = tf.stack([chunk_len, k_const])
                best_dist = tf.fill(best_shape, large_val)
                best_idx = tf.zeros(best_shape, dtype=tf.int32)

                def _inner_cond(col_start, *_):
                    return tf.less(col_start, n)

                def _inner_body(col_start, best_d, best_i):
                    col_end = tf.minimum(n, col_start + chunk_const)
                    cols = tf.range(col_start, col_end)
                    x_cols = tf.gather(x, cols)
                    col_sq = tf.gather(x_sq, cols)
                    dist = (
                        tf.expand_dims(chunk_sq, 1)
                        + tf.expand_dims(col_sq, 0)
                        - 2.0 * tf.matmul(x_chunk, x_cols, transpose_b=True)
                    )
                    dist = tf.maximum(dist, 0.0)
                    same = tf.cast(
                        tf.equal(tf.expand_dims(rows, 1), tf.expand_dims(cols, 0)),
                        dist.dtype,
                    )
                    dist = dist + same * 1e9

                    combined_dist = tf.concat([best_d, dist], axis=1)
                    tiled_cols = tf.tile(
                        tf.expand_dims(tf.cast(cols, tf.int32), 0), [chunk_len, 1]
                    )
                    combined_idx = tf.concat([best_i, tiled_cols], axis=1)

                    neg_dist = -combined_dist
                    vals, top_idx = tf.math.top_k(neg_dist, k=k)
                    new_best_dist = -vals
                    new_best_idx = tf.gather(combined_idx, top_idx, batch_dims=1)
                    return col_end, new_best_dist, new_best_idx

                start_inner = tf.constant(0, dtype=tf.int32)
                _, best_final, idx_final = tf.while_loop(
                    _inner_cond,
                    _inner_body,
                    (start_inner, best_dist, best_idx),
                    parallel_iterations=1,
                )
                ta_handle = ta_handle.write(write_idx, idx_final)
                return end, ta_handle, write_idx + 1

            start0 = tf.constant(0, dtype=tf.int32)
            write0 = tf.constant(0, dtype=tf.int32)
            _, ta_final, _ = tf.while_loop(
                _cond, _body, (start0, ta, write0), parallel_iterations=1
            )
            return ta_final.concat()

    return tf.cond(tf.equal(n, 0), _empty, _build)


def _knn_to_adj(knn_idx: tf.Tensor, n_nodes: int | tf.Tensor) -> tf.sparse.SparseTensor:
    """
    Convert (N, K) knn indices to normalized (N, N) sparse adjacency matrix.
    Values are 1.0/K (row-normalized).
    """
    knn_idx = tf.cast(knn_idx, tf.int64)
    N = tf.shape(knn_idx)[0]
    K = tf.shape(knn_idx)[1]
    
    # Construct indices: (row, col)
    # rows: [0,0,..,0, 1,1,..,1, ...]
    row_idx = tf.repeat(tf.range(N, dtype=tf.int64), repeats=K)
    col_idx = tf.reshape(knn_idx, [-1])
    
    indices = tf.stack([row_idx, col_idx], axis=1) # (N*K, 2)
    
    # Values: 1/K
    val = tf.cast(1.0 / tf.cast(K, tf.float32), tf.float32)
    values = tf.fill([N * K], val)
    
    # Sort indices (required for sparse operations)
    # Since we constructed row_idx sequentially, it should be sorted by row, 
    # but strictly allow sparse_reorder to ensure correctness if col order matters or implementation changes.
    sp = tf.sparse.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=[tf.cast(n_nodes, tf.int64), tf.cast(n_nodes, tf.int64)]
    )
    return tf.sparse.reorder(sp)


# -----------------------------
# Networks
# -----------------------------

class ParamEncoder(tf.keras.layers.Layer):
    """Encode normalized preload vector (P_hat) to a condition vector z."""
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.in_dim = int(getattr(cfg, "in_dim", 0) or 0)
        self.mlp = MLP(
            width=cfg.width,
            depth=cfg.depth,
            act=cfg.act,
            final_dim=cfg.out_dim,
        )

    def call(self, P_hat: tf.Tensor) -> tf.Tensor:
        # Ensure 2D: (B,3)
        if P_hat.shape.rank == 1:
            P_hat = tf.reshape(P_hat, (1, -1))
        P_hat = self._normalize_dim(P_hat)
        return self.mlp(P_hat)  # (B, out_dim)

    def _normalize_dim(self, P_hat: tf.Tensor) -> tf.Tensor:
        """Pad/trim ``P_hat`` to match the configured encoder input width."""

        target = self.in_dim
        if target <= 0:
            return P_hat

        # 闈欐€佸舰鐘跺凡鍖归厤鍒欑洿鎺ヨ繑鍥?
        if P_hat.shape.rank is not None and P_hat.shape[-1] == target:
            P_hat.set_shape((None, target))
            return P_hat

        cur = tf.shape(P_hat)[-1]
        target_tf = tf.cast(target, tf.int32)

        # Avoid tf.cond to prevent trace-time Optional type inconsistencies when using
        # mixed precision (half vs int32). We pad with zeros only when needed, then
        # slice to the target width so both under- and over-length inputs are handled
        # in a single branch with consistent dtypes.
        pad_width = tf.maximum(target_tf - cur, 0)
        pad_zeros = tf.zeros((tf.shape(P_hat)[0], pad_width), dtype=P_hat.dtype)
        padded = tf.concat([P_hat, pad_zeros], axis=-1)
        adjusted = padded[:, :target_tf]
        adjusted.set_shape((None, target))
        return adjusted


class DisplacementNet(tf.keras.Model):
    """
    Core field network: input features = [x_feat, z_broadcast] -> u
    - x_feat = pe(x) if PE enabled else x
    - z is per-parameter vector; we broadcast to match number of spatial samples
    """
    def __init__(self, cfg: FieldConfig):
        super().__init__()
        self.cfg = cfg
        self.use_graph = bool(cfg.use_graph)
        self.use_film = bool(getattr(cfg, "use_film", False))
        self.use_finite_spectral = bool(getattr(cfg, "use_finite_spectral", False))
        self.use_engineering_semantics = bool(getattr(cfg, "use_engineering_semantics", False))
        self.stress_branch_early_split = bool(getattr(cfg, "stress_branch_early_split", False))
        self.use_eps_guided_stress_head = bool(getattr(cfg, "use_eps_guided_stress_head", False))
        self.contact_stress_hybrid_enabled = bool(getattr(cfg, "contact_stress_hybrid_enabled", False))
        self.adaptive_depth_enabled = bool(getattr(cfg, "adaptive_depth_enabled", False))
        self.adaptive_depth_mode = str(getattr(cfg, "adaptive_depth_mode", "hard") or "hard").strip().lower()
        if self.adaptive_depth_mode not in {"hard", "soft"}:
            raise ValueError(f"Unsupported adaptive_depth_mode='{self.adaptive_depth_mode}', expect 'hard' or 'soft'.")
        self.adaptive_depth_shallow_layers = max(
            1, int(getattr(cfg, "adaptive_depth_shallow_layers", 1) or 1)
        )
        self.adaptive_depth_threshold = float(getattr(cfg, "adaptive_depth_threshold", 0.5))
        self.adaptive_depth_temperature = max(
            1.0e-6, float(getattr(cfg, "adaptive_depth_temperature", 1.0))
        )
        self.adaptive_depth_route_source = str(
            getattr(cfg, "adaptive_depth_route_source", "z_norm") or "z_norm"
        ).strip().lower()
        if self.adaptive_depth_route_source not in {"z_norm", "contact_residual"}:
            raise ValueError(
                "Unsupported adaptive_depth_route_source="
                f"'{self.adaptive_depth_route_source}', expect 'z_norm' or 'contact_residual'."
            )
        self._contact_residual_hint = tf.Variable(
            0.0,
            trainable=False,
            dtype=tf.float32,
            name="contact_residual_hint",
        )

        # Fourier PE (used if not in DFEM mode)
        self.pe = GaussianFourierFeatures(
            in_dim=cfg.in_dim_coord,
            num=cfg.fourier.num,
            sigma=cfg.fourier.sigma,
            sigmas=cfg.fourier.sigmas,
            trainable=cfg.fourier.trainable,
        )
        self.finite_pe = FiniteSpectralFeatures(
            in_dim=cfg.in_dim_coord,
            modes=int(getattr(cfg, "finite_spectral_modes", 0)),
            with_distance=bool(getattr(cfg, "finite_spectral_with_distance", True)),
        )
        self._node_semantic_features: Optional[tf.Tensor] = None
        self._contact_surface_semantic_features: Optional[tf.Tensor] = None

        # DFEM mode: learnable node embeddings instead of positional encoding
        self.dfem_mode = cfg.dfem_mode
        base_feat_dim = cfg.node_emb_dim if self.dfem_mode else self.pe.out_dim
        if self.use_finite_spectral:
            base_feat_dim += self.finite_pe.out_dim

        if self.dfem_mode:
            if cfg.n_nodes is None or cfg.n_nodes <= 0:
                raise ValueError(
                    "FieldConfig.dfem_mode=True requires n_nodes > 0, "
                    f"got {cfg.n_nodes}"
                )
            self.n_nodes = cfg.n_nodes
            # Learnable embeddings for each node
            self.node_embeddings = tf.Variable(
                tf.random.normal((self.n_nodes, cfg.node_emb_dim), stddev=0.02),
                trainable=True,
                name="node_embeddings"
            )
            in_dim_total = base_feat_dim + cfg.cond_dim
        else:
            in_dim_total = base_feat_dim + cfg.cond_dim

        # MLP fallback (used when graph is disabled or input is not full mesh)
        self.mlp_act = _get_activation(cfg.act)
        self.mlp_layers: list[tf.keras.layers.Layer] = []
        for _ in range(int(cfg.depth)):
            self.mlp_layers.append(
                tf.keras.layers.Dense(
                    cfg.width,
                    kernel_initializer="he_uniform",
                )
            )
        self.mlp_out = tf.keras.layers.Dense(
            cfg.out_dim,
            kernel_initializer="glorot_uniform",
        )
        self.stress_branch_mlp_split_index = _stress_split_index(len(self.mlp_layers))
        self.stress_branch_mlp_layers: list[tf.keras.layers.Layer] = []
        if self.stress_branch_early_split and cfg.stress_out_dim > 0:
            mlp_branch_depth = max(1, len(self.mlp_layers) - self.stress_branch_mlp_split_index)
            for bi in range(mlp_branch_depth):
                self.stress_branch_mlp_layers.append(
                    tf.keras.layers.Dense(
                        cfg.width,
                        kernel_initializer="he_uniform",
                        name=f"stress_branch_mlp_{bi}",
                    )
                )
        self.mlp_out_shallow = None
        self.mlp_out_deep = None
        if self.adaptive_depth_enabled:
            self.mlp_out_shallow = tf.keras.layers.Dense(
                cfg.out_dim,
                kernel_initializer="glorot_uniform",
                name="mlp_head_shallow",
            )
            self.mlp_out_deep = tf.keras.layers.Dense(
                cfg.out_dim,
                kernel_initializer="glorot_uniform",
                name="mlp_head_deep",
            )

        self.graph_proj = tf.keras.layers.Dense(
            cfg.graph_width,
            kernel_initializer="he_uniform",
        )
        self.graph_layers = [
            GraphConvLayer(
                hidden_dim=cfg.graph_width,
                k=cfg.graph_k,
                act=cfg.act,
                dropout=cfg.graph_dropout,
                chunk_size=cfg.graph_knn_chunk,
            )
            for _ in range(cfg.graph_layers)
        ]
        # FiLM 璋冨埗锛氫负姣忓眰鍑嗗 纬/尾锛屽垵濮嬩负鎭掔瓑锛埼?1, 尾=0锛?
        self.stress_branch_graph_split_index = _stress_split_index(len(self.graph_layers))
        self.stress_branch_graph_layers: list[GraphConvLayer] = []
        self.stress_branch_graph_norm = None
        if self.stress_branch_early_split and cfg.stress_out_dim > 0:
            graph_branch_depth = max(1, len(self.graph_layers) - self.stress_branch_graph_split_index)
            self.stress_branch_graph_layers = [
                GraphConvLayer(
                    hidden_dim=cfg.graph_width,
                    k=cfg.graph_k,
                    act=cfg.act,
                    dropout=cfg.graph_dropout,
                    chunk_size=cfg.graph_knn_chunk,
                )
                for _ in range(graph_branch_depth)
            ]
            self.stress_branch_graph_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.film_gamma: list[tf.keras.layers.Layer] = []
        self.film_beta: list[tf.keras.layers.Layer] = []
        if self.use_film:
            for li in range(cfg.graph_layers):
                self.film_gamma.append(
                    tf.keras.layers.Dense(
                        cfg.graph_width,
                        kernel_initializer="zeros",
                        bias_initializer="ones",
                        name=f"film_gamma_{li}",
                    )
                )
                self.film_beta.append(
                    tf.keras.layers.Dense(
                        cfg.graph_width,
                        kernel_initializer="zeros",
                        bias_initializer="zeros",
                        name=f"film_beta_{li}",
                    )
                )
        self.graph_norm = tf.keras.layers.LayerNormalization(axis=-1)
        self.graph_out = tf.keras.layers.Dense(
            cfg.out_dim,
            kernel_initializer="glorot_uniform",
        )
        self.graph_out_shallow = None
        self.graph_out_deep = None
        if self.adaptive_depth_enabled:
            self.graph_out_shallow = tf.keras.layers.Dense(
                cfg.out_dim,
                kernel_initializer="glorot_uniform",
                name="graph_head_shallow",
            )
            self.graph_out_deep = tf.keras.layers.Dense(
                cfg.out_dim,
                kernel_initializer="glorot_uniform",
                name="graph_head_deep",
            )
        self.stress_out = None
        self.stress_out_mlp = None
        self.stress_out_shallow = None
        self.stress_out_deep = None
        self.stress_out_mlp_shallow = None
        self.stress_out_mlp_deep = None
        self.stress_out_eps = None
        self.stress_out_eps_mlp = None
        self.stress_semantic_proj_mlp = None
        self.stress_semantic_proj_graph = None
        self.stress_contact_surface_proj_mlp = None
        self.stress_contact_surface_proj_graph = None
        if cfg.stress_out_dim > 0:
            self.stress_out = tf.keras.layers.Dense(
                cfg.stress_out_dim,
                kernel_initializer="glorot_uniform",
                name="stress_head_graph",
            )
            self.stress_out_mlp = tf.keras.layers.Dense(
                cfg.stress_out_dim,
                kernel_initializer="glorot_uniform",
                name="stress_head_mlp",
            )
            if self.adaptive_depth_enabled:
                self.stress_out_shallow = tf.keras.layers.Dense(
                    cfg.stress_out_dim,
                    kernel_initializer="glorot_uniform",
                    name="stress_head_graph_shallow",
                )
                self.stress_out_deep = tf.keras.layers.Dense(
                    cfg.stress_out_dim,
                    kernel_initializer="glorot_uniform",
                    name="stress_head_graph_deep",
                )
                self.stress_out_mlp_shallow = tf.keras.layers.Dense(
                    cfg.stress_out_dim,
                    kernel_initializer="glorot_uniform",
                    name="stress_head_mlp_shallow",
                )
                self.stress_out_mlp_deep = tf.keras.layers.Dense(
                    cfg.stress_out_dim,
                    kernel_initializer="glorot_uniform",
                    name="stress_head_mlp_deep",
                )
            if self.use_eps_guided_stress_head:
                self.stress_out_eps = tf.keras.layers.Dense(
                    cfg.stress_out_dim,
                    kernel_initializer="glorot_uniform",
                    name="stress_head_graph_eps",
                )
                self.stress_out_eps_mlp = tf.keras.layers.Dense(
                    cfg.stress_out_dim,
                    kernel_initializer="glorot_uniform",
                    name="stress_head_mlp_eps",
                )
            if self.use_engineering_semantics and int(getattr(cfg, "semantic_feat_dim", 0) or 0) > 0:
                self.stress_semantic_proj_mlp = tf.keras.layers.Dense(
                    cfg.width,
                    kernel_initializer="glorot_uniform",
                    name="stress_semantic_proj_mlp",
                )
                self.stress_semantic_proj_graph = tf.keras.layers.Dense(
                    cfg.graph_width,
                    kernel_initializer="glorot_uniform",
                    name="stress_semantic_proj_graph",
                )
            self.stress_contact_surface_proj_mlp = tf.keras.layers.Dense(
                cfg.width,
                kernel_initializer="glorot_uniform",
                name="stress_contact_surface_proj_mlp",
            )
            self.stress_contact_surface_proj_graph = tf.keras.layers.Dense(
                cfg.graph_width,
                kernel_initializer="glorot_uniform",
                name="stress_contact_surface_proj_graph",
            )
        self.uncertainty_out = None
        self.uncertainty_out_mlp = None
        self.uncertainty_out_shallow = None
        self.uncertainty_out_deep = None
        self.uncertainty_out_mlp_shallow = None
        self.uncertainty_out_mlp_deep = None
        if int(getattr(cfg, "uncertainty_out_dim", 0) or 0) > 0:
            uod = int(cfg.uncertainty_out_dim)
            self.uncertainty_out = tf.keras.layers.Dense(
                uod,
                kernel_initializer="glorot_uniform",
                name="uncertainty_head_graph",
            )
            self.uncertainty_out_mlp = tf.keras.layers.Dense(
                uod,
                kernel_initializer="glorot_uniform",
                name="uncertainty_head_mlp",
            )
            if self.adaptive_depth_enabled:
                self.uncertainty_out_shallow = tf.keras.layers.Dense(
                    uod,
                    kernel_initializer="glorot_uniform",
                    name="uncertainty_head_graph_shallow",
                )
                self.uncertainty_out_deep = tf.keras.layers.Dense(
                    uod,
                    kernel_initializer="glorot_uniform",
                    name="uncertainty_head_graph_deep",
                )
                self.uncertainty_out_mlp_shallow = tf.keras.layers.Dense(
                    uod,
                    kernel_initializer="glorot_uniform",
                    name="uncertainty_head_mlp_shallow",
                )
                self.uncertainty_out_mlp_deep = tf.keras.layers.Dense(
                    uod,
                    kernel_initializer="glorot_uniform",
                    name="uncertainty_head_mlp_deep",
                )
        # 鍏ㄥ眬閭绘帴缂撳瓨锛堝彲閫夛級
        self._global_knn_idx: Optional[tf.Tensor] = None
        self._global_adj: Optional[tf.sparse.SparseTensor] = None
        self._global_knn_n: Optional[int] = None

        # 杈撳嚭缂╂斁锛堝彲閫夊彲璁粌锛夛紝渚夸簬寰皬浣嶇Щ閲忕骇鐨勬暟鍊肩ǔ瀹?
        scale_init = tf.constant(getattr(cfg, "output_scale", 1.0), dtype=tf.float32)
        if getattr(cfg, "output_scale_trainable", False):
            self.output_scale = tf.Variable(scale_init, trainable=True, name="output_scale")
        else:
            self.output_scale = tf.cast(scale_init, tf.float32)

    def _uses_mlp_stress_head(self, stress_feat: tf.Tensor) -> bool:
        return bool(
            self.stress_out_mlp is not None
            and (stress_feat.shape.rank is None or stress_feat.shape[-1] != self.cfg.graph_width)
        )

    def predict_stress_from_features(
        self,
        stress_feat: tf.Tensor,
        eps_bridge: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        if self.stress_out is None:
            raise ValueError("stress head disabled (stress_out_dim<=0)")

        stress_feat = tf.convert_to_tensor(stress_feat)
        use_mlp_head = self._uses_mlp_stress_head(stress_feat)

        if eps_bridge is not None:
            eps_bridge = tf.cast(eps_bridge, stress_feat.dtype)
            fused = tf.concat([stress_feat, eps_bridge], axis=-1)
            if use_mlp_head and self.stress_out_eps_mlp is not None:
                return self.stress_out_eps_mlp(fused)
            if self.stress_out_eps is not None:
                return self.stress_out_eps(fused)

        if use_mlp_head:
            return self.stress_out_mlp(stress_feat)
        return self.stress_out(stress_feat)

    def set_node_semantic_features(self, features: np.ndarray | tf.Tensor):
        """Attach per-node engineering semantic features (N_nodes, F)."""

        feats = tf.convert_to_tensor(features, dtype=tf.float32)
        feats = tf.ensure_shape(feats, (None, None))
        expected_dim = int(getattr(self.cfg, "semantic_feat_dim", 0) or 0)
        if self.dfem_mode and self.cfg.n_nodes is not None:
            n = int(self.cfg.n_nodes)
            if feats.shape.rank is not None and feats.shape[0] is not None and int(feats.shape[0]) != n:
                raise ValueError(
                    f"semantic feature rows must match n_nodes={n}, got {int(feats.shape[0])}"
                )
        if expected_dim > 0 and feats.shape.rank is not None and feats.shape[-1] is not None:
            if int(feats.shape[-1]) != expected_dim:
                raise ValueError(
                    f"semantic feature columns must match semantic_feat_dim={expected_dim}, "
                    f"got {int(feats.shape[-1])}"
                )
        self._node_semantic_features = feats

    @staticmethod
    def assemble_contact_surface_semantic_features(
        normals: np.ndarray | tf.Tensor,
        t1: np.ndarray | tf.Tensor,
        t2: np.ndarray | tf.Tensor,
    ) -> tf.Tensor:
        normals = tf.ensure_shape(tf.convert_to_tensor(normals, dtype=tf.float32), (None, 3))
        t1 = tf.ensure_shape(tf.convert_to_tensor(t1, dtype=tf.float32), (None, 3))
        t2 = tf.ensure_shape(tf.convert_to_tensor(t2, dtype=tf.float32), (None, 3))
        n_rows = tf.shape(normals)[0]
        contact_flag = tf.ones((n_rows, 1), dtype=tf.float32)
        return tf.concat([contact_flag, normals, t1, t2], axis=-1)

    def set_contact_surface_semantic_features(self, features: np.ndarray | tf.Tensor):
        """Attach per-sample contact-surface semantics for pointwise stress evaluation."""

        feats = tf.convert_to_tensor(features, dtype=tf.float32)
        feats = tf.ensure_shape(feats, (None, None))
        if feats.shape.rank is not None and feats.shape[-1] is not None:
            if int(feats.shape[-1]) != CONTACT_SURFACE_SEMANTIC_DIM:
                raise ValueError(
                    "contact-surface semantic feature columns must match "
                    f"{CONTACT_SURFACE_SEMANTIC_DIM}, got {int(feats.shape[-1])}"
                )
        self._contact_surface_semantic_features = feats

    def set_contact_surface_frame(
        self,
        normals: np.ndarray | tf.Tensor,
        t1: np.ndarray | tf.Tensor,
        t2: np.ndarray | tf.Tensor,
    ):
        """Assemble and attach contact-surface semantics from a local surface frame."""

        self.set_contact_surface_semantic_features(
            self.assemble_contact_surface_semantic_features(normals, t1, t2)
        )

    def clear_contact_surface_semantic_features(self):
        self._contact_surface_semantic_features = None

    def _resolve_semantic_features(self, n_rows: tf.Tensor, *, dtype: tf.dtypes.DType) -> Optional[tf.Tensor]:
        if not self.use_engineering_semantics:
            return None
        sem_dim = int(getattr(self.cfg, "semantic_feat_dim", 0) or 0)
        if sem_dim <= 0:
            return None
        if self._node_semantic_features is not None:
            n_sem = tf.shape(self._node_semantic_features)[0]
            sem = tf.cond(
                tf.equal(n_sem, n_rows),
                lambda: self._node_semantic_features,
                lambda: tf.zeros((n_rows, sem_dim), dtype=tf.float32),
            )
        else:
            sem = tf.zeros((n_rows, sem_dim), dtype=tf.float32)
        return tf.cast(sem, dtype)

    def _resolve_contact_surface_semantic_features(
        self,
        n_rows: tf.Tensor,
        *,
        dtype: tf.dtypes.DType,
    ) -> Optional[tf.Tensor]:
        if self._contact_surface_semantic_features is None:
            return None
        n_sem = tf.shape(self._contact_surface_semantic_features)[0]
        sem = tf.cond(
            tf.equal(n_sem, n_rows),
            lambda: self._contact_surface_semantic_features,
            lambda: tf.zeros((n_rows, CONTACT_SURFACE_SEMANTIC_DIM), dtype=tf.float32),
        )
        return tf.cast(sem, dtype)

    def _fuse_stress_semantics(
        self,
        stress_feat: tf.Tensor,
        semantic_feat: Optional[tf.Tensor],
    ) -> tf.Tensor:
        stress_feat = tf.convert_to_tensor(stress_feat)
        fused = stress_feat

        if semantic_feat is not None:
            semantic_feat = tf.cast(semantic_feat, stress_feat.dtype)
            use_graph_proj = (
                self.stress_semantic_proj_graph is not None
                and stress_feat.shape.rank is not None
                and stress_feat.shape[-1] == self.cfg.graph_width
            )
            proj_layer = self.stress_semantic_proj_graph if use_graph_proj else self.stress_semantic_proj_mlp
            if proj_layer is not None:
                fused = fused + tf.cast(proj_layer(semantic_feat), stress_feat.dtype)

        contact_surface_feat = self._resolve_contact_surface_semantic_features(
            tf.shape(stress_feat)[0],
            dtype=stress_feat.dtype,
        )
        if contact_surface_feat is not None:
            use_graph_proj = (
                self.stress_contact_surface_proj_graph is not None
                and stress_feat.shape.rank is not None
                and stress_feat.shape[-1] == self.cfg.graph_width
            )
            proj_layer = (
                self.stress_contact_surface_proj_graph
                if use_graph_proj
                else self.stress_contact_surface_proj_mlp
            )
            if proj_layer is not None:
                fused = fused + tf.cast(proj_layer(contact_surface_feat), stress_feat.dtype)

        return fused

    def _contact_mask_from_semantics(
        self,
        semantic_feat: Optional[tf.Tensor],
        *,
        dtype: tf.dtypes.DType,
    ) -> Optional[tf.Tensor]:
        if not self.contact_stress_hybrid_enabled:
            return None
        if semantic_feat is None:
            raise ValueError(
                "contact_stress_hybrid_enabled=True requires engineering semantic features with a contact mask."
            )
        return tf.cast(tf.greater(semantic_feat[:, 0:1], 0.5), dtype)

    def _blend_contact_stress_features(
        self,
        stress_feat: tf.Tensor,
        local_stress_feat: Optional[tf.Tensor],
        semantic_feat: Optional[tf.Tensor],
    ) -> tf.Tensor:
        if local_stress_feat is None:
            return stress_feat
        contact_mask = self._contact_mask_from_semantics(semantic_feat, dtype=stress_feat.dtype)
        if contact_mask is None:
            return stress_feat
        one = tf.cast(1.0, stress_feat.dtype)
        local_stress_feat = tf.cast(local_stress_feat, stress_feat.dtype)
        return (one - contact_mask) * stress_feat + contact_mask * local_stress_feat

    def prebuild_adjacency(self, X_nodes: tf.Tensor | np.ndarray):
        """
        Pre-build and cache the adjacency graph using node coordinates.
        Should be called once during initialization with all mesh node coordinates.
        
        Args:
            X_nodes: (N_nodes, 3) node coordinates
        """
        if not self.dfem_mode:
            # For traditional PINN, this is optional but can still cache
            pass
            
        X_nodes = tf.convert_to_tensor(X_nodes, dtype=tf.float32)
        n_nodes = tf.shape(X_nodes)[0]
        
        # Build KNN graph
        knn_idx = _build_knn_graph(X_nodes, self.cfg.graph_k, self.cfg.graph_knn_chunk)
        adj = _knn_to_adj(knn_idx, n_nodes)
        
        # Cache
        self._global_knn_idx = knn_idx
        self._global_adj = adj
        self._global_knn_n = int(n_nodes.numpy() if hasattr(n_nodes, 'numpy') else n_nodes)
        
        print(f"[DisplacementNet] Pre-built adjacency graph: {self._global_knn_n} nodes, k={self.cfg.graph_k}")

    def call(
        self,
        x: tf.Tensor,
        z: tf.Tensor,
        training: bool | None = False,
        return_stress: bool = False,
        return_stress_features: bool = False,
        return_uncertainty: bool = False,
        force_pointwise: bool = False,
    ) -> tf.Tensor | Tuple[tf.Tensor, tf.Tensor] | Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        x : (N,3) coordinates (already normalized if you閲囩敤褰掍竴鍖?
        z : (B,cond_dim) or (cond_dim,)
        Returns:
            u: (N,3)
        """
        x = tf.convert_to_tensor(x)
        z = tf.convert_to_tensor(z)
        
        # Ensure z is 2D: (B, cond_dim)
        # Static shape check if possible, otherwise dynamic
        if z.shape.rank is not None and z.shape.rank == 1:
            z = tf.reshape(z, (1, -1))
        
        # Broadcast z to N samples
        # logic: if B != 1 and B != N, fallback to B=1; then broadcast B=1 to N
        
        N = tf.shape(x)[0]
        B = tf.shape(z)[0]

        # --- 淇鐐?1锛氬鐞?Fallback 閫昏緫 ---
        # 鍘熶唬鐮? if tf.not_equal(B, 1) and tf.not_equal(B, N): ...
        # 鏂颁唬鐮? 浣跨敤 tf.cond
        condition_fallback = tf.logical_and(tf.not_equal(B, 1), tf.not_equal(B, N))
        z = tf.cond(condition_fallback, lambda: z[:1], lambda: z)
        
        # 鏇存柊 B (鍥犱负 z 鍙兘鍙樹簡)
        B = tf.shape(z)[0]

        # --- 淇鐐?2锛氬鐞嗗箍鎾€昏緫 (浣犵幇鍦ㄧ殑鎶ラ敊鐐? ---
        # 鍘熶唬鐮? if tf.equal(B, 1): ... else: ...
        # 鏂颁唬鐮? 浣跨敤 tf.cond
        zb = tf.cond(
            tf.equal(B, 1), 
            lambda: tf.repeat(z, repeats=N, axis=0), 
            lambda: z
        )

        # --- 鍚庣画閫昏緫淇濇寔涓嶅彉 ---
        feat_dtype = x.dtype
        if zb.dtype != feat_dtype:
            zb = tf.cast(zb, feat_dtype)

        # DFEM mode: use node embeddings; Traditional: use positional encoding
        if self.dfem_mode:
            # x should contain node indices in DFEM mode: (N,) or (N,1) or (N,3) ignored
            # We use implicit indexing: x[i] corresponds to node i
            node_indices = tf.range(N, dtype=tf.int32)
            if self.cfg.n_nodes is not None and int(self.cfg.n_nodes) > 0:
                node_indices = tf.math.mod(node_indices, tf.cast(self.cfg.n_nodes, tf.int32))
            x_feat = tf.gather(self.node_embeddings, node_indices)  # (N, node_emb_dim)
            if self.use_finite_spectral:
                x_spec = self.finite_pe(tf.cast(x, tf.float32))
                x_feat = tf.concat([x_feat, tf.cast(x_spec, x_feat.dtype)], axis=-1)
        else:
            # Traditional PINN: positional encoding of coordinates
            x_feat = self.pe(x)  # (N, fourier_dim)
            if self.use_finite_spectral:
                x_spec = self.finite_pe(tf.cast(x, tf.float32))
                x_feat = tf.concat([x_feat, tf.cast(x_spec, x_feat.dtype)], axis=-1)

        semantic_feat = self._resolve_semantic_features(N, dtype=feat_dtype)

        if x_feat.dtype != feat_dtype:
            x_feat = tf.cast(x_feat, feat_dtype)
        
        h = tf.concat([x_feat, zb], axis=-1)

        def _apply_output(
            u_out: tf.Tensor,
            coords: tf.Tensor,
            hfeat: tf.Tensor,
            stress_hfeat: Optional[tf.Tensor] = None,
            semantic_feat: Optional[tf.Tensor] = None,
            local_stress_feat: Optional[tf.Tensor] = None,
        ):
            # Output scaling: network predicts normalized displacement first.
            scale = tf.cast(self.output_scale, u_out.dtype)
            u_out = u_out * scale

            # Optional hard BC mask for points inside the constrained hole radius.
            if self.cfg.hard_bc_radius is not None and float(self.cfg.hard_bc_radius) > 0.0:
                cx, cy = self.cfg.hard_bc_center
                dx = coords[:, 0] - tf.cast(cx, coords.dtype)
                dy = coords[:, 1] - tf.cast(cy, coords.dtype)
                r2 = dx * dx + dy * dy
                mask = tf.cast(
                    r2 > tf.cast(self.cfg.hard_bc_radius, coords.dtype) ** 2,
                    u_out.dtype,
                )
                dof_mask = tf.convert_to_tensor(self.cfg.hard_bc_dims, dtype=u_out.dtype)
                u_out = u_out * mask[:, None] * dof_mask

            stress_feat = stress_hfeat if stress_hfeat is not None else hfeat
            if return_stress or return_stress_features:
                stress_feat = self._blend_contact_stress_features(stress_feat, local_stress_feat, semantic_feat)
                stress_feat = self._fuse_stress_semantics(stress_feat, semantic_feat)
            sigma_out = None
            if return_stress:
                sigma_out = self.predict_stress_from_features(stress_feat)

            log_var = None
            if return_uncertainty:
                if self.uncertainty_out is None:
                    raise ValueError("uncertainty head disabled (uncertainty_out_dim<=0)")
                if (
                    self.uncertainty_out_mlp is not None
                    and (hfeat.shape.rank is None or hfeat.shape[-1] != self.cfg.graph_width)
                ):
                    log_var = self.uncertainty_out_mlp(hfeat)
                else:
                    log_var = self.uncertainty_out(hfeat)

            if return_stress_features and return_uncertainty:
                return u_out, stress_feat, log_var
            if return_stress_features:
                return u_out, stress_feat
            if return_stress and return_uncertainty:
                return u_out, sigma_out, log_var
            if return_stress:
                return u_out, sigma_out
            if return_uncertainty:
                return u_out, log_var
            return u_out

        def _apply_output_adaptive(
            u_shallow: tf.Tensor,
            u_deep: tf.Tensor,
            coords: tf.Tensor,
            hfeat_shallow: tf.Tensor,
            hfeat_deep: tf.Tensor,
            *,
            use_mlp_head: bool,
            stress_hfeat: Optional[tf.Tensor] = None,
            semantic_feat: Optional[tf.Tensor] = None,
            local_stress_feat: Optional[tf.Tensor] = None,
        ):
            alpha = self._sample_route_alpha(z, u_deep.dtype)
            one = tf.cast(1.0, u_deep.dtype)
            u_out = (one - alpha) * u_shallow + alpha * u_deep
            scale = tf.cast(self.output_scale, u_out.dtype)
            u_out = u_out * scale

            if self.cfg.hard_bc_radius is not None and float(self.cfg.hard_bc_radius) > 0.0:
                cx, cy = self.cfg.hard_bc_center
                dx = coords[:, 0] - tf.cast(cx, coords.dtype)
                dy = coords[:, 1] - tf.cast(cy, coords.dtype)
                r2 = dx * dx + dy * dy
                mask = tf.cast(
                    r2 > tf.cast(self.cfg.hard_bc_radius, coords.dtype) ** 2,
                    u_out.dtype,
                )
                dof_mask = tf.convert_to_tensor(self.cfg.hard_bc_dims, dtype=u_out.dtype)
                u_out = u_out * mask[:, None] * dof_mask

            sigma_out = None
            stress_feat = stress_hfeat
            if return_stress or return_stress_features:
                if stress_hfeat is not None:
                    stress_feat = self._blend_contact_stress_features(stress_hfeat, local_stress_feat, semantic_feat)
                    stress_feat = self._fuse_stress_semantics(stress_feat, semantic_feat)
                    if return_stress:
                        sigma_out = self.predict_stress_from_features(stress_feat)
                else:
                    if use_mlp_head:
                        shallow_head = self.stress_out_mlp_shallow or self.stress_out_mlp
                        deep_head = self.stress_out_mlp_deep or self.stress_out_mlp
                    else:
                        shallow_head = self.stress_out_shallow or self.stress_out
                        deep_head = self.stress_out_deep or self.stress_out
                    stress_feat = (one - alpha) * hfeat_shallow + alpha * hfeat_deep
                    stress_feat = self._blend_contact_stress_features(stress_feat, local_stress_feat, semantic_feat)
                    stress_feat = self._fuse_stress_semantics(stress_feat, semantic_feat)
                    if return_stress:
                        sigma_out = self.predict_stress_from_features(stress_feat)

            log_var = None
            if return_uncertainty:
                if self.uncertainty_out is None:
                    raise ValueError("uncertainty head disabled (uncertainty_out_dim<=0)")
                if use_mlp_head:
                    shallow_head = self.uncertainty_out_mlp_shallow or self.uncertainty_out_mlp
                    deep_head = self.uncertainty_out_mlp_deep or self.uncertainty_out_mlp
                else:
                    shallow_head = self.uncertainty_out_shallow or self.uncertainty_out
                    deep_head = self.uncertainty_out_deep or self.uncertainty_out
                log_var_shallow = shallow_head(hfeat_shallow)
                log_var_deep = deep_head(hfeat_deep)
                log_var = (one - alpha) * log_var_shallow + alpha * log_var_deep

            if return_stress_features and return_uncertainty:
                return u_out, stress_feat, log_var
            if return_stress_features:
                return u_out, stress_feat
            if return_stress and return_uncertainty:
                return u_out, sigma_out, log_var
            if return_stress:
                return u_out, sigma_out
            if return_uncertainty:
                return u_out, log_var
            return u_out

        def _run_stress_branch_mlp(shared_feat: tf.Tensor) -> tf.Tensor:
            stress_feat = shared_feat
            for layer in self.stress_branch_mlp_layers:
                stress_feat = self.mlp_act(layer(stress_feat))
            return stress_feat

        def _run_stress_branch_graph(
            shared_feat: tf.Tensor,
            coords: tf.Tensor,
            knn_idx: tf.Tensor,
            adj: tf.sparse.SparseTensor | None,
        ) -> tf.Tensor:
            stress_feat = shared_feat
            for layer in self.stress_branch_graph_layers:
                stress_feat = layer(stress_feat, coords, knn_idx, adj=adj, training=training)
            if self.stress_branch_graph_norm is not None:
                stress_feat = self.stress_branch_graph_norm(stress_feat)
            return stress_feat

        def mlp_forward():
            hcur = h
            stress_split = None
            if return_stress and self.stress_branch_early_split and self.stress_branch_mlp_layers:
                stress_split = self.stress_branch_mlp_split_index
            stress_source = hcur if stress_split == 0 else None
            if not self.adaptive_depth_enabled:
                for li, layer in enumerate(self.mlp_layers, start=1):
                    hcur = self.mlp_act(layer(hcur))
                    if stress_split is not None and li == stress_split:
                        stress_source = hcur
                if stress_split is not None and stress_source is None:
                    stress_source = hcur
                u_out = self.mlp_out(hcur)
                stress_hfeat = None
                if stress_split is not None and stress_source is not None:
                    stress_hfeat = _run_stress_branch_mlp(stress_source)
                return _apply_output(u_out, x, hcur, stress_hfeat=stress_hfeat, semantic_feat=semantic_feat)

            shallow_depth = min(
                self.adaptive_depth_shallow_layers,
                max(1, len(self.mlp_layers)),
            )
            h_shallow = None
            for li, layer in enumerate(self.mlp_layers, start=1):
                hcur = self.mlp_act(layer(hcur))
                if stress_split is not None and li == stress_split:
                    stress_source = hcur
                if li == shallow_depth:
                    h_shallow = hcur
            if stress_split is not None and stress_source is None:
                stress_source = hcur
            if h_shallow is None:
                h_shallow = hcur
            h_deep = hcur

            shallow_head = self.mlp_out_shallow or self.mlp_out
            deep_head = self.mlp_out_deep or self.mlp_out
            u_shallow = shallow_head(h_shallow)
            u_deep = deep_head(h_deep)
            stress_hfeat = None
            if stress_split is not None and stress_source is not None:
                stress_hfeat = _run_stress_branch_mlp(stress_source)
            return _apply_output_adaptive(
                u_shallow,
                u_deep,
                x,
                h_shallow,
                h_deep,
                use_mlp_head=True,
                stress_hfeat=stress_hfeat,
                semantic_feat=semantic_feat,
            )

        def graph_forward():
            coords = x
            n_nodes = tf.shape(coords)[0]

            def _build_dynamic():
                knn_dyn = _build_knn_graph(coords, self.cfg.graph_k, self.cfg.graph_knn_chunk)
                adj_dyn = _knn_to_adj(knn_dyn, n_nodes)
                return tf.cast(knn_dyn, tf.int32), adj_dyn

            if self._global_knn_idx is None:
                knn_idx, adj = _build_dynamic()
            else:
                if self._global_knn_n is not None:
                    cached_n = tf.cast(self._global_knn_n, n_nodes.dtype)
                else:
                    cached_n = tf.cast(tf.shape(self._global_knn_idx)[0], n_nodes.dtype)

                use_cached = tf.equal(n_nodes, cached_n)

                def _use_cache():
                    knn_cached = tf.cast(self._global_knn_idx, tf.int32)
                    if self._global_adj is not None:
                        return knn_cached, self._global_adj
                    return knn_cached, _knn_to_adj(knn_cached, n_nodes)

                knn_idx, adj = tf.cond(use_cached, _use_cache, _build_dynamic)

            hcur = self.graph_proj(h)
            local_stress_feat = None
            if (return_stress or return_stress_features) and self.contact_stress_hybrid_enabled:
                local_stress_feat = hcur
            stress_split = None
            if return_stress and self.stress_branch_early_split and self.stress_branch_graph_layers:
                stress_split = self.stress_branch_graph_split_index
            stress_source = hcur if stress_split == 0 else None
            film_gamma = self.film_gamma if self.use_film else None
            film_beta = self.film_beta if self.use_film else None
            shallow_depth = min(
                self.adaptive_depth_shallow_layers,
                max(1, len(self.graph_layers)),
            )
            h_shallow = None
            for li, layer in enumerate(self.graph_layers, start=1):
                hcur = layer(hcur, coords, knn_idx, adj=adj, training=training)
                if film_gamma is not None and film_beta is not None:
                    gamma = film_gamma[li - 1](zb)
                    beta = film_beta[li - 1](zb)
                    gamma = tf.cast(gamma, hcur.dtype)
                    beta = tf.cast(beta, hcur.dtype)
                    hcur = gamma * hcur + beta
                if stress_split is not None and li == stress_split:
                    stress_source = hcur
                if li == shallow_depth:
                    h_shallow = hcur

            if stress_split is not None and stress_source is None:
                stress_source = hcur
            if h_shallow is None:
                h_shallow = hcur

            if not self.adaptive_depth_enabled:
                hcur = self.graph_norm(hcur)
                u_out = self.graph_out(hcur)
                stress_hfeat = None
                if stress_split is not None and stress_source is not None:
                    contact_mask = self._contact_mask_from_semantics(semantic_feat, dtype=hcur.dtype)
                    if (
                        contact_mask is not None
                        and tf.executing_eagerly()
                        and bool(tf.reduce_all(contact_mask > 0.5).numpy())
                    ):
                        stress_hfeat = local_stress_feat
                        local_stress_feat = None
                    else:
                        stress_hfeat = _run_stress_branch_graph(stress_source, coords, knn_idx, adj)
                return _apply_output(
                    u_out,
                    coords,
                    hcur,
                    stress_hfeat=stress_hfeat,
                    semantic_feat=semantic_feat,
                    local_stress_feat=local_stress_feat,
                )

            h_shallow_norm = self.graph_norm(h_shallow)
            h_deep_norm = self.graph_norm(hcur)
            shallow_head = self.graph_out_shallow or self.graph_out
            deep_head = self.graph_out_deep or self.graph_out
            u_shallow = shallow_head(h_shallow_norm)
            u_deep = deep_head(h_deep_norm)
            stress_hfeat = None
            if stress_split is not None and stress_source is not None:
                contact_mask = self._contact_mask_from_semantics(semantic_feat, dtype=hcur.dtype)
                if (
                    contact_mask is not None
                    and tf.executing_eagerly()
                    and bool(tf.reduce_all(contact_mask > 0.5).numpy())
                ):
                    stress_hfeat = local_stress_feat
                    local_stress_feat = None
                else:
                    stress_hfeat = _run_stress_branch_graph(stress_source, coords, knn_idx, adj)
            return _apply_output_adaptive(
                u_shallow,
                u_deep,
                coords,
                h_shallow_norm,
                h_deep_norm,
                use_mlp_head=False,
                stress_hfeat=stress_hfeat,
                semantic_feat=semantic_feat,
                local_stress_feat=local_stress_feat,
            )
        # --- Decide graph vs MLP ---
        if force_pointwise:
            return mlp_forward()
        if not self.use_graph:
            return mlp_forward()
        if self._global_knn_idx is None:
            # No cached adjacency available: fall back to dynamic graph build.
            return graph_forward()
        if self._global_knn_n is not None:
            cached_n = tf.cast(self._global_knn_n, N.dtype)
        else:
            cached_n = tf.cast(tf.shape(self._global_knn_idx)[0], N.dtype)
        use_graph = tf.equal(N, cached_n)
        return tf.cond(use_graph, graph_forward, mlp_forward)

    def set_global_graph(self, coords: tf.Tensor):
        """Precompute and cache global kNN adjacency for full-mesh forward passes."""

        coords = tf.convert_to_tensor(coords, dtype=tf.float32)
        k = self.cfg.graph_k
        self._global_knn_idx = _build_knn_graph(coords, k, self.cfg.graph_knn_chunk)
        self._global_knn_n = int(coords.shape[0]) if coords.shape.rank else None
        
        # Precompute sparse adj
        self._global_adj = _knn_to_adj(self._global_knn_idx, self._global_knn_n)

    def set_contact_residual_hint(self, value: float | tf.Tensor):
        """Update sample-level routing hint from contact residual statistics."""

        v = tf.cast(tf.convert_to_tensor(value), tf.float32)
        if v.shape.rank != 0:
            v = tf.reshape(v, ())
        v = tf.where(tf.math.is_finite(v), v, tf.zeros_like(v))
        v = tf.maximum(v, tf.constant(0.0, dtype=tf.float32))
        self._contact_residual_hint.assign(v)

    def _sample_route_alpha(self, z: tf.Tensor, dtype: tf.dtypes.DType) -> tf.Tensor:
        """
        Return deep-path weight alpha in [0,1] for sample-level routing.
        alpha=0 -> shallow head, alpha=1 -> deep head.
        """
        if self.adaptive_depth_route_source == "contact_residual":
            score = tf.cast(self._contact_residual_hint, tf.float32)
        else:
            z_sample = tf.reduce_mean(tf.cast(z, tf.float32), axis=0, keepdims=True)
            score = tf.sqrt(tf.reduce_mean(tf.square(z_sample)))
        threshold = tf.cast(self.adaptive_depth_threshold, tf.float32)
        temperature = tf.cast(self.adaptive_depth_temperature, tf.float32)
        if self.adaptive_depth_mode == "soft":
            alpha = tf.math.sigmoid((score - threshold) / tf.maximum(temperature, 1.0e-6))
        else:
            alpha = tf.cast(score >= threshold, tf.float32)
        return tf.cast(tf.reshape(alpha, (1, 1)), dtype)


# -----------------------------
# Wrapper model with unified u_fn
# -----------------------------

class DisplacementModel:
    """
    High-level wrapper that holds:
      - ParamEncoder (P_hat -> z)
      - DisplacementNet ([x_feat, z] -> u)

    Provides:
      - u_fn(X, params): unified forward callable for energy modules.
    """
    def __init__(self, cfg: ModelConfig):
        _maybe_mixed_precision(cfg.mixed_precision)
        self.cfg = cfg
        self.encoder = ParamEncoder(cfg.encoder)
        # Ensure field.cond_dim == encoder.out_dim
        if cfg.field.cond_dim != cfg.encoder.out_dim:
            print(f"[pinn_model] Adjust cond_dim from {cfg.field.cond_dim} -> {cfg.encoder.out_dim}")
            cfg.field.cond_dim = cfg.encoder.out_dim
        self.field = DisplacementNet(cfg.field)
        # Alias stress head for backward compatibility with previously traced graphs
        # that referenced `self.stress_out` directly.
        self.stress_out = self.field.stress_out
        self.uncertainty_out = self.field.uncertainty_out

    def _normalize_inputs(self, X: tf.Tensor, params: Optional[Dict]) -> Tuple[tf.Tensor, tf.Tensor]:
        """Validate/convert inputs and ensure stable shapes for tf.function trace reuse."""
        if params is None:
            raise ValueError("params must contain 'P_hat' or 'P'.")

        if "P_hat" in params:
            P_hat = params["P_hat"]
        elif "P" in params:
            # normalize: (P - shift)/scale
            shift = tf.cast(self.cfg.preload_shift, tf.float32)
            scale = tf.cast(self.cfg.preload_scale, tf.float32)
            P_hat = (tf.convert_to_tensor(params["P"], dtype=tf.float32) - shift) / scale
        else:
            raise ValueError("params must have 'P_hat' or 'P'.")

        P_hat = tf.convert_to_tensor(P_hat, dtype=tf.float32)
        if P_hat.shape.rank == 1:
            P_hat = tf.expand_dims(P_hat, axis=0)

        # P_hat may include staged metadata (mask/last/rank), so keep trailing
        # dimension flexible but enforce rank-2.
        tf.debugging.assert_rank(P_hat, 2, message="P_hat must be rank-2 after normalization.")
        P_hat.set_shape((None, None))

        X = tf.convert_to_tensor(X, dtype=tf.float32)
        if X.shape.rank == 1:
            X = tf.expand_dims(X, axis=0)
        # Do not use tf.ensure_shape here: under tf.function+ForwardAccumulator it
        # can trigger trace_type errors in some TF versions.
        tf.debugging.assert_rank(X, 2, message="X must be rank-2 with shape (N,3).")
        tf.debugging.assert_equal(tf.shape(X)[-1], 3, message="X last dimension must be 3.")
        X.set_shape((None, 3))

        return X, P_hat

    @staticmethod
    def _extract_contact_surface_frame(params: Optional[Dict]):
        if not isinstance(params, dict):
            return None
        normals = params.get(CONTACT_SURFACE_NORMALS_KEY)
        t1 = params.get(CONTACT_SURFACE_T1_KEY)
        t2 = params.get(CONTACT_SURFACE_T2_KEY)
        if normals is None or t1 is None or t2 is None:
            return None
        return normals, t1, t2

    @contextmanager
    def _contact_surface_stress_context(self, params: Optional[Dict]):
        frame = self._extract_contact_surface_frame(params)
        if frame is None:
            yield
            return

        previous = self.field._contact_surface_semantic_features
        self.field.set_contact_surface_frame(*frame)
        try:
            yield
        finally:
            if previous is None:
                self.field.clear_contact_surface_semantic_features()
            else:
                self.field.set_contact_surface_semantic_features(previous)

    def _us_fn_runtime(
        self,
        X: tf.Tensor,
        P_hat: tf.Tensor,
        *,
        params: Optional[Dict] = None,
        force_pointwise: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.field.stress_out is None:
            raise ValueError("stress head disabled (stress_out_dim<=0)")

        contact_surface_active = self._extract_contact_surface_frame(params) is not None
        strict_mixed_default_eps_bridge = bool(
            getattr(self.field.cfg, "strict_mixed_default_eps_bridge", False)
        )
        strict_mixed_contact_pointwise_stress = bool(
            getattr(self.field.cfg, "strict_mixed_contact_pointwise_stress", False)
        )
        force_pointwise = bool(
            force_pointwise
            or (contact_surface_active and strict_mixed_contact_pointwise_stress)
        )
        use_eps_bridge = bool(
            self.field.use_eps_guided_stress_head
            or (contact_surface_active and strict_mixed_default_eps_bridge)
        )

        with self._contact_surface_stress_context(params):
            if use_eps_bridge:
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(X)
                    z = self.encoder(P_hat)
                    u, stress_feat = self.field(
                        X,
                        z,
                        return_stress_features=True,
                        force_pointwise=force_pointwise,
                    )
                    u = tf.cast(u, tf.float32)
                eps_bridge = _engineering_strain_from_tape(tape, X, u)
                del tape
                sigma = self.field.predict_stress_from_features(stress_feat, eps_bridge=eps_bridge)
                return tf.cast(u, tf.float32), tf.cast(sigma, tf.float32)

            z = self.encoder(P_hat)
            u, sigma = self.field(X, z, return_stress=True, force_pointwise=force_pointwise)
            return tf.cast(u, tf.float32), tf.cast(sigma, tf.float32)

    @tf.function(
        jit_compile=False,
        reduce_retracing=True,
        input_signature=(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32, name="X"),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32, name="P_hat"),
        ),
    )
    def _u_fn_compiled(self, X: tf.Tensor, P_hat: tf.Tensor) -> tf.Tensor:
        z = self.encoder(P_hat)          # (B, cond_dim)
        u = self.field(X, z)             # (N,3)
        # Physics operators鍜岃兘閲忕畻瀛愰兘鍋囧畾杈撳叆涓?float32锛涜嫢鍚敤娣峰悎绮惧害锛?
        # 缃戠粶鍐呴儴浼氬湪 float16/bfloat16 涓嬭绠楋紝姝ゅ缁熶竴 cast 鍥?float32锛?
        # 浠ラ伩鍏嶅 tie/boundary 绾︽潫涓嚭鐜?"expected float but got half" 鐨勬姤閿欍€?
        return tf.cast(u, tf.float32)

    @tf.function(
        jit_compile=False,
        reduce_retracing=True,
        input_signature=(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32, name="X"),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32, name="P_hat"),
        ),
    )
    def _u_fn_pointwise_compiled(self, X: tf.Tensor, P_hat: tf.Tensor) -> tf.Tensor:
        z = self.encoder(P_hat)
        u = self.field(X, z, force_pointwise=True)
        return tf.cast(u, tf.float32)

    def u_fn(self, X: tf.Tensor, params: Optional[Dict] = None) -> tf.Tensor:
        """
        Unified forward:
            X: (N,3) float tensor (coordinates; normalized outside if閲囩敤褰掍竴鍖?
            params: dict with either
                - 'P_hat': (3,) or (N,3) normalized preload
                - or 'P': (3,) real preload in N + cfg.preload_shift/scale provided
        """
        X, P_hat = self._normalize_inputs(X, params)
        return self._u_fn_compiled(X, P_hat)

    def u_fn_pointwise(self, X: tf.Tensor, params: Optional[Dict] = None) -> tf.Tensor:
        """Forward that always uses the pointwise MLP path."""

        X, P_hat = self._normalize_inputs(X, params)
        return self._u_fn_pointwise_compiled(X, P_hat)

    @tf.function(
        jit_compile=False,
        reduce_retracing=True,
        input_signature=(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32, name="X"),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32, name="P_hat"),
        ),
    )
    def _us_fn_compiled(self, X: tf.Tensor, P_hat: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.field.stress_out is None:
            raise ValueError("stress head disabled (stress_out_dim<=0)")
        if self.field.use_eps_guided_stress_head:
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(X)
                z = self.encoder(P_hat)
                u, stress_feat = self.field(X, z, return_stress_features=True)
                u = tf.cast(u, tf.float32)
            eps_bridge = _engineering_strain_from_tape(tape, X, u)
            del tape
            sigma = self.field.predict_stress_from_features(stress_feat, eps_bridge=eps_bridge)
            return tf.cast(u, tf.float32), tf.cast(sigma, tf.float32)

        z = self.encoder(P_hat)          # (B, cond_dim)
        u, sigma = self.field(X, z, return_stress=True)
        return tf.cast(u, tf.float32), tf.cast(sigma, tf.float32)

    @tf.function(
        jit_compile=False,
        reduce_retracing=True,
        input_signature=(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32, name="X"),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32, name="P_hat"),
        ),
    )
    def _us_fn_pointwise_compiled(self, X: tf.Tensor, P_hat: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.field.stress_out is None:
            raise ValueError("stress head disabled (stress_out_dim<=0)")
        if self.field.use_eps_guided_stress_head:
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(X)
                z = self.encoder(P_hat)
                u, stress_feat = self.field(
                    X,
                    z,
                    return_stress_features=True,
                    force_pointwise=True,
                )
                u = tf.cast(u, tf.float32)
            eps_bridge = _engineering_strain_from_tape(tape, X, u)
            del tape
            sigma = self.field.predict_stress_from_features(stress_feat, eps_bridge=eps_bridge)
            return tf.cast(u, tf.float32), tf.cast(sigma, tf.float32)

        z = self.encoder(P_hat)
        u, sigma = self.field(X, z, return_stress=True, force_pointwise=True)
        return tf.cast(u, tf.float32), tf.cast(sigma, tf.float32)

    def us_fn(self, X: tf.Tensor, params: Optional[Dict] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        甯﹀簲鍔涘ご鐨勫墠鍚戯細杩斿洖浣嶇Щ u 鍜岄娴嬬殑搴斿姏鍒嗛噺 sigma銆?
        sigma 鐨勭淮搴︾敱 cfg.field.stress_out_dim 鍐冲畾锛堥粯璁?6锛夈€?
        """
        X, P_hat = self._normalize_inputs(X, params)
        if self._extract_contact_surface_frame(params) is not None:
            return self._us_fn_runtime(X, P_hat, params=params, force_pointwise=False)
        return self._us_fn_compiled(X, P_hat)

    def sigma_fn(self, X: tf.Tensor, params: Optional[Dict] = None) -> tf.Tensor:
        """Stress-only forward in canonical Voigt layout [...,6]."""

        _, sigma = self.us_fn(X, params)
        return tf.cast(sigma, tf.float32)

    def forward_mixed(
        self,
        X: tf.Tensor,
        params: Optional[Dict] = None,
        cache: Optional[MixedForwardCache] = None,
    ) -> MixedFieldBatch:
        """Return mixed outputs with optional single-forward cache reuse."""

        cache_obj = cache if cache is not None else MixedForwardCache()
        cache_key = (id(X), id(params))
        if cache_obj.key == cache_key and cache_obj.batch is not None:
            return cache_obj.batch

        u, sigma = self.us_fn(X, params)
        batch = MixedFieldBatch(
            u=tf.cast(u, tf.float32),
            sigma_vec=tf.cast(sigma, tf.float32),
            cache_key=cache_key,
        )
        cache_obj.key = cache_key
        cache_obj.batch = batch
        return batch

    def us_fn_pointwise(self, X: tf.Tensor, params: Optional[Dict] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Stress forward that always uses the pointwise MLP path."""

        X, P_hat = self._normalize_inputs(X, params)
        if self._extract_contact_surface_frame(params) is not None:
            return self._us_fn_runtime(X, P_hat, params=params, force_pointwise=True)
        return self._us_fn_pointwise_compiled(X, P_hat)

    @tf.function(
        jit_compile=False,
        reduce_retracing=True,
        input_signature=(
            tf.TensorSpec(shape=(None, 3), dtype=tf.float32, name="X"),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32, name="P_hat"),
        ),
    )
    def _uvar_fn_compiled(self, X: tf.Tensor, P_hat: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        if self.field.uncertainty_out is None:
            raise ValueError("uncertainty head disabled (uncertainty_out_dim<=0)")
        z = self.encoder(P_hat)
        u, log_var = self.field(X, z, return_uncertainty=True)
        return tf.cast(u, tf.float32), tf.cast(log_var, tf.float32)

    def uvar_fn(self, X: tf.Tensor, params: Optional[Dict] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward with uncertainty head: returns (u, log_var)."""

        X, P_hat = self._normalize_inputs(X, params)
        return self._uvar_fn_compiled(X, P_hat)


def create_displacement_model(cfg: Optional[ModelConfig] = None) -> DisplacementModel:
    """Factory function to create the high-level displacement model."""
    return DisplacementModel(cfg or ModelConfig())


# -----------------------------
# Minimal smoke test
# -----------------------------
if __name__ == "__main__":
    cfg = ModelConfig(
        encoder=EncoderConfig(in_dim=3, width=64, depth=2, act="silu", out_dim=64),
        field=FieldConfig(
            in_dim_coord=3,
            fourier=FourierConfig(num=8, sigma=3.0),
            cond_dim=64,
            width=256, depth=7, act="silu", residual_skips=(3,6),
            out_dim=3
        ),
        mixed_precision=None,
        preload_shift=200.0, preload_scale=800.0
    )

    model = create_displacement_model(cfg)

    # Fake inputs
    N = 1024
    X = tf.random.uniform((N, 3), minval=-1.0, maxval=1.0)     # assume normalized coords
    P = tf.constant([500.0, 800.0, 300.0], dtype=tf.float32)   # N
    out = model.u_fn(X, {"P": P})
    print("u shape:", out.shape)  # expect (N,3)
