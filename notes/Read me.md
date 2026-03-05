# 项目任务与代码架构详解（基于当前代码快照）

## 1. 文档目标与范围
- 目标：基于你当前仓库代码，完整说明“这个项目在做什么、怎么做、每个模块负责什么、训练流程如何执行、关键配置如何生效”。
- 范围：`main.py`、`config.yaml`、`src/` 全部模块、根目录测试文件、`tools/` 工具脚本。
- 快照说明：本说明按当前工作区代码组织生成；仓库存在未提交修改与大量未跟踪文件，本文件只描述代码行为，不回滚任何现有内容。

## 2. 当前项目任务定义（你这套代码在解决什么问题）

### 2.1 总任务
你这套系统是在做一个 **3D 镜面结构（mirror）在螺母拧紧载荷路径下的形变预测** 问题，采用的是：
- 几何/网格来自 ANSYS `CDB`。
- 位移场用 PINN/DFEM 神经网络建模（TensorFlow）。
- 物理约束包含：体弹性、法向接触、切向摩擦、边界条件、螺母拧紧。
- 训练目标是最小化综合物理损失，并输出镜面变形云图与 SavedModel。

### 2.2 业务目标（从代码和计划文档推断）
- 主目标：提高镜面 `Total Deformation` 预测质量。
- 工程约束：保持接触/摩擦路径可训练且稳定（ALM + staged preload + adaptive loss）。
- 创新钩子（已在代码中落地基础版）：
  - 有限域谱特征（finite spectral）。
  - CDB 工程语义特征（接触/边界/镜面/材料）。
  - 不确定性头 + 残差驱动校准接口。

## 3. 入口与执行链路

### 3.1 入口文件
- 主入口：`main.py`
- CLI：`python main.py [--export <saved_model_dir>]`

### 3.2 主流程（`main.py`）
1. `_setup_run_logs()`：将 stdout/stderr 同步写入 `train.log` / `train.err`。
2. `_load_yaml_config()`：读取 `config.yaml`。
3. `_prepare_config_with_autoguess()`：
   - 读取网格路径（当前仅支持 `.cdb`）。
   - 解析镜面/接触/材料/预紧/优化器/网络/损失等配置。
   - 构建 `TrainerConfig` 并将配置映射到 `cfg`。
4. `_run_training(cfg, asm)`：
   - 为本次训练创建独立 checkpoint 子目录。
   - 初始化 `Trainer(cfg)` 并执行 `trainer.run()`。
   - 导出 SavedModel（`--export` 或默认 `outputs/saved_model_<timestamp>`）。

### 3.3 当前配置快照（来自现有 `config.yaml`）
- 网格：`inp_path = mir111.cdb`
- 接触：`contact_mode = mortar`，`contact_mortar_gauss = 3`
- 接触点（sample 模式参数仍存在）：`n_contact_points_per_pair = 2000`
- 接触重采样：`resample_contact_every = 0`
- RAR：`contact_rar_enabled = false`，`volume_rar_enabled = false`
- 增量训练：`incremental_mode = true`
- Stage：`stage_inner_steps = 5`，`stage_alm_every = 1`，`stage_schedule_steps = [400, 400, 400, 300]`
- 优化：Adam，`learning_rate = 1e-5`，`epochs = 1500`，`log_every = 50`
- Elasticity：`use_forward_mode = true`，`n_points_per_step = 8192`
- Loss：`mode = residual`，`w_int/w_cn/w_ct/w_bc/w_tight/w_sigma/w_eq = 1.0`
- 输出：`output_config.save_path = ./results`，开启可视化与 surface mesh 导出。

## 4. 代码架构总览（分层）

### 4.1 配置与调度层
- `main.py`
- `src/train/trainer.py` 中 `TrainerConfig` + `Trainer`

### 4.2 几何与网格层
- `src/inp_io/cdb_parser.py`：ANSYS CDB 解析到 `AssemblyModel`
- `src/inp_io/inp_parser.py`：Abaqus INP 结构定义与兼容解析
- `src/mesh/surface_utils.py`：表面三角化、采样、投影
- `src/mesh/contact_pairs.py`：构建接触点/法向/切向/面积权重
- `src/mesh/volume_quadrature.py`：体积分点和体积权重

### 4.3 物理算子层
- `src/physics/elasticity_residual.py`
- `src/physics/contact/contact_normal_alm.py`
- `src/physics/contact/contact_friction_alm.py`
- `src/physics/contact/contact_operator.py`
- `src/physics/boundary_conditions.py`
- `src/physics/tightening_model.py`
- `src/model/loss_energy.py`（TotalEnergy 汇总）

### 4.4 神经网络层
- `src/model/pinn_model.py`
  - `ParamEncoder`
  - `DisplacementNet`
  - `DisplacementModel`

### 4.5 可视化与导出层
- `src/viz/mirror_viz.py`
- `Trainer.export_saved_model()`

### 4.6 工具与验证层
- 根目录 `test_*.py`
- `tools/*.py`

## 5. 核心对象与数据模型

### 5.1 AssemblyModel（`src/inp_io/inp_parser.py`）
`AssemblyModel` 是系统内几何/网格的核心容器，主要字段：
- `parts`: part 名 -> `PartMesh`
- `surfaces`: surface 名 -> `SurfaceDef`
- `contact_pairs`: 接触对定义
- `boundaries`: 边界条件原始行
- `nodes/elements/element_types`: 全局节点和单元索引
- `interactions`: 接触属性（摩擦系数等）

### 5.2 ContactMap（`src/mesh/contact_pairs.py`）
`ContactMap` 把所有接触对样本打平为训练可用数组：
- `xs/xm`: slave 采样点与 master 投影点
- `n/t1/t2`: 法向和切向基
- `w_area`: 面积权重
- `xs_node_idx/xs_bary`、`xm_node_idx/xm_bary`: 插值元数据（可从节点位移重建点位移）
- `pair_id/slave_tri_idx/master_tri_idx/dist`

### 5.3 TrainerConfig（`src/train/trainer.py`）
`TrainerConfig` 是全系统运行配置中心，覆盖：
- 输入输出路径与日志
- 模型结构配置（`model_cfg`）
- 物理配置（`elas_cfg/contact_cfg/tightening_cfg/total_cfg`）
- staged preload / incremental mode
- Adam / L-BFGS / early-exit
- RAR 开关与参数
- 可视化与 SavedModel 相关参数

## 6. 训练构建阶段（`Trainer.build()`）

`Trainer.build()` 的固定顺序（进度条也按这个顺序）：
1. **Load Mesh**：读 CDB 到 `self.asm`
2. **Volume/Materials**：
   - 用 `MaterialLibrary` 建材料枚举
   - `build_volume_points()` 生成 `X_vol/w_vol/mat_id`
3. **Elasticity**：初始化 `ElasticityResidual`
4. **Contact**：
   - 从 config 或自动识别生成接触对
   - `build_contact_map(...)` -> `ContactOperator.build_from_cat(...)`
5. **Tightening**：
   - 读取 `nuts` 生成 `NutSpec`
   - 初始化 `NutTighteningPenalty`
6. **Ties/BCs**：BC 在 `run()` 里通过 `attach_bcs_from_asm()` 挂载
7. **Model/Opt**：
   - `create_displacement_model(cfg.model_cfg)`
   - 可选预构建 kNN 图并做缓存
   - 可选附加工程语义特征
8. **Checkpoint**：初始化 tf checkpoint manager

## 7. 训练主循环（`Trainer.run()`）

### 7.1 每步固定 4 阶段
每个训练 step 都是：
1. 接触重采样（按 `resample_contact_every` / incremental 规则）
2. 前向+反传（核心 `_train_step` 或 incremental 路径）
3. ALM 乘子更新（按 `alm_update_every`）
4. 日志与 checkpoint（按 `log_every`）

### 7.2 早停机制
`_check_early_exit()` 监控：
- 非有限值连续次数（non-finite streak）
- 梯度爆炸 + `Pi` 的 EMA 持续恶化
- warmup 之后按 `early_exit_check_every` 检查

### 7.3 RAR（残差自适应重采样）
- 接触 RAR：基于 `contact.last_sample_metrics()`（gap/fric_res）缓存更新。
- 体积分点 RAR：基于 `elasticity.last_sample_metrics()`（psi）缓存更新。
- 当前配置里二者均关闭，但代码路径完整保留。

### 7.4 增量/分阶段
- `incremental_mode=true` 时使用 staged 参数构造与阶段内反传。
- 支持 `force_then_lock` 等 preload stage 模式。
- 可按 `stage_schedule_steps` 动态控制激活阶段数。

## 8. 损失函数与物理项（`src/model/loss_energy.py` + trainer）

### 8.1 总损失组合
`TotalEnergy` 维护分项：
- `E_int`: 体弹性能
- `E_cn`: 法向接触
- `E_ct`: 切向摩擦
- `E_bc`: 边界条件
- `E_tight`: 螺母拧紧
- `E_sigma`: 应力监督
- `E_eq`: 平衡残差（div sigma）
- `E_reg`: 正则
- `E_bi`: 双势函数相关项
- `E_ed`: 增量耗散一致性项
- `E_unc`: 不确定性代理项

### 8.2 组合形式
在 `TotalEnergy._combine_parts` 中按权重线性加权求和：
- `Pi = Σ (w_i * E_i)`
- trainer 中若启用 adaptive loss，则由 `combine_loss(parts, loss_state)` 组合。

### 8.3 关键创新项
- `compute_incremental_ed_penalty(...)`：增量能量耗散不一致惩罚。
- 摩擦双势项：`contact_friction_alm.py` 内 `use_bipotential_residual` 路径。
- 不确定性代理：`Trainer._compute_uncertainty_proxy_loss_tf(...)`。

## 9. 物理算子细节

### 9.1 ElasticityResidual
- 支持 reverse-mode 与 forward-mode（JVP）应变计算。
- `energy()` 返回体能积分及 `psi` 统计。
- `residual_cache()` 可返回 `sigma_phys/sigma_pred/div_sigma` 缓存。
- 支持 sample-level 子采样（配合 RAR）。

### 9.2 NormalContactALM
- gap: `g = ((xs+u_s) - (xm+u_m)) · n`
- `phi = softplus(-g; beta)`
- energy（alm 模式）：`Σ w (λ phi + 0.5 μ_n phi^2)`
- residual 支持 FB 或 projection 形式。
- `update_multipliers`: `λ <- max(0, λ + η μ_n phi)`

### 9.3 FrictionContactALM
- 切向滑移 `s_t` -> 试探应力 `tau_trial = lambda_t + k_t s_t`
- 摩擦圆锥半径 `tau_c = mu_f * p_eff`
- 支持三种路径：
  - 严格 ALM 残差能量
  - smooth friction 能量
  - smooth/strict blend
- 可选 bipotential residual 项并写入 `E_bi`。

### 9.4 ContactOperator
- 聚合 normal + friction 两个子算子。
- 对外提供统一 `energy/residual/update_multipliers/last_sample_metrics`。

### 9.5 BoundaryPenalty
- 支持 `penalty | hard | alm` 三种边界模式。
- `attach_ties_bcs.py` 从 `asm.boundaries` 解析并挂到 `TotalEnergy`。

### 9.6 NutTighteningPenalty
- 从螺母端面采样，按旋转角构造目标位移罚项。
- 支持分块前向求值（`forward_chunk`）控制显存。

## 10. 模型架构细节（`src/model/pinn_model.py`）

### 10.1 输入条件编码
- `ParamEncoder`: 将 `P_hat` 编成条件向量 `z`。
- 兼容 `params['P']` 或 `params['P_hat']`。

### 10.2 位移场主干
`DisplacementNet` 支持：
- 传统 PINN（坐标 + Fourier）
- DFEM 模式（learnable node embedding）
- Graph 分支（kNN + GraphConv）与 MLP fallback
- FiLM 条件调制

### 10.3 创新扩展钩子
- Finite spectral features
- Engineering semantic features（按节点）
- Stress head (`us_fn`) 与 uncertainty head (`uvar_fn`)
- Sample-level adaptive depth routing（`hard/soft`，`z_norm` 或 `contact_residual` 触发）

## 11. 网格与接触几何管线

### 11.1 CDB 解析
`cdb_parser.load_cdb()` 解析：
- `NBLOCK/EBLOCK/ETBLOCK/CMBLOCK`
- 接触组件命名规则识别
- `D,` 边界命令
- 自动组装 `AssemblyModel.parts/surfaces/contact_pairs`

### 11.2 表面处理
`surface_utils.py` 提供：
- surface key 容错匹配
- element-surface -> triangle surface
- 表面采样、最近点投影、三角几何属性

### 11.3 接触数据构建
`contact_pairs.py` 支持：
- `sample` 随机采样
- `mortar` 高斯点采样（deterministic）
- `two_pass`
- 重采样函数 `resample_contact_map`

### 11.4 体积分点
`volume_quadrature.py`:
- C3D4/C3D8/SOLID185 的质心+体积权重
- 输出 `X_vol/w_vol/mat_id`

## 12. 可视化输出（`src/viz/mirror_viz.py`）
- 将镜面表面三角化并投影到平面。
- 批量调用 `u_fn` 求位移，生成云图（smooth/contour/flat）。
- 支持只画表面、也可评估全装配后裁切表面显示。
- 支持导出：
  - 位移采样 txt
  - surface mesh ply
  - 对比图和额外诊断信息（blank region diagnostics）

## 13. 测试覆盖情况

### 13.1 `test_model_innovation_hooks.py`
覆盖创新钩子行为：
- finite spectral + semantic + uncertainty 输出形状
- 动态 kNN fallback
- adaptive depth 路由
- pointwise path 绕过 graph path

### 13.2 `test_innovation_physics_losses.py`
覆盖物理创新项：
- `compute_incremental_ed_penalty`
- friction bipotential stats 暴露
- uncertainty proxy sigma

### 13.3 `test_trainer_optimization_hooks.py`
覆盖 trainer 工程行为：
- step 采样/日志门控
- apply_gradients kwargs 兼容
- early-exit
- route hint 推送

### 13.4 `test_performance_sync_guards.py`
覆盖性能与配置读取守卫：
- RAR 配置读取
- elasticity metrics cache 开关

### 13.5 `test_forward_mode_normalize_inputs.py`
覆盖 forward accumulator + 输入归一化兼容性。

## 14. 工具脚本（`tools/`）
主要用途：
- `viz_saved_model.py`: 加载 SavedModel 后快速可视化
- `audit_preload.py`: preload 几何审计
- `bolt_region_metrics.py`: 螺栓区域指标
- `compare_deflections.py`: 变形结果对比
- `cdb_to_json.py`: CDB 快速导出
- `export_from_ckpt.py`: 从 checkpoint 导出模型
- `visualize_contact_tightening_plotly.py`: 接触/拧紧 3D 动画可视化

## 15. 当前代码的关键工程特征与风险点

### 15.1 强项
- 架构分层清晰：几何、物理、模型、训练、可视化解耦。
- 接触与摩擦算子可配置路径丰富（alm/smooth/blend/bipotential）。
- staged preload + incremental mode 对工程载荷路径友好。
- 测试已覆盖关键创新钩子与训练守卫行为。

### 15.2 风险/注意事项
- `trainer.py` 体量较大（4k+行），后续维护成本高。
- 大量 feature flag 组合复杂，若改动缺少回归测试易引入行为漂移。
- 配置注释存在编码问题（部分注释显示乱码），建议统一 UTF-8 清理。
- 当前工作区已有未提交改动，继续开发前建议分支隔离+最小化提交范围。

## 16. 文档结论
- 你当前代码并不是“单一 PINN 脚本”，而是一个完整的 **CDB->物理算子->增量训练->可视化->导出** 工程系统。
- 训练主链路和创新钩子已经齐备，后续重点应放在：
  - 模块拆分（尤其 `trainer.py`）
  - 配置治理（schema/默认值/注释编码）
  - 指标化验证（以镜面 full-field 误差与稳定性为主）。

---

以下附录为自动抽取的 **文件行数统计** 与 **模块符号清单**，用于“逐文件细节追踪”。
## Appendix A: File Line Counts

| Lines | File |
|---:|---|
| 4275 | `src/train/trainer.py` |
| 1445 | `src/model/pinn_model.py` |
| 1390 | `src/viz/mirror_viz.py` |
| 1134 | `tools/visualize_contact_tightening_plotly.py` |
| 940 | `src/mesh/surface_utils.py` |
| 903 | `main.py` |
| 815 | `src/model/loss_energy.py` |
| 735 | `src/physics/contact/contact_friction_alm.py` |
| 671 | `src/assembly/surfaces.py` |
| 662 | `src/mesh/contact_pairs.py` |
| 629 | `src/inp_io/inp_parser.py` |
| 557 | `src/inp_io/cdb_parser.py` |
| 528 | `src/physics/contact/contact_normal_alm.py` |
| 508 | `src/train/loss_weights.py` |
| 406 | `src/physics/contact/contact_operator.py` |
| 383 | `src/physics/elasticity_residual.py` |
| 331 | `config.yaml` |
| 281 | `src/physics/tightening_model.py` |
| 269 | `tools/cdb_to_json.py` |
| 269 | `tools/audit_preload.py` |
| 265 | `tools/viz_saved_model.py` |
| 258 | `test_model_innovation_hooks.py` |
| 249 | `src/physics/boundary_conditions.py` |
| 241 | `test_trainer_optimization_hooks.py` |
| 216 | `tools/compare_deflections.py` |
| 193 | `tools/bolt_region_metrics.py` |
| 173 | `src/physics/material_lib.py` |
| 160 | `src/mesh/volume_quadrature.py` |
| 92 | `src/train/attach_ties_bcs.py` |
| 91 | `tools/export_from_ckpt.py` |
| 90 | `test_innovation_physics_losses.py` |
| 82 | `test_performance_sync_guards.py` |
| 48 | `src/mesh/interp_utils.py` |
| 43 | `test_forward_mode_normalize_inputs.py` |
| 41 | `src/train/uncertainty_calibration.py` |
| 30 | `src/physics/elasticity_config.py` |
| 26 | `tools/visualize_contact_tightening.py` |
| 0 | `src/viz/__init__.py` |
| 0 | `src/train/__init__.py` |
| 0 | `src/physics/contact/__init__.py` |
| 0 | `src/physics/__init__.py` |
| 0 | `src/model/__init__.py` |
| 0 | `src/mesh/__init__.py` |
| 0 | `src/inp_io/__init__.py` |
| 0 | `src/__init__.py` |

## Appendix B: Symbol Inventory

### `main.py`
- class `_Tee` (L44): `__init__`(L45), `write`(L53), `flush`(L63), `__getattr__`(L67)
- def `_strip_ansi` (L71)
- def `_setup_run_logs` (L76)
- def `_default_saved_model_dir` (L103)
- def `_load_yaml_config` (L118)
- def `_auto_resolve_surface_keys` (L128)
- def `_prepare_config_with_autoguess` (L153)
- def `_run_training` (L854)
- def `main` (L886)

### `src/__init__.py`
- (no top-level class/def)

### `src/assembly/surfaces.py`
- def `_debug` (L32)
- class `ElementFaceRef` (L42): (no methods)
- class `SurfaceDef` (L49): (no methods)
- class `SurfaceResolvers` (L75): (no methods)
- def `to_points` (L93)
- def `sample_surface_by_key` (L118)
- def `_element_surface_to_points` (L162)
- def `_sample_on_polygon` (L275)
- def `_face_normal_and_area` (L299)
- def `_node_surface_to_points` (L328)
- def `_resolve_node_coords` (L353)
- def `_attach_resolvers` (L394)
- def `_as_element_face` (L492)
- def `_is_elset_tuple` (L505)
- def `_empty_Xnw` (L512)
- def `_as_float32` (L518)
- def `_normalize` (L525)
- def `_is_int_like` (L533)
- def `_pca_normals` (L541)
- def `_strip_asm_prefix` (L569)
- def `_check_surface` (L577)
- def `surface_def_to_points` (L585)

### `src/inp_io/__init__.py`
- (no top-level class/def)

### `src/inp_io/cdb_parser.py`
- def `_parse_fixed_width` (L38)
- def `_safe_int` (L51)
- def `_safe_float` (L66)
- def `_expand_range_stream` (L78)
- def `_parse_etblock` (L95)
- def `_parse_nblock` (L118)
- def `_parse_eblock` (L162)
- def `_parse_cmblock` (L220)
- def `_etype_name_from_code` (L252)
- def `_is_contact_component` (L270)
- def `_is_combined_component` (L277)
- def `load_cdb` (L286)
- def `main` (L540)

### `src/inp_io/inp_parser.py`
- class `ElementBlock` (L39): (no methods)
- class `PartMesh` (L46): (no methods)
- class `ContactPair` (L53): (no methods)
- class `TieConstraint` (L60): (no methods)
- class `BoundaryEntry` (L66): (no methods)
- class `InstanceDef` (L70): (no methods)
- class `SetDef` (L75): (no methods)
- class `InteractionProp` (L86): (no methods)
- class `AssemblyModel` (L92): `summary`(L109), `_dequote`(L128), `_aliases`(L132), `_strip_suffix_S`(L144), `expand_elset`(L148), `get_face_nodes`(L196), `finalize`(L234), `get_friction_mu`(L246)
- def `_is_comment_or_empty` (L280)
- def `_extract_param` (L284)
- def `_parse_kw_params` (L289)
- def `_collect_set_items` (L302)
- def `_normalize_surface_items` (L312)
- def `load_inp` (L328)
- def `_print_quick_summary` (L583)
- def `main` (L615)

### `src/mesh/__init__.py`
- (no top-level class/def)

### `src/mesh/contact_pairs.py`
- class `ContactPairSpec` (L41): (no methods)
- class `ContactPairData` (L49): (no methods)
- class `ContactMap` (L84): `concatenate`(L91), `__len__`(L126)
- def `_orthonormal_tangent_basis` (L134)
- def `_fetch_xyz` (L159)
- def `_triangle_gauss_rule` (L174)
- def `_mortar_points_on_surface` (L216)
- def `_compute_area_weights` (L255)
- def `_sorted_node_ids` (L279)
- def `_map_node_ids_to_idx` (L283)
- def `build_contact_pair_data` (L306)
- def `build_contact_pair_data_mortar` (L371)
- def `build_contact_map` (L437)
- def `resample_contact_map` (L563)
- def `guess_surface_key` (L599)

### `src/mesh/interp_utils.py`
- def `interp_bary_tf` (L19)

### `src/mesh/surface_utils.py`
- def `_normalize_surface_key` (L32)
- class `TriSurface` (L83): `__len__`(L98)
- def `_ordered_unique` (L154)
- def `_normalize_etype_conn` (L169)
- def `_face_map_for_type` (L187)
- def `_expand_elset_ids_fallback` (L205)
- def `_emit_tris_from_face` (L234)
- def `_plane_basis` (L254)
- def `_convex_hull_indices` (L273)
- def `_order_contact_nodes` (L300)
- def `resolve_surface_to_tris` (L316)
- def `triangulate_part_boundary` (L588)
- def `_fetch_xyz` (L695)
- def `compute_tri_geometry` (L709)
- def `sample_points_on_surface` (L740)
- def `_closest_pt_on_triangle` (L771)
- def `project_points_onto_surface` (L822)
- def `_coord_provider_for_ts` (L893)
- def `build_contact_surfaces` (L904)

### `src/mesh/volume_quadrature.py`
- def `build_volume_points` (L44)
- def `_volume_points_for_part` (L93)
- def `_centroid_weight_c3d4_block` (L126)
- def `_centroid_weight_c3d8_block` (L142)

### `src/model/__init__.py`
- (no top-level class/def)

### `src/model/pinn_model.py`
- class `FourierConfig` (L43): (no methods)
- class `EncoderConfig` (L50): (no methods)
- class `FieldConfig` (L58): (no methods)
- class `ModelConfig` (L111): (no methods)
- def `_get_activation` (L123)
- def `_maybe_mixed_precision` (L135)
- class `GaussianFourierFeatures` (L148): `__init__`(L157), `build`(L174), `call`(L185), `out_dim`(L200)
- class `FiniteSpectralFeatures` (L207): `__init__`(L210), `call`(L222), `out_dim`(L247)
- class `MLP` (L256): `__init__`(L259), `call`(L294)
- class `GraphConvLayer` (L304): `__init__`(L307), `call`(L327)
- def `_build_knn_graph` (L413)
- def `_knn_to_adj` (L513)
- class `ParamEncoder` (L548): `__init__`(L550), `call`(L561), `_normalize_dim`(L568)
- class `DisplacementNet` (L595): `__init__`(L601), `set_node_semantic_features`(L844), `prebuild_adjacency`(L857), `call`(L883), `set_global_graph`(L1202), `set_contact_residual_hint`(L1213), `_sample_route_alpha`(L1223)
- class `DisplacementModel` (L1246): `__init__`(L1255), `_normalize_inputs`(L1269), `_u_fn_compiled`(L1312), `_u_fn_pointwise_compiled`(L1328), `u_fn`(L1333), `u_fn_pointwise`(L1344), `_us_fn_compiled`(L1358), `_us_fn_pointwise_compiled`(L1373), `us_fn`(L1380), `us_fn_pointwise`(L1388), `_uvar_fn_compiled`(L1402), `uvar_fn`(L1409)
- def `create_displacement_model` (L1416)

### `src/physics/__init__.py`
- (no top-level class/def)

### `src/physics/boundary_conditions.py`
- class `BoundaryConfig` (L46): (no methods)
- class `BoundaryPenalty` (L57): `__init__`(L60), `build_from_numpy`(L94), `reset_for_new_batch`(L138), `build`(L142), `energy`(L159), `residual`(L203), `update_multipliers`(L226), `set_alpha`(L243), `multiply_weights`(L246)

### `src/physics/contact/__init__.py`
- (no top-level class/def)

### `src/physics/contact/contact_friction_alm.py`
- class `FrictionALMConfig` (L90): (no methods)
- class `FrictionContactALM` (L113): `__init__`(L130), `link_normal`(L171), `build_from_numpy`(L175), `snapshot_state`(L241), `restore_state`(L252), `last_slip`(L272), `capture_reference`(L276), `commit_reference`(L288), `reset_reference`(L294), `build_from_cat`(L298), `_absolute_slip_t`(L318), `_relative_slip_t`(L356), `_effective_normal_pressure`(L375), `energy`(L398), `residual`(L568), `update_multipliers`(L641), `set_mu_t`(L681), `set_k_t`(L684), `set_mu_f`(L687), `set_smooth_friction`(L691), `set_s0`(L695), `set_smooth_blend`(L699), `multiply_weights`(L708), `reset_for_new_batch`(L713), `reset_multipliers`(L726)

### `src/physics/contact/contact_normal_alm.py`
- def `_to_tf` (L57)
- def `softplus_neg` (L64)
- def `fb_residual` (L81)
- class `NormalALMConfig` (L87): (no methods)
- class `NormalContactALM` (L98): `__init__`(L114), `build_from_numpy`(L145), `build_from_cat`(L219), `_auto_orient_normals`(L244), `_gap`(L263), `energy`(L294), `update_multipliers`(L352), `residual`(L389), `set_beta`(L446), `set_mu_n`(L450), `multiply_weights`(L454), `_compute_effective_pressure`(L468), `effective_normal_pressure`(L475), `reset_for_new_batch`(L481), `reset_multipliers`(L495)
- def `tfp_median` (L506)
- def `_ensure_2d` (L524)

### `src/physics/contact/contact_operator.py`
- class `ContactOperatorConfig` (L45): (no methods)
- class `ContactOperator` (L63): `__init__`(L74), `_friction_active`(L101), `build_from_cat`(L118), `reset_for_new_batch`(L172), `reset_multipliers`(L181), `energy`(L193), `residual`(L234), `update_multipliers`(L260), `last_sample_metrics`(L286), `last_meta`(L309), `snapshot_stage_state`(L316), `restore_stage_state`(L322), `last_friction_slip`(L327), `set_beta`(L335), `set_mu_n`(L339), `set_mu_t`(L343), `set_k_t`(L347), `set_mu_f`(L351), `multiply_weights`(L356), `N`(L369), `built`(L373)

### `src/physics/elasticity_config.py`
- class `ElasticityConfig` (L13): (no methods)

### `src/physics/elasticity_residual.py`
- class `ElasticityResidual` (L23): `__init__`(L24), `set_sample_indices`(L85), `set_sample_metrics_cache_enabled`(L92), `last_sample_metrics`(L97), `_cache_metrics`(L100), `_select_points`(L113), `_eval_u_on_nodes`(L123), `_compute_strain`(L126), `_compute_strain_reverse_mode`(L131), `_compute_strain_forward_mode`(L167), `_sigma_from_eps`(L221), `energy`(L232), `residual_cache`(L266)

### `src/physics/material_lib.py`
- def `lame_from_E_nu` (L41)
- def `isotropic_C_6x6` (L48)
- class `MaterialSpec` (L73): (no methods)
- class `MaterialLibrary` (L79): `__init__`(L91), `tags`(L124), `num_materials`(L127), `encode_tags`(L130), `id_of`(L141), `C_table_np`(L144), `C_table_tf`(L148), `summary`(L157)

### `src/physics/tightening_model.py`
- class `NutSpec` (L28): (no methods)
- class `TighteningConfig` (L36): (no methods)
- class `NutSampleData` (L44): (no methods)
- def `_sorted_node_ids` (L54)
- def `_map_node_ids_to_idx` (L58)
- def `_compute_area_weights` (L74)
- def `_normalize_axis` (L88)
- def `_auto_axis_from_nodes` (L98)
- class `NutTighteningPenalty` (L114): `__init__`(L115), `build_from_specs`(L119), `_u_fn_chunked`(L165), `_angle_to_rad`(L179), `_rotation_displacement`(L185), `energy`(L207), `residual`(L271)

### `src/train/__init__.py`
- (no top-level class/def)

### `src/train/loss_weights.py`
- def `_to_float` (L40)
- class `LossWeightState` (L51): `from_config`(L124), `as_dict`(L199)
- def `update_loss_weights` (L208)
- def `combine_loss` (L463)

### `src/train/uncertainty_calibration.py`
- def `calibrate_sigma_by_residual` (L10)

### `src/viz/__init__.py`
- (no top-level class/def)

### `src/viz/mirror_viz.py`
- def `_coerce_params_for_forward` (L42)
- def `_eval_displacement_batched` (L80)
- def `_with_new_stem` (L106)
- def `_eval_surface_or_assembly` (L116)
- def `_refine_surface_samples` (L171)
- def `_build_vertex_adjacency` (L236)
- def `_interpolate_displacement_on_refined` (L245)
- def `_smooth_scalar_on_tri_mesh` (L274)
- class `BlankRegionDiagnostics` (L324): `summary_lines`(L339), `primary_cause`(L352)
- def `_convex_hull_area` (L368)
- def `_triangle_area_sum` (L403)
- def `_collect_boundary_loops` (L414)
- def `_loop_area` (L471)
- def `_diagnose_blank_regions` (L483)
- def `_mask_tris_with_loops` (L545)
- def `_fit_rigid_transform` (L605)
- def `_remove_rigid_body_motion` (L633)
- def `_apply_rigid_correction` (L656)
- def `_fit_plane_basis` (L671)
- def `_unique_nodes_from_tris` (L695)
- def `_export_surface_mesh` (L707)
- def `_project_to_plane` (L736)
- def `plot_mirror_deflection` (L750)
- def `plot_mirror_deflection_by_name` (L1328)

### `test_forward_mode_normalize_inputs.py`
- class `ForwardModeNormalizeInputsTests` (L21): `test_normalize_inputs_supports_forward_accumulator_in_tf_function`(L22)

### `test_innovation_physics_losses.py`
- class `InnovationPhysicsLossTests` (L25): `test_incremental_ed_penalty_positive_when_violated`(L26), `test_incremental_ed_penalty_zero_when_satisfied`(L37), `test_friction_bipotential_term_exposed_in_stats`(L48), `test_uncertainty_proxy_sigma_shape_and_positive`(L81)

### `test_model_innovation_hooks.py`
- def `_make_minimal_asm` (L27)
- class `InnovationHookTests` (L49): `test_displacement_model_supports_finite_spectral_semantic_and_uncertainty`(L50), `test_build_node_semantic_features_shape_and_value_range`(L90), `test_residual_driven_sigma_calibration_is_monotonic`(L106), `test_graph_mode_without_prebuilt_graph_uses_dynamic_knn`(L119), `test_sample_level_adaptive_depth_routes_easy_and_hard_samples`(L163), `test_pointwise_forward_bypasses_graph_path`(L222)

### `test_performance_sync_guards.py`
- class `_MatLib` (L26): (no methods)
- class `PerformanceSyncGuardTests` (L30): `test_prepare_config_reads_rar_flags_from_yaml`(L31), `test_elasticity_metrics_cache_can_be_disabled`(L57)

### `test_trainer_optimization_hooks.py`
- class `_OptWithAggregateArg` (L26): `apply_gradients`(L27)
- class `_OptNoAggregateArg` (L32): `apply_gradients`(L33)
- class `TrainerOptimizationHookTests` (L38): `test_savedmodel_module_run_disables_autograph`(L39), `test_contact_route_update_interval_gate`(L59), `test_step_scalar_collection_uses_log_and_early_exit_intervals`(L69), `test_detect_apply_gradients_kwargs_for_supported_optimizer`(L84), `test_detect_apply_gradients_kwargs_for_plain_optimizer`(L88), `test_static_weight_vector_cache_for_non_adaptive_mode`(L92), `test_format_energy_summary_is_skipped_when_step_bar_disabled`(L112), `test_volume_sampling_falls_back_to_uniform_before_rar_cache_ready`(L129), `test_early_exit_triggers_after_nonfinite_streak`(L154), `test_early_exit_triggers_on_sustained_divergence`(L178), `test_contact_residual_route_metric_and_hint_push`(L205), `test_contact_multiplier_updates_are_plain_python_methods`(L232)

### `tools/audit_preload.py`
- def `_load_yaml` (L28)
- def `_resolve_surface_key` (L36)
- def `_parse_vec` (L55)
- def `_find_latest_saved_model` (L62)
- class `BoltAudit` (L73): (no methods)
- def `_audit_preload_geometry` (L82)
- def `_print_geometry_report` (L107)
- def `_make_u_fn_from_saved_model` (L129)
- def `main` (L152)

### `tools/bolt_region_metrics.py`
- def `_load_yaml` (L31)
- def `_resolve_surface_key` (L39)
- class `DeflectionData` (L59): (no methods)
- def `_load_deflection_txt` (L67)
- class `BoltCenter` (L92): (no methods)
- def `_load_bolt_centers` (L97)
- def `_region_stats` (L136)
- def `main` (L157)

### `tools/cdb_to_json.py`
- def `_parse_fixed_width` (L18)
- def `_safe_int` (L31)
- def `_safe_float` (L46)
- def `_expand_range_stream` (L58)
- def `_parse_cdb` (L72)
- def `main` (L233)

### `tools/compare_deflections.py`
- class `DeflectionData` (L25): (no methods)
- def `_load_txt` (L34)
- def `_ensure_same_nodes` (L65)
- def `_pearson` (L75)
- def `_summarize_one` (L84)
- def `_pair_metrics` (L101)
- def `_save_scatter` (L142)
- def `_iter_txt_paths` (L157)
- def `main` (L168)

### `tools/export_from_ckpt.py`
- def `_default_export_dir` (L21)
- def `main` (L26)

### `tools/visualize_contact_tightening.py`
- (no top-level class/def)

### `tools/visualize_contact_tightening_plotly.py`
- def `_load_config` (L37)
- def `_resolve_mesh_path` (L42)
- def `_load_asm` (L56)
- def `_normalize_contact_pairs` (L63)
- def `_normalize_axis` (L90)
- def `_auto_axis_from_nodes` (L103)
- def `_rotate_points` (L115)
- def `_tri_surface_to_mesh` (L128)
- def `_lighten` (L150)
- def `_rgb` (L157)
- def `_nut_specs_from_cfg` (L162)
- def `_tighten_angles_from_cfg` (L180)
- def `_order_from_cfg` (L189)
- def `_build_frame_angles` (L203)
- def `_color_cycle` (L222)
- def `_make_part_color_map` (L237)
- def `_build_vertex_neighbors` (L250)
- def `_laplacian_step` (L260)
- def `_taubin_smooth` (L270)
- def `_bounds_from_verts` (L287)
- def `_merge_bounds` (L293)
- def `_pad_bounds` (L306)
- def `_hover_template` (L313)
- def `_clip_faces_to_cylindrical_bounds` (L339)
- def `_mesh_trace` (L366)
- def `main` (L399)

### `tools/viz_saved_model.py`
- def `_load_yaml` (L29)
- def `_find_latest_saved_model` (L37)
- def `_parse_vec` (L47)
- def `_resolve_path` (L54)
- def `_load_saved_u_fn` (L61)
- def `_default_cases` (L93)
- def `main` (L117)
