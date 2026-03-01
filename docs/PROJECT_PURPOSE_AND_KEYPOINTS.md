# 项目目的与关键点（PINN + 摩擦接触 + 镜面变形）

## 1. 这个代码的目的
- 输入工程结构（`CDB/INP` 网格）与螺栓拧紧路径参数。
- 在 PINN/DFEM 框架下求解结构变形与接触响应。
- 输出镜面区域的全场变形云图（可扩展到应力/应变/不确定度）。
- 支持少量真值工况下的物理约束训练与校准。

## 2. 主流程（从输入到输出）
1. `main.py` 读取 `config.yaml`，构建 `TrainerConfig`。  
2. `trainer.build()` 解析 CDB，构建体积分点、接触点、拧紧约束、模型与优化器。  
3. `trainer.run()` 组装 `TotalEnergy`，循环训练，周期更新接触/边界 ALM 乘子。  
4. 训练后使用 `viz/mirror_viz.py` 导出镜面变形图与文本数据。  

## 3. 关键模块
- `src/model/pinn_model.py`  
  - 位移网络主干（DFEM/GCN/MLP）。
  - 已加入：有限域谱编码、工程语义输入、不确定度头 `uvar_fn`。
- `src/model/loss_energy.py`  
  - 总损失装配（弹性、接触、边界、拧紧、应力/平衡等）。
  - 已加入：增量能量-耗散罚项 `E_ed`、摩擦双势项 `E_bi`、不确定度项 `E_unc` 通道。
- `src/physics/contact/contact_normal_alm.py`  
  - 法向接触 ALM（含 FB/投影残差）。
- `src/physics/contact/contact_friction_alm.py`  
  - 切向摩擦 ALM（严格/平滑摩擦路径）。
  - 已加入：双势一致残差（bipotential-inspired）项。
- `src/train/trainer.py`  
  - 训练循环、权重调度、日志、导出。
  - 已加入：CDB 语义特征构建与挂载、不确定度代理损失计算。

## 4. 当前创新点映射（与你的论文设定一致）
1. 增量能量-耗散一致性（Innovation A）  
   - 位置：`loss_energy.py`（`compute_incremental_ed_penalty` + staged loss `E_ed`）。
2. 摩擦公式创新（双势一致残差）  
   - 位置：`contact_friction_alm.py`（`use_bipotential_residual`、`E_bi`、`R_bi_comp`）。
3. 几何泛化：有限域谱编码  
   - 位置：`pinn_model.py`（`FiniteSpectralFeatures`）。
4. 几何泛化：CDB 工程语义编码  
   - 位置：`trainer.py`（`build_node_semantic_features`）+ `pinn_model.py`（语义拼接）。
5. 可信度：物理残差驱动不确定性  
   - 位置：`pinn_model.py`（`uvar_fn`）+ `trainer.py`（`E_unc` 代理）+ `train/uncertainty_calibration.py`（后校准）。

## 5. 配置入口（建议）
- `loss_config.base_weights`：`w_bi`, `w_ed`, `w_unc`。
- `loss_config.energy_dissipation`：`enabled`, `external_scale`, `margin`, `use_relu`, `squared`。
- `friction_config`：`use_bipotential_residual`, `bipotential_weight`, `bipotential_eps`。
- `network_config`：`use_finite_spectral`, `finite_spectral_modes`, `use_engineering_semantics`, `semantic_feat_dim`, `uncertainty_out_dim`。
- `uncertainty_config`：`loss_weight`, `sample_points`, `proxy_scale`, `logvar_min`, `logvar_max`。

## 6. 当前验证状态
- 新增测试：
  - `test_model_innovation_hooks.py`
  - `test_innovation_physics_losses.py`
- 均已通过（本地执行通过）。

