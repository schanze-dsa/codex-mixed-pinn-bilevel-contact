# Tail Alpha Implementation

日期：2026-03-19

## 实现

在 `src/physics/contact/contact_inner_solver.py` 中：

- 将常规 residual-driven alpha 序列扩展为：
  - `1.0, 0.5, 0.25, 0.125, 0.0625`
- 当常规序列未找到 accepted step 时，继续进入 tail 搜索：
  - `0.03125, 0.015625, 0.0078125`
- 若进入 tail 搜索，`tangential_step_mode` 标记为 `residual_driven_tail`

## 测试

新增单测：

- `test_residual_driven_tail_alpha_search_finds_smaller_effective_step`

它验证：

- 常规 alpha 全失败
- `0.03125` 能带来真实 residual 下降
- solver 会接受该 tail alpha，而不是停在 `effective_alpha_scale = 0.0`

## 结果

- focused 回归通过
- synthetic case 上 tail alpha 可工作
- 真实 batch tail trace 仍未找到 accepted tail step，说明这一刀还没有解决真实尾段停滞
