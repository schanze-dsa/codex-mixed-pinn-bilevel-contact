# Tail Alpha Design

日期：2026-03-19

## 问题

`tangential residual-driven step` 在真实 `normal_ready` batch 上进入尾段后：

- `ft_residual_after == ft_residual_before`
- `effective_alpha_scale == 0.0`
- `tangential_backtrack_steps` 已打满

说明问题不再是收敛判据，也不再是总预算，而是尾段找不到可接受的更小步长。

## 设计

保持现有的 residual-driven proposal：

`lambda_t_next = Pi(lambda_t - alpha * F_t(lambda_t), mu * lambda_n)`

在常规 alpha 序列失败后，继续追加更小的 tail alpha 搜索：

- 常规：`1.0, 0.5, 0.25, 0.125, 0.0625`
- tail：`0.03125, 0.015625, 0.0078125`

## 目标

- 不改 normal block
- 不改 residual 定义
- 不再继续加 `max_inner_iters`
- 只验证“更小 alpha 是否能在尾段重新找到 accepted step”

## 预期

- synthetic 单点 case 上能从 `alpha=0.0` 变成接收更小 tail alpha
- 真实 batch 上若仍然 `alpha=0.0`，则说明方案 A 方向对但力度仍不足，下一刀应进入更强的 tail strategy 或更局部 Newton / quasi-Newton
