# Step-Level GRPO Advantage Migration

## 背景

当前 Stage 2 GRPO 训练在 **day-level scalar reward** 上算 advantage：

```
R_day_i = sum_{t=1}^{48} r_{i,t}
A_i     = (R_day_i - mean_i R_day_i) / (std_i R_day_i + eps)
```

然后把同一个 `A_i` 广播到 rollout i 里所有 48 个 step 的 `log π(a_{i,t} | s_{i,t})` 上。

问题：
- 一天 48 步共享同一个 advantage，credit assignment 完全糊掉。
- 同一 rollout 里好的决策和坏的决策在 loss 里贡献等同。
- 远期 reward 和近期 reward 被同等回贴，没有时间结构。

目标：
- 在不引入 value model 的前提下，给每一步独立的 credit signal。
- 保持 `1 day rollout → 1 optimizer step` 的训练节奏不变。
- 作为未来要不要上 VAPO-style critic 的 baseline 判据。

## 改动范围

**只改 advantage 计算与 loss aggregation；采样、optimizer、KL anchor、checkpoint 节奏都不动。**

### 当前等效伪代码

```python
# 每天一次
rollouts = sample_G_rollouts(day)               # G × 48
rewards  = [sum(r[i]) for i in range(G)]        # day-level scalar
A        = zscore(rewards)                      # G 个
loss     = -sum(A[i] * sum_t log_pi[i, t] for i in range(G))
optimizer.step()
```

### 改后伪代码

```python
# 每天一次
rollouts = sample_G_rollouts(day)                   # G × 48
r        = reward_matrix(rollouts)                  # shape (G, 48)
G_ret    = discounted_return_to_go(r, gamma)        # shape (G, 48)
A        = zscore_per_step(G_ret)                   # shape (G, 48)
loss     = -sum(A[i, t] * log_pi[i, t]
                for i in range(G) for t in range(48))
optimizer.step()
```

### 具体函数

```python
def discounted_return_to_go(r, gamma):
    # r: (G, T)
    G_ret   = np.zeros_like(r)
    running = np.zeros(r.shape[0])
    for t in reversed(range(r.shape[1])):
        running     = r[:, t] + gamma * running
        G_ret[:, t] = running
    return G_ret

def zscore_per_step(G_ret, eps=1e-8):
    # G_ret: (G, T)
    mean = G_ret.mean(axis=0, keepdims=True)     # (1, T)
    std  = G_ret.std(axis=0,  keepdims=True)     # (1, T)
    return (G_ret - mean) / (std + eps)
```

## 关键确认点

- **reward 本来就是 per-step**：当前 reward 公式 `-0.01 * (1.0 * net_building_energy_kwh + 50.0 * pmv_violation * occupancy)` 每步都算一次，只是在训练管道里被 sum 成 day scalar。step-level 直接用原始 `r_t` 即可，无需改 reward 定义。
- **episode 固定 horizon=48**：Stage 2 Miami 30-min timestep × 24h = 48 步。末步 `G_{i, 47} = r_{i, 47}`，无需 value bootstrap。
- **std=0 保护**：当 G 条 rollout 在某个 t 上 return-to-go 完全一致时，std=0；用 `eps` 或把该步 advantage 置零。Early training 不太会碰到，但 mode 塌到单一分支、或某段 occupancy=0 reward 常数化时可能出现，要 guard 住。
- **KL 和 clip ratio 不变**：这俩作用在 token 级 log ratio 上，和 advantage 是 day-level 还是 step-level 无关。
- **per-step reward 标准化暂不做**：是否对 `r_t` 本身除以 running std 是独立问题，先不动，避免一次改两件事导致归因困难。

## γ 选择

| γ    | 有效 horizon (steps) | 有效 horizon (hours, 30-min step) | 备注 |
|------|----------------------|-----------------------------------|------|
| 1.00 | ∞                    | ∞                                 | 等价于每步用 full-day return，和现状最接近 |
| 0.99 | ~100                 | ~50h                              | 近似 undiscounted，先验证 step-level attribution 本身是否生效 |
| 0.95 | ~20                  | ~10h                              | 覆盖典型 thermal inertia + forecast 6h window |
| 0.90 | ~10                  | ~5h                               | 激进，测 step-level credit 在更短时间尺度上是否有效 |

**建议起步：γ = 0.99**。先观察"从 day-level 换成 step-level 本身有没有用"；确认有效后再跑 γ ∈ {0.95, 0.90} 的 ablation 测时间尺度敏感度。

## 需要改的文件

| 文件 | 改动 |
|------|------|
| `train_qwen3_houston_gspo_stage2.py` | advantage 计算段：day-level z-score → per-step return-to-go + per-step z-score；loss aggregation 段：`A[i]` 广播 → `A[i, t]` 按步贴 |
| `grpo_miami_bandit.py` | rollout 输出保留 per-step `r_t` 数组（如果现在已经是 sum 成 day scalar 就改成保留数组；day scalar 继续作为 logging metric 保留） |
| WandB / JSONL 日志 | 新增 `step_advantage_mean`, `step_advantage_std`, `step_advantage_max_abs`；保留现有 `day_reward_mean` 用于和历史曲线对齐 |

## 验证 checklist

起一个短 run（1-2 episode × 16 weekdays）先确认：

1. `A[i, t]` 在每个 t 上均值 ≈ 0、std ≈ 1（按定义应当如此，验证实现正确性）。
2. 同一 day 内不同 step 的 `A[i, t]` 存在非平凡差异（不是 day-level 广播的退化情况）。
3. `day_reward_mean` 短期内不应明显劣于现有 day-level baseline（方差稍大可接受，这是 step-level MC 的固有代价）。
4. `total_kl`、`raw_grad_norm` 量级与现 `recover24_klfix_gn2` 曲线大致可比；显著偏大意味着 step-level advantage 引入了过强的 per-token 更新信号，需要回看 clip / grad norm 配置。

## 对照跑建议

直接 fork 现在的 setpoint-only weather macro fresh run：

- 起点 checkpoint：`result/gspo/stage1_checkpoints/miami_qwen3_8b_klguard_gpu0_checkpoint-16_for_stage2_20260414`
- KL reference：同上
- γ = 0.99
- `max_grad_norm = 2.0`（与现 GPU0 gn2 对照保持一致）
- `--max-steps 80`（约 5 episodes）先看趋势，再决定是否拉到 160

WandB run 命名建议（延续现有风格）：
`miami_stage2_qwen3_8b_gpu?_setpointonly_weather_steplevel_gamma099_20260416`

Tmux session 命名建议：
`asim_stage2_setpointonly_weather_gpu?_steplevel_gamma099_20260416`

## 与 VAPO-style critic 的关系

这条路径**不是 VAPO**，但是 VAPO 的**判据 baseline**。核心逻辑：

- step-level group-normalized return-to-go ≈ "用同组 rollout 对 `V(s_t)` 做隐式估计"。
- 如果这个隐式估计在当前 setup 下已经足够稳（reward 曲线、KL、grad norm 都健康），那就没必要显式引入 value network。
- 如果 step-level advantage 方差过大（表现为训练抖动显著、reward curve 不收敛），说明需要真正的 critic 来降方差 → 那时再考虑 VAPO 的 `Value-Pretraining + Decoupled-GAE + 独立小 critic`（小 critic 而非 LLM backbone 共享的 value head，因为 state 是 low-dim 的 building physical state，不需要 8B 参数编码）。

换句话说，这步实验的结论直接决定要不要走 VAPO 那条更重的路。

## 后续（可选）

一旦 step-level baseline 稳定，可以增量尝试：

- **γ ablation**：{0.99, 0.95, 0.90} 三条曲线比较。
- **per-step reward 归一化**：对 `r_t` 除以 running std，观察是否进一步降方差。
- **混合 advantage**：`A_mix = α * A_step + (1 - α) * A_day`，把 step-level 和 day-level 信号加权混合，作为过渡。
- **独立小 critic**：引入 MLP / 轻量 LSTM 作 `V(s_t)`，用 Stage 1 rollout 做 Value-Pretraining，切到 GAE advantage。
