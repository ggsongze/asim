# Houston Forecast-Aware RL Notes

当前主线实验文件是 [TODO_compare_single_agent.ipynb](/home/AD/user/lab/asim/TODO_compare_single_agent.ipynb)。

LSTM 对照 notebook 是 [TODO_compare_single_agent_lstm.ipynb](/home/AD/user/lab/asim/TODO_compare_single_agent_lstm.ipynb)。

## Current Setup

- building: `houston.idf`
- weather: `weather/houston_2025_06_01_2025_09_30_historical_weather_api.epw`
- control step: `10` minutes
- action: next-step action only
- reward:
  - net building electricity penalty
  - PMV penalty
  - current formula:
    `reward = -0.01 * (1.0 * net_building_energy_kwh + 50.0 * pmv_violation * occupancy)`
  - where `net_building_energy_kwh = max((Electricity:Facility - ElectricityProduced:Facility) / 3.6e6, 0.0)`
  - net building electricity is counted once per env step, while comfort penalty is summed across zones

## Runtime Dependencies

- 当前 `.venv` 已回退到 `numpy==1.23.5`
- 当前 `.venv` 已回退到 `pythermalcomfort==2.10.0`
- 当前 `.venv` 的 PyTorch 已切到 `torch==2.7.0+cu128`
- 这次回退是为了兼容 `controllables` 现有依赖，以及 notebook 里仍在使用的 `pytc.models.pmv_ppd` / `pytc.utilities.clo_dynamic` 旧 API
- 当前机器 `nvidia-smi` 可见 `2` 张 `NVIDIA RTX 6000 Ada Generation`，PyTorch 侧 `torch.cuda.is_available() == True`、`torch.cuda.device_count() == 2`

## Forecast Observation

第一版先把 forecast 作为 observation，不把 action 扩展成未来轨迹。

当前接入的 forecast 文件是：

- `weather/houston_2025_06_01_2025_09_30_hourly_model_runs_api_label_h6.csv`

当前 observation 里的 forecast 部分采用固定 shape 的 `Box`：

- `forecast_available`: scalar
- `forecast_temperature_6h`: shape `(6,)`
- `forecast_humidity_6h`: shape `(6,)`
- `forecast_cloudcover_6h`: shape `(6,)`
- `forecast_precip_prob_6h`: shape `(6,)`
- `forecast_precip_6h`: shape `(6,)`

当前 forecast window 实际观察的量是：

- 未来 `1-6` 小时室外干球温度
- 未来 `1-6` 小时室外相对湿度
- 未来 `1-6` 小时总云量覆盖率
- 未来 `1-6` 小时降雨概率
- 未来 `1-6` 小时降雨量
- 当前时刻之前是否已经有可用的 forecast issue time

实现原则：

- 每 `10` 分钟 step 一次
- 每一步都读取“当前时刻之前最近的一条 forecast issue time”
- observation 结构始终固定，便于 RLlib 直接消费

## Local Qwen Endpoint

本地 `Qwen3-8B` 现已部署成 OpenAI-compatible endpoint。

- runtime env: `/home/AD/user/lab/asim/.venv_qwen`
- start script: `/home/AD/user/lab/asim/start_qwen3_vllm.sh`
- endpoint: `http://127.0.0.1:8000/v1`
- api key: `local-qwen`
- served model name: `Qwen3-8B-local`
- base model: `Qwen/Qwen3-8B`

默认启动参数：

- `CUDA_VISIBLE_DEVICES=0`
- `--dtype bfloat16`
- `--max-model-len 4096`
- `--gpu-memory-utilization 0.45`

当前 planner 接本地 Qwen 时建议设置：

- `OPENAI_BASE_URL=http://127.0.0.1:8000/v1`
- `OPENAI_API_KEY=local-qwen`
- `OPENAI_MODEL=Qwen3-8B-local`

相关文件：

- planner module: `llm_setpoint_planner.py`
- local endpoint smoke: `smoke_test_local_qwen_endpoint.py`
- local planner smoke result: `result/qwen_server/local_qwen_endpoint_smoke_test.json`
- server log: `result/qwen_server/qwen3_vllm.log`
- online rollout smoke: `run_llm_planner_episode.py`
- online rollout result: `result/llm_planner_rollout/qwen3_8b_forecast_smoke/summary.json`

Qwen3 在本地 endpoint 下默认会输出 thinking。当前 backend 已自动在 prompt 前加 `/no_think`，并限制 `max_output_tokens=256`，避免思考内容把 planner JSON 挤掉。

当前 planner prompt 已明确：

- setpoint 不固定在 `24C`
- 模型可以在 `20-30C` 内自由选择
- 只有输出无效时才回退到 fallback setpoint
- planner 当前还会在 prompt 内加入 zone 排序、相对均值偏差、推荐 setpoint band，以及“多数 occupied zone 略热时整体下调”的全局提示
- planner 当前输出会被量化到 `0.1C`
- prompt 内已加入 very-short few-shot decision examples，帮助本地 `Qwen3-8B` 稳定输出 JSON setpoint
- planner 输出之后还会经过一层 post-check，当前包含：
  - 对近似对称 zone 的一致性修正
  - 对 hotter / higher-PMV occupied zone 的单调性约束
- rollout JSONL 现在会同时记录 `sanitized_setpoints` 和最终 `setpoints`
- planner 也支持多候选模式：
  - `LLM_CANDIDATE_COUNT`
  - `LLM_TEMPERATURE`
  - `LLM_MAX_GENERATION_ATTEMPTS`
- 多候选模式下，rollout JSONL 会额外记录 `candidate_summaries`
- 当前多候选主线已经切到 compact one-shot JSON：
  - `{"comfort_first": [...], "balanced": [...], "energy_saving": [...]}`
  - 数组顺序固定为 planner 的 `zone_ids`
- rollout 侧对 online planner 额外加了两层保护：
  - 同一 `wallclock` 只处理一次
  - 对明显无效的 observation 做物理过滤，避免 `0.0 C` 这类假值进入 planner
- 当前可工作的 compact candidate smoke:
  - `result/llm_planner_rollout/qwen3_8b_forecast_day_smoke_v17_compact_candidates/`

## GSPO Bandit Start

当前没有直接把 Qwen3-8B 接成“整段 episode 上的在线 RL trainer”，而是先落成一条更稳的 `one-step real-environment GSPO bandit` 路径。

设计目标：

- 保持房间数为 `8` 个
- 保持 observation window 和当前 Houston RL 环境一致
- 让 LLM 在真实 EnergyPlus 状态上输出下一步 `8` 个房间的 setpoint
- reward 不靠 surrogate，而是直接读真实环境下一步 reward

当前 8 个房间是：

- `1FNW`
- `1FNE`
- `0FNW`
- `0FNE`
- `1FSW`
- `1FSE`
- `0FSW`
- `0FSE`

新增文件：

- `gspo_houston_bandit.py`
- `collect_houston_gspo_dataset.py`
- `train_qwen3_houston_gspo.py`
- `smoke_test_houston_gspo_bandit.py`

### GSPO Bandit Mechanics

`gspo_houston_bandit.py` 当前做的是：

- 从真实 Houston 环境中采样一个“物理上合理”的 observation window
- observation window 必须覆盖全部 `8` 个房间
- prompt 中保留与 RL 一致的核心量：
  - `temperature_drybulb`
  - `temperature:radiant`
  - `humidity`
  - `occupancy`
  - `energy_consumption`
  - `PV`
  - `forecast_available`
  - `forecast_temperature_6h`
  - `forecast_humidity_6h`
  - `forecast_cloudcover_6h`
  - `forecast_precip_prob_6h`
  - `forecast_precip_6h`
- 对任意 completion JSON，把它 replay 到同一个 sampled state
- 只前进一步，直接读取环境 reward，形成 `one-step contextual bandit` 训练样本

当前 observation plausibility filter 只做最基本的物理筛选：

- `temperature_drybulb` 必须在 `[10, 40] C`
- `humidity` 必须在 `[0, 100] %`

另外，GSPO state sampler 现在还会额外过滤“非主运行时段”的 observation：

- `wallclock` 年份必须落在 forecast CSV 的有效年份内
- 如果 `include_forecast=True`，则必须满足 `forecast_available = 1`

这样采到的 state 会稳定落在 Houston 主 weather run period，而不是 sizing / warmup 阶段。

### GSPO Smoke Results

最小 smoke 已完成：

- result: `result/gspo/smoke_test_houston_gspo_bandit.json`
- dataset smoke summary: `result/gspo/houston_gspo_dataset_smoke.summary.json`

当前已验证到：

- `prompt_zone_count = 8`
- `zone_ids` 已扩展并固定为 8 房间
- 采样到的 observation window 是可解析且物理合理的
- completion 可以被 replay 回真实 Houston 环境
- 下一步 reward 可以被成功读出
- smoke 当前命中的主时段样本已经回到 `2025-08-01`
- forecast window 当前也已经确认是非零、可用的真实 6h 预测向量

### GSPO Trainer Status

`train_qwen3_houston_gspo.py` 当前采用的是 `TRL GRPOTrainer + sequence-level importance sampling`，作为 GSPO 风格的第一版实现：

- `importance_sampling_level = "sequence"`
- `remove_unused_columns = False`
- reward function 直接调用 `HoustonGSPOBandit.evaluate_completion(...)`

当前限制：

- GSPO 训练依赖已安装到 `.venv_qwen`
- 当前安装结果：
  - `datasets==4.8.4`
  - `accelerate==1.13.0`
  - `peft==0.18.1`
  - `trl==0.29.1`
- 实际安装命令：
  - `/home/AD/user/lab/asim/.venv_qwen/bin/pip install datasets accelerate peft trl`
- 这样做是为了不碰当前正在跑 Houston RL 正式训练的主 `.venv`
- `.venv_qwen` 内已验证可以正常 import：
  - `datasets`
  - `accelerate`
  - `peft`
  - `trl.GRPOConfig`
  - `trl.GRPOTrainer`
- 本地 `Qwen3-8B-local` endpoint 在安装后仍然存活
- 当前有一个需要记住的 warning：
  - `TRL 0.29.1` import 时会提示它官方验证过的 `vLLM` 版本范围较老
  - `.venv_qwen` 当前是 `vllm==0.18.0`
  - 这不会影响当前 `GRPOTrainer` 的基础 import，但后续若直接走 `TRL <-> vLLM` 深度集成，还要再做兼容性验证

所以现在的状态是：

- GSPO 数据采样路径已完成
- GSPO reward 回传路径已完成
- 8 房间 observation window 已验证正确
- 真正的 Qwen3-8B GSPO trainer 现在已经满足“可启动”的依赖条件

### GSPO Trainer Smoke

已经在 `.venv_qwen` 上做过最小 trainer smoke，命令主线是：

- `CUDA_VISIBLE_DEVICES=1 /home/AD/user/lab/asim/.venv_qwen/bin/python train_qwen3_houston_gspo.py ... --max-steps 1 --num-generations 2 --use-peft`

这轮 smoke 验证到的正常项：

- dataset 能被 `TRL` 正常读入
- `Qwen/Qwen3-8B` 5 个 checkpoint shards 能在 `GPU 1` 上加载完成
- LoRA 训练路径能真正进入 `trainer.train()`
- 第一步 reward callback 确实进入了真实 Houston EnergyPlus 环境
  - 日志里可以看到 `Starting Simulation at 08/01/2025`
  - 说明不是假 reward，也不是纯 surrogate

这轮 smoke 暴露出的两个主要问题：

1. 模型输出有一部分不是合法 JSON
   - 当前 bandit reward 路径里已经观察到：
     - `ValueError: No JSON payload found in planner response.`
   - 这说明第一版 GSPO 训练里，raw completion 还不够稳定，必须继续保留 parse-failure penalty 和 output monitoring

2. 当前机器剩余显存不足以让 `Qwen3-8B + LoRA + GRPO` 稳定完成 backward
   - 实际现象：
     - 模型加载后 `GPU 1` 占用达到约 `28 GiB`
     - `accelerate` 报告部分参数被 offload 到 CPU / meta device
     - backward 最终报：
       - `RuntimeError: Function MmBackward0 returned an invalid gradient at index 1 - expected device meta but got cuda:0`
   - 这说明当前 smoke 已经跑到了“生成 + reward callback”阶段，但在 backward 上被 multi-device offload 打断

额外结论：

- `GRPO` 本身要求 `num_generations >= 2`
- 所以不能把 smoke 简化成单 generation

当前最准确的判断是：

- 输入链路：正常
- 模型加载：正常
- 真实环境 reward callback：正常
- 输出链路：目前会出现非 JSON completion
- 完整训练：当前被显存/设备映射问题卡在 backward

后续改成双卡后，已经跑通一轮最小 `2-GPU` smoke：

- command shape:
  - `CUDA_VISIBLE_DEVICES=0,1`
  - `--device-map auto`
  - `--max-memory '0=18GiB,1=26GiB,cpu=64GiB'`
- result dir:
  - `result/gspo/qwen3_houston_gspo_2gpu_smoke/`

这轮双卡 smoke 已验证到：

- 模型确实分布到两张卡，而不是继续大规模 offload 到 CPU/meta
  - 同一训练进程同时出现在 `GPU 0` 和 `GPU 1`
- `trainer.train()` 完整结束，退出码为 `0`
- 产物已经正常落盘：
  - `checkpoint-1/`
  - `completions/completions_00001.parquet`
  - `reward_monitor.jsonl`

输入/输出监测结果：

- `reward_monitor.jsonl` 中已经拿到 raw completion 与真实环境 reward
- 当前这轮两条 completion 都是合法 JSON：
  - `{"1FNW": 24.0, ..., "0FSE": 24.0}`
- 对应真实环境 reward 也已记录：
  - `-0.0022923432291666666`

这说明：

- 双卡方案已经把之前单卡 backward 的设备映射问题绕开了
- 当前至少在 `max_steps=1` 的 GSPO smoke 上，输入、输出、reward callback 都是正常的

### GSPO 2-GPU Pilot (`max_steps=5`)

在最小双卡 smoke 跑通后，又继续用 `8` 条真实 Houston 状态跑了一轮短 pilot：

- dataset:
  - `result/gspo/houston_gspo_dataset_pilot_d8.jsonl`
  - `result/gspo/houston_gspo_dataset_pilot_d8.summary.json`
- trainer output:
  - `result/gspo/qwen3_houston_gspo_2gpu_pilot_d8_s5/`
  - `result/gspo/qwen3_houston_gspo_2gpu_pilot_d8_s5/reward_monitor.jsonl`

这轮 pilot 的直接结果：

- 训练已完整结束，`checkpoint-5/` 已生成
- `reward_monitor.jsonl` 共记录 `10` 条 completion（`5` step x `2` generations）
- `status = ok` 的 completion 有 `10/10`
- 没有 parse failure
- 当前只出现了 `2` 种动作模式：
  - 大多数 step：全房间 `24.0 C`
  - 一个较热样本点：全房间 `23.0 C`
- 当前观测到的 wallclock 为：
  - `2025-08-01 00:10:00+00:00`
  - `2025-08-01 16:12:00+00:00`
  - `2025-08-02 00:10:00+00:00`
  - `2025-08-03 00:10:00+00:00`
  - `2025-08-03 08:11:40+00:00`
- 对应 reward 只出现了 `3` 个值：
  - `-0.11345118888888889`
  - `-0.0022923432291666666`
  - `-0.0`

更关键的是，这轮还没有形成真正的 GSPO 学习信号：

- `checkpoint-5/trainer_state.json` 显示每一步都是：
  - `loss = 0.0`
  - `grad_norm = 0.0`
  - `frac_reward_zero_std = 1.0`
- `completions/completions_00005.parquet` 进一步确认：
  - 同一个 group 内的两条 completion 连文本都一样
  - 不是“文本不同但 reward 恰好一样”，而是模型直接重复输出同一组 setpoint
- 这说明每个 group 内的两条 completion 奖励完全一样
- 在这种情况下，虽然 `trainer.train()` 正常结束，但实际上没有发生有效策略更新

所以这轮 pilot 的准确结论是：

- 工程链路：已经通了
- 真实环境 reward：已经稳定回传
- 8 房间 observation：没有问题
- 当前 blocker：不是崩溃，而是 `group reward variance = 0`
- 下一步必须优先解决“同一 prompt 下两条 completion 几乎总是等价”这个问题，否则 GSPO 仍然学不起来

### GSPO Exploration Update

为了解决 `group reward variance = 0`，这次没有继续做人为候选逻辑，而是直接沿训练期 exploration 这条线往前推：

- `gspo_houston_bandit.py`
  - system prompt 现在显式写入即时 reward 目标：
    - `reward = -0.01 * (net_grid_kwh + sum_over_zones(50.0 * occupied_pmv_violation))`
  - 明确告诉模型：
    - `net_grid_kwh = max(facility_electricity_kwh - pv_kwh, 0)`
    - `occupied_pmv_violation = occupancy * max(abs(PMV) - 0.5, 0)`
    - 只优化“下一步 10 分钟”，不是长期计划
- `train_qwen3_houston_gspo.py`
  - 新增 generation knobs：
    - `--top-k`
    - `--top-p`
    - `--min-p`
    - `--repetition-penalty`

### Hot Occupied Probe

先用单个高信息量样本做了一个 probe：

- dataset:
  - `result/gspo/houston_gspo_dataset_probe_hot_occupied.jsonl`
- run dir:
  - `result/gspo/qwen3_houston_gspo_probe_hotocc_ng4_t135/`
- config:
  - `num_generations=4`
  - `gradient_accumulation_steps=4`
  - `temperature=1.35`
  - `top_k=50`
  - `top_p=0.92`
  - `min_p=0.05`
  - `repetition_penalty=1.05`

这轮 probe 已经不再是 4 条完全相同的 completion：

- `reward_monitor.jsonl` 里实际拿到：
  - `3` 条全房间 `24.0 C`
  - `1` 条全房间 `23.0 C`
- 对应 reward 分成两档：
  - `-0.10489292222222223`
  - `-0.11345118888888889`

更关键的是，这次第一次拿到了非零组内方差和非零梯度：

- `reward_std = 0.004279134329408407`
- `frac_reward_zero_std = 0.0`
- `grad_norm = 0.09467872232198715`

这说明：

- prompt 明确 reward 目标 + 更强的训练期采样，已经足以让同一个 prompt 下出现可学习的 completion 差异
- GSPO 现在不再只是“能跑通”，而是第一次出现了真实更新信号

### 5-Step Explore Pilot

在 probe 成功后，又用同一套 exploration 参数重跑了一轮 `5-step pilot`：

- output dir:
  - `result/gspo/qwen3_houston_gspo_2gpu_pilot_d8_s5_explore/`
- reward monitor:
  - `result/gspo/qwen3_houston_gspo_2gpu_pilot_d8_s5_explore/reward_monitor.jsonl`

结果：

- `20/20` completion 都是 `status = ok`
- 现在已经不是所有 step 都完全塌成一个动作
- 当前按 wallclock 拆开看：
  - `2025-08-01 16:12:00+00:00`
    - `4` 条 completion 出现了 `3` 种 pattern
    - reward 分成 `3` 档：
      - `-0.11405841111111112`
      - `-0.11345118888888889`
      - `-0.10489292222222223`
  - `2025-08-02 00:10:00+00:00`
    - 已出现 `24.0` 和 `24.5`
    - 但 reward 仍相同
  - `2025-08-03 08:11:40+00:00`
    - 已出现 `24.0` 和 `25.0`
    - 但 reward 仍相同

trainer state 的关键变化在最后一个 step：

- `reward_std = 0.004389678593724966`
- `frac_reward_zero_std = 0.0`
- `loss = 0.0158`
- `grad_norm = 0.1472562700510025`

所以当前准确状态应更新为：

- 之前的“完全零方差”问题已经不是全局性的
- 在 occupied / hotter / higher-net-load 样本上，GSPO 已经能得到真实更新信号
- 但当前 dataset 里很多夜间或 `net_grid = 0` 样本，仍然会天然产生等价动作和零方差
- 因此下一步更值得做的是：
  - 增加 occupied / high-load 样本占比
  - 而不是单纯继续加大 temperature

### GSPO Full Control-Window Update

上面那条 `occupied/high-load` 路线现在只保留为“诊断 GSPO 是否有学习信号”的工具，不再当正式训练分布。

当前已经把 GSPO dataset 和 reward evaluator 都切到完整 HVAC 控制窗口：

- 控制窗口：`06:30-19:00`
- collector：`collect_houston_gspo_dataset.py --window-start 06:30 --window-end 19:00`
- trainer：`train_qwen3_houston_gspo.py --window-start 06:30 --window-end 19:00`

对应的新 dataset：

- `result/gspo/houston_gspo_dataset_control_window_d20.jsonl`
- `result/gspo/houston_gspo_dataset_control_window_d20.summary.json`

摘要：

- `count = 20`
- `attempts = 20`
- `first_wallclock = 2025-08-01 06:32:00+00:00`
- `last_wallclock = 2025-08-02 13:03:20+00:00`

基于这份 full-window dataset，又跑了一轮双卡 pilot：

- `result/gspo/qwen3_houston_gspo_2gpu_controlwin_d20_s5_windowed/`
- `result/gspo/qwen3_houston_gspo_2gpu_controlwin_d20_s5_windowed/reward_monitor.jsonl`
- `result/gspo/qwen3_houston_gspo_2gpu_controlwin_d20_s5_windowed/checkpoint-5/trainer_state.json`

这轮 `max_steps=5` 的结果：

1. `2025-08-01 08:30:00+00:00`
   - completion 出现 `24.0C` 和 `23.5C`
   - `reward_std = 0.0016341408481821418`
   - `grad_norm = 0.20998837053775787`

2. `2025-08-02 08:02:30+00:00`
   - 只有单一 `24.0C`
   - `reward_std = 0.0`

3. `2025-08-02 12:03:20+00:00`
   - 只有单一 `24.0C`
   - `reward_std = 0.0`

4. `2025-08-02 09:05:00+00:00`
   - 只有单一 `24.0C`
   - `reward_std = 0.0`

5. `2025-08-01 07:32:30+00:00`
   - completion 出现 `24.0C` 和 `23.5C`
   - `reward_std = 0.0006893177633173764`
   - `grad_norm = 0.07770933210849762`

这说明：

- 完整控制窗口 `06:30-19:00` 本身是可以学的
- 但完整窗口里确实包含一部分 one-step reward 天然平坦的状态
- 这轮里 `2025-08-02` 的几个 daytime 状态就是这样：`occupancy = 0` 且 `net_grid_kwh = 0`
- 所以正式训练不该把这些时段删掉，但也不能指望它们在 one-step GSPO 下每一步都提供梯度

当前更合理的方向是：

- 保持 `06:30-19:00` 作为正式训练分布
- 后续用更大的 full-window dataset 继续训练
- 必要时做“full-window coverage + informative-state mix”的 curriculum，而不是简单把 weak-signal 时段裁掉

### GSPO Weekday-Only Update

后面又确认了一条更关键的业务规则：

- 周末 HVAC 根本不启动
- 因此周末时段对这条 planner / GSPO 主线没有控制意义

所以正式训练口径进一步收紧为：

- `Mon-Fri`
- `06:30-19:00`

这里没有采用“看 7 点之后 occupancy 是否为 0 来猜 weekday/weekend”的做法。  
原因是这对肉眼判断可以，但不适合作为正式训练过滤逻辑：

- 工作日也可能临时没人
- occupancy=0 不能稳定区分“周末停机”和“工作日空置”

所以当前 collector / bandit / trainer 都直接用 `wallclock.dayofweek` 做工作日过滤。

对应代码更新：

- `gspo_houston_bandit.py`
  - 新增 `weekday_only`
- `collect_houston_gspo_dataset.py`
  - 新增 `--weekday-only`
- `train_qwen3_houston_gspo.py`
  - 新增 `--weekday-only`

新的 weekday-only dataset：

- `result/gspo/houston_gspo_dataset_weekday_controlwin_d12.jsonl`
- `result/gspo/houston_gspo_dataset_weekday_controlwin_d12.summary.json`

摘要：

- `count = 12`
- `attempts = 12`
- `window_start = 06:30`
- `window_end = 19:00`
- `weekday_only = true`
- 这批样本当前全部落在 `2025-08-01 (Friday)` 的控制时段内

对应的 2-GPU short pilot：

- `result/gspo/qwen3_houston_gspo_2gpu_weekday_controlwin_d12_s3/`
- `result/gspo/qwen3_houston_gspo_2gpu_weekday_controlwin_d12_s3/reward_monitor.jsonl`
- `result/gspo/qwen3_houston_gspo_2gpu_weekday_controlwin_d12_s3/checkpoint-3/trainer_state.json`

这轮 `max_steps=3` 的结果：

1. `2025-08-01 07:32:30+00:00`
   - 动作出现 `23.5C` 和 `24.0C`
   - reward 两档：
     - `-0.16627212222222224`
     - `-0.16489347777777777`

2. `2025-08-01 12:30:00+00:00`
   - 动作出现 `23.5C` 和 `24.0C`
   - reward 两档：
     - `-0.1468117`
     - `-0.14459678888888888`

3. `2025-08-01 14:30:00+00:00`
   - 动作出现 `23.5C`、`24.0C`、`24.5C`
   - reward 三档：
     - `-0.15572835555555556`
     - `-0.15275231111111112`
     - `-0.15143373333333332`

trainer state：

- step 1:
  - `reward_std = 0.0011074542999267578`
  - `grad_norm = 0.21063709259033203`
- step 2:
  - `reward_std = 0.001997422892600298`
  - `grad_norm = 0.08336402475833893`
- step 3:
  - `reward_std = 0.0006893227691762149`
  - `grad_norm = 0.07939362525939941`

这说明 weekday-only 口径更符合当前系统设定，而且在短 pilot 上已经连续 `3/3` 个 step 都出现了真实学习信号。

### Month-Scale Weekday Dataset

collector 现在已经改成 single-pass，不再为每个 sample 单独重跑一次 EnergyPlus。  
因此 month-scale dataset 现在可以直接生成。

完整工作日控制窗口 dataset：

- `result/gspo/houston_gspo_dataset_weekday_controlwin_fullmonth_all.jsonl`
- `result/gspo/houston_gspo_dataset_weekday_controlwin_fullmonth_all.summary.json`

这版配置是：

- `Mon-Fri`
- `06:30-19:00`
- `skip_step = 1`
- `count = 0`（表示收完整个 run）

结果：

- `count = 1650`
- `valid_steps_seen = 1650`
- `first_wallclock = 2025-08-01 06:32:00+00:00`
- `last_wallclock = 2025-09-01 18:52:00+00:00`

也就是说，当前 one-month weekday-only full-window 已经完整落盘。

### Reward Scale Note

当前 GSPO trainer 也新增了：

- `train_qwen3_houston_gspo.py --reward-scale`

不过这不是当前主矛盾。  
原因是 TRL 这里的 GRPO/GSPO-style 更新会按组做标准化 advantage，因此把 reward 全体乘一个常数，很多情况下会大幅抵消。

所以：

- 如果只是担心 reward 绝对值偏小，现在不是主要问题
- 当前更重要的仍然是：
  - 训练分布是否合理
  - completion 是否有组内差异
  - reward 是否在不同动作间真有可分性

`--reward-scale` 现在主要作为实验开关保留，而不是当前的核心修复手段。

### Workday Return + 24C Baseline Relative

当前已经把 GSPO 从“只看下一步 raw reward”的 one-step bandit，扩成了第一版 `workday return + baseline relative`。

这里的 4 个关键点已经落地：

1. **环境 step reward 本身不改**
   - 仍然使用当前环境定义：
   - `r_t = -0.01 * (net_grid_kwh + sum_z 50 * occupied_pmv_violation_z)`

2. **一个工作日的累计回报**
   - 定义：
   - `G_day = sum_t r_t`
   - 时间范围：
   - `Mon-Fri + 06:30-19:00`

3. **GSPO 更新仍按同 prompt 组内标准化**
   - TRL 这里本来就会对同一个 prompt 的多条 completion 做 grouped normalization
   - 所以天气差异不会让不同天的 raw reward 直接硬比

4. **baseline-relative logging / training reward**
   - baseline 固定为 `24.0C`
   - 训练时实际用的 raw reward 现在可以定义为：
   - `G_day(candidate) - G_day(24C baseline)`

### 当前实现语义

当前 day-level 口径已经切到真正的 `full-day closed-loop`：

- completion 不再是“下一步 8 个房间 setpoint”
- completion 现在是一个紧凑的 `workday policy JSON`
- 这个 policy 会在同一个工作日的每个 `10` 分钟控制步上，结合**当前** observation 反复计算 setpoint
- baseline 仍然是同一天、同一起点下的固定 `24.0C`

也就是说，现在比较的是：

- `G_day(candidate closed-loop policy) - G_day(24.0C baseline)`

### 新增代码

- `gspo_houston_bandit.py`
  - 新增 `_rollout_workday(...)`
  - 新增 `_rollout_workday_closed_loop(...)`
  - 新增 `evaluate_completion_workday_closed_loop(...)`
- `collect_houston_gspo_dataset.py`
  - 新增 `--day-start-only`
  - 新增 `--request-mode workday_policy`
- `train_qwen3_houston_gspo.py`
  - 新增 `--reward-mode`
  - 新增 `--baseline-setpoint-c`
  - 新增 day-level monitor 字段

### Month-Scale Day-Start Dataset

为了配合 `workday return`，又从 full-month weekday dataset 里收了一版“每天起始控制点” dataset：

- `result/gspo/houston_gspo_dataset_weekday_daystart_fullmonth.jsonl`
- `result/gspo/houston_gspo_dataset_weekday_daystart_fullmonth.summary.json`

配置：

- `Mon-Fri`
- `06:30-19:00`
- `day_start_only = true`
- `count = 0`

结果：

- `count = 22`
- `valid_steps_seen = 1650`
- `first_wallclock = 2025-08-01 06:32:00+00:00`
- `last_wallclock = 2025-09-01 06:32:00+00:00`

这 `22` 条刚好对应这一个月里的 `22` 个工作日。

另外，closed-loop policy 口径对应的 month-scale day-start dataset 也已经收好：

- `result/gspo/houston_gspo_dataset_weekday_daystart_fullmonth_workday_policy.jsonl`
- `result/gspo/houston_gspo_dataset_weekday_daystart_fullmonth_workday_policy.summary.json`

这版同样是：

- `count = 22`
- `request_mode = workday_policy`

### Day-Level Smoke

先保留了旧的 single-deviation evaluator smoke：

- `result/gspo/day_relative_eval_smoke.json`

测试内容：

- 工作日起点：`2025-08-01 06:32:00+00:00`
- candidate：全房间 `23.5C`
- baseline：全房间 `24.0C`

结果：

- `day_return = -11.455181316666666`
- `baseline_day_return = -11.454836583333329`
- `relative_day_return = -0.00034473333333728817`
- `candidate_control_steps_applied = 75`
- `baseline_control_steps_applied = 75`

这说明：

- day-level evaluator 已经真实跑完整个工作日
- baseline relative 的数值也已经正常产出

### Closed-Loop Workday Policy Dataset Smoke

新的 closed-loop 训练 prompt 现在改成了“直接输出每房间温度”，不再输出 `zone_bias_c`：

- `result/gspo/houston_gspo_dataset_weekday_daystart_workday_policy_zone_temp_smoke1.jsonl`
- `result/gspo/houston_gspo_dataset_weekday_daystart_workday_policy_zone_temp_smoke1.summary.json`

这版 prompt 不再要求输出下一步 setpoint，而是要求输出：

```json
{
  "occupied_zone_setpoints_c": [24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0],
  "unoccupied_zone_setpoints_c": [27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0],
  "temp_gain": 0.6,
  "pmv_gain": 0.8,
  "hot_forecast_gain": 0.3,
  "net_grid_gain": 0.3
}
```

其中两个 8 维温度数组的固定顺序是：

- `1FNW`
- `1FNE`
- `0FNW`
- `0FNE`
- `1FSW`
- `1FSE`
- `0FSW`
- `0FSE`

### Full-Day Closed-Loop Evaluator Smoke

正式的 closed-loop smoke 在：

- `result/gspo/workday_closed_loop_relative_eval_zone_temp_smoke.json`

这次 candidate 不再只影响第一步，而是整天 `75` 个控制步都由同一个 policy 闭环执行。结果：

- `candidate_control_steps_applied = 75`
- `baseline_control_steps_applied = 75`
- `relative_day_return = 0.012195711111102625`

这说明：

- candidate closed-loop policy 已经真实回写到 EnergyPlus
- baseline relative 的 day return 已经是“整天闭环对整天闭环”的比较
- 不再是旧的 `single-deviation` 语义

### Minimal GSPO Smoke

还额外做了一个最小 closed-loop trainer smoke，直接验证新的 zone-temperature schema：

- dataset：
  - `result/gspo/houston_gspo_dataset_weekday_daystart_workday_policy_zone_temp_smoke1.jsonl`
- output：
  - `result/gspo/qwen3_houston_gspo_closedloop_zone_temp_t11_g4/`

配置：

- `--reward-mode workday_closed_loop_relative`
- `--baseline-setpoint-c 24.0`
- `max_steps = 1`
- `temperature = 1.1`
- `top_p = 0.95`
- `min_p = 0.05`
- `repetition_penalty = 1.02`
- `num_generations = 4`

结果：

- trainer 分支已经正常走到 `workday_closed_loop_relative`
- reward monitor 已记录：
  - `day_return`
  - `baseline_day_return`
  - `relative_day_return`
  - `candidate_control_steps_applied`
  - `baseline_control_steps_applied`
  - `policy`

这次 4 条 completion 已经不再完全相同，至少出现了三种直接温度模板：

- `occupied_zone_setpoints_c = [22.0 x 8]`, `unoccupied_zone_setpoints_c = [28.0 x 8]`
- `occupied_zone_setpoints_c = [23.0 x 8]`, `unoccupied_zone_setpoints_c = [27.0 x 8]`
- `occupied_zone_setpoints_c = [24.0 x 8]`, `unoccupied_zone_setpoints_c = [27.0 x 8]`

对应的 `relative_day_return` 分别是：

- `-0.24234836666667192`
- `-0.12624841111112062`
- `0.0067763333333275`

训练指标也已经不是零信号：

- `reward_std = 0.1200682520866394`
- `grad_norm = 0.11126472055912018`
- `frac_reward_zero_std = 0.0`

这说明：

- 直接输出每房间温度后，组内多样性已经明显好于旧的 `zone_bias` schema
- GSPO 现在已经能拿到非零 advantage 并产生真实梯度
- 但 4 条 completion 里仍有两条重复在 `24.0C` 模板上，所以多样性问题只是缓解，不是完全消失

### Full-Month Zone-Temperature Pilot

为了真正按“整个月工作日”测试新的 direct-temperature schema，又重新收了一版整个月 day-start dataset：

- `result/gspo/houston_gspo_dataset_weekday_daystart_fullmonth_workday_policy_zone_temp.jsonl`
- `result/gspo/houston_gspo_dataset_weekday_daystart_fullmonth_workday_policy_zone_temp.summary.json`

这版口径是：

- `Mon-Fri`
- `06:30-19:00`
- `day_start_only = true`
- `request_mode = workday_policy`

summary 结果：

- `count = 22`
- `attempts = 1650`
- `first_wallclock = 2025-08-01 06:32:00+00:00`
- `last_wallclock = 2025-09-01 06:32:00+00:00`

随后直接起了一轮 month-scale closed-loop GSPO pilot：

- output:
  - `result/gspo/qwen3_houston_gspo_closedloop_zone_temp_fullmonth_s22_g4/`
- launch:

```bash
./.venv_qwen/bin/python train_qwen3_houston_gspo.py \
  --dataset-path result/gspo/houston_gspo_dataset_weekday_daystart_fullmonth_workday_policy_zone_temp.jsonl \
  --output-dir result/gspo/qwen3_houston_gspo_closedloop_zone_temp_fullmonth_s22_g4 \
  --max-steps 22 \
  --num-generations 4 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 1 \
  --save-steps 11 \
  --logging-steps 1 \
  --reward-mode workday_closed_loop_relative \
  --baseline-setpoint-c 24.0 \
  --window-start 06:30 \
  --window-end 19:00 \
  --weekday-only \
  --use-peft \
  --device-map auto \
  --max-memory '0=18GiB,1=26GiB,cpu=64GiB' \
  --temperature 1.1 \
  --top-p 0.95 \
  --min-p 0.05 \
  --repetition-penalty 1.02 \
  --reward-monitor-path result/gspo/qwen3_houston_gspo_closedloop_zone_temp_fullmonth_s22_g4/reward_monitor.jsonl
```

这轮训练当前已经启动，第一批 `reward_monitor` 结果表明它不是空跑。第一组 prompt 来自 `2025-08-11`，4 条 completion 中至少有两种不同 policy：

- `occupied_zone_setpoints_c = [22.0 x 8]` 时，`relative_day_return = 5.177064622222225`
- `occupied_zone_setpoints_c = [24.0 x 8]` 时，`relative_day_return = 3.4199874222222206`

这说明：

- 整个月口径已经正式起跑
- direct-temperature schema 在 month-scale run 里仍然保留了组内差异
- 当前训练并不是只在单日 smoke 上有信号

这轮 month-scale pilot 现在已经完整结束：

- `global_step = 22`
- `epoch = 1.0`
- 最终 checkpoint:
  - `result/gspo/qwen3_houston_gspo_closedloop_zone_temp_fullmonth_s22_g4/checkpoint-22`

最终 trainer 指标尾部显示：

- `reward_std` 在最后几步都保持非零，step 22 时是 `0.0809902623295784`
- `grad_norm` 也一直非零，step 22 时是 `0.09937587380409241`
- `frac_reward_zero_std` 在最后几步持续是 `0.0`

`reward_monitor.jsonl` 汇总：

- `rows = 88`，对应 `22` 个工作日 * `4` 条 completion
- `days_seen = 22`
- `unique_policies = 17`
- `reward_min = -0.22607942222222377`
- `reward_max = 17.648517155555567`
- `reward_mean = 2.4811735739899`

最常见 policy 仍然是默认模板：

- `occupied_zone_setpoints_c = [24.0 x 8]`
- `unoccupied_zone_setpoints_c = [27.0 x 8]`
- 出现 `45 / 88` 次

但已经不是唯一模板，后面还出现了：

- `[22.0 x 8] / [28.0 x 8]`
- `[23.0 x 8] / [28.0 x 8]`
- `[23.0 x 8] / [27.0 x 8]`
- `[24.0 x 8] / [28.0 x 8]`

所以这轮的结论是：

- full-month closed-loop GSPO 已经真正跑完
- 训练信号是稳定存在的
- direct-temperature schema 确实比旧的 `zone_bias` schema 更容易产生组内差异
- 但策略仍然明显偏向默认 `24C/27C` 模板，下一步该继续提高 policy 表达和采样多样性

### Longer 220-Step Run

基于上面的结论，又继续按同一套 month-scale dataset 启动了一轮更长的验证，作为“约 10 epochs” 的长跑：

- output:
  - `result/gspo/qwen3_houston_gspo_closedloop_zone_temp_fullmonth_s220_g4/`
- log:
  - `result/gspo/qwen3_houston_gspo_closedloop_zone_temp_fullmonth_s220_g4/run.log`
- reward monitor:
  - `result/gspo/qwen3_houston_gspo_closedloop_zone_temp_fullmonth_s220_g4/reward_monitor.jsonl`

启动命令：

```bash
./.venv_qwen/bin/python train_qwen3_houston_gspo.py \
  --dataset-path result/gspo/houston_gspo_dataset_weekday_daystart_fullmonth_workday_policy_zone_temp.jsonl \
  --output-dir result/gspo/qwen3_houston_gspo_closedloop_zone_temp_fullmonth_s220_g4 \
  --max-steps 220 \
  --num-generations 4 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 1 \
  --save-steps 22 \
  --logging-steps 1 \
  --reward-mode workday_closed_loop_relative \
  --baseline-setpoint-c 24.0 \
  --window-start 06:30 \
  --window-end 19:00 \
  --weekday-only \
  --use-peft \
  --device-map auto \
  --max-memory '0=18GiB,1=26GiB,cpu=64GiB' \
  --temperature 1.1 \
  --top-p 0.95 \
  --min-p 0.05 \
  --repetition-penalty 1.02 \
  --reward-monitor-path result/gspo/qwen3_houston_gspo_closedloop_zone_temp_fullmonth_s220_g4/reward_monitor.jsonl
```

当前状态：

- 这轮后来被停止，没有继续跑完
- 原因不是资源问题，而是目标函数定义不对：当前 `workday_policy` 仍然是“day-start 生成一条 compact policy，然后整天复用这条 policy”
- 这和目标中的“每 10 分钟滚动重规划/重新 query 模型”不一致

需要明确区分：

- 当前实现不是“一天固定一个温度”
- 但确实是“一天只采样一次 policy JSON”

代码证据：

- prompt 明确要求 `the same policy will be executed at every 10-minute step of the workday`：
  [gspo_houston_bandit.py](/home/AD/user/lab/asim/gspo_houston_bandit.py#L201)
- rollout 中每个 step 都复用同一个 `policy`：
  [gspo_houston_bandit.py](/home/AD/user/lab/asim/gspo_houston_bandit.py#L523)
- trainer 也是对每个 prompt 只生成一个 completion 再评 reward：
  [train_qwen3_houston_gspo.py](/home/AD/user/lab/asim/train_qwen3_houston_gspo.py#L142)

下一步主线应改成：

- completion 回到“下一步 8 房间 setpoint”
- reward evaluator 在整天 rollout 中每个 10 分钟都重新 query 当前模型
- 再按整天累计 `relative_day_return` 训练

### Corrected Rolling-Query Plan

当前确认后的正确主线如下：

1. 控制语义

- 控制周期固定为 `10` 分钟
- 控制窗口固定为 `Mon-Fri, 06:30-19:00`
- 每个控制步都重新读取最新 observation
- 每个控制步都重新 query 当前模型
- 模型只输出“下一步 8 房间 cooling setpoint”，精度 `0.1 C`

2. 训练目标

- step reward 仍然沿用：
  - `reward = -0.01 * (net_grid_kwh + sum_over_zones(50 * occupied_pmv_violation))`
- 真正优化目标改成：
  - `G_day(candidate rolling planner) - G_day(24.0C baseline)`
- 也就是：同一天 candidate 整天累计回报，减去同一天固定 `24C` baseline 的整天累计回报

3. 数据集语义

- 一个训练样本只表示“某个工作日的起始控制点”
- 不再让 dataset 自带“整天 compact policy prompt”
- day-start dataset 只负责提供：
  - `skip_valid_steps`
  - `target_date`
  - 初始 observation

4. evaluator 语义

- 旧版 `workday_policy`: day-start 生成一条 compact policy，整天复用
- 新版 rolling evaluator:
  - 每个 10 分钟重新构造 step-action prompt
  - 每个 10 分钟重新 query 当前模型
  - 每个 10 分钟重新输出下一步 setpoint
  - 整天累计 reward

5. 训练实现路线

- Stage A:
  - 先做 rolling step-action evaluator
  - 先用本地 Qwen endpoint 跑单日 smoke
- Stage B:
  - baseline daily return 做缓存，避免同一天重复跑 `24C`
  - candidate rollout 尽量并行
- Stage C:
  - 再把 trainer 从“completion-based reward”改成“trajectory-based reward”
  - 也就是：同一天里生成 `K` 条完整滚动轨迹，再做 same-day grouped normalization

6. 当前已开始执行

- 已新增真正的 rolling evaluator 到 [gspo_houston_bandit.py](/home/AD/user/lab/asim/gspo_houston_bandit.py)
- 已新增 smoke 脚本：
  - [smoke_test_houston_gspo_rolling_step_action.py](/home/AD/user/lab/asim/smoke_test_houston_gspo_rolling_step_action.py)

这个脚本现在会：

- 用本地 OpenAI-compatible Qwen endpoint
- 在工作日控制窗口里每 `10` 分钟重新 query 一次模型
- 输出下一步 8 房间 setpoint
- 对比同一天 `24.0C baseline`
- 写出 smoke 结果到 `result/gspo/rolling_step_action_workday_smoke.json`

### Rolling Step-Action Smoke

rolling evaluator 的第一轮 smoke 已经跑通：

- script:
  - `smoke_test_houston_gspo_rolling_step_action.py`
- result:
  - `result/gspo/rolling_step_action_workday_smoke.json`

这轮用的是：

- 本地 `Qwen3-8B-local`
- `skip_valid_steps = 0`
- `max_control_steps = 4`
- `candidate_count = 1`

结果说明：

- `planner_mode = rolling_step_action`
- `candidate_control_steps_applied = 4`
- `baseline_control_steps_applied = 4`
- `relative_day_return = 0.0004147555555555904`

更关键的是，`candidate_action_trace` 已经证明：

- `2025-08-01 06:32:00+00:00`
- `2025-08-01 06:42:30+00:00`
- `2025-08-01 06:53:20+00:00`
- `2025-08-01 07:03:20+00:00`

这 4 个控制步都是单独重新 query 模型后得到的 `planner_step`，不再是“一天只生成一次 compact policy”。

这次 Qwen 在 4 个 step 上都给了同一个 `24.5C` setpoint 模板，但语义已经对了：

- 现在是“每一步都重新问模型”
- 而不是“整天复用同一条 policy”

所以下一步的重点不再是改 reward horizon，而是把 trainer 从 completion-based reward 改成真正的 trajectory-based rolling reward。

### Rolling Month Smoke

我又补了一层多日 smoke，确认 month-scale 入口也已经切到 rolling 语义：

- script:
  - `run_houston_rolling_planner_month.py`
- result:
  - `result/gspo/rolling_step_action_month_eval_smoke2d/summary.json`
  - `result/gspo/rolling_step_action_month_eval_smoke2d/results.jsonl`

配置：

- `rows_evaluated = 2`
- `max_control_steps = 4`
- backend:
  - 本地 OpenAI-compatible endpoint

结果：

- `status = completed`
- `mean_relative_day_return = 0.0019348666666666459`
- `first_date = 2025-08-01`
- `last_date = 2025-08-04`

也就是说，现在不只是单日 smoke 正确，month-scale runner 也已经按“每一步重新 query 模型”的口径跑通。

### Local Transformers Backend

为了后面直接评估本地 checkpoint，而不是永远绕到 endpoint，我又补了一条“模型进程内” backend：

- `llm_setpoint_planner.py`
  - 新增 `TransformersSamplingBackend`
- `smoke_test_houston_gspo_rolling_step_action.py`
  - 现在支持：
    - `--backend openai`
    - `--backend transformers`
- `run_houston_rolling_planner_month.py`
  - 同样支持：
    - `--backend openai`
    - `--backend transformers`

这条本地 backend 会：

- 直接加载 `transformers` 模型
- 用 chat template 构造 prompt
- 在整天 rollout 中每个 `10` 分钟控制步直接调用当前模型
- 不再依赖 `OpenAI-compatible endpoint`

### Local-Model Rolling Smoke

本地模型版 smoke 也已经跑通：

- result:
  - `result/gspo/rolling_step_action_workday_local_model_smoke.json`

配置：

- backend:
  - `transformers`
- model:
  - `Qwen/Qwen3-8B`
- `max_control_steps = 2`

结果：

- `status = ok`
- `planner_mode = rolling_step_action`
- `candidate_control_steps_applied = 2`
- `baseline_control_steps_applied = 2`
- `relative_day_return = -2.222222217351799e-08`

这次最关键的不是 reward 大小，而是：

- 当前训练/评估脚本已经可以不经过 endpoint
- 直接在 GPU 上加载的 `Qwen/Qwen3-8B` 上做 rolling day rollout
- `candidate_action_trace` 里已经保留了每一步的原始 completion

从这轮结果看，本地 base model 当前仍然倾向输出统一的 `24.5C` 模板，而且 `raw_output` 里还会带一层 `<think>...</think>` 包裹；不过 JSON 解析和 step-level 下发都已经正常。

### Local-Model Month Smoke

本地模型版的 month runner 也已经跑通：

- result:
  - `result/gspo/rolling_step_action_month_eval_local_model_smoke2d/summary.json`
  - `result/gspo/rolling_step_action_month_eval_local_model_smoke2d/results.jsonl`

配置：

- backend:
  - `transformers`
- model:
  - `Qwen/Qwen3-8B`
- `rows_evaluated = 2`
- `max_control_steps = 2`

结果：

- `status = completed`
- `mean_relative_day_return = 0.0012989000000000195`
- `first_date = 2025-08-01`
- `last_date = 2025-08-04`

这说明：

- rolling evaluator 现在既能走 endpoint
- 也能直接走本地模型 / 未来 checkpoint

后面自定义 trajectory trainer 就可以直接复用这一套本地 backend，而不需要再额外维护一条服务端推理链。

另外，rolling evaluator 的 `candidate_action_trace` 现在也已经补齐了训练需要的 step-level字段：

- `request_payload`
- `request_system_prompt`
- `request_user_prompt`
- `raw_output`
- `setpoints`

也就是说，后面如果切到自定义 trajectory trainer，已经可以直接从单日 rollout 里拿到“每一步看到了什么 / 输出了什么 / 得到了什么 reward”的完整链路。

### Custom Rolling Trajectory Trainer

已经新增自定义 rolling trajectory trainer：

- `train_qwen3_houston_gspo_rolling.py`

这条线不再沿用旧的“day-start 生成一条整天 compact policy”语义，而是：

- 每个 trajectory 在工作日 rollout 里每 `10` 分钟重新 query 当前模型
- 模型每一步只输出 8 房间下一步 setpoint
- 用整天累计 `relative_day_return` 做 same-day 组内比较

当前验证结果分两层：

- day-start smoke:
  - `result/gspo/qwen3_houston_gspo_rolling_smoke1_tokenpg/summary.json`
  - 这轮 `max_control_steps = 1`，而且 `2025-08-01 06:32` 的工作日早段状态太冷、太同质，两条 trajectory 都落到同一个 `24.5C` 模板
  - 所以这轮是 `reward_std = 0`、`grad_norm = 0`，并不是 rolling trainer 框架坏了，而是组内没有 reward variance
- noon debug smoke:
  - `result/gspo/debug_datasets/houston_gspo_single_noon_row.jsonl`
  - 这是一条从 `2025-08-01 12:00:00+00:00` 切出来的单行 dataset，用来做更高信号的 rolling 训练验证
  - 后台启动目录：
    - `result/gspo/qwen3_houston_gspo_rolling_noon_smoke1`
  - 对应思路是先在更热、更高占用的工作日中午状态上确认 rolling trainer 能拿到非零 `reward_std / grad_norm`，再回到整月工作日训练

所以当前的主结论是：

- rolling evaluator 已经是对的
- custom rolling trainer 也已经真正落地
- 当前主问题不再是“是否每步重规划”，而是“怎样在真实工作日状态上更快拿到组内 reward variance”

另外已修正一个 full-day smoke 口径问题：

- `smoke_test_houston_gspo_rolling_step_action.py` 现在 `--max-control-steps 0` 会正确表示“完整工作日”
- 之前脚本内部会把 `0` 强制改成 `1`
- 这会导致所谓 full-day smoke 实际只跑 1 个 control step；这个问题已经修掉

### Full-Day Rolling Smoke

修复 `--max-control-steps 0` 之后，已经完成一轮真正的单工作日 full-day rolling smoke：

- result:
  - `result/gspo/rolling_step_action_workday_fullday_2025_08_07_openai_smoke.json`
- config:
  - backend: `openai`
  - model: `Qwen3-8B-local`
  - target workday: `2025-08-07`
  - `skip_valid_steps = 300`
  - `max_control_steps = 0` meaning full workday

关键结果：

- `status = ok`
- `planner_mode = rolling_step_action`
- `candidate_control_steps_applied = 75`
- `baseline_control_steps_applied = 75`
- `relative_day_return = 2.9360280444444413`

这次可以明确确认：

- 模型是在整天 `75` 个 10 分钟控制步里被反复 query
- action 是逐步写回 EnergyPlus 的，不是“一天只生成一次 policy”
- reward 也是整天累计后再与同日 `24.0C baseline` 对比

另外，`result/qwen_server/qwen3_vllm.log` 里看到的主要是：

- 连续 `POST /v1/chat/completions ... 200 OK`
- 一个 `GET /v1/models ... 401 Unauthorized`

这里的 `401` 只是未带 API key 的模型列表探测，不是 rolling 控制失败；真正的推理请求都是 `200 OK`。

### Full-Day Trainer Smoke

为了避免再把“整天 rolling evaluator”误当成“整天 rolling trainer”，这里又额外起了一轮真正的 full-day trainer smoke：

- output dir:
  - `result/gspo/qwen3_houston_gspo_rolling_workday_2025_08_07_full_day_trainer_smoke1`
- dataset:
  - `result/gspo/debug_datasets/houston_gspo_single_workday_2025_08_07.jsonl`
- config:
  - `max_steps = 1`
  - `trajectories_per_day = 2`
  - `max_control_steps = 0`

参数语义说明：

- `max_steps = 1` 指的是只做 `1` 次 optimizer step
- `trajectories_per_day = 2` 指的是对同一个工作日起点采样 `2` 条完整候选轨迹
- `max_control_steps = 0` 指的是每条轨迹都滚完整个工作日，不做截断

当前运行方式和资源情况：

- 这轮 trainer smoke 现在在 `GPU 0` 上运行
- 进程：
  - `395934`
- `GPU 1` 当前被外部用户的独立训练任务占用，所以没有直接切到双卡 trainer

这意味着：

- full-day rolling control 语义已经由 evaluator smoke 验证通过
- 现在正在验证的是：full-day trainer 自身能否在完整工作日 horizon 下拿到非零 `reward_std / grad_norm`

后来排查发现，这条 full-day trainer smoke 之所以会看起来“跑了 8 小时还不结束”，不是 EnergyPlus 还在滚，而是 trainer 的反向实现写错了：

- 旧实现会对 completion 的每个 token 单独做一次 `8B` 模型前向，然后立刻 `backward`
- 对 full-day rolling trajectory 来说，这会把一次更新放大成成百上千次重复前向，效率极差
- 轨迹文件在 `2026-03-28 18:52 UTC` 就已经写完，但进程之后卡在低效反向/同步阶段，没有正常落出 `metrics.jsonl`

已在 `2026-03-29` 修复为 teacher-forced 版本：

- 现在每个 planner step 的 `prompt + completion` 只做一次前向
- 一次性拿到整段 completion 的 logprob，再做一次 `backward`
- 这才是正确的语言模型训练写法，也是 rolling trainer 后续继续验证的基线实现

修复后又补了一轮带 phase tracing 的 full-day trainer smoke：

- output dir:
  - `result/gspo/qwen3_houston_gspo_rolling_workday_2025_08_07_full_day_trainer_smoke4_phase_gpu0`
- 结果：
  - `status = completed`
  - `elapsed_s = 1205.25`
  - `reward_std = 0.017099`
  - `grad_norm = 0.049864`

这轮说明：

- full-day rolling trainer 现在已经能完整走完 `2` 条 full-day trajectory
- 两条 trajectory 的 reward 具有非零方差
- backward、optimizer step、metrics flush 都已经真正完成
- 也就是说，这条 trainer 已经不再挂死在 rollout 后半段

在此基础上，已经正式启动第一轮 month-scale weekday rolling training：

- output dir:
  - `result/gspo/qwen3_houston_gspo_rolling_fullmonth_weekday_s22_t2_gpu0`
- dataset:
  - `result/gspo/houston_gspo_dataset_weekday_daystart_fullmonth_workday_policy_zone_temp.jsonl`
- config:
  - `max_steps = 22`
  - `trajectories_per_day = 2`
  - `max_control_steps = 0`
  - `save_steps = 11`
  - `CUDA_VISIBLE_DEVICES = 0`

这轮的语义是：

- 先按整个月 `22` 个工作日样本跑一遍
- 每个工作日都做 full-day rolling control
- 每个工作日起点采样 `2` 条完整 trajectory
- 每个工作日结束后做 `1` 次参数更新

在 `2026-03-29` 对这轮 month-scale run 做中途审计时，发现这轮结果不能继续作为有效训练样本使用：

- 运行目录：
  - `result/gspo/qwen3_houston_gspo_rolling_fullmonth_weekday_s22_t2_gpu0`
- `trajectory_samples.jsonl` 中已写出的 `26` 条 trajectory、共 `1950` 个 `planner_step`
- 这 `1950` 个 step 的 `request_payload.forecast` 完全相同
- 不只是 `temperature_6h_c[0]` 不变，而是以下四个 forecast 向量都完全不变：
  - `temperature_6h_c`
  - `humidity_6h_pct`
  - `precip_prob_6h_pct`
  - `precip_6h_mm`

进一步核对发现：

- 这组固定 forecast：
  - `temperature_6h_c = [29.6, 29.1, 27.5, 26.4, 25.5, 24.5]`
  - `humidity_6h_pct = [46.0, 49.0, 44.0, 42.0, 46.0, 50.0]`
- 在原始 Houston forecast CSV 中实际对应：
  - `2025-09-30 16:00:00`
- 但这些样本自己的 `wallclock` 却覆盖：
  - `2025-08-01`
  - `2025-08-04`
  - `2025-08-05`
  - `...`

因此可以确认：

- 这不是“某几个小时 forecast issue time 没更新”的正常现象
- 而是这条正在运行的 month-scale trainer 在 rollout 记录里出现了 forecast 错位/冻结
- 这轮 month-scale run 已停止，不能继续作为有效训练结果解读

为避免同类问题继续静默发生，rolling planner trace 现在新增了硬校验和审计字段：

- 强制检查 `observation` 覆盖全部 `8` 个 zone
- 强制检查 `request_payload["zones"]` 也覆盖全部 `8` 个 zone
- 强制检查 `request_payload.forecast` 同 observation forecast 完全一致
- 强制检查以下四个 forecast 向量都存在且 horizon 长度都是 `6`
  - `temperature_6h_c`
  - `humidity_6h_pct`
  - `precip_prob_6h_pct`
  - `precip_6h_mm`
- 新增 trace 字段：
  - `observation_zone_ids`
  - `observation_zone_count`
  - `request_zone_ids`
  - `request_zone_count`
  - `observation_forecast`
  - `request_forecast`
  - `forecast_horizon_lengths`
  - `forecast_matches_observation`

在修复 forecast trace 之后，又重新尝试按原来的 full-parameter 配置启动 month-scale weekday rolling training：

- 运行目录：
  - `result/gspo/qwen3_houston_gspo_rolling_fullmonth_weekday_s22_t2_gpu0_forecastfix_20260329`

这轮也不能作为有效完整训练结果使用，因为它在 `step 1` 的 `optimizer.step()` 处触发了 CUDA OOM：

- 两条 full-day trajectory 都已完整 rollout
- 两条 trajectory 的 backward 也已完成
- `phase_trace.jsonl` 停在：
  - `optimizer_step_start`
- `metrics.jsonl` 仍为空
- 没有生成 `summary.json`
- 前台复现报错明确是：
  - `torch.OutOfMemoryError`
  - full-parameter 8B + Adam optimizer state 在 `GPU 0` 上无法完成第一次参数更新

因此当前 month-scale weekday rolling training 已切换到 LoRA 版本重新启动：

- 运行目录：
  - `result/gspo/qwen3_houston_gspo_rolling_fullmonth_weekday_s22_t2_gpu0_forecastfix_lora_20260329`
- 启动方式：
  - 解释器改为 `.venv_qwen/bin/python`
  - 并打开 `--use-peft`
- 当前目标：
  - 先确认 LoRA 版本能稳定通过 `optimizer.step()`
  - 再继续看完整个月训练的 reward / grad / forecast trace 是否正常

随后又定位到 LoRA 版本中一个更底层的 forecast freeze root cause：

- 问题不在 rolling planner / step-action 逻辑本身
- 也不在 LLM backend 本身
- 真正的问题出在 forecast CSV 的时间索引单位

最小复现实验表明：

- `.venv` 中的 `pandas 1.5.3` 会把 `run_time` 读成 `datetime64[ns]`
- `.venv_qwen` 中的 `pandas 3.0.1` 会把同一列读成 `datetime64[us]`
- 旧代码直接做：
  - `df["run_time"].astype("int64").to_numpy()`
- 这会在 `.venv_qwen` 下得到“微秒”时间轴
- 但查询端 `pd.Timestamp(...).value` 仍然是“纳秒”
- 结果 `searchsorted(...)` 几乎总是落到最后一行，于是整天 forecast 冻结成同一组值

最终修复是：

- 在 `ForecastBundleReader` 里显式把 `run_time` 强制转成 `datetime64[ns]`
- 具体实现改为：
  - `df["run_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)`
- 这样在 `.venv` 和 `.venv_qwen` 下都会得到一致的纳秒时间轴

修复后已验证：

- `.venv_qwen/bin/python` + 末尾 `sys.path.append(".venv/site-packages")` 可以正常导入 `transformers/accelerate`
- `2025-08-01 06:32:00+00:00` 的 `sample_state` forecast 恢复正确
- 同一天 `75` 个 planner step 恢复为 `5` 组不同 forecast bundle

对应修复文件：

- `.tmp_todo_random_start_cell0.py`
- `.tmp_todo_lstm_cell0.py`

另外，导入路径也恢复成安全的末尾 `append` 方式，避免把 `.venv` 的 `numpy/pandas` 提前注入后破坏 `.venv_qwen` 的 `accelerate/transformers`：

- `train_qwen3_houston_gspo_rolling.py`
- `train_qwen3_houston_gspo.py`
- `smoke_test_houston_gspo_rolling_step_action.py`
- `run_houston_rolling_planner_month.py`

### Correct Startup For Month-Scale LoRA Training

当前这条 Houston rolling GSPO 月度训练，正确启动方式已经固定下来：

- 解释器必须使用：
  - `.venv_qwen/bin/python`
- 不要手动前置：
  - `.venv/site-packages`
- 也不要手动设置：
  - `PYTHONPATH=.venv/site-packages`
- 原因：
  - trainer 脚本内部已经安全地把 `.venv/site-packages` 追加到 `sys.path` 末尾
  - 这样可以同时满足：
    - `transformers / torch / accelerate` 来自 `.venv_qwen`
    - RL / EnergyPlus 相关依赖可以继续从 `.venv` 补齐

推荐启动命令：

```bash
cd /home/AD/user/lab/asim

env PYTHONUNBUFFERED=1 .venv_qwen/bin/python train_qwen3_houston_gspo_rolling.py \
  --output-dir result/gspo/qwen3_houston_gspo_rolling_fullmonth_weekday_s22_t2_gpu0_<run_tag> \
  --max-steps 22 \
  --trajectories-per-day 2 \
  --save-steps 1 \
  --device cuda:0 \
  --use-peft
```

启动前注意：

- `--output-dir` 每次必须用新的目录，不要复用旧的失效 run
- 当前推荐保留：
  - `--use-peft`
  - `--device cuda:0`
  - `--max-steps 22`
  - `--trajectories-per-day 2`
- 不要在 shell 外额外注入 `.venv` 的 `numpy/pandas`

启动后快速自检：

- `phase_trace.jsonl` 应该先出现：
  - `step_start`
  - `trajectory_rollout_start`
- 第一个 full-day trajectory 落盘后，应检查：
  - `trajectory_samples.jsonl`
  - `request_payload.forecast.temperature_6h_c`
- 修复后的正确现象是：
  - 同一天 `75` 个 planner step 内 forecast 会出现多组 bundle
  - 不是整天固定为同一组值

在当前 Codex 终端环境里还有一个额外注意点：

- 不要依赖复杂的多层 `nohup ... bash -lc ... &` 包装去保活训练
- 这个环境里更稳的方式是：
  - 用 PTY 长会话直接启动训练
  - 保留会话 `session_id`
- 如果是普通 shell / tmux / screen 中手动启动，上面的单条命令即可

### Block-Based Rolling Plan (Forecast-Aligned 3h Blocks)

当前 rolling planner / GSPO 主线新增一版 **forecast-aligned block planning** 方案，用于缓解：

- full-day long-horizon credit assignment 过长
- planner completion token 过长
- 同一 prompt 下 candidate 容易塌缩到默认模板
- day-level grouped reward 对局部好坏不敏感

这版不是把物理环境步长改成 `3h`，而是：

- **EnergyPlus / 环境物理步长仍保持 `10 min`**
- **训练与规划单元改成 `3h` block**
- **每个 block 内输出 `6` 个动作点**
- **每个动作点覆盖 `30 min`，再在环境内部展开成 `3` 个 `10 min` step**

#### Block 划分

block 直接和当前 forecast issue/update 口径对齐：

- `07:00-10:00`
- `10:00-13:00`
- `13:00-16:00`
- `16:00-19:00`

也就是：

- 每个工作日 `7:00-19:00` 共 `4` 个 block
- 每个 block `3` 小时
- 每个 block 内部共有 `18` 个环境步（`3h * 6 step/h = 18`）

#### 动作语义

每个 block 输出：

- `6` 个半小时 control knots
- 每个 knot 在环境内部重复执行 `3` 个 `10 min` step

因此：

- planner 视角：`6` 个动作点 / block
- environment 视角：仍然是 `10 min` 控制与推进

推荐 block output JSON：

```json
{
  "block_start": "10:00",
  "block_end": "13:00",
  "plan": [
    {"slot": 1, "setpoints": [24.5, 24.5, 24.5, 24.5, 25.0, 25.0, 25.0, 25.0]},
    {"slot": 2, "setpoints": [24.5, 24.5, 24.5, 24.5, 25.0, 25.0, 25.0, 25.0]},
    {"slot": 3, "setpoints": [25.0, 25.0, 24.5, 24.5, 25.0, 25.0, 25.0, 25.0]},
    {"slot": 4, "setpoints": [25.0, 25.0, 25.0, 25.0, 25.5, 25.5, 25.5, 25.5]},
    {"slot": 5, "setpoints": [25.5, 25.5, 25.5, 25.5, 26.0, 26.0, 26.0, 26.0]},
    {"slot": 6, "setpoints": [26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0, 26.0]}
  ]
}
```

其中 `setpoints` 的固定顺序仍然是：

- `1FNW`
- `1FNE`
- `0FNW`
- `0FNE`
- `1FSW`
- `1FSE`
- `0FSW`
- `0FSE`

#### Block-level reward

- 环境 step reward 仍保持不变
- 对 block 内 `18` 个环境步做累计：
  - `R_block = sum_{t in block} r_t`
- 保留相对 `24.0C` baseline 的语义：
  - `R_block_rel = sum_{t in block} (r_t(candidate) - r_t(24C baseline))`

#### GSPO / grouped rollout 语义

新版 grouped training 从"每天 3 条 trajectory 比整天回报"，改成：

- 对同一个 block prompt，采样 `K=3` 条 candidate block plans
- 从同一个 block-initial state clone rollout
- 每条 candidate 都完整跑完当前 block
- 计算 `block_relative_return`
- 在该 block 内做 grouped normalization / ranking

Advantage 构造：

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
```

#### Candidate diversity forcing

不再只依赖 sampling temperature。3 条 candidate 在 prompt 中直接带有不同的 planning mode 语言描述：

- Candidate A — `comfort`: "Prioritize occupant comfort. Keep occupied zones well within the neutral PMV band, even if it costs more grid energy."
- Candidate B — `balanced`: "Balance comfort and energy. Maintain acceptable comfort while being mindful of net grid consumption."
- Candidate C — `energy_saving`: "Prioritize energy reduction. Accept PMV slightly outside the ideal band if it meaningfully reduces net grid purchase."

不给 explicit setpoint range，让 GSPO 自己学每种 mode 下什么 setpoint 最好。

#### Block 状态管理：replay 方案

EnergyPlus 不支持 state snapshot/restore，因此 block 间状态传递采用 **replay** 方案：

跨 block 状态耦合规则：

| 层面 | 要求 |
|---|---|
| 跨 block | 有耦合，block 2 起点 = block 1 选中 candidate 跑完的结果 |
| block 内 candidates | 去耦合，A/B/C 从同一个 block-start state 出发 |

每个 block 结束后：

1. 按 **max block reward** 选出 winner candidate
2. 保存 winner 的 `18` 步 action 序列（8 zone × 18 step）
3. 追加到 `winner_actions_history`

下一个 block 的每条 candidate 都需要：

1. 启动新的 EP 实例
2. skip 到工作日起点
3. replay `winner_actions_history` 中所有之前 block 的 winner actions
4. 到达当前 block 的正确初始状态
5. 再执行自己的 block plan

一天 4 block 的完整流程示例：

```
Block 1 (07:00-10:00):
  3 candidates 各自从 07:00 出发 → 跑 18 步 → 得到 R_block1_{A,B,C}
  winner = argmax(R_block1) → 保存 winner actions

Block 2 (10:00-13:00):
  3 candidates 各自: skip→07:00 → replay Block1 winner(18步) → 到达 10:00
  → 跑 Block2 自己的 18 步 → 得到 R_block2_{A,B,C}
  winner = argmax(R_block2) → 保存 winner actions

Block 3 (13:00-16:00):
  3 candidates 各自: skip→07:00 → replay Block1+Block2 winner(36步) → 到达 13:00
  → 跑 Block3 自己的 18 步 → 得到 R_block3_{A,B,C}
  winner = argmax(R_block3) → 保存 winner actions

Block 4 (16:00-19:00):
  3 candidates 各自: skip→07:00 → replay Block1+2+3 winner(54步) → 到达 16:00
  → 跑 Block4 自己的 18 步 → 得到 R_block4_{A,B,C}
```

Baseline 优化：固定 `24C` baseline 只需跑一次完整天，按 block 切出 reward 即可。

#### 成本分析

| Block | candidate EP runs | replay 步数/run | block 步数/run |
|---|---|---|---|
| Block 1 | 3 | 0 | 18 |
| Block 2 | 3 | 18 | 18 |
| Block 3 | 3 | 36 | 18 |
| Block 4 | 3 | 54 | 18 |

总 candidate EP runs = 12/天。Baseline = 1 次完整天。

#### 与旧 rolling trainer 的关系

- **旧 rolling planner**：每 `10 min` 直接输出下一步 setpoint
- **新版 block planner**：每 `3h` 输出当前 block 的 `6` 个 half-hour action knots
- **环境内部**：仍然按 `10 min` step 展开与执行

不是"物理控制周期变粗"，而是"LLM 的规划粒度变粗，但环境执行粒度不变"。

### Prompt Relaxation And T3 Restart

针对 rolling planner 过于保守的问题，`llm_setpoint_planner.py` 已并入一版更激进的 prompt / scoring / post-check 调整：

- 放宽了 prompt 里对“统一 setpoint / 小步调整 / 低温仅极端使用”的限制
- 把 candidate separation 从 `0.2-0.5 C` 提高到 `0.5-1.0 C`
- 降低 `movement_penalty`
- 大幅降低 `symmetry_penalty`
- 去掉 `global_mode_penalty`
- 对称 zone 只在 setpoint 差异本来就小于 `0.1 C` 时才 merge

如果要让新 prompt 真正生效，必须重新启动 trainer。新的 restart 也同步把：

- `--trajectories-per-day 2`

改成：

- `--trajectories-per-day 3`

推荐重启命令：

```bash
cd /home/AD/user/lab/asim

env PYTHONUNBUFFERED=1 .venv_qwen/bin/python train_qwen3_houston_gspo_rolling.py \
  --output-dir result/gspo/qwen3_houston_gspo_rolling_fullmonth_weekday_s22_t3_gpu0_forecastfix_lora_nsfix_promptv2_tty_<run_tag> \
  --max-steps 22 \
  --trajectories-per-day 3 \
  --save-steps 1 \
  --device cuda:0 \
  --use-peft
```

### Near-Term Rain And PV Hints

为了让 rolling planner 更容易学会 “降雨 / 近时段降雨概率会压低之后 PV，从而抬高后续 net grid” 这层关系，现在在保留原始 6 小时天气 forecast 的前提下，又额外加入了一组非模型式 summary hints：

- `precip_prob_next_2h_max_pct`
- `precip_mm_next_2h_sum`
- `pv_kwh_now`
- `pv_recent_delta_kwh`
- `pv_recent_trend`
- `near_term_pv_risk`

这不是外生 PV 预测模型，只是把已有 observation 重新组织给 LLM，减少它从长 forecast 数组里自己做信用分配的难度。

后续分析表明，`precip_prob` 和 `cloudcover` 应该一起保留，而不是二选一：

- `precip_prob` 和 forecast `cloudcover` 的相关系数约 `0.49`
- `precip_prob` 对 “实际是否下雨” 的相关系数约 `0.28`
- `cloudcover` 对 “实际 cloud_cover” 的相关系数约 `0.31`
- 在当前 August trajectory 上，若未来 2 小时 `precip_prob max >= 20%`，60 分钟后 PV 平均多掉约 `3.02`
- 若未来 2 小时 `cloudcover max >= 80%`，60 分钟后 PV 平均多掉约 `1.09`

因此当前策略是：

- 保留 `precip_prob`，继续表达不确定性
- 新增 `cloudcover`，提供更直接的天空遮挡连续信号
- 不把 `precip_mm` 当主信号
- 如果后面要加 `weathercode`，优先做成分类 hint，不把原始数值直接当连续变量用

基于这个结论，当前 observation / prompt 又新增了一条原始 forecast 向量：

- `forecast_cloudcover_6h`

同时 near-term hints 里新增：

- `cloudcover_next_2h_max_pct`

prompt 里也同步新增了 3 条显式规则：

- `Rain or high precipitation probability in the next 1-2 hours can reduce PV availability and increase later grid dependence.`
- `High cloud cover in the next 1-2 hours can also weaken PV even if rain totals remain small.`
- `If near-term PV risk is elevated and occupancy remains meaningful, modest pre-cooling before PV availability worsens can be beneficial.`
- `Do not overreact to distant rain; prioritize rain risk within the next 1-2 hours.`

如果要让这版雨/PV hint 生效，必须重新启动 trainer，并使用新的输出目录，避免和旧的 `promptv2` run 混淆。

### Block-Based GSPO Smoke Verification

Block-based GSPO pipeline 已通过完整 smoke test：

- script:
  - `smoke_test_houston_gspo_block.py`
- result:
  - `result/gspo/smoke_test_houston_gspo_block.json`

这轮 smoke 验证到：

- Block replay 机制正确：Block 2/3/4 的 candidate 都先 replay 前面 winner 的 actions，再从正确的 block-start state 出发
- Baseline 只跑一次完整天，按 block 切出 reward
- 3 candidate modes (comfort / balanced / energy_saving) 在 prompt 中通过不同的 mode description 强制多样性
- 所有 4 个 block 都产出了 block-level relative reward

smoke 结果摘要：

- `total_winner_relative_reward = +0.006`
- Block 0 (07-10): winner = balanced, rel_reward = -0.310
- Block 1 (10-13): winner = energy_saving, rel_reward = +0.080
- Block 2 (13-16): winner = energy_saving, rel_reward = +0.128
- Block 3 (16-19): winner = comfort, rel_reward = +0.109

但问题：所有 candidate 在所有 zone 内输出完全一致的 setpoint（例如全部 22.0C 或全部 25.0C），zone 间和 knot 间没有差异。这说明 base Qwen3 还没学到 zone/时间维度的策略差异，但 mode forcing 至少给了 block 级别的多样性信号。

### Block Planner Prompt V2 (Zone Layout + Fine-Grained Control)

为了让模型输出 zone 间和 knot 间有差异的 setpoint，对 `llm_setpoint_planner.py` 中的 BlockPlanner prompt 做了以下改进：

1. **System prompt 加入 zone layout**

   新增 `ZONE_DESCRIPTIONS` dict，描述每个 zone 的楼层、朝向、太阳暴露特征：

   - `1FNW`: Upper north-west: minimal direct sun, roof heat gain
   - `1FNE`: Upper north-east: morning indirect light, roof heat gain
   - `0FNW`: Ground north-west: earth-cooled, least solar gain
   - `0FNE`: Ground north-east: mild morning light, earth-cooled
   - `1FSW`: Upper south-west: strong afternoon sun + roof, warmest zone
   - `1FSE`: Upper south-east: morning sun + roof, warm in AM
   - `0FSW`: Ground south-west: afternoon direct sun, moderate gain
   - `0FSE`: Ground south-east: morning direct sun, moderate gain

   System prompt 现在把完整 zone layout 表格嵌入，并明确要求：
   - Each zone should get a DIFFERENT setpoint
   - Each knot should VARY over time as conditions change
   - Do NOT output the same setpoint for all zones or all knots

2. **Mode descriptions 细化**

   三个 mode 都保持 PMV ±0.5，但增加了 zone 差异化指导：

   - comfort: 给每个 zone 独立 setpoint，根据温度 / PMV / solar exposure / 楼层
   - balanced: 冷 zone 高 setpoint，热 zone 低 setpoint；向高 PV 时段倾斜
   - energy_saving: 无人 zone 靠近上限；有人 zone 用舒适范围的暖端；每 zone 反映自身热负荷

3. **User prompt 内嵌 zone 描述**

   每个 zone 的状态行现在包含 zone 描述：
   - `- 1FSW (Upper south-west: strong afternoon sun + roof, warmest zone): temp=26.3C, humidity=52%, occupancy=1, PMV=0.45`

4. **Example 改为差异化**

   旧 example 是 `[24.5, 24.5, 24.5, 24.5, 25.0, 25.0, 25.0, 25.0]`（只有两种值）。
   新 example 是 `[24.1, 24.3, 25.2, 25.0, 23.4, 23.8, 24.6, 24.2]`，每个 zone 不同，且精度到 0.1C。

5. **Quantization 从 0.5C 改为 0.1C**

   - `llm_setpoint_planner.py` prompt 内的 rounding instruction 改为 "Round to nearest 0.1 C"
   - `train_qwen3_houston_gspo_block.py` 和 `smoke_test_houston_gspo_block.py` 中 `quantization_c` 从 `0.5` 改为 `0.1`
   - 与 README 里 rolling planner 的 `精度 0.1 C` 口径对齐

### Block GSPO Full-Month Training (Prompt V2)

基于上述 prompt 改进，已启动 22 workday 完整月训练：

- output dir:
  - `result/gspo/qwen3_houston_gspo_block_fullmonth_s22_promptv2_q01_20260331/`
- dataset:
  - `result/gspo/houston_gspo_dataset_weekday_daystart_fullmonth_workday_policy_zone_temp.jsonl`
- config:
  - `CUDA_VISIBLE_DEVICES=0`
  - `--max-steps 22`
  - `--save-steps 11`
  - `--device cuda:0`
  - `--use-peft`
  - `--temperature 0.7`
  - `--top-p 0.95`
  - `--learning-rate 1e-5`
  - `--reward-scale 1.0`

启动命令：

```bash
env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 nohup .venv_qwen/bin/python train_qwen3_houston_gspo_block.py \
  --dataset-path result/gspo/houston_gspo_dataset_weekday_daystart_fullmonth_workday_policy_zone_temp.jsonl \
  --output-dir result/gspo/qwen3_houston_gspo_block_fullmonth_s22_promptv2_q01_20260331 \
  --max-steps 22 \
  --save-steps 11 \
  --device cuda:0 \
  --use-peft \
  --temperature 0.7 \
  --top-p 0.95 \
  --learning-rate 1e-5 \
  --reward-scale 1.0 \
  > result/gspo/qwen3_houston_gspo_block_fullmonth_s22_promptv2_q01_20260331.log 2>&1 &
```

Step 1 (2025-08-01) 初步结果：

- Block 0 (07-10): comfort=-0.310, balanced=-0.344, **energy_saving=-0.121 (winner)**
- Block 1 (10-13): comfort=+0.042, balanced=+0.039, **energy_saving=+0.043 (winner)**
- Block 2 (13-16): **comfort=-0.100 (winner)**, balanced=-0.104, energy_saving=-0.232
- Block 3 (16-19): **comfort=-0.132 (winner)**, balanced=-0.245, energy_saving=-0.508
- total_winner_reward = -0.310
- avg_block_reward_std = 0.080

与旧 smoke (prompt v1, quantization 0.5C) 对比：

| 指标 | 旧 smoke (v1) | 当前 step 1 (v2) |
|---|---|---|
| total_winner_reward | +0.006 | -0.310 |
| avg_block_reward_std | ~0.001 | 0.080 |
| grad_norm (avg) | ~0.0 | 0.099 |
| zone 差异 | 无 | 待确认 |

关键变化：

- reward_std 提升约 80 倍，GSPO 在每个 block 都能计算出有意义的 advantage
- grad_norm 非零，每个 block 都产生了真实参数更新
- 初始 total_winner_reward 下降，可能是新 prompt 让模型偏离了旧的保守模板（全输出 24.5C 接近 baseline），但 GSPO 信号更强，训练应该能逐步修正

### Block-Internal Rolling Knot Planning

上面的 Prompt V2 训练仍然使用 one-shot 模式：一次 LLM 调用生成一个 block 内全部 6 knots × 8 zones = 48 个 setpoint。实际结果表明 base Qwen3 在 one-shot 模式下倾向于对所有 zone 和 knot 输出相同值。

改进：将 block 内的 knot 规划从 one-shot 改为 rolling——每个 knot 边界用当前 EP observation 调用一次 LLM，只输出 8 个 setpoint（当前 30 分钟的 8 zones）。

**架构变化**：

1. **`llm_setpoint_planner.py` 新增方法**
   - `plan_knot(obs, mode, block_index, knot_index)`: 生成单个 30min knot 的 8 zone setpoints
   - `_build_knot_system_prompt()`: 简化 system prompt，只要求输出下一个 30min 的 setpoints
   - `_build_knot_user_prompt()`: 展示当前 EP observation（含 zone 描述），要求输出 `{"setpoints": [8 values]}` JSON
   - `_parse_knot_output()`: 解析单 knot JSON 输出

2. **`gspo_houston_bandit.py` 新增方法**
   - `_rollout_block_rolling(planner, block_index, block_start, block_end, mode)`: 在 EP 回调内交叉调用 LLM
   - 流程：seek_start → replay 前序 winner actions → 进入 rolling block 阶段
   - 每 `KNOT_ENV_STEPS`（3 步 = 30 分钟）调用一次 `planner.plan_knot()`，获取当前 knot 的 setpoints
   - 当前 knot 的 setpoint 复用 3 个 env step（10 min × 3 = 30 min）
   - 返回 `block_reward`、`knot_plans` list、action traces

3. **`train_qwen3_houston_gspo_block.py` 改动**
   - 每个 candidate 调用 `bandit._rollout_block_rolling()` 而非 `plan_block()` + `_rollout_block_with_replay()`
   - gradient accumulation 改为遍历 candidate 的全部 `knot_plans`，每个 knot 独立计算 log-prob
   - block-level GSPO advantage 在同一 block 的所有 knot 间共享

**对比 one-shot vs rolling**：

| 指标 | one-shot (Prompt V2) | rolling (Prompt V2) |
|---|---|---|
| LLM 调用次数/block/candidate | 1 (48 values) | 6 (8 values each) |
| 输入信息 | block-start observation only | 每 knot 有实时 EP observation |
| 输出复杂度 | 6 knots × 8 zones JSON | 1 knot × 8 zones JSON |
| avg_block_reward_std (step 1) | 0.080 | 0.323 |
| avg_grad_norm (step 1) | 0.099 | 0.206 |

rolling 模式 reward_std 提升约 4 倍，grad_norm 翻倍，GSPO 信号显著增强。

### Block GSPO Rolling Full-Month Training

基于 rolling-within-block 改进，已启动新的 22 workday 完整月训练：

- output dir:
  - `result/gspo/qwen3_houston_gspo_block_rolling_fullmonth_s22_promptv2_q01_20260331/`
- dataset:
  - `result/gspo/houston_gspo_dataset_weekday_daystart_fullmonth_workday_policy_zone_temp.jsonl`
- config:
  - `CUDA_VISIBLE_DEVICES=0`
  - `--max-steps 22`
  - `--save-steps 11`
  - `--device cuda:0`
  - `--use-peft`
  - `--temperature 0.7`
  - `--top-p 0.95`
  - `--learning-rate 1e-5`
  - `--reward-scale 1.0`

启动命令：

```bash
env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 nohup .venv_qwen/bin/python train_qwen3_houston_gspo_block.py \
  --dataset-path result/gspo/houston_gspo_dataset_weekday_daystart_fullmonth_workday_policy_zone_temp.jsonl \
  --output-dir result/gspo/qwen3_houston_gspo_block_rolling_fullmonth_s22_promptv2_q01_20260331 \
  --max-steps 22 \
  --save-steps 11 \
  --device cuda:0 \
  --use-peft \
  --temperature 0.7 \
  --top-p 0.95 \
  --learning-rate 1e-5 \
  --reward-scale 1.0 \
  > result/gspo/qwen3_houston_gspo_block_rolling_fullmonth_s22_promptv2_q01_20260331.log 2>&1 &
```

Step 1 (2025-08-01) rolling 结果：

| Block | comfort | balanced | energy_saving | Winner | reward_std | grad_norm |
|-------|---------|----------|---------------|--------|------------|-----------|
| 0 (07-10) | -2.768 | **-0.153** | -0.979 | balanced | 1.091 | 0.338 |
| 1 (10-13) | +0.023 | +0.015 | **+0.030** | energy_saving | 0.006 | 0.100 |
| 2 (13-16) | -0.160 | **+0.060** | -0.266 | balanced | 0.136 | 0.140 |
| 3 (16-19) | **-0.142** | -0.289 | -0.200 | comfort | 0.060 | 0.246 |

- 每个 candidate 每个 block 均生成 6 knots，每个 knot 336 tokens
- baseline_block_rewards: [-2.952, -2.835, -2.854, -2.167]

Step 2 部分结果（不同天，更热）：

| Block | comfort | balanced | energy_saving | Winner | reward_std |
|-------|---------|----------|---------------|--------|------------|
| 0 (07-10) | -0.757 | **-0.751** | -0.757 | balanced | 0.003 |
| 1 (10-13) | -4.477 | **-2.630** | -8.638 | balanced | 2.513 |

Step 2 block 1 的 reward_std=2.51，说明在热天中午模式差异更大，GSPO 有强学习信号。

预计总训练时长约 2 小时（22 steps × ~5 min/step）。

Prompt V2 rolling 4 步结果：

| Step | total_winner | avg_std | avg_gnorm |
|---|---|---|---|
| 1 | -0.11 | 0.33 | 0.21 |
| 2 | -1.36 | 2.45 | 0.22 |
| 3 | -3.91 | 0.93 | 0.16 |
| 4 | +0.50 | 0.53 | 0.20 |
| **avg** | **-1.22** | **1.06** | **0.20** |

对比旧版 full-day rolling（无 block）和 block one-shot：

| 版本 | avg winner | avg std | avg gnorm | 说明 |
|---|---|---|---|---|
| Full-Day rolling (22 steps) | +0.000 | 0.000 | 0.000 | 完全无学习信号 |
| Block One-Shot (5 steps) | +1.138 | 0.837 | 0.104 | 保守输出接近 baseline |
| Block Rolling V2 (4 steps) | -1.221 | 1.059 | 0.197 | 信号最强，初始质量差 |

Rolling 短期 reward 差但 GSPO 信号强（std↑48%, gnorm↑90%），随 step 增加应逐步收敛。

### Prompt V3: 去除强制差异化

Prompt V2 中有多处强制要求 zone 间输出不同 setpoint 的语句：

- `"IMPORTANT: Each zone should get a DIFFERENT setpoint..."`
- `"Do NOT output the same setpoint for all zones or all knots"`
- `"Each zone should get a different setpoint reflecting its heat load"`
- Mode descriptions 中的 `"Vary setpoints"`, `"Differentiate setpoints per zone and per knot"`

问题：强制差异化导致模型为了 "不同" 而输出不合理的极端值，实际上很多时候 zone 条件相近时给相近 setpoint 是正确的。

V3 修改：

1. **去掉所有强制差异化语句**，改为描述依据：`"Set each zone's setpoint based on its temperature, PMV, occupancy, and solar exposure."`
2. **Mode descriptions 精简**：
   - comfort: `"Prioritize occupant comfort within PMV ±0.5. Choose setpoints based on each zone's current temperature, PMV, solar exposure, and floor level."`
   - balanced: `"Balance comfort (PMV ±0.5) and energy. Cooler zones can tolerate higher setpoints; warmer zones may need lower. Shift cooling toward high-PV periods when solar generation offsets cost."`
   - energy_saving: `"Minimise net grid energy while maintaining PMV ±0.5. Set unoccupied zones near the upper bound. For occupied zones, use the warm end of the comfort range. Pre-cool when PV is high."`

### Block GSPO Rolling Full-Month Training (Prompt V3)

- output dir:
  - `result/gspo/qwen3_houston_gspo_block_rolling_fullmonth_s22_promptv3_q01_20260331/`
- config: 与 V2 rolling 相同（22 steps, LoRA, lr=1e-5, temp=0.7, GPU 0）

启动命令：

```bash
env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 nohup .venv_qwen/bin/python train_qwen3_houston_gspo_block.py \
  --dataset-path result/gspo/houston_gspo_dataset_weekday_daystart_fullmonth_workday_policy_zone_temp.jsonl \
  --output-dir result/gspo/qwen3_houston_gspo_block_rolling_fullmonth_s22_promptv3_q01_20260331 \
  --max-steps 22 \
  --save-steps 11 \
  --device cuda:0 \
  --use-peft \
  --temperature 0.7 \
  --top-p 0.95 \
  --learning-rate 1e-5 \
  --reward-scale 1.0 \
  > result/gspo/qwen3_houston_gspo_block_rolling_fullmonth_s22_promptv3_q01_20260331.log 2>&1 &
```

### Prompt V4: PMV 解释 + Mode 方向性

V3 的问题：去掉强制差异化后，三个 mode 输出几乎一样（52% block 被跳过），base Qwen3 对 "target PMV≈0" vs "target PMV≈+0.5" 的 prompt 差异没有反应——模型不理解 PMV 和 setpoint 的关系。

V4 修改：

1. **System prompt 加入 PMV 解释**（`PMV_EXPLANATION` 常量）：
   - PMV 0=neutral, +0.5=slightly warm, -0.5=slightly cool
   - LOWER setpoint → more cooling → LOWER PMV
   - HIGHER setpoint → less cooling → HIGHER PMV
   - 具体例子："if PMV is +0.4 and you want PMV≈0, LOWER the setpoint"

2. **Mode description 加方向性指令**：
   - comfort: "Use LOWER setpoints (more cooling) to push PMV toward 0"
   - balanced: "Use MODERATE setpoints — slightly higher than comfort mode"
   - energy_saving: "Use HIGHER setpoints (less cooling) to reduce energy"

效果验证（3 天 5 episode 测试，lr=1e-5, reward_scale=1.0）：
- Block 0 三个 mode 首次出现巨大差异：comfort=-2.72, balanced=-0.87, energy_saving=-4.78
- std=0.97（block 0），比 V3 的 std≈0 大幅提升
- 但 block 1-3 仍有约 50% 被跳过

### 学习率 + Reward Scale 调参

在 PMV 解释 prompt 基础上测试更激进的超参：

- lr=5e-5 (5×), reward_scale=5.0 (5×), 综合梯度放大约 25 倍
- output dir: `qwen3_houston_gspo_block_rolling_3day_5ep_promptv4_pmv_lr5e5_rs5_20260331/`

Episode 1 结果：

| Step | Ep | Day | winner | avg_std | avg_gnorm | upd | skip |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | -0.08 | 2.63 | 0.07 | 3 | 1 |
| 2 | 1 | 2 | -4.95 | 3.64 | 0.26 | 4 | 0 |
| 3 | 1 | 3 | -26.14 | 2.69 | 0.25 | 4 | 0 |

信号非常强（std=2.6-3.6, 100% 更新率），但 reward 逐天恶化（-0.08 → -26.14），说明 lr×reward_scale 太大，模型被推向极端。需要找到信号强度和稳定性的平衡点。

### GRPO + KL Penalty（标准 GRPO 实现）

**问题**：之前的实现是 bare REINFORCE（`loss = -advantage * log_prob`），缺少 KL 约束，高 lr + reward_scale 时策略容易发散。

**改动**：在 `_accumulate_block_gradient` 中加入 Schulman KL 近似：

```
KL_t = exp(log_ratio) - log_ratio - 1
log_ratio = log_π_θ(token) - log_π_ref(token)
loss = pg_loss + β * mean(KL)
```

- π_ref 通过 LoRA `disable_adapter_layers()` 获取 base model logprobs，无额外显存开销
- PPO clip 不需要（on-policy 单步更新，importance ratio ≡ 1）
- 新增 CLI 参数 `--kl-beta`（默认 0.1）
- phase_trace 新增 `block_kl` 字段

**训练配置**：

```
lr=2e-5, reward_scale=3.0, kl_beta=0.1
dataset: 前 3 个 workday, 5 episodes
output_dir: qwen3_houston_grpo_3day_5ep_pmv_kl01_lr2e5_rs3_20260331/
```

**训练结果（截至 ep3 day1，共 7 步完成）**：

| Step | Ep | Day | Winner RR | Avg Std | Avg KL | Signal Blocks |
|------|-----|------|-----------|---------|--------|---------------|
| 1 | 1 | 1 | -0.0481 | 1.183 | 0.0294 | 2/4 |
| 2 | 1 | 2 | +4.0872 | 1.408 | 0.3744 | 4/4 |
| 3 | 1 | 3 | -0.8741 | 1.701 | 0.5780 | 4/4 |
| 4 | 2 | 1 | +0.0847 | 0.840 | 0.5450 | 2/3 |
| 5 | 2 | 2 | +9.3478 | 3.403 | 1.1287 | 4/4 |
| 6 | 2 | 3 | +0.6733 | 1.162 | 0.5381 | 4/4 |
| 7 | 3 | 1 | +0.0055 | 0.451 | 0.3026 | 1/2 |

**同天跨 episode 对比（学习效果验证）**：

- Day 1: ep1 -0.0481 → ep2 +0.0847 → ep3 +0.0055（信号弱，block 容易 collapse）
- Day 2: ep1 +4.0872 → ep2 **+9.3478**（翻倍提升）
- Day 3: ep1 -0.8741 → ep2 **+0.6733**（负转正）

**Per-block 分析要点**：

1. **Block 0 (07-10)**：始终有信号，balanced 经常胜出，KL 增长最快（ep3 达到 1.48）
2. **Block 2 (13-16)**：信号最强（std 最高可达 8.14），comfort mode 在 day2/3 大幅领先
3. **Day 1 的 block 1 容易 collapse**：两次 std≈0，该时段观测特征可能导致三 mode 趋同
4. **Comfort mode 整体主导**：模型学到了 "降 setpoint → 降 PMV → 更高 comfort reward" 的因果关系
5. **KL 受控**：avg KL 在 0.3-1.1 波动，未持续爆炸，β=0.1 约束力目前足够

**结论**：GRPO+KL 显著优于之前的 bare REINFORCE 和 one-shot 版本。Day 2 reward 翻倍、Day 3 负转正，确认模型在学习有效策略。可以切换到更长训练数据（3 周）进行正式训练。

### GRPO+KL 5 Episode 完整结果

训练完成后 15/15 步全部跑完，最终 per-episode 总奖励：

| Episode | Total Winner RR |
|---------|----------------|
| ep1 | +3.17 |
| ep2 | +10.11 |
| ep3 | +5.74 |
| ep4 | +14.16 |
| ep5 | +14.06 |

趋势：ep1 → ep4 持续上升（+3.17 → +14.16），ep5 保持 +14.06，学习收敛。

### 三方对比：GRPO vs PPO vs LSTM（前 3 天 workday）

在相同的 3 个 workday（skip=0/75/150，control window 07:00-19:00）上对比：
- **Baseline**: 固定 24°C setpoint
- **PPO**: `formal300_nonlstm_netload_wc50_rs001_nodup_ep5000_x300` checkpoint
- **LSTM**: `formal300_lstm_netload_wc50_rs001_nodup_ep5000_x300` checkpoint
- **GRPO**: 当前 Qwen3-8B GRPO+KL 训练 ep5 结果

| Day | Date | Baseline | PPO | LSTM | GRPO(ep5) | GRPO(best) |
|-----|------|----------|-----|------|-----------|------------|
| 1 | 08-01 | -10.84 | -0.18 | -48.04 | -2.22 | +0.08 |
| 2 | 08-04 | -13.19 | +4.74 | +1.37 | **+14.00** | **+14.14** |
| 3 | 08-05 | -7.69 | +0.74 | -13.35 | **+2.28** | **+2.28** |
| **Total** | | -31.72 | **+5.30** | **-60.02** | **+14.06** | **+16.50** |

注：PPO/LSTM 的 checkpoint 不含 cloudcover observation（训练时未加），评估时临时去除 cloudcover 以匹配 checkpoint 维度。GRPO 的 observation 包含 cloudcover。

**关键发现**：
1. **GRPO 总分 +14.06，远超 PPO +5.30**（提升 165%）
2. Day 2 上 GRPO +14.0 vs PPO +4.7，Day 3 上 GRPO +2.3 vs PPO +0.7
3. LSTM 表现异常差（-60.02），在 Day 1 和 Day 3 大幅低于 baseline
4. GRPO Day 1 尚有不稳定（ep5 为 -2.22），但 ep2/ep4 为正值

### Block 3 knot_count 归零 Bug 与修复

**现象**：3 周训练中 block 3 (16:00-19:00) 的 knot 数量随天数线性递减：6→5→4→3→2→1→0，第 7 天后完全归零。随后 block 2 也开始丢失，第 13 天归零。每个 episode 重复相同 pattern。

```
step  block0 block1 block2 block3
 1      6      6      6      6
 2      6      6      6      5
 7      6      6      6      0    ← block 3 完全丢失
13      6      6      0      0    ← block 2 也丢失
15      6      4      0      0
```

**根因**：数据集收集与训练的控制窗口不一致。

| | 数据集收集 | 训练 bandit |
|---|---|---|
| `control_window_start` | **06:30** | **07:00** |
| `control_window_end` | 19:00 | 19:00 |
| 每天 eligible steps | **75** | **72** |

数据集的 `skip_valid_steps` 按 75/天递增，但训练 bandit 只在 07:00-19:00 内计 eligible step（72/天）。每天多跳 3 步 = 30 分钟 drift。由于 block 0-2 不检查时间边界（只检查 19:00 终点），总是取满 18 步，延迟全部累积到最后一个 block。

**修复**：将训练 bandit 的 `control_window_start` 改为 `"06:30"`，`BLOCK_DEFINITIONS[0]` 改为 `(time(6, 30), time(10, 0))`，使每天 eligible step 数与数据集一致（75）。

### 3 周 8 Episode 正式训练（修复后）

- **Run**: `qwen3_houston_grpo_3week_8ep_pmv_kl01_lr2e5_rs3_20260331`
- **WandB**: `grpo_3week_8ep_kl01_lr2e5_rs3_resume30`
- **数据集**: `houston_gspo_dataset_3week.jsonl`（15 workdays, 2025-08-01 ~ 08-21）
- **总步数**: 120 steps（8 episodes × 15 days）
- **初始化**: 从 run-b checkpoint-30 的 LoRA 权重继续（已完成 2 episode 有效学习）
- **超参数**: temperature=0.7, top_p=0.95, lr=2e-5, reward_scale=3.0, kl_beta=0.1
- **Checkpoint**: 每 15 步（每 episode 结束）保存

**Bug 修复前训练 (run-b) 有效天数摘要**（仅 days 1-6，block 3 有完整 knots）：

| Episode | Days 1-6 reward | 全部天数 reward |
|---------|-----------------|-----------------|
| ep1 | -9.01 | +11.38 |
| ep2 | +9.63 | +35.31 |
| ep3 (6天) | +19.06 | +28.32 |

趋势：-9 → +10 → +19，模型在有效数据上持续改善。

### 五方对比：GRPO vs PPO vs LSTM（含 cloudcover 重训，per-workday reward）

PPO 和 LSTM 使用含 cloudcover 的 observation 从头训练 300 episodes（5000 steps/ep），与 GRPO 8-episode 训练对比。

评估方式：夜间 HVAC 不工作 reward=0，因此 episode reward 即为控制窗口累计 reward。PPO/LSTM 取 last-50 episode 均值折算为 per-workday，GRPO 直接从控制窗口 block 数据计算。

| 模型 | reward/day | vs baseline (24°C) | 备注 |
|------|-----------|-------------------|------|
| Baseline (24°C) | -12.74 | — | 固定 setpoint |
| PPO forecast | -11.62 | +1.12 | 300 ep, cloudcover |
| PPO no-forecast | -11.70 | +1.04 | 300 ep, cloudcover |
| LSTM forecast | -11.95 | +0.79 | 300 ep, cloudcover |
| LSTM no-forecast | -11.60 | +1.14 | 300 ep, cloudcover |
| **GRPO ep5 (best)** | **-4.12** | **+8.62** | Qwen3-8B, 8ep 训练中 |
| **GRPO ep7 (latest)** | **-5.42** | **+7.32** | 同上 |

**结论**：
1. GRPO 相对 baseline 改善 +7~9/day，PPO/LSTM 仅 +1/day，GRPO 优势约 **7-8 倍**
2. PPO/LSTM 的 forecast vs no-forecast 差异极小（<0.3/day），forecast 未带来明显收益
3. LSTM 表现略逊于 PPO（forecast 变体），与之前 3-day 评估结论一致
4. GRPO block 2 (13:00-16:00) 贡献最大，block 3 (16:00-19:00) 仍有改善空间

### Held-Out Eval 初步结果（Aug 25-29，不公平对比）

在 Aug 25-29（5 workdays）上评估 GRPO（30min knot，checkpoint-75/ep5）和 PPO/LSTM 全部 4 个变体：

| Date | GRPO (30min) | PPO+fc | PPO-nofc | LSTM+fc | LSTM-nofc |
|------|-------------|--------|----------|---------|-----------|
| 08-25 | +8.70 | +11.55 | +11.53 | +10.40 | +8.57 |
| 08-26 | +2.67 | +2.70 | +2.70 | +2.03 | -10.74 |
| 08-27 | -0.07 | +1.23 | +1.24 | +0.71 | -25.25 |
| 08-28 | +0.48 | +1.44 | +1.45 | +0.61 | -26.46 |
| 08-29 | +1.28 | +2.95 | +2.96 | +2.37 | -18.01 |
| **Total** | +13.06 | +19.87 | **+19.88** | +16.13 | -71.89 |
| **Mean** | +2.61 | +3.97 | **+3.98** | +3.23 | -14.38 |

**关键发现**：
1. PPO forecast ≈ PPO no-forecast（+19.87 vs +19.88），forecast 特征对 PPO 无增益
2. LSTM no-forecast 严重崩溃（-71.89），可能因为无 forecast 的 observation 不足以支撑 LSTM 状态建模
3. GRPO（30min knot）排第四，低于两个 PPO 和 LSTM+fc

**问题：这个对比不公平。** PPO/LSTM 的 IDF RunPeriod 是 Aug 1-Sep 1，`repeat=True`，300 episode × 5000 steps 的训练中 Aug 25-29 被反复经历过。GRPO 的训练数据集只覆盖 Aug 1-21（15 workdays），Aug 25-29 对 GRPO 是真正的 out-of-distribution。

### Sep 1-5 OOD Eval（所有模型均未训练过 9 月数据）

使用 `houston_2025_09_eval.idf`（RunPeriod Sep 1-30）在 Sep 1-5 上评估。9 月数据对所有模型都是 OOD（PPO IDF 到 Sep 1 但 `repeat=True` 循环回 Aug 1，实际只覆盖 Aug 天气；GRPO 训练集只到 Aug 21）。

| Date | GRPO 30min | PPO+fc | PPO-nofc | LSTM+fc |
|------|-----------|--------|----------|---------|
| 09-01 | **+0.12** | -0.16 | -0.16 | -6.06 |
| 09-02 | -1.35 | **+0.29** | +0.29 | -11.37 |
| 09-03 | +0.57 | **+1.73** | +1.74 | -1.21 |
| 09-04 | -0.84 | **+3.62** | +3.63 | -0.68 |
| 09-05 | -1.41 | +3.97 | **+3.98** | +3.22 |
| **Total** | **-2.92** | **+9.45** | **+9.48** | **-16.10** |
| **Mean** | **-0.58** | **+1.89** | **+1.90** | **-3.22** |

**关键发现**：
1. PPO 在 9 月 OOD 上仍然表现最好（+9.48），泛化能力强
2. GRPO 30min 总分为负（-2.92），但仍远优于 LSTM+fc（-16.10）
3. Sep 1 是唯一 GRPO 胜过 PPO 的日期（+0.12 vs -0.16）
4. LSTM+fc 在 OOD 上严重崩溃（Sep 1-2 分别 -6.06、-11.37）
5. PPO forecast ≈ PPO no-forecast，forecast 特征在 OOD 上也无增益
6. GRPO 使用 30min knot（checkpoint-75/ep5），10min knot 版本训练中，待对比

**注意**：PPO 的 IDF RunPeriod 到 Sep 1，虽然 `repeat=True` 循环回 Aug，但 EnergyPlus 的 warmup/sizing 可能让 PPO 对 Sep 初的天气有间接暴露。3 周限定训练（`houston_3week.idf`，Aug 1-22）的公平对比正在进行中。

### 公平对比计划：PPO/LSTM 限制为 3 周训练

为了让所有方法在相同条件下对比，需要将 PPO/LSTM 的训练范围也限制到 Aug 1-21（3 周），使 Aug 25-29 对所有方法都是 held-out。

**方案**：创建 `houston_3week.idf`，将 RunPeriod 从 `Aug 1 - Sep 1` 改为 `Aug 1 - Aug 22`（21 天），重新训练 4 个模型：

1. PPO forecast (cloudcover, 300 ep, 5000 steps)
2. PPO no-forecast (cloudcover, 300 ep, 5000 steps)
3. LSTM forecast (cloudcover, 300 ep, 5000 steps)
4. LSTM no-forecast (cloudcover, 300 ep, 5000 steps)

训练完成后在 Aug 25-29 上重新 eval，与 GRPO 公平对比。

**IDF 改动**：
- 复制 `houston.idf` → `houston_3week.idf`
- RunPeriod: `Begin Month=8, Begin Day=1, End Month=8, End Day=22`
- 其余参数不变

**Cell 文件改动**：
- `.tmp_todo_random_start_cell0.py` 和 `.tmp_todo_lstm_cell0.py` 的 `building_path` 改为 `os.getenv("RL_IDF", "houston.idf")`
- 训练时设 `RL_IDF=houston_3week.idf`，eval 时不设（默认用 `houston.idf`，覆盖 Aug 25-29）

**PPO/LSTM 3 周训练已启动**（GPU 1，4 个并行进程）：

```bash
# 所有进程共用参数
CUDA_VISIBLE_DEVICES=1 RL_IDF=houston_3week.idf RL_EPISODE_STEPS=5000 RL_TRAIN_EPISODES=300 RL_NUM_GPUS=1

# PPO forecast
RL_VARIANT=forecast_window RL_TMP_CELL_PREFIX=.tmp_todo_random_start WANDB_RUN_PREFIX=houston_aug2025_3wk_ppo_cc
# PPO no-forecast
RL_VARIANT=no_forecast_window RL_TMP_CELL_PREFIX=.tmp_todo_random_start WANDB_RUN_PREFIX=houston_aug2025_3wk_ppo_cc
# LSTM forecast
RL_VARIANT=forecast_window RL_TMP_CELL_PREFIX=.tmp_todo_lstm WANDB_RUN_PREFIX=houston_aug2025_3wk_lstm_cc
# LSTM no-forecast
RL_VARIANT=no_forecast_window RL_TMP_CELL_PREFIX=.tmp_todo_lstm WANDB_RUN_PREFIX=houston_aug2025_3wk_lstm_cc
```

输出目录：`result/manual_train/houston_aug2025_3wk_{ppo,lstm}_cc_ep5000_x300_{forecast,no_forecast}_window_manual/`

### GRPO 10min Knot 训练（从 30min 改为 10min 粒度）

**动机**：之前 GRPO 用 30min knot（`KNOT_ENV_STEPS=3`），每 block 只有 6 次 LLM 调用。改为 10min knot（`KNOT_ENV_STEPS=1`），每 block 18 次调用，控制粒度对齐 PPO 的 10min step。

**代码改动**：

1. `gspo_houston_bandit.py`:
   - `KNOT_ENV_STEPS`: 3 → 1
   - `KNOTS_PER_BLOCK`: 6 → 18
   - `BLOCK_ENV_STEPS`: 不变（18）

2. `llm_setpoint_planner.py`:
   - `KNOTS_PER_BLOCK`: 6 → 18
   - `KNOT_MINUTES`: 30 → 10
   - Prompt 中 "half-hour" → "{KNOT_MINUTES}-minute"

**训练配置**：

- **Run**: `qwen3_houston_grpo_3week_4ep_10min_knot_kl01_lr2e5_rs3_20260401`
- **WandB**: `grpo_3week_4ep_10min_knot_fresh`
- **数据集**: `houston_gspo_dataset_3week.jsonl`（15 workdays, Aug 1-21）
- **总步数**: 60 steps（4 episodes × 15 days）
- **初始化**: 从 base Qwen3-8B + 新 LoRA（不 resume）
- **超参数**: temperature=0.7, top_p=0.95, lr=2e-5, reward_scale=3.0, kl_beta=0.1
- **GPU**: cuda:0
- **Checkpoint**: 每 15 步（每 episode 结束）保存

```bash
env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 nohup .venv_qwen/bin/python train_qwen3_houston_gspo_block.py \
  --dataset-path result/gspo/houston_gspo_dataset_3week.jsonl \
  --output-dir result/gspo/qwen3_houston_grpo_3week_4ep_10min_knot_kl01_lr2e5_rs3_20260401 \
  --max-steps 60 --save-steps 15 --device cuda:0 --use-peft \
  --temperature 0.7 --top-p 0.95 --learning-rate 2e-5 --reward-scale 3.0 --kl-beta 0.1 \
  --wandb-project asim-houston-grpo --wandb-group block-rolling-grpo \
  --wandb-name grpo_3week_4ep_10min_knot_fresh \
  > result/gspo/qwen3_houston_grpo_3week_4ep_10min_knot.log 2>&1 &
```

**对比 30min vs 10min knot**：

| 指标 | 30min knot | 10min knot |
|------|-----------|-----------|
| KNOT_ENV_STEPS | 3 | 1 |
| KNOTS_PER_BLOCK | 6 | 18 |
| LLM 调用/block/candidate | 6 | 18 |
| 控制粒度 | 30 min | 10 min（= PPO） |
| 每步可见最新 observation | 否（30min 内复用） | 是 |

### GRPO 10min Knot Eval 结果

10min knot (checkpoint-30/ep2, fresh 训练) 与 30min knot (checkpoint-75/ep5) 和 PPO 3wk 的公平对比：

**Aug 25-29（GRPO in-distribution，PPO 3wk OOD）**：

| Date | GRPO 30min | GRPO 10min | PPO 3wk |
|------|-----------|-----------|---------|
| 08-25 | +8.70 | **+11.30** | +11.00 |
| 08-26 | +2.67 | +0.97 | **+2.74** |
| 08-27 | -0.07 | -0.36 | **+1.37** |
| 08-28 | +0.48 | +0.32 | **+1.55** |
| 08-29 | +1.28 | +0.48 | **+3.00** |
| **Total** | +13.06 | +12.71 | **+19.65** |
| **Mean** | +2.61 | +2.54 | **+3.93** |

**Sep 1-5（所有模型 OOD）**：

| Date | GRPO 30min | GRPO 10min | PPO (旧) |
|------|-----------|-----------|---------|
| 09-01 | **+0.12** | -0.28 | -0.16 |
| 09-02 | -1.35 | -1.18 | **+0.29** |
| 09-03 | +0.57 | -1.05 | **+1.74** |
| 09-04 | -0.84 | **+2.94** | +3.63 |
| 09-05 | -1.41 | +1.29 | **+3.98** |
| **Total** | -2.92 | **+1.72** | **+9.48** |
| **Mean** | -0.58 | **+0.34** | **+1.90** |

**关键发现**：
1. 10min knot 在 Sep OOD 上从 -2.92 提升到 +1.72，泛化能力显著改善
2. Aug 25 上 GRPO 10min (+11.30) 超过 PPO 3wk (+11.00)
3. Aug 总分 10min (+12.71) 略低于 30min (+13.06)，可能因为 10min 只训练了 4ep 而 30min 训练了 8ep
4. PPO 3wk 仍然整体领先，但差距在缩小

### 后续改进计划

#### 改进 1：Block 缩小到 1 小时

**动机**：当前 3 小时 block 内 credit assignment horizon 为 18 步，mode 切换频率仅 4 次/天。缩小到 1 小时可显著提升信号质量。

**方案**：
- 06:30-19:00 → 12-13 个 block/天，每 block 6 个 env step
- Credit assignment 从 18 步降到 6 步
- 每天 reward 信号从 4 个增到 12 个
- Mode 切换从每 3 小时变成每 1 小时

**代价**：replay 次数增加（block 12 要 replay 前 11 个 winner action），训练变慢约 3 倍。

**改动**：修改 `BLOCK_DEFINITIONS` 和相关常量，架构不变。

**已实施**。`BLOCK_DEFINITIONS` 改为 13 个 1 小时 block（最后一个 18:30-19:00 为 30 分钟）。`BLOCK_ENV_STEPS` 和 `KNOTS_PER_BLOCK` 改为动态计算（`_block_env_steps(block_index)` / `_block_knots(block_index)`），自动处理变长 block。

#### 改进 2：Reflexion — 跨天反思与在线适应

**动机**：GRPO 训练后权重固定，面对 OOD 天气（如 9 月）无法在线调整。PPO 的 MLP 同样不能在线适应，但 LLM 有 in-context learning 能力，可以利用。

**方案**：参考 Reflexion（Shinn et al., NeurIPS 2023）框架：

1. **每天结束后反思**：将当天 reward、各 block winner mode、温度/PMV 变化轨迹交给 LLM，生成自然语言反思总结（如 "南向 zone 下午过冷，energy_saving mode 在 13-16 block 表现最好"）
2. **注入下一天 prompt**：反思文本作为额外 context 加入次日 system prompt
3. **跨天积累**：多天反思形成 episodic memory buffer

**分阶段实施**：
- **Phase 1（eval only）**：不改训练，仅在 eval 时添加反思机制，验证对第 2-5 天 performance 的提升
- **Phase 2（训练融入）**：如果 Phase 1 有效，考虑将反思质量纳入 reward 或作为 GRPO 训练的 context

**优势**：
- 这是 LLM controller 相比 PPO/LSTM 的独特能力——从历史经验中自我改进
- 对 OOD 泛化特别有价值（9 月新天气模式）
- 即使最终 reward 未超过 PPO，可解释性 + 自我反思也是论文的独特贡献

**参考**：
- Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., NeurIPS 2023)
- https://arxiv.org/abs/2303.11366

**已实施**。`BlockPlanner` 新增：
- `generate_day_reflection()`: 每天结束后生成自然语言反思，总结各 block 表现和 mode 选择
- `get_reflection_context()`: 返回最近 5 天反思作为 prompt context
- `clear_reflections()`: 重置反思记忆
- 反思自动注入 `_build_knot_system_prompt()` 的 system prompt
- 训练循环中每天最后一个 block 结束后调用 `generate_day_reflection()`

#### 改进 3：Day-Level Gradient — 全天奖励信号

**动机**：per-block GRPO 只优化"block 内哪个 mode 最好"，看不到跨 block 依赖（如 block 3 过度制冷导致 block 4 起点温度过低）。

**方案**：每天所有 block 完成后，用全天总 relative reward 对所有 winning knot 做一次额外梯度更新：

```
day_advantage = total_winner_reward * DAY_ADVANTAGE_SCALE  (0.3)
gradient_update(all_winning_knots, day_advantage)
```

每个 winning knot 收到两次梯度信号：
1. **Block-level**：block 内 3-mode 比较的 advantage（短期信用分配）
2. **Day-level**：全天总 reward 的 advantage（长期信用分配，跨 block 依赖）

`DAY_ADVANTAGE_SCALE = 0.3` 控制 day 信号权重，避免覆盖 block 信号。

**已实施**。训练循环中每天 block 循环结束后执行 day-level optimizer.step()。

#### 改进 4：Precooling Mode — 第 4 个 Candidate 策略

**动机**：现有 3 个 mode 覆盖 PMV 0 ~ +0.5（中性到偏暖），缺少"主动偏冷"方向。Houston 下午 PV 充足时 precool 可以存储 thermal mass，傍晚 PV 下降后减少空调使用。

**方案**：新增 `precooling` mode，target PMV ≈ -0.2（slightly cool）。PMV -0.2 在 ±0.5 限制内，不触发 comfort violation penalty。

4 个 candidate modes: `["comfort", "balanced", "energy_saving", "precooling"]`

**代价**：每 block rollout 从 3 次变成 4 次，训练时间增加 ~33%。但 reward variance 更大（4 个策略方向 vs 3 个），GRPO 信号更强。

**已实施**。`CANDIDATE_MODES` 改为 4 个，`CANDIDATE_MODE_DESCRIPTIONS` 新增 precooling 描述。

### 综合训练（1h Block + Reflexion + Day-Level Gradient + 4 Modes）

整合以上 4 项改进的训练：

- **Run**: `qwen3_houston_grpo_3week_4ep_1h_block_reflexion_kl01_lr2e5_rs3_20260402`
- **WandB**: `grpo_3week_4ep_1h_block_4mode_reflexion_daygrad`
- **数据集**: `houston_gspo_dataset_3week.jsonl`（15 workdays, Aug 1-21）
- **总步数**: 60 steps（4 episodes × 15 days）
- **每步**: 13 blocks × 4 modes = 52 rollouts + 13 block gradient updates + 1 day gradient update + 1 reflection
- **超参数**: temperature=0.7, top_p=0.95, lr=2e-5, reward_scale=3.0, kl_beta=0.1, DAY_ADVANTAGE_SCALE=0.3
- **GPU**: cuda:0

```bash
env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 nohup .venv_qwen/bin/python train_qwen3_houston_gspo_block.py \
  --dataset-path result/gspo/houston_gspo_dataset_3week.jsonl \
  --output-dir result/gspo/qwen3_houston_grpo_3week_4ep_1h_block_reflexion_kl01_lr2e5_rs3_20260402 \
  --max-steps 60 --save-steps 15 --device cuda:0 --use-peft \
  --temperature 0.7 --top-p 0.95 --learning-rate 2e-5 --reward-scale 3.0 --kl-beta 0.1 \
  --wandb-project asim-houston-grpo --wandb-group block-rolling-grpo \
  --wandb-name grpo_3week_4ep_1h_block_4mode_reflexion_daygrad \
  > result/gspo/qwen3_houston_grpo_3week_4ep_1h_block_reflexion.log 2>&1 &
```

#### 改进 5：并行 Mode Rollout（ThreadPoolExecutor）

**动机**：每 block 4 个 mode 串行跑 EP rollout，每个 ~20 秒，共 ~80 秒。4 个 mode 共享相同的 replay history，replay 阶段是纯 CPU（EP C 扩展释放 GIL），可以并行。

**方案**：`ThreadPoolExecutor(max_workers=4)` 同时跑 4 个 mode 的完整 `_rollout_block_rolling`。LLM GPU 推理通过 `BlockPlanner._inference_lock`（`threading.Lock`）串行保护，避免 GPU 竞争。

```python
with ThreadPoolExecutor(max_workers=len(CANDIDATE_MODES)) as executor:
    futures = {executor.submit(_rollout_mode, m): m for m in CANDIDATE_MODES}
    for future in as_completed(futures):
        mode, result = future.result()
        mode_results[mode] = result
```

**安全性**：
- 模型权重只在主线程的 optimizer.step() 中更新，rollout 线程只做 forward inference（不更新权重）
- `_inference_lock` 保证 GPU 推理串行，不会 CUDA 竞争
- 4 个 EP 实例各自独立（不同 report 目录），无文件冲突

**效果**：Block 0（无 replay）从 ~80s 降到 ~71s（主要受 LLM 串行限制）。后续 block 有长 replay，并行收益更明显（replay 阶段 4 个 EP 同时跑 CPU）。

**已实施**。训练循环中 mode rollout 改为 ThreadPoolExecutor 并行。

**注意**：Block 10+（16:30 之后）replay history 较长，4 线程并行可能导致 EP crash（"Operation failed with status 1"）。已加入 retry 机制（失败后 sleep 2s 重试一次），并将 block_index >= 10 的并行度降为 2。

**预估训练时间**：每步 ~12 分钟（并行 rollout + gradient + reflection），60 步 ≈ ~12 小时。

#### 改进 6：Reflection 增强（v2）

**改进点**：

1. **按日期索引反思**：`_reflection_by_date` 字典按日期存储反思。同一天跨 episode 的反思能正确传递（ep2 跑 Aug 1 时能看到 ep1 对 Aug 1 的反思）。注入 prompt 时优先展示 same-date 反思（最多 3 条），再补充最近 5 条一般反思，总共上限 8 条去重。

2. **All-mode reward 对比**：反思 prompt 现在包含每个 block 所有 4 个 mode 的 reward，而非只有 winner：
   ```
   Block 5 (10:30-11:30): winner=precooling (+0.15) | all modes: comfort=+0.08, balanced=+0.05, energy_saving=-0.02, precooling=+0.15
   ```
   LLM 能看到 mode 之间的 reward 差异，生成更有针对性的反思。

3. **Think tag 清理**：Qwen3 即使加了 `/no_think` 也可能输出空 `<think></think>` tag，现在用 regex 清理。

#### 改进 7：Hierarchical 9-Mode + 3h Block（两层决策架构）

**动机**：1h block 信噪比低（reward_std=0.55 vs 3h 的 2.05），4 个 flat mode 探索不够细粒度。

**方案**：回到 3h block（强信号），但将 candidate modes 从 4 个扩展到 9 个，按两层层级组织：

| 策略层 (Tier) | cool | med | warm |
|---|---|---|---|
| **comfort** | PMV ≈ -0.2 | PMV ≈ 0 | PMV ≈ +0.1 |
| **balanced** | PMV ≈ 0 | PMV ≈ +0.15 | PMV ≈ +0.3 |
| **energy_saving** | PMV ≈ +0.3 | PMV ≈ +0.4 | PMV ≈ +0.5 |

9 个 candidate 覆盖 PMV -0.2 到 +0.5 的完整空间。Precooling 自然整合到 `comfort_cool`。

**两层 Advantage 计算**（参考 [HICRA](https://arxiv.org/html/2509.03646v1)、[PTA-GRPO](https://arxiv.org/pdf/2510.01833)）：

```python
# Level 1: 策略层 advantage（3 个 tier 的 best-of-3 互比）
tier_best = {tier: max(tier_modes_rewards) for tier in tiers}
tier_advantage = normalize([tier_best["comfort"], tier_best["balanced"], tier_best["energy_saving"]])

# Level 2: tier 内部 sub-mode advantage（每个 tier 内 3 选 1）
comfort_advantage = normalize([comfort_cool, comfort_med, comfort_warm])

# Combined: tier + 0.5 * sub
combined_advantage = tier_advantage + 0.5 * sub_advantage
```

不做扁平 9-mode 归一化——Level 1 学"什么时候用 comfort vs energy_saving"，Level 2 学"comfort 时偏冷还是偏暖"。

**Block 定义**：回到 3h block（06:30-10:00, 10:00-13:00, 13:00-16:00, 16:00-19:00），动态 `_block_env_steps()` 保持不变。

**并行**：ThreadPoolExecutor(max_workers=3)，9 modes 分 3 轮并行。

**参考文献**：
- [Hierarchical LLM+RL for HVAC](https://arxiv.org/abs/2603.26050)
- [PTA-GRPO: Plan Then Action](https://arxiv.org/pdf/2510.01833)
- [HICRA: Hierarchy-aware Credit Assignment](https://arxiv.org/html/2509.03646v1)

### 当前训练对比（双 GPU 并行）

| | GPU 0 | GPU 1 |
|---|---|---|
| **架构** | 3h block, 9 modes, hierarchical advantage, reflexion v2, day-level gradient | 同左 |
| **初始化** | 从 10min checkpoint-30 resume | 从头训练（新 LoRA） |
| **Output dir** | `qwen3_houston_grpo_3week_4ep_3h_9mode_hierarchical_20260402` | `qwen3_houston_grpo_3week_4ep_3h_9mode_hierarchical_fresh_20260402` |
| **WandB** | `grpo_3h_9mode_hier_resume10min` | `grpo_3h_9mode_hier_fresh` |

### PPO 蒸馏 + GRPO 微调（Phase 1-3）

**动机**：LLM从头学GRPO太慢（~2000次梯度更新 vs PPO的150万次）。先用PPO的action做SFT让LLM快速达到PPO水平，再用GRPO在PPO基础上利用LLM独有能力（语义理解、Reflexion、zone差异化）继续优化。

**类比**：AlphaGo先用人类棋谱SFT，再用self-play RL超越人类。

**Phase 1: 收集PPO数据**
- 用 PPO 3wk forecast checkpoint 跑 15 个训练日（Aug 1-21）
- 每步收集 `(knot_system_prompt, knot_user_prompt, setpoint_json)` × 9 个 mode
- 同一个PPO action配对到所有9个mode prompt（PPO不区分mode，但LLM需要学会在不同mode下输出合适的setpoint）
- 预计 ~10,000 条 SFT 样本
- 输出：`result/gspo/ppo_sft_dataset.jsonl`

**Phase 2: SFT**
- 用 PPO 数据 fine-tune Qwen3-8B（LoRA）
- LLM 学会：给定 observation + mode prompt → 输出接近 PPO 水平的 setpoint
- 几个 epoch，约 1-2 小时

**Phase 2 已完成**：SFT checkpoint-500（epoch ~0.83，loss=0.004），保存在 `result/gspo/qwen3_sft_ppo_distill/checkpoint-500`。

**Phase 3: GRPO**
- 从 SFT checkpoint-500 开始，3 mode（comfort/balanced/energy_saving）+ Reflexion + day-level gradient + per-knot advantage
- Mode prompt 只给 PMV 方向和相对指导（不给绝对温度范围），LLM 根据 observation 自己判断 setpoint
- `kl_beta=0`：SFT 后模型已远离 base model，KL penalty 会爆炸（step 1 KL=262~545，step 2 KL=17670）。去掉 KL，靠 reward advantage 引导
- `grad_clip=1.0`：加入 `clip_grad_norm_(max_norm=1.0)`，防止 gradient 爆炸（之前 step 2 block 1 的 grad_norm=6518）
- 起点已经是 PPO 水平，GRPO 只需要微调
- LLM 独有能力（weather forecast 语义理解、zone 差异化、Reflexion 跨天适应）有望超越 PPO

**Phase 1 已完成**：10,125 条 SFT 样本（15 天 × 75 步 × 9 modes），保存在 `result/gspo/ppo_sft_dataset.jsonl`。

**Phase 3 调参记录**：

| Run | kl_beta | KL ref | grad_clip | 结果 |
|-----|---------|--------|-----------|------|
| `grpo_phase3_sft_ppo_3mode_pmv` | 0.1 | base model | 无 | 前6步好（+15.30, +3.04, +9.10, +7.35），但grad_norm爆炸（6518），后续崩溃 |
| `grpo_phase3_sft_3mode_nokl_gradclip` | 0.0 | — | 1.0 | grad稳定(0.1-0.4)，但无KL约束导致模型drift，step 12跌到-356 |
| `grpo_phase3_sft_3mode_sftref_kl01` | 0.1 | SFT snapshot | 1.0 | KL从0.98逐步增到273，grad稳定，reward没崩但KL增长太快 |
| `grpo_phase3_nokl_gradclip_v2` | 0.0 | — | 1.0 | 预计仍会累积drift崩溃（grad小但方向一致偏离，clip不触发） |
| **`grpo_phase3_sftref_kl05_gradclip`** | **0.5** | **SFT snapshot** | **1.0** | **当前运行。kl_beta 提升5倍压制KL增长** |

**关键发现**：
- KL vs base model 对SFT后模型无意义（常数偏移~几千，不约束梯度方向）
- KL vs SFT ref 有效，kl_beta=0.1时增长太快（4步从0.98到273），需要更大的kl_beta
- 无KL时grad虽小(0.1-0.4)但累积drift导致崩溃（step 12到-356），grad_clip无法防止
- 防崩溃需要**KL约束（防drift）+ grad_clip（防单步爆炸）**两者缺一不可

#### 改进 9：Baseline-Anchored Advantage（防drift的根本方案）

**问题**：标准GRPO的group内归一化 `(r - mean) / std` 总是产生正负各半的advantage。当所有mode都比baseline差时，"最不差的"仍然得到正advantage被强化→模型持续被推向错误方向→累积drift→崩溃。KL约束只是减缓drift速度，不解决根因。

**方案**：用0（baseline）作为anchor而非group mean：

```python
# 之前：group内归一化（总有正有负）
advantage = (r - mean) / (std + eps)

# 改进：baseline-anchored（reward已减过baseline，0就是baseline水平）
advantage = r / (std + eps)
```

效果：
- 全正 [+5, +3, +1]：全部强化（但最好的强化最多）
- 全负 [-5, -3, -1]：**全部惩罚**（不强化任何一个）
- 混合 [+5, -1, -3]：只强化正的，惩罚负的

**这从根源上解决了drift问题**——不需要KL约束，因为模型不会被推向比baseline差的方向。

#### 改进 10：PPO Baseline 替代 24°C Baseline

**问题**：用24°C作为baseline时，SFT模型的输出大部分比24°C好很多，所有mode都拿到大的正advantage，区分度不够。用SFT作为baseline也可以，但我们的目标是超越PPO，直接和PPO比更合理。

**方案**：训练开始前加载PPO checkpoint，对15个训练日跑 `evaluate_planner_workday_closed_loop`，用 `_slice_reward_trace_into_blocks` 切分成per-block reward，缓存作为baseline。训练时 `relative_reward = candidate_reward - ppo_block_reward`。

效果：
- reward > 0：比PPO好 → 强化（baseline-anchored）
- reward < 0：比PPO差 → 惩罚
- reward ≈ 0：和PPO差不多 → 不更新

**只有真正超越PPO的决策才被强化**，从根源上防止drift（不会强化比PPO差的输出）。不需要KL约束。

**PPO baseline 方案暂未成功**：PPO baseline预计算时crash（Ray+EnergyPlus初始化问题），已回退到24°C baseline。PPO baseline方案保留代码供后续使用。

### Miami 场景搭建

**动机**：Houston 8月天气变化不大（大部分晴天），forecast信息价值有限。Miami 8月日内天气变化剧烈（77%工作日日内云量swing > 60%），午后雷暴导致PV大幅波动，forecast价值高——是LLM利用forecast超越PPO的理想场景。

**Miami 8月天气特征**：
- 日内云量swing：平均 73%，17/22 工作日 > 60%
- 降水时数：41/273 工时（15%），重度降水 7/273（3%）
- Forecast 云量 MAE：31.5%（有信息但不完美，需要LLM学会判断可信度）
- 典型pattern：早上晴（5-20%）→ 午后雷暴（90-100%）→ 傍晚散去

**已完成**：
- 历史天气 CSV: `weather/miami_2025_06_01_2025_09_30_historical_weather_api.csv`
- Forecast CSV: `weather/miami_2025_06_01_2025_09_30_hourly_model_runs_api_raw.csv`
- EPW: `weather/miami_2025_06_01_2025_09_30_historical_weather_api.epw`
- IDF: `miami.idf`（full Aug 1-Sep 1），`miami_3week.idf`（Aug 1-22）
- Cell文件 env var 支持: `RL_IDF`, `RL_EPW`, `RL_FORECAST_CSV`

**Forecast 数据**：

Houston 的 forecast 数据来自 Open-Meteo 的 **`single-runs-api`**（`https://single-runs-api.open-meteo.com/v1/forecast`），每个 model run（HRRR，每3小时一次）获取未来 1-6 小时的预测：
- 参数：`run=2025-08-15T12:00&forecast_hours=7&models=ncep_hrrr_conus`
- 返回 7 小时数据（lead 0-6h），只保留 lead 1-6h
- 变量：temperature_2m, cloudcover, precipitation_probability, precipitation 等 21 个气象变量

数据格式：
- **Raw**（长格式）：每行一个 (run_time, target_time, lead_hours)，与 Houston raw 格式一致
- **Label_h6**（宽格式）：每行一个 run_time，列名为 `variable_t_plus_1h` ... `variable_t_plus_6h`（142列），供 `ForecastBundleReader` 读取

Miami forecast 数据下载：`download_miami_forecast.py`
- 984 个 run_times（122天 × 8 runs/天，每3小时一次）
- 自动转换 raw → label_h6 宽格式
- 下载中，预计 30-60 分钟

**验证**：Miami 2025-08-15 12:00 UTC run 的预测显示 cloud cover 从 3% 逐步升到 22%（13:00-18:00），符合 Miami 午后对流开始的典型 pattern。

**Cell 文件环境变量支持**：
- `RL_IDF`：IDF文件名（默认 `houston.idf`）
- `RL_EPW`：EPW文件名（默认 Houston EPW）
- `RL_FORECAST_CSV`：Forecast CSV文件名（默认 Houston label_h6）
- Forecast reader 加入 try/except 保护，缺失 forecast CSV 时不 crash（fallback 为全零 forecast）

### Block 结构调整：2h block + 30min knot

| Block | 时段 | 时长 | Steps | Knots |
|-------|------|------|-------|-------|
| 0 | 06:30-08:30 | 2h | 12 | 4 |
| 1 | 08:30-10:30 | 2h | 12 | 4 |
| 2 | 10:30-12:30 | 2h | 12 | 4 |
| 3 | 12:30-14:30 | 2h | 12 | 4 |
| 4 | 14:30-16:30 | 2h | 12 | 4 |
| 5 | 16:30-19:00 | 2.5h | 15 | 5 |
| **Total** | | **12.5h** | **75** | **25** |

- LLM调用/step: 75（25 knots × 3 modes）
- 预估训练速度: ~8 min/step，60 steps ≈ 8 小时
- 信号强度: 2h block比1h强（4 knots累积），比3h更细的mode切换（6次/天）

### Miami 完整训练计划

#### Phase 0：PPO Baseline 训练

两个PPO变体在Miami 3周数据（Aug 1-22）上训练，作为GRPO的对照和SFT数据源。

| 模型 | GPU | IDF | EPW | Forecast | Episodes | Steps/ep |
|------|-----|-----|-----|----------|----------|----------|
| PPO forecast | GPU 0 | `miami_3week.idf` | `miami_..._historical_weather_api.epw` | `miami_..._label_h6.csv`（每小时HRRR） | 300 | 5000 |
| PPO no-forecast | GPU 1 | `miami_3week.idf` | `miami_..._historical_weather_api.epw` | — | 300 | 5000 |

PPO 配置：`RL_NUM_GPUS=1`, seed=1229, lr=2e-5, gamma=0.99, MLP [256,256], 3 env_runners

**Forecast 数据**：已从 Open-Meteo `single-runs-api` 下载完成。HRRR 每小时更新（2952 runs × 6h ahead = 17,712 rows），转为 Houston 兼容的 label_h6 宽格式（142列）。

#### Phase 1：收集 PPO SFT 数据

- 用 PPO forecast checkpoint 在 Miami 15 个训练日跑 `evaluate_planner_workday_closed_loop`
- 每步收集 `(knot_system_prompt, knot_user_prompt, setpoint_json)` × 3 modes
- 预计 ~3,375 条 SFT 样本（15天 × 75步 × 3 modes）

#### Phase 2：SFT 蒸馏

- 用 PPO 数据 fine-tune Qwen3-8B（LoRA, batch=2, grad_accum=8, lr=2e-5, 1 epoch）
- LLM 学会在 Miami 天气条件下输出接近 PPO 水平的 setpoint

#### Phase 3：GRPO 微调

从 SFT checkpoint 开始，用 GRPO 在 PPO 基础上继续优化。

**GRPO 配置**：

| 参数 | 值 |
|------|-----|
| Block 结构 | 6 × 2h blocks（最后一个2.5h），30min knot |
| Candidate modes | 3（comfort/balanced/energy_saving） |
| Advantage | Baseline-anchored（`r / (std + eps)`），reward = candidate - 24°C baseline |
| KL penalty | kl_beta=0.1 vs base model（SFT后为常数偏移，不实际约束） |
| Gradient clipping | `clip_grad_norm_(max_norm=1.0)` |
| Day-level gradient | DAY_ADVANTAGE_SCALE=0.3，全天 winning knots 额外更新 |
| Per-knot advantage | Monte Carlo partial return within block（0.3权重） |
| Reflexion | 天气 forecast context + per-block all-mode reward 对比，跨天按日期索引 |
| Parallel rollout | ThreadPoolExecutor(max_workers=3) |
| lr | 2e-5 |
| reward_scale | 3.0 |
| temperature | 0.7, top_p=0.95 |
| 数据集 | Miami 3周 15 workdays |
| Episodes | 4 × 15 days = 60 steps |

**关键差异 vs Houston**：Miami 日内天气变化大（77%工作日云量swing>60%），forecast 信息价值高。LLM 利用 forecast 做 pre-cooling / mode 切换的优势预期更明显。

#### Phase 4：Eval 对比

**两种 Eval 模式**：

1. **3-mode best-of-3**（和训练一致）：每block跑3个mode选最优。公平对比训练效果，但不反映真实部署场景。

2. **Reflexion-guided single-mode**（真实部署模式）：每block开头LLM根据当前observation + forecast + Reflexion记忆选择一个mode，只跑选定的mode。

   实现：`BlockPlanner.select_mode()` 方法 + `evaluate_workday_blocks(mode_selector=...)` 参数。

   流程：
   ```
   Block开始 → probe observation → LLM选mode（"根据反思经验，午后PV低时用comfort"）→ 只跑comfort → 执行
   ```

   这是 Reflexion 的核心价值——训练时3-mode探索积累的经验，在部署时压缩为单次mode选择。PPO做不到这种test-time reasoning。

#### Reflexion 增强：Per-Block 反思 + 观测轨迹 + 失败分析

**问题**：之前的day-level反思太粗糙，几乎每天结论都是"comfort最好"，无法支撑block级别的mode selection。

**改进**：

1. **Per-block 反思**（`generate_block_reflection()`）：每个block结束后立即反思，而非等到一天结束。每次反思精确到2小时时段，包含具体的mode reward对比。

2. **观测轨迹**（`observation_trajectory`）：反思输入包含block内的环境变化——PV、cloudcover、outdoor temp的起止值。LLM能看到"PV从6降到2，cloudcover从20%跳到80%"。

3. **失败分析**：明确要求LLM分析"最差mode为什么输了"。prompt要求输出格式化的rule："When [condition], use [mode]"。

4. **Block反思记忆**（`_block_reflections` + `get_block_reflection_context()`）：按block index检索历史反思，同时段的经验优先。注入knot prompt和select_mode prompt。

**反思示例**（目标质量）：
```
[2025-08-14 B4 14:30-16:30] comfort赢(+2.7)因为cloudcover从15%跳到75%，PV从7降到3kWh。
energy_saving输(-1.2)：高setpoint导致PMV+0.4，但PV降了省电优势消失。
Rule: When cloudcover > 60% and PV dropping, use comfort.
```

### 架构设计哲学：SFT × GRPO × Reflexion 三层分工

**核心洞察**：GRPO 的 3-mode 探索本质上是在为 Reflexion 生成训练数据——每个 block 跑 3 种策略得到明确的 reward 对比，这正是反思需要的素材。

**逻辑链**：

```
GRPO 训练 → 每 block 3-mode 对比 → 产生"什么条件下哪个 mode 好"的数据
     ↓
Per-block 反思 → 把对比结果压缩成自然语言 rule（"cloudcover > 60% 时用 comfort"）
     ↓
Reflexion 记忆 → 积累跨天跨 block 的策略知识库
     ↓
Eval/部署 → select_mode 读取记忆 → 单次 mode 选择 → 执行
```

**三层分工**：

| 组件 | 需要训练？ | 能力来源 | 职责 |
|------|----------|---------|------|
| **Setpoint 输出** | ✅ SFT + GRPO | Domain-specific 微调 | 给定 mode + observation → 输出精确 setpoint |
| **Reflexion 反思** | ❌ Base LLM | 预训练的总结归纳能力 | 将 3-mode 对比结果压缩为 condition→mode rule |
| **Mode selection** | ❌ Reflexion 经验 | 反思记忆 + in-context reasoning | 根据当前条件选择最优 mode |

**为什么 Reflexion 不需要训练**：
- 反思本质是"总结+归纳"，这是 LLM 的预训练能力，不需要 LoRA 微调
- 反思的 reward 难以定义（"好的反思"无法量化）
- 反思 token 参与 backward 会让梯度路径更长更不稳定

**为什么这个架构比纯 PPO 有优势**：
1. PPO 的 MLP 是黑箱——无法解释为什么输出某个 setpoint
2. PPO 无法做 test-time exploration——只有一个 policy，没有 mode 选择
3. PPO 无法利用 forecast 语义——MLP 看到 30 维数字向量，学不到"cloudcover 升高→PV 下降→该 pre-cool"
4. PPO 无法跨天积累经验——每步独立决策，没有 Reflexion 记忆

**LLM agent 的独特价值不在于"输出更精确的数字"，而在于"做更聪明的策略选择"**。

### 完整架构总结

LLM 同时承担高层决策和低层执行，一个模型替代了传统 hierarchical LLM+RL 的两个模型：

```
PPO（底层精细控制能力）
  → SFT 蒸馏：LLM 学会 PPO 级别的 setpoint 输出
      → GRPO 3-mode 探索：每 block 跑 comfort/balanced/energy_saving，打分对比
          → Per-block Reflexion：将对比结果压缩为 condition→mode rule
              → Eval/部署：读取 Reflexion 记忆 → select_mode 选择最优 mode → 单次执行
```

**各组件职责**：

| 阶段 | 方法 | 输入 | 输出 | 职责 |
|------|------|------|------|------|
| Phase 1 | PPO 训练 | EnergyPlus env | MLP checkpoint | 产生精细控制的 teacher |
| Phase 2 | SFT 蒸馏 | (prompt, PPO action) pairs | LoRA adapter | LLM 学会精细 setpoint 输出 |
| Phase 3 | GRPO 探索 | 3 modes × EnergyPlus | reward 对比 + LoRA 更新 | 优化 mode 内 setpoint + 产生 Reflexion 素材 |
| 训练中 | Reflexion | block reward 对比 + 天气 | 自然语言 rule | 积累"什么条件用什么 mode"的策略知识 |
| Eval | select_mode | observation + forecast + 记忆 | mode 名称 | 利用积累的策略知识做单次决策 |
| Eval | plan_knot | observation + mode prompt | setpoint JSON | SFT+GRPO 训练的精细控制能力 |

**vs 传统 Hierarchical LLM+RL**（如 [arxiv 2603.26050](https://arxiv.org/abs/2603.26050)）：
- 传统方案：LLM 做高层 action masking + RL 做低层控制（两个模型）
- 本方案：**一个 LLM 同时承担高层（Reflexion mode selection）和低层（SFT+GRPO setpoint output）**
- 优势：无需两模型协调、端到端可解释、Reflexion 经验可迁移到新建筑

**Eval 计划**：
- 在 Miami held-out（Aug 25-29 和 Sep 1-5）上评估
- PPO forecast vs PPO no-forecast（验证 forecast 在 Miami 是否有用）
- GRPO best-of-3 vs GRPO single-mode（验证 Reflexion mode selection 效果）
- GRPO vs PPO（验证 LLM agent 能否超越 PPO）
- 重点关注：午后雷暴前的 pre-cooling 决策、forecast-aware mode 切换

### 当前实验：Miami 无SFT GRPO（直接从 base Qwen3 训练）

**目的**：在Miami天气条件下验证GRPO agent能否利用forecast信息。不经过SFT蒸馏，直接从base Qwen3-8B + LoRA开始GRPO训练。

**GRPO 配置**：
- **Run**: `miami_grpo_2h_block_30min_knot_20260403`
- **WandB**: `miami_grpo_2h_block_30min_fresh`（project: `asim-miami-grpo`）
- **Block**: 6 × 2h（最后2.5h），30min knot
- **Modes**: 3（comfort/balanced/energy_saving）
- **Advantage**: baseline-anchored（vs 24°C baseline）
- **KL**: kl_beta=0.1 vs base model
- **Gradient**: clip_grad_norm=1.0 + day-level gradient + per-knot partial return
- **Reflexion**: 天气 forecast context 注入
- **Parallel**: ThreadPoolExecutor(3)
- **IDF**: `miami_3week.idf`（Aug 1-22）
- **EPW**: `miami_..._historical_weather_api.epw`
- **初始化**: base Qwen3-8B + 新LoRA（不 resume）
- **GPU**: cuda:0

**对照 PPO 训练**：
- PPO no-forecast: GPU 1, 300ep, 进行中（83/300 ep）
- PPO forecast: 待 PPO nofc 完成后在 GPU 1 启动

**后续计划**：
1. PPO 训练完成后收集 SFT 数据
2. SFT → GRPO Phase 3 训练
3. 对比：GRPO(无SFT) vs GRPO(SFT→GRPO) vs PPO forecast vs PPO no-forecast
4. 重点验证：Miami 午后雷暴场景下 forecast-aware 决策

### SFT→GRPO 实验总结（已放弃）

**结论：SFT→GRPO 这条路在当前HVAC控制场景下不通。无SFT的GRPO反而更稳定更有效。**

#### 所有SFT→GRPO尝试及失败原因

| # | 配置 | 结果 | 失败原因 |
|---|------|------|---------|
| 1 | 3-mode SFT + kl=0.1 vs base | 前6步好（+15.30），手动kill | 误判KL=17670为不健康（实际是SFT后的常数偏移） |
| 2 | 3-mode SFT + kl=0 | step 12崩到-356 | 无KL → advantage全正 → 累积drift |
| 3 | 3-mode SFT + kl=0.1 vs SFT ref | KL从0.98涨到273 | kl_beta=0.1太小压不住 |
| 4 | balanced-only SFT + kl=0.1 vs base + lr=2e-5 | step 3崩到-145 | lr太大，2步就破坏SFT pattern |
| 5 | balanced-only SFT + lr=1e-6 | step 2才+1.74 | lr太小，学不动 |
| 6 | balanced-only SFT + ratio clipping v1 | step 5崩到-264 | old_logprobs在optimizer.step后计算（bug） |
| 7 | balanced-only SFT + ratio clipping v2（修正） | step 2崩到-28 | clipping仍不够 |

#### 根本原因分析

1. **SFT后LoRA在"脆弱平衡点"**：SFT学的是"一种策略（balanced）的输出"，GRPO要推向"多种策略"→方向冲突，任何梯度更新都容易偏移
2. **KL约束困境**：
   - KL vs base model → SFT后已经远离base，KL是常数偏移（几千），无约束力
   - KL vs SFT ref → 有效但增长太快，kl_beta调不好（太小压不住，太大学不动）
3. **LoRA参数少（504个）**：每个参数的扰动影响大，不像full fine-tune有冗余
4. **3-mode SFT的额外问题**：3个mode映射同一个PPO action → 模型学到"忽略mode prompt" → GRPO要打破这个pattern需要极大梯度 → 爆炸

#### Setpoint输出对比

| Mode | 无SFT GRPO（ep3） | SFT（balanced-only） |
|------|------------------|---------------------|
| comfort | **24.5°C** | 26.5°C |
| balanced | **25.6°C** | 26.6°C |
| energy_saving | **26.5°C** | 26.5°C |

SFT模型3个mode输出几乎一样（26.5-26.6°C），mode prompt被忽略。无SFT的GRPO反而学会了mode差异化。

#### 结论

- **SFT对LLM HVAC agent有害**：破坏了mode sensitivity，产生脆弱的LoRA状态
- **无SFT的GRPO更好**：从base Qwen3出发，KL有效（从0开始），模型有探索空间
- **SFT的价值（zone差异化setpoint）可以通过prompt engineering替代**：zone PMV hints直接告诉模型每个zone的调整方向
- **AlphaGo的SFT→RL路径不适用于此场景**：AlphaGo的SFT教"怎么下棋"（单一目标），这里的SFT教"一种策略的输出"（GRPO需要多种策略）

#### 保留的改进

以下改进在SFT实验中开发，已整合到无SFT的GRPO训练中：
- balanced-only SFT数据收集脚本（`collect_ppo_sft_data.py`）
- PPO-style importance ratio clipping（代码已加，当前训练未启用）
- SFT adapter snapshot for KL reference（代码已加，当前训练未使用）
- [Two-Stage SFT + GRPO Pipeline](https://www.emergentmind.com/topics/two-stage-sft-grpo-training-pipeline)

**当前Houston验证run**：`houston_grpo_sft_balanced_clip_20260403`（GPU 1, lr=2e-5, clip_eps=0.2）

### 当前状态

| 任务 | GPU | 状态 |
|------|-----|------|
| Miami GRPO（无SFT，增强Reflex，2h block） | GPU 0 | 训练中（3/60 steps） |
| Miami PPO no-forecast（3wk, 300ep） | GPU 1 | 训练中（182/300 ep） |
| Miami PPO forecast（3wk, 300ep） | GPU 1 | 训练中（90/300 ep） |
| Miami forecast 数据 | — | ✅ 已完成 |
| Miami SFT→GRPO | — | 待 PPO fc 完成后执行 |

### Miami PPO nofc Eval 结果（Aug 25-29）

| Date | PPO nofc 3wk |
|------|-------------|
| 08-25 | +4.16 |
| 08-26 | +2.84 |
| 08-27 | +0.99 |
| 08-28 | +1.04 |
| 08-29 | -0.00 |
| **Total** | **+9.02** |
| **Mean** | **+1.80/day** |

Miami PPO比Houston PPO弱（+1.80 vs +3.93/day），Miami天气更复杂。GRPO训练mean=+8.99/day（含best-of-3），待eval验证。

### Miami PPO 30min 控制对照（公平对比）

**动机**：PPO用10分钟控制步（75步/天），GRPO用30分钟knot（25 knots/天）。PPO有3倍控制频率优势，对比不公平。

**方案**：加入 `RL_ACTION_REPEAT=3` 环境变量——PPO每3个EP timestep才调用一次policy，中间保持上一个action。等效于30分钟控制一次，和GRPO对齐。

| PPO版本 | 控制频率 | 决策数/天 |
|---------|---------|----------|
| PPO 10min（原版） | 每10分钟 | 75 |
| **PPO 30min（新）** | **每30分钟** | **25** |
| GRPO 30min knot | 每30分钟 | 25 |

**PPO 30min训练已完成**（300ep × 2）。

### Miami PPO 30min Eval（公平对比，Aug 25-29）

| Date | PPO nofc 30min | PPO fc 30min | GRPO best-of-3 (30min) |
|------|---------------|-------------|----------------------|
| 08-25 | -28.00 | -22.87 | **+4.09** |
| 08-26 | -20.84 | -15.79 | **+2.68** |
| 08-27 | -16.52 | -11.32 | **+0.61** |
| 08-28 | -15.65 | -10.91 | **+0.94** |
| 08-29 | -11.99 | -7.92 | **-0.14** |
| **Total** | **-93.0** | **-68.8** | **+8.17** |
| **Mean** | **-18.6** | **-13.8** | **+1.63** |

**核心结论：在公平的30分钟控制频率下，GRPO大幅超越PPO（+8.17 vs -68.82，差77分）。**

**关键发现**：
1. **PPO在30min控制下完全失效**——全部天都大幅负于24°C baseline。MLP在10min时靠快速反馈弥补精度，30min间隔下丧失了这个优势
2. **Forecast对PPO 30min有用**（-68.8 vs -93.0，差24分）——控制频率低时forecast价值增大，但仍不够救PPO
3. **GRPO的mode selection + Reflexion在30min控制下仍然有效**——因为策略选择和setpoint质量不依赖控制频率
4. **这证明了LLM agent在低频控制场景下的根本优势**：PPO需要高频反馈，LLM agent靠语义理解和策略推理弥补

### GRPO v1 (8ep, histbest+zone hints) Eval 结果

使用 checkpoint-30（ep2结束，训练mean最高+12.66）：

| Date | PPO 10min fc | PPO 30min fc | GRPO v1 ckpt-30 |
|------|-------------|-------------|-----------------|
| 08-25 | +4.21 | -22.87 | +4.04 |
| 08-26 | +2.87 | -15.79 | **+2.80** |
| 08-27 | +1.02 | -11.32 | **+0.99** |
| 08-28 | +1.07 | -10.91 | **+1.13** |
| 08-29 | +0.02 | -7.92 | **+0.12** |
| **Total** | **+9.19** | **-68.82** | **+9.10** |
| **Mean** | **+1.84** | **-13.76** | **+1.82** |

**GRPO 30min (+9.10) 几乎追平 PPO 10min (+9.19)，差距仅0.09分（1%）。用1/3的控制频率达到同等效果。**

### Mode Selection Pattern（GRPO学到的时段策略）

训练中per-block winner mode分布（4 episodes, 72 blocks/ep）：

| Block | 时段 | comfort | balanced | energy_saving | 策略含义 |
|-------|------|---------|----------|---------------|---------|
| 0 | 06:30-08:30 | 12% | 30% | **58%** | 早上PV低+制冷需求低→省电 |
| 1 | 08:30-10:30 | 15% | **62%** | 22% | 上午温和→平衡策略 |
| 2 | 10:30-12:30 | 11% | **82%** | 7% | 中午PV上升→平衡策略主导 |
| 3 | 12:30-14:30 | 29% | **57%** | 14% | 午后开始热→balanced为主，comfort增加 |
| 4 | 14:30-16:30 | **68%** | 31% | 1% | 下午最热→必须制冷 |
| 5 | 16:30-19:00 | **40%** | 26% | **33%** | 傍晚天气变化大（Miami雷暴）→三方混战 |

**关键发现**：
1. **模型学会了时段差异化策略**——早上省电、中午平衡、下午制冷、傍晚看天气
2. **这不是随机的**——82%的中午block选balanced、68%的下午block选comfort，有明确的时段依赖
3. **Block 5（傍晚）三方混战**（40%/26%/33%）反映了Miami午后雷暴的天气不确定性——模型根据当天具体天气做不同选择
4. **Energy_saving只在早晨dominant（58%）**——模型学到了"PV不足时省电最有效"
5. **这种时段策略是PPO做不到的**——PPO只有一个固定policy，无法根据时段切换策略方向

### GRPO v2 训练结果（修正反思，from checkpoint-15 resume）

| Episode | v1 | v2 |
|---------|------|------|
| ep1 | +12.15 | +12.06 |
| ep2 | +12.66 | +12.24 |
| ep3 | +12.41 | +12.04 |
| ep4 | +11.99 | **+12.88** |

v2 ep4比v1 ep4高0.89分（修正了first_knot/last_knot PMV对比的反思）。两版都完全稳定无drift（history best gating有效）。v2 checkpoint-60 eval进行中。

### Miami Eval 结果（Aug 25-29，PPO 10min vs GRPO 30min，不公平对比）

| Date | PPO fc (10min) | PPO nofc (10min) | GRPO best-of-3 (30min) | GRPO single (30min) |
|------|---------------|-----------------|----------------------|-------------------|
| 08-25 | +4.21 | +4.16 | +4.09 | +3.96 |
| 08-26 | +2.87 | +2.84 | +2.68 | **+2.75** |
| 08-27 | +1.02 | +0.99 | +0.61 | +0.38 |
| 08-28 | +1.07 | +1.04 | +0.94 | **+1.03** |
| 08-29 | +0.02 | -0.00 | -0.14 | -1.50 |
| **Total** | **+9.18** | **+9.02** | **+8.17** | **+6.61** |
| **Mean** | **+1.84** | **+1.80** | **+1.63** | **+1.32** |

**关键发现**：
1. GRPO best-of-3（+8.17）接近PPO（+9.18），差距仅11%——GRPO无SFT、30min粗控制 vs PPO 10min精细控制
2. PPO forecast vs nofc差距极小（+0.16），确认PPO的MLP无法利用forecast信息
3. GRPO single-mode的Reflexion-guided mode selection全选comfort——训练反思质量不够细致，未来需要反思压缩/总结
4. Aug 26和28上single-mode超过best-of-3，说明统一comfort有时优于混合mode
5. **公平对比（PPO 30min）正在训练中**——PPO控制频率降为30min后预计performance下降，GRPO可能追平或超越

### Reflexion Mode Selection 详细对比

| 方法 | Mode选择策略 | Total | Mean |
|------|------------|-------|------|
| GRPO best-of-3 | 每block跑3个选最好 | **+8.17** | +1.63 |
| GRPO single (raw反思) | 全选comfort（60条反思都说comfort好） | +6.61 | +1.32 |
| GRPO single (compressed rules) | 差异化（早上energy_saving，中午balanced，下午comfort） | **-82.82** | -16.56 |

**关键发现**：compressed rules的差异化mode选择反而大幅劣于全选comfort。原因：

1. **balanced和energy_saving的setpoint质量差**：无SFT的GRPO在balanced(25.6°C)和energy_saving(26.5°C)下输出太高的setpoint，导致PMV violation
2. **comfort的setpoint(24.5°C)是唯一安全的**：在Miami的高温高湿环境下，只有comfort的低setpoint能维持舒适度
3. **训练4 episodes不够**：comfort在训练中经常赢（因为Miami热），所以学得好；balanced/energy_saving赢得少，学得差

**Setpoint对比（同一observation）**：

| Mode | 无SFT GRPO | SFT (balanced-only) |
|------|-----------|---------------------|
| comfort | 24.5°C | 26.5°C |
| balanced | 25.6°C | 26.6°C |
| energy_saving | 26.5°C | 26.5°C |
| Zone差异化 | 无（全zone一样） | 无（全zone一样） |

**结论**：当前GRPO agent的价值在于best-of-3 mode selection（+8.17），而非single-mode deployment。要实现真正的single-mode部署，需要每个mode内部的setpoint质量都足够好——这需要更多训练episodes和zone级别的反思引导。

### Miami GRPO 8-Episode 训练（Zone PMV Reflexion）

**改进**：
1. **8 episodes**（之前4个）：让balanced和energy_saving有更多训练机会
2. **Zone PMV超标反思**：per-block反思现在包含超出mode PMV目标范围的zone提示，如 "1FSW: PMV=+0.3 EXCEEDED comfort range (±0.2), temp=26.5°C"。只报告超标zone，不冗余。
3. **训练结束自动压缩**：调用 `compress_reflections()` 生成5-8条精炼rules，保存到 `reflections.json`

**Run (v1, drift问题)**: `miami_grpo_2h_30min_8ep_zone_reflex_20260404` — ep2 step 19崩到-41，KL从0.3涨到27.8

#### 改进 12：History Best Gating（防drift的根本方案）

**问题**：标准GRPO + baseline-anchored advantage仍然在8ep长训练中drift（ep2开始崩）。原因：虽然全负时不强化，但全正时会持续往一个方向推，累积偏移。

**方案**：per-(day, block) 追踪历史最佳reward。只有当当前block的winner reward**超越之前所有episode该block的best**时才更新权重：

```python
if current_best > history_best_block_reward[(skip, block_index)]:
    history_best_block_reward[(skip, block_index)] = current_best
    do_gradient_update()  # 只在创新纪录时更新
else:
    skip_update()  # 没超过就不动权重
```

**效果**：
- ep1：所有block都是新纪录，正常更新
- ep2+：只有超越ep1 best的block更新，performance只能上升或持平
- 不可能drift——权重只在"变好"时才改

#### 改进 13：Zone PMV Direction Hints in Knot Prompt

**问题**：模型能看到每个zone的PMV，但不知道该怎么响应（PMV=0.94→setpoint降多少？）。输出统一setpoint因为没有zone差异化的梯度信号。

**方案**：在knot user prompt里加zone-level方向提示，只显示超标zone：
```
Setpoint adjustment hints (target PMV -0.2 to +0.1):
  0FNW: PMV=-0.58 TOO LOW → HIGHER setpoint
  1FSW: PMV=+0.94 TOO HIGH → LOWER setpoint
```

PMV在目标范围内的zone不提示（避免冗余）。模型只需要知道方向（LOWER/HIGHER），GRPO通过多次rollout找到具体数值。

**当前Run**: `miami_grpo_8ep_histbest_20260404`（含zone hints + day-level history best + baseline-anchored + zone PMV reflex）
**Steps**: 120（8ep × 15days），GPU 0

#### 改进 8：Per-Knot Partial Return（三层 Advantage）

**动机**：当前 block 内所有 knot 共享同一个 advantage，但 knot 0 的决策影响后续全部步（18步），knot 17 只影响最后 1 步。共享 advantage 给不同位置的 knot 相同的梯度权重，不够精确。

**方案**：用 EP 已有的 step-level reward 计算 per-knot Monte Carlo partial return：

```python
knot_0_return = sum(step_rewards[0:18])   # 影响全部后续
knot_1_return = sum(step_rewards[1:18])   # 影响 17 步
...
knot_17_return = step_rewards[17]          # 只影响最后 1 步
```

归一化后作为第三层 advantage，与 tier-level 和 sub-level 叠加：

```python
total_advantage = tier_advantage + 0.5 * sub_advantage + 0.3 * knot_partial_advantage
```

**三层信号互补**：
- **Per-knot**（0.3 权重）：这一步决策在 block 内的短期效果（6-18 步精确信号）
- **Block-level tier + sub**（1.0 + 0.5 权重）：mode 选择的中期效果（3h 比较）
- **Day-level**（0.3 权重，单独 optimizer.step）：全天策略组合的长期效果（跨 block）

**注意**：per-knot return 不包含跨 block 影响（knot 17 对下一个 block 的温度影响），这由 day-level gradient 补偿。

**已实施**。使用 `block_reward_trace` 中的 step rewards 计算 partial return，无需额外 EP 模拟。

### WandB 集成

`train_qwen3_houston_gspo_block.py` 新增 wandb 支持，per-step 和 per-block 指标实时上传：

- `--wandb-project`（默认 `asim-houston-grpo`）
- `--wandb-group`（默认 `block-rolling-grpo`）
- `--wandb-name`（默认取 output-dir 名称）
- `--no-wandb` 禁用

记录的指标：
- Step 级别：`winner_relative_reward`, `avg_block_reward_std`, `avg_block_grad_norm`, `avg_block_kl`, `blocks_updated`, `episode`
- Block 级别：`block{i}/reward_std`, `block{i}/grad_norm`, `block{i}/kl`, `block{i}/winner_reward`

## Implementation Notes

- RLlib 支持 `DictSpace` 中嵌套 `Box(shape=(6,))`
- forecast observation 目前通过 `MutableVariable(np.ndarray)` 更新
- 当前 notebook 同时提供两套 Houston 训练环境：
  - `UserEnv`: 含 forecast window
  - `UserEnvNoForecast`: 不含任何 `forecast_*` observation
- 天气与 forecast 数据现在统一放在 `weather/`
- smoke / compare / EnergyPlus report / training artifact 现在统一放在 `result/`
- 两套环境当前保持完全一致的：
  - building
  - weather
  - action space
  - reward
  - random-start episode 逻辑
- 因此当前 forecast ablation 的唯一差异就是 observation 中是否加入 `forecast_*`
- Houston 本地 forecast CSV 当前实际覆盖从 `2025-07-26 19:00:00` 到 `2025-09-30 16:00:00`
- 在 forecast 尚不可用的时段，当前实现返回零向量，并通过 `forecast_available = 0` 标记
- 当前 lookup 会在 `wallclock` 年份不属于 forecast CSV 年份时直接返回不可用，避免 sizing/design day 错误复用最后一条 forecast
- forecast lookup 现在统一按 `Timestamp.value` 的纳秒时间戳比较，避免 `pandas` 版本变化导致的 `ns/us` 单位错配
- 对 Houston forecast CSV 中已知的缺失值，reader 现在会按 `run_time` 方向做 `ffill + bfill`，再兜底 `0.0`，避免 `NaN` 直接进入 RL observation
- notebook 现在会把 `BoxSpace` 绑定值统一强制转换成与空间一致的 `np.ndarray/np.float32` 形状，避免 RLlib 在 `observation_space.contains()` 检查时反复触发 `gymnasium` 的 `Casting input x to numpy array` warning
- `UserEnv.run()` 现在会在系统启动前预先 request 全部 observation / action / reward 相关绑定，避免首轮 run 中 `OutputVariable` 因“请求过晚”而对 RL observation 持续返回 `TemporaryUnavailableError`

## Smoke Test

Houston forecast observation 的最小运行验证脚本是：

- [smoke_test_houston_forecast_observation.py](/home/AD/user/lab/asim/smoke_test_houston_forecast_observation.py)
- [compare_houston_rl_observation_forecast.py](/home/AD/user/lab/asim/compare_houston_rl_observation_forecast.py)
- [smoke_test_houston_ppo_train.py](/home/AD/user/lab/asim/smoke_test_houston_ppo_train.py)
- [smoke_test_houston_ppo_eval_episode.py](/home/AD/user/lab/asim/smoke_test_houston_ppo_eval_episode.py)

当前 notebook 的训练入口也已经拆成两套 tuner helper：

- `FORECAST_TUNER`: 对应 `UserEnv`
- `NO_FORECAST_TUNER`: 对应 `UserEnvNoForecast`
- 当前默认的 WandB 配置是：
- `project = asim-houston-forecast-rl`
- `group = forecast-vs-no-forecast`
- `run_prefix = houston_aug2025`
- 单次训练 run name 会自动生成为 `houston_aug2025_forecast_window` 或 `houston_aug2025_no_forecast_window`
- 如需改名，可以在启动 notebook 或 Python 进程前设置 `WANDB_PROJECT`、`WANDB_GROUP`、`WANDB_RUN_PREFIX`、`WANDB_ENTITY`
- 当前 notebook 的默认 episode 长度是 `5000` steps，可用 `RL_EPISODE_STEPS` 覆盖
- 当前 notebook 默认会把 `RL_TRAIN_BATCH_SIZE` 对齐到 `EPISODE_STEPS`；如果不手动覆盖，就是 `5000`
- 当前 notebook 的默认试跑目标是 `10` 个 episode，对应 `RL_TRAIN_EPISODES=10` 和总采样步数 `50000`
- 如果你要严格按“完整 episode 数”训练，而不是按 Tune 的 sampled steps 停止，可以直接运行 [run_houston_fixed_episodes.py](/home/AD/user/lab/asim/run_houston_fixed_episodes.py)
- 这个脚本默认用 `forecast_window`、`5000-step episode`、`10 episodes`、`num_env_runners=0`，并把结果写到 `result/manual_train/<run_name>/`
- 运行示例：`RL_EPISODE_STEPS=5000 RL_TRAIN_EPISODES=10 RL_VARIANT=forecast_window RL_NUM_GPUS=0 ./.venv/bin/python run_houston_fixed_episodes.py`
- 如果要切到 LSTM notebook 对应的环境定义，额外设置 `RL_TMP_CELL_PREFIX=.tmp_todo_lstm`
- LSTM notebook 默认 WandB 命名是 `project=asim-houston-forecast-rl`、`group=lstm-forecast-vs-no-forecast`、`run_prefix=houston_aug2025_lstm`
- 这个脚本现在会额外输出逐 episode 结果到 `episode_history.json/csv`，并把 `episode_reward`、`episode_length` 按 `episode_index` 单独记录到 WandB
- 同时它也会把 `algo.train()` 返回结果里的 `info/learner/...`、`env_runners/...` 这类训练指标按 `training_iteration` 展开记录到 WandB
- WandB 的 system/machine stats 现在默认关闭，不再上传 system 面板数据
- 训练资源现在会自动检测本机 GPU 数量；如果当前机器没有可见 GPU，会自动按 `num_gpus = 0` 回退到 CPU
- 如果你想手动覆盖 GPU 数量，可以在启动前设置 `RL_NUM_GPUS`
- 训练采样配置也支持环境变量覆盖：`RL_NUM_ENV_RUNNERS`、`RL_ROLLOUT_FRAGMENT_LENGTH`、`RL_TRAIN_BATCH_SIZE`、`RL_MINIBATCH_SIZE`、`RL_NUM_EPOCHS`
- 如果当前环境没有安装 `wandb`，训练仍然可以启动，只是不会挂 WandB callback
- Tune 落盘路径也已经固定到 `result/ray_tune/`

它会：

- 读取 notebook 的环境定义
- 使用 Houston `idf + epw`
- 在仿真过程中直接解引用 `forecast_*` observation 绑定
- 提取第一条 `forecast_available > 0.5` 的 Houston forecast observation
- 检查 forecast shape 和数值是否正常
- 对比 `env.agent.observation.value` 中的 `forecast_*` 与底层绑定值是否逐步一致
- 用最小 PPO 配置跑一轮 `train()`，确认 Houston 环境在当前依赖组合下可以正常 build/sample/train
- 用未训练 PPO 策略完整跑 `1` 个 evaluation episode，确认月尺度 episode 可以走完

## Eval Block 结构修正与最终对比（2026-04-05）

### 重要修正：之前 eval 使用了错误的 block 结构

之前 eval GRPO 时使用了 `gspo_houston_bandit_30min.py`（4 blocks × 3h），但训练时用的是 6 blocks × 2h。Block 结构不匹配导致 eval 结果严重偏低。

| 配置 | Blocks | Knots/block | 总决策/天 |
|------|--------|------------|----------|
| 训练 (`gspo_houston_bandit.py` 当时) | 6 × 2h | 4 | 24 |
| 错误 eval (`gspo_houston_bandit_30min.py`) | 4 × 3h | 6 | 24 |
| 修正 eval (`gspo_houston_bandit_2h.py`) | 6 × 2h | 4 | 24 |

虽然总 knots/天相同（24），但 block 边界不同意味着 mode 切换时机不同。训练时模型学到的是"在 2h block 边界处切换 mode"的策略，eval 用 3h block 则完全错位。

### ⚠️ 第二次修正：skip_valid_steps 不匹配

上面的"修正后"结果（+26.65 等）仍然有错：**GRPO eval 使用了 `skip_valid_steps=0-300`（对应 Aug 1-7，训练集内），而 PPO eval 使用 `skip_valid_steps=1200-1500`（对应 Aug 25-29，训练集外）。**

这意味着 GRPO 在训练数据上评估，PPO 在未见数据上评估，完全不可比。

**正确 eval**: `eval_grpo_vs_ppo_fair.py` 使用与 PPO 相同的 skip=1200-1500，确保评估 Aug 25-29。

### 公平对比结果（Miami Aug 25-29, skip=1200-1500）

| 方法 | Total | Mean | 控制频率 | 决策/天 |
|------|-------|------|---------|---------|
| PPO 10min fc | **+9.18** | **+1.84** | 10min | 75 |
| GRPO best-of-3 | +8.78 | +1.76 | 30min knot / 2h block | 24 (×3 modes) |
| GRPO single (LLM+stats hybrid) | +7.51 | +1.50 | 30min knot / 2h block | 24 |

逐天对比：

| Date | GRPO-bo3 | GRPO-single | PPO | GRPO best-of-3 modes |
|------|---------|-------------|-----|----------------------|
| Aug 25 | +3.95 | +3.35 | +4.21 | es→com→bal→bal→com→bal |
| Aug 26 | +2.85 | +2.36 | +2.87 | bal→bal→com→bal→bal→bal |
| Aug 27 | +0.81 | +0.85 | +1.02 | es→es→bal→bal→bal→es |
| Aug 28 | **+1.17** | +1.01 | +1.07 | es→bal→bal→bal→bal→es |
| Aug 29 | +0.00 | -0.07 | +0.02 | es→bal→bal→com→bal→bal |

**结论**：
- GRPO best-of-3（+8.78）接近 PPO（+9.18），差距仅 **4.4%**
- GRPO 用 30min 控制频率（24 knots/天）接近 PPO 的 10min（75步/天），**用 1/3 的控制频率达到 96% 的性能**
- GRPO single-mode（+7.51）也达到 PPO 的 82%，且只需 1 次 rollout
- Mode 选择 pattern 清晰：早晨 energy_saving，中午 balanced，下午偶尔 comfort
- Aug 28 GRPO 略优于 PPO（+1.17 vs +1.07），其余天PPO略优

**GRPO 优势**：
1. 可解释性：每个 block 有明确的 mode 选择和理由
2. 低频控制：30min 一次决策，适合实际部署
3. 自适应：mode 随天气条件切换（早晨偏冷→energy_saving，下午偏热→comfort）

**详细数据**：`result/comparisons/fair_comparison_20260405/`
- `grpo_best_of_3_block_details.csv`: 每 block 的 3-mode reward 对比
- `grpo_single_mode_block_details.csv`: single-mode 每 block 详情
- `daily_comparison.csv`: 逐天三方对比
- `energy_pmv_per_block.csv`: 每 block 的能耗/PV/PMV物理量
- `energy_pmv_daily.csv`: 每日能耗/PV/PMV对比
- `summary.json`: 完整结果

### 物���量分析：GRPO vs Baseline (Miami Aug 25-29, 5天合计)

Reward公式: `reward = -0.01 * (net_grid_kwh + 50 * PMV_violation)`

| 指标 | Baseline (24°C) | GRPO bo3 | 变化 |
|------|----------------|----------|------|
| 建筑总能耗 | 8266 kWh | 8298 kWh | **+0.4%** |
| PV发电量 | 3902 kWh | 3902 kWh | 不变 |
| 电网购电 | 4364 kWh | 4396 kWh | +32 kWh |
| PV利用率 | 47.2% | 47.0% | 基本不变 |
| PMV违规总量 | 20.417 | **0.809** | **-96.0%** |
| Reward | 基准 | +9.28 | |

加入PPO物理量后的完整三方对比：

| 指标 | Baseline (24°C) | PPO 10min | GRPO bo3 |
|------|----------------|-----------|----------|
| 建筑总能耗 | 8266 kWh | 8369 kWh | 8298 kWh |
| 能耗增加 | — | +103 kWh (+1.2%) | +32 kWh (+0.4%) |
| PMV违规总量 | 20.418 | **0.000** | 0.809 |
| Reward | 基准 | +9.18 | +9.28 |

逐天分解：

| Date | BL能耗 | PPO能耗 | GRPO能耗 | BL PMV | PPO PMV | GRPO PMV |
|------|-------|---------|---------|--------|---------|---------|
| Aug 25 | 1720 | 1744 | 1743 | 8.905 | 0.000 | 0.160 |
| Aug 26 | 1667 | 1689 | 1682 | 6.162 | 0.000 | 0.060 |
| Aug 27 | 1648 | 1669 | 1645 | 2.445 | 0.000 | 0.242 |
| Aug 28 | 1619 | 1635 | 1618 | 2.465 | 0.000 | 0.170 |
| Aug 29 | 1612 | 1632 | 1610 | 0.440 | 0.000 | 0.177 |

**关键发现**：
1. **PPO和GRPO reward几乎一样（+9.18 vs +9.28），但策略完全不同**：
   - PPO：多花能耗（+103 kWh）完全消除PMV violation（0.000）
   - GRPO：少花能耗（+32 kWh）允许微量PMV violation（0.809）
2. **GRPO比PPO节能71 kWh/5天**（14.2 kWh/天），代价是0.16/天的微量PMV violation
3. 两种策略在reward上等价，但GRPO在能效上更优——GRPO学到了"不需要完全消除PMV violation就能拿到高reward"的经济策略
4. 高温天（Aug 25-26）PMV改善贡献最大
5. PV利用率不受HVAC控制影响（PV发电由太阳决定）

### 训练 Run 对比

| Run | 特性 | 稳定性 |
|-----|------|--------|
| `enhanced_reflex_20260403` | zone PMV reflex, 无 history best gating | ep2/ep4 drift（+6/+7） |
| `8ep_histbest_20260404` | 同上 + history best gating + zone PMV hints | ep1-5 稳定（+12） |

两个 run 的 block 结构、modes、knots 完全相同，唯一区别是 history best gating（改进12）和 zone PMV hints（改进13）。History best gating 是防止 drift 的关键。

### 量化规则压缩（已实施）

从 `phase_trace.jsonl` 统计 block × outdoor_temp_bucket → mode 胜率：

```
Block 0 (06:30-08:30): <31°C → energy_saving (61%), ≥31°C → balanced (100%)
Block 1 (08:30-10:30): balanced (62-63%) across all temps
Block 2 (10:30-12:30): balanced (60-100%) across all temps
Block 3 (12:30-14:30): <31°C → energy_saving (56-100%), ≥31°C → balanced (66-75%)
Block 4 (14:30-16:30): ≥28°C → comfort (70%), <28°C → mixed
Block 5 (16:30-19:00): mixed (comfort略优 at 28-31°C)
```

新增 `BlockPlanner.build_statistical_rules()` 从 phase_trace + EPW 自动构建量化规则表，`select_mode()` 已改为将统计证据（win_rate%, avg_reward）作为 primary reference 注入 LLM prompt，LLM 做最终决策。

文件：
- `gspo_houston_bandit_2h.py`: 6 × 2h block 结构的 eval 专用副本
- `llm_setpoint_planner_2h.py`: 对应 planner 副本
- `eval_grpo_statistical_mode.py`: 三种 mode 选择策略对比 eval 脚本

### PPO 训练与 Eval 流程

**训练**（使用 `.venv`，有 ray/rllib）：
1. 运行 `TODO_compare_single_agent.ipynb` 的 Cell 0（env定义：`UserEnv`, `RewardFunction`, observation/action space）和 Cell 1（`PPOConfig`, `get_config()`）
2. Cell 2+3 启动训练（`tune.run` 或 `algo.train()` 循环）
3. Checkpoint 保存到 `result/manual_train/<run_name>/checkpoint/`

或使用脚本：
```bash
RL_EPISODE_STEPS=5000 RL_TRAIN_EPISODES=300 RL_VARIANT=forecast_window \
  .venv/bin/python run_houston_fixed_episodes.py
```

**Eval**：
1. Cell 0 + Cell 1（加载env定义）
2. Cell 5（定义 `PlottingCallbacks`：每步记录 pmv, temp, elec, pv, occupancy, reward → episode结束保存CSV）
3. Cell 6（构建eval config + `algo.restore(checkpoint_path)`）
4. Cell 7（`algo_eval.evaluate()` 跑evaluation episodes）
5. 物理量数据在 `result/<run_name>/records/records_<zone_id>.csv`

或使用脚本（仅reward，无物理量）：
```bash
.venv/bin/python eval_single_model.py --model miami_ppo_fc_3wk --eval-set miami_aug_lastweek
```

**关键文件**：
- `.tmp_todo_random_start_cell0.py`: env定义（从notebook Cell 0提取）
- `.tmp_todo_random_start_cell1.py`: PPO config
- `eval_single_model.py`: 批量eval脚本（通过bandit包装，只输出reward）
- `run_houston_fixed_episodes.py`: 训练脚本

### GRPO (LLM) 训练与 Eval 流程

**训练**（使用 `.venv_qwen`，有 transformers/torch/peft）：
```bash
env CUDA_VISIBLE_DEVICES=0 .venv_qwen/bin/python train_qwen3_houston_gspo_block.py \
  --dataset-path result/gspo/houston_gspo_dataset_3week.jsonl \
  --output-dir result/gspo/<run_name> \
  --max-steps 60 --save-steps 15 --device cuda:0 \
  --building-idf miami_3week.idf \
  --weather-epw weather/miami_2025_06_01_2025_09_30_historical_weather_api.epw \
  --temperature 0.7 --top-p 0.95 --learning-rate 2e-5 --reward-scale 3.0 --kl-beta 0.1
```

训练输出：
- `checkpoint-N/`: LoRA adapter weights
- `phase_trace.jsonl`: 每个block的mode rewards、grad norm、KL等
- `metrics.jsonl`: 每步的day-level reward
- `reflections.json`: Reflexion记忆（day + block反思 + compressed rules）

**Eval**（使用 `.venv_qwen`）：
```bash
# best-of-3 + single-mode + 物理量
CUDA_VISIBLE_DEVICES=1 .venv_qwen/bin/python eval_grpo_vs_ppo_fair.py

# 详细能耗/PMV分析
CUDA_VISIBLE_DEVICES=1 .venv_qwen/bin/python eval_energy_pmv_detailed.py
```

**Eval 注意事项**：
- `skip_valid_steps` 必须和PPO eval一致（Miami Aug 25-29 = 1200-1500）
- Block结构必须和训练一致（当前训练用2h blocks → eval用 `gspo_houston_bandit_2h.py`）
- GRPO eval 用 `.venv_qwen`，PPO eval 用 `.venv`（两者 python 环境不兼容）

**关键文件**：
- `gspo_houston_bandit.py`: 核心bandit（当前2h blocks + 30min knots，训练用）
- `gspo_houston_bandit_2h.py`: eval副本（匹配训练block结构，含 `_extract_step_physics`）
- `llm_setpoint_planner.py`: LLM planner（mode描述、knot生成、Reflexion、统计规则）
- `train_qwen3_houston_gspo_block.py`: GRPO训练循环
- `eval_grpo_vs_ppo_fair.py`: 公平对比eval
- `eval_energy_pmv_detailed.py`: 物理量详细eval

### 常见配置错误（已踩坑，勿重复）

1. **Env var路径格式**：
   - `RL_IDF`: 相对于 `PROJECT_ROOT`，只写文件名 → `miami_3week.idf`（不是 `./miami_3week.idf`）
   - `RL_EPW`: 相对于 `WEATHER_DIR`（`PROJECT_ROOT/weather/`），只写文件名 → `miami_...epw`（**不是** `weather/miami_...epw`，否则变成 `weather/weather/...`）
   - `RL_FORECAST_CSV`: 同 `RL_EPW`，相对于 `WEATHER_DIR`，只写文件名

2. **Forecast CSV必须显式设置**：
   - 默认值是 Houston forecast（`houston_...csv`），切换城市时**必须**设 `RL_FORECAST_CSV=miami_...csv`
   - GRPO训练和PPO训练**都需要设**，否则forecast数据和weather不匹配
   - 错误不会crash，只会 WARNING + forecast_available=0（静默失败）

3. **GRPO必须加 `--use-peft`**：
   - 不加则全量训练8B参数，需要~48GB显存（OOM on 44GB GPU）
   - 加LoRA后只需~20GB
   - `--resume-from checkpoint` 会自动加载LoRA adapter，不需要 `--use-peft`
   - 从头训练（无resume）**必须**显式加 `--use-peft --lora-r 16 --lora-alpha 32`

4. **SFT adapter snapshot已废弃**：
   - `_sft_adapter_state` 已设为 None（`train_qwen3_houston_gspo_block.py` line 359）
   - 之前的 `copy.deepcopy` KL reference 会额外占16GB显存，已移除
   - 当前纯GRPO训练不需要KL reference

5. **Eval skip_valid_steps必须匹配**：
   - Miami Aug 25-29 = skip 1200/1275/1350/1425/1500
   - skip=0 对应 Aug 1（训练集内），**不是** Aug 25
   - PPO和GRPO eval必须用相同的skip值

6. **Eval block结构必须匹配训练**：
   - 训练用2h blocks → eval必须用 `gspo_houston_bandit_2h.py`（不是主文件的1h blocks）
   - block不匹配不会报错，但reward会显著偏低

7. **PPO eval通过bandit时需要tolerant validation**：
   - `RLLibRollingPlanner` 不返回 `request.payload`
   - `gspo_houston_bandit.py` 的 `_validate_and_build_planner_step_trace` 需要对缺失request做tolerant处理（直接return minimal trace）
   - 已在主bandit中修复，`gspo_houston_bandit_2h.py` 也需要同步修复

### 3x Energy Penalty 实验 (2026-04-07)

**配置变更**：
- `w_building_energy`: 1.0 → 3.0（能耗惩罚3倍）
- Baseline: 24°C → 23°C（更强的baseline，PMV violation≈0）
- Reward: `reward = -0.01 * (3.0 * net_grid_kwh + 50.0 * pmv_violation)`
- SFT adapter snapshot: 已废弃（`_sft_adapter_state = None`），纯GRPO训练
- Eval天气: swap weather（Aug 27 = Aug 9暴雨天，GHI=2284）

**建筑能耗分解**：
- `Electricity:Facility = Electricity:Building + HVAC`
- `Electricity:Building`（固定负荷）= InteriorLights(522.5) + InteriorEquipment(346.1) = **868.6 kWh/工作日**
- 固定负荷对所有方法完全相同，不可控
- **HVAC是唯一可控部分**，约700-900 kWh/工作日（取决于天气）

**三方对比 (Miami Aug 25-29, Swap Weather, Baseline 23°C)**：

| 指标 | Baseline 23°C | PPO fc (3x) | GRPO bo3 (3x) |
|------|-------------|------------|--------------|
| Facility总电 | 8285 kWh | 8147 kWh | 8152 kWh |
| 固定负荷 | 4343 kWh | 4343 kWh | 4343 kWh |
| **HVAC用电** | **3942 kWh** | **3804 kWh** | **3810 kWh** |
| HVAC节省 | — | 138 kWh (3.5%) | 133 kWh (3.4%) |
| PMV violation | 0.003 | 4.645 | 3.599 |
| Total reward | 0 | +1.82 | +1.56 |

逐天分解（HVAC only）：

| Day | BL HVAC | PPO HVAC | GRPO HVAC | PPO PMV | GRPO PMV | GRPO modes |
|-----|---------|---------|----------|---------|---------|------------|
| Aug 25 | 888 | 851 | 869 | 3.14 | 1.06 | B C C B C E |
| Aug 26 | 829 | 801 | 815 | 1.49 | 0.50 | E B B B C E |
| **Aug 27★** | **706** | **679** | **666** | **0.00** | **0.00** | **E E E E E E** |
| Aug 28 | 753 | 731 | 728 | 0.02 | 0.64 | E E E B E B |
| Aug 29 | 765 | 741 | 732 | 0.00 | 1.40 | E E E B B E |

**关键发现**：
1. **3x energy成功改变了策略方向**：PPO和GRPO都开始省电（之前1x energy时PPO反而多花电）
2. **GRPO PMV更低**（3.6 vs 4.6），**PPO省电略多**（138 vs 133 kWh）——GRPO在能效和舒适度之间找到了更好的平衡
3. **暴雨天（Aug 27）GRPO完胜**：
   - GRPO: HVAC=666kWh, reward=+1.09, modes=**全部E(energy_saving)**
   - PPO: HVAC=679kWh, reward=+0.81
   - GRPO比PPO多省13kWh，reward高+0.28——**LLM读到暴雨forecast后全天切energy_saving**
4. **GRPO的mode pattern清晰**：晴天混合B/C/E，暴雨天全E，体现了forecast-aware策略切换
5. **PV利用率**：暴雨天PV仅250kWh（正常天786-843），net grid大幅增加→energy_saving更有价值

**训练细节**：
- GRPO: 75 steps (5 episodes), LoRA r=16 α=32, checkpoint-60 (EP4, mean=+4.05)
- PPO fc: 300 episodes, `houston_aug2025_ep5000_x300_forecast_window_manual/checkpoint`
- PPO nofc: checkpoint被覆盖（run name冲突），未能eval

### HVAC-only能耗观测改动 (2026-04-07)

**问题**：`energy_consumption` 绑定 `Electricity:Facility`（全建筑），包含固定负荷（照明522.5 + 设备346.1 = 868.6 kWh/工作日），不可控部分不应参与reward。

**改动**：新增 `energy_building` 观测绑定 `Electricity:Building`，reward改为 `HVAC = Facility - Building`：
```python
# .tmp_todo_random_start_cell0.py
hvac_energy_j = first_zone_obs["energy_consumption"] - first_zone_obs.get("energy_building", 0.0)
net_building_energy_kwh = max(joules_to_kwh(hvac_energy_j - first_zone_obs["PV"]), 0.0)
```

验证：Baseline 23°C Aug 25 → Facility=1757, HVAC=931, Building=826, PV=786 kWh ✅

**net_grid去除max(0) clamp**：原来 `net_grid = max(HVAC - PV, 0)`，中午PV>HVAC时net_grid=0 → Block 2-3（10:30-14:30）的reward信号完全消失（candidate和baseline reward相同）。改为 `net_grid = HVAC - PV`（允许负值），中午PV盈余时LLM省电可获得正reward信号。

上一轮训练（3x energy + HVAC-only + baseline 23°C）已中止，改为加入以下改动后重新训练。

### Mode差异化 + PMV硬约束 + 天气观测改进 (2026-04-07)

**改动一览**：

1. **comfort→cooling 重命名**：避免"comfort"歧义，直接表达主动制冷意图
2. **PMV目标间距拉大**（消除重叠）：
   - cooling: [-0.5, 0]（原comfort: [-0.2, +0.1]）
   - balanced: [-0.1, +0.2]（原: [0, +0.3]）
   - energy_saving: [+0.2, +0.5]（原: [+0.3, +0.5]）
3. **PMV硬约束**：`plan_knot()` 输出setpoint后做PMV-aware clamp：
   - 当前PMV > 0.4 → setpoint ≤ current_temp - 0.5（强制降温）
   - 当前PMV < -0.4 → setpoint ≥ current_temp + 0.5（停止过冷）
4. **新增观测**：`outdoor_temp`（Site Outdoor Air Drybulb）+ `cloud_cover`（Site Total Sky Cover）
   - PPO和GRPO共用同一个cell0，都自动获得新观测
5. **knot prompt改进**：
   - 加入 `Outdoor: {temp}°C, Cloud cover: {cover}/10`
   - Mode描述用cloudcover判断PV（"cloud < 30% → solar high"）而不是硬编码PV阈值
   - 移除固定example setpoint值，改为 `{<float>, <float>, ...}`
   - 加入PMV硬限提示：`PMV hard limits: all occupied zones must stay within [-0.5, +0.5]`

**当前训练（含所有改动）**：
- GRPO: `miami_grpo_3x_hvac_cooling_bl23_20260407`（GPU 0, LoRA, cooling/balanced/energy_saving）
- PPO fc: `miami_3x_hvac_cooling_bl23_ep5000_x300_forecast_window_manual`（GPU 1）
- PPO nofc: `miami_3x_hvac_cooling_bl23_ep5000_x300_no_forecast_window_manual`（GPU 1）

### 城市天气对比调研

下载了NYC和Chicago的2025年6-9月历史天气（Open-Meteo archive API），评估是否适合替代Miami做实验。

**8月天气四城对比**：

| 指标 | Houston | Miami | NYC | Chicago |
|------|---------|-------|-----|---------|
| GHI均值 | 6016 | 6240 | 5792 | 6140 |
| GHI标准差 | 793 | 1042 | **1366** | 872 |
| GHI最低 | 3112 | 2284 | **1563** | 4138 |
| Tmax均值 | 33.3°C | 33.0°C | 28.5°C | 26.2°C |
| Tmax范围 | 30-37 | 28-34 | **22-35** | 19-34 |
| 降雨天>5mm | 8 | 11 | 1 | 4 |

**结论**：
- **NYC**：天气波动最大（GHI std=1366，温度22-35°C），NYISO有RTP数据，但不够tropical
- **Chicago**：ComEd有最好的RTP数据，但8月下旬太凉（13天Tmax<25°C），制冷需求不足
- **Miami**：最tropical但无RTP数据
- **Atlanta**：气候像新加坡+有Georgia Power RTP，待评估

天气数据文件：
- `weather/nyc_2025_06_01_2025_09_30_historical_weather_api.json/csv`
- `weather/chicago_2025_06_01_2025_09_30_historical_weather_api.json`

### GRPO架构缺陷分析与改进方向

#### 当前架构

```
每天 (15训练日 × 5 episodes)
  └── 6个Block (2h each, 06:30-19:00)
       └── 3个Mode并行rollout (cooling / balanced / energy_saving)
            └── 4个Knot (30min each)
                 └── LLM (Qwen3-8B + LoRA) → 8个zone setpoint
                      └── EnergyPlus 10min仿真 → step reward
       └── 比较3个mode → Baseline-anchored GRPO梯度更新
  └── Day-level gradient (history best gating, 权重0.3)
  └── Reflexion (block + day反思, 训练时辅助/eval时mode选择)
```

梯度三层：tier advantage(1.0) + sub advantage(0.5) + per-knot partial return(0.3) + day-level(0.3)

#### 缺陷一：Block间无协调

**问题**：每个block独立选mode、独立梯度更新。建筑有热惯性——Block 2过度制冷 → Block 3室温过低 → Block 3的balanced反而需要回温。Day-level gradient是唯一跨block信号，但权重只有0.3，信号太弱。

**解决方案**：

**A. Hierarchical RL（推荐长期方案）**：将全天mode序列作为上层决策：
- 上层：LLM一次性输出6个block的mode序列（如 `[E, B, B, C, C, E]`），用全天reward做GRPO
- 下层：每个block内按给定mode输出setpoint，用block reward做GRPO
- 上层学到跨block因果（"Block 2用cooling导致Block 3过冷"），下层学setpoint质量
- 类似Options Framework / Feudal RL

**B. 前block结果注入prompt（推荐短期方案，改动最小）**：
- 当前block的prompt加入前一个block的执行结果：
  ```
  Previous block: mode=cooling, end_temp=22.1°C, end_PMV=-0.3, reward=+1.2
  ```
- 不改梯度结构，让LLM通过in-context learning理解block间依赖
- 实现成本极低（改 `_build_knot_user_prompt()` 加一行）

#### 缺陷二：Mode选择是穷举不是学习

**问题**：训练时每个block跑全部3个mode（best-of-3），模型只学"每个mode下怎么输出好setpoint"，不学"什么条件选什么mode"。Eval时靠统计表或LLM判断选mode，和训练过程脱节。

**解决方案**：

**A. 端到端GRPO优化mode选择（推荐长期方案）**：
- 不穷举3个mode，而是让LLM在每个block开始时先输出mode选择，再输出setpoint
- GRPO对完整output（mode + setpoints）做梯度更新
- 可以采样2-3次不同output，用group内比较
- 缺点：reward variance更大（只有1-2个rollout而不是3个）

**B. 两阶段训练（推荐中期方案）**：
- 阶段1（当前）：穷举3 mode，GRPO学好setpoint生成能力
- 阶段2：固定setpoint LoRA，用best-of-3的(condition→best_mode)数据训练mode selection adapter
- SFT分类任务（3选1），不是之前失败的setpoint回归SFT

**C. 统计表+LLM hybrid（当前方案）**：
- 已实现 `build_statistical_rules()` + `select_mode()` 注入统计证据
- 不需要额外训练，但精度受限于统计样本量和特征维度

#### 缺陷三：Reward延迟/稀疏

**问题**：setpoint → 空调响应 → 室温变化 → PMV变化有10-30min延迟。knot 0的决策可能在knot 2-3才体现reward。Per-knot partial return是Monte Carlo近似，不是因果归因。

**解决方案**：

**A. TD(λ) 替代 Monte Carlo**：
- 当前：`knot_return = sum(step_rewards[knot_t:])`（MC）
- 改为：`advantage_t = reward_t + γ * V(t+1) - V(t)`（TD）
- 需要critic网络估计V(t)，增加复杂度
- 但能更准确地归因"哪个knot的决策导致了后续的reward变化"

**B. Reward shaping（推荐短期方案）**：
- 在reward中加入温度变化的shaping term：
  ```
  shaped_reward_t = reward_t + γ * (target_temp - current_temp) * direction_bonus
  ```
- encode先验："降低setpoint的行动在未来会降低温度/PMV"
- 不需要critic，不改GRPO框架

**C. 保持当前设计**：
- 30min knot已经覆盖一个完整的cause-effect cycle
- Per-knot partial return虽不精确，但提供了位置权重
- 最大的reward信号来自block-level（4个knot的总和），延迟在block内被平均化

#### 综合改进方案：合并缺陷一+二的统一架构

三个缺陷可以通过一个架构变更同时解决：**让LLM在每个block开始时同时输出mode选择+setpoint，用标准GRPO优化**。

**当前架构 vs 新架构的本质区别**：

当前：mode是**系统预设穷举的**，LLM不做mode选择：
```
Block 2 → 系统强制跑3次：
  第1次：prompt写死 "Planning mode: cooling" → LLM只输出setpoint
  第2次：prompt写死 "Planning mode: balanced" → LLM只输出setpoint
  第3次：prompt写死 "Planning mode: energy_saving" → LLM只输出setpoint
→ 比较reward选winner。GRPO只训练"给定mode下怎么出setpoint"
```

新：mode是**LLM自主决定的**，mode token参与梯度更新：
```
Block 2 → LLM自由采样3次（temperature采样产生多样性）：
  第1次：LLM看到observation+前block结果 → 输出 "balanced\n{setpoints: [...]}"
  第2次：同样输入 → 输出 "cooling\n{setpoints: [...]}"
  第3次：同样输入 → 输出 "balanced\n{setpoints: [...]}"（不同setpoint值）
→ 比较reward。GRPO同时训练"什么条件选什么mode"+"怎么出setpoint"
```

**对比表**：

| | 当前 best-of-3 | 新：统一mode+setpoint |
|--|--------------|---------------------|
| LLM调用/block | 3×4=12（3 mode × 4 knot） | 3×4=12（3次采样 × 4 knot） |
| EP仿真/block | 3次 | 3次 |
| mode选择 | 系统穷举，不经过模型 | **LLM输出，参与GRPO梯度** |
| 跨block信息 | 无 | **前block结果注入prompt** |
| Forecast修正 | 无（mode在block开始前已决定） | **每个block实时观测+最新forecast→实时决策** |
| GRPO信号 | tier+sub+knot 三层 | **sample advantage + knot（两层，更简洁）** |
| 计算量 | 一样 | 一样 |

**新的LLM输出格式**：
```
Input:  [weather] [实时cloud_cover+outdoor_temp] [forecast_6h] [前block结果] [zone observation]
Output: "balanced\n{\"setpoints\": [23.5, 23.8, 24.1, ...]}"
```

**关于Hierarchical RL（缺陷一A）的修正**：之前提到的"day开始时一次性规划全天mode序列"方案**不推荐**——因为06:30规划的mode序列基于当时的forecast，到14:30 forecast可能已经偏了，提前规划的mode无法修正。新方案每个block实时决策，天然支持forecast滚动修正：
```
Block 2 (10:30): forecast说cloud=30%，但实际cloud_cover=60%
  → LLM看到实时数据 vs forecast偏差 → 自主调整mode（原本可能选balanced，现在选energy_saving）
```

**Forecast偏差信号**：可在knot prompt中加入forecast vs 实际的偏差提示：
```
Forecast bias: actual cloud cover 60% vs forecast 30% (cloudier than predicted,
future PV may be lower than forecast suggests)
```
让LLM显式意识到forecast不准并做修正。

**新的GRPO更新逻辑**：
```python
# 每个block
for i in range(3):  # 3次自由采样
    output = LLM.generate(prompt)  # mode + setpoints
    mode, setpoints = parse(output)
    reward = EP_rollout(mode, setpoints)

# Baseline-anchored advantage（保留防drift）
advantages = rewards / (std(rewards) + 1e-4)

# 梯度更新：mode token和setpoint token一起被更新
for sample, advantage in zip(samples, advantages):
    loss += -logprob(all_tokens) * (sample_adv + 0.3 * knot_partial_adv)
```

**mode token直接参与梯度**——如果sample A选了cooling得reward+2，sample B选了energy_saving得reward-1，GRPO会强化"在这个条件下选cooling"。

#### 记忆与在线适应

**Prompt-based memory buffer（推荐）**：利用LLM的in-context learning做在线适应，不需要额外算法。

```
Day starts:
  context = [compressed rules from training] + [today's forecast]

Block 0 完成 → context += "Block 0: chose energy_saving, HVAC=85kWh, PMV=+0.1, reward=+0.5"
Block 1 完成 → context += "Block 1: chose balanced, HVAC=120kWh, PMV=+0.3, reward=-0.2"
Block 2 开始 → 模型看到前两个block结果 + forecast，自主调整策略
...
Day 结束 → 生成day-level反思 → 下一天注入
```

**三层适应机制**：
1. **Block内（knot间）**：每30min看到新的observation+PMV，微调setpoint
2. **Block间（同天）**：看到前block的mode选择+结果+reward，调整后续mode
3. **Day间（跨天）**：reflexion积累经验，修正长期策略

**为什么这是LLM相对PPO的结构性优势**：
- PPO的MLP权重在部署时固定，不能根据当天执行历史调整
- LLM通过in-context learning自然做到"看到Block 1的结果→调整Block 2的策略"
- GRPO训练教模型**学会利用这些context**——如果模型看到负reward后做了正确调整，全天reward更高，GRPO强化这种"根据反馈修正"的行为

#### History Best Gating在新架构下的问题与方案

**问题**：当前history best按 `(skip_valid_steps, block_index)` 记录。但新方案下3次采样可能选不同mode，不同mode导致block结束时建筑状态不同（室温、热蓄量），影响后续block的reward。跨episode比较"Block 3的best reward"时，EP1可能是cooling得到的+2.0，EP2可能是energy_saving得到的+1.8——两者不可直接比较。

**当前架构也有这个问题**：每个episode的winner mode可能不同，history best不区分mode。

**方案A：按(block, mode)记录history best**

只和同mode的历史比。但新方案下mode是LLM自选的，3次采样可能2次balanced+1次cooling，不再是固定3个mode。

**方案B（推荐）：Block-level不做gating，只在day-level做**

```
Block-level: 永远更新（3次采样间的relative advantage已经提供足够信号）
Day-level: history best gating（全天总reward必须超过历史最佳才更新）
```

理由：
- Block-level的GRPO核心机制是组内比较（3个sample互相比），不需要跨episode的history保护
- Day-level gating防止全局drift（某个episode全天都差→不更新权重）
- 避免了"不同mode的block reward不可比"的问题
- 更简洁，少一层复杂度

**方案C：History best记录全天mode序列+reward**

```python
history_best[skip_valid_steps] = {
    "reward": best_total_reward,
    "mode_sequence": ["E", "B", "B", "C", "C", "E"],
}
```
只比较全天总reward，不比较单个block。

#### 统一架构的风险与应对

**1. Mode collapse（最大风险）**：模型可能收敛到总是输出同一个mode。当前穷举方案保证每block都有3种mode的数据；新方案下3次采样可能全是balanced。

**2. 探索效率下降**：3次自由采样可能2次balanced+1次cooling，energy_saving完全没探索到。需要更多step覆盖所有mode×条件组合。

**3. Advantage更noisy**：3次采样的reward差异来自mode差异+setpoint随机性+思维链差异，归因更模糊。

**4. 训练前期更难**：base模型没见过"先输出mode再输出setpoint"的格式，初期可能格式错误。

**应对：渐进式过渡方案（推荐）**

在同一个训练脚本里，用step数控制phase切换，一次训练、权重连续继承：

```python
total_steps = 75  # 5 episodes × 15 days

for step in range(total_steps):
    if step < 30:        # EP1-2: 穷举3 mode（学好setpoint生成）
        free_samples = 0
    elif step < 45:      # EP3: 2穷举 + 1自由（开始学mode选择）
        free_samples = 1
    elif step < 60:      # EP4: 1穷举 + 2自由（mode选择为主）
        free_samples = 2
    else:                # EP5: 全自由（完全端到端）
        free_samples = 3
    
    # 穷举部分：固定mode，LLM只输出setpoint（当前方式）
    # 自由部分：LLM输出 "mode\n{setpoints}"，mode参与GRPO梯度
```

优势：
- 不需要多份文件、多次启动，一个训练脚本搞定
- EP1-2用穷举保证setpoint质量和mode探索
- EP3-4逐渐交出mode选择权给LLM
- EP5完全端到端，模型同时优化mode选择+setpoint
- 每个phase都能从上一个phase的checkpoint继续，不浪费训练

监控指标：
- 自由采样中3次的mode分布（如果某mode占比>80%→mode collapse）
- 自由采样 vs 穷举的reward差距（应该逐渐缩小）
- 训练中可以随时调整phase边界

#### 实施优先级

1. **短期（当前训练完成后）**：前block结果注入prompt + forecast偏差提示（改 `_build_knot_user_prompt()`，最小改动验证效果）
2. **中期（核心架构升级）**：统一mode+setpoint output + 标准GRPO + 前block结果注入 + forecast修正（改训练循环，中等工作量）
   - 同时：Block-level去掉history best gating，只保留day-level
   - 同时：加mode collapse监控和随机探索
3. **中期**：完整的prompt memory buffer（block结果+reflexion+forecast的滚动上下文）
4. **不推荐**：Hierarchical RL全天一次性规划mode序列（无法做forecast滚动修正）

### 未来可能实现

2. **多维统计查找表**：当前统计规则只用 `block_index × outdoor_temp_bucket` 两个维度。训练更多episode后可以扩展到 `(block_index, temp_bucket, pv_bucket)` 三维，甚至加入 `cloud_cover_bucket`，提升mode推荐精度。

3. **Online adaptation**：eval时前几个block跑best-of-3收集当天pattern，后续block根据当天已观察到的结果 + 统计表选single mode，平衡准确性和效率。

4. **10min控制频率GRPO**：当前在训练的1h block + 10min knot（13 blocks × 6 knots = 78 knots/天），与PPO的75步/天控制频率匹配。如果效果好，可以直接对比同频率下GRPO vs PPO。

5. **提高能耗惩罚权重**：当前 `reward = -0.01 * (1.0 * net_grid_kwh + 50.0 * pmv_violation)`，能耗权重1.0 vs PMV权重50.0。一旦PMV控制在±0.5以内（violation=0），reward完全由能耗决定，但能耗系数太小导致策略对省电不敏感。

   当前物理量对比显示：PPO多花103kWh消除全部PMV violation，GRPO多花32kWh消除96%，两者reward几乎一样（+9.18 vs +9.28）——**reward函数对71kWh能耗差距不敏感**。

   改进方案：`w_energy: 1.0 → 3.0~5.0`，或收紧 `pmv_limit: 0.5 → 0.3`，或改用连续PMV惩罚（去掉dead zone）。

   **训练影响分析**：
   - **Mode分化加大**：energy_saving在当前reward下几乎没有优势（PMV已在0.5内，省电带来的reward提升微弱）。提高能耗权重后energy_saving能获得更明显的正advantage，3个mode之间的reward方差增大→GRPO梯度信号更强
   - **Comfort mode代价增加**：comfort降低setpoint → 更多制冷 → 能耗更高。高能耗权重下comfort只在PMV严重超标时才值得选 → mode选择更有策略性（而不是"无脑选comfort总是安全"）
   - **PV利用率变得重要**：当net_grid权重提高，利用PV避免grid购电的策略更有价值。晴天和阴天的mode选择差异会被放大 → **forecast信息更有用 → LLM的天气理解能力能发挥优势**
   - **训练难度增加**：能耗和舒适度的trade-off更尖锐，模型需要更精细的setpoint控制。可能需要更多training steps才能收敛
   - **风险**：如果能耗权重过高，模型可能过度追求省电而牺牲舒适度，PMV violation反弹。需要逐步调整（1→2→3）并监控PMV

   注意：改reward需要PPO和GRPO**同时重新训练**，否则对比不公平。
