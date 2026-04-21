# cycloneCopilot 旧项目 Review：教训、可复用资产与新数据集规范

## 1. 项目对比概况

| 维度 | 旧项目 (cycloneCopilot) | 新项目 (Cyclone_next) |
|------|------------------------|----------------------|
| 训练样本量 | 978 train / 64 val / 57 test | 3,984 train / 1,370 val / 1,632 test |
| 数据切分 | 随机 split 字段（同风暴可交叉） | 事件级时间切分 + 冻结 manifest |
| 基座模型 | Qwen3-14B / QwQ-32B / DeepSeek-R1-70B | TBD |
| 训练框架 | Unsloth (4-bit QLoRA) + TRL | TBD |
| 观测数据 | 仅 HRES track + CDS 环境 | GOES + ASCAT + Recon + HRES + CDS + ATCF |
| 防泄露 | 无（同风暴可同时出现在 train/test） | 事件级切分 + 匿名化 + 泄露审计 |
| RL 方法 | GRPO (混合奖励, ~0 均值) | GRPO (结构化 verification 奖励) |
| 数据格式 | prompt/response 对 (COT + FORECAST 混合) | chat 格式 (system/user/assistant) |
| Prompt 语言 | 英文 | 英文 |
| CDS 环境描述 | 中文长文 (~200 字/特征) | 英文压缩 (value + level + 一句话摘要) |

---

## 2. 核心问题诊断（修正版）

### 问题 1: 数据量严重不足 — 978 条训练样本

旧项目仅有 978 条训练样本，这对 14B 参数的模型来说远远不够：
- Qwen3-14B 有 ~140 亿参数，即使 LoRA 只训练 ~0.5% 参数，也需要数千到数万高质量样本
- 6 epoch 训练导致严重过拟合（验证损失回升、生成文本重复）
- GRPO 仅 300 步，reward 始终在 0.001-0.006 极低区间震荡

**新项目优势**：3,984 条训练样本，是旧项目的 **4 倍**，且每条包含更丰富的多源数据（GOES/ASCAT/Recon/HRES/ATCF/CDS）。

### 问题 2: GRPO 奖励信号几乎为零 — 训练无效

旧项目 GRPO 训练的关键指标：
- `reward mean` 始终在 **0.001-0.006** 区间（几乎为零）
- `frac_reward_zero_std` 高达 **0.85-0.94**（90% 的样本 reward 为零）
- KL 散度在 0.4-1.5 波动（不稳定）

根因分析：
1. **轨迹提取困难**：模型输出自由文本，需要 regex + LLM 提取轨迹点，提取失败率极高
2. **forecast_truth 覆盖不全**：仅有 8 个时效点（INIT/12H/.../120H），难以对齐
3. **奖励函数过度复杂**：forecast_reward(55%) + cot_reward(35%) + base_reward(10%)，三路奖励都有噪声
4. **生成长度仅 ~220 tokens**：GRPO max_completion_length=1200，但实际生成远短于此，说明模型学到了"安全但无信息"的输出策略
5. **格式奖励主导**：后期 GRPO 配置中 forecast_weight=1.0, cot_weight=0.0，但仍需 regex 提取预报表格，失败率极高

**新项目改进**：
- verification_targets 直接提供 B-deck 最佳路径，无需从自由文本提取
- reward 基于结构化数值对比（track MAE + intensity MAE），信号更清晰
- 多时效点验证（每 6h 一个点），覆盖更完整

### 问题 3: COT + FORECAST 混合输出 — 格式学习困难

旧项目要求模型输出 `### COT ### ... ### FORECAST ###` 的混合格式：
- COT 部分是自然语言推理（~1500 tokens）
- FORECAST 部分是格式化表格（~500 tokens）
- 模型很难同时学好两种截然不同的输出模式
- SFT 后模型生成的 COT 经常是"模板式解释"，缺少具体要点（Recall 仅 0.46）
- FORECAST 表格覆盖率仅 63%（SFT），GRPO 反而降到 32%

**新项目改进**：
- target 分层：track_intensity_table（结构化）+ risk_messages + reasoning_text
- 格式更紧凑，减少格式学习负担
- 结构化输出和自由文本分离，降低学习难度

### 问题 4: 过拟合严重 — 验证集表现不如训练集

- SFT 3 epoch 后验证损失回升
- 6 epoch 时生成文本重复、退化
- `num_train_epochs: 3` + `max_steps: 240` 的组合在小样本上仍然过多
- LoRA dropout=0.0 加剧了过拟合

**新项目改进**：
- 训练样本 4 倍增加，天然缓解过拟合
- 应设 `lora_dropout: 0.05`
- 建议 2 epoch + early stopping
- 监控 eval loss，出现回升立即停止

### 问题 5: 无防泄露机制 — 训练/测试可能信息泄漏

旧项目使用 `split` 字段做随机切分，没有：
- 事件级隔离（同一风暴可能同时在 train 和 test）
- 时间切分（test 可能包含 train 年份的风暴）
- 匿名化（storm_name, date 直接暴露）
- 泄露审计

这意味着模型可能通过"记忆事件"而非"理解规律"获得高分。

**新项目改进**：
- 事件级时间切分（train ≤ 2020, val 2021-2022, test ≥ 2023）
- 零交叉污染验证（180 train / 67 val / 78 test 风暴，互不重叠）
- 4 组测试集（主/匿名/结构化/扰动）
- 完整泄露审计

### ~~问题 6: Prompt 设计过于"引导式"~~ → 修正：Guidance 是合法预报输入

> **重要修正**：旧项目将 HRES 未来轨迹放入 prompt 作为 Guidance，**这不是数据泄露，而是正确的做法**。实际预报业务中，预报员确实参考 HRES/GFS 等模式输出。模型需要学习"在模式指导基础上做出合理判断"，而非"从零独立推理完整预报"。

旧项目 Guidance 部分的真正问题是：
1. **Guidance 过于详细**：完整暴露 HRES 120h 轨迹每一个时效点的 lat/lon/wind/pressure，模型可能过度依赖（"抄 HRES"），但这不是泄露，而是训练信号稀释问题
2. **缺少多模式对比**：只有 HRES 单模式，没有 GFS/UKMET 等对比，模型无法学习"多模式分歧时如何抉择"
3. **OFCL/CARQ 泄露**：guidance 中包含了 OFCL（官方预报本身），这才是真正的泄露——模型可以直接抄官方预报

**新项目改进**：
- guidance_inputs 包含 HRES + ATCF 多模式共识/离散度（56+ 模型）
- OFCL/CARQ 已从 guidance 中排除（ATCF_BLOCKED_GUIDANCE_MODELS）
- 只提供共识统计量 + spread，不暴露单模式完整轨迹（降低"抄模式"倾向，但不排除合法参考）

---

## 3. 旧项目可复用资产评估

### 3.1 可直接复用

| 资产 | 路径 | 评估 |
|------|------|------|
| **奖励函数架构** | `src/cyclone_copilot/evaluation/reward.py` | 核心可复用。haversine 距离、track error、intensity RMSE、event detection 逻辑完全可迁移。但需适配新的结构化输出格式（不再需要 regex 提取） |
| **SFT/GRPO 训练流水线** | `src/cyclone_copilot/finetune/sft.py`, `grpo.py` | 框架可复用（基于 Unsloth + TRL）。chat template 渲染、gradient checkpointing、LoRA 配置等逻辑可直接使用 |
| **训练配置模板** | `configs/sft_config_*.yaml`, `grpo_config_*.yaml` | 可作为新配置的起点。注意修正 lora_dropout=0.05, epochs=2 |
| **GRPO 停止字符串补丁** | `grpo.py` 中的 `_patch_transformers_stop_strings_tokenizer` | 直接可用。解决了 TRL GRPO + stop_strings 的兼容性问题 |

### 3.2 需要适配

| 资产 | 路径 | 需要的修改 |
|------|------|-----------|
| **奖励计算** | `evaluation/reward.py` → `RewardScorerConfig` | 需要从"regex 提取 + 对齐"改为"直接对比 verification_targets"。新的 reward 不再需要 `_align_timesteps`（已有时效对齐数据），直接按 lead_from_issue_h 做 MAE 计算 |
| **Prompt 模板结构** | 旧 prompt v4/v5 | 整体结构可参考：当前状态 → 环境诊断 → 模式指导 → 任务指令。但新版已改为 chat 格式，system prompt 已内置，无需在 user prompt 中重复任务指令 |
| **GRPO 默认奖励函数** | `grpo.py` 中的 `default_reward` | 基于 SequenceMatcher 的字符串相似度完全不可用。需替换为基于 verification_targets 的结构化数值奖励 |
| **EventRule 系统** | `reward.py` → `EventRule` | 概念可复用（RI/landfall/track turn 检测），但需基于 verification_targets 的 future_best_track 而非从模型输出提取 |

### 3.3 不应复用

| 资产 | 原因 |
|------|------|
| **旧数据格式 (prompt/response)** | 新项目使用 chat 格式 (system/user/assistant)，更符合 LLM 训练范式 |
| **COT + FORECAST 混合输出格式** | 已证明此格式导致模型顾此失彼，FORECAST 覆盖率从 63% 降到 32% |
| **regex 轨迹提取逻辑** | 根因之一。新项目 verification_targets 直接提供结构化 B-deck 数据，无需提取 |
| **随机数据切分** | 已证明有泄露风险，新项目使用事件级时间切分 |
| **lora_dropout=0.0 配置** | 在小样本上严重过拟合，必须设为 0.05 |

### 3.4 Prompt 模板演进对比

**旧项目 v4/v5 Prompt**（单轮文本，~2400 chars）：
```
You are a tropical cyclone forecaster. Using only the data below, draft the
official forecast discussion that would be issued at {issue_time} UTC.

### DATA CONTEXT ###
Basin: {basin} | Storm: {name} ({id}) | Issue: {issue}
Best track: start ... latest ...
Guidance (multi-model): Model: HRES v1 Track: 0h ... 120h ...
Environment (multi-model): Env 0h: MT(...) | TROUGH(...) ...
Reanalysis environment: {cds_desc_chinese}

### TASK ### Write the forecast discussion in NHC TCD style. Include:
### DISCUSSION ### (analysis text)
### FORECAST ### (8-line table: INIT/12H/24H/36H/48H/72H/96H/120H)
Hard rules: 1) Keep hemisphere consistent ... Never change N/S ...
```

**新项目 Prompt**（chat 格式，~690-1250 chars user content）：
```
[System] You are a tropical cyclone forecaster. Based on the current
observations, environmental diagnostics, and model guidance provided, issue
the official forecast including track/intensity projections, risk messages,
and reasoning.

[User]
## Current State
- Position: {lat}°N {lon}°W | Moving: {dir} at {spd}kt
- Intensity: {vmax}kt, {pmin}mb

## Environmental Diagnostics
- Vertical wind shear: {vws} m/s (strong) — shear significantly inhibits development
- Upper-level divergence: {div} ×10⁻⁵ s⁻¹ (moderate)
- Sea surface temp / OHC: {sst} °C (low) — insufficient ocean energy
- Subtropical high: {sh} gpm (moderate) — moderate steering
...

## Observation Evidence
Satellite IR (GOES): cloud_top_temp_min_k: 195.2 K (QC:ok) ...
Scatterometer (ASCAT): wind_max_inner_kt: 45.3 kt ...
Reconnaissance aircraft: vdm_min_slp_mb: 985 mb ...

## Model Guidance
HRES forecast:
  +24h: 28.5°, -75.2° | 35kt, 1005hPa
  +48h: 30.1°, -77.8° | 30kt, 1008hPa
  ...
Multimodel consensus (56 models):
  Lead  Lat     Lon     Vmax  MSLP   TrackSpread WindSpread
  +12h  28.2°   -75.8°  33.2kt 1006hPa   45km     3.1kt
  +24h  29.0°   -76.5°  30.5kt 1007hPa   82km     5.2kt
  ...
```

**关键改进**：
1. **Chat 格式**：system prompt 定义角色，user 提供数据，assistant 输出预报——更自然
2. **CDS 英文化**：中文长描述压缩为 `{value} {unit} ({level}) — one-line summary`，token 减少 ~60%
3. **观测数据结构化**：GOES/ASCAT/Recon 以 key-value 列出，无需自然语言描述
4. **多模式共识表**：不再只列 HRES 单模式，提供 spread 信息
5. **OFCL/CARQ 排除**：guidance 中不含官方预报（防泄露）
6. **输出分离**：track table + risk info + reasoning，不再混在 COT + FORECAST 中

---

## 4. 新数据集完整规范

### 4.1 数据规模与切分

| 切分 | 风暴数 | 样本数 | 时间范围 | 说明 |
|------|--------|--------|----------|------|
| Train | 180 | 3,984 | 2016-2020 | 含 ASCAT/GOES/Recon/HRES/ATCF/CDS |
| Val | 67 | 1,370 | 2021-2022 | 同上 |
| Test A (主) | 78 | 1,632 | 2023-2025 | 完整信息 |
| Test B (匿名) | 78 | 1,636 | 2023-2025 | 风暴名/日期匿名化 |
| Test C (结构化) | 78 | 1,636 | 2023-2025 | 仅结构化数据，无自然语言推理 |
| Test D (扰动) | 78 | 1,636 | 2023-2025 | 环境参数轻微扰动 |

**零交叉污染**：train/val/test 的 storm_id 完全不重叠，已通过 split_manifest_v1.json 冻结。

### 4.2 Token 分布

| 指标 | SFT Train | SFT Val | SFT Test | RL Train |
|------|-----------|---------|----------|----------|
| Prompt 中位数 | 439 | 420 | 330 | 439 |
| Prompt P90 | 770 | 469 | 417 | 770 |
| Prompt 最大 | 921 | 921 | 482 | 921 |
| Target 中位数 | 137 | 144 | 132 | — |
| Target P90 | 155 | 375 | 140 | — |
| Target 最大 | 1,300 | 1,329 | 1,293 | — |
| 总计中位数 | 593 | 567 | 460 | 439 |

**关键观察**：
- 总 token 中位数 ~590，P90 ~970，远低于旧项目的 ~4500（旧项目 COT + FORECAST 格式臃肿）
- Target 中位数 ~137 tokens，远低于旧项目的 ~700 tokens
- RL 格式仅含 prompt + verification，不含 target（由模型生成）

### 4.3 SFT 数据格式

```jsonl
{
  "sample_id": "Atlantic_2020186N30289_2020-07-06T03:00:00Z_007",
  "messages": [
    {
      "role": "system",
      "content": "You are a tropical cyclone forecaster. Based on the current observations, environmental diagnostics, and model guidance provided, issue the official forecast including track/intensity projections, risk messages, and reasoning."
    },
    {
      "role": "user",
      "content": "## Current State\n- Position: 27.1°N 30.8°W | Moving: NORTHEAST at 12kt\n- Intensity: 45kt, 990mb\n\n## Environmental Diagnostics\n- Vertical wind shear: 10.31 m/s (strong) — shear significantly inhibits development\n- Upper-level divergence: -2.53 ×10⁻⁵ s⁻¹ (negative)\n- Sea surface temp / OHC: 22.05 °C (low) — insufficient ocean energy, likely weakening\n- Subtropical high: 57668.2 gpm (moderate) — moderate steering\n...\n\n## Observation Evidence\n[GOES/ASCAT/Recon structured data or 'No observation data available']\n\n## Model Guidance\nHRES forecast:\n  +24h: 28.5°, -75.2° | 35kt, 1005hPa\n  ...\nMultimodel consensus (56 models):\n  Lead  Lat     Lon     Vmax  MSLP   TrackSpread WindSpread\n  +12h  28.2°   -75.8°  33.2kt 1006hPa   45km     3.1kt\n  ..."
    },
    {
      "role": "assistant",
      "content": "Official forecast:\n  Day14 0600Z: 28.9°, -29.7° | 45kt\n  Day14 1800Z: 32.0°, -28.2° | 45kt\n  ...\n\n## Risk Information\nPublic advisory summary:\n  LOCATION...27.1N 30.8W\n  ...\n\n## Forecast Reasoning\nCurrent analysis: ...\nForecast reasoning: ...\nAdditional: ..."
    }
  ]
}
```

**Target 结构完整率**：track table 99.3%, risk info 99.9%, reasoning 100%。

### 4.4 RL/GRPO 数据格式

```jsonl
{
  "sample_id": "Atlantic_2020186N30289_2020-07-06T03:00:00Z_007",
  "messages": [
    {"role": "system", "content": "...(同 SFT)"},
    {"role": "user", "content": "...(同 SFT)"}
  ],
  "verification": {
    "future_best_track": [
      {
        "lead_from_issue_h": 3,
        "valid_time_utc": "2016-01-14T00:00:00Z",
        "lat": 27.9,
        "lon": -30.4,
        "vmax_kt": 55.0,
        "min_pressure_mb": 988.0,
        "storm_phase": "BEST",
        "source_used": "atcf_b_deck"
      },
      ...  // 平均 12-14 个点，每 6h 一个
    ],
    "best_track_at_issue": {
      "matched_datetime_utc": "2016-01-13T21:00:00Z",
      "lat": 27.2,
      "lon": -30.9,
      "storm_speed": 24.8,
      "storm_direction": 41.6
    },
    "reward_config": {
      "track_error_weights": {
        "12h": 0.08, "24h": 0.12, "48h": 0.18,
        "72h": 0.22, "96h": 0.20, "120h": 0.20
      },
      "intensity_error_weight": 0.4,
      "track_error_weight": 0.4,
      "reasoning_quality_weight": 0.2,
      "high_impact_bonus": {
        "rapid_intensification": 0.5,
        "landfall": 0.3,
        "track_turn": 0.2
      }
    }
  }
}
```

**RL vs SFT 的关键区别**：
- RL 不含 assistant 消息（由模型生成）
- RL 包含 verification 字段（用于 reward 计算）
- verification 中的 `future_best_track` 是 B-deck 真实路径（不进入 prompt，仅用于奖励）

### 4.5 奖励函数设计

```python
# 建议的 GRPO 奖励架构（基于旧项目 reward.py 改进）

def compute_reward(model_output: str, verification: dict) -> float:
    """结构化数值奖励，不再依赖 regex 提取。"""

    # 1. 从 model_output 解析结构化预报表格
    forecast_points = parse_track_table(model_output)  # 简单 regex，格式已由 SFT 学会

    # 2. 对齐 forecast_points 与 verification.future_best_track
    truth = verification["future_best_track"]
    track_mae = compute_weighted_track_mae(
        forecast_points, truth,
        weights=verification["reward_config"]["track_error_weights"]
    )
    intensity_mae = compute_intensity_mae(forecast_points, truth)

    # 3. 加权组合
    track_reward = exp(-track_mae / 250.0)  # 250km scale
    intensity_reward = exp(-intensity_mae / 15.0)  # 15kt scale
    reasoning_reward = compute_reasoning_quality(model_output)  # 可选

    total = (
        0.4 * track_reward +
        0.4 * intensity_reward +
        0.2 * reasoning_reward
    )

    # 4. 高影响事件奖励（RI/landfall/track turn 正确识别）
    bonus = detect_high_impact_events(forecast_points, truth, verification["reward_config"])
    total += bonus

    return max(total, 0.0)
```

**与旧项目奖励函数的关键区别**：
1. **不再需要 regex 从自由文本提取轨迹**：SFT 已教会模型输出结构化表格
2. **直接对比 B-deck verification**：无需 `_align_timesteps` 插值，lead_from_issue_h 直接对齐
3. **奖励分布有足够方差**：MAE 在 0-1000km 范围内，`exp(-x/250)` 给出 0-1 的平滑奖励
4. **高影响事件检测**：使用 verification 中的 truth 而非模型输出

### 4.6 数据源覆盖

| 数据源 | 训练集可用率 | 说明 |
|--------|------------|------|
| NOAA Forecast Advisory | ~100% | 当前状态 + 官方预报表 |
| NOAA Forecast Discussion | ~100% | 推理文本（target only） |
| CDS Environment | ~100% | 7 项环境诊断（英文压缩） |
| HRES Track/Env | ~85% | 模式指导 |
| ATCF A-deck | ~70% | 多模式共识 + spread |
| GOES Satellite IR | ~60% | 9 项 IR 指标 |
| ASCAT Scatterometer | ~40% | 9 项风场指标 |
| Recon Aircraft | ~15% | 8 项实测指标 |
| ATCF B-deck (verification) | ~70% | 仅用于 reward，不进 prompt |

---

## 5. 训练建议（供专家判断）

### 5.1 任务定义

**核心任务**：给定当前观测 + 环境诊断 + 模式指导，生成包含以下三部分的官方热带气旋预报：
1. **Track/Intensity Table**：未来 120h 的位置和强度预报（结构化表格）
2. **Risk Messages**：watch/warning 信息 + 公众预警摘要
3. **Forecast Reasoning**：当前分析 + 预报逻辑 + 额外背景

这是一个**结构化生成 + 自然语言推理的混合任务**。

### 5.2 SFT 阶段建议

```yaml
# 基于旧项目教训的推荐配置
model: Qwen3-14B 或更大  # 旧项目验证了 14B 的可行性
lora_r: 32              # 14B+ 用 32（旧项目用 16 可能不够）
lora_alpha: 32
lora_dropout: 0.05      # 旧项目用 0.0 导致严重过拟合
learning_rate: 1e-4
num_train_epochs: 2     # 旧项目 6 epoch 严重过拟合
max_seq_length: 2048    # 新数据 P90 ~970 tokens，2048 足够
batch_size: 4
gradient_accumulation: 4  # 有效 batch=16
warmup_ratio: 0.06
eval_steps: 100
save_steps: 200
early_stopping_patience: 3
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
# 旧项目已验证这些 target modules 的有效性
```

### 5.3 GRPO/RL 阶段建议

```yaml
# 关键改进：基于 verification_targets 的结构化奖励
adapter_path: <sft_checkpoint>/final  # 从 SFT 检查点继续
beta: 0.2               # 旧项目 GRPO 用的 0.2
max_steps: 200           # 旧项目 300 步 reward 已经饱和，200 步足够
num_generations: 4       # 每个样本生成 4 个候选
max_completion_length: 1024
temperature: 0.3         # 旧项目 0.2 太保守
learning_rate: 3e-5      # RL 用更小的学习率

# 奖励：结构化数值对比，不再用 regex 提取
reward_components:
  track_mae_weight: 0.4
  intensity_mae_weight: 0.4
  reasoning_quality_weight: 0.2
```

### 5.4 模型选择建议

| 模型 | 参数量 | 优点 | 缺点 | 推荐度 |
|------|--------|------|------|--------|
| Qwen3-8B | 8B | 显存友好，训练快 | 容量可能不足，旧项目未验证 | 低 |
| Qwen3-14B | 14B | 旧项目已验证可行性，平衡性能与成本 | 需 4bit + LoRA | 中 |
| Qwen3-32B | 32B | 更强推理能力 | 显存需求大，旧项目 QwQ-32B SFT 效果更好 | 高 |
| DeepSeek-R1-Distill-70B | 70B | 最强推理 | 训练慢，需多卡，旧项目验证了可行性但 reward 改善有限 | 中 |

**推荐路径**：Qwen3-14B 起（快速验证数据流水线），验证后升级到 32B。

### 5.5 必须避免的错误（来自旧项目教训）

1. **不要让模型同时输出 COT + 格式化表格** — 旧项目 FORECAST 覆盖率从 63% 降到 32%
2. **不要在小样本上跑过多 epoch** — 2 epoch + early stopping
3. **不要用 lora_dropout=0** — 至少 0.05
4. **不要用随机切分** — 必须事件级时间切分
5. **不要在 GRPO 中用从自由文本提取的 reward** — 用结构化数值对比
6. **不要在 test 集上反复调参** — 冻结 test，只用 val 调参
7. **不要在 guidance 中包含 OFCL/CARQ** — 这是真正的泄露
8. **不要把 reasoning_text 放入 prompt** — 它是 target 的一部分
9. **不要把 verification_targets 放入 prompt** — 硬约束

### 5.6 评估策略

基于旧项目的评估教训（模型精度仅 0.201 vs 人类 0.491），建议：

1. **SFT 后立即评估 track/intensity MAE**（旧项目在 SFT 后就开始生成测试）
2. **GRPO 训练中监控 reward 分布**：确保非零奖励比例 > 50%，reward 方差 > 0.1
3. **使用 4 组 test 集做全面评估**：
   - Test A：主评估（完整信息）
   - Test B：匿名化（检测记忆 vs 推理）
   - Test C：结构化（检测对自然语言的依赖）
   - Test D：扰动（鲁棒性）
4. **与 HRES 基线对比**：旧项目发现"模型差于 HRES"是常态，需要至少持平才有价值

---

## 6. 文件清单

### 新项目关键文件

| 文件 | 用途 |
|------|------|
| `scripts/build_dataset_sample_preview_v0_1.py` | 单样本构建器（EDOUARD 示例） |
| `scripts/build_dataset_batch.py` | 批量构建（manifest 驱动） |
| `scripts/dataset_formatter.py` | SFT/RL 格式转换 |
| `scripts/data_leakage_prevention.py` | 防泄露流水线 |
| `data/training/sft_{train,val,test}.jsonl` | SFT 训练数据 |
| `data/training/rl_{train,val,test}.jsonl` | RL 训练数据 |
| `data/training/sft_test_{anonymous,structured_only,perturbation}.jsonl` | 测试集变体 |
| `data/training/raw/{train,val,test}/*.json` | 完整 JSON 快照 |
| `data/interim/leakage_prevention/split_manifest_v1.json` | 冻结的切分方案 |
| `data/training/build_report.json` | 构建统计 |

### 旧项目可复用文件

| 文件 | 用途 |
|------|------|
| `src/cyclone_copilot/evaluation/reward.py` | 奖励函数架构（haversine, event detection） |
| `src/cyclone_copilot/finetune/sft.py` | SFT 训练流水线 |
| `src/cyclone_copilot/finetune/grpo.py` | GRPO 训练流水线（含 stop_strings 补丁） |
| `src/cyclone_copilot/finetune/config.py` | 训练配置类定义 |
| `configs/sft_config_qwen3_14b.yaml` | SFT 配置模板 |
| `configs/grpo_config_qwen3_14b_v4_format_150.yaml` | GRPO 配置模板 |
