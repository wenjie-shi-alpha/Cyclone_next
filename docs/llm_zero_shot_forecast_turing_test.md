# LLM 零样本预报修正图灵测试方案

## 1. 背景与动机

### 1.1 两个项目的关系

- **MOSNet** (`/root/MOSNet/`): 研究热带气旋预报误差的物理来源，以及预报员（专家）对数值模式预报的修正技能。核心发现是：预报误差并非随机，而是通过 6 条可识别的物理路径（failure pathway）组织起来的，而预报员在特定路径被触发时做出方向性的、针对性的修正。
- **Cyclone_next** (`/root/Cyclone_next/`): 将预报员在发布预报时参考的全部信息（环境诊断、观测证据、模式指导等）结构化后，通过大语言模型微调（SFT + GRPO）学习从输入到官方预报表的映射。

### 1.2 核心问题

微调后的大语言模型是否真正"学会"了预报员的物理推理过程，还是仅仅在记忆模式输出？一个直接的检验方式是：**将同样的输入信息直接给未经微调的大语言模型 API，看它能否做出类似预报员的修正行为**——这本质上是一个图灵测试。

## 2. 预报误差的物理来源（MOSNet）

MOSNet 识别了 6 条物理失效路径，每条路径有明确的环境控制变量和误差响应变量：

| # | 失效路径 | 控制变量 | 响应变量 | 可修复比例 |
|---|---------|---------|---------|-----------|
| 1 | 副高引导位移 (Ridge-steering displacement) | 副高距离 (km) | 横向误差 (km) | 67.5% |
| 2 | 槽时序错配 (Trough-timing misalignment) | 槽距离 TC (km) | 横向误差 (km) | 97.4% |
| 3 | 弱引导持续 (Weak-steering persistence) | 引导气流速度 (m/s) | 沿轨漂移 (km) | 92.8% |
| 4 | 切变-强度解耦 (Shear-intensity decoupling) | 切变强度 (m/s) | 风速误差 (kt) | 61.0% |
| 5 | 季风槽组织误差 (Monsoon-trough organization) | 轴线长度 (km) | 路径误差 (km) | 56.0% |
| 6 | 海洋-外流快速增强低估 (Ocean-outflow RI) | 暖水面积 (10³ km²) | RI 低估 (kt) | 88.9% |

### 2.1 专家修正的具体数据

预报员对数值模式预报的修正效果（HRES 模式为例）：

| 机制区 | 样本数 | 模式误差 (km) | 官方误差 (km) | 专家增益 (km) | 改善率 | 统计显著性 |
|-------|--------|-------------|-------------|-------------|-------|----------|
| Regime 0 | 16,507 | 125.9 | 117.1 | 8.7 | 6.9% | 显著 |
| Regime 1 | 80 | 207.0 | 165.6 | 41.4 | 20.0% | 显著 |
| Regime 5 | 5,664 | 242.2 | 189.8 | 52.4 | 21.6% | 显著 |

GFS 模式在 Regime 5 的改善更显著（52.4 km，21.6%），说明预报员在弱引导/高不确定性场景中发挥了更大的修正价值。

### 2.2 预报员推理的语言证据

从 NOAA 历史讨论文本中可以看到，预报员的修正决策基于明确的物理推理：

- **副高修正**: "The subtropical ridge to the south of Bonnie located over southern Florida is beginning to shift slowly northward" → 预报路径向西北调整
- **槽时序修正**: "Bonnie has finally made the long expected turn to the north through a break in the subtropical ridge" → 预报加速向东北
- **切变修正**: "Abundance of very dry air surrounding the cyclone...that Bonnie will have to navigate through for the next 24 to 48 hours" → 强度预测偏保守
- **引导减弱**: "Should begin to accelerate significantly to the northeast in 24-30 hours" → 速度修正

## 3. 微调数据集的输入-输出结构（Cyclone_next）

当前微调使用的唯一数据集位于 `data/training_rebuilt_v2_20260414_guidancefix/`，包含 SFT 和 RL 两套数据。

### 3.1 System Prompt

```
You are a tropical cyclone forecaster. Use only the evidence and guidance
provided in the prompt. Return only the official forecast table. The first
line must be exactly 'Official forecast:'. Each remaining non-empty line
must be exactly '- DayDD HHMMZ | LAT LON | NN kt'. Do not output reasoning,
risk text, markdown headings, prose, or any additional text.
```

### 3.2 User Prompt 结构（预报员接收的全部信息）

用户提示按以下 6 个板块组织：

#### 板块 1: 时间锚点
```
## Time Anchor
- Advisory issue Day13 2100Z
```

#### 板块 2: 当前状态
```
## Current State
- Position 27.1°N 30.8°W | Motion NORTHEAST at 12 kt | Intensity 45 kt / 990 mb
```

#### 板块 3: 环境诊断
```
## Environmental Diagnostics
- Vertical wind shear: 10.3 m/s (strong) — generally unfavorable for organization
- Upper-level divergence: -2.53 ×10⁻⁵ s⁻¹ (negative) — upper-level convergence, unfavorable
- Sea surface temp / OHC: 24.8 °C (marginal) — marginal ocean support
- Subtropical high: 5820 gpm (moderate) — moderate steering influence
- Westerly trough: 5540 gpm (weak)
```

这 5 项环境诊断恰好对应 MOSNet 的 5 条主要失效路径的驱动因子：
- 垂直风切变 → 切变-强度解耦 (Pathway 4)
- 上层辐散 → 海洋-外流快速增强低估 (Pathway 6)
- 海温/OHC → 海洋-外流快速增强低估 (Pathway 6)
- 副高 → 副高引导位移 (Pathway 1) + 弱引导持续 (Pathway 3)
- 西风槽 → 槽时序错配 (Pathway 2)

#### 板块 4: 发报前 HRES 分析（可选）
```
## Pre-Issue HRES Analysis
- 27.5°N 30.0°W | 42.0 kt, 992.0 hPa
```

#### 板块 5: 观测证据
```
## Observation Evidence
- Coverage: GOES + ASCAT available; missing Recon
- GOES: min Tb 171.9 K; p10 Tb 202.4 K; inner cold frac 0.37
- ASCAT: max wind 48 kt; mean wind 28 kt; 34-kt radius 165 km
```

观测数据来源：
- **GOES**: 红外卫星观测（最低亮温、冷云比例等）
- **ASCAT**: 散射计地面风观测（最大风速、平均风速、大风半径）
- **Recon**: 飞机侦察观测（VDM 最低气压、飞行层风速）

#### 板块 6: 模式指导
```
## Model Guidance

Consensus guidance (56 models):
- Day14 0600Z 27.0°N 27.7°W | 39.2 kt, 993 hPa | spread 1167 km/6.84 kt
- Day14 1800Z 29.5°N 25.1°W | 40.0 kt, 992 hPa | spread 1345 km/8.12 kt
- Day15 0600Z 32.1°N 23.8°W | 41.5 kt, 990 hPa | spread 1520 km/9.50 kt
- Day15 1800Z 35.0°N 22.5°W | 40.0 kt, 992 hPa | spread 1790 km/11.2 kt
- Day16 0600Z 37.8°N 21.0°W | 38.0 kt, 994 hPa | spread 2100 km/13.5 kt

Deterministic HRES:
- Day14 0600Z 27.8°N 28.5°W | 42 kt, 991 hPa
- Day14 1800Z 30.2°N 26.0°W | 44 kt, 989 hPa
- Day15 0600Z 33.0°N 24.5°W | 43 kt, 990 hPa
```

### 3.3 期望输出格式

```
Official forecast:
- Day14 0600Z | 28.9°N 29.7°W | 45 kt
- Day14 1800Z | 32.0°N 28.2°W | 45 kt
- Day15 0600Z | 36.3°N 27.5°W | 45 kt
- Day15 1800Z | 39.5°N 25.8°W | 42 kt
- Day16 0600Z | 42.0°N 24.0°W | 40 kt
```

### 3.4 验证数据（RL 训练使用，不提供给模型）

```json
{
  "future_best_track": [
    {"lead_from_issue_h": 3, "lat": 27.9, "lon": -30.4, "vmax_kt": 55.0},
    {"lead_from_issue_h": 15, "lat": 30.5, "lon": -28.0, "vmax_kt": 50.0}
  ],
  "best_track_at_issue": {
    "lat": 27.2, "lon": -30.9, "storm_speed": 24.8, "storm_direction": 41.6
  },
  "forecast_slots": [
    {"valid_time_utc": "2016-01-14T06:00:00Z"},
    {"valid_time_utc": "2016-01-14T18:00:00Z"}
  ]
}
```

## 4. 图灵测试实验设计

### 4.1 核心思路

将 Cyclone_next 中预报员使用的输入信息，原封不动地交给未经微调的大语言模型 API，观察：

1. **格式遵循能力**: 能否严格遵循 `Official forecast:` 表格格式
2. **物理一致性**: 输出是否与环境诊断信息一致（如强切变下强度偏保守）
3. **修正行为**: 是否对模式指导做出了类似预报员的方向性修正
4. **误差结构**: 零样本输出与最佳路径的误差分布，与微调模型和原始模式输出的误差分布对比

### 4.2 实验变量

| 变量 | 取值 |
|-----|------|
| 模型 | Claude Opus 4.6 / Claude Sonnet 4.6 / GPT-4o / Gemini 2.5 Pro |
| Prompt 策略 | A: 仅 strict forecast system prompt; B: 加入物理推理指引; C: Chain-of-thought |
| 测试集 | sft_test.jsonl (1,613 样本) |
| 评估指标 | 路径误差 (km), 强度误差 (kt), 格式合规率, 修正方向一致性 |

### 4.3 Prompt 策略设计

#### 策略 A: 零样本严格格式（与微调训练同构）

直接使用与 SFT 训练相同的 system prompt 和 user prompt 结构，不做任何额外提示。

#### 策略 B: 零样本 + 物理推理指引

在 system prompt 中补充 6 条失效路径的物理知识：

```
You are a tropical cyclone forecaster. When adjusting model guidance,
consider these known error pathways:

1. Ridge-steering displacement: When the subtropical high is mispositioned,
   models may have cross-track bias. If ridge is stronger/closer than model
   assumes, adjust track toward the ridge side.
2. Trough-timing misalignment: If a westerly trough is approaching but
   model timing is uncertain, consider earlier or later recurvature.
3. Weak-steering persistence: In weak steering environments, models tend
   to maintain momentum; consider deceleration or stalling.
4. Shear-intensity decoupling: Strong vertical wind shear decouples
   upper-level outflow from surface circulation; bias intensity downward.
5. Monsoon-trough organization: Monsoon trough interactions can cause
   unexpected track changes; consider track variability.
6. Ocean-outflow RI support: High OHC + favorable outflow can trigger
   rapid intensification that models systematically underestimate; bias
   intensity upward in these conditions.

Return only the official forecast table. The first line must be exactly
'Official forecast:'. Each remaining non-empty line must be exactly
'- DayDD HHMMZ | LAT LON | NN kt'.
```

#### 策略 C: Chain-of-Thought 推理

要求模型先输出推理过程，再输出预报表：

```
You are a tropical cyclone forecaster. First analyze the environmental
conditions and model guidance step by step. Identify which physical
pathways may cause model errors. Then adjust the guidance accordingly.
Format your reasoning as bullet points under '## Analysis', then output
the forecast table.

## Analysis
- [your reasoning here]

Official forecast:
- DayDD HHMMZ | LAT LON | NN kt
```

### 4.4 评估框架

#### 4.4.1 格式合规评估

使用与微调训练相同的正则表达式解析输出：

```python
pattern = r"^-\s*Day(?P<day>\d{1,2})\s+(?P<hh>\d{2})(?P<mm>\d{2})Z\s*\|\s*" \
          r"(?P<lat>\d+(?:\.\d+)?)°(?P<lat_hemi>[NS])\s+" \
          r"(?P<lon>\d+(?:\.\d+)?)°(?P<lat_hemi>[EW])\s*\|\s*" \
          r"(?P<vmax>\d+(?:\.\d+)?)\s*kt$"
```

指标：
- 格式合规率: 能被成功解析的预报行占比
- 行数匹配率: 输出行数与指导行数一致的比例

#### 4.4.2 预报误差评估

使用与 `cyclone_training/rewards.py` 相同的计算方式：
- **路径误差**: Haversine 距离 (km)
- **强度误差**: 绝对风速误差 (kt)
- **时间匹配容差**: ±6 小时

对比以下三者的误差分布：
1. 原始模式指导（Consensus/HRES）
2. 零样本 LLM 输出
3. 微调 LLM 输出
4. 官方预报（NHC OFCL）

#### 4.4.3 修正方向一致性评估

这是图灵测试的核心：**LLM 是否做出了与预报员相同方向的修正？**

对于每个样本，计算：
- **模式→官方修正向量**: `adjustment = official_forecast - model_guidance`
- **模式→LLM 修正向量**: `adjustment = llm_forecast - model_guidance`
- **修正方向一致性**: 两个修正向量的方向是否一致（cosine similarity > 0）

按 MOSNet 的 6 条失效路径分层统计，特别关注：
- 在高置信度路径触发样本中，LLM 是否做出了正确的方向性修正
- 修正幅度是否合理（与专家修正幅度对比）

#### 4.4.4 路径分层分析

利用 MOSNet 的路径评分系统，将测试样本按主导路径分组：

| 路径 | 预期 LLM 行为 | 关键指标 |
|-----|-------------|---------|
| Ridge-steering | 在副高偏强时将路径向副高一侧调整 | cross-track 修正方向 |
| Trough-timing | 在有西风槽接近时考虑更早/晚转向 | cross-track 修正方向 |
| Weak-steering | 在弱引导下考虑减速或停滞 | along-track 修正方向 |
| Shear-intensity | 在强切变下偏保守估计强度 | 强度偏移方向 |
| Monsoon-trough | 考虑季风槽交互带来的路径变化 | 路径变异性 |
| Ocean-outflow RI | 在高 OHC+有利外流下偏强估计强度 | 强度偏移方向 |

### 4.5 样本选择策略

从 1,613 个测试样本中，选择以下子集进行重点分析：

1. **高专家增益样本** (n ≈ 200): 专家修正幅度最大的样本（官方预报 vs 模式指导差距大），这是图灵测试最有区分力的场景
2. **路径触发样本** (n ≈ 400): 按 6 条路径各选 60-70 个高置信度样本
3. **全量样本** (n = 1,613): 用于整体统计

### 4.6 实验流程

```
1. 数据准备
   ├── 从 sft_test.jsonl 提取 user prompt
   ├── 从 rl_test.jsonl 提取 verification 数据
   ├── 从 MOSNet 路径评分系统获取路径标签
   └── 按样本选择策略划分子集

2. API 调用
   ├── 对每个样本 × 每个模型 × 每个 prompt 策略，调用 LLM API
   ├── 记录原始输出、token 使用量、延迟
   └── 对格式不合规的输出做容忍处理（尝试宽松解析）

3. 误差计算
   ├── 解析 LLM 输出为结构化预报
   ├── 与 best_track 对齐计算路径/强度误差
   ├── 计算修正向量（LLM输出 - 模式指导）
   └── 与官方修正向量对比

4. 统计分析
   ├── 整体误差分布对比（模式 vs 零样本LLM vs 微调LLM vs 官方）
   ├── 路径分层修正方向一致性
   ├── 物理合理性检验（环境条件与修正方向的一致性）
   └── 图灵测试判别：能否区分零样本LLM和预报员
```

## 5. 预期结果与解读

### 5.1 可能的实验结果

| 场景 | 零样本表现 | 解读 |
|-----|----------|------|
| A: 零样本接近随机 | 格式不合规，误差远大于模式 | LLM 无内生的热带气旋预报能力，微调提供了核心知识 |
| B: 零样本接近模式 | 格式合规，输出 ≈ 模式指导 | LLM 学会了"抄模式"，但不做修正；微调的价值在于学会修正 |
| C: 零样本部分修正 | 某些路径有方向性修正，但不完整 | LLM 有部分物理推理能力（来自预训练知识），但不系统；微调补全了系统修正能力 |
| D: 零样本接近预报员 | 修正方向与官方高度一致 | LLM 预训练中已编码了足够的气象知识，微调主要是格式适配 |

### 5.2 各结果的教学意义

- **场景 A/B**: 证明微调不仅是格式适配，而是真正灌入了领域知识
- **场景 C**: 最可能的结果。说明预训练提供了一定的物理推理基底，微调将其系统化和强化。可进一步分析哪些路径的推理来自预训练、哪些来自微调
- **场景 D**: 说明当前微调的效率不高（零样本已接近最优），可考虑更小的微调数据量或更高效的微调方法

## 6. 与 MOSNet 的深度整合

### 6.1 利用 MOSNet 路径评分作为评估框架

MOSNet 的路径评分系统可以为图灵测试提供物理可解释的分层：

```
MOSNet 路径评分 → 测试样本路径标签 → 分层评估 LLM 修正行为
```

这比简单的误差统计更有诊断价值，因为它能回答：**LLM 在哪条物理路径上做出了正确修正，在哪条路径上失败？**

### 6.2 利用 MOSNet 环境特征构建增强 Prompt

MOSNet 的 257 维环境特征（8 个天气系统对象）比 Cyclone_next 当前使用的 5 项环境诊断更丰富。可以考虑：

- 将 MOSNet 的天气系统对象描述（如副高边界点、槽底部位置等）融入 prompt
- 观察更丰富的环境描述是否能提升零样本修正的准确性
- 这也能验证：MOSNet 识别的关键物理特征是否真的对 LLM 推理有帮助

### 6.3 构建反事实测试

MOSNet 的反事实修复实验（counterfactual repair）提供了一个天然的测试集：

- 在 MOSNet 已证明"修复某个环境变量可大幅降低误差"的样本中
- 测试 LLM 是否在修正该环境变量描述后，能做出相应的修正
- 这相当于控制变量的物理实验，可以精确验证 LLM 的物理推理链

## 7. 实施清单

- [ ] 从 `sft_test.jsonl` 提取测试样本的 user prompt
- [ ] 从 `rl_test.jsonl` 提取对应的 verification 数据
- [ ] 编写 LLM API 调用脚本（支持多模型、多 prompt 策略）
- [ ] 编写输出解析器（严格格式 + 宽松格式兼容）
- [ ] 编写误差计算模块（复用 `cyclone_training/rewards.py` 的逻辑）
- [ ] 从 MOSNet 获取路径标签，关联到测试样本
- [ ] 选择高专家增益子集和路径触发子集
- [ ] 运行零样本 API 测试
- [ ] 统计分析与可视化
