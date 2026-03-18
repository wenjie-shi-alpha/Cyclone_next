---

# Paper 2 (B)：完整研究设计

---

# 一、Title

**首选：**

> **Learning to reason about tropical cyclones: An AI system that surpasses human expert forecasts through physical understanding and learned judgment**

**备选：**

> **Beyond post-processing: A reasoning AI that integrates physical diagnosis with expert judgment to set new benchmarks in tropical cyclone forecasting**

> **Physical reasoning meets expert intuition: A large language model framework for tropical cyclone forecast optimization**

**推荐首选**——"Learning to reason"准确捕捉了核心贡献，区别于所有已有工作。"surpasses human expert forecasts"是最强的结果声明。

---

# 二、Abstract

> Current AI weather models have transformed atmospheric prediction by learning physical dynamics from data, but they address only one half of the operational forecast chain: the numerical integration of governing equations. The other half — the reasoning process by which human experts interpret model outputs, assess their reliability under specific environmental conditions, and synthesize competing information sources into a single forecast decision — has remained beyond the reach of artificial intelligence. This expert reasoning, recently shown to be precisely concentrated in environmentally conditioned error regimes where numerical models exhibit structured failures, represents the last major untapped source of forecast skill.

> Here we present [SystemName], an AI framework that learns to reason about tropical cyclone forecasts at three integrated levels. First, a physical diagnosis module ingests object-based environmental representations from model-predicted fields and identifies which error regime the current forecast is likely operating in, providing structured awareness of model vulnerability. Second, a large language model, fine-tuned on decades of expert forecast adjustment records contextualized with environmental diagnostics, learns the conditional reasoning patterns that forecasters deploy — when to trust which model, how to weight competing signals, and how much to adjust under different storm–environment configurations. Third, reinforcement learning with verification-based rewards refines this learned reasoning, retaining the expert's genuine skill while filtering out systematic human biases including conservatism, anchoring, and under-reaction to emerging threats.

> Evaluated on North Atlantic and eastern North Pacific tropical cyclones (2016–2025), [SystemName] produces forecasts that systematically outperform all individual numerical models, multi-model consensus, and human expert forecasts for both track and intensity across all lead times, with the largest gains during rapid intensification and complex multi-system interaction scenarios — precisely the situations that matter most for protective decision-making. Crucially, each forecast is accompanied by an explicit reasoning trace that references specific physical mechanisms, enabling meteorologists to evaluate, trust, or override the AI's judgment.

> Ablation experiments demonstrate that neither the physical diagnosis module nor the language model architecture alone accounts for the full improvement: it is their integration — structured physical awareness enabling contextual reasoning — that produces gains unattainable by conventional machine learning approaches operating on the same input information. Our results establish that the next frontier in AI weather prediction is not higher-resolution physics emulation, but the learning of situated judgment: the capacity to reason about what forecasts mean, when they might fail, and how to act on imperfect information.

---

# 三、Narrative（叙事逻辑）

## 全文一句话叙事

> **预报的最后一公里不是计算——是判断。我们构建了第一个能像专家一样推理、并最终超越专家的 AI 预报推理系统。**

## 展开的六段叙事

**第一段：AI 天气预报的盲区**

> AI 天气模型（Pangu-Weather, GraphCast, FuXi）在替代数值积分方面取得了惊人进展。但业务预报链条中，数值积分只是上游——下游是人类专家的判断：看多个模式、评估哪个可信、判断当前环境下模式可能犯什么错、然后做出一个综合决策。这个"判断"环节至今没有被任何 AI 系统建模过。

**第二段：Paper 1 的发现为什么至关重要**

> 我们的前序工作 [Paper 1, Author et al.] 揭示了一个关键事实：专家判断的价值不是泛泛的"经验丰富"，而是高度选择性的——它精确集中在那些环境配置导致模式产生结构化错误的 regime 中。这意味着专家做的不是"随机微调"，而是一种**有物理根基的条件化推理**：识别当前环境配置 → 判断模式在此配置下的可信度 → 做出有方向性的调整。

**第三段：为什么这是一个"推理"任务而不是"回归"任务**

> 如果专家只是在做 $f(x) \to y$ 的映射（输入特征→调整量），那任何 ML 模型都够了。但专家做的是：在不同条件下采用不同的推理策略——有时信任 ECMWF 而忽略 GFS，有时反过来；有时做大幅调整，有时故意不动；有时基于最新卫星信息推翻所有模式。这种条件化、多策略、需要整合异质信息的认知过程，本质上是推理，而不是回归。这就是为什么需要语言模型——它是目前唯一能建模这种推理结构的架构。

**第四段：三层架构的逻辑**

> 我们构建了三层递进的系统。第一层让 AI "看懂"环境——通过物理诊断模块识别当前预报处于哪种误差 regime，以及模式可能在哪里犯什么错。第二层让 AI "学会推理"——通过在大量专家决策记录上微调 LLM，使其学会专家在不同环境配置下的推理模式。第三层让 AI "超越人类"——通过 RL 对齐真实验证结果，保留专家的真实技能，过滤掉专家自身的系统性偏差。

**第五段：结果——不只是更准，而是更透明**

> 系统不仅在所有维度上超越了现有预报（包括专家），而且每次预报都附带一个可阅读的推理过程——说明它为什么做出这个调整、基于什么物理判断、信任了哪个模式。这意味着预报员可以审查、理解、信任或推翻 AI 的建议——这不是一个黑箱替代人类，而是一个可协作的推理伙伴。

**第六段：更大图景**

> 这不仅仅是一个更好的 TC 预报系统。它示范了一种新的 AI 范式：不是替代物理计算，而是学习人类在不确定性下的判断过程。这种"学会推理"的能力可以推广到任何存在"模式输出→专家判断→最终决策"链条的领域。

---

# 四、核心科学问题

| # | 问题 | 意义 |
|---|------|------|
| **Q1** | 人类专家的条件化推理模式能否被 AI 系统学会？ | 确立"专家推理可学习"的基本可行性 |
| **Q2** | 将物理诊断信息（误差 regime 识别）注入推理过程，是否显著增强 AI 的预报调整能力？ | 证明物理理解对推理的必要性 |
| **Q3** | 语言模型是否比传统 ML 架构更适合建模专家的条件化推理？如果是，具体在哪些场景下优势最大？ | 为"LLM"的使用提供不可替代性的证据 |
| **Q4** | RL 能否在保留专家真实技能的同时系统性地消除人类认知偏差？RL 过滤了哪些偏差？ | 证明"超越专家"的机制 |
| **Q5** | AI 推理系统是否能发现专家从未做过、但应该做的调整模式？ | 最高层次的贡献——AI 做了人类未做到的事 |
| **Q6** | AI 产生的推理过程在物理上是否正确、可被领域专家理解和信任？ | 可解释性的硬证据 |

---

# 五、需要做的具体工作

## 阶段一：数据与输入构建

| 任务 | 具体内容 | 说明 |
|------|---------|------|
| **1.1** 预报记录收集 | 所有模式（GFS, ECMWF, HWRF/HAFS, UKMO 等）对每个 TC 每个报次的 track + intensity guidance | ATCF |
| **1.2** 专家预报收集 | NHC Official forecast (OFCL)，逐报次逐时效 | ATCF |
| **1.3** 验证数据 | Best track | IBTrACS / ATCF |
| **1.4** 环境特征 | Paper 1 的对象化环境表示——从模式预报场中提取 | 复用 Paper 1 框架 |
| **1.5** 误差 regime 标签 | Paper 1 的误差模态分类器对每个预报实例的诊断结果 | 复用 Paper 1 模型 |
| **1.6** 专家调整计算 | Track adjustment (CTE/ATE) + Intensity adjustment (signed) = Official − Consensus/Guidance | 计算 |
| **1.7** 多模式共识构建 | 各报次的 consensus track + intensity（多种共识定义） | 计算 |
| **1.8** 附加信息（如可获取） | Forecast Discussion text（NHC 的文字讨论）、近期模式表现统计 | NHC archives |

## 阶段二：训练样本构建

| 任务 | 具体内容 | 说明 |
|------|---------|------|
| **2.1** 输入 prompt 设计 | 将每个预报实例编码为结构化文本输入 | 核心设计决策 |
| **2.2** 输出 target 设计 | 专家调整量 + （可选）推理过程的文本化 | 需要仔细设计 |
| **2.3** 训练/验证/测试划分 | 时间切分（严格避免未来泄漏） | 如 2016-2021 train, 2022-2023 val, 2024-2025 test |

**2.1 输入 prompt 的详细设计**：

```
=== TROPICAL CYCLONE FORECAST CONTEXT ===

Storm: [Name], [Basin], [Date/Time]
Current Position: [lat/lon]
Current Intensity: [kt] ([Category])
Current Motion: [direction] at [speed] kt
Recent Trend: [intensifying/steady/weakening] over past [N] hours

=== MODEL GUIDANCE ===

         12h    24h    48h    72h    96h    120h
GFS:     [lat/lon, kt] ...
ECMWF:   [lat/lon, kt] ...
HWRF:    [lat/lon, kt] ...
UKMO:    [lat/lon, kt] ...
Consensus: [lat/lon, kt] ...

Model Spread (track): [km at each lead time]
Model Spread (intensity): [kt at each lead time]

=== ENVIRONMENTAL DIAGNOSIS ===

[From Paper 1's object-based representation]

Subtropical Ridge: [position relative to TC, strength, trend]
Upper-Level Trough: [position, approach rate, depth]
Vertical Wind Shear: [magnitude, direction relative to motion, trend]
SST/OHC: [current, along-track forecast]
Other TCs: [if any, position/distance]

=== ERROR REGIME DIAGNOSIS ===

[From Paper 1's classifier]

Predicted Error Regime: [regime name] (probability: [X%])
Historical bias in this regime: Track [direction, magnitude], 
                                 Intensity [sign, magnitude]
Expert typically adds value in this regime: [Yes/No, magnitude]

=== RECENT MODEL PERFORMANCE ===

[Over past 5 forecasts for this storm]
GFS track bias: [magnitude, direction]
ECMWF track bias: [magnitude, direction]
GFS intensity bias: [magnitude, sign]
ECMWF intensity bias: [magnitude, sign]

=== TASK ===

Based on all available information, provide:
1. Your forecast adjustment relative to consensus
2. Your reasoning for this adjustment
```

**2.2 输出 target 的设计**：

**SFT 阶段**：

```
=== ADJUSTMENT ===
Track: [CTE adjustment km, ATE adjustment km] at each lead time
Intensity: [adjustment kt] at each lead time

=== REASONING ===
[由人工或规则生成的推理文本，解释为什么专家做了这个调整]
```

**关于 Reasoning 的生成**：
- 如有 NHC Forecast Discussion 文本 → 直接提取关键判断句
- 如无 → 基于 adjustment 方向 + 环境特征 + 误差 regime，用规则模板生成合理的推理文本
- 这是一个设计决策：SFT 不需要完美的推理文本，只需要方向正确且与物理一致

## 阶段三：物理诊断模块（Layer 1）

| 任务 | 具体内容 | 方法 |
|------|---------|------|
| **3.1** 误差 regime 分类器 | 复用 Paper 1 的分类器，或训练一个改进版本 | 来自 Paper 1 |
| **3.2** 条件化偏差估计 | 在每种 regime 下，估计各模式的条件化偏差分布 | 条件统计 |
| **3.3** 物理诊断文本化 | 将诊断结果转化为 prompt 中的文本段落 | 模板 + 规则 |

**这一层不需要单独的 DL 模型——它是 Paper 1 框架的直接应用，为 LLM 提供结构化的物理背景知识。**

## 阶段四：LLM 监督微调（Layer 2 — SFT）

| 任务 | 具体内容 | 方法 |
|------|---------|------|
| **4.1** 基础模型选择 | Llama-3 / Qwen-2.5 / Mistral 等开源 LLM | 比较多个基座 |
| **4.2** SFT 训练 | 在（prompt, expert_adjustment）对上微调 | Standard SFT, LoRA/QLoRA |
| **4.3** 推理格式训练 | 同时学习 adjustment 数值和 reasoning 文本 | Chain-of-thought style |
| **4.4** 超参数调优 | 学习率、epoch、LoRA rank 等 | 验证集上调优 |
| **4.5** SFT 评估 | 在验证集上评估 adjustment accuracy + reasoning quality | 定量+定性 |

## 阶段五：强化学习优化（Layer 3 — RL）

| 任务 | 具体内容 | 方法 |
|------|---------|------|
| **5.1** Reward 设计 | 基于验证结果的综合 reward | 见下文详细设计 |
| **5.2** RL 训练 | 在 SFT 模型基础上进行 RL 优化 | PPO / GRPO / DPO |
| **5.3** 约束设计 | 防止 RL 退化（reasoning 崩溃、过拟合极端值） | KL penalty, reward shaping |
| **5.4** RL 评估 | 与 SFT 对比：skill 提升、偏差消除、推理质量变化 | 全面对比 |

**Reward 设计**：

$$R = -\alpha \cdot E_{\text{track}} - \beta \cdot E_{\text{intensity}} + \gamma \cdot \text{Bonus}_{\text{high-impact}}$$

其中：
- $E_{\text{track}}$：track MAE（多时效加权）
- $E_{\text{intensity}}$：intensity MAE（多时效加权）
- $\text{Bonus}_{\text{high-impact}}$：在 RI / 路径转折 / 登陆前场景给出额外奖励（正确时奖励更大，错误时惩罚更大）
- 权重 $\alpha, \beta, \gamma$ 在验证集上调优

**备选 RL 策略——DPO**：
- 构建 preference pairs：对同一输入，SFT 产生多个候选 adjustment
- 选择验证误差更小的作为 preferred response
- 用 DPO 训练偏好对齐——更稳定，不需要显式 reward model

## 阶段六：基线系统构建

| 基线 | 描述 | 目的 |
|------|------|------|
| **B1** Raw Guidance (GFS) | 单模式原始预报 | 下界 |
| **B2** Raw Guidance (ECMWF) | 单模式原始预报 | 下界 |
| **B3** Multi-model Consensus | 简单多模式平均 | 经典基线 |
| **B4** HCCA / Weighted Consensus | 历史校准的加权共识 | 业务基线 |
| **B5** Statistical Post-processing | MOS-type regression，用标量环境特征 | 传统后处理 |
| **B6** ML baseline — XGBoost | 表格特征（含对象化环境特征）→ 调整量 | ML 基线 |
| **B7** ML baseline — MLP | 同上，换为 MLP | ML 基线 |
| **B8** ML baseline — CNN/ConvLSTM | 直接从网格场学习调整 | DL 基线 |
| **B9** LLM-SFT (no physics) | LLM 只看模式数值输出，不给环境诊断 | 消融：物理信息的价值 |
| **B10** LLM-SFT (no regime) | LLM 看环境特征但不给误差 regime 诊断 | 消融：regime 诊断的价值 |
| **B11** LLM-SFT (full) | 完整输入的 SFT | 消融：RL 的价值 |
| **B12** Human Expert (OFCL) | NHC 官方预报 | 真正的上界对比 |
| **B13** [SystemName] (full) | 完整系统（物理 + SFT + RL） | 我们的系统 |

## 阶段七：评估与分析

| 任务 | 具体内容 | 方法 |
|------|---------|------|
| **7.1** 总体性能 | 所有系统在测试集上的 Track MAE, Intensity MAE, 分时效 | 标准评估 |
| **7.2** Regime-conditional 性能 | 在每种误差 regime 下分别评估所有系统 | 条件评估 |
| **7.3** 高影响场景性能 | RI, 路径转折, 登陆前 | 子集评估 |
| **7.4** 消融分析 | B9-B13 的逐层对比 | 消融 |
| **7.5** LLM vs ML 深度对比 | 在哪些场景下 LLM 优势明显？在哪些场景下差异不大？ | 条件对比 |
| **7.6** RL 偏差消除分析 | SFT → RL 过程中，哪些系统性偏差被消除了？ | 偏差分解 |
| **7.7** RL 新发现分析 | RL 学到了哪些专家从未做过但有效的调整模式？ | 对比分析 |
| **7.8** 推理质量评估 | 请领域专家评估 reasoning trace 的物理正确性 | 专家评审 |
| **7.9** 案例研究 | 3-4 个经典案例的完整分析 | 深入个案 |

---

# 六、Figure 设计

## Figure 1: The Reasoning Gap in AI Weather Prediction

**内容**：定位论文——AI 天气预报的"最后一公里"在哪里？

**形式**：概念示意图，水平流程

```
┌───────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ Observations  │───▶│  NWP / AI     │───▶│  Expert       │───▶│  Forecast
│  │               │    │  Models       │    │  Judgment     │    │  Decision
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                 │
│  "感知大气状态"       "计算未来演化"       "推理 → 决策"         │
│                                                                 │
│   ✅ AI has             ✅ AI has            ❌ AI has NOT       │
│   addressed this        addressed this       addressed this     │
│   (data assimilation)   (Pangu, GraphCast)   (THIS PAPER)      │
│                                                                 │
└───────────────────────────────────────────────────────────────┘
```

**下半部分**：展示"推理"的内涵——不是回归

| 回归任务 | 推理任务 |
|---------|---------|
| 输入特征 → 输出数值 | 识别情境 → 评估可信度 → 整合证据 → 形成判断 |
| 固定函数映射 | 条件化策略选择 |
| 不需要解释 | 过程可追溯 |
| XGBoost 可以做 | **需要语言模型** |

**设计细节**：
- 上半部分清晰展示预报链条中的定位
- 下半部分用具体对比说明"为什么是推理而不是回归"
- 可在右侧用一个小面板展示 Paper 1 的关键发现作为动机："Expert value is concentrated in structured error regimes"

---

## Figure 2: System Architecture

**内容**：[SystemName] 的三层架构和数据流

**形式**：技术架构图

```
┌─────────────────────────────────────────────────────────┐
│                                                           │
│  INPUT AT FORECAST ISSUANCE TIME                         │
│                                                           │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Multi-Model  │  │ Object-Based │  │ Recent Model   │  │
│  │ Guidance     │  │ Environmental│  │ Performance    │  │
│  │ (track, int) │  │ Features     │  │ Statistics     │  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘  │
│         └────────────────┼──────────────────┘            │
│                          ▼                                │
│  ┌─────────────────────────────────────────────┐         │
│  │  LAYER 1: Physical Diagnosis                 │         │
│  │                                               │         │
│  │  Error Regime Classifier (from Paper 1)       │         │
│  │  ──▶ Regime: "Trough-Interaction Timing"     │         │
│  │  ──▶ Historical bias: Track late recurve     │         │
│  │  ──▶ Expert value: High                       │         │
│  │  ──▶ Conditional model reliability ranking   │         │
│  └──────────────────────┬───────────────────────┘         │
│                          ▼                                │
│  ┌─────────────────────────────────────────────┐         │
│  │  LAYER 2: Expert Reasoning (LLM-SFT)        │         │
│  │                                               │         │
│  │  Structured prompt (all info above)           │         │
│  │  ──▶ LLM generates:                          │         │
│  │       • Reasoning trace (text)               │         │
│  │       • Forecast adjustment (numbers)        │         │
│  └──────────────────────┬───────────────────────┘         │
│                          ▼                                │
│  ┌─────────────────────────────────────────────┐         │
│  │  LAYER 3: Self-Improvement (RL)              │         │
│  │                                               │         │
│  │  Reward: verification-based                   │         │
│  │  ──▶ Retain expert skill                     │         │
│  │  ──▶ Filter human biases                     │         │
│  │  ──▶ Discover new adjustment patterns        │         │
│  └──────────────────────┬───────────────────────┘         │
│                          ▼                                │
│  OUTPUT                                                   │
│  ┌──────────────────────────────────────────┐            │
│  │ Adjusted Forecast + Reasoning Trace       │            │
│  │ "Shifting track 60km NE at 48h because    │            │
│  │  trough interaction will accelerate        │            │
│  │  recurvature; ECMWF handles this better   │            │
│  │  than GFS historically in this regime"    │            │
│  └──────────────────────────────────────────┘            │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

**设计细节**：
- 三层用不同颜色区分（蓝/绿/橙）
- 每层标注其来源（Layer 1 来自 Paper 1，Layer 2 来自 SFT，Layer 3 来自 RL）
- 右侧用注释标出关键设计决策
- 底部的输出框展示一个实际的 reasoning trace 示例

---

## Figure 3: Performance Ladder

**内容**：核心定量结果——逐层性能提升

**形式**：水平柱状图 + 分时效折线图

**Panel (a)**: 总体性能——水平柱状图

```
                           Track MAE (km)        Intensity MAE (kt)
                           ──────────────        ──────────────────
GFS                        ████████████████       ██████████████████
ECMWF                      ██████████████         ████████████████
HWRF                       ███████████████        ███████████████
Multi-model Consensus      ████████████           ██████████████
Weighted Consensus (HCCA)  ███████████            █████████████
XGBoost (same features)    ██████████             ████████████
MLP (same features)        ██████████             ████████████
Human Expert (OFCL)        █████████              ███████████
LLM-SFT (no physics)      ████████               ██████████
LLM-SFT (full)            ███████                █████████       
[SystemName] (full)        █████                  ███████           ★
```

- 颜色分组：模式（灰色系）、传统方法（蓝色系）、ML（紫色系）、人类（金色）、我们（红色渐变）
- 最右侧标注相对 Expert 的改进百分比
- 星号标注统计显著优于所有其他系统

**Panel (b)**: 分时效性能——折线图

- X 轴：Lead time（12h, 24h, 36h, 48h, 72h, 96h, 120h）
- Y 轴：MAE
- 线条：Consensus, Expert, XGBoost, LLM-SFT, [SystemName]
- 分为 track（左）和 intensity（右）两个子面板
- **关键观察**：[SystemName] 在所有时效上优于 Expert；优势在 48-120h 最大

**Panel (c)**: 改进的统计稳健性
- Bootstrap confidence intervals 或 paired permutation test p-values
- 展示 [SystemName] vs Expert 的逐时效显著性

---

## Figure 4: Regime-Conditional Performance Decomposition

**内容**：在不同误差 regime 下，各系统和各层的表现

**形式**：热力图 + 分解柱状图

**Panel (a)**: 热力图——误差 regime × 系统

- 行：各误差 regime（从 Paper 1 的分类）
- 列：Consensus, Expert, XGBoost, LLM-SFT-no-physics, LLM-SFT-full, [SystemName]
- 色值：MAE（蓝=低误差=好）
- **关键观察**：
  - 在"简单"regime 中所有系统差异小
  - 在"困难"regime 中[SystemName] 拉开大差距
  - XGBoost 在某些 regime 中接近 LLM，在其他 regime 中远不如

**Panel (b)**: 层贡献分解——堆叠柱状图

- X 轴：各误差 regime
- Y 轴：相对于 Consensus 的误差减少量
- 堆叠三种颜色：
  - Layer 1 贡献（物理诊断 → 蓝色）
  - Layer 2 贡献（SFT → 绿色）
  - Layer 3 贡献（RL → 橙色）

**预期发现示例**：

| Regime | Layer 1 | Layer 2 | Layer 3 | 解释 |
|--------|---------|---------|---------|------|
| Ridge-steering | **大** | 中 | 小 | 物理修正主导 |
| Trough-timing | 中 | **大** | 中 | 需要推理哪个模式更可信 |
| RI under-predict | 中 | 中 | **大** | 专家也保守，RL 大胆调整 |
| Low-error config | 小 | 小 | 小 | 无需大幅调整 |

**Panel (c)**: LLM vs XGBoost 的 regime 条件差异

- X 轴：各误差 regime
- Y 轴：LLM-SFT MAE − XGBoost MAE（负值 = LLM 更好）
- **关键观察**：LLM 在复杂 regime 中优势大，在简单 regime 中与 XGBoost 持平
- 这直接回答了 "Why LLM?"

---

## Figure 5: Ablation — What Each Component Contributes

**内容**：严格的消融实验——每个组件的不可替代性

**形式**：表格可视化 + 条件差异图

**Panel (a)**: 消融矩阵

| Configuration | Physics | Env Features | Regime Diag | LLM | RL | Track | Intensity |
|---|:---:|:---:|:---:|:---:|:---:|---|---|
| Consensus baseline | ✗ | ✗ | ✗ | ✗ | ✗ | ref | ref |
| XGBoost (scalar features) | ✗ | scalar | ✗ | ✗ | ✗ | −A% | −B% |
| XGBoost (object features) | ✗ | ✓ | ✗ | ✗ | ✗ | −C% | −D% |
| XGBoost (+ regime) | ✗ | ✓ | ✓ | ✗ | ✗ | −E% | −F% |
| MLP (full features) | ✗ | ✓ | ✓ | ✗ | ✗ | −G% | −H% |
| LLM-SFT (guidance only) | ✗ | ✗ | ✗ | ✓ | ✗ | −I% | −J% |
| LLM-SFT (+ env features) | ✗ | ✓ | ✗ | ✓ | ✗ | −K% | −L% |
| LLM-SFT (+ regime) | ✗ | ✓ | ✓ | ✓ | ✗ | −M% | −N% |
| LLM-RL (full) | ✗ | ✓ | ✓ | ✓ | ✓ | **−O%** | **−P%** |

形式：将表格可视化为点图（dot plot），每行一个点在 Track 和 Intensity 轴上

**Panel (b)**: 关键消融对比的效应量

四个关键对比，用成对的柱状图展示：

| 对比 | 揭示 |
|------|------|
| XGBoost(full) vs LLM-SFT(full) | LLM 架构的价值 |
| LLM-SFT(no env) vs LLM-SFT(+ env) | 环境特征的价值 |
| LLM-SFT(no regime) vs LLM-SFT(+ regime) | Regime 诊断的价值 |
| LLM-SFT(full) vs LLM-RL(full) | RL 的价值 |

每个对比在整体和分 regime 两个层面展示

**Panel (c)**: 交互效应

- 物理诊断 + LLM 的联合效应是否大于两者单独效应之和？
- 如果是 → 协同效应存在（物理理解使推理更有效）
- 形式：2×2 矩阵图（有/无 physics × 有/无 LLM），展示 interaction term

---

## Figure 6: Reasoning Traces — Interpretability Evidence

**内容**：系统产生的推理过程展示和质量评估

**形式**：案例展示 + 定量评估

**Panel (a)**: 三个对比案例的推理过程展示

每个案例展示三行：

| | Case 1: Trough Interaction | Case 2: RI Event | Case 3: High Model Spread |
|---|---|---|---|
| **环境配置** | [简明示意图] | [简明示意图] | [简明示意图] |
| **AI 推理** | *"An upper-level trough is approaching from the NW at 15 m/s. GFS positions the trough 3° too far west compared to ECMWF, resulting in delayed recurvature in GFS. Historical analysis shows GFS trough-position bias of 2.1° in this configuration. Adjusting track 80km NE at 48-72h, weighting toward ECMWF solution."* | *"SST 29.8°C with OHC >80 kJ/cm². Shear is 8kt and decreasing. The storm has developed a clear eye in recent satellite imagery analog. All models under-predict intensification in this low-shear warm-ocean regime by 15kt on average. Adjusting intensity +20kt at 24-48h."* | *"GFS and ECMWF diverge by 300km at 96h. GFS maintains westward track (ridge-dominant), ECMWF shows sharp recurvature (trough capture). Current trough position and strength favor the ECMWF solution with 68% confidence based on regime classification. Adjusting toward ECMWF at 72-120h."* |
| **实际结果** | [验证] | [验证] | [验证] |

**设计细节**：
- 推理文本中的关键判断用**粗体**标注
- 每个物理论断旁标注 ✓（正确）或 ✗（错误）
- 这让审稿人看到推理不是"话术"，而是有物理内容的

**Panel (b)**: 推理质量的定量评估

- 请 3-5 位 TC 领域专家对 100 条随机采样的 reasoning traces 评分
- 评分维度（1-5 分）：
  - 物理正确性（Physical correctness）
  - 与实际环境的一致性（Environmental relevance）
  - 推理逻辑（Logical coherence）
  - 是否提供了有用信息（Actionability）
- 形式：雷达图（radar chart）展示各维度平均分
- 与 GPT-4 生成的 baseline reasoning 对比

**Panel (c)**: 推理内容的统计分析

- 对所有 reasoning traces 提取提到的物理因素
- 统计哪些因素被提到最多
- 对比：在不同 regime 下，系统是否提到了正确的物理因素？
- 形式：Regime × Physical Factor 的频率热力图

---

## Figure 7: What RL Learned Beyond Experts

**内容**：RL 超越专家的机制——消除了什么偏差？发现了什么新模式？

**形式**：偏差分解图 + 新模式发现图

**Panel (a)**: Expert bias 的识别和消除

- X 轴：场景类别
- 双柱对比：SFT 的调整模式（≈专家模式） vs RL 的调整模式
- 揭示的 expert biases：

| 偏差类型 | 表现 | RL 修正 |
|---------|------|--------|
| **保守主义（Conservatism）** | 专家对 RI 的调整幅度不够大 | RL 学会在高置信 RI regime 下做更大调整 |
| **锚定偏差（Anchoring）** | 专家过度锚定上一次预报 | RL 更愿意在新信息出现时大幅修改 |
| **等距偏差（Centering）** | 专家倾向于取模式中间值 | RL 学会在特定 regime 下偏向某个模式 |
| **消散延迟** | 专家不愿意过早减弱预报 | RL 在消散 regime 下更果断减弱 |

**Panel (b)**: SFT→RL 调整分布的变化

- 对每种 regime：
  - 左：SFT 调整量的分布（直方图）
  - 右：RL 调整量的分布
  - 叠加验证最优调整量的分布
- **关键观察**：RL 的分布更接近验证最优分布

**Panel (c)**: RL 发现的"新模式"

- 识别 RL 做出了专家从未做过但效果好的调整
- 条件：|RL adjustment| > threshold，但 |Expert adjustment| ≈ 0（专家没调整，但 RL 调整了，且 RL 是对的）
- 展示 2-3 个这样的场景
- 分析其物理合理性

**Panel (d)**: 从 SFT 到 RL 的推理变化

同一输入，对比 SFT 和 RL 的 reasoning trace：

```
SFT: "Models show moderate intensification. Adjusting slightly 
      above consensus (+5kt) consistent with warm SST."

RL:  "SST is 30.1°C with OHC 95 kJ/cm², shear dropping to 5kt, 
      and an emerging upper-level outflow channel to the NE. This 
      configuration historically supports explosive intensification. 
      Models systematically under-predict by 20kt in this regime. 
      Adjusting +25kt at 24-48h, prioritizing HWRF which better 
      resolves inner-core processes."
```

> SFT 模仿了专家的保守调整。RL 学会了在物理证据强时大胆超越专家。

---

## Figure 8: Operational Case Studies

**内容**：3-4 个经典案例的完整端到端展示

**形式**：每个案例一大行，多列信息

**案例选择标准**：
- **Case A**：[SystemName] 显著超越所有系统的高影响事件（如一个被模式和专家严重低估的 RI 事件）
- **Case B**：路径转折案例——系统正确识别了应信任哪个模式
- **Case C**：一个传统 ML 与 LLM 差异最大的案例——展示推理的价值
- **Case D**：系统失败的案例——诚实分析局限

**每个案例的展示**：

| 列 | 内容 | 形式 |
|----|------|------|
| **Col 1** | 路径/强度总览 | Best track + 各系统预报的叠加图 |
| **Col 2** | 环境配置 + Regime 诊断 | 对象化环境图 + regime 标签 |
| **Col 3** | [SystemName] 的推理过程 | 文字框，关键判断高亮 |
| **Col 4** | 性能对比 | 柱状图：Track/Intensity error for each system |

**Case A 详细设计（RI 案例）**：

> **Hurricane Michael (2018) — Explosive intensification to Category 5**

**Col 1**：
- 上面板：路径图，Best track（黑色）+ GFS（蓝色）+ ECMWF（红色）+ OFCL（金色）+ [SystemName]（绿色）
- 下面板：强度时间序列，同色方案
- 标注关键时刻："RI onset", "Cat 5 peak", "Landfall"

**Col 2**：
- 对象化环境图（TC 为中心）
- 低切变 ↓ + 暖 SST 🔴 + 外流通道 ↗
- Regime 诊断：**"RI under-prediction regime"**，置信度 82%

**Col 3**：
```
[SystemName] Reasoning at 2018-10-08 00Z:

"Storm is currently Cat 1 (75kt). Environment is highly 
favorable for rapid intensification:
- SST 29.5°C, OHC 85 kJ/cm²
- Deep-layer shear only 7kt, forecast to decrease
- Strong upper-level anticyclone providing robust outflow

All models intensify the storm but none bring it above Cat 3. 
However, this environmental configuration falls squarely in 
the 'RI under-prediction' regime where models historically 
underestimate peak intensity by 25kt (mean) to 40kt (75th pctl).

HWRF shows the most aggressive intensification (Cat 3), 
consistent with its better representation of inner-core 
convective processes.

ADJUSTMENT: Intensity +30kt above consensus at 36-48h. 
Shifting toward HWRF intensity solution.
Track adjustment minimal — models agree on NNW motion."
```

**Col 4**：

```
Intensity Error at 48h (kt):
GFS:          ████████████████  45kt
ECMWF:        █████████████     35kt
Consensus:    ██████████████    38kt
Expert(OFCL): ████████████      30kt
XGBoost:      █████████████     33kt
[SystemName]: █████             12kt  ★
```

---

## 补充图表（Extended Data / Supplementary）

| # | 内容 | 作用 |
|---|------|------|
| **ED Fig 1** | 训练数据统计：年份×盆地×样本量×regime 分布 | 数据基础 |
| **ED Fig 2** | Prompt 设计详细示例（完整输入输出） | 方法论透明性 |
| **ED Fig 3** | LLM 基座模型选择对比：Llama vs Qwen vs Mistral | 模型选择 |
| **ED Fig 4** | SFT 超参数敏感性分析 | 训练稳健性 |
| **ED Fig 5** | RL reward 设计消融：不同 reward 函数的影响 | RL 设计 |
| **ED Fig 6** | 分盆地结果：Atlantic vs East Pacific | 泛化性 |
| **ED Fig 7** | 分年份结果：性能是否稳定 | 时间稳健性 |
| **ED Fig 8** | 分强度段结果：TD/TS vs Cat1-2 vs Major | 条件性能 |
| **ED Fig 9** | Reasoning trace 的完整专家评估问卷和结果 | 可解释性证据 |
| **ED Fig 10** | 计算成本分析：推理延迟、训练时间、可部署性 | 实用性 |
| **ED Table 1** | 所有系统、所有时效、两个盆地的完整数值结果 | 完整结果 |
| **ED Table 2** | 统计显著性检验表 | 统计严格性 |
| **ED Table 3** | 100 条 reasoning traces 的专家评分详表 | 评分细节 |

---

# 七、图的叙事链条

```
Fig 1: "AI 天气预报解决了计算问题，但推理问题没人碰过——这就是 gap"
  ↓
Fig 2: "我们设计了一个三层系统：理解物理 → 学会推理 → 超越专家"
  ↓
Fig 3: "它在所有维度、所有时效上都超越了所有现有系统，包括人类专家"
  ↓
Fig 4: "不是均匀超越——在复杂 regime 下优势最大，这证明推理真的有用"
  ↓
Fig 5: "每个组件都不可替代——物理+LLM+RL 的协同效应是关键"
  ↓
Fig 6: "它不是黑箱——它产生的推理过程在物理上是正确的、可被专家理解的"
  ↓
Fig 7: "它不只是模仿专家——RL 让它消除了人类偏差、发现了新的调整模式"
  ↓
Fig 8: "看具体案例：在 Hurricane Michael 这样的极端事件中，它做对了人类没做到的事"
```

> **从"AI 不会推理"到"AI 推理得比专家好且更透明"——这是一个完整的认知颠覆。**

---

# 八、Paper 1 与 Paper 2 的关系

```
Paper 1 (A+C)                           Paper 2 (B)
──────────────                          ──────────────
回答 WHAT 和 WHY                         回答 HOW 和 BEYOND

发现：误差有结构                          行动：学会利用这个结构
发现：专家精确互补                        行动：让 AI 学会这种互补
发现：脆弱性可诊断                        行动：基于诊断做出更好的预报
                                        超越：RL 去除人类偏差

    Paper 2 引用 Paper 1:
    "Building on the structured error regimes and 
     expert compensation architecture revealed in 
     [Paper 1], we ask whether an AI system can 
     learn — and ultimately surpass — the reasoning 
     process that makes human experts valuable..."

    Paper 1 在 Discussion 中预告 Paper 2:
    "The structured nature of expert compensation 
     suggests that this reasoning may be learnable 
     by AI systems capable of conditional, 
     multi-source inference..."
```

**两篇论文形成完美的科学叙事链：Paper 1 是诊断，Paper 2 是治疗并超越。**