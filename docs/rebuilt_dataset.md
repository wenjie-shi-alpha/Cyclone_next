# Cyclone Forecast / Process Supervision 重构开发文档

## 1. 文档目的

本文档用于指导 Codex 对当前台风预报训练体系进行一次**面向长期扩展的重构**。

目标不是简单做出一个临时可跑通的 `forecast-only` 数据集，而是：

1. 先把**统一的数据主结构（canonical schema）**设计正确；
2. 在该主结构之上，派生多个训练视图（views）；
3. 训练时采用**分阶段策略**，优先做最稳定、最可验证的 forecast 主任务；
4. 为后续“学习专家推理过程”预留明确接口，而不是把自由文本直接混进主任务 target 中；
5. 最终形成一套可持续扩展的 `diagnose -> forecast -> explain` 体系。

---

## 2. 总体原则

### 2.1 核心结论

本次重构遵循以下逻辑：

* **数据结构一开始就完整设计好**；
* **训练分阶段进行**；
* 第一阶段只激活 `forecast-only` 主任务；
* 后续再补 `diagnostic/process supervision` 与 `reasoning/explanation`；
* 不要一开始把“严格结构化预报结果”和“自由文本推理”混在同一个 target 中训练。

### 2.2 为什么不能把 forecast 和 reasoning 混成一个主任务

原因如下：

1. 当前最稳定、最可验证、最适合 RL 的目标是 **strict forecast table**；
2. reasoning 文本奖励不稳定，自动评估难，噪声大；
3. reasoning 覆盖率低，且不完整，直接混入主任务会污染目标分布；
4. 混合训练会显著增加“接近正确但不可解析”的输出；
5. 如果主任务格式边界被破坏，后续的 parse / verification / reward 体系都会失效。

因此，当前正确方向不是：

* 让模型一步输出“推理 + 结论”；

而是：

* 先让模型稳定输出严格 forecast；
* 再让模型学习如何解释该 forecast；
* 更进一步，让模型学习专家的**结构化中间判断过程**。

### 2.3 对“学习专家推理”的正确理解

本项目不应把“学习专家推理”简单理解为：

* 学会写一段很像专家的话。

而应理解为：

* 学会专家在预报时依赖的**关键判断节点、诊断因子与决策结构**。

因此，真正应当补充的是：

* `structured diagnostics / decision states`

而不是仅仅补更多自由文本 explanation。

---

## 3. 目标架构

### 3.1 长期目标链路

长期应形成以下三段式链路：

1. **Diagnose**：根据观测、环境场、模式指导，输出结构化诊断结果；
2. **Forecast**：根据输入与诊断结果，输出严格可解析的 forecast table；
3. **Explain**：基于已确定的 forecast 与诊断结果，生成 reasoning / narrative explanation。

即：

```text
input -> diagnostics -> forecast -> explanation
```

### 3.2 当前阶段的实际训练顺序

虽然长期目标是三段式，但当前阶段的训练顺序应为：

```text
Phase 1: forecast-only SFT -> GRPO
Phase 2: diagnostic/process supervision SFT
Phase 3: reasoning/explanation SFT
Phase 4: optional multi-task distillation / unification
```

注意：

* **架构现在就按长期目标设计**；
* **训练现在先做最稳的一段**。

这两件事不能混淆。

---

## 4. Codex 需要完成的核心任务

Codex 本轮开发应完成以下工作：

1. 定义新的 canonical schema；
2. 将现有原始数据映射到新的 canonical schema；
3. 在 canonical schema 基础上导出多个 training views；
4. 明确 view eligibility / quality flags；
5. 为后续 diagnostics 与 reasoning 预留字段与导出逻辑；
6. 不改坏当前可用于 strict forecast 训练的主链路；
7. 为训练、评估、推理三侧提供一致接口。

---

## 5. 新的数据设计逻辑

## 5.1 设计原则

数据设计必须满足：

1. **一个样本对应一个明确的 forecast issuance unit**；
2. 所有输入证据围绕这个 unit 对齐；
3. forecast、diagnostic、reasoning、risk 等内容都作为该 unit 的不同 target / annotation；
4. 各类训练任务不是各自独立拼装，而是从同一个 canonical record 派生；
5. 允许字段缺失，但不允许 schema 缺位；
6. 必须支持后续多 adapter、多阶段训练。

### 5.2 不能采用的错误做法

Codex 必须避免以下错误：

1. 直接复制现有 forecast-only 数据，临时拼一个新数据集，后续再补；
2. 为 reasoning 单独定义另一套不兼容 schema；
3. forecast / reasoning / risk 使用不同样本键，导致后续无法严格对齐；
4. 把中间过程信息塞进 prompt 文本里，但不落结构化字段；
5. 在 schema 中省略 quality flags、eligibility flags、source metadata；
6. 在 view 导出阶段做过多“隐式逻辑”，导致样本来源不可追踪。

---

## 6. Canonical Schema 设计要求

下面给出推荐的主结构。Codex 可以根据工程现状调整字段命名，但必须保持语义完整。

```json
{
  "sample_id": "...",
  "storm_id": "...",
  "basin": "...",
  "issue_time": "...",
  "lead_times": [24, 48, 72, 96, 120],
  "source_split": "train|val|test",
  "time_anchor_complete": true,
  "input_window_spec": {
    "obs_start": "...",
    "obs_end": "...",
    "env_start": "...",
    "env_end": "...",
    "guidance_cycle": "..."
  },
  "inputs": {
    "observation_context": "...",
    "environment_context": "...",
    "model_guidance": "...",
    "historical_track_context": "..."
  },
  "targets": {
    "official_forecast_table": "...",
    "forecast_parseable": true,
    "verification_target": {
      "track": "...",
      "intensity": "..."
    },
    "reasoning_text": "...",
    "risk_text": "..."
  },
  "diagnostics": {
    "track_control_signal": null,
    "turning_signal": null,
    "intensity_support_signal": null,
    "shear_constraint_level": null,
    "land_interaction_level": null,
    "model_agreement_level": null,
    "main_uncertainty_source": null,
    "forecast_confidence_level": null,
    "expert_decision_notes": null
  },
  "flags": {
    "has_forecast": true,
    "has_reasoning": false,
    "has_risk": false,
    "has_diagnostics": false,
    "forecast_view_eligible": true,
    "diagnostic_view_eligible": false,
    "reasoning_view_eligible": false
  },
  "metadata": {
    "raw_source_ids": [],
    "parser_version": "...",
    "canonical_version": "...",
    "quality_flags": []
  }
}
```

### 6.1 字段层级要求

Codex 必须保证以下层级明确存在：

#### A. 基础标识层

* `sample_id`
* `storm_id`
* `issue_time`
* `basin`
* `source_split`

#### B. 输入证据层

* 观测信息
* 环境场信息
* 模式指导
* 历史轨迹上下文
* 时间窗说明

#### C. 主目标层

* strict forecast table
* verification target
* parse / format 相关标记

#### D. 过程层

* diagnostics
* reasoning_text
* risk_text
* expert_decision_notes

#### E. 可训练性与质量控制层

* has_x flags
* view eligibility flags
* quality flags
* parser / schema version

### 6.2 缺失值原则

允许：

* `reasoning_text = null`
* `diagnostics.* = null`
* `risk_text = null`

不允许：

* 因为字段暂时缺失，就在 schema 中删除该字段；
* 用空字符串、占位话术、伪造文本代替真正缺失。

缺失值必须显式、可检测、可统计。

---

## 7. View 导出设计

Codex 不应直接维护多个互不兼容的数据集，而应从 canonical schema 导出多个 training views。

### 7.1 Forecast-only View

#### 输入

* observation_context
* environment_context
* model_guidance
* historical_track_context

#### 输出

* `official_forecast_table`

#### 要求

* 严格保持可解析格式；
* 不得混入 reasoning / risk 文本；
* 必须适配后续 strict SFT 与 GRPO。

#### eligibility

* `flags.forecast_view_eligible == true`
* `targets.official_forecast_table != null`
* 时间锚点完整
* 通过最基本 parse 校验

### 7.2 Diagnostic View

#### 输入

* 与 forecast-only 相同

#### 输出

* `diagnostics.*`

#### 要求

* 优先输出结构化/离散化诊断信号；
* 不要先做成长篇自由文本；
* 即使当前标签稀疏，也要先把 view 管线搭好。

#### eligibility

* `flags.diagnostic_view_eligible == true`
* 至少一个 diagnostic 字段非空

### 7.3 Reasoning-only View

#### 输入

* observation_context
* environment_context
* model_guidance
* historical_track_context
* **official_forecast_table**
* 可选：diagnostics

#### 输出

* `reasoning_text`

#### 要求

* reasoning 是对 forecast 的解释，不是重新做决策；
* 不允许让 reasoning 与 forecast 相互竞争输出空间；
* 保持和 forecast 强绑定。

#### eligibility

* `flags.reasoning_view_eligible == true`
* `targets.reasoning_text != null`
* `targets.official_forecast_table != null`

### 7.4 Risk View（可选）

当前不应作为主任务，但可为后续扩展保留：

#### 输入

* context + forecast

#### 输出

* `risk_text`

---

## 8. 训练顺序要求

## 8.1 Phase 1：只做 forecast 主任务

当前最优先训练的是：

```text
strict forecast-only SFT -> GRPO
```

### Phase 1 目标

1. 提高 strict parse rate；
2. 稳定 forecast format；
3. 提高 verification reward；
4. 提高结构化结果的准确性与稳定性；
5. 为后续 explanation 链路提供可靠中间结果。

### Phase 1 明确禁止

1. 把 reasoning 与 forecast 拼成同一 target；
2. 为了“更像专家”而放宽 strict format；
3. 将 risk 或低覆盖率字段混入主训练集；
4. 因为一小部分 reasoning 存在，就在所有样本中引入 explanation prompt 分支。

## 8.2 Phase 2：补结构化过程监督

在 forecast 训练稳定之后，再引入 `diagnostic/process supervision`。

### 这一步的目标

让模型学习专家判断的**中间状态**，而不是只学结果。

### 优先应补的 diagnostics 字段

可从以下候选开始：

* `track_control_signal`
* `turning_signal`
* `intensity_support_signal`
* `shear_constraint_level`
* `land_interaction_level`
* `model_agreement_level`
* `main_uncertainty_source`
* `forecast_confidence_level`

### 注意

这一步不是要求 Codex 立即自动生成高质量标签，而是要求：

1. 先把 schema 和导出管线搭好；
2. 允许后续通过规则抽取 / 弱监督 / LLM 辅助标注逐步补全。

## 8.3 Phase 3：Reasoning / Explanation 训练

在 forecast 已稳定、diagnostic 体系已初步成型后，再做 explanation。

### 正确输入

* 原始 context
* 已确定的 forecast table
* 可选 diagnostics

### 正确输出

* reasoning_text

### 注意

explanation 的角色是：

* 解释 forecast 为什么成立；

不是：

* 作为 forecast 生成的主决策路径。

## 8.4 Phase 4：可选统一

未来如有需要，可考虑：

* 同一 base model + 多 adapter 长期共存；
* 或者做 task-tag 蒸馏 / 多任务统一。

但本轮开发不应直接做这一步。

## 8.5 当前执行决策（2026-04-11）

基于当前重构结果，本项目**不应等待把 diagnostics / reasoning 数据一次性“全部做好”之后再开始训练**；
而应当执行：

```text
先用 rebuilt dataset 的 forecast-only view 完成第一轮 SFT -> GRPO，
再根据第一轮结果决定 diagnostics / reasoning 的补标、清洗与接入方式。
```

### 当前为什么应先做第一轮 forecast-only 微调

原因如下：

1. `forecast-only` 是当前唯一同时满足**高覆盖率、严格可解析、可自动评估、可直接接 RL** 的任务；
2. 当前 canonical schema 和多 view 导出已经完成，说明“长期接口”已经搭好，不需要再为了后续任务推迟 Phase 1；
3. 当前 `reasoning` 覆盖率仍明显低于 forecast 主任务，且文本质量、忠实性、一致性都还需要结合模型行为再回看；
4. 当前 `diagnostic` view 虽然已经可导出，但标签本质上仍属于**弱监督 / 规则派生的 v1 信号**，不应把它当作阻塞 Phase 1 的前置条件；
5. 如果不先做一轮 `forecast-only` 训练，就无法知道当前真正的瓶颈到底是：
   * prompt 设计问题；
   * strict format 学习问题；
   * reward 设计问题；
   * 还是后续才需要由 diagnostics / reasoning 来弥补的问题。

### 当前阶段的正确策略

正确策略不是：

* 继续无上限地扩写后续任务数据，试图在第一次训练前把所有任务都做“完整”；

而是：

* 保持 canonical schema 与多 view 结构不变；
* 先激活 `forecast-only` 主任务；
* 跑第一轮 SFT；
* 再跑第一轮 GRPO；
* 用这一轮结果反推 diagnostics / reasoning 数据还缺什么、该怎么接入。

### 当前阶段不建议做的事

当前不建议：

1. 在第一轮训练前，把 diagnostics / reasoning 一起并入主训练集；
2. 为了提高“像专家”的感觉，提前把 explanation 文本混回 forecast target；
3. 在没有看过第一轮 forecast-only 失败模式之前，就大规模重写 diagnostics 标签体系；
4. 把弱监督 diagnostics 当成已经可以稳定监督主模型决策的成熟标签。

### Phase 1 完成后再决定 Phase 2 / Phase 3 的依据

完成第一轮 `forecast-only SFT -> GRPO` 后，再根据以下信号决定后续数据改造：

1. strict parse rate 是否已经稳定；
2. verification reward / track / intensity error 的主要失分段落在哪些 lead time；
3. 模型失败是否主要表现为：
   * 结构格式问题；
   * 轨迹偏差问题；
   * 强度偏差问题；
   * 模式分歧时决策不稳；
   * 解释与结果不一致；
4. 当前弱监督 diagnostics 是否真的能解释这些失败模式；
5. reasoning 样本是否需要先做忠实性筛选、去噪和一致性审查，再接入训练。

### 结论

**结论非常明确：**

> 现在就开始第一轮 `forecast-only` 微调；
> 不等待 diagnostics / reasoning 数据一次性做满；
> 但保持 canonical schema 和多 view 管线不变，
> 让后续 Phase 2 / Phase 3 能在第一轮训练结果基础上有针对性接入。

---

## 9. 模型与 Adapter 设计建议

Codex 在文档、配置、导出命名中，应显式支持以下结构：

* `forecast_adapter`
* `diagnostic_adapter`
* `reasoning_adapter`

推荐策略：

* 同一个 base model；
* 不同 adapter 分别训练；
* 推理时按阶段切换 adapter。

### 当前最稳的推理链路

```text
Step 1: forecast_adapter -> forecast table
Step 2: reasoning_adapter -> explanation
```

### 后续理想链路

```text
Step 1: diagnostic_adapter -> diagnostics
Step 2: forecast_adapter -> forecast table
Step 3: reasoning_adapter -> explanation
```

Codex 应在设计数据导出与推理接口时，为这两种链路都预留兼容性。

---

## 10. 评估设计要求

虽然本轮重点是数据重构，但 Codex 设计时必须为后续评估保留字段与接口。

### 10.1 Forecast 主评估

* parse rate
* strict format pass rate
* verification reward
* track / intensity error
* lead-time 分段表现

### 10.2 Diagnostic 评估

后续应支持：

* 结构化标签准确率
* 与 forecast 一致性
* 与专家标注一致性
* 不确定性判断的校准能力

### 10.3 Reasoning 评估

后续应支持：

* reasoning 与 forecast 是否一致
* reasoning 是否引用了正确关键因子
* 是否与 diagnostics 冲突
* 是否出现与最终 forecast 不一致的说明

即：

* explanation quality 不能只看“像不像人写的”，而要看是否**支持并忠实于 forecast/diagnostics**。

---

## 11. Codex 实现时的边界要求

## 11.1 要做的事情

Codex 应当：

1. 重构 canonical schema；
2. 重写数据构建 / 导出逻辑；
3. 为多个 training views 建立统一导出接口；
4. 增加 eligibility flags 和 quality flags；
5. 保证样本 ID 与对齐关系稳定；
6. 保证导出逻辑可复现、可追踪；
7. 保留版本信息。

## 11.2 不要做的事情

Codex 不应：

1. 擅自改变 forecast strict output format；
2. 擅自把 reasoning 拼接到 forecast target 中；
3. 因 reasoning 样本稀疏而删除相关 schema；
4. 为了“看起来完整”伪造缺失字段；
5. 让不同 view 的样本切片逻辑不一致；
6. 引入过度复杂但不可追踪的隐式 prompt 规则；
7. 将训练逻辑硬编码进数据 schema；
8. 将 schema 设计成只适用于当前一次实验的临时结构。

---

## 12. 开发顺序建议

建议 Codex 按以下顺序推进：

### Step 1：梳理当前数据构建链路

需要明确：

* 当前 canonical 数据来自哪些原始源；
* 样本唯一键目前如何定义；
* 时间锚点在哪里丢失或不完整；
* forecast / reasoning / risk 当前如何挂载。

### Step 2：定义 v2 canonical schema

输出：

* 明确字段定义；
* 明确 null / missing 规则；
* 明确 versioning。

### Step 3：实现原始数据到 canonical v2 的转换器

要求：

* 可重跑；
* 可记录日志；
* 可输出 coverage 统计；
* 可统计各字段缺失率。

### Step 4：实现多 view 导出器

至少支持：

* forecast-only
* diagnostic-only
* reasoning-only

### Step 5：实现 format / eligibility / quality report

至少输出：

* 总样本数
* forecast eligible 数量
* reasoning eligible 数量
* diagnostic eligible 数量
* parseable 比例
* 时间锚点完整率
* 各字段缺失率

### Step 6：回归检查

确保：

* forecast-only view 不劣化当前 strict training 能力；
* 样本对齐关系未被破坏；
* 输出格式与现有训练器兼容。

---

## 13. 产出物要求

Codex 至少应产出以下内容：

1. 新版 canonical schema 定义；
2. 数据转换脚本；
3. view 导出脚本；
4. 统计报告脚本；
5. 关键 README / 使用说明；
6. 示例输出目录结构；
7. 核心单元测试或最小回归测试。

推荐产物示例：

```text
/data/
  canonical_v2/
    train.jsonl
    val.jsonl
    test.jsonl
    schema.json
    build_report.json
/views/
  forecast_only/
    train.jsonl
    val.jsonl
    test.jsonl
    report.json
  diagnostic_only/
    train.jsonl
    val.jsonl
    test.jsonl
    report.json
  diagnostic_slot_turn_correction_only/
    train.jsonl
    val.jsonl
    test.jsonl
    report.json
  reasoning_only/
    train.jsonl
    val.jsonl
    test.jsonl
    report.json
/scripts/
  build_canonical_v2.py
  export_views.py
  summarize_dataset.py
/docs/
  DATASET_V2_SPEC.md
```

---

## 14. 开发验收标准

以下条件满足，才算本轮开发完成：

### 14.1 数据结构层

* canonical schema 已统一；
* 不同任务不再各用一套独立样本结构；
* 字段缺失显式可统计；
* 版本与来源可追踪。

### 14.2 视图层

* forecast-only view 稳定导出；
* reasoning-only view 可导出；
* diagnostic-only view 已搭好管线，即使当前标签稀疏也可以运行；
* eligibility 与 quality report 正常。

### 14.3 训练兼容层

* forecast-only view 与当前 strict SFT / GRPO 兼容；
* 不要求本轮完成 reasoning / diagnostic 训练，但必须具备后续直接接入的接口。

### 14.4 逻辑层

系统结构必须体现以下原则：

* **schema 先完整设计**；
* **训练后分阶段激活**；
* **forecast 是当前主任务**；
* **diagnostics 是未来过程监督核心**；
* **reasoning 是解释层，不是主决策层**。

---

## 15. 给 Codex 的特别提醒

### 15.1 不要把“学习专家推理”等同于“学习长文本推理”

真正要学习的是：

* 专家使用哪些证据；
* 专家如何在关键信号之间做判断；
* 专家如何形成置信度与不确定性判断；
* 专家如何在模式分歧中做取舍。

因此，后续最值得投入的数据不是单纯更多 explanation prose，而是：

* 更好的 structured diagnostics
* 更好的 decision-state annotations
* 更好的 consistency checks

### 15.2 不要为了长期目标破坏当前最稳主链路

长期目标是 `diagnose -> forecast -> explain`，但当前阶段最重要的是：

* 保住 strict forecast 训练与 RL 路径
* 不让格式边界被污染
* 不让主任务可验证性下降

### 15.3 先把接口做对，比一开始把所有标签做满更重要

当前最重要的不是“立即拥有完美的 diagnostics 标注”，而是：

* schema 先能承载它们；
* pipeline 先能导出它们；
* 统计先能看到 coverage；
* 后续补标注时不需要推倒重来。

---

## 16. 最终一句话要求

请 Codex 按以下原则开发：

> **现在就把数据体系按“结果 + 过程 + 解释”完整设计好；但训练上先只激活 forecast 主任务，后续再逐步接入 diagnostics 与 reasoning。**

不要做成“先临时凑一个 forecast-only 数据集，后面再补”；
而要做成“同一个 canonical schema 下派生多个 views，并支持未来的 diagnose -> forecast -> explain 演进路径”。
