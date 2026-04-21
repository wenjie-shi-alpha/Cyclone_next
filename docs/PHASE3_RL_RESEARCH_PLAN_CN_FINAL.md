# Phase 3 强化学习研究计划（定稿）

状态：Phase 3 定稿版。本文档用于冻结 Phase 3 的研究目标、阶段边界、解锁条件、奖励设计原则、评估纪律与待办事项，不表示在未完成前置准备前立即启动训练。

## 1. 阶段定位

Phase 2 已经完成了进入 RL 所需的最重要前置工作：

- 找到了一个**狭义但可执行的中间接口**，并且它确实能够改善下游 forecast 行为
- 将该接口修复到 raw diagnostic head 本身也稳定、可执行
- 确认当前最佳 predicted mainline 不是 sample-level 偶然结果，而是在 held-out 上经过 confirm 的结果
- 同时也澄清了当前剩余瓶颈是什么，以及**不是什么**

因此，Phase 3 不应该被表述为“先跑一版 RL 看看有没有用”。

它应该被表述为：

- 以 **已经确认有效的可执行修正接口** 为起点
- 进一步逼近 **策略层面的 expert-gap closing 行为**
- 同时保留 Phase 2 已经建立好的 contract、renderer 和评估纪律

换句话说，Phase 3 的目标不只是让模型分数再高一点。
Phase 3 的目标是验证：**在相同证据条件下，模型是否能够学会更接近预报专家的、可执行的 forecast-correction policy。**

## 2. 从 Phase 2 冻结继承的起点

### 2.1 当前已确认主线

当前推荐的 predicted mainline 为：

- `diagnostic_slot_turn_correction_only v3 + baseline intensity + scale 1.20`
- 对应 forecast-side confirmed variant：
  - `predicted_slot_turn_track_plus_baseline_intensity_scale_1p20_v3`

held-out full confirm 结果：

- `track_error_km = 145.58`
- `mean_track_diff_vs_official_km = 125.26`
- `reward_mean = 0.3686`
- `intensity_error_kt = 11.82`

相对于 `baseline_forecast_sft_v2`，这条 mainline 已经缩小了有意义的一部分 expert gap，应当作为 Phase 3 的正确起始 control。

### 2.2 当前保留控制组

以下系统在整个 Phase 3 过程中都必须被冻结保留，作为 reporting controls：

- `predicted_slot_turn_track_plus_baseline_intensity_scale_1p20_v1`
- `predicted_slot_correction v1 + baseline intensity + scale 1.20`
- `baseline_forecast_sft_v2`
- `baseline_forecast_grpo_physical_v2`
- `baseline_forecast_grpo_reward_v2`

### 2.3 当前机制层面的认识

Phase 2 已经澄清了若干必须直接塑造 Phase 3 设计的事实：

- 当前 confirmed gain 主要来自 `lon semantics`
- `lat bucket` 仍然是最弱的语义轴
- 当前 blocker 已经不再是 JSON contract 或 turning-signal collapse
- 当前项目 gate 是 dual-track，而不是 single-track
- RL 应优化的是 **integrated forecast behavior**，而不是 standalone diagnostic proxy score

因此，Phase 3 不应再退回到以下方向：

- generic end-to-end free-form RL
- 仅针对 standalone diagnostic head、且以 proxy labels 为最终目标的 RL
- 只优化 `reward_mean` 或 `coverage` 的 reward 设计

## 3. 中心科学问题

Phase 3 应回答一个比当前 Phase 2 RL 草案更严格的问题。

不只是：

- RL 能不能让当前 frozen mainline 再好一点？

而是：

- 在固定证据条件下，RL 能否学到一种修正策略，在保持结构可执行、forecast 合法的前提下，进一步缩小模型与官方专家预报之间的差距？

这个问题自然可以拆成三个嵌套的子问题：

1. **Residual policy question**
   - 在 `slot_turn_correction v3` 接口被冻结的前提下，forecast adapter 是否还存在 residual headroom？

2. **Action-policy question**
   - 如果将 RL 直接施加到可执行修正接口本身，模型能否学到比监督训练阶段更好的 correction action？

3. **System-coordination question**
   - 如果 action policy 得到提升，forecast adapter 是否还能进一步被对齐，从而更有效地消费这套策略，同时不破坏既有接口？

Phase 3 应该按照这三个问题的顺序组织。

## 4. Phase 3 总体设计原则

Phase 3 应遵循一个严格原则：

**RL 必须作用在“最小但又能直接对应科学问题”的可训练对象上。**

这意味着：

- **不要**一开始就做 joint end-to-end RL
- **不要**在第一步就把 action generation、forecast rendering 和 free-form reasoning 混成一个 policy 来训
- **不要**让 reward function 成为正确性的唯一来源

相反，Phase 3 应按三个逐步扩大的层次推进：

- Layer A：面向下游 forecast adapter 的 residual RL
- Layer B：面向可执行修正接口本身的 action-level RL
- Layer C：在 action policy 与 forecast adapter 之间做 constrained alternating optimization

## 5. Phase 3 结构设计

### 5.0 阶段解锁规则（冻结）

为避免后续将局部负结果误读为整条 Phase 3 路线失败，现冻结如下解锁规则：

- **Phase 3A 失败，不阻断 Phase 3B**
- **Phase 3B 是 Phase 3 的真正主线**
- **Phase 3C 只有在 Phase 3B 正向后才允许解锁**
- **不得因为 Phase 3A 的负结果而否定 action-level RL 的科学价值**
- **不得在 Phase 3B 未完成前提前启动 joint end-to-end RL**

### 5.1 Phase 3A — 面向 Forecast 侧的 Residual RL 探针

#### 目标

这是最窄的一层 RL 入口。
它的目的不是解决整个问题。
它的目的是测量：**在 Phase 2 接口被冻结之后，forecast 侧本身是否还存在 residual headroom。**

#### 可训练对象

- 仅训练 forecast adapter

#### 冻结对象

- `slot_turn_correction v3` diagnostic adapter
- prompt override schema
- deterministic slot-locked renderer
- baseline intensity integration path
- 当前 calibration path

#### 为什么仍然保留这一层

这一层仍有价值，因为它回答的是一个非常干净的问题：

- 当可执行接口已经固定之后，forecast model 在“如何消费这套接口”上，是否仍然存在可以学习的下游策略价值？

如果这一层为正，说明 downstream 仍有 residual headroom。
如果这一层为负，**并不能**否定整个 Phase 3；它只说明 downstream forecast adapter 可能已经不是当前主要瓶颈。

#### 成功标准

一个正向的 Phase 3A 结果应满足：

- 至少一个 primary track metric 在 matched overridden held-out evaluation 上改善
- 另一个 primary track metric 不发生实质性退化
- intensity 守住 guardrail
- parseability 与 slot-time alignment 保持完整

#### 在整个 Phase 3 中的角色

Phase 3A 是一个 **residual probe**，不是最终的 RL 方案。

### 5.2 Phase 3B — 面向可执行修正策略的 Action-Level RL

#### 目标

这应当是 Phase 3 的真正主线。

不是让 RL 直接作用在最终 forecast token 上，
而是让 RL 直接作用在 **可执行的 correction action** 上：

- `slot_turn_correction` 风格的结构化输出
- 通过 deterministic renderer 渲染成 forecast 行为
- 最终只通过 downstream forecast gain 来评估

这一步与研究目标最接近：
在固定证据条件下学习更接近专家的修正策略。

#### 可训练对象

- 仅训练 correction-policy adapter

#### 初始化

- 从当前已经通过 confirm 的 Phase 2 action policy 初始化：
  - `slot_turn_correction v3 final_adapter`

#### 冻结环境

- deterministic renderer
- calibration path
- baseline intensity integration
- forecast-side evaluation stack
- evidence regime

#### 核心思想

此时被优化的 policy 不再是“forecast text generation”。

而是：

- 在相同证据下选择更好的结构化修正动作
- 使渲染后的 forecast 能进一步缩小 expert gap

这是第一层真正直接优化你们在 Phase 2 花大量力气构建出来的对象的 RL。

#### 必要约束

因为这一层训练的是 correction interface 本身，所以必须强约束：

- 对 `slot_turn_correction v3` 加 KL 或 behavior-cloning anchor
- 在 rollout 与训练环路中执行 hard schema validation
- 单独监控 turning-signal distribution drift
- 单独监控 bucket-distribution drift
- 单独报告 `lat` / `lon` 分轴表现
- 单独报告 weakest-lat 与 strongest-lon 字段变化
- 这一层不允许修改 renderer
- 这一层不允许把 contract 稳定性退化换取短期 reward 改善

#### 防 shortcut 条款（冻结）

考虑到当前 confirmed gain 主要来自 `lon semantics`，而 `lat bucket` 仍然较弱，现冻结以下防 shortcut 约束：

- 不允许仅凭 aggregate dual-track 改善就宣布机制性成功
- 任何正向结论都必须附带 `lat` / `lon` 分轴结果
- 若 gain 仅来自更强的 `lon` 轴放大，而 `lat` 轴、turning 或 bucket stability 明显恶化，则该结果只能记为“局部 gain”，不得升级为机制性胜利
- 如 action-level RL 导致 contract drift、turning collapse 或 bucket 分布显著失真，则即便 aggregate reward 上升，也不得判定为通过

#### 为什么这一层比纯 forecast-only RL 更符合科学目标

Forecast-only RL 最多只是在 frozen interface 下优化模型的“使用方式”。
而 action-level RL 则是在直接检验：模型能否把修正策略本身学得更好。

这更接近：

- expert adjustment learning
- policy transfer
- human-like forecast correction behavior

#### 成功标准

一个正向的 Phase 3B 结果应满足：

- 渲染后的 forecast 在 dual-track expert-gap 指标上改善
- action contract 仍然可执行
- 增益不是格式伪影造成的
- 增益并不只来自 trivial 或 degenerate slot behavior

### 5.3 Phase 3C — 受约束的 Alternating RL / SFT 协同优化

#### 目标

如果 Phase 3B 成功，下一步问题就是：下游 forecast adapter 在消费这套 improved action policy 时，是否仍然是次优的。

这一层**不应**一上来就做 full joint RL。
应先从 **alternating optimization** 开始：

1. 冻结 forecast adapter，提升 action policy
2. 冻结 action policy，提升 forecast adapter 的消费能力
3. 与 fully frozen control 做比较
4. 一旦收益来源不清晰或训练不稳定，就立刻停止

#### 可训练对象

交替优化，初版不允许同时自由训练：

- action policy adapter
- forecast adapter

#### 为什么要 alternating，而不是直接 joint RL

当前主线是一个 composed system，不是 monolithic checkpoint。
如果一开始同时训练两个部分，就会失去归因能力：

- 是 action policy 变好了？
- 是 forecast adapter 变好了？
- 还是两边只是偶然互相补偿？

Alternating optimization 能更好地保留可解释性。

#### 成功标准

一个正向的 Phase 3C 结果应满足：

- alternating system 优于当前最佳的 Phase 3B action-policy 结果
- 增益在 held-out full confirm 上稳定存在
- correction interface 的可解释性仍然可接受
- 系统不丢失 slot validity、schema stability 或 intensity discipline

## 6. Reward 设计策略

### 6.1 基本原则

Phase 3 的 reward 必须围绕 **在可执行约束下缩小 expert gap**，而不是一般性的 scalar reward 最大化。

### 6.2 推荐主 reward

Phase 3 推荐的主 reward 应包括：

- truth-side track improvement
- official-gap closing improvement

在具体实现上，应直接建立在：

- `track_error_km`
- `mean_track_diff_vs_official_km`

这两个 primary metric 之上，因为当前项目 gate 本身就是 dual-track。

### 6.3 Guardrails

以下指标应继续作为 guardrails，而不是 primary reward target：

- `intensity_error_kt`
- `mean_intensity_diff_vs_official_kt`
- `strict_parseable_rate`
- `slot_time_match_rate_vs_official`
- executable schema validity
- turning-signal distribution stability
- bucket distribution drift

### 6.4 Phase 3 的 reward 形式

Phase 3 推荐的 reward 形式不应只是 absolute-score reward。
更合适的形式是：**相对 frozen control 的 gap-closing reward**。

也就是说，一个 candidate policy 的 reward，应该取决于：

- 在 matched prompt regime 下
- 它是否比当前 frozen confirmed mainline 关闭了更多 gap

这种定义更符合研究目标：

- 不是泛泛地“变好”
- 而是“比当前最佳已知策略更进一步缩小 expert gap”

### 6.5 Dual-track 标量化规则（冻结）

为避免 pilot 阶段再次退回到临时解释 checkpoint 的状态，现冻结如下规则：

- 主 reward 必须同时包含：
  - `track_error_km` 改善项
  - `mean_track_diff_vs_official_km` 改善项
- 两个 primary 项在 Phase 3 初版中按**同等优先级**处理
- 若实现为加权和，默认采用 **1:1** 的归一化后权重；任何偏离都必须先更新文档并重跑 pilot
- 若一个 primary 指标改善而另一个退化，则不得只凭 aggregate reward 上升宣布通过
- Phase 3 的正式过线标准始终要求：
  - 至少一个 primary metric 有具有实际意义的改善
  - 另一个 primary metric 只能在预先冻结的容忍范围内非退化

### 6.6 Non-regression 与 Guardrail 约束（冻结）

冻结以下初版容忍规则：

- 对 secondary primary metric，默认 non-regression tolerance 为：
  - 不得劣化超过 `1 km`
- 对 intensity guardrail，默认容忍为：
  - `intensity_error_kt` 不得劣化超过 `0.3 kt`
- `strict_parseable_rate` 必须保持 `>= 0.995`
- `slot_time_match_rate_vs_official` 必须保持 `= 1.0000`
- executable schema validity 视作硬约束，不作为可牺牲项
- 若后续 pilot 证明这些阈值需要调整，必须先修订文档，再进入 formal

### 6.7 不应作为主 reward 的指标

以下信号不应被用作 Phase 3 的 main objective：

- 单独的 `reward_mean`
- 单独的 `coverage`
- standalone diagnostic `macro-F1`
- 单独的 contract metrics
- 单一语义轴的 proxy score

这些可以用于监控或诊断，但不能成为最终优化目标。

## 7. 评估协议

### 7.1 严格的数据切分纪律

Phase 3 必须清晰区分：

- training split
- pilot-tuning validation split
- locked formal test split

formal test split 不应在 pilot tuning 阶段被反复打开。

#### 必须遵守的规则（冻结）

- 所有 pilot model selection 都只能基于 validation
- locked formal test 只能用于预先声明好的 confirmatory run
- 不允许使用同一块 held-out 数据既调 reward / 权重 / checkpoint，又作为最终结论依据
- 若 formal test 已被用于 pilot 决策，则该轮不得再被称为正式 confirmatory evidence
- 所有正式结论必须明确说明它们来自 validation 还是 locked formal test

### 7.2 Prompt-regime matching

每一个 comparison block 都必须在相同 prompt regime 下进行。

例如：

- overridden-prompt RL candidate 对 overridden-prompt frozen control
- action-policy rendered candidate 对 action-policy rendered frozen control

不能比较不同 prompt regime 下的系统，然后把差异解释成干净的 RL effect。

#### Prompt-regime matching 规则（冻结）

- overridden view 只能与相同 overridden view 下的 control 做比较
- rendered policy 只能与相同 rendered policy 下的 control 做比较
- 不允许将 original non-overridden baseline 与 overridden RL candidate 直接比较后，宣称这是干净的 RL 增益
- materialized override view 必须记录 provenance

### 7.3 Reporting blocks

每一轮 formal compare 至少应报告以下四块：

1. **aggregate forecast metrics**
   - `track_error_km`
   - `mean_track_diff_vs_official_km`
   - `intensity_error_kt`
   - `mean_intensity_diff_vs_official_kt`
   - `strict_parseable_rate`
   - `slot_time_match_rate_vs_official`

2. **action validity / contract stability**
   - JSON parse rate
   - executable schema validity
   - turning-signal distribution
   - bucket distribution drift

3. **semantic-axis breakdown**
   - `lat` 相关切片
   - `lon` 相关切片
   - turning vs non-turning
   - bucket-only slices

4. **pathway / regime slices**
   - 项目中已经在使用的物理 failure modes
   - 尤其是那些专家修正价值已知更高的 slices

### 7.4 统计纪律

Phase 3 不应仅凭一个幸运 checkpoint 就宣布胜利。

最低建议：

- 对 frozen control 做 paired comparison
- 对主指标计算 bootstrap confidence intervals
- 任何想要被称为“positive”的结果，至少重复一个 seed

## 8. 决策逻辑

### 8.1 什么算 Phase 3A 正向

满足以下条件时可判为正向：

- 至少一个 primary track metric 有具有实际意义的改善
- 另一个 primary track metric 在容忍范围内非退化
- guardrails 全部满足

满足以下情况时判为负向：

- reward 上升，但 track metrics 持平或退化
- 增益只来自格式伪影
- 增益在 locked formal confirm 中消失

对负向结果的解释应为：

- forecast-only RL 不是当前主要的剩余杠杆

而不是：

- RL 对这个项目没有价值

### 8.2 什么算 Phase 3B 正向

满足以下条件时可判为正向：

- action-level RL 改善了 downstream dual-track metrics
- 接口仍然可执行
- 增益在 locked held-out confirm 上仍然存在
- 改善不局限于 trivial shortcut behavior

对正向结果的解释应为：

- 可执行 correction interface 本身包含可以通过 RL 进一步优化的 expert-gap-closing policy

这是 Phase 3 最核心的科学成功标准。

### 8.3 什么算 Phase 3C 正向

满足以下条件时可判为正向：

- alternating optimization 优于最佳 Phase 3B 冻结消费路径
- 增益仍可解释
- action contract 与 forecast validity 都没有崩塌

对正向结果的解释应为：

- correction policy 与 downstream consumer 两侧都仍然存在可以学习的协调空间

## 9. 主要风险

1. **Reward misalignment**
   - 如果 reward 设计没有和 dual-track 项目目标对齐，RL 会优化到错误行为

2. **Shortcut amplification**
   - 当前 confirmed gain 仍更强地与 `lon semantics` 相关，而不是 `lat semantics`
   - RL 可能进一步放大容易的轴，而不是修复最弱的语义轴

3. **Split contamination**
   - 若 pilot 选择与 final confirm 反复使用同一 held-out compare，会模糊研究结论

4. **Attribution collapse**
   - 如果过早做 joint training，就无法判断增益来自哪里

5. **Interface drift**
   - 如果 action-level RL 没有足够强的 anchor，可能会破坏掉正是 Phase 2 成功的那个 contract

## 10. 当前推荐推进顺序

Phase 3 应按以下顺序推进：

1. 先正式冻结 Phase 3 的 reward、guardrails 与 split discipline
2. 运行 Phase 3A residual forecast-only RL，作为一个有边界的 probe
3. 无论 Phase 3A 结果如何，都要继续准备 Phase 3B action-level RL，把它作为真正主线
4. 只有在 Phase 3B 为正后，才考虑 Phase 3C alternating optimization
5. 在上述顺序完成之前，不启动 joint end-to-end RL

## 11. 预期交付物

预期的 Phase 3 交付物包括：

- 一份冻结后的 Phase 3 设计文档
- 一份 reward-spec 文档
- 一份 split / evaluation protocol 文档
- Phase 3A 配置与结果报告
- Phase 3B action-level RL 配置与结果报告
- 若解锁，则继续产出 Phase 3C alternating optimization 设计与结果报告
- 一张与所有 retained controls 的最终比较总表
- 一份解释增益来源的 mechanism summary

## 12. Phase 3 TODO 跟踪表

### 12.1 已完成的 Phase 2 前置条件

- [x] 确认 `slot_turn_correction v3 + baseline intensity + scale 1.20` 为当前 Phase 2 推荐主线
- [x] 确认当前推荐主线在更大 held-out full confirm 上成立
- [x] 确认当前 gain 主要与 `lon semantics` 相关
- [x] 确认当前 blocker 已不再是 JSON contract 或 turning-signal collapse
- [x] 确认当前已经可以进入 RL 准备阶段
- [x] 冻结 Phase 2 的核心判断：RL 应优化 integrated forecast behavior，而不是 standalone diagnostic proxy score
- [x] 明确当前 RL-v0 计划只是一个狭义的 forecast-only residual 入口，而不是完整研究级 RL 方案
- [x] 冻结整个 Phase 3 必须持续保留的 reporting controls

### 12.2 Phase 3 设计任务

- [x] 正式将新的 RL 阶段命名为 `Phase 3`
- [x] 在文档中冻结一版正式的 `Phase 3` 目标表述
- [x] 冻结 train / pilot-val / locked formal test 的 split discipline
- [x] 冻结所有 compare block 的 prompt-regime matching 规则
- [ ] 写出正式的 Phase 3 reward specification
- [ ] 写出正式的 Phase 3 guardrail specification
- [x] 定义 dual-track main reward 的 scalarization / weighting rule
- [x] 定义 secondary primary metric 的 non-regression tolerance rule
- [ ] 补齐 dual-track RL reward 所需的 official-forecast slot 字段
- [ ] 扩展 RL reward implementation，使其支持 official-gap closing
- [ ] 增加一个专用的 Phase 3 preflight checker
- [ ] 物化 overridden RL views，用于 matched-prompt evaluation
- [x] 冻结 aggregate、semantic-axis、pathway-slice 三层 reporting template

### 12.3 Phase 3A — residual forecast-only RL probe

- [ ] 写出 `Phase 3A` smoke config
- [ ] 写出 `Phase 3A` pilot config
- [ ] 写出 `Phase 3A` formal config
- [ ] 跑 smoke
- [ ] 跑 pilot
- [ ] 仅基于 validation 选择 checkpoint
- [ ] 跑 locked formal confirm
- [ ] 写出 Phase 3A 结果报告
- [ ] 判断 forecast-only RL 是否仍存在有意义的 residual headroom

### 12.4 Phase 3B — action-level RL mainline

- [ ] 写出 action-level RL formulation document
- [x] 冻结 RL 中 action policy 的输出 contract
- [ ] 确定对 `slot_turn_correction v3` 的 anchor strategy
- [ ] 实现 action-level RL 所需的数据 / rollout plumbing
- [ ] 在 RL loop 中加入 hard schema validation
- [ ] 加入 turning-signal distribution monitoring
- [ ] 加入 bucket-drift monitoring
- [ ] 加入 `lat/lon` 分轴 reporting
- [ ] 写出 `Phase 3B` smoke config
- [ ] 写出 `Phase 3B` pilot config
- [ ] 写出 `Phase 3B` formal config
- [ ] 跑 smoke
- [ ] 跑 pilot
- [ ] 仅基于 validation 选择 checkpoint
- [ ] 跑 locked formal confirm
- [ ] 写出 Phase 3B 结果报告
- [ ] 判断 executable correction policy 本身是否包含可被 RL 优化的 expert-gap-closing signal

### 12.5 Phase 3C — constrained alternating coordination

- [x] 冻结 Phase 3C 的解锁条件
- [ ] 写出 alternating-optimization design note
- [ ] 定义 attribution-safe compare logic
- [ ] 写出 `Phase 3C` config set
- [ ] 跑 smoke
- [ ] 跑 pilot
- [ ] 仅基于 validation 选择 checkpoint
- [ ] 跑 locked formal confirm
- [ ] 写出 Phase 3C 结果报告
- [ ] 判断 alternating coordination 是否在 Phase 3B 之外继续提供增益

### 12.6 最终综合

- [ ] 构建一张面向所有 retained controls 的最终 Phase 3 comparison table
- [ ] 形成一份解释 RL gain 来源的 mechanism summary
- [ ] 明确当前主要增益究竟来自 residual forecast consumption、action-policy learning，还是 alternating coordination
- [ ] 明确判断 RL 是否以科学上有意义的方式改善了 expert-gap closing
- [ ] 冻结 Phase 3 结束后的下一研究阶段建议
