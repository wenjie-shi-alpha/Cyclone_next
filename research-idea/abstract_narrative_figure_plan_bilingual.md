# Updated Abstract, Narrative, and Figure Plan

This file updates the abstract and narrative based on the revised storyline in `research-idea/research-idea-pro.md`, while keeping the overall scientific writing style relatively close to the previous draft. It also proposes a bilingual main-text figure plan for the paper.

---

## 中文版

### Updated Abstract

热带气旋预报是高风险环境预测的典型场景，直接影响疏散、基础设施保护和应急响应。尽管数值天气预报、统计-动力方法和 AI 天气模型持续提高了对风暴演变的预测能力，官方预报仍不是任何单一模式或简单共识的直接输出，而是专家综合观测、环境诊断和多模式指导后的最终裁决。专家判断是否在客观指导之外，尤其在模式结构化失效情境下，包含系统性、可学习且可验证的额外预测技能，至今仍未得到清晰回答，因为预报时刻信息、归档指导、官方输出和事后验证通常没有被因果一致地分离。这里我们表明，在严格的 issuance-time information set 下，热带气旋业务预报可以被形式化为一个以 expert adjudication 为中心的条件化推理问题。我们构建了一个显式分离现状输入、客观指导、官方输出和验证真值的 issuance-aligned dataset，将官方预报表示为客观指导加上专家调整，并提出结合物理诊断、大语言模型推理学习和 verification-aware offline reinforcement learning 的三层框架。该 formulation 将 forecaster-added skill 转化为可学习、可审计和可检验的科学对象，为识别人类判断何时真正增加了可验证信息，以及构建更透明的人机协同热带气旋预报系统，提供了基础。

### Updated Narrative

#### 一句话叙事

业务预报的最后一公里不是计算，而是条件化推理；我们的目标不是让 AI 简单模仿官方产品，而是让它学会在模式会出错的地方做出可验证、可解释、并最终可能超越专家的判断。

#### Dark Spot｜真正的盲区

当前热带气旋 AI 研究的主战场仍然是“更准确地预测未来风暴状态”，无论是数值模式、统计后处理还是新一代 AI 天气模型，核心都集中在预报链条的上游计算环节。但在真实业务流程中，最终发布的官方预报并不是任何单一模式或简单共识的直接输出，而是预报员在观测证据、环境诊断和多模式分歧之间作出的最终裁决。真正的 dark spot 因而不在于我们还不会预测风暴本身，而在于我们尚未把这层从 objective guidance 到 official forecast 的专家 adjudication 作为一个独立科学对象来研究。

#### Why the Dark Spot Still Exists｜为什么这个盲区仍然存在

这个盲区长期存在，并不是因为它不重要，而是因为它在数据和方法上都很难被干净地隔离出来。现有研究通常要么直接预测事后真值，要么把官方预报当成标签进行模仿，却很少严格区分预报时刻可得输入、归档指导、官方输出和事后验证。这样一来，专家到底是在重复模式、修正模式，还是在模式结构化失效时增加了额外信息，就被混在一起了。更进一步说，前序研究已经表明 expert-added skill 主要集中在少数 structured error regimes 中，而不是在所有样本上均匀出现；如果缺少 issuance-aligned 数据组织和 regime-aware 分析框架，这种选择性增益就会被平均误差和标签模仿逻辑掩盖掉。

#### What We Do｜我们做什么

为了解决这一问题，我们首先构建一个严格按 issuance time 因果对齐的业务预报数据集，显式分离 `now inputs`、`guidance inputs`、`official outputs` 和 `verification targets`，并将官方预报形式化为客观指导加上专家调整。在此基础上，我们把业务预报重新定义为一个条件化推理任务：模型需要先识别当前处于哪一种误差 regime、哪些模式在该环境下更脆弱，再决定是否跟随共识、偏向哪个指导、以及应调整多少。围绕这一 formulation，我们提出一个三层框架：第一层用物理诊断模块识别环境配置、误差 regime 和模式风险；第二层用大语言模型的监督微调学习专家在不同情境下的推理模式与基础调整行为；第三层再用 verification-aware offline reinforcement learning 优化模型何时以及如何偏离指导，并尽可能过滤保守、锚定和过度折中等人类偏差。

#### What Changes, How Understanding Shifted, and Why It Matters｜改变了什么、理解如何转变、为什么重要

这一框架改变的，不只是模型结构，而是问题本身的定义方式。它把研究问题从“谁最接近真值”推进为“谁在同一信息集下真正增加了可验证信息”，也把官方预报从一个待模仿的标签，转变为一个可以被学习、比较和审计的 situated judgment 对象。理解上的转变在于，专家价值不再被视为模糊经验，而被具体化为在模式脆弱性与环境条件共同约束下的条件化推理过程。其重要性在于，这使我们第一次可以在因果一致的框架下直接检验 expert-added skill 是否稳定、可迁移、可被机器学习，以及 RL 是否能够在保留专家有效技能的同时进一步优化决策边界。更广泛地说，这项研究为透明的人机协同热带气旋预报系统提供了基础，也为所有存在“模型输出 - 专家裁决 - 最终决策”链条的高风险预测领域提供了一种新的 AI 研究路径。

### Main-Text Figure Plan

建议主文使用 6 张图。这个数量足以形成完整叙事链，同时把数据统计、训练细节、更多 case study 和更细的专家评估留给 Extended Data 或 Supplementary。

#### Figure 1. The Reasoning Gap in Operational Tropical-Cyclone Forecasting

- 核心内容：界定本文的科学缺口，说明现代 AI/NWP 主要解决了“计算未来状态”的问题，但业务预报最后仍需要专家在模式分歧和环境脆弱性下做判断。
- 回答的问题：为什么这不是一个普通的后处理问题，而是一个尚未被 AI 真正建模的推理问题？
- 对叙事的贡献：作为开篇图，建立全文问题意识，把“官方预报不是模式直接输出”这件事讲清楚，并自然引出 expert-added skill 的研究对象。
- 建议形式：概念示意图，分成上下两个 panel。
- 推荐 panel：
  - Panel a：业务预报链条示意，`Observations -> NWP/AI guidance -> Expert adjudication -> Official forecast`，明确标注前两段已有大量 AI 工作，最后一段是本文要解决的 gap。
  - Panel b：引用 Paper 1 的核心发现，用小型热图、散点分箱图或 regime 条件柱状图展示“专家增益集中在 structured error regimes”，把 gap 从概念问题落到经验事实。

#### Figure 2. Issuance-Aligned Dataset and Three-Layer Learning Framework

- 核心内容：说明数据如何按预报发布时间进行因果对齐，并把 learning target 定义为“在给定信息集下学习官方预报中的条件化判断”。
- 回答的问题：我们到底学习什么？使用哪些信息？如何避免把事后真值泄漏进训练输入？
- 对叙事的贡献：这是全文的方法学支点，把抽象叙事正式转化为一个可检验的科学设计。
- 建议形式：双层结构图或左右两栏图。
- 推荐 panel：
  - Panel a：时间轴式数据示意，清楚分出 `now inputs`, `guidance inputs`, `official outputs`, `verification targets`，并标注 verification 只用于评估或 RL reward。
  - Panel b：三层系统架构图，展示 `physical diagnosis -> LLM reasoning/SFT -> offline RL refinement` 的数据流与输出形式。

#### Figure 3. Overall Performance Ladder

- 核心内容：给出全文最重要的总体结果，对比单模式、共识、传统 ML、官方专家预报、SFT 版本和完整系统的 track/intensity 技巧。
- 回答的问题：学习 expert reasoning 是否真的能带来经过验证的预报增益？这种增益相对专家和客观指导有多大？
- 对叙事的贡献：这是全文的主结果图，负责把“概念新意”转化为“性能成立”。
- 建议形式：主图用柱状图或点图，辅以分时效折线图。
- 推荐 panel：
  - Panel a：总体 `Track MAE` 和 `Intensity MAE` 的横向柱状图或 forest plot。
  - Panel b：按 lead time 展示各主要系统的误差曲线，体现增益是否稳定、在哪些时效最强。
  - Panel c：bootstrap confidence intervals 或 paired significance test，证明相对 expert/consensus 的优势具有统计稳健性。

#### Figure 4. Where the Gains Occur: Regime-Conditional and High-Impact Performance

- 核心内容：分解系统在不同误差 regime、快速增强、路径转折、登陆前、高分歧样本等关键子集中的表现。
- 回答的问题：系统的增益是均匀存在，还是主要集中在最困难、最需要判断的情境中？
- 对叙事的贡献：这一图把方法有效性的“位置”讲清楚，证明我们学到的不是平均意义上的小修小补，而是最有业务价值的 conditional reasoning。
- 建议形式：热图加子集柱状图，或 small multiples。
- 推荐 panel：
  - Panel a：`Regime x Model` 的性能热图，直接展示各系统在不同误差 regime 下的强弱变化。
  - Panel b：RI、recurvature、landfall 和 large-spread cases 的 grouped bar charts，突出高影响场景。
  - Panel c：可附一个专家增益 vs 模型脆弱性的关系图，进一步呼应 Figure 1 的问题设定。

#### Figure 5. Why the System Works: Component Attribution and Ablation

- 核心内容：严格分析环境特征、regime 诊断、LLM 架构和 RL 优化分别贡献了什么，并与 XGBoost/MLP 等传统后处理路线比较。
- 回答的问题：为什么必须是“物理诊断 + 推理模型 + RL”，而不是更简单的表格 ML 或直接回归？
- 对叙事的贡献：这是机制层面的关键图，用来支撑“这不是模型堆料，而是任务表述正确”。
- 建议形式：消融矩阵加效应量点图。
- 推荐 panel：
  - Panel a：完整 ablation matrix，逐步加入 `env features`, `regime diagnosis`, `LLM`, `RL`。
  - Panel b：几个关键对比的效应量图，例如 `XGBoost(full) vs LLM-SFT(full)`、`LLM-SFT vs LLM-RL`。
  - Panel c：如果结果支持，可加入一个 interaction plot，证明物理诊断与 LLM 之间存在协同增益，而非简单相加。

#### Figure 6. Reasoning Traces, Bias Correction, and Operational Case Studies

- 核心内容：用少量高质量案例展示系统如何解释自己的判断、如何在复杂环境中选择信任哪些指导，以及 RL 如何从模仿专家走向超越专家。
- 回答的问题：模型是否可解释？它学到的究竟是物理上合理的判断，还是只是在数值上拟合标签？RL 是否真的修正了人类偏差？
- 对叙事的贡献：作为收束图，这一图把“能不能做”推进到“为什么值得信任、为什么它不只是模仿”，支撑全文 broader significance。
- 建议形式：案例研究图，多 panel 组合。
- 推荐 panel：
  - Panel a：1-2 个代表性案例的路径和强度对比图，叠加 best track、official forecast 和系统输出。
  - Panel b：对应案例的 reasoning trace 摘录，配合环境示意图或 regime 标签，突出模型依据了哪些物理机制。
  - Panel c：`SFT vs RL` 调整分布对比，或偏差分解图，展示 RL 如何缓解 conservatism、anchoring 或 over-centering。

---

## English Version

### Updated Abstract

Tropical-cyclone forecasting is a high-stakes environmental prediction problem with direct consequences for evacuation and emergency response. Although numerical weather prediction, statistical-dynamical methods and AI weather models have improved forecasts of storm evolution, official forecasts are still not direct outputs of any single model or simple consensus, but expert adjudications that integrate observations, environmental diagnosis and multi-model guidance. Whether these expert adjustments contain systematic, learnable and verifiable predictive skill beyond objective guidance, especially in structured model-failure regimes, remains unresolved because forecast-time information, archived guidance, official outputs and later verification are rarely separated in a causally consistent way. Here we show that, under a strict issuance-time information set, operational tropical-cyclone forecasting can be reformulated as a conditional-reasoning problem centered on expert adjudication. We construct an issuance-aligned dataset that separates current inputs, objective guidance, official outputs and later verification, formalize the official forecast as guidance plus expert adjustment, and propose a three-layer framework combining physical diagnosis, large-language-model reasoning and verification-aware offline reinforcement learning. This formulation turns forecaster-added skill into a learnable, auditable and testable scientific object, opening a path to transparent human-AI forecasting systems and to a clearer understanding of when human judgement adds verifiable value.

### Updated Narrative

#### One-Sentence Narrative

The last mile of operational forecasting is not computation but conditional reasoning; the goal is not to make AI merely imitate the official product, but to make it learn how to judge model failure regimes in ways that are verifiable, interpretable and potentially better than expert adjudication.

#### Dark Spot

The main blind spot in current tropical-cyclone AI is not the prediction of storm evolution itself, but the adjudication layer between objective guidance and the final operational forecast. Numerical models, statistical post-processing and AI weather systems have advanced the upstream calculation of future states, yet the official forecast is still not the direct output of any single model or simple consensus. It is the result of expert judgement exercised across observations, environmental diagnosis and multi-model disagreement. The true dark spot is therefore the expert decision layer that turns guidance into the forecast that is actually issued.

#### Why the Dark Spot Still Exists

This dark spot persists not because it is secondary, but because it has been difficult to isolate cleanly in both data and methodology. Existing studies typically either predict post hoc truth directly or treat the official forecast as a label to imitate, while rarely separating forecast-time inputs, archived guidance, official outputs and later verification under a strict causal structure. As a result, it remains unclear whether experts are merely repeating guidance, slightly smoothing it, or adding predictive information precisely when models fail in structured ways. Previous analysis further suggests that expert-added skill is concentrated in a limited set of structured error regimes rather than spread uniformly across all cases; without issuance-aligned data and regime-aware analysis, that selective gain is easily washed out by average metrics and label-imitation objectives.

#### What We Do

We address this problem by constructing an issuance-aligned operational dataset that explicitly separates `now inputs`, `guidance inputs`, `official outputs` and `verification targets`, and by formalizing the official forecast as objective guidance plus expert adjustment. On that basis, we recast operational forecasting as a conditional-reasoning task: the model must identify the current error regime, infer which guidance sources are vulnerable under the present environment, and decide whether to follow consensus, favor a particular model or depart more substantially from all of them. We then propose a three-layer framework: a physical-diagnosis layer that identifies environmental structure, regime type and model risk; a supervised large-language-model layer that learns expert reasoning patterns and baseline adjustment behavior; and a verification-aware offline reinforcement-learning layer that optimizes when and how the system should depart from guidance while filtering human biases such as conservatism, anchoring and over-centering.

#### What Changes, How Understanding Shifted, and Why It Matters

What changes here is not only the model architecture, but the definition of the scientific problem itself. The question shifts from who best predicts truth to who adds verified information under the same operational information set, and the official forecast shifts from a label to be imitated into a situated-judgement object that can be learned, audited and compared. The conceptual shift is that expert value is no longer treated as diffuse experience, but as conditional reasoning expressed under jointly structured environmental conditions and model vulnerabilities. This matters because it makes it possible, for the first time, to test under causal consistency whether expert-added skill is stable, transferable and learnable by machines, and whether reinforcement learning can preserve genuine human skill while improving the decision boundary beyond expert behavior. More broadly, the framework provides a foundation for transparent human-AI tropical-cyclone forecasting and suggests a general path for AI in other high-stakes domains that also depend on the chain of model output, expert adjudication and final decision.

### Main-Text Figure Plan

I recommend 6 main-text figures. That is enough to build a complete narrative arc while leaving dataset statistics, training details, additional case studies and fuller expert-evaluation material to Extended Data or the Supplement.

#### Figure 1. The Reasoning Gap in Operational Tropical-Cyclone Forecasting

- Core content: define the scientific gap by showing that modern AI/NWP systems mainly address the calculation of future states, whereas operational forecasting still requires expert judgement under model disagreement and environmental vulnerability.
- Question answered: why is this not just another post-processing task, but a reasoning problem that AI has not yet modeled well?
- Contribution to the narrative: this is the opening figure; it establishes the problem statement and clarifies why the official forecast is a distinct scientific object.
- Suggested visual form: a conceptual figure with two panels.
- Recommended panels:
  - Panel a: an operational forecast-chain schematic, `Observations -> NWP/AI guidance -> Expert adjudication -> Official forecast`, with the first two steps marked as already heavily studied by AI and the third as the gap addressed here.
  - Panel b: a compact empirical panel inspired by Paper 1, such as a heat map, binned scatter or regime-wise bar plot, showing that expert gains are concentrated in structured error regimes.

#### Figure 2. Issuance-Aligned Dataset and Three-Layer Learning Framework

- Core content: show how the data are causally aligned at forecast issuance and define the learning target as conditional judgement under a strict forecast-time information set.
- Question answered: what exactly is being learned, what information is available, and how is leakage from post hoc verification prevented?
- Contribution to the narrative: this is the methodological anchor of the paper; it turns the conceptual story into a formal and testable design.
- Suggested visual form: a two-level schematic or two-column figure.
- Recommended panels:
  - Panel a: a timeline-style data diagram separating `now inputs`, `guidance inputs`, `official outputs` and `verification targets`, with verification explicitly reserved for evaluation or RL reward only.
  - Panel b: a three-layer architecture diagram showing `physical diagnosis -> LLM reasoning/SFT -> offline RL refinement` and the resulting forecast-plus-reasoning output.

#### Figure 3. Overall Performance Ladder

- Core content: present the headline quantitative result by comparing single models, consensus systems, conventional ML, the official human forecast, the SFT model and the full system on track and intensity skill.
- Question answered: does learning expert reasoning produce verified forecast gains, and how large are those gains relative to objective guidance and human experts?
- Contribution to the narrative: this is the primary results figure; it converts conceptual novelty into demonstrated forecasting value.
- Suggested visual form: a bar or dot plot as the main view, complemented by lead-time curves.
- Recommended panels:
  - Panel a: horizontal bars or a forest plot for overall `Track MAE` and `Intensity MAE`.
  - Panel b: lead-time error curves for the major systems to show whether gains are stable and where they are largest.
  - Panel c: bootstrap confidence intervals or paired significance tests to establish statistical robustness versus expert and consensus baselines.

#### Figure 4. Where the Gains Occur: Regime-Conditional and High-Impact Performance

- Core content: decompose performance across error regimes, rapid intensification, recurvature, pre-landfall periods, large-spread cases and other hard subsets.
- Question answered: are the gains uniform, or are they concentrated in the hardest situations where expert judgement matters most?
- Contribution to the narrative: this figure localizes the value of the method and shows that the system is not merely making small average improvements, but learning the most operationally valuable forms of conditional reasoning.
- Suggested visual form: a heat map combined with subset bar charts or small multiples.
- Recommended panels:
  - Panel a: a `Regime x Model` performance heat map showing which systems succeed or fail in each structured regime.
  - Panel b: grouped bar charts for RI, recurvature, landfall and large-spread subsets, highlighting high-impact situations.
  - Panel c: if useful, a relationship plot linking expert gain to model vulnerability, reinforcing the setup from Figure 1.

#### Figure 5. Why the System Works: Component Attribution and Ablation

- Core content: isolate the roles of environmental features, regime diagnosis, LLM-based reasoning and RL optimization, and compare them with simpler tabular or neural post-processing baselines.
- Question answered: why is the proposed combination of physical diagnosis, reasoning architecture and RL necessary, rather than a simpler regression pipeline?
- Contribution to the narrative: this is the mechanism figure; it supports the claim that the framing is correct, not just that the model is larger.
- Suggested visual form: an ablation matrix plus effect-size plots.
- Recommended panels:
  - Panel a: a full ablation matrix with incremental addition of `env features`, `regime diagnosis`, `LLM` and `RL`.
  - Panel b: effect-size plots for key comparisons such as `XGBoost(full) vs LLM-SFT(full)` and `LLM-SFT vs LLM-RL`.
  - Panel c: if supported by results, an interaction plot showing that physical diagnosis and LLM reasoning produce a synergistic benefit rather than a simple additive one.

#### Figure 6. Reasoning Traces, Bias Correction, and Operational Case Studies

- Core content: use a small number of strong cases to show how the system explains its decisions, how it chooses which guidance to trust in complex environments, and how RL moves the model from expert imitation toward beyond-expert behavior.
- Question answered: is the model interpretable, is its reasoning physically meaningful, and does RL truly correct human biases rather than only fit labels numerically?
- Contribution to the narrative: this is the closing figure; it turns the result from “it works” into “it is trustworthy and scientifically interesting,” supporting the broader significance of the paper.
- Suggested visual form: a multi-panel case-study figure.
- Recommended panels:
  - Panel a: track and intensity overlays for one or two representative storms, including best track, official forecast and system output.
  - Panel b: excerpts of the reasoning trace paired with an environmental schematic or regime label, highlighting the physical mechanisms referenced by the model.
  - Panel c: an `SFT vs RL` adjustment-distribution comparison or a bias-decomposition panel showing how RL reduces conservatism, anchoring or over-centering.
