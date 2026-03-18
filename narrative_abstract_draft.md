# Narrative and Abstract Draft

本稿基于 [idea.md](idea.md) 与 [research_action_plan.md](research_action_plan.md) 的研究设想撰写，写法向 Nature、Science 及其环境与可持续领域子刊靠拢：先建立广义问题背景，再收束到一个关于 expert-added forecast skill 的科学问题，用一条明确的 Here we show 句提出核心贡献，最后落到更广泛的科学与应用意义。

## 1. Core Claim | 核心论点

### 中文
本研究的核心论点是：热带气旋预报中的关键科学对象，不只是风暴本身的可预报性，也是专家在业务流程中作出 final call 时相对于客观指导所增加的可预报信息。当前官方预报并不是任何单一模式或简单共识的直接输出，而是预报员在多源观测、环境诊断和多模式分歧基础上作出的最终裁决，并且这种 judgement 往往能够带来额外的准确性增益。只要把这种 final call 放在严格的 forecast-time information set 下定义并建模，人工智能研究就可以从模仿标签转向识别和学习 expert-added forecast skill。

### English
The central claim of this study is that a key scientific object in tropical cyclone forecasting is not only the predictability of the storm itself, but also the additional predictive signal introduced when experts make the final operational call. Official forecasts are not direct outputs of any single model or simple consensus; they are final judgements made by forecasters who weigh observations, environmental diagnosis and model disagreement, and those judgements can improve forecast accuracy. If this final-call layer is defined and modelled under a strict forecast-time information set, AI research can move beyond label imitation towards identifying and learning expert-added forecast skill.

## 2. Narrative Structure Template | Narrative 结构模板

### Background | What We Know | 背景：我们已知什么

#### 中文
热带气旋预报是环境科学中最典型的高风险预测决策场景之一。当前主导框架主要有三类：一类是以动力学和统计-动力模型为核心、持续提升轨迹和强度预测精度的物理预报框架；一类是将再分析场、卫星信息或模式输出直接映射到未来风暴状态的数据驱动框架；还有一类是近年兴起的 AI 天气模型，它们强调大尺度场的快速演算与预测能力。这些框架共同推动了热带气旋预报能力提升，但在真实业务链条中，最终发布的官方预报仍由专家作出 final call。换言之，业务系统默认承认一个事实：在多模型竞争、观测不完备和高影响天气场景下，专家 judgement 仍然能够改善最终预报。

#### English
Tropical cyclone forecasting is one of the clearest high-stakes prediction settings in environmental science. Three dominant frameworks currently shape the field: physics-based and statistical-dynamical systems that aim to improve track and intensity prediction, data-driven approaches that map reanalysis fields, satellite signals or model outputs directly to future storm states, and emerging AI weather models that emphasize fast prediction of atmospheric fields at scale. Together these approaches have advanced forecast capability, yet in real operations the official forecast is still a final call made by human experts. The forecasting system therefore already encodes a strong empirical premise: under model disagreement, incomplete observations and high-impact conditions, expert judgement can improve the final forecast.

### Gap | What Cannot Be Seen, Why It Matters, Why It Remains Unresolved | 缺口：什么看不见、为什么重要、为什么尚未解决

#### 中文
真正的缺口并不是我们还没有更详细地描述专家如何写出讨论文本，而是我们仍然缺少一个科学框架去回答更根本的问题：专家 judgement 是否包含相对于客观模式指导的系统性、可学习、可泛化的额外预报信息，以及这种信息在什么条件下能够提升准确性。这个问题之所以重要，是因为它关系到热带气旋可预报性的边界究竟由大气动力学决定，还是也受到人类对不确定性和模型失配进行仲裁的能力所约束。若这一层 added skill 无法被识别，我们就无法判断自动化预报的上限，也无法回答人机协同为何在极端天气中仍然必要。该问题长期未解决，是因为预报时刻可得信息、专家 final call 和事后真值通常没有被严格区分，现有数据集难以隔离“专家比模型多做了什么”这一科学信号。

#### English
The real gap is not that we have yet to describe in more detail how experts write forecast discussions, but that we still lack a scientific framework for a deeper question: does expert judgement contain a systematic, learnable and generalizable predictive signal beyond objective model guidance, and under what conditions does that signal improve accuracy? This matters scientifically because it bears on whether the limits of tropical cyclone predictability are set only by atmospheric dynamics, or also by the ability of humans to arbitrate uncertainty and model mismatch. If this added skill cannot be isolated, we cannot define the ceiling of automated forecasting or explain why human-machine forecasting remains necessary for extremes. The problem has persisted because forecast-time inputs, expert final calls and later verification are rarely kept strictly separate, making it difficult to isolate the scientific signal of what experts add beyond the models.

### Novelty | Why This Study, Why It Uniquely Solves the Problem | 新意：为什么提出这项研究、为什么它能独特解决问题

#### 中文
本研究的创新点不在于再提出一个更强的轨迹或强度预测器，而在于把热带气旋预报重述为一个“专家 final call 的可学习对象”。我们引入一个面向大模型的统一数据框架，严格区分四类对象：预报时刻的现状输入、预报时刻已可获得的未来指导、专家最终发布的官方输出，以及仅用于事后评估的验证真值。这样做的关键不是复原文本层面的推理细节本身，而是把专家 judgment 作为一个可以与模式指导和最终真值进行比较的科学对象，从而首次允许我们定量研究专家何时跟随共识、何时偏离共识，以及这种偏离何时真正带来准确性增益。

#### English
The novelty of this study is not that it proposes a stronger predictor of cyclone track or intensity, but that it reformulates tropical cyclone forecasting around the expert final call as a learnable scientific target. We introduce a unified data framework for large models that keeps four entities strictly separate: current inputs available at issuance, future guidance already accessible to the forecaster, official outputs issued by the expert, and verification targets reserved for later evaluation. The key contribution is therefore not the reconstruction of textual reasoning for its own sake, but the elevation of expert judgement into an object that can be compared directly with model guidance and later truth. This makes it possible, for the first time, to quantify when experts follow consensus, when they depart from it and when such departures actually improve forecast accuracy.

### Results | What Changed, How Understanding Shifted, Why It Matters | 结果：改变了什么、理解如何转变、为什么重要

#### 中文
本研究带来的根本变化，是把热带气旋 AI 研究从“预测风暴未来状态”推进到“识别并学习专家 final call 所携带的额外预报技巧”。在这一框架下，核心问题不再只是哪个模型最接近真值，而是专家在何种分歧结构、环境条件和不确定性水平下能够系统性地改进客观指导。理解上的转变在于，AI 的角色不再是简单替代模式或复述官方文本，而是去刻画一个此前几乎未被显式定义的 scientific layer: expert-added predictability。其意义在于，这不仅为可解释的人机协同预报提供了方法，也为研究极端天气业务决策中的“可预报性增益来自哪里”提供了新的科学路径。

#### English
The main change introduced by this study is a shift from predicting storm evolution alone to identifying and learning the additional forecast skill carried by expert final calls. Under this framework, the key question is no longer merely which model is closest to truth, but under what structures of disagreement, environmental regimes and uncertainty levels experts can systematically improve on objective guidance. The conceptual shift is that AI is not used simply to replace models or restate official text, but to characterize a scientific layer that has rarely been made explicit: expert-added predictability. This matters because it opens a route both to interpretable human-AI forecasting and to a deeper understanding of where forecast skill for high-impact extremes actually comes from.

## 3. Abstract Draft | 摘要草稿

### 中文
热带气旋预报的进步通常归因于数值模式、统计-动力方法和新兴人工智能天气模型对风暴演变可预报性的持续挖掘。然而在真实业务流程中，官方预报并不是任何单一模式或简单共识的直接输出，而是由预报员在多源观测、环境诊断和多模式分歧基础上作出的 final call，并且这种 judgement 往往能够提升最终预报准确性。一个尚未解决的科学问题是，专家 judgment 是否包含相对于客观指导的系统性、可学习的额外预报信息，以及这种增益在什么条件下出现。现有数据集难以回答这一问题，因为它们常常混合预报时刻可得信息与事后真值，也缺少对“模式指导-官方 final call-最终验证”这一关键关系的严格刻画。这里我们表明，若将 NOAA 预报讨论、官方 advisory、多模式指导和环境再分析数据按预报发布时间进行因果对齐，热带气旋预报就可以被重构为一个 expert final-call learning problem，而不仅是一个风暴状态预测问题。这一框架严格区分现状输入、未来指导、官方输出与验证真值，从而能够识别专家何时跟随模式共识、何时偏离共识，以及这种偏离何时真正转化为准确性提升。通过把 expert-added skill 变成显式、可监督且可审计的研究对象，该方法为构建更透明、更有效的人机协同气旋预警系统提供了基础。

### English
Progress in tropical cyclone forecasting is usually framed around improved prediction of storm evolution by numerical models, statistical-dynamical methods and, increasingly, AI weather systems. In operations, however, the official forecast is not the direct output of any single model or simple consensus, but a final call made by forecasters who weigh heterogeneous observations, environmental diagnosis and model disagreement, and that judgement can improve forecast accuracy. A central unresolved scientific question is therefore whether expert judgement contains a systematic, learnable predictive signal beyond objective guidance, and under what conditions that added skill emerges. Existing datasets are poorly suited to this question because they often mix forecast-time information with post hoc truth and do not represent the critical relationship between model guidance, the official final call and later verification. Here we show that tropical cyclone forecasting can be reformulated as an expert final-call learning problem by causally aligning NOAA forecast discussions, official advisories, multi-model guidance and environmental reanalysis at forecast issuance. This alignment yields a data structure that separates current inputs, future guidance, official outputs and later verification, making it possible to identify when experts follow consensus, when they depart from it and when such departures improve accuracy. By turning expert-added skill into an explicit, supervised and auditable target, this framework offers a route to more transparent and more effective human-AI warning systems for climate-related extremes.

## 4. Sentence To Update After Experiments | 实验完成后建议替换的一句

### 中文
如果后续已经拿到定量结果，建议在摘要中把“这里我们表明”后面的第一句替换为一条更强的结果句，例如：与只依赖客观指导或直接学习输入到输出映射的基线相比，该框架更准确地识别了专家偏离模式共识并提升最终预报准确性的情形。

### English
Once quantitative results are available, the sentence immediately after “Here we show” should be strengthened with a concrete finding, for example: compared with baselines that rely only on objective guidance or direct input-to-output mapping, the proposed framework more accurately identifies cases in which expert departures from model consensus improve the final forecast.