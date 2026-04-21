# Phase 2 Diagnostic Supervision Design

## 1. Positioning

本项目的二阶段不是把 `reasoning_text` 混回 forecast 主任务，而是引入 `diagnostic/process supervision`，验证某种环境 / 过程信息接口是否真的能帮助 forecast。

当前推荐直接建立在新的共享数据根目录上：

- `data/training_rebuilt_v2_20260414_guidancefix`

这套数据已经修复了 guidance 对齐和伪 consensus 问题，因此可以作为 `/root/Cyclone_next` 和 `AIExpert` 共用的 shared-evidence 底座。

二阶段的目标不是“多做一个分类任务”，而是回答两个更窄的问题：

1. 能否设计出一种环境 / 过程信息接口，在 oracle 注入时真实降低 `track_error_km` 或 `mean_track_diff_vs_official_km`。
2. 只有在 oracle 通过后，模型能否从同样的输入证据中恢复该接口，并把一部分 oracle 增益带回 forecast。

### 1.1 Status Reset: 2026-04-16

`diagnostic_track_turn_only` 的两轮正式实验已经完成：

- `phase2_diagnostic_track_turn_v0_20260416_131804`
- `phase2_diagnostic_track_turn_v0_1_20260416_205033`

当前结论已经足够明确：

- standalone diagnostic 有小幅提升，但不是决定性提升
- `oracle diagnostics + forecast` 没有改善 track 主指标
- `predicted diagnostics + forecast` 也没有改善 track 主指标
- 当前 `track_control_signal + turning_signal` 这套环境标签接口，只能保留为狭义 probe / explanation 分支，不能继续当作 forecast-accuracy 主线

从这一版文档开始，后文所有 “Phase 2 objective / go-no-go / next step” 都以这个 reset 为准，不再沿用“只要 diagnostics 有一点 standalone 提升就继续扩字段”的隐含假设。

### 1.2 Status Reset: 2026-04-18

`diagnostic_track_correction_only` 的 formal oracle gate 也已经完成，结论同样是 negative：

- run root: `runs/phase2_diagnostic_track_correction_oracle_v0_20260418_144247`
- baseline `track_error_km = 184.6035`
- oracle `track_error_km = 184.5662`
- baseline `mean_track_diff_vs_official_km = 165.8990`
- oracle `mean_track_diff_vs_official_km = 169.2600`

这说明：

- “把环境信息再结构化一点” 本身不是目标
- 即使改成 `48h/72h` 的 fixed-lead correction anchors，接口仍然太间接
- Phase 2 当前真正要验证的是：**能不能把可见 guidance 直接变成 slot-locked 的 track correction interface，并在 oracle 注入时缩小 forecast track gap**

因此从这一节开始，当前 active mainline 切换为：

- `diagnostic_slot_correction_only`

它不再输出抽象环境标签，也不再输出固定 `48h/72h` lead 的 anchor，而是对 official forecast 已有槽位直接预测：

- `slot_1..slot_6` 的 `lat/lon correction bucket`
- 相对对象是 prompt 中已经可见的 ATCF representative points
- 不允许创建、删除或平移 `Day/HHMMZ` 槽位
- 不承担 intensity 修正语义

### 1.3 Status Reset: 2026-04-18 (After Slot-Correction Oracle Gate)

`diagnostic_slot_correction_only` 的 formal oracle gate 现已完成，并且这是当前 **唯一通过** Phase 2A gate 的接口：

- run root: `runs/phase2_diagnostic_slot_correction_oracle_v0_20260418_154645`
- baseline `track_error_km = 184.6035`
- oracle `track_error_km = 104.6487`
- baseline `mean_track_diff_vs_official_km = 165.8990`
- oracle `mean_track_diff_vs_official_km = 49.2995`
- baseline `reward_mean = 0.2374`
- oracle `reward_mean = 0.3990`

这个结果的意义是：

- 当前第一次出现了显著、方向明确、且可解释的 `track` 主指标增益
- 增益并不是“visible ATCF 本来就更好”，因为 `visible_atcf_consensus_passthrough_v0` 反而更差
- Phase 2 已经从 interface redesign 阶段推进到 `predicted interface recovery`

因此，当前默认下一步不再是继续 redesign，而是：

- 训练 `diagnostic_slot_correction_only` predicted adapter
- 检查它能否恢复足够比例的 oracle track gain

### 1.4 Status Reset: 2026-04-18 (After Slot-Correction Predicted v0 Gate)

`diagnostic_slot_correction_only` 的第一轮 predicted recovery 已完成：

- run root: `runs/phase2_diagnostic_slot_correction_v0_20260418_161630`
- standalone `json_parse_rate = 0.0000`
- standalone `joint_exact = 0.0000`
- standalone `mean_field_acc = 0.1033`
- standalone `mean_macro_f1 = 0.0408`
- predicted forecast `reward_mean = 0.3264`
- predicted forecast `track_error_km = 167.69`
- predicted forecast `mean_track_diff_vs_official_km = 148.35`
- predicted forecast `coverage = 1.0000`
- predicted forecast `slot_time_match_rate_vs_official = 1.0000`
- predicted forecast `intensity_error_kt = 14.73`

这轮结果说明：

- `predicted slot_correction` 已经在 forecast 侧拿回一部分 oracle track gain
- 但 raw diagnostic adapter 仍未守住 JSON contract，standalone 仍属失败
- 当前 recovery 主要依赖 null -> `near_consensus` fallback，强度问题依旧未解

目前 root-cause audit 已经定位到 output contract / prompt budget：

- `diagnostic_slot_correction_only` 的 val prompt 中位数约 `1792` tokens
- v0 训练 `max_seq_length = 1536`
- v0 compare 默认 `MAX_PROMPT_TOKENS = 1024`
- 大部分样本在训练与评测两侧都发生截断
- 旧 recovery 对半结构化字段行利用不足，仍会把大量 parse fail 压成全-null payload

因此当前下一步仍然不是 redesign，而是 `slot_correction` recovery 修复：

- 压缩 `diagnostic_slot_correction_only` prompt 与 target
- 为 predicted diagnostics 增加 line-based recovery gate
- 用 `slot_correction v0 final_adapter` 作为 staged init，启动 `slot_correction v1` 微调

### 1.5 Status Reset: 2026-04-18 (After Slot-Correction Predicted v1 Gate)

`diagnostic_slot_correction_only` 的第二轮 predicted recovery 现已完成，`v1` 相比 `v0` 已经确认修复了 output contract 主问题：

- run root: `runs/phase2_diagnostic_slot_correction_v1_20260418_184506`
- best checkpoint: `checkpoint-160`
- standalone `json_parse_rate = 0.8850`
- standalone `exact_keyset_rate = 0.8850`
- standalone `joint_exact = 0.0100`
- standalone `mean_field_acc = 0.4104`
- standalone `mean_macro_f1 = 0.3593`
- predicted forecast `reward_mean = 0.3287`
- predicted forecast `track_error_km = 163.44`
- predicted forecast `mean_track_diff_vs_official_km = 143.34`
- predicted forecast `coverage = 1.0000`
- predicted forecast `slot_time_match_rate_vs_official = 1.0000`
- predicted forecast `intensity_error_kt = 14.73`

这轮结果说明：

- `slot_correction v1` 已经不再是“纯 fallback 撑出来的 gain”，raw diagnostic adapter 本身开始守住 JSON contract
- predicted forecast 侧继续逼近 oracle 上限，但增益仍然集中在 `track`，不是 `intensity`
- 当前 `track_error_km` 口径下已恢复约 `26.5%` 的 oracle gain，但 `mean_track_diff_vs_official_km` 口径下仅恢复约 `19.4%`

当前最重要的新结论是：

- Phase 2 当前不再卡在 schema redesign，也不再主要卡在 training hygiene
- `slot_correction` 作为 track-only interface 已经基本坐实是正确方向
- 下一步最应该做的不是继续重做 `slot_correction` schema，而是解决 **track gain 已拿到、intensity 仍被 visible-consensus renderer 锁死** 的 integration 缺口
- 按当前正式 gate，`track_error_km` 和 `mean_track_diff_vs_official_km` 两个主轨迹指标都要过线；因此 `slot_correction v1` 还不能算 Phase 2B 正式通过

因此从这一节开始，当前主线更新为：

- 冻结 `diagnostic_slot_correction_only` 为 Phase 2 的 track 主线接口
- 保留现有 `slot-locked` track renderer 不动
- 下一步优先做 `intensity` 通道解耦 / recovery gate，而不是恢复更宽 schema
- 在是否恢复 `track_core` / `core` 之前，先满足新的 Phase 2B gate：`track_error_km` 和 `mean_track_diff_vs_official_km` 两项都稳定过线，且 intensity 不再系统性恶化

当前所处研究阶段可以直接概括为：

- 已经完成“找到正确轨迹接口”这一步
- 正在进行“让模型把这个接口稳定学会，并把 oracle 增益真正兑现成 predicted gain”这一步
- 还没有进入“恢复更宽 schema / 更完整 diagnostics 扩展”这一步

### 1.6 Status Reset: 2026-04-19 (After Slot-Correction Predicted v2 Gate)

`diagnostic_slot_correction_only v2` 的去塌缩微调已经正式完成，但结论是 negative：

- run root: `runs/phase2_diagnostic_slot_correction_v2_20260418_224242`
- best checkpoint: `checkpoint-160`
- standalone `json_parse_rate = 0.8250`
- standalone `joint_exact = 0.0150`
- standalone `mean_field_acc = 0.4417`
- standalone `mean_macro_f1 = 0.3450`
- predicted forecast `reward_mean = 0.3307`
- predicted forecast `track_error_km = 170.67`
- predicted forecast `mean_track_diff_vs_official_km = 150.50`
- predicted forecast `intensity_error_kt = 14.73`

这轮结果说明：

- `v2` 虽然让部分 standalone 指标上升，但没有把提升转成更好的 forecast track 主指标
- 相比当前最优的 `v1`，`v2` 在两个主轨迹指标上都退化了
- 以当前正式 gate 看，`v2` 证明“继续在 slot_correction 头上做同类重采样微调”已经不是最优主线

因此从这一节开始，主线再次收敛：

- 当前最优 predicted adapter 仍然是 `slot_correction v1`
- `slot_correction v2` 作为负结果保留，不再继续沿同一路径做 `v3`
- 下一条主线切换为：**保持 `slot_correction v1` 的轨迹修正不动，只替换强度来源，验证 intensity integration gate**

### 1.7 Status Reset: 2026-04-19 (After Intensity Integration Gate v0)

`slot_correction v1 track + baseline forecast intensity` 的第一轮 integration gate 已完成，结论是 positive but incomplete：

- run root: `runs/phase2_slot_correction_intensity_gate_v0_20260419_025200`
- predicted hybrid `reward_mean = 0.3350`
- predicted hybrid `track_error_km = 163.44`
- predicted hybrid `mean_track_diff_vs_official_km = 143.34`
- predicted hybrid `intensity_error_kt = 13.21`
- predicted hybrid `mean_intensity_diff_vs_official_kt = 10.62`

这轮结果说明：

- 只替换强度来源，不动 `slot_correction v1` 轨迹修正，本身就是有价值的
- track 两项主指标与 `slot_correction v1` 保持不变，没有被强度替换破坏
- intensity 指标明显好于原先的 visible-consensus intensity，已经从 `14.73 / 13.42` 收敛到 `13.21 / 10.62`
- 但强度仍然略差于 baseline，因此这一步还不能算整个 gate 完成

因此当前主线继续收敛为：

- `slot_correction v1 + baseline intensity` 是比 `slot_correction v1 + consensus intensity` 更好的 integration 基线
- 既然 intensity 通道已经基本被拆开，当前剩余的主要 blocker 就只剩 `track`，尤其是 `mean_track_diff_vs_official_km`
- 下一步不再优先做新的 intensity 训练头，而是先在这条 hybrid 基线上做 **track calibration sweep**

### 1.8 Status Reset: 2026-04-19 (After Track Calibration Sweep v0)

`slot_correction v1 track + baseline intensity` 上的第一轮全局 `offset_scale` sweep 已完成，结论是 clear positive：

- run root: `runs/phase2_slot_correction_scale_sweep_v0_20260419_104026`
- scale sweep: `0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20`
- best current candidate: `scale = 1.20`
- predicted scaled hybrid `reward_mean = 0.3472`
- predicted scaled hybrid `track_error_km = 153.33`
- predicted scaled hybrid `mean_track_diff_vs_official_km = 132.48`
- predicted scaled hybrid `intensity_error_kt = 13.21`
- predicted scaled hybrid `mean_intensity_diff_vs_official_kt = 10.62`

这轮结果说明：

- 从 `1.00 -> 1.20`，两个主轨迹指标都是单调改善，说明当前不是 correction 方向错了，而是校准幅度整体偏保守
- 在“predicted 至少恢复 oracle gain 的 25%”这一定义下，`scale = 1.15` 与 `scale = 1.20` 都已经让 `track_error_km` 和 `mean_track_diff_vs_official_km` 两项同时过线
- intensity 仍然略差于 baseline，因此这一步不能被写成“forecast 全面通过”；但当前主 blocker 已经不再是 track dual gate 本身

因此当前主线再次更新为：

- 把 `slot_correction v1 + baseline intensity + scaled track calibration` 视为新的正式候选 mainline
- 不再继续泛化 sweep 结论本身，而是将 `scale = 1.20` 接入正式 compare 入口
- 下一步启动一条新的 formal gate，验证 `predicted_slot_locked_track_plus_baseline_intensity_scale_1p20_v1`

### 1.9 Status Reset: 2026-04-19 (After Intensity Integration Gate v1)

`slot_correction v1 + baseline intensity + scale 1.20` 的 formal gate 已完成，结论是 positive：

- run root: `runs/phase2_slot_correction_intensity_gate_v1_20260419_104510`
- predicted scaled hybrid `reward_mean = 0.3472`
- predicted scaled hybrid `track_error_km = 153.33`
- predicted scaled hybrid `mean_track_diff_vs_official_km = 132.48`
- predicted scaled hybrid `intensity_error_kt = 13.21`
- predicted scaled hybrid `mean_intensity_diff_vs_official_kt = 10.62`

这轮结果说明：

- 当前正式 mainline 已经不只是 sweep 候选，而是 sample-200 formal gate 真正过线
- 按当前 Phase 2B 标准，`track_error_km` 和 `mean_track_diff_vs_official_km` 两项都已通过 25% oracle-gain recovery line
- 但第二项的安全边际还不算很厚，同时强度仍略差于 baseline，因此当前最合理的下一步不是继续 redesign，而是先跑更稳的 confirmatory run

因此当前主线再次更新为：

- 把 `slot_correction v1 + baseline intensity + scale 1.20` 冻结为当前最优 predicted integration mainline
- Phase 2B 在 sample-200 formal gate 上可视为已通过
- 下一步优先做更大 held-out 的 confirmatory run，确认这不是抽样偶然，并检查剩余 intensity 小幅尾差

### 1.10 Status Reset: 2026-04-19 (After Intensity Integration Gate v1 Confirmatory Run)

`slot_correction v1 + baseline intensity + scale 1.20` 的更大 held-out confirmatory run 也已经完成，结论仍然是 positive：

- run root: `runs/phase2_slot_correction_intensity_gate_v1_confirm_20260419_112630`
- evaluated sample count: `1454`
- predicted scaled hybrid `reward_mean = 0.3624`
- predicted scaled hybrid `track_error_km = 153.72`
- predicted scaled hybrid `mean_track_diff_vs_official_km = 133.53`
- predicted scaled hybrid `intensity_error_kt = 11.82`
- predicted scaled hybrid `mean_intensity_diff_vs_official_kt = 9.55`

这轮结果说明：

- `slot_correction v1 + baseline intensity + scale 1.20` 不只是 sample-200 偶然过线，而是在更大 held-out 上也守住了双主轨迹指标
- 按当前 Phase 2B 标准，`track_error_km` 与 `mean_track_diff_vs_official_km` 两项都已正式通过
- 强度仍然略差于 baseline，但差距已经很小，当前不再值得把主线时间继续耗在这类尾差打磨上

因此当前阶段正式更新为：

- `slot_correction v1 + baseline intensity + scale 1.20` 视为当前二阶段已经通过的主线
- Phase 2B 可以视为正式完成，不再继续做同类 `slot_correction` 微调或 calibration 小修
- 下一步应从“把窄接口训到过线”切到“在已通过主线上恢复更宽的 diagnostic 信息”

### 1.11 Status Reset: 2026-04-19 (After Track-Core Phase 2C Probe v1)

`diagnostic_track_core_only v1` 的第一轮 Phase 2C 扩展训练与评测已经完成，但结论是 negative：

- run root: `runs/phase2_diagnostic_track_core_v1_20260419_153108`
- best checkpoint: `checkpoint-120`
- standalone `json_parse_rate = 1.0000`
- standalone `joint_exact = 0.0350`
- standalone `mean_field_acc = 0.3362`
- standalone `mean_macro_f1 = 0.2823`
- oracle forecast `track_error_km = 186.48`
- oracle forecast `mean_track_diff_vs_official_km = 168.98`
- predicted forecast `track_error_km = 183.64`
- predicted forecast `mean_track_diff_vs_official_km = 166.98`

这轮结果说明：

- `track_core` 作为更宽 schema 的第一条扩展探针，训练本身可以跑通，输出 contract 也稳定
- 但它没有把 richer diagnostics 变成更好的 forecast track gain；oracle 自己都没改善双主指标，因此这不是“训练不够”，而是接口扩展方向当前不对
- predicted 虽然在 `track_error_km` 上比 baseline 小幅好一点，但 `mean_track_diff_vs_official_km` 反而更差，因此不能算通过

因此当前阶段进一步更新为：

- `diagnostic_track_core_only v1` 作为 Phase 2C 的第一条扩展 probe，当前判定为 negative
- 当前已通过的主线仍然只有 `slot_correction v1 + baseline intensity + scale 1.20`
- Phase 2C 不能简单沿“generic richer diagnostic injection”继续铺开；下一步需要重新定义更窄、更可执行的扩展接口，而不是直接继续做 `track_core v2`

### 1.12 Status Update: 2026-04-20 (After Slot-Turn Correction Phase 2C Probe v0 Confirmatory Compare)

基于 `track_core v1` 的 negative 结论，Phase 2C 的下一条 probe 改成了更窄、且仍可执行的：

- `diagnostic_slot_turn_correction_only`

这轮训练、sample-200 formal compare、以及更大 held-out confirmatory compare 都已完成：

- run root: `runs/phase2_diagnostic_slot_turn_correction_v0_20260419_180019`
- confirm run root: `runs/phase2_diagnostic_slot_turn_correction_v0_confirm_20260419_220736`
- best checkpoint: `checkpoint-160`
- confirm standalone `json_parse_rate = 0.0825`
- confirm standalone `joint_exact = 0.0056`
- confirm standalone `mean_field_acc = 0.3899`
- confirm standalone `mean_macro_f1 = 0.2646`
- confirm standalone `turning_signal exact_accuracy = 0.1159`
- confirm standalone `turning_signal macro_f1 = 0.0805`
- confirm predicted forecast `track_error_km = 157.28`
- confirm predicted forecast `mean_track_diff_vs_official_km = 138.02`
- confirm predicted forecast + baseline intensity `track_error_km = 149.94`
- confirm predicted forecast + baseline intensity `mean_track_diff_vs_official_km = 129.32`
- confirm predicted forecast + baseline intensity `reward_mean = 0.3672`
- relative to current confirmed mainline, confirm predicted `slot_turn + baseline intensity` improves:
  - `reward_mean` by `+0.0048`
  - `track_error_km` by `-3.78 km`
  - `mean_track_diff_vs_official_km` by `-4.21 km`
  - intensity metrics remain unchanged because both variants still use baseline intensity integration

这轮结果说明：

- 更窄的 `slot + turning_signal` executable interface，不只是 `sample_count = 200 / seed = 3407` 上的偶然正向信号，而是在更大 held-out confirmatory compare 上也继续改善了 forecast-side track 指标
- 在当前所有已完成 confirmatory compare 的 predicted 分支里，`slot_turn + baseline intensity` 现在给出了最好的 track 结果
- 但 standalone diagnostic adapter 本身仍然没有守住 output contract；`turning_signal` 的预测分布依然高度塌缩到 `<null>/steady`
- 因此这轮仍然不能直接取代 `slot_correction v1 + baseline intensity + scale 1.20` 这条稳定主线；它更准确的定位是：**confirmed forecast-side winner, standalone contract negative**

因此当前阶段再更新为：

- `diagnostic_slot_correction_only v1 + baseline intensity + scale 1.20` 仍然是当前 stability-first confirmed mainline
- `diagnostic_slot_turn_correction_only v0` 已经升级为当前最强的 Phase 2C executable extension branch，但还没有升级为默认主线
- 下一步最合理的选择不再是重跑同一类 confirmatory compare，也不是恢复更宽 schema；而是启动 `slot_turn_correction v1` 的定向 contract-repair 分支：
  - 继续固定当前 `slot-turn` renderer / calibration / compare gate
  - 从当前 `slot_turn_correction v0 final_adapter` 继续做局部修复，而不是重新回到 generic redesign
  - 优先解决 `turning_signal` collapse、JSON contract 和 parseability
  - 只有在 forecast-side gain 不回退的前提下，才考虑把它升级为新的默认主线

### 1.13 Phase 2C Closure Definition

当前 `Phase 2C` 的收口目标不是继续做更宽 schema，也不是继续把 expert gap 直接往下打，而是把已经确认有效的 **narrow executable extension** 修成可以稳定接管主线的接口。

对当前 `diagnostic_slot_turn_correction_only` 而言，收口标准需要同时满足两条：

1. `forecast-side gain` 不能回退  
   相对当前 confirmed mainline `slot_correction v1 + baseline intensity + scale 1.20`，至少要继续守住：
   - 更低的 `track_error_km`
   - 更低的 `mean_track_diff_vs_official_km`
   - 不恶化现有 intensity integration
2. `standalone contract` 不再明显失效  
   当前最直接的 blocker 不是 renderer 本身，而是：
   - `json_parse_rate` 过低
   - `joint_exact` 过低
   - `turning_signal` 高度塌缩到 `<null>/steady`
   - 过度依赖 `line_kv` fallback 才能被 forecast integration 使用

因此 `Phase 2C` 当前最合理的阶段目标应被固定为：

- 保持 `slot-turn` 这条 executable extension 的已确认 track gain
- 将它从“forecast-side can help”修到“raw diagnostic head 本身也能稳定输出可执行 contract”
- 只有在这两点同时成立后，才决定是否替换当前 stability-first confirmed mainline

### 1.14 Status Update: 2026-04-20 (After Slot-Turn Correction v1 Full Confirm)

`slot_turn_correction v1` 的训练、sample-200 formal compare、以及更大 held-out full confirm 现已全部完成：

- train run root: `runs/phase2_diagnostic_slot_turn_correction_v1_20260420_095215`
- confirm run root: `runs/phase2_diagnostic_slot_turn_correction_v1_confirm_20260420_141813`
- best training checkpoint: `checkpoint-40`
- confirm standalone `json_parse_rate = 1.0000`
- confirm standalone `joint_exact = 0.0236`
- confirm standalone `mean_field_acc = 0.4522`
- confirm standalone `mean_macro_f1 = 0.3246`
- confirm standalone `turning_signal exact_accuracy = 0.8246`
- confirm standalone `turning_signal macro_f1 = 0.8309`
- confirm predicted rendered `track_error_km = 159.59`
- confirm predicted rendered `mean_track_diff_vs_official_km = 140.12`
- confirm predicted rendered + baseline intensity `track_error_km = 151.96`
- confirm predicted rendered + baseline intensity `mean_track_diff_vs_official_km = 131.04`
- confirm predicted rendered + baseline intensity `reward_mean = 0.3665`

相对 `slot_turn_correction v0 confirm`，`v1` 的含义是：

- standalone contract 从 failure 修成了 stable executable interface：`json_parse_rate 0.0825 -> 1.0000`
- `turning_signal` 从高度塌缩修成了接近真实分布：`exact_accuracy 0.1159 -> 0.8246`
- forecast-side 只发生了很小的数值回撤：
  - `reward_mean -0.0007`
  - `track_error_km +2.02`
  - `mean_track_diff_vs_official_km +1.72`

相对之前的 confirmed mainline `slot_correction v1 + baseline intensity + scale 1.20`，`slot_turn_correction v1 + baseline intensity` 在更大 held-out 上仍然继续小幅改善：

- `reward_mean +0.0041`
- `track_error_km -1.76`
- `mean_track_diff_vs_official_km -2.49`
- intensity 指标维持不变，因为两条路径仍共享 baseline intensity integration

这轮 full confirm 的结论因此很明确：

- `Phase 2C Closure Definition` 现在已经满足
- `diagnostic_slot_turn_correction_only v1 + baseline intensity + scale 1.20` 升级为新的当前 confirmed mainline / recommended variant
- `diagnostic_slot_correction_only v1 + baseline intensity + scale 1.20` 转为 stability / ablation control，不再是默认推荐主线
- `Phase 2C` 可以正式视为完成；后续工作不再以“修 contract / 修 turning”作为阶段 blocker

同时，这轮结果也把剩余问题收敛得更清楚：

- 去掉 `turning_signal` 后，12 个 slot bucket 字段的 `bucket-only mean macro-F1` 只从 `0.2800` 小幅升到 `0.2824`
- 最弱字段仍然明显塌向 `near_consensus`，例如：
  - `slot_1_lat_bias_vs_consensus_bucket macro_f1 = 0.1414`
  - `slot_1_lon_bias_vs_consensus_bucket macro_f1 = 0.1846`
  - `slot_6_lat_bias_vs_consensus_bucket macro_f1 = 0.2171`

因此下一阶段的优化目标不再是 output contract，也不再是 `turning_signal` 本身，而是：**只针对 12 个 slot bucket 的语义分辨率做定向提升。**

### 1.15 Next Phase: Slot-Bucket Semantic Resolution

既然 `Phase 2C` 已经完成，下一阶段最合理的动作不是继续扩大 schema，也不是回到 generic diagnostic injection，而是固定当前已验证通过的 executable interface，只改与 slot bucket 语义分辨率直接相关的部分。

下一阶段建议固定不动的部分：

- 保持 `diagnostic_slot_turn_correction_only` 的 13 字段 contract 不变
- 保持单行、定序、`turning_signal` 优先的 JSON 输出 contract 不变
- 保持当前 `slot-locked deterministic renderer` / calibration / compare gate 不变
- 保持 baseline intensity integration 不变
- 保持 staged init 起点为 `slot_turn_correction v1 full confirm final_adapter`

下一阶段只改动以下三类内容：

1. `slot bucket` 重采样从“兼顾 turning”改成“只盯 bucket”
   - `turning_signal` 不再作为主优化对象，只要求不回退
   - 12 个 `slot_i lat/lon bias bucket` 变成 resampling 主目标
   - 明确提高 `non-near_consensus`、方向性 `small/large` 桶位的最小采样倍数
   - 对当前最弱字段优先加权：`slot_1 lat/lon`、`slot_6 lat/lon`

2. prompt 结构只做 `slot-local semantic alignment`
   - 不增加任何新证据源
   - 压缩通用叙述，减少与 bucket 判别无关的 token 占用
   - 将可见 evidence 按 `slot_1 -> slot_6` 的顺序重排，使每个 slot 的 lat/lon 语义 cue 更靠近对应输出字段

3. 评估口径新增 `bucket-only` 指标
   - 除总 `mean_macro_f1` 外，单独报告去掉 `turning_signal` 后的 12 字段 `bucket-only mean macro-F1`
   - 单独跟踪最弱 4 个 bucket 字段的 macro-F1
   - 这样可以避免 `turning_signal` 的提升掩盖 bucket 语义仍未解决这一事实

下一阶段建议的 go / no-go 目标：

- `bucket-only mean macro-F1` 相对当前 `0.2824` 至少提升到 `>= 0.31`
- 最弱 4 个 bucket 字段的 macro-F1 至少各提升 `0.03`
- `turning_signal macro-F1` 不允许相对当前 `0.8309` 明显回退
- full confirm forecast 侧不得低于当前 mainline；理想情况是再拿到 `1-3 km` 量级的 track 改善

也就是说，从阶段管理角度看：

- `Phase 2C` 现在已经完成，不再需要额外 blocker-clearing experiment
- 真正剩下的，是一个新的、更窄的阶段目标：把当前 executable interface 中最弱的 `slot bucket semantics` 提升起来
- 因此现在可以正式进入下一阶段

### 1.16 Status Update: 2026-04-21 (After Slot-Bucket Semantic Resolution v2 Sample-200)

第一轮 `slot bucket semantic resolution` probe 现已完成：

- run root: `runs/phase2_diagnostic_slot_turn_correction_v2_20260420_211349`
- best training checkpoint: `checkpoint-40`
- standalone `json_parse_rate = 1.0000`
- standalone `joint_exact = 0.0100`
- standalone `mean_field_acc = 0.4558`
- standalone `mean_macro_f1 = 0.3850`
- sample-200 predicted rendered `track_error_km = 158.60`
- sample-200 predicted rendered `mean_track_diff_vs_official_km = 140.39`
- sample-200 predicted rendered + baseline intensity `track_error_km = 149.56`
- sample-200 predicted rendered + baseline intensity `mean_track_diff_vs_official_km = 131.65`
- sample-200 predicted rendered + baseline intensity `reward_mean = 0.3516`

这轮 `v2` 的结构性结论是：

- standalone bucket semantics 确实提升了
  - `bucket-only mean exact_accuracy: 0.4192 -> 0.4271`
  - `bucket-only mean macro-F1: 0.3057 -> 0.3488`
- 提升几乎全部集中在 `lon bucket`
  - `slot_1_lon macro-F1: 0.1867 -> 0.3048`
  - `slot_2_lon macro-F1: 0.3153 -> 0.4295`
  - `slot_3_lon macro-F1: 0.3180 -> 0.4046`
  - `slot_4_lon macro-F1: 0.3259 -> 0.4046`
  - `slot_5_lon macro-F1: 0.3165 -> 0.3941`
- `lat bucket` 基本没有被真正带起来
  - `slot_2_lat` 到 `slot_6_lat` 相对 `v1 sample-200` 基本不动
  - 当前最弱字段仍然是 `slot_1_lat`、`slot_6_lat`、`slot_6_lon`
- `turning_signal` 没有退化，但也不是新的瓶颈

forecast integration 侧，这轮不能视为主线升级：

- 相对 `slot_turn_correction v1 sample-200`
  - `reward_mean +0.0014`
  - `track_error_km +1.98`
  - `mean_track_diff_vs_official_km +3.34`
- 相对当前 `v1 full confirm mainline`
  - `track_error_km` 仍然略低，但只是 sample-200 抽样波动量级
  - `mean_track_diff_vs_official_km` 没有进一步改善
  - 不能据此替换当前 confirmed mainline

因此这轮 `v2` 的正式判定应为：

- **diagnostic-side positive**：说明 `slot-local anti-collapse` 和 targeted resampling 方向正确
- **forecast-side neutral to slightly negative**：说明 gain 还没有传导为主指标改善
- **mainline decision unchanged**：当前 confirmed mainline 仍然保持 `diagnostic_slot_turn_correction_only v1 + baseline intensity + scale 1.20`

### 1.17 Next Phase: Lat-Bucket Semantic Resolution

既然 `v2` 已经证明当前剩余瓶颈主要集中在 `lat bucket semantics`，下一轮就不再泛化地做 “all bucket resolution”，而是只对 `north/south` 判别做定向改动。

下一轮固定不动：

- 保持 `diagnostic_slot_turn_correction_only` 的 13 字段 contract 不变
- 保持 `slot-locked renderer` / calibration / baseline intensity integration 不变
- 保持 `turning_signal` 只作为守住不回退的辅助字段，不再作为主优化目标

下一轮只改三件事：

1. `lat-only resampling`
   - 只对 6 个 `slot_i_lat_bias_vs_consensus_bucket` 做重采样
   - 不再继续放大 `lon bucket`
   - 对 `slot_1_lat` / `slot_6_lat` 给更强的 directional bucket 最小倍数

2. `lat/lon explicit decoupling` prompt
   - 在 system prompt 中明确写出：`lat` 与 `lon` 必须分开判别
   - 明确声明：`east/west` 贴近 consensus 不能作为 `near_consensus lat` 的证据
   - 在 slot correction cues 中把 `north/south displacement` 的判别规则写得更硬

3. `v2 -> v3 staged init`
   - 不回退到 `v1`
   - 直接从 `slot_turn_correction v2 final_adapter` 继续
   - 目的是保留已经拿到的 `lon semantics` 增益，只补 `lat` 这半边

下一轮 go / no-go 目标收窄为：

- `lat-bucket mean macro-F1` 明显提升
- `slot_1_lat` 与 `slot_6_lat` macro-F1 至少继续抬升
- `turning_signal` 不明显回退
- forecast integration 不能劣于当前 `v1 full confirm mainline`

### 1.18 Status Update: 2026-04-21 (After Lat-Bucket Semantic Resolution v3 Sample-200)

第二轮 `slot bucket semantics` probe 现已完成：

- run root: `runs/phase2_diagnostic_slot_turn_correction_v3_20260421_010206`
- best training checkpoint: `checkpoint-100`
- standalone `json_parse_rate = 1.0000`
- standalone `joint_exact = 0.0050`
- standalone `mean_field_acc = 0.4565`
- standalone `mean_macro_f1 = 0.4278`
- sample-200 predicted rendered `track_error_km = 153.13`
- sample-200 predicted rendered `mean_track_diff_vs_official_km = 135.02`
- sample-200 predicted rendered + baseline intensity `track_error_km = 143.14`
- sample-200 predicted rendered + baseline intensity `mean_track_diff_vs_official_km = 125.36`
- sample-200 predicted rendered + baseline intensity `reward_mean = 0.3561`

这轮 `v3` 的结果和原始假设不完全一致：

- forecast-side 是目前最强的 sample-200 结果
  - 相对 `v2 sample-200`
    - `reward_mean +0.0045`
    - `track_error_km -6.43`
    - `mean_track_diff_vs_official_km -6.29`
  - 相对 `v1 sample-200`
    - `reward_mean +0.0059`
    - `track_error_km -4.45`
    - `mean_track_diff_vs_official_km -2.95`
- standalone 总体也继续提升
  - `mean_macro_f1: 0.3850 -> 0.4278`
  - `bucket-only mean macro-F1: 0.3488 -> 0.3961`

但真正需要强调的是：

- 这轮并没有把 `lat bucket semantics` 修起来
  - `lat mean macro-F1: 0.3289 -> 0.3262`
  - `slot_2_lat` 到 `slot_6_lat` 基本不动
  - `slot_1_lat macro-F1` 甚至从 `0.2040` 回到 `0.1877`
- 增益仍然几乎全部来自 `lon bucket`
  - `lon mean macro-F1: 0.3687 -> 0.4660`
  - `slot_2_lon`, `slot_3_lon`, `slot_4_lon`, `slot_5_lon` 均继续明显抬升

因此这轮 `v3` 的正式判定应为：

- **forecast-side positive**
- **standalone aggregate positive**
- **hypothesis mismatch**：`lat-only resampling` 并没有真正修复 `lat bucket`，但 `lat/lon decoupling` prompt 继续释放了 `lon bucket` 的可分性

从主线管理角度看，当前最合理的动作不是再开新 probe，而是：

- 先把 `v3` 作为 **sample-200 provisional winner**
- 立即启动 `full confirm`
- 只有当更大 held-out 也继续优于当前 `v1 full confirm mainline` 时，才允许升级主线

### 1.19 Status Update: 2026-04-21 (After Slot-Turn Correction v3 Full Confirm)

`slot_turn_correction v3` 的更大 held-out full confirm 现已完成，结论是 positive：

- confirm run root: `runs/phase2_diagnostic_slot_turn_correction_v3_confirm_20260421_090002`
- confirm evaluated sample count: `1454`
- confirm standalone `json_parse_rate = 1.0000`
- confirm standalone `joint_exact = 0.0229`
- confirm standalone `mean_field_acc = 0.4567`
- confirm standalone `mean_macro_f1 = 0.3418`
- confirm standalone `bucket-only mean macro-F1 = 0.3015`
- confirm standalone `turning_signal exact_accuracy = 0.8221`
- confirm standalone `turning_signal macro_f1 = 0.8256`
- confirm predicted rendered `track_error_km = 154.14`
- confirm predicted rendered `mean_track_diff_vs_official_km = 135.26`
- confirm predicted rendered + baseline intensity `track_error_km = 145.58`
- confirm predicted rendered + baseline intensity `mean_track_diff_vs_official_km = 125.26`
- confirm predicted rendered + baseline intensity `reward_mean = 0.3686`
- confirm predicted rendered + baseline intensity `intensity_error_kt = 11.82`

相对当前 `v1 full confirm mainline`，`slot_turn_correction v3 + baseline intensity` 在更大 held-out 上继续改善：

- `reward_mean +0.0021`
- `track_error_km -6.38`
- `mean_track_diff_vs_official_km -5.78`
- intensity 指标维持不变，因为两条路径仍共享 baseline intensity integration

相对 baseline `baseline_forecast_sft_v2`，`v3 full confirm` 也给出了更明确的双主指标改善：

- `reward_mean +0.1169`
- `track_error_km -36.33`
- `mean_track_diff_vs_official_km -38.74`
- `coverage +0.1978`

同时，这轮结果也把增益来源讲得更清楚：

- standalone aggregate 仍然小幅上升：`mean_macro_f1 0.3246 -> 0.3418`
- `turning_signal` 基本持平：`macro_f1 0.8309 -> 0.8256`
- `bucket-only mean macro-F1` 只从 `0.2824 -> 0.3015`
- 其中主要提升仍来自 `lon bucket`
  - `lat mean macro-F1: 0.2703 -> 0.2743`
  - `lon mean macro-F1: 0.2946 -> 0.3286`

这轮 full confirm 的正式结论因此是：

- `diagnostic_slot_turn_correction_only v3 + baseline intensity + scale 1.20` 升级为新的当前 confirmed mainline / recommended variant
- `diagnostic_slot_turn_correction_only v1 + baseline intensity + scale 1.20` 转为 stability control / ablation control
- `v3` 的 gain 已不再只是 sample-200 抽样信号，而是 held-out confirm 口径下的真实改善
- 但当前剩余短板依然是 bucket semantics，尤其不是 `turning_signal` 或 output contract

### 1.20 RL Readiness Assessment

基于这轮 `v3 full confirm`，当前阶段已经不再卡在 schema、contract、或 “predicted 有没有 gain” 这种 blocker 上；但这还不等于现在就应该直接开始 RL。

当前判断应明确写成：

- **可以开始准备 RL**
- **但在真正启动 RL 训练前，先冻结 objective 与 guardrail**

原因有三点：

1. 当前 gain 已经 confirmed，但 diagnostic 语义还没有收敛到“值得用 RL 放大”的程度
   - `joint_exact` 仍只有 `0.0229`
   - `bucket-only mean macro-F1` 仍只有 `0.3015`
   - 最弱字段仍明显塌向 `near_consensus`
   - 这意味着如果现在直接上 RL，最容易放大的不是“真正更懂轨迹修正”，而是当前已经有效但仍偏单侧的 shortcut

2. `v3` 的正向 gain 目前主要来自 `lon bucket`，而不是原假设中的 `lat bucket repair`
   - 这不是坏事，但说明当前改进机制还没有被充分解释
   - 在这种状态下直接上 RL，风险是 reward 会继续强化已有偏向，而不是补齐最弱的 lat 语义

3. 当前可用的 reward 仍然应该严格依赖 forecast-side dual-track gate，而不是 diagnostic-side proxy
   - `reward_mean`、`coverage`、standalone label 指标都不能替代 `track_error_km` 与 `mean_track_diff_vs_official_km`
   - 因此如果要做 RL，正确对象应是 **integrated forecast behavior**，而不是单独对 diagnostic head 做 proxy-RL

因此，距离“可以安全启动 RL”还差的不是大规模 redesign，而是两项更窄的准备工作：

1. 先完成一轮 **post-confirm semantic audit**
   - 这一步已经完成
   - 当前结论应固定为：`v3` 的 held-out forecast gain 主要来自 `lon semantics`
   - `lat bucket` 仍然是最弱语义轴，但当前证据不支持把新的 `lat-only targeted SFT cleanup` 作为 RL 前的必经 blocker

2. 先写清楚 **RL objective 与 guardrail**
   - 优化目标必须直接绑定 `track_error_km` 与 `mean_track_diff_vs_official_km`
   - `intensity_error_kt`、`strict_parseable_rate`、`slot_time_match_rate_vs_official` 应作为 guardrail，而不是主 reward
   - 训练单元应该以 integrated predicted variant 为对象，而不是把 raw diagnostic JSON 当最终优化终点

换句话说，当前距离 RL 不是“还差一个大阶段”，而是“还差一次语义收口和一次 reward 设计收口”。

- 如果只问工程 readiness：**已经可以开始准备 RL**
- 如果问研究判定是否已经该开跑：**可以进入 RL 设计与准备阶段；真正开跑前先把 reward/guardrail 定死更稳妥**

### 1.21 Post-Confirm Semantic Audit Verdict

`v3 full confirm` 的 post-confirm semantic audit 现已完成：

- audit note: `runs/phase2_diagnostic_slot_turn_correction_v3_confirm_20260421_090002/post_confirm_semantic_audit.md`
- compared systems:
  - `diagnostic_slot_turn_correction_only v1` vs `v3`
  - `predicted_slot_turn_track_plus_baseline_intensity_scale_1p20_v1` vs `v3`
  - shared held-out intersection: `1454` samples

审计结论可以固定为：

- `v3` 的 confirmed forecast gain **主要来自 `lon semantics`**
- `lat bucket` 仍然是最弱语义轴，但它不是当前 gain 的主要解释变量
- 因此当前不需要把新的 `lat-only targeted SFT cleanup` 作为 RL 前的 mandatory blocker

最关键的样本级证据是：

- `lat_delta` 与 `track_improve` 的相关性约为 `-0.0578`
- `lon_delta` 与 `track_improve` 的相关性约为 `0.5149`
- best-200 winner slice 的平均 `lon_delta = +0.94`，而平均 `lat_delta = +0.02`
- worst-200 loser slice 的平均 `lon_delta = -0.57`，而平均 `lat_delta = +0.025`

因此当前的正式管理结论应写成：

- **accept**：当前 `v3` gain 主要由 `lon semantics` 驱动
- **defer**：`lat-only targeted SFT cleanup` 降级为后续可选 probe，而不是 RL 前必做门槛
- **proceed**：下一步可以直接进入 RL objective / guardrail design

## 2. Current Readiness

当前仓库已经具备二阶段的最小数据底座，不需要回到 raw NOAA / ERA5 / ATCF / HRES 重做一遍。

- canonical 底座仍然是 `canonical_v2`
- shared-evidence 输入已经通过 `forecast_only` 视图修正
- `diagnostic_only` 视图已经可导出
- `diagnostic_track_turn_only` 视图已经可导出
- `diagnostic_track_core_only` 视图已经可导出
- `diagnostic_track_correction_only` 视图已经可导出
- `diagnostic_slot_correction_only` 视图已经可导出
- `diagnostic_slot_turn_correction_only` 视图已经可导出
- `diagnostic_core_only` 视图已经可导出
- 根目录兼容文件已经存在：
  - `sft_diagnostic_train.jsonl`
  - `sft_diagnostic_val.jsonl`
  - `sft_diagnostic_test.jsonl`
  - `sft_diagnostic_track_turn_train.jsonl`
  - `sft_diagnostic_track_turn_val.jsonl`
  - `sft_diagnostic_track_turn_test.jsonl`
  - `sft_diagnostic_track_core_train.jsonl`
  - `sft_diagnostic_track_core_val.jsonl`
  - `sft_diagnostic_track_core_test.jsonl`
  - `sft_diagnostic_track_correction_train.jsonl`
  - `sft_diagnostic_track_correction_val.jsonl`
  - `sft_diagnostic_track_correction_test.jsonl`
  - `sft_diagnostic_slot_correction_train.jsonl`
  - `sft_diagnostic_slot_correction_val.jsonl`
  - `sft_diagnostic_slot_correction_test.jsonl`
  - `sft_diagnostic_slot_turn_correction_train.jsonl`
  - `sft_diagnostic_slot_turn_correction_val.jsonl`
  - `sft_diagnostic_slot_turn_correction_test.jsonl`
  - `sft_diagnostic_core_train.jsonl`
  - `sft_diagnostic_core_val.jsonl`
  - `sft_diagnostic_core_test.jsonl`

当前新数据根目录中的规模如下：

| view | train | val | test |
| --- | ---: | ---: | ---: |
| `forecast_only` | 3959 | 1354 | 1613 |
| `diagnostic_only` | 3959 | 1354 | 1613 |
| `diagnostic_track_turn_only` | 3959 | 1354 | 1613 |
| `diagnostic_track_core_only` | 3959 | 1354 | 1613 |
| `diagnostic_track_correction_only` | 3959 | 1354 | 1613 |
| `diagnostic_slot_correction_only` | 3959 | 1354 | 1613 |
| `diagnostic_slot_turn_correction_only` | 3959 | 1354 | 1613 |
| `diagnostic_core_only` | 3959 | 1354 | 1613 |
| `reasoning_only` | 385 | 138 | 148 |

这意味着二阶段不受样本规模限制；真正的限制是标签质量和 schema 设计。

当前 `diagnostic_only` 的导出方式已经明确：

- 输入：与 `forecast_only` 共享同一证据底座；当前 diagnostic 视图会额外追加基于 ATCF/HRES guidance 的 `ridge/trough competition cues` 与 `turning signal cues` 摘要，但不引入任何新信息
- 输出：固定 key 的 JSON 对象
- 导出脚本：`scripts/export_views_v2.py`
- 字段来源：`scripts/dataset_v2.py` 中的 `heuristic_v2`

因此，二阶段现在不需要先改训练主干；现有 SFT pipeline 已经可以直接训练 `messages` 格式的 diagnostic 数据。

### 2.1 Phase 1 Formal Baseline Freeze

旧数据口径上的 `step30` / `step70` 已经失效，不能再作为二阶段 baseline、历史最优或 go / no-go 依据。

当前 Phase 2 只承认建立在：

- `data/training_rebuilt_v2_20260414_guidancefix`

之上的正式 Phase 1 baseline。当前冻结的 forecast-side adapter 如下：

- `baseline_forecast_sft_v2`
  - checkpoint: `runs/phase1_baseline_v2_formal_20260415_013403/sft/final_adapter`
- `baseline_forecast_grpo_physical_v2`
  - checkpoint: `runs/phase1_baseline_v2_formal_20260415_013403/grpo/adapter_step-000050`
- `baseline_forecast_grpo_reward_v2`
  - checkpoint: `runs/phase1_baseline_v2_formal_20260415_013403/grpo/adapter_reward-0.7544_step-000003`

基于当前正式 run 的 `sample_count = 200` checkpoint compare，结果如下：

| system | reward | coverage | track error (km) | intensity error (kt) |
| --- | ---: | ---: | ---: | ---: |
| `baseline_forecast_sft_v2` | `0.2943` | `0.9467` | `199.8` | `13.65` |
| `baseline_forecast_grpo_physical_v2` | `0.2886` | `0.9442` | `200.4` | `13.45` |
| `baseline_forecast_grpo_reward_v2` | `0.2909` | `0.9417` | `202.0` | `13.54` |

当前可以确认：

1. 这轮正式 Phase 1 中，默认 forecast-side 参考 adapter 应该使用 `baseline_forecast_sft_v2`。
2. `baseline_forecast_grpo_physical_v2` 作为 physical-error-led GRPO baseline 保留，用于观察二阶段是否缩小物理误差差距。
3. `baseline_forecast_grpo_reward_v2` 作为 reward-led GRPO baseline 保留，用于 reward 侧对照。
4. `adapter_reward-0.4389_step-000001` 在当前 200 样本 compare 上与 `baseline_forecast_sft_v2` 逐样本完全等价，因此不再单列为独立 baseline。

因此，二阶段策略必须建立在新的正式 baseline 冻结上：

- 不是沿用旧 `step30` / `step70`
- 不是泛泛地做 diagnostics
- 而是围绕“如何缩小新 baseline 与专家之间的 gap”来做 diagnostics

尤其需要注意：

- 当前最大的收益空间仍然在 `track / turning / uncertainty handling`
- 因此二阶段不应从 full diagnostic 全量字段直接启动，而应采用 `track-first rollout`

### 2.2 Phase 2 Bootstrap Artifacts

当前仓库中已经具备以下二阶段 bootstrap 物料：

- 数据导出：
  - `views/diagnostic_only`
  - `views/diagnostic_track_turn_only`
  - `views/diagnostic_track_correction_only`
  - `views/diagnostic_slot_correction_only`
  - `views/diagnostic_slot_turn_correction_only`
  - `views/diagnostic_track_core_only`
  - `views/diagnostic_core_only`
- 训练配置：
  - `configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_track_turn_stage_v0.yaml`
  - `configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_track_turn_stage_v0_1.yaml`
  - `configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_slot_correction_stage_v0.yaml`
  - `configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_slot_correction_stage_v1.yaml`
  - `configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_slot_correction_stage_v2.yaml`
  - `configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_slot_turn_correction_stage_v0.yaml`
  - `configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_track_core_v0.yaml`
  - `configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_track_core_v1.yaml`
  - `configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_core_v0.yaml`
- 运行入口：
  - `scripts/run_phase2_diagnostic_track_turn_formal.sh`
  - `scripts/run_phase2_diagnostic_track_turn_v0_1_formal.sh`
  - `scripts/run_phase2_diagnostic_track_turn_checkpoint_sweep_formal.sh`
  - `scripts/check_phase2_diagnostic_track_turn.py`
  - `scripts/run_phase2_diagnostic_track_correction_oracle_formal.sh`
  - `scripts/run_phase2_diagnostic_track_correction_oracle_background.sh`
  - `scripts/check_phase2_diagnostic_track_correction_oracle.py`
  - `scripts/run_phase2_diagnostic_slot_correction_oracle_formal.sh`
  - `scripts/run_phase2_diagnostic_slot_correction_oracle_background.sh`
  - `scripts/check_phase2_diagnostic_slot_correction_oracle.py`
  - `scripts/run_phase2_diagnostic_slot_correction_formal.sh`
  - `scripts/run_phase2_diagnostic_slot_correction_background.sh`
  - `scripts/run_phase2_diagnostic_slot_correction_v1_formal.sh`
  - `scripts/run_phase2_diagnostic_slot_correction_v1_background.sh`
  - `scripts/run_phase2_diagnostic_slot_correction_v2_formal.sh`
  - `scripts/run_phase2_diagnostic_slot_correction_v2_background.sh`
  - `scripts/run_phase2_diagnostic_slot_turn_correction_formal.sh`
  - `scripts/run_phase2_diagnostic_slot_turn_correction_background.sh`
  - `scripts/run_phase2_diagnostic_track_core_v1_formal.sh`
  - `scripts/run_phase2_diagnostic_track_core_v1_background.sh`
  - `scripts/check_phase2_diagnostic_slot_correction.py`
  - `scripts/check_phase2_diagnostic_slot_turn_correction.py`

当前已经完成并且必须回填到结论中的环节：

- standalone diagnostic eval summary
- oracle diagnostics -> forecast ablation
- predicted diagnostics -> forecast ablation

当前执行口径再明确一次：

- **Phase 2 objective**：`oracle-first environment-interface validation for track-gap closing`
- **Phase 2A**：先验证某个环境接口在 oracle 注入下是否真的改善 track 主指标
- **Phase 2B**：只有在 `Phase 2A` 通过后，才训练 predicted diagnostic adapter
- **Phase 2C**：只有在 `Phase 2A + 2B` 都通过后，才允许扩展到 `track_core` / `core` / `extended`

当前正式结果摘要：

| variant | standalone joint exact | standalone mean macro-F1 | forecast track error (km) | model-vs-official track diff (km) | slot-time match | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_forecast_sft_v2` | N/A | N/A | `186.30` | `167.29` | `0.7747` | reference |
| `oracle track_turn` | N/A | N/A | `192.17` | `175.22` | `0.7897` | slot-only gain, no track gain |
| `predicted track_turn v0` | `0.305` | `0.4867` | `189.85` | `173.04` | `0.7913` | no track gain |
| `predicted track_turn v0.1` | `0.330` | `0.4958` | `189.44` | `172.42` | `0.7913` | no track gain |
| `oracle track_inflection` | N/A | N/A | `190.12` | `175.82` | `0.7838` | worse track, stop |
| `oracle track_correction` | N/A | N/A | `184.57` | `169.26` | `0.7655` | almost flat track error, worse vs official |
| `oracle slot_correction` | N/A | N/A | `104.65` | `49.30` | `1.0000` | **first clear oracle pass** |
| `predicted slot_correction v0` | `0.000` | `0.0408` | `167.69` | `148.35` | `1.0000` | partial recovery; contract failure |
| `predicted slot_correction v1` | `0.010` | `0.3593` | `163.44` | `143.34` | `1.0000` | current best predicted variant; still below dual-track gate |
| `predicted slot_correction v2` | `0.015` | `0.3450` | `170.67` | `150.50` | `1.0000` | standalone up, forecast down; stop |
| `predicted slot_correction v1 + baseline intensity` | `0.010` | `0.3593` | `163.44` | `143.34` | `1.0000` | better integration baseline; remaining blocker is track calibration |
| `predicted slot_correction v1 + baseline intensity + scale 1.20` | `0.010` | `0.3593` | `153.72` | `133.53` | `1.0000` | former stability-first mainline; now retained as control |
| `predicted track_core v1` | `0.035` | `0.2823` | `183.64` | `166.98` | `0.7863` | Phase 2C first probe; negative |
| `predicted slot_turn_correction v0` | `0.006` | `0.2646` | `157.28` | `138.02` | `1.0000` | larger held-out confirm positive; forecast-side gain confirmed, contract still failed |
| `predicted slot_turn_correction v0 + baseline intensity` | `0.006` | `0.2646` | `149.94` | `129.32` | `1.0000` | forecast-side winner before contract repair; superseded by v1 |
| `predicted slot_turn_correction v1` | `0.024` | `0.3246` | `159.59` | `140.12` | `1.0000` | contract repaired; predicted gain retained |
| `predicted slot_turn_correction v1 + baseline intensity` | `0.024` | `0.3246` | `151.96` | `131.04` | `1.0000` | former confirmed mainline; now retained as stability control |
| `predicted slot_turn_correction v2` | `0.010` | `0.3850` | `158.60` | `140.39` | `1.0000` | standalone bucket semantics up, but forecast gain did not improve; keep as probe only |
| `predicted slot_turn_correction v2 + baseline intensity` | `0.010` | `0.3850` | `149.56` | `131.65` | `1.0000` | sample-200 close to mainline, but not enough to replace `v1 full confirm` |
| `predicted slot_turn_correction v3` | `0.005` | `0.4278` | `153.13` | `135.02` | `1.0000` | current best sample-200 forecast result; gain still not explained by `lat` repair |
| `predicted slot_turn_correction v3 + baseline intensity` | `0.023` | `0.3418` | `145.58` | `125.26` | `1.0000` | current confirmed mainline / recommended variant after held-out full confirm |

### 2.3 Phase 2 TODO Tracker

当前以 `training_rebuilt_v2_20260414_guidancefix` 为唯一合法数据底座。下面 checklist 只跟踪新数据口径下的二阶段工作，不再引用旧 `step30` / `step70`。规则如下：

- `[x]` = 已完成
- `[ ]` = 待完成

- [x] 冻结 Phase 2 唯一合法数据根目录：`data/training_rebuilt_v2_20260414_guidancefix`
- [x] 冻结默认 forecast-side adapter：`baseline_forecast_sft_v2`
- [x] 冻结 physical-error-led baseline：`baseline_forecast_grpo_physical_v2`
- [x] 冻结 reward-led baseline：`baseline_forecast_grpo_reward_v2`
- [x] 导出 `diagnostic_only` / `diagnostic_track_turn_only` / `diagnostic_track_core_only` / `diagnostic_core_only`
- [x] 写好 `diagnostic_track_turn_only` / `diagnostic_track_core_only` / `diagnostic_core_only` 的 SFT 配置
- [x] 写好 `scripts/eval_diagnostic_heldout.py`
- [x] 写好 `scripts/compare_diagnostic_models.py`
- [x] 写好 `scripts/build_forecast_prompt_overrides.py`
- [x] 写好 `scripts/compare_diagnostic_forecast_integration.py`
- [x] 为 SFT 补齐 staged init、label-level resampling、best checkpoint selection、early stopping
- [x] 将 `scripts/eval_strict_forecast_heldout.py` 扩展为支持 `prompt_overrides`、`forecast vs official` 指标、`expert_official` 汇总
- [x] 完成脚本烟测
  - [x] `python3 -m py_compile scripts/eval_diagnostic_heldout.py scripts/compare_diagnostic_models.py scripts/eval_strict_forecast_heldout.py scripts/build_forecast_prompt_overrides.py scripts/compare_diagnostic_forecast_integration.py`
  - [x] `scripts/compare_diagnostic_models.py` 的 `rule_echo` / `majority_label` 小样本 compare 已通过
  - [x] `scripts/build_forecast_prompt_overrides.py` 的 oracle 小样本落盘已通过
  - [x] `eval_strict_forecast_heldout.py` 的 expert-official 聚合路径已通过小样本 smoke
- [x] 训练 `diagnostic_adapter_track_turn` `v0`
- [x] 训练 `diagnostic_adapter_track_turn` `v0.1`
- [x] 跑 standalone `diagnostic_track_turn_only` held-out compare，并输出 `diagnostic_eval_summary`
- [x] 用 `baseline_forecast_sft_v2` 跑 oracle `track_turn diagnostics -> forecast` compare
- [x] 用同一个 forecast adapter 跑 predicted `track_turn diagnostics -> forecast` compare
- [x] 回填结论：当前 `track_turn` 接口对 forecast track accuracy 没有产生可用增益
- [x] 设计 `diagnostic_track_inflection_only` 并完成 formal oracle gate
- [x] 回填结论：`diagnostic_track_inflection_only v0` 仍未改善 forecast track 主指标
- [x] 基于 case-level failure audit，定义下一条 redesign mainline：`diagnostic_track_correction_only`
- [x] 审计 `48h/72h official - ATCF consensus` 偏差分布，并冻结一版 provisional correction bucket cut points
- [x] 实现 `diagnostic_track_correction_only` derivation / export / oracle prompt block
- [x] 在 prompt override 中加入 track-only guardrail：禁止改动 `Day/HHMMZ` 槽位与 intensity
- [x] 只对 `diagnostic_track_correction_only` 先跑 oracle compare；formal 结果为 negative
- [x] 基于该失败结论，定义新的 redesign mainline：`diagnostic_slot_correction_only`
- [x] 实现 `diagnostic_slot_correction_only` derivation / export / oracle compare 入口
- [x] 将 `diagnostic_slot_correction_only` 重导出到 shared dataset root
- [x] 只对 `diagnostic_slot_correction_only` 先跑 formal oracle compare；formal 结果为 positive
- [x] 回填结论：`diagnostic_slot_correction_only` 是当前唯一通过 oracle gate 的接口
- [x] 写好 `configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_slot_correction_stage_v0.yaml`
- [x] 写好 `scripts/check_phase2_diagnostic_slot_correction.py`
- [x] 写好 `scripts/run_phase2_diagnostic_slot_correction_formal.sh`
- [x] 写好 `scripts/run_phase2_diagnostic_slot_correction_background.sh`
- [x] 启动 `diagnostic_slot_correction_only` predicted training
  - run root: `runs/phase2_diagnostic_slot_correction_v0_20260418_161630`
- [x] 跑 predicted `slot_correction diagnostics -> forecast` gate
- [x] 回填结论：`slot_correction v0` 已恢复部分 oracle track gain，但 standalone contract failure 仍然严重
- [x] 审计 root cause：当前 `slot_correction` prompt budget 在 train/eval 两侧均发生截断
- [x] 为 `slot_correction` predicted diagnostics 补 line-based recovery gate
- [x] 为 `slot_correction` 视图补 compact prompt / compact target 导出
- [x] 写好 `configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_slot_correction_stage_v1.yaml`
- [x] 写好 `scripts/run_phase2_diagnostic_slot_correction_v1_formal.sh`
- [x] 写好 `scripts/run_phase2_diagnostic_slot_correction_v1_background.sh`
- [x] 启动 `diagnostic_slot_correction_only v1` predicted training
  - run root: `runs/phase2_diagnostic_slot_correction_v1_20260418_184506`
- [x] 跑 predicted `slot_correction v1 diagnostics -> forecast` gate
- [x] 回填结论：`slot_correction v1` 已修复大部分 output contract 问题，并继续逼近 oracle track 上限
- [x] 冻结新的 Phase 2B 判定标准：`track_error_km` 和 `mean_track_diff_vs_official_km` 两项都要过线
- [x] 回填当前阶段定位：已完成“找接口”，正在做 predicted recovery，尚未进入更宽 schema 扩展
- [x] 设计 `slot_correction v2` 的定向去塌缩训练方案
  - 关闭会放大 `near_consensus` 的通用 rarity resampling
  - 改为只对方向性 correction buckets 做 targeted min multipliers
  - staged init from `slot_correction v1 final_adapter`
- [x] 写好 `configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_slot_correction_stage_v2.yaml`
- [x] 写好 `scripts/run_phase2_diagnostic_slot_correction_v2_formal.sh`
- [x] 写好 `scripts/run_phase2_diagnostic_slot_correction_v2_background.sh`
- [x] 启动 `diagnostic_slot_correction_only v2` predicted training
  - run root: `runs/phase2_diagnostic_slot_correction_v2_20260418_224242`
- [x] 跑 predicted `slot_correction v2 diagnostics -> forecast` gate
- [x] 回填结论：`slot_correction v2` 未改善双主指标，且相对 `v1` 退化
- [x] 停止继续在 `slot_correction` 头上做同类重采样微调
- [x] 定义下一条主线：`slot_correction v1 track + baseline forecast intensity` integration gate
- [x] 写好 `scripts/run_phase2_slot_correction_intensity_gate_v0_formal.sh`
- [x] 写好 `scripts/run_phase2_slot_correction_intensity_gate_v0_background.sh`
- [x] 启动 `slot_correction v1 track + baseline intensity` integration gate
  - run root: `runs/phase2_slot_correction_intensity_gate_v0_20260419_025200`
- [x] 回填结论：baseline intensity 替换明显改善 intensity，但 track 双主指标仍停在 `v1` 水平
- [x] 冻结新的 integration baseline：`slot_correction v1 track + baseline intensity`
- [x] 在 hybrid baseline 上做 track calibration sweep，优先冲 `mean_track_diff_vs_official_km`
- [x] 回填结论：全局 `offset_scale` sweep 单调改善 track 双主指标，`scale = 1.20` 为当前最佳候选
- [x] 将 `scale = 1.20` 接入正式 compare 入口，并跑新的 formal gate
- [x] 回填结论：`scale = 1.20` 的 sample-200 formal gate 已通过双主指标线
- [x] 启动更大 held-out 的 confirmatory run，验证 `scale = 1.20` 的稳定性
- [x] 回填结论：更大 held-out confirmatory run 也已通过双主指标线
- [x] 在已通过的 `slot_correction` 主线上，设计并启动第一条更宽 schema 扩展分支
- [x] 第一优先级候选：恢复 `diagnostic_track_core_only`，验证 richer track diagnostics 是否还能带来额外稳定增益
- [x] 写好 `configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_track_core_v1.yaml`
- [x] 写好 `scripts/run_phase2_diagnostic_track_core_v1_formal.sh`
- [x] 写好 `scripts/run_phase2_diagnostic_track_core_v1_background.sh`
- [x] 启动 `diagnostic_track_core_only v1` 训练与评测
  - run root: `runs/phase2_diagnostic_track_core_v1_20260419_153108`
- [x] 回填结论：`diagnostic_track_core_only v1` 作为 Phase 2C 第一条扩展 probe 为 negative
- [x] 基于这次 negative result，重新定义下一条 Phase 2C 扩展接口
- [x] 定义新的窄扩展接口：`diagnostic_slot_turn_correction_only`
- [x] 实现 `diagnostic_slot_turn_correction_only` derivation / export / slot-locked renderer calibration 接口
- [x] 写好 `configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_slot_turn_correction_stage_v0.yaml`
- [x] 写好 `scripts/check_phase2_diagnostic_slot_turn_correction.py`
- [x] 写好 `scripts/run_phase2_diagnostic_slot_turn_correction_formal.sh`
- [x] 写好 `scripts/run_phase2_diagnostic_slot_turn_correction_background.sh`
- [x] 启动 `diagnostic_slot_turn_correction_only v0` 训练与评测
  - run root: `runs/phase2_diagnostic_slot_turn_correction_v0_20260419_180019`
- [x] 回填结论：`diagnostic_slot_turn_correction_only v0` 在 sample-200 forecast compare 上给出正向 track signal，但 standalone contract failure 仍然明显
- [x] 对 `diagnostic_slot_turn_correction_only v0` 跑更大 held-out confirmatory compare，并回填结论：forecast-side gain 已确认，但 standalone contract 仍未过关
- [x] 启动 `diagnostic_slot_turn_correction_only v1` contract-repair 分支，优先修复 `turning_signal` collapse / JSON contract，并验证 track gain 不回退
- [x] 对 `diagnostic_slot_turn_correction_only v1` 跑 full confirm，并确认 standalone contract 与 forecast-side gain 同时成立
- [x] 将 `diagnostic_slot_turn_correction_only v1 + baseline intensity + scale 1.20` 升级为当前 confirmed mainline / recommended variant
- [x] 启动第一轮 `slot bucket semantic resolution` 分支，只针对 12 个 slot bucket 字段提升语义分辨率
- [x] 回填结论：`slot_turn_correction v2` 主要提升了 `lon bucket semantics`，但没有转化为更好的 forecast 主指标
- [x] 确认当前剩余瓶颈已经收敛到 `lat bucket semantics`
- [x] 启动 `lat-bucket semantic resolution` 分支，只对 6 个 `slot_i_lat_bias_vs_consensus_bucket` 做定向修复
- [x] 回填结论：`slot_turn_correction v3` 给出当前最强 sample-200 forecast 结果，但 standalone gain 仍主要来自 `lon bucket`
- [x] 对 `slot_turn_correction v3 + baseline intensity + scale 1.20` 启动 full confirm，并验证其可替换当前 `v1 full confirm mainline`
- [x] 对 `slot_turn_correction v3` 做 post-confirm semantic audit，并固定结论：当前 held-out gain 主要来自 `lon semantics`
- [ ] 冻结 RL 入口定义：只允许以 integrated forecast variant 为优化对象，并显式绑定 dual-track reward + guardrail
- [x] 写好首版 RL run plan draft：`docs/PHASE2_RL_RUN_PLAN_DRAFT.md`
- [ ] 将 RL run plan draft 收口成最终执行版，并冻结 smoke / pilot / formal 三套配置

## 3. Design Principles

二阶段数据设计建议遵守以下原则：

1. **Shared evidence 一致**
   所有 diagnostic 样本必须继续使用与 `forecast_only` 相同的输入证据，不允许为 diagnostic 任务额外加入人类不可见信息。

2. **One canonical record, multiple aligned views**
   保持 `sample_id`、split、canonical record 不变，只在 view / export 层增加新的 diagnostic 变体。

3. **Structured output only**
   二阶段主任务只输出固定 schema 的 JSON，不输出 forecast table，不输出 reasoning prose。

4. **Closed label space first**
   优先做闭集、低歧义、覆盖高的字段；不要一开始就把稀疏文本字段作为主目标。

5. **Probe-first, expand-later**
   先做最窄、最直接面向 track 修正的接口 probe；只有 oracle 通过后，才做 core / extended 扩展。

6. **Do not start with RL**
   二阶段 v0 应该先做 SFT，不建议直接接 RL。原因是 structured diagnostics 目前没有稳定 reward，且标签本身属于弱监督。

7. **Downstream benefit is the real gate**
   diagnostic 模型本身的 held-out 准确率不是最终目标；最终 gate 是它能否帮助 forecast。

8. **Oracle-first before predicted modeling**
   在 oracle 还没证明接口有效之前，不要继续训练 predicted diagnostic adapter。

9. **Track metrics before task expansion**
   当前最显著的差距仍然在 track error，因此扩字段的前提必须是 track 主指标先出现改善。

10. **Do not use slot-time / reward / coverage as surrogate track gain**
    `slot_time_match_rate_vs_official`、`reward`、`coverage` 可以作为辅助信号，但不能代替 `track_error_km` 与 `mean_track_diff_vs_official_km`。

11. **Do not use exact official-match as the main gate**
    官方预报 exact match 可以作为描述性指标，但不应作为二阶段 go / no-go 主标准。主标准应是对 truth 的 skill 和对 expert gap 的 closing。

12. **If oracle fails, redesign the interface instead of tuning training**
    一旦 oracle 注入都无法改善 track 主指标，应优先修改 schema / 注入方式，而不是继续做 resampling、checkpoint sweep 或 prompt 微调。

13. **Prefer fixed-lead anchors over relative timing phrases**
    如果接口里的时间信息会被模型误解成 forecast `Day/HHMMZ` 槽位平移，那么它就不是安全的 track 接口。优先使用固定 lead 的 anchor / correction，而不是 `turn_timing_bucket` 这类相对时机标签。

14. **Track-only injections must not alter slot timing or intensity**
    如果二阶段接口的目标是修正轨迹，那么 prompt override 必须明确约束它只影响 `lat/lon` path，不允许顺带改动 forecast 时槽或 intensity。

15. **Prefer correction relative to visible guidance over abstract environment labels**
    当模型已经在 prompt 中看到了 ATCF / HRES guidance，再给它抽象环境标签通常仍然需要一层“从标签编译到轨迹”的映射。下一版应优先使用相对 guidance 的 correction anchors。

## 4. Current Diagnostic Schema

当前导出字段为：

- `track_control_signal`
- `turning_signal`
- `intensity_support_signal`
- `shear_constraint_level`
- `land_interaction_level`
- `model_agreement_level`
- `main_uncertainty_source`
- `forecast_confidence_level`
- `expert_decision_notes`

当前实际导出的标签空间如下：

| field | current values |
| --- | --- |
| `track_control_signal` | `subtropical_high`, `midlatitude_trough`, `competing_ridge_and_trough` |
| `turning_signal` | `steady`, `notable_turn`, `recurvature` |
| `intensity_support_signal` | `supportive`, `mixed`, `constraining` |
| `shear_constraint_level` | `weak`, `moderate`, `strong` |
| `land_interaction_level` | `moderate`, `high` |
| `model_agreement_level` | `high`, `medium`, `low` |
| `main_uncertainty_source` | `model_spread`, `ridge_evolution`, `midlatitude_trough_interaction`, `land_interaction`, `vertical_wind_shear`, `intensity_change` |
| `forecast_confidence_level` | `high`, `medium`, `low` |
| `expert_decision_notes` | nullable free text |

当前训练集中的非空覆盖率如下：

| field | train non-null | train rate | recommendation |
| --- | ---: | ---: | --- |
| `track_control_signal` | 3959 | 100.0% | P0 |
| `turning_signal` | 3793 | 95.8% | P0 |
| `intensity_support_signal` | 3959 | 100.0% | P0 |
| `shear_constraint_level` | 3959 | 100.0% | P0 |
| `model_agreement_level` | 3959 | 100.0% | P0 |
| `forecast_confidence_level` | 3959 | 100.0% | P0 |
| `main_uncertainty_source` | 859 | 21.7% | P1 |
| `land_interaction_level` | 175 | 4.4% | P2 |
| `expert_decision_notes` | 369 | 9.3% | P2 / separate |

需要特别注意的一点：

- `main_uncertainty_source` 在 train 非空率只有 `21.7%`，但 test 非空率高达 `67.8%`

这说明它存在明显 split distribution shift。这个字段不适合一开始就作为二阶段主 KPI，必须先做分布审计。

另一个已经被正式实验验证的问题是：

- `diagnostic_track_turn_only` 的 train 标签并不极端失衡：`track_control_signal` 中 `competing_ridge_and_trough = 2175`，`subtropical_high = 1744`
- 但在 `v0.1` held-out 评测里，模型仍把 `track_control_signal` 预测为 `subtropical_high` 达 `183 / 200`
- 这说明当前 `track_control_signal` 标签空间虽然闭集、覆盖率高，但对模型来说仍然过粗、过易塌缩，不足以充当 forecast 侧的强接口

## 5. Recommended Dataset Variants

### 5.1 Current Reference Probe: `diagnostic_track_turn_only`

当前唯一完成了 standalone + oracle + predicted 正式闭环的变体，是 `diagnostic_track_turn_only`。

它只保留两个字段：

- `track_control_signal`
- `turning_signal`

它当前的用途已经被重新定义为：

- 验证一个最窄环境接口是否有 forecast 价值
- 作为后续 schema redesign 的失败参考
- 作为 explanation / diagnostic 输出分支保留

它当前不再承担的用途是：

- 作为 forecast-accuracy 主线继续迭代
- 作为扩展到 `track_core` / `core` 的自动解锁入口

当前推荐输出示例仍然如下，但这只是 interface probe，不再代表它已经被证明“对 forecast 有用”：

```json
{
  "track_control_signal": "competing_ridge_and_trough",
  "turning_signal": "notable_turn"
}
```

### 5.2 Previous Oracle-Failed Mainline: `diagnostic_track_inflection_only`

`diagnostic_track_inflection_only` 已经完成 formal oracle gate，但结果为 negative，因此它现在只保留为 redesign reference，不再是当前主线。

它的 4 个字段如下：

- `steering_regime_phase`
- `turn_timing_bucket`
- `turn_direction_family`
- `turn_magnitude_bucket`

这套 candidate 原先的设计目标是：

- 对 forecast track 修正有更直接的因果解释，而不是泛化的环境名词
- 能做闭集或小型结构化 schema，避免自由文本噪声
- 能被 oracle 注入到 forecast prompt 中做同样本 compare
- 其字段设计能区分 “会不会转” 与 “往哪里偏、何时偏、偏得多不多”

但 formal 结果已经表明：

- 它没有改善 `track_error_km`
- 它没有改善 `mean_track_diff_vs_official_km`
- `turn_timing_bucket` 这类相对 timing 标签还会在少量样本上把 forecast `Day/HHMMZ` 槽位整体推偏

因此它当前的用途改为：

- 作为 oracle-failed reference case
- 作为“不要再把抽象 inflection 标签直接塞回 forecast prompt”的证据
- 作为下一版 schema redesign 的失败边界

### 5.3 Previous Oracle-Failed Mainline: `diagnostic_track_correction_only`

`diagnostic_track_correction_only` 曾经是从抽象环境接口转向更直接轨迹修正接口的一步，但它的 formal oracle gate 已经完成，当前也被判定为 failure reference，而不是 active mainline。

它的 schema 是：

- `diagnostic_track_correction_only`
- `lat_bias_vs_consensus_48h_bucket`
- `lon_bias_vs_consensus_48h_bucket`
- `lat_bias_vs_consensus_72h_bucket`
- `lon_bias_vs_consensus_72h_bucket`

它当时的设计思路是：

- 不再让模型从抽象环境标签自己“编译”出轨迹
- 只描述 `48h / 72h` 这两个关键 lead 的 coarse correction anchors
- 参考对象使用 prompt 中已经可见的 ATCF consensus guidance，而不是不可见基准
- 标签只作用于 track path，不承担 intensity / confidence / slot-time 语义

推荐值族：

- 对 `lat_*` 字段：
  - `south_large`
  - `south_small`
  - `near_consensus`
  - `north_small`
  - `north_large`
- 对 `lon_*` 字段：
  - `east_large`
  - `east_small`
  - `near_consensus`
  - `west_small`
  - `west_large`

当前已经基于 train split 的 `official - ATCF consensus` 偏差分布做过一轮 quantile audit，因此可以先冻结一版 **provisional cut points**：

- `48h`
  - `near_consensus`: `|delta| < 50 km`
  - `*_small`: `50-150 km`
  - `*_large`: `>= 150 km`
- `72h`
  - `near_consensus`: `|delta| < 75 km`
  - `*_small`: `75-175 km`
  - `*_large`: `>= 175 km`

对应 train 覆盖率：

- `48h` anchor matched: `3587 / 3987`，约 `90.0%`
- `72h` anchor matched: `3145 / 3987`，约 `78.9%`

补充说明：

- `60h` anchor 在 train 上只有 `692 / 3987` 可对齐，明显差于 `72h`
- 因此下一版仍然优先保留 `48h + 72h`，而不是改成 `48h + 60h`

但 formal oracle 结果已经说明，它仍然不够直接：

- `track_error_km` 几乎不变
- `mean_track_diff_vs_official_km` 反而变差
- 这说明“固定 lead 的 correction anchor”仍然要求 forecast model 再做一层抽象编译

所以它当前只保留为：

- oracle-failed redesign reference
- “比环境标签更近，但仍然不够近”的证据
- slot-locked direct correction 的上一版失败边界

### 5.4 Current Mainline And First Oracle-Passed Interface: `diagnostic_slot_correction_only`

当前 active candidate 不再按固定 `48h/72h` lead 输出 correction anchors，而是直接对 official forecast 的可见槽位输出 **slot-locked track correction buckets**。

推荐的新 candidate 名称：

- `diagnostic_slot_correction_only`

推荐字段：

- `slot_1_lat_bias_vs_consensus_bucket`
- `slot_1_lon_bias_vs_consensus_bucket`
- `slot_2_lat_bias_vs_consensus_bucket`
- `slot_2_lon_bias_vs_consensus_bucket`
- `slot_3_lat_bias_vs_consensus_bucket`
- `slot_3_lon_bias_vs_consensus_bucket`
- `slot_4_lat_bias_vs_consensus_bucket`
- `slot_4_lon_bias_vs_consensus_bucket`
- `slot_5_lat_bias_vs_consensus_bucket`
- `slot_5_lon_bias_vs_consensus_bucket`
- `slot_6_lat_bias_vs_consensus_bucket`
- `slot_6_lon_bias_vs_consensus_bucket`

设计思路：

- 不再让 forecast model 把 `48h/72h` anchor 再翻译回完整 track table
- 每个字段直接绑定到 official forecast 已有的 `slot_1..slot_6`
- 每个槽位的参考对象是 prompt 中已经可见的 ATCF representative point
- `Day/HHMMZ` 槽位固定，不允许新增、删除、平移
- 只修正 track 位置，不修正 intensity
- 当后续槽位没有可见 ATCF point 时，对应字段置 `null`

当前 train 分布已经说明 variable-length slot schema 比 fixed-lead schema 更贴近真实输出：

- official forecast slot count 分布：
  - `5: 2621`
  - `6: 602`
  - `4: 364`
  - `3: 204`
  - `2: 106`
  - `1: 62`
  - `0: 28`
- 非空覆盖：
  - `slot_1`: `3959`
  - `slot_2`: `3897`
  - `slot_3`: `3791`
  - `slot_4`: `3587`
  - `slot_5`: `3223`
  - `slot_6`: `602`

当前冻结的 label family 与 fixed-lead correction 一致：

- `lat`:
  - `south_large`
  - `south_small`
  - `near_consensus`
  - `north_small`
  - `north_large`
- `lon`:
  - `east_large`
  - `east_small`
  - `near_consensus`
  - `west_small`
  - `west_large`

这条主线之所以值得继续，不是因为“看起来更合理”，而是因为它已经在同一 `sample_count = 200 / seed = 3407` 子集上给出强 offline sanity signal：

- visible ATCF passthrough:
  - `track_error_km = 232.0359`
  - `mean_track_diff_vs_official_km = 217.6116`
- simple constant median shift:
  - `track_error_km = 132.9103`
  - `mean_track_diff_vs_official_km = 104.8409`
- oracle slot-locked rendering:
  - `track_error_km = 104.5653`
  - `mean_track_diff_vs_official_km = 49.2633`

这条主线现在不再只是“推荐 gate candidate”，而是已经完成 formal oracle 验证的 **当前主线**。

formal oracle 结果如下：

- run root: `runs/phase2_diagnostic_slot_correction_oracle_v0_20260418_154645`
- baseline `track_error_km = 184.60`
- oracle `track_error_km = 104.65`
- baseline `mean_track_diff_vs_official_km = 165.90`
- oracle `mean_track_diff_vs_official_km = 49.30`
- baseline `reward_mean = 0.2374`
- oracle `reward_mean = 0.3990`
- baseline `coverage = 0.8015`
- oracle `coverage = 1.0000`

同时必须一起看的 control 是：

- `visible_atcf_consensus_passthrough_v0`
  - `track_error_km = 232.04`
  - `mean_track_diff_vs_official_km = 217.61`

这说明真正有效的是 “slot-locked correction”，不是“直接把 visible ATCF 当 forecast”。

所以当前 gate 已从 oracle redesign 阶段推进到 predicted training 阶段：

1. 先跑 `visible_consensus` control
2. 再跑 `oracle_slot_locked_forecast_correction_v0`
3. 只有当 oracle 相对 baseline 在 `track_error_km` 或 `mean_track_diff_vs_official_km` 上给出真实增益时，才训练 predicted diagnostic adapter

这套 candidate 的核心假设是：

- forecast 模型更容易把“slot 对齐的局部 north/south/east/west 修正”转成具体轨迹
- slot-locked correction 比相对 timing 语言更不容易污染 forecast 时槽
- 只传输 track correction，不再让诊断块顺带影响 intensity

推荐输出示例：

```json
{
  "slot_1_lat_bias_vs_consensus_bucket": "north_small",
  "slot_1_lon_bias_vs_consensus_bucket": "west_small",
  "slot_2_lat_bias_vs_consensus_bucket": "near_consensus",
  "slot_2_lon_bias_vs_consensus_bucket": "west_small",
  "slot_3_lat_bias_vs_consensus_bucket": "north_large",
  "slot_3_lon_bias_vs_consensus_bucket": "west_large"
}
```

注入约束也必须同步改变。当前 renderer / prompt block 应显式声明：

- 这些字段只对应固定 official slot 的 track correction
- 不允许因为这段诊断而修改 forecast `Day/HHMMZ` 槽位
- 不允许因为这段诊断而联动修改 intensity
- 如果某个后续 slot 的 consensus anchor 缺失，则对应字段为 `null`

### 5.4.1 Current Phase 2C Executable Probe: `diagnostic_slot_turn_correction_only`

在 `diagnostic_slot_correction_only` 已经完成 oracle / predicted 主线验证之后，当前 Phase 2C 不再回到 generic `track_core` 注入，而是先尝试一个更窄但仍然可执行的扩展：

- `diagnostic_slot_turn_correction_only`

它的 schema 是：

- 保留 `diagnostic_slot_correction_only` 的 12 个 `slot_1..slot_6 lat/lon bias bucket`
- 只额外增加 1 个字段：`turning_signal`

这条 probe 的关键约束是：

- forecast integration 仍然走 `slot-locked deterministic renderer`
- `turning_signal` 只是进入 renderer 的条件 calibration，不重新退回 generic prompt override
- `Day/HHMMZ` 槽位继续固定
- intensity 仍然不由 diagnostic renderer 直接生成

当前 formal run、contract-repair 训练、以及更大 held-out full confirm 都已完成：

- v0 run root: `runs/phase2_diagnostic_slot_turn_correction_v0_20260419_180019`
- v1 train run root: `runs/phase2_diagnostic_slot_turn_correction_v1_20260420_095215`
- v1 confirm run root: `runs/phase2_diagnostic_slot_turn_correction_v1_confirm_20260420_141813`
- v1 best checkpoint: `checkpoint-40`
- v1 confirm standalone `json_parse_rate = 1.0000`
- v1 confirm standalone `mean_macro_f1 = 0.3246`
- v1 confirm standalone `turning_signal exact_accuracy = 0.8246`
- v1 confirm standalone `turning_signal macro_f1 = 0.8309`
- v1 confirm predicted rendered `track_error_km = 159.59`
- v1 confirm predicted rendered `mean_track_diff_vs_official_km = 140.12`
- v1 confirm predicted rendered + baseline intensity `track_error_km = 151.96`
- v1 confirm predicted rendered + baseline intensity `mean_track_diff_vs_official_km = 131.04`
- v1 confirm predicted rendered + baseline intensity `reward_mean = 0.3665`

当前 verdict 需要分开看：

- **standalone diagnostic contract**：positive。`slot_turn_correction v1` 已经把 `json_parse_rate` 从 `0.0825` 修到 `1.0000`
- **turning semantics**：positive。`turning_signal` 不再塌缩，full confirm `macro_f1 = 0.8309`
- **forecast-side**：positive。相对此前 confirmed mainline `slot_correction v1 + baseline intensity + scale 1.20`，`slot-turn v1 + baseline intensity` 在更大 held-out 上仍小幅改善了 `track_error_km` `1.76 km`，并改善了 `mean_track_diff_vs_official_km` `2.49 km`
- **overall promotion status**：promoted。`Phase 2C` 的 contract + forecast-side 双条件已经同时满足

因此这条 probe 当前的正式定位已经更新为：

- 作为当前新的 **confirmed mainline / recommended variant**
- 证明 “只额外加一维 turning information，并把它接进 deterministic renderer calibration” 不只是 forecast-side winner，而且已经是可稳定执行的主线接口
- 下一步不再是继续做 contract repair，而是只针对 12 个 slot bucket 的语义分辨率做定向提升

### 5.5 Deferred Legacy Variants: `diagnostic_track_core_only` / `diagnostic_core_only` / `diagnostic_extended_only`

以下变体继续保留，但当前只作为条件扩展入口，不是默认主线：

- `diagnostic_track_core_only`
- `diagnostic_core_only`
- `diagnostic_extended_only`

它们只有在下列条件同时满足时才允许恢复：

1. 新的 environment interface candidate 先通过 oracle track gate
2. predicted diagnostics 也恢复出可观的 oracle 增益
3. forecast 主指标至少有一项开始稳定向 expert 靠近

### 5.6 Not Recommended as Phase 2 Default: `expert_decision_notes`

`expert_decision_notes` 当前更像“稀疏 explanation 残片”，不是稳定的 structured diagnostics：

- 非空率低
- 文本风格噪声更大
- 与 Phase 3 reasoning 的边界容易混淆

建议：

- 保留在 canonical 中
- 继续保留在现有 full diagnostic 导出中
- 但不要把它放进二阶段默认训练目标

如果后续需要，可单独做：

- `diagnostic_notes_only`
- 或 `diagnostic_json + short_notes` 的辅助实验

## 6. Current Data Layout

当前根目录兼容层已经导出以下视图文件：

```text
data/training_rebuilt_v2_20260414_guidancefix/
  views/
    diagnostic_only/
      train.jsonl
      val.jsonl
      test.jsonl
      report.json
    diagnostic_track_turn_only/
      train.jsonl
      val.jsonl
      test.jsonl
      report.json
    diagnostic_slot_correction_only/
      train.jsonl
      val.jsonl
      test.jsonl
      report.json
    diagnostic_slot_turn_correction_only/
      train.jsonl
      val.jsonl
      test.jsonl
      report.json
    diagnostic_track_correction_only/
      train.jsonl
      val.jsonl
      test.jsonl
      report.json
    diagnostic_track_core_only/
      train.jsonl
      val.jsonl
      test.jsonl
      report.json
    diagnostic_core_only/
      train.jsonl
      val.jsonl
      test.jsonl
      report.json
  sft_diagnostic_train.jsonl
  sft_diagnostic_val.jsonl
  sft_diagnostic_test.jsonl
  sft_diagnostic_track_turn_train.jsonl
  sft_diagnostic_track_turn_val.jsonl
  sft_diagnostic_track_turn_test.jsonl
  sft_diagnostic_slot_correction_train.jsonl
  sft_diagnostic_slot_correction_val.jsonl
  sft_diagnostic_slot_correction_test.jsonl
  sft_diagnostic_slot_turn_correction_train.jsonl
  sft_diagnostic_slot_turn_correction_val.jsonl
  sft_diagnostic_slot_turn_correction_test.jsonl
  sft_diagnostic_track_correction_train.jsonl
  sft_diagnostic_track_correction_val.jsonl
  sft_diagnostic_track_correction_test.jsonl
  sft_diagnostic_track_core_train.jsonl
  sft_diagnostic_track_core_val.jsonl
  sft_diagnostic_track_core_test.jsonl
  sft_diagnostic_core_train.jsonl
  sft_diagnostic_core_val.jsonl
  sft_diagnostic_core_test.jsonl
```

当前策略：

- 保留现有 `diagnostic_only` 作为 full schema 版本
- 保留 `diagnostic_track_turn_only` 作为当前 formal reference probe / failure case
- 将 `diagnostic_track_correction_only` 归档为失败 redesign reference
- 将 `diagnostic_slot_correction_only` 作为当前 stability control / pre-2C mainline reference
- 将 `diagnostic_slot_turn_correction_only v3` 作为当前 confirmed mainline / recommended variant
- 将 `diagnostic_slot_turn_correction_only v1` 保留为 current stability control
- 冻结 `diagnostic_track_core_only` 与 `diagnostic_core_only`，但暂停把它们作为默认扩展主线
- 所有变体共用同一套 `sample_id` 和 split

## 7. Training Strategy

### 7.1 Baseline Selection And Primary Gate

二阶段开始前，必须先明确 baseline 口径。

研究口径上保留三层 baseline：

- `baseline_forecast_sft_v2`
  - **Operational default**
  - 当前默认 forecast-side adapter
  - 当前正式 sample-200 compare 上整体最强
- `baseline_forecast_grpo_physical_v2`
  - **Secondary control**
  - 当前 **physical-error-led** GRPO baseline
  - 对应 `grpo/adapter_step-000050`
- `baseline_forecast_grpo_reward_v2`
  - **Secondary control**
  - 当前 **reward-led** GRPO baseline
  - 对应 `grpo/adapter_reward-0.7544_step-000003`

执行层当前应按下面这句理解：

- **Operational default**：`baseline_forecast_sft_v2`
- **Secondary controls**：`baseline_forecast_grpo_physical_v2`、`baseline_forecast_grpo_reward_v2`

二阶段所有 forecast-side 评估都应继续汇报 **expert-gap closing**，但主 gate 现在更明确：

- 第一主指标：`track_error_km`
- 第二主指标：`mean_track_diff_vs_official_km`
- 次级 guardrail：`reward`、`coverage`
- 次级描述信号：`slot_time_match_rate_vs_official`、`intensity_error_kt`

对于越低越好的指标，例如 track / intensity error：

```text
gap_closed = (baseline_error - new_error) / (baseline_error - expert_error)
```

对于越高越好的指标，例如 reward / coverage：

```text
gap_closed = (new_metric - baseline_metric) / (expert_metric - baseline_metric)
```

也就是说，二阶段现在不再接受“辅助指标更好但 track 主指标没更好”的乐观解释。

### 7.2 Revised Phase 2 Definition

当前 Phase 2 改成三段式：

1. **Phase 2A: oracle-first interface validation**
   只验证某个环境接口在 oracle 注入下是否真的改善 track 主指标。
2. **Phase 2B: predicted interface recovery**
   只有 `Phase 2A` 通过后，才训练 predicted diagnostic adapter。
3. **Phase 2C: schema expansion**
   只有 `Phase 2A + 2B` 都通过后，才允许恢复 `track_core` / `core` / `extended` 扩展。

当前已经完成的 `diagnostic_track_turn_only` 应被视为 `Phase 2A` 的第一份 formal reference failure，而不是一条还应继续打磨的默认主线。
当前已经完成的 `diagnostic_slot_correction_only` 则应被视为 `Phase 2A` 的第一份 formal positive interface。

### 7.3 Current Reference Failure: `diagnostic_track_turn_only`

当前 `track_turn` 接口已经完成了从 standalone 到 forecast integration 的完整闭环：

- `v0`：`phase2_diagnostic_track_turn_v0_20260416_131804`
- `v0.1`：`phase2_diagnostic_track_turn_v0_1_20260416_205033`

正式结果说明：

- `v0.1` standalone 相比 `v0` 只出现小幅提升：
  - `joint_exact_match_rate`: `0.305 -> 0.330`
  - `mean_field_exact_accuracy`: `0.5850 -> 0.5975`
  - `mean_field_macro_f1`: `0.4867 -> 0.4958`
- 但 forecast 主指标没有被拉正：
  - baseline `track_error_km = 186.30`
  - oracle `track_error_km = 192.17`
  - predicted `v0.1 track_error_km = 189.44`
  - baseline `mean_track_diff_vs_official_km = 167.29`
  - oracle `mean_track_diff_vs_official_km = 175.22`
  - predicted `v0.1 mean_track_diff_vs_official_km = 172.42`
- 当前唯一稳定改善的是 `slot_time_match_rate_vs_official`，但这不足以证明 forecast track accuracy 有收益

因此：

- `track_turn` 当前只保留为 explanation / interface-audit 分支
- 不再继续把它当作 forecast-accuracy 主线
- 不再因为 standalone 提升、checkpoint 变好、slot-match 变好就继续扩字段

### 7.4 Oracle Interface Gate

在把任何新的 predicted diagnostics 接回 forecast 前，必须先做一个更严格的 oracle 试验：

1. 先定义新的 environment interface candidate
2. 用真实标签构造 oracle prompt override
3. 与 `baseline_forecast_sft_v2` 在同样本集上比较
4. 只有在以下任一主指标改善时，才算 oracle 通过：
   - `track_error_km` 下降
   - `mean_track_diff_vs_official_km` 下降
5. `slot_time_match_rate_vs_official` 单独上升，不再视为通过条件

如果 oracle 都不能改善上述主指标，那么说明问题主要在：

- 接口设计本身过粗或不对题
- forecast prompt / 注入方式没把这类信息转成轨迹修正

这种情况下应停止 predicted-side 调参，回到 schema 和注入方式本身。

### 7.4.1 Current Oracle Gate Result: `diagnostic_track_inflection_only v0`

`diagnostic_track_inflection_only` 的 formal oracle gate 已完成：

- run root: `phase2_diagnostic_track_inflection_oracle_v0_20260418_133608`
- sample count: `200`
- baseline: `baseline_forecast_sft_v2`
- oracle: `oracle_track_inflection_candidate_v0_plus_forecast`

正式结果：

- baseline `track_error_km = 184.60`
- oracle `track_error_km = 190.12`
- baseline `mean_track_diff_vs_official_km = 165.90`
- oracle `mean_track_diff_vs_official_km = 175.82`
- baseline `reward_mean = 0.2374`
- oracle `reward_mean = 0.2317`
- baseline `coverage = 0.8015`
- oracle `coverage = 0.7978`
- baseline `slot_time_match_rate_vs_official = 0.7822`
- oracle `slot_time_match_rate_vs_official = 0.7838`

结论：

- `diagnostic_track_inflection_only v0` **未通过 oracle gate**
- 两个主指标都变差：
  - `track_error_km`: `+5.52 km`
  - `mean_track_diff_vs_official_km`: `+9.92 km`
- `slot_time_match_rate_vs_official` 的轻微上升不能抵消主指标恶化
- 因此当前 schema 不允许进入 predicted training

### 7.4.2 Failure Audit: Why `track_inflection v0` Still Failed

对 sample-level compare 做进一步审计后，可以把这次失败拆成两层：

1. **少量 catastrophic timing-shift cases**
   - oracle 赢 `68` 个样本，输 `100` 个样本，平 `32` 个样本
   - 只有 `7` 个样本出现明显的 `coverage` / `slot-time` 退化
   - 但这 `7` 个样本的平均：
     - `track_error_km delta = +149.77 km`
     - `mean_track_diff_vs_official_km delta = +174.77 km`
   - 这说明 `turn_timing_bucket` 之类的相对时机标签，会在少量样本上被模型误解成 forecast `Day/HHMMZ` 槽位平移

2. **去掉 timing-shift 后仍然没有稳定 track gain**
   - 在其余样本上，平均 `track_error_km delta` 约为 `+0.00 km`
   - 但 `mean_track_diff_vs_official_km delta` 仍约为 `+3.89 km`
   - 同时整体统计仍然是：
     - `track_error_km` 改善 `49` 个样本、恶化 `98` 个样本、持平 `52` 个样本
     - `mean_track_diff_vs_official_km` 改善 `47` 个样本、恶化 `104` 个样本、持平 `48` 个样本

因此，不能把这次失败简单理解成“删掉 `turn_timing_bucket` 就行”。更准确的结论是：

- 当前 inflection schema 既存在 timing leakage 风险
- 又依然过于抽象，不能稳定地把 oracle 信息转成对 forecast path 的可用修正

### 7.4.3 Redesign Decision After `track_inflection v0`

下一版 redesign 必须同时满足：

1. 改用固定 lead 的 track anchors，而不是相对 timing 语言
2. 改用相对 prompt 内可见 guidance 的 correction，而不是全局环境标签
3. 明确把诊断块限制为 track-only，不允许联动时槽和 intensity

因此当前默认下一条 candidate 已切换为：

- `diagnostic_track_correction_only`

### 7.4.4 Oracle Result Archive After Redesigns

到目前为止，Phase 2 的 oracle interface archive 可以收敛成下面这组结论：

1. `diagnostic_track_turn_only`
   - failed
   - 原因：环境标签太抽象，track 主指标没有改善
2. `diagnostic_track_inflection_only`
   - failed
   - 原因：既有 timing leakage 风险，又仍然太抽象
3. `diagnostic_track_correction_only`
   - failed
   - 原因：固定 `48h/72h` anchors 仍然要求模型自己重建完整 path
4. `diagnostic_slot_correction_only`
   - passed oracle gate
   - 原因：它直接绑定 official slot，并把修正语义压缩到 prompt 中可见 guidance 的局部 correction

这意味着当前唯一允许进入 `Phase 2B predicted training` 的接口是：

- `diagnostic_slot_correction_only`

### 7.5 Predicted Interface Gate

只有在 oracle 通过后，才允许做 predicted diagnostics：

1. 训练新的 predicted diagnostic adapter
2. 跑同样本的 forecast integration compare
3. 要求 predicted 至少恢复 `25%` 的 oracle 在同一主指标上的正向增益：
   - `track_error_km`，或
   - `mean_track_diff_vs_official_km`
4. 同时保持 truth-side guardrail：
   - `reward` 不下降超过 `0.02`
   - `coverage` 不下降超过 `0.02`

这里要强调：

- standalone diagnostic 提升不是 unlock 条件
- best checkpoint selection、early stopping、resampling、staged init 只是训练 hygiene，不是战略通过条件

### 7.6 Expansion Gate

只有在以下条件同时满足时，才允许恢复 `diagnostic_track_core_only`、`diagnostic_core_only` 或 `diagnostic_extended_only`：

1. standalone diagnostic eval 不是崩溃式失败
2. oracle interface gate 已满足
3. predicted interface gate 已满足
4. forecast-to-truth 或 forecast-to-official 的 track 主指标至少一项开始稳定向 expert 靠近

如果上述条件不满足，则默认策略是：

- 停止扩字段
- 停止在当前接口上继续加训练轮次
- 优先修改接口定义与注入方式

### 7.7 Updated Go / No-Go Summary

为避免再次出现“跑完以后再解释结果”，当前统一使用下面这套口径：

`continue current interface`

1. standalone 没有 schema 崩溃
2. oracle 改善 `track_error_km` 或 `mean_track_diff_vs_official_km`
3. predicted 恢复至少 `25%` 的 oracle 增益
4. `reward`、`coverage` 没有明显回撤

`stop current interface`

1. oracle 只改善 `slot_time_match_rate_vs_official`，但 track 主指标不改善
2. standalone 有提升，但 predicted forecast 仍不改善 track 主指标
3. 继续调训练只是在修 label score，没有把 forecast 主目标修好

## 8. Evaluation Design

当前仓库已经补齐 diagnostic held-out 评测脚本与 forecast integration compare 脚手架，因此这里的重点从“补工程入口”转成“冻结执行判定口径并运行实验”。

当前 `scripts/eval_diagnostic_heldout.py` 应至少支持以下指标：

1. JSON parse rate
2. Per-field exact accuracy
3. Per-field macro-F1
4. Null vs non-null detection F1
5. Core-field joint exact match
6. Confusion matrix for each categorical field
7. Split distribution report
8. 预测 diagnostics 注入前后的 forecast gap-closing report

评测要同时给出两套口径：

- `track-turn fields only`
- `track-core fields only`
- `core fields only`
- `full fields`

此外，二阶段的 forecast-side 评测必须补三套表：

1. **forecast vs truth**
   继续沿用 reward / coverage / track error / intensity error

2. **forecast vs official**
   新增：
   - mean track diff vs official
   - mean intensity diff vs official
   - slot-time match rate vs official

3. **forecast vs expert gap closing**
   用上文公式直接汇报 gap closed 比例

建议至少比较以下 baseline：

1. majority-class baseline
2. rule echo baseline
   直接复用 `heuristic_v2` 导出的标签作为上限参考不成立，但可以作为“标签源本身”的对照审计
3. `diagnostic_adapter`

注意：

- `exact_match_rate_vs_official` 不应作为主 KPI
- 只要 slot 时间、轨迹、强度略有偏移，整表 exact match 就会掉到 0
- 二阶段应该看“是否更接近专家”，而不是“是否逐字符复制专家”
- `slot_time_match_rate_vs_official` 只能作为辅助描述，不再单独充当扩展 unlock 条件

## 9. Required Work

### 9.1 Data / Export

1. 保留现有 `diagnostic_track_turn_only` / `diagnostic_track_core_only` / `diagnostic_core_only` 导出，不再改动 canonical 或 split
2. 将 `diagnostic_track_turn_only` 明确标记为 formal reference failure / explanation branch
3. 冻结 `diagnostic_track_inflection_only` 的 label space，并把它作为 formal failure reference
4. 基于 failure audit 定义新的 `diagnostic_track_correction_only`
5. 基于 `track_correction` oracle negative result，继续 redesign 为 `diagnostic_slot_correction_only`
6. 继续审计稀疏字段，如 `main_uncertainty_source`，但这不再是当前主线 blocker

当前状态：

- 第 1-5 项已完成
- `diagnostic_track_inflection_only` 已冻结 label space，并完成 oracle compare
- `diagnostic_track_correction_only` 已完成 formal oracle compare，当前 verdict 是 negative
- `diagnostic_slot_correction_only` 已完成 formal oracle compare，当前 verdict 是 positive

### 9.2 Training

1. 停止在当前 `track_turn` 接口上继续做 `v0.x` tuning
2. `diagnostic_slot_correction_only` 的第一步不是训练，而是 oracle compare
3. 只有 `diagnostic_slot_correction_only` 的 oracle 通过后，才训练新的 predicted diagnostic adapter
4. 训练 hygiene 可以继续沿用：
   - staged init
   - label-level resampling
   - best checkpoint selection
   - early stopping
5. 但这些只用于提升训练效率，不再作为继续主线的理由

当前状态：

- 训练基础设施已完成
- `slot_correction v0` 与 `v1` predicted recovery 结果都已经拿到
- 当前 `track_turn` 分支进一步训练已暂停
- 当前 `track_inflection v0` 分支进一步训练也已暂停
- 当前 `track_correction` 分支进一步训练也已暂停
- `slot_correction v0` predicted training 已完成：
  - `runs/phase2_diagnostic_slot_correction_v0_20260418_161630`
- `slot_correction v1` predicted training 也已完成：
  - `runs/phase2_diagnostic_slot_correction_v1_20260418_184506`
  - best checkpoint: `checkpoint-160`
  - compact prompt / compact target 已生效
  - line-based recovery 已接入
  - staged init from `slot_correction v0 final_adapter`
- `slot_correction v2` predicted training 已启动：
- `slot_correction v2` predicted training 已完成：
  - `runs/phase2_diagnostic_slot_correction_v2_20260418_224242`
  - staged init from `slot_correction v1 final_adapter`
  - 停用会抬高 `near_consensus` 的通用 rarity resampling
  - 改成针对 `north/west` 主导 correction 与少数 `south/east` 尾部标签的 targeted min multipliers
- `slot_correction v2` 已证明同类重采样微调不是最优下一步
- 当前最直接的下一步改成：保留 `slot_correction v1` 作为最优轨迹修正头，单独测试强度来源替换

### 9.3 Evaluation

1. 新增 `scripts/eval_diagnostic_heldout.py`
2. 新增 diagnostic report 聚合脚本
3. 产出 `runs/.../diagnostic_eval_summary.md`
4. 新增 `expert_gap_closing` 汇总逻辑
5. 在正式结果中明确标记：
   - primary metrics
   - secondary metrics
   - pass / stop verdict

当前状态：

- 第 1-2 项已完成：
  - `scripts/eval_diagnostic_heldout.py`
  - `scripts/compare_diagnostic_models.py`
- 第 3 项已有正式 run 结果可引用
- 第 4 项已在 forecast-side compare 逻辑中实现：
  - `scripts/eval_strict_forecast_heldout.py`
  - `scripts/compare_diagnostic_forecast_integration.py`
- 第 5 项需要在后续实验汇报模板中继续保持
- `slot_correction v1` 当前正式 standalone 结果为：
  - `json_parse_rate = 0.8850`
  - `mean_field_acc = 0.4104`
  - `mean_macro_f1 = 0.3593`
  - `line_recovered_rate = 0.1150`
- `slot_correction v2` 当前正式 standalone 结果为：
  - `json_parse_rate = 0.8250`
  - `mean_field_acc = 0.4417`
  - `mean_macro_f1 = 0.3450`
  - 结论：standalone 上升并没有转成更好的 downstream track

### 9.4 Forecast Integration

1. 保留现有 prompt override 框架
2. 当前主线已经切到 `diagnostic_slot_turn_correction_only v1 + baseline intensity` 的 predicted gate
3. 如果 predicted 恢复不了足够比例的 oracle 增益，立即停止该接口
4. 如果 predicted 通过，再做 schema expansion gate
5. 只有两级 gate 都通过后，才允许恢复更宽 schema 的 forecast integration

当前状态：

- 第 1 项已完成，注入格式已固化在 `scripts/build_forecast_prompt_overrides.py`
- `track_turn` 的 oracle / predicted compare 已正式完成，并给出 stop verdict
- `diagnostic_track_inflection_only` 的 oracle gate 也已正式完成，并给出 stop verdict
- `diagnostic_track_correction_only` 的 oracle gate 也已正式完成，并给出 stop verdict
- `diagnostic_slot_correction_only` 当前不再依赖 prompt override 改写 forecast，而是用 slot-locked deterministic renderer：
  - 固定 official `Day/HHMMZ`
  - 固定 slot count
  - intensity 先保持 visible consensus
- 当前 formal compare 会同时保留 `visible_atcf_consensus_passthrough_v0` control
- `diagnostic_slot_correction_only` 的 oracle gate 已正式完成，并给出 go verdict：
  - `track_error_km`: `184.60 -> 104.65`
  - `mean_track_diff_vs_official_km`: `165.90 -> 49.30`
- `diagnostic_slot_correction_only` 的 predicted v0 compare 也已完成：
  - `track_error_km`: `184.60 -> 167.69`
  - `mean_track_diff_vs_official_km`: `165.90 -> 148.35`
  - `reward`: `0.2374 -> 0.3264`
  - `intensity_error_kt`: `13.10 -> 14.73`
- `diagnostic_slot_correction_only` 的 predicted v1 compare 也已完成：
  - `track_error_km`: `184.60 -> 163.44`
  - `mean_track_diff_vs_official_km`: `165.90 -> 143.34`
  - `reward`: `0.2374 -> 0.3287`
  - `intensity_error_kt`: `13.10 -> 14.73`
  - payload recovery mode: `json=173`, `line_kv=27`
- `diagnostic_slot_correction_only` 的 predicted v2 compare 也已完成：
  - `track_error_km`: `184.60 -> 170.67`
  - `mean_track_diff_vs_official_km`: `165.90 -> 150.50`
  - `reward`: `0.2374 -> 0.3307`
  - `intensity_error_kt`: `13.10 -> 14.73`
  - 结论：相比 `v1`，两个主轨迹指标都退化
- 当前未完成环节已从 “是否有 predicted gain” 变成：
  - 如何在保留 `slot_correction v1` 轨迹收益的同时，把 intensity 从 visible-consensus passthrough 中解耦出来
  - 如何验证“只替换强度来源”本身能否改善整体 gate，而不先引入新的训练头
- 第 5 项已完成接线：
  - `reward`
  - `coverage`
  - `track error`
  - `intensity error`
  - `mean track diff vs official`
  - `mean intensity diff vs official`
  - `slot-time match rate vs official`
  - `expert-gap closing`

### 9.5 Documentation and Tracking

1. 记录当前 active objective，而不是保留过时目标
2. 记录每个字段 / 接口的 allowed values
3. 记录 split-level label distribution
4. 记录 oracle gain、predicted gain 与最终 verdict
5. 记录已经踩过的坑，避免把训练 hygiene 误判成 forecast 进展

当前状态：

- 第 1-5 项已在本文和现有 formal run 结果中开始落地
- 后续任何新接口都应先补本文，再开正式 run

## 10. Recommended Execution Order

建议按以下顺序启动二阶段：

1. 基于 `training_rebuilt_v2_20260414_guidancefix` 冻结正式 Phase 1 baseline
2. 固定二阶段 operational default forecast adapter：
   - `baseline_forecast_sft_v2`
3. 将 `baseline_forecast_grpo_physical_v2` 与 `baseline_forecast_grpo_reward_v2` 标记为 confirmatory controls，而不是默认全跑主线
4. 冻结 `track_turn` 的正式负结论，不再继续在当前 schema 上追加 tuning
5. 冻结 `track_inflection v0` 的正式负结论，不再继续在当前 schema 上追加 tuning
6. 冻结 `diagnostic_track_correction_only` 的正式负结论，不再继续在当前 schema 上追加 tuning
7. 使用 `diagnostic_slot_correction_only` 作为当前 candidate
8. 保留 `visible_atcf_consensus_passthrough_v0` 作为必须同时报告的 control
9. 承认 oracle `slot-locked correction -> forecast` gate 已通过
10. 承认 `slot_correction v1` 虽然继续逼近 oracle，但在新的双主指标 gate 下仍未正式通过
11. 承认 `slot_correction v2` 已经给出 formal negative result，不再继续沿这一路径追加 `v3`
12. 下一步切到 `slot_correction v1 track + baseline intensity` integration gate
13. 只有 `track + intensity` integration 都稳定后，才考虑恢复 `diagnostic_track_core_only`
14. 最后才考虑 `diagnostic_core_only` 和 `diagnostic_extended_only`

## 11. What Not To Do

当前不建议：

1. 把 reasoning 文本拼进二阶段主目标
2. 直接用 full diagnostic + free-text notes 当默认训练集
3. 把二阶段成功定义成“diagnostic 标签准确率变高”
4. 在没有 standalone eval 的情况下上 GRPO
5. 在 oracle 没过关时继续做 predicted diagnostics 链路
6. 把 `slot_time_match_rate_vs_official`、`reward`、`coverage` 的改善当成 track gain 的替代品
7. 在当前 `track_turn` schema 上继续做 checkpoint sweep、resampling 加码、prompt 小修小补
8. 在当前 `track_inflection v0` schema 上继续做 predicted training 或 prompt 小修小补
9. 继续使用相对 timing 标签直接指导 forecast，而不加 slot-time guardrail
10. 在 `diagnostic_track_correction_only` 这个 fixed-lead schema 上继续做 predicted training
11. 在没有看到 track 主指标改善的情况下就扩字段
12. 再次改动 canonical / sample split / shared-evidence 输入定义

## 12. Pitfalls And Lessons From The Current Runs

1. **把 standalone 提升误当成 forecast 进展**
   `v0.1` 的 standalone 指标比 `v0` 好，但 forecast track 主指标仍然劣于 baseline。以后不能再用 label score 的小幅提升替代 downstream 结论。

2. **把过松的 gate 当成 go 信号**
   oracle 和 predicted 都提高了 `slot_time_match_rate_vs_official`，但 `track_error_km` 与 `mean_track_diff_vs_official_km` 都没有改善。以后 `slot-time` 只能当辅助指标。

3. **在失败接口上继续磨训练细节**
   staged init、label-level resampling、checkpoint sweep、best-checkpoint selection、early stopping 都是有价值的工程能力，但它们没把失败接口变成有效接口。以后要先问“接口对不对”，再问“训练够不够”。

4. **低估了标签空间塌缩问题**
   `track_control_signal` 的 train 分布不算特别失衡，但 held-out 预测仍然高度塌缩到 `subtropical_high`。这说明当前标签定义过粗、过弱，不能把轨迹修正所需的信息传给模型。

5. **把 predicted failure 归因到模型，而忽略 oracle failure**
   当前更关键的事实是 oracle 自己都没把 track 主指标拉正。这意味着问题首先在接口设计或注入方式，而不是仅仅在 diagnostic adapter 的预测精度。

6. **把训练 hygiene 当成战略证据**
   best checkpoint 选得更准、训练提前停止、standalone 更稳定，这些都是必要的，但它们只说明训练更规范，不说明目标已经对了。

7. **相对 timing 标签会污染 forecast 时槽**
   `track_inflection v0` 的最差样本里，oracle 并不是把轨迹坐标改到完全离谱，而是把整条 forecast 的 `Day/HHMMZ` 往后推，导致 `coverage`、`slot_time_match_rate_vs_official` 和 track 主指标一起崩掉。以后只要接口里有 timing 信息，就必须优先问“它会不会让模型改时槽”。

8. **抽象 inflection 语义仍然离 forecast table 太远**
   即使把那几个 catastrophic timing-shift case 单独拿掉，`track_inflection v0` 仍然没有稳定缩小 `mean_track_diff_vs_official_km`。这说明“regime / direction / magnitude”这种全局语义，依然需要模型自己再做一层编译，离实际 forecast path 仍然太远。

9. **固定 `48h/72h` correction anchors 仍然太间接**
   `diagnostic_track_correction_only` 已经比环境标签更接近轨迹修正，但 formal oracle 结果仍然几乎没有改善 `track_error_km`，且 `mean_track_diff_vs_official_km` 还变差。这说明只给少数固定 lead 的 correction anchors，依然要求模型自己重建整条 forecast path，接口还是太远。

10. **必须把 visible-consensus control 一起报**
   如果不同时报告 `visible_atcf_consensus_passthrough_v0`，就很难判断提升到底来自“接口有用”，还是仅仅来自“把 forecast 替换成 ATCF-like path”。slot-locked redesign 之后，这个 control 必须成为 formal compare 的标配。

11. **track-only oracle 通过，不等于 intensity 一起通过**
   `diagnostic_slot_correction_only` 的 oracle 确实大幅改善了 `track_error_km` 与 `mean_track_diff_vs_official_km`，但当前 deterministic renderer 仍然沿用 visible-consensus intensity，因此 `intensity_error_kt` 与 `mean_intensity_diff_vs_official_kt` 还差于 baseline。这个 caveat 必须保留，避免把“track-only oracle pass”误写成“forecast 全面通过”。

12. **contract 修好以后，瓶颈会从 diagnostic 头转移到 integration 头**
   `slot_correction v1` 已经把 standalone `json_parse_rate` 修到 `0.8850`，forecast payload 中 `173/200` 样本直接走标准 JSON，另有 `27/200` 走 `line_kv` recovery。这说明当前瓶颈已经不再主要是“模型完全不会输出可用 schema”，而是“renderer 仍把 intensity 锁死在 visible consensus”。下一轮如果继续只磨 prompt / parse，大概率收益会迅速递减。

13. **standalone 上升不等于双主指标会继续上升**
   `slot_correction v2` 的 `mean_field_acc` 虽然升到 `0.4417`，但 forecast 侧反而从 `163.44 / 143.34` 退回到 `170.67 / 150.50`。这再次证明当前阶段不能拿 standalone 进展替代 downstream gate，尤其不能据此继续在同类重采样上加码。

## 13. Bottom Line

如果现在继续二阶段，最稳的做法不再是“继续把现有 diagnostics 训得更准”，而是：

1. 保持 `canonical_v2` 和 shared-evidence 输入不动
2. 先承认当前模型与专家之间仍存在显著 gap，尤其是 track gap
3. 冻结 `diagnostic_track_turn_only` 这次 formal negative result
4. 承认 `diagnostic_track_inflection_only v0` 也已经给出 formal oracle negative result
5. 承认 `diagnostic_track_correction_only` 也已经给出 formal oracle negative result
6. 不在当前 `track_inflection v0` 或 `track_correction` schema 上启动 predicted training
7. 承认 `diagnostic_slot_correction_only` 已经给出当前唯一的 formal oracle positive result
8. 这条新主线直接绑定 official slot，而不是固定 `48h/72h` lead
9. 这条新主线固定 `Day/HHMMZ`、slot count，并暂时冻结 intensity
10. 当前所处位置不再是“继续找接口”或“继续把窄接口训到过线”，而是准备进入下一阶段
11. `slot_correction v1 + baseline intensity + scale 1.20` 已经在 sample-200 formal gate 和更大 held-out confirmatory run 上都把 predicted 侧双主指标推过线
12. 但 `slot_correction v2` 已经证明，同类重采样微调不是更优路线
13. 因此下一步不再新增 `slot_correction v3`，也不再回到 schema redesign，更不需要继续打磨当前这点强度尾差
14. 更合理的候选是：
    - track 仍使用当前 `slot_correction v1` renderer
    - intensity 继续沿用当前 baseline forecast adapter 输出即可
15. 下一步不应继续在 `slot_correction` 本身上做同类小修，但也不能把“更宽 schema”简单理解成 generic diagnostic injection
16. `diagnostic_track_core_only v1` 已经证明：更宽信息如果还是以 generic prompt override 方式注入，oracle 自己都不能改善双主指标
17. `diagnostic_slot_turn_correction_only v0` 说明了更窄、仍可执行的 `slot + turning_signal` 扩展是对的，但它自己还没有守住 standalone contract
18. `diagnostic_slot_turn_correction_only v3` 的 full confirm 已进一步说明：这条 executable extension 现在不仅继续满足 contract 稳定、turning 语义可恢复、forecast-side gain 不回退三个条件，而且在更大 held-out 上继续优于 `v1` control

也就是说，二阶段的核心已经从“继续磨环境标签”或者“再加一点 correction anchor”，进一步收敛成五个连续动作：先确认 **slot-locked actionable interface** 是对的，再把它在 predicted 侧真正训到双主指标过线，然后只做 **窄而可执行** 的扩展，再把这个扩展修到可稳定接管主线，最后才考虑恢复更宽 schema。到目前为止，`track_turn`、`track_inflection v0`、`track_correction`、`slot_correction v2`、`track_core v1` 都应视为失败分支；`diagnostic_slot_turn_correction_only v3 + baseline intensity + scale 1.20` 现已升级为当前 confirmed mainline / recommended variant；`diagnostic_slot_turn_correction_only v1 + baseline intensity + scale 1.20` 与 `diagnostic_slot_correction_only v1 + baseline intensity + scale 1.20` 则转为 stability controls。`Phase 2C` 本身现在可以视为完成；当前更合理的下一步也不再是继续补 contract，而是先完成 post-confirm semantic audit 与 RL objective 收口，再决定是否进入 RL。
