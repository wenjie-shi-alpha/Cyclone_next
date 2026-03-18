# 微调数据集构建 Guide（v0.1）

更新时间：2026-03-12  
适用范围：`Cyclone_next` 当前 `v0.1.x` 样本构建链路

---

## 1. 目标与范围

本 Guide 用于约束“从多源原始数据到可训练微调样本”的落地流程，目标是：
1. 保证无信息泄漏（严格区分输入、官方输出、事后验证）。
2. 保证字段口径可解释（单位、时间窗、缺失原因可追溯）。
3. 控制 prompt token 开销（保留可学习信号，删除高冗余标识串）。

本 Guide 面向：
- `task2_data_source_mapping_and_dataset_construction.md` 对应的数据工程执行。
- `scripts/build_dataset_sample_preview_v0_1.py` 与后续批量建集脚本。

---

## 2. 数据集契约（必须遵守）

样本层级：
1. `prompt.now_inputs`：`issue_time` 时刻可见信息（观测/环境/当前状态）。
2. `prompt.guidance_inputs`：`issue_time` 时刻可见的未来指导（EC/ATCF 多模型共识）。
3. `target.official_outputs`：NOAA 当期官方输出。
4. `verification_targets`：B-deck/IBTrACS 事后真值，仅用于评估或奖励。

硬约束：
1. `verification_targets` 绝不能回流到 `prompt`。
2. `valid_time <= issue_time` 的记录不能进入未来指导字段。
3. 所有缺失必须显式记录 `status + missing_reason`，禁止静默丢样本。
4. 所有关键输入都要保留 `source_file/source_trace` 以便审计与重算。

---

## 3. 推荐构建流程（当前工作流）

### 3.1 GOES 结构化特征（已修复口径）

全量重跑入口：

```bash
bash scripts/run_goes_full_controlled.sh eminent-glider-467006-r0 --force
```

说明：
1. 提取端已在 GEE 内做 `CMI_C13 * 0.1` 单位修复。
2. 采用严格 QC，不再依赖事后 `÷10` 启发式“救回”坏值。
3. 输出会同步到：
- `data/interim/goes/goes_observation_features_full.csv`
- `data/interim/goes/goes_observation_features.csv`

### 3.2 预览样本构建

```bash
source .venv/bin/activate
python scripts/build_dataset_sample_preview_v0_1.py
```

输出：
- `data/interim/schema/dataset_v0_1_sample_preview_ec_single_source.json`

### 3.3 ATCF token 精简（已落地）

`multimodel_guidance_a_deck` 仅保留：
1. `model_count`
2. `consensus_*`
3. `track_spread_km / wind_spread_kt`

已移除高 token 冗余：
1. 顶层 `model_ids`
2. 逐时效点 `models` 长字符串

---

## 4. 构建过程中的关键注意点

### 4.1 时间与泄漏

1. 所有输入必须满足 `可在 issue_time 时刻获得`。
2. 建议统一容差：`|delta_t| <= 3h`，超阈值标记而非硬删。
3. 文本泄漏检查要覆盖：discussion 原文、public 摘要、verify 字段。

### 4.2 单位与物理范围

1. GOES C13 温度必须为 K（非 DN、非 K*10）。
2. 阈值类特征（如 cold cloud）要保证阈值与数据单位一致。
3. 关键特征需做物理范围 QC（如温度、风速、SST、切变）。

### 4.3 缺失语义分层

不要把所有缺失混为一类。建议至少区分：
1. `no_data_in_window`（真实不可得）
2. `invalid_after_qc`（有数据但口径/质量不合格）

两类缺失对训练策略和后续修复优先级完全不同。

### 4.4 Token 预算

1. 优先保留统计量与物理量（可学习）。
2. 删除高基数 ID 串（难学习且占 token）。
3. 明细 ID 放到 `source_trace` 或离线表，不放进 prompt 主体。

### 4.5 可追溯与可复算

1. 每个阶段产物保留 summary（rows/available/missing/reason 分布）。
2. 关键脚本参数（窗口、阈值、scale）要固定并写入文档。
3. 产物命名保持 `full` 与 `canonical` 同步策略，避免读到旧文件。

---

## 5. 当前已知问题（截至 2026-03-12）

### 5.1 GOES 覆盖与质量

现状（`goes_observation_features_full_summary.json`）：
1. `rows_written=7085`
2. `available_rows=3230`
3. `missing_rows=3855`

缺失原因分布（`goes_observation_features_full.csv`）：
1. `invalid_goes_temperature_metrics_after_qc = 2719`
2. `no_goes_image_in_window = 1136`

含义：
1. 当前不是“纯覆盖不足”，还包含“质量口径导致不可用”的缺失。
2. 后续需要分开治理 `invalid_qc` 与 `no_image`。

### 5.2 观测证据层未闭环

1. ASCAT 结构化表：缺失，但 Sherlock Slurm 提交链路已补齐，待执行全量提取。
2. Recon/VDM 结构化表：部分可用（本地执行模式已打通）。

Recon 当前进展（本地）：
1. 烟雾测试：`rows_written=20`, `available_rows=14`, `missing_rows=6`。
2. 全量合并（2016-2025）：`rows_written=6981`, `available_rows=3551`, `missing_rows=3430`。
3. 分年覆盖：2016 `0.566372`，2017 `0.554362`，2018 `0.409613`，2019 `0.630015`，2020 `0.345606`，2021 `0.485636`，2022 `0.685976`，2023 `0.497076`，2024 `0.509646`，2025 `0.330275`。
4. 产物位置：
- `data/interim/recon/recon_observation_features_smoke.csv`
- `data/interim/recon/recon_observation_features_full.csv`
- `data/interim/recon/recon_observation_features_full_summary.json`

影响：
1. `now_inputs.observation_evidence_structured` 仍主要依赖 GOES + 部分 Recon，ASCAT 尚缺。
2. 对强度与低层风场证据支持不足。

### 5.3 ATCF 跨源映射仍有缺口

1. `storm_id_crosswalk` 仍有未映射 token（重点在近年）。
2. 部分样本需依赖 IBTrACS 回退验证。

### 5.4 ERA5 对齐口径

1. `CDS_real` 无显式 `storm_id`，依赖时间+位置匹配。
2. 需要持续维护匹配规则与误差审计。

---

## 6. 发布前质量门禁（Checklist）

每次发版前至少检查：
1. 数据规模：总样本、可用样本、缺失样本。
2. 缺失原因 TopN：是否出现新异常类型。
3. 单位与范围：关键数值是否落在物理合理区间。
4. 泄漏检查：`leakage_audit` 全部关键项为预期值。
5. token 预算：ATCF 块不含 `model_ids/models`。
6. 路径一致性：构建脚本默认读取的 canonical 文件是最新产物。

---

## 7. 下一步执行建议

1. 对 GOES `invalid_qc` 分布做 year/basin/storm 级诊断，形成可调整清单。
2. 优先接入 ASCAT，并继续补齐 Recon 剩余年份与扩展子目录（`AHONT1/AHOPN1`）。
3. 在批量建集中固化“缺失分层标签”，用于训练采样与评估分桶。
4. 保持本 Guide 与 `task2_data_source_mapping_and_dataset_construction.md` 同步更新。
