# Task 2 数据方案：从多源数据到预报微调数据集（详尽版）

## 0. 文档目标与核心原则

本文件对应 `idea.md` 的第 2 个任务：  
“根据 NOAA 预报高频信息定义所需数据，将 CDS / AWS S3 Open Data / Google Research Data / GEE 等数据源映射到可提取字段，并组织成可用于预报微调的数据集。”

同时吸收 Appendix 的约束：
1. 要严格区分“输入可得信息”和“预报结果输出”。
2. 之前 `TianGong-AI-Cyclone` / `TianGong-AI-Cyclone-GFS` 的资产不废弃，改造成可复用特征层。
3. 目标是“可执行的数据工程方案”，不是只做概念框架。

配套执行文档：
- `finetune_dataset_build_guide.md`（构建流程、门禁检查、当前问题与 token 预算策略）

---

## 1. 从 Step 1 到 Step 2：数据需求总览

基于 `noaa_forecast_step1_deconstruction.md`，我们把训练样本拆成四层：

1. `now_inputs`：预报时刻可用的现状信息（观测/分析/历史状态）
2. `guidance_inputs`：预报时刻可用的未来指导信息（A-deck、多模式、未来环境场指导）
3. `official_outputs`：NOAA 预报员本时次输出（轨迹/强度/风险/讨论）
4. `verification_targets`：事后验证目标（B-deck/IBTrACS 真值，供评估/RFT奖励）

关键因果关系：
- 输入：`now_inputs + guidance_inputs`
- 输出：`official_outputs`
- 验证：`verification_targets`（不能回流到同一样本输入，避免信息泄漏）

---

## 2. 数据源映射（字段 -> 数据集）

## 2.1 P0/P1/P2 优先级

### P0（必须）
1. 本地 NOAA 文本归档（当前仓库 `noaa/`）
2. ATCF A-deck / B-deck（NHC FTP）
3. ERA5（CDS）

### P1（建议）
1. AWS NODD: GFS / GOES
2. GEE: ERA5、GFS、OISST、GOES 数据集

### P2（扩展）
1. Google Research / WeatherBench2（大规模历史重分析与预报基线）

---

## 2.2 详细字段映射表

| 训练层 | 字段组 | 关键字段 | 首选数据源 | 备选数据源 | 备注 |
|---|---|---|---|---|---|
| `now_inputs` | 当前状态 | `current_lat/lon`, `vmax`, `pmin`, `motion` | NOAA `forecast_advisory` / `public_advisory`（本地） | ATCF `CARQ/WRNG` | 与预报员可见信息口径最一致 |
| `now_inputs` | 观测证据文本 | 卫星结构、侦察机、散射计、雷达叙述 | NOAA `forecast_discussion` | - | 作为 reasoning 监督文本 |
| `now_inputs` | 环境现状诊断 | `vws_200_850`, `rh_mid`, `div200`, `sst_local` | ERA5（CDS） | GEE ERA5 | 用再分析近似“可感知现状” |
| `guidance_inputs` | 多模式轨迹/强度 | `model_track_fcst`, `model_intensity_fcst` | ATCF A-deck | GEE GFS 补充 | 核心未来输入 |
| `guidance_inputs` | 共识与分歧 | `track_spread`, `wind_spread`, `consensus` | A-deck 派生 | 旧 TianGong 结果 | 直接支撑不确定性表达 |
| `guidance_inputs` | 未来环境场指导 | 未来切变/脊槽/湿度/SST 变化 | GFS（AWS/GEE） | WeatherBench2 HRES/ERA5-forecast | 用于解释未来变化机理 |
| `official_outputs` | 官方数值输出 | 官方轨迹、强度、风圈、相态 | NOAA `forecast_advisory`（本地） | A-deck 中 `OFCL` 技术行 | 主监督标签 |
| `official_outputs` | 官方推理文本 | 轨迹/强度/不确定性解释 | NOAA `forecast_discussion`（本地） | - | SFT 文本目标 |
| `official_outputs` | 官方风险产品 | `watch/warning`, hazard, wind prob | `public_advisory` + `wind_speed_probabilities` | - | 风险沟通监督 |
| `verification_targets` | 事后真值 | 实际路径、强度 | B-deck / IBTrACS | HURDAT2（区域） | 仅用于评估/RFT奖励 |

---

## 3. 各数据源可执行提取方案

## 3.1 NOAA 本地文本（当前仓库现有）

输入路径（已在仓库）：
- `noaa/<year>/<basin>/<storm>/forecast_discussion/*.txt`
- `noaa/<year>/<basin>/<storm>/forecast_advisory/*.txt`
- `noaa/<year>/<basin>/<storm>/public_advisory/*.txt`
- `noaa/<year>/<basin>/<storm>/wind_speed_probabilities/*.txt`

提取目标：
1. 解析每个 advisory 的 `issue_time` 和 `advisory_no`
2. 从 `forecast_advisory` 抽结构化表（INIT/12H/24H...）
3. 从 `forecast_discussion` 抽推理段落（现状、轨迹、强度、不确定性）
4. 从 `public_advisory` 抽风险沟通和预警状态
5. 从 `wind_speed_probabilities` 抽地点阈值概率矩阵

输出建议：
- `data/interim/noaa_cycles.parquet`（每时次一行）
- `data/interim/noaa_outputs.parquet`（官方输出结构化）
- `data/interim/noaa_reasoning_text.jsonl`

---

## 3.2 ATCF（A-deck / B-deck）

主要入口：
- A-deck（实时与近实时）：`https://ftp.nhc.noaa.gov/atcf/aid_public/`
- B-deck（最佳路径）：`https://ftp.nhc.noaa.gov/atcf/btk/`
- 历史归档：`https://ftp.nhc.noaa.gov/atcf/archive/`
- ABR 格式说明：`https://science.nrlmry.navy.mil/atcf/docs/database/new/abrdeck.html`

### A-deck（未来可得输入）
用途：
1. 多模式轨迹/强度指导（GFS/ECMWF/HWRF/HAFS/统计模型等）
2. 计算共识与分歧（spread）
3. 提取官方技巧行（如 `OFCL`）用于与 NOAA 文本互校

关键字段（ABR）：
- `BASIN, CY, YYYYMMDDHH, TECH, TAU, LatN/S, LonE/W, VMAX, MSLP, ...`

清洗规则：
1. 经纬度字符串转十进制度（`N/S/E/W`）
2. `TAU` 转 `lead_hour`
3. 统一 `init_time_utc` 与 `valid_time_utc = init + TAU`
4. 按 `storm_id + init_time + TECH + TAU` 去重（保留最后版本）
5. 模型别名归一（如同一模型不同后缀）

### B-deck（事后验证）
用途：
1. 训练后离线评估（track/intensity error）
2. RFT 奖励（对官方输出或候选输出进行真实性打分）

注意：
- B-deck 是后验最佳路径，不能作为同一样本输入。

---

## 3.3 ERA5（CDS）

主要入口：
- CDS API 指南：`https://cds.climate.copernicus.eu/en/how-to-api`
- ERA5 数据说明：`https://confluence.ecmwf.int/x/Oi1EDg`

核心数据集短名（CDS）：
1. `reanalysis-era5-single-levels`
2. `reanalysis-era5-pressure-levels`

推荐变量（对齐 NOAA 高频推理）：
1. `u`, `v` at 200/850 hPa -> 垂直风切变
2. `relative_humidity` at 700/500 hPa -> 中层湿度/干空气
3. `divergence` at 200 hPa -> 高层外流代理
4. `geopotential` at 500 hPa -> 脊/槽背景
5. `sea_surface_temperature` -> SST 环境
6. `mean_sea_level_pressure` -> 大尺度环流背景

空间时间采样建议：
1. 以 `issue_time` 对齐（容差 <= 3h）
2. 以风暴中心为圆形窗口（500km）或环形窗口（200-800km）
3. 输出窗口统计（mean/max/std/azimuthal asymmetry）

---

## 3.4 AWS S3 Open Data（NODD）

建议使用：
1. GFS：`https://registry.opendata.aws/noaa-gfs-bdp-pds/`
2. GOES：`https://registry.opendata.aws/noaa-goes/`

用途映射：
1. GFS：未来环境场指导（`guidance_inputs`）
2. GOES：云系结构/对流组织代理特征（`now_inputs`）

工程建议：
1. 若 CDS 节点下载外部慢，AWS/HPC 节点先拉取再下采样成小表回传
2. 大体积网格不直接入训练样本，仅保留特征摘要

---

## 3.5 Google Earth Engine（GEE）

推荐集合：
1. ERA5 Hourly：`ECMWF/ERA5/HOURLY`
2. GFS：`NOAA/GFS0P25`
3. OISST：`NOAA/CDR/OISST/V2_1`
4. GOES（示例）：`NOAA/GOES/19/MCMIPC`

用途：
1. 快速按轨迹点批量提取环境统计量
2. 在 GEE 端完成空间聚合，导出轻量表（CSV/TFRecord）
3. 作为 CDS/AWS 的替代或补齐路径

---

## 3.6 Google Research Data（WeatherBench2）

入口：
- 数据指南：`https://weatherbench2.readthedocs.io/en/latest/data-guide.html`
- 站点：`https://sites.research.google/gr/weatherbench/`

可用价值：
1. 大规模历史 ERA5 / HRES / AI 预报样本（研究用途）
2. 可用于构建“环境未来变化指导”或模型候选输出池

定位建议：
- 作为 P2 扩展数据源，不替代 ATCF A-deck 在热带气旋任务中的主地位。

---

## 4. 样本构建：从“时次”出发的统一主键

主键定义：
`sample_id = basin + storm_num + year + issue_time_utc + advisory_no`

每条样本必须包含：
1. `now_inputs`（全部字段有效时间 <= issue_time）
2. `guidance_inputs`（模型初始化时间 <= issue_time，预测目标时间 > issue_time）
3. `official_outputs`（该 issue 对应 NOAA 官方输出）
4. `verification_targets`（可选，供评估/RFT）

## 4.1 按预报时刻注入旧资产（与你现有组织逻辑一致）

你的旧数据（`GFS_forecast` / `HRES_forecast`）是按预报时次组织的，这非常适合直接注入 `guidance_inputs`。  
建议把“时次+台风ID”作为注入锚点，规则如下：

1. 时次锚点：
- 以样本 `issue_time_utc` 为主键时间（你当前逻辑为每日 00/12）。
- 若后续接入 NOAA 6 小时制时次（常见 00/06/12/18 或业务发布时次），允许 `±3h` 最近邻匹配。

2. 风暴 ID 锚点：
- 统一成 `storm_id_norm`（如 `2025067S12085` 或 ATCF 规范 ID）。
- 建一张 `storm_id_crosswalk`（内部ID <-> ATCF ID <-> NOAA文本ID），避免跨源命名不一致。

3. 文件定位：
- 轨迹：`*_track/track_<storm_id>_*_<init_time>*`
- 环境系统：`*_system/*_TC_Analysis_<storm_id>.json`
- 批处理索引：`*_system/_analysis_manifest.json`

4. 注入内容：
- `track/*.csv` -> `guidance_inputs.model_track_fcst`
- `*_TC_Analysis_*.json.time_series[*].environmental_systems` -> `guidance_inputs.environment_guidance`
- 从 `time_idx` 或 `time` 推导 `tau_h`（0/6/12/.../240）

5. 起始点规则（与你现有做法一致）：
- 以“离 `issue_time` 最近的路径点”作为 `tau=0` 的注入起点；
- 再拼接后续 `tau` 的环境系统时间序列（建议至少保留 12/24/48/72/96/120h）。

6. 质量控制：
- 若 `environmental_systems=[]`，保留样本并标记 `env_guidance_coverage=0`；
- 若同一时刻有重复轨迹点，按“最低 `msl` + 最高 `wind` 优先”或固定去重规则保留 1 条。

该策略的好处是：不需要重跑旧流程即可把历史资产直接转成“未来环境场指导输入”。

---

## 5. 关键特征提取方法（可直接编码）

## 5.1 轨迹/强度分歧（A-deck）

对每个 `lead_hour in {12,24,36,48,72,96,120}`：
1. 收集全部可用模型在该 `valid_time` 的位置和强度
2. 计算：
   - `track_spread_km_tau`: 对共识中心的距离标准差或中位绝对偏差
   - `wind_spread_kt_tau`: `vmax` 标准差
   - `model_count_tau`

输出：
- `guidance_spread` 子结构

## 5.2 环境场诊断（ERA5/GFS）

1. `vws_200_850 = |V200 - V850|`
2. `rh_mid = mean(RH700, RH500)`
3. `div200 = divergence@200hPa`
4. `z500_grad` / `z500_anom`（脊槽背景）
5. `sst_local` / `sst_anom`

输出：
- `environment_now`
- `environment_guidance`

## 5.3 文本证据提取（forecast_discussion）

提取四段：
1. `current_analysis_text`
2. `track_reasoning_text`
3. `intensity_reasoning_text`
4. `uncertainty_hazard_text`

并保留指针：
- `source_file`
- `source_advisory_no`
- `source_issue_time`

---

## 6. 旧资产复用（Appendix 对接方案）

你之前的路径“模型轨迹与环境场冲突检测”不再作为主任务，但旧数据可转为高价值输入：

1. 复用为 `path_environment_timeline`：
- 每个模型路径每个 `tau` 对应局地环境特征（切变、湿度、SST等）
- 不再强调“冲突标签”，而强调“候选路径条件描述”

2. 复用为 `high_disagreement_case` 筛选器：
- 用既有结果快速识别“高分歧/高不确定”样本
- 用于 curriculum（先训普通样本，再训困难样本）

3. 复用为 prompt 压缩摘要：
- 将高维环境场转换为短文本诊断，减少 token 开销

4. 复用为 RFT 候选动作池：
- 候选输出可来自“不同模型路径 + 环境特征解释”
- 用 B-deck/官方输出打分形成偏好对

---

## 7. 微调数据集组织方案

## 7.1 SFT 数据（监督微调）

### 输入（prompt）
1. `storm_meta`（storm_id, issue_time, basin）
2. `now_inputs`（结构化摘要 + 关键文本证据）
3. `guidance_inputs`（模型共识/分歧 + 未来环境指导）

### 输出（target）
1. `official_outputs.track_intensity_table`
2. `official_outputs.risk_messages`
3. `official_outputs.reasoning_text`（来自 discussion）

推荐 JSONL 结构：
```json
{
  "sample_id": "AL042025_2025-08-06T09:00:00Z_010",
  "prompt": {
    "storm_meta": {},
    "now_inputs": {},
    "guidance_inputs": {}
  },
  "target": {
    "official_outputs": {}
  }
}
```

## 7.2 RFT 数据（偏好/奖励微调）

构造方式：
1. 固定同一 `prompt`
2. 生成多个候选 `candidate_forecast`（模型引导、规则修正、历史模板）
3. 用奖励函数排序，形成 `chosen/rejected`

奖励函数示例：
1. 与官方输出的一致性（训练 imitate-expert）
2. 与 B-deck 的误差（训练 outcome-aware）
3. 物理一致性惩罚（如极端不合理增弱）
4. 风险沟通完整性（是否覆盖关键 hazard）

---

## 8. 防泄漏与时间对齐规则（必须执行）

1. 输入截断：
- `now_inputs.source_time <= issue_time`
- `guidance_inputs.init_time <= issue_time`

2. 输出定义：
- 本时次 NOAA 产品一律归 `official_outputs`

3. 验证隔离：
- `B-deck/IBTrACS` 只用于离线评估与奖励，不进入同一样本 prompt

4. 时间容差：
- 默认 `|delta_t| <= 3h`；超阈值打 `coverage_flag=0`

5. 缺测处理：
- 不丢弃样本；字段置空并写 `missing_reason`

---

## 9. 建议的数据目录与产物契约

```text
data/
  raw/
    noaa_text/
    atcf/
      aid_public/
      btk/
    era5/
    gfs/
    goes/
  interim/
    cycles.parquet
    now_inputs.parquet
    guidance_inputs.parquet
    official_outputs.parquet
  features/
    spread_features.parquet
    env_now_features.parquet
    env_guidance_features.parquet
  training/
    sft_train.jsonl
    sft_valid.jsonl
    rft_pairwise.jsonl
  reports/
    coverage_report.md
    leakage_audit.md
```

字段契约最小集（每条样本）：
1. `sample_id`
2. `storm_id`, `issue_time_utc`, `advisory_no`
3. `now_inputs.*`
4. `guidance_inputs.*`
5. `official_outputs.*`
6. `coverage_flag`, `delta_t_hours`, `source_trace`

---

## 10. 实施里程碑（建议）

### Milestone A（1周）：核心管线打通（P0）
1. NOAA 本地文本解析
2. A-deck/B-deck 下载与清洗（已完成，2026-03-04）
3. ERA5 关键变量提取
4. 产出首版 `sft_train.jsonl`（小样本）

### Milestone B（1-2周）：增强输入（P1）
1. 接入 GFS（AWS/GEE）未来环境指导
2. 接入 OISST/GOES 观测代理特征
3. 完成 spread + 环境诊断摘要模板

### Milestone C（1周）：RFT 数据与评估
1. 生成 candidate forecasts
2. 建立奖励函数与 pairwise 数据
3. 输出 `rft_pairwise.jsonl` + 评估报告

---

## 11. 与你当前项目状态的衔接结论（更新于 2026-03-05）

1. 你当前仓库已具备 NOAA 输出语料（官方输出层），可直接支撑 `official_outputs` 构建。  
2. ATCF 已完成下载、解压、清洗和按 `storm_id` 聚合，A-deck/B-deck 不再是“缺数据”，已进入可用状态。  
3. verify groundtruth 已落地为“`B-deck` 主、`IBTrACS` 回退”：按 storm 产出 `verification_groundtruth_preferred.csv`，并保留来源字段（`source_used`）。  
4. `build_dataset_sample_preview_v0_1.py` 已升级为优先读取真实 ATCF 聚合结果（A-deck multimodel + preferred verification），缺失时才回退占位。  
5. 旧 TianGong 资产继续作为“路径-环境时间序列特征库”使用，主链路不再依赖“冲突检测结论”。  
6. 训练目标仍建议两阶段：
   - 阶段1：imitation（向 NOAA 官方输出学习）
   - 阶段2：outcome-aware（叠加 B-deck/IBTrACS 奖励）
7. GOES 结构化观测已接入主链路：`build_dataset_sample_preview_v0_1.py` 默认读取 `data/interim/goes/goes_observation_features.csv`，样本可直接注入 `now_inputs.observation_evidence_structured`。

---

## 12. 现有数据系统 Review（基于当前仓库）

状态标记：
- `READY`：已有数据可直接满足目标字段（只需工程清洗/拼装）。
- `PARTIAL`：已有可用数据，但覆盖率或口径尚有缺口。
- `MISSING`：当前仓库无可用数据，需新增请求/下载。

### 12.1 资产盘点（截至 2026-03-09）
1. NOAA 本地语料：
- `forecast_discussion` 18,491
- `forecast_advisory` 18,505
- `public_advisory` 17,273
- `wind_speed_probabilities` 16,571

2. 旧逻辑模型资产：
- `HRES_forecast/HRES_track`：3,553 文件
- `HRES_forecast/HRES_system`：3,553 JSON（3,552 非空环境系统）
- `GFS_forecast/GFS_track`：9,560 文件（可用 CSV 4,780 + `:Zone.Identifier` 4,780）
- `GFS_forecast/GFS_system`：3,769 JSON（3,553 非空环境系统）+ `:Zone.Identifier` 3,769
- `GFS` 总 `:Zone.Identifier` 冗余文件 8,549 个（track+system，需清理）

3. ERA5 实况环境资产：
- `CDS_real`：209 个按月 JSON（50,269 条时刻环境记录）
- 系统类型较完整（`VerticalWindShear/UpperLevelDivergence/OceanHeatContent/...`）
- 仍缺显式 `storm_id`/`basin` 字段，但已通过 `time + tc_position` 与 GroundTruth 对齐（50,269/50,269 命中）

4. 轨迹真值资产：
- `GroundTruth_Cyclones/matched_cyclone_tracks.csv`：53,186 行，870 个唯一 `storm_id`

5. ATCF 资产（新增，已入工作流）：
- `data/raw/atcf/by_category/a_deck`：2016-2025 共 421 文件  
- `data/raw/atcf/by_category/b_deck`：2016-2025 共 398 文件  
- `data/processed/atcf/plaintext/a_deck`：约 3.6G，34,638,335 行  
- `data/processed/atcf/plaintext/b_deck`：约 4.7M，18,414 行  
- `data/interim/atcf/storm_id_crosswalk.csv`：398 行（B-deck token 级映射）  
- `data/interim/atcf/by_storm/<storm_id>/`：870 个目录，包含：
  - `a_deck_guidance.csv`（325 storm）
  - `a_deck_spread.csv`（325 storm）
  - `b_deck_best_track.csv`（325 storm）
  - `ibtracs_best_track.csv`（870 storm）
- `verification_groundtruth_preferred.csv`（870 storm，B-deck优先）
- 汇总文件：`data/interim/atcf/summary.json`、`data/interim/atcf/a_deck_file_summary.csv`

6. GOES 结构化观测资产（新增，已入工作流）：
- `data/interim/goes/goes_request_manifest_full.csv`：8,010 行（其中 `storm_id` 已匹配 7,085 行）
- `data/interim/goes/goes_observation_features_full.csv`：7,085 行（`available=3,230`，`missing=3,855`，覆盖率 `0.455893`）
- `data/interim/goes/goes_observation_features_full_summary.json`：分年覆盖统计（2016=0.0，2017=0.463087，2018=0.324873，2019=0.613670，2020=0.616390，2021=0.589603，2022=0.684451，2023=0.394152，2024=0.496988，2025=0.601399）
- `data/interim/goes/goes_observation_features.csv` 已与 full 文件同步（用于样本构建默认读取）
- GOES 已完成单位修复并全量重跑：提取端采用严格 QC（不再做事后 `÷10` 启发式补救）

### 12.2 目标字段可用性标注（MARK）

| 目标层 | 目标字段/能力 | 现有来源 | 状态 | 结论 |
|---|---|---|---|---|
| `official_outputs` | 官方轨迹/强度/风圈/相态 | `noaa/*/forecast_advisory` | `READY` | 可直接解析为监督标签 |
| `official_outputs` | 官方推理文本 | `noaa/*/forecast_discussion` | `READY` | 可直接用于 reasoning target |
| `official_outputs` | 风险与概率产品 | `public_advisory` + `wind_speed_probabilities` | `READY` | 可直接构建风险沟通输出 |
| `now_inputs` | 当前中心/强度基础态 | `GroundTruth_Cyclones` + NOAA INIT行 | `READY` | 可形成当前状态输入 |
| `now_inputs` | 当前环境诊断（切变/外流/海温/引导背景） | `CDS_real`（EC/ERA5链路） | `PARTIAL` | 已可对齐样本主键，但需固定特征口径与单位 |
| `guidance_inputs` | EC 单模式轨迹/环境指导 | `HRES_track` + `HRES_system` | `PARTIAL` | v0 可用，但覆盖受 HRES 时窗限制 |
| `guidance_inputs` | A-deck 多模型轨迹/强度 | `data/interim/atcf/by_storm/*/a_deck_guidance.csv` | `PARTIAL` | 已落地，受 `storm_id_crosswalk` 覆盖限制（当前 325 storm） |
| `guidance_inputs` | 共识/离散度（多模型） | `data/interim/atcf/by_storm/*/a_deck_spread.csv` | `PARTIAL` | 已落地，仍需模型白名单与质量过滤规则 |
| `verification_targets` | B-deck 事后真值 | `data/interim/atcf/by_storm/*/b_deck_best_track.csv` | `PARTIAL` | 已落地，尚有 73 个 B-deck token 未映射到 GroundTruth |
| `verification_targets` | IBTrACS 聚合真值（回退） | `GroundTruth_Cyclones/matched_cyclone_tracks.csv` + `by_storm/*/ibtracs_best_track.csv` | `READY` | 已可作为 B-deck 缺失 storm 的回退验证来源 |
| `verification_targets` | 优先验证真值（主用） | `by_storm/*/verification_groundtruth_preferred.csv` | `READY` | 已执行 `B-deck > IBTrACS` 的 storm 级优先策略 |
| `now_inputs` 证据层 | GOES 红外结构化观测 | `data/interim/goes/goes_observation_features_full.csv` | `PARTIAL` | 已接入并完成严格 QC 重跑，当前覆盖 `3230/7085`；缺口由 `no_goes_image_in_window` 与 `invalid_goes_temperature_metrics_after_qc` 构成 |
| `now_inputs` 证据层 | ASCAT 结构化观测 | 当前仓库无结构化观测表；Sherlock Slurm 提交入口已补齐 | `MISSING` | 待在 Sherlock 执行全量提取并落盘 |
| `now_inputs` 证据层 | Recon 结构化观测 | `data/interim/recon/recon_observation_features_full.csv` | `PARTIAL` | 已有本地全量结果，但覆盖仍不完整 |
| 数据治理 | `storm_id` 跨源映射表 | `data/interim/atcf/storm_id_crosswalk.csv` | `PARTIAL` | crosswalk 已产出，但存在未映射 token（重点在 2025） |

### 12.3 当前最关键缺口（已从“无 ATCF”切换到“补齐覆盖”）
1. `storm_id_crosswalk` 仍有 73 个 B-deck token 未映射（398 总量中已映射 325），这些 storm 当前由 IBTrACS 回退承担验证。  
2. 未映射主要集中在近期年份，2025 年未映射 22 个 token，说明 GroundTruth/命名对齐仍需补链路。  
3. A-deck 共 421 token，而 B-deck 共 398 token，存在 23 个 A-only token；这类样本缺标准 B-deck 验证。  
4. `CDS_real` 仍无显式 `storm_id` 字段，需保持时间+位置匹配并形成稳定产物表。  
5. GOES 证据层已完成“修复+重跑”：`goes_observation_features_full.csv` 当前覆盖 7,085 个样本时次中的 3,230 个（严格 QC 后口径）。  
6. GOES 缺失原因已分层：`invalid_goes_temperature_metrics_after_qc = 2,719`，`no_goes_image_in_window = 1,136`。  
7. 样本 token 治理已落地：`multimodel_guidance_a_deck` 已移除高冗余 `model_ids` 与逐点 `models` 字段，仅保留 `model_count + consensus/spread` 可学习信号。
8. ASCAT/Recon 仍缺结构化输入，观测证据层仍未完全闭环。

### 12.4 B-deck -> GroundTruth 映射覆盖（token 级，2016-2025）

来源：`data/interim/atcf/storm_id_crosswalk.csv`（由 `scripts/organize_atcf_for_workflow.py` 生成）

| 年份 | B-deck token 总数 | 已映射 | 未映射 |
|---|---:|---:|---:|
| 2016 | 39 | 34 | 5 |
| 2017 | 39 | 33 | 6 |
| 2018 | 42 | 36 | 6 |
| 2019 | 41 | 34 | 7 |
| 2020 | 52 | 43 | 9 |
| 2021 | 40 | 36 | 4 |
| 2022 | 36 | 31 | 5 |
| 2023 | 42 | 36 | 6 |
| 2024 | 34 | 31 | 3 |
| 2025 | 33 | 11 | 22 |
| **合计** | **398** | **325** | **73** |

### 12.5 当前可构造样本规模（保持原口径 + 新增 ATCF能力）
1. 维持原 `EC-only` 口径时，`dataset_v0` 可直接构造 4,027 样本（2016-2025，`±3h`）。  
2. 现在 ATCF 已可提供 `guidance_inputs.multimodel`，verify 已可提供 `verification_targets.preferred(B-deck>IBTrACS)`，但“NOAA issue 时次级覆盖矩阵”尚未批量统计，需在下一步 P0 产出。  
3. 当前 verify 聚合规模：
- `verification_preferred_storm_count = 870`
- `verification_preferred_rows_written = 43,692`
- 行级来源：`atcf_b_deck = 9,899`，`ibtracs_matched_groundtruth = 33,793`
4. 单样本验证已通过：`Atlantic_2020186N30289_2020-07-06T03:00:00Z_007` 中 A-deck、preferred verification、GOES structured observation 均为 `available`。
5. 当前 GOES 缺失原因已分为两类：`invalid_goes_temperature_metrics_after_qc`（2,719 行）与 `no_goes_image_in_window`（1,136 行），后续需分别处理“口径”与“覆盖”问题。

---

## 13. 数据 TODO List（按优先级，已按新状态重排）

### P0（仅基于现有数据，立即可做）
1. 产出 NOAA 时次级覆盖矩阵：
- 维度：`sample_id` x `{ec_guidance, atcf_a_deck, atcf_b_deck, cds_now}`
- 目标：明确“可直接训练/仅SFT可用/仅评估可用/缺块”四类样本数量。

2. 攻关 `storm_id_crosswalk` 未映射 73 token：
- 优先处理 2025 的 22 token；
- 引入 `storm_name + season + basin + time/position` 二级匹配；
- 保留 `unresolved_reason` 字段（如 GroundTruth 暂无、命名冲突、时间偏差）。

3. 固化 A-deck 模型治理规则（训练可用模型白名单）：
- 过滤 `CARQ/WRNG` 与异常条目；
- 统一模型族别名（例如 AVNO/GFSO 等）；
- 给每个 `tau` 计算有效模型数下限（例如 `model_count >= 3`）作为 `guidance_qc`。

4. 将 ATCF 聚合结果接入批量建集（不仅 sample preview）：
- `guidance_inputs`: 读取 `a_deck_guidance.csv + a_deck_spread.csv`
- `verification_targets`: 读取 `b_deck_best_track.csv`
- 对缺失 ATCF 的样本打 `missing_reason`，不丢弃。

5. 持续保留并执行已完成的前置检查（作为 pipeline gate）：
- `basin_scope_check`（CDS 对齐）
- `guidance_data_qc`（轨迹与环境系统可用性）

6. GOES 质量分层与补洞（限范围，避免无限重跑）：
- 将 GOES 缺失拆分为 `no_goes_image_in_window` 与 `invalid_goes_temperature_metrics_after_qc` 两类，分别统计到 year/basin/storm 维度；
- 对 `invalid_*` 做规则诊断（阈值、窗口、空间尺度）后再决定是否放宽 QC；不允许恢复事后 `÷10` 启发式；
- 对 `no_goes_image_in_window` 维持“可训练但缺观测证据”分层，不阻塞主建集。

7. 固化 prompt token 预算策略（ATCF）：
- 保留 `model_count + consensus_* + spread_*`；
- 去除 `model_ids` 与逐时效 `models` 长字符串；
- 在 `source_trace` 保留上游文件路径，保证可追溯。

### P1（需要新增数据请求/补充）
1. **ASCAT/Scatterometer 风场（中高优先）**：
- 作用：补充地面风场观测证据，改善初始强度与风圈判断。
- 验收目标：形成可按 `issue_time` 对齐的结构化表。

2. **Recon/VDM（可得范围内，中优先）**：
- 作用：补充强度/中心定位高价值观测证据（大西洋优先）。
- 验收目标：可追溯到样本 `source_trace` 的结构化记录。

3. **ERA5 原始变量集（pressure + single levels，中优先）**：
- 作用：从原始网格重算 `vws/rh/div/z500/sst/mslp`，提升可复现性。
- 验收目标：每条特征都带 `formula_version + source_dataset_version`。

4. **EC/HRES 2022-2025 补齐（中优先）**：
- 作用：提升 `EC-only` 样本覆盖上限，减少对 GFS 可选分支依赖。

5. **IBTrACS 原始源文件归档与版本锁定（中优先）**：
- 作用：让当前聚合版 `GroundTruth_Cyclones` 有可追溯上游来源，便于审计与复算。
- 验收目标：保留原始文件下载指纹（版本/时间/校验）并可重建 `matched_cyclone_tracks.csv`。

### P2（增强与质量收敛）
1. 建立 `baseline(v0) vs enriched(v1)` A/B 数据集。  
2. 增加自动质检：
- 时间泄漏检查；
- 单位一致性检查；
- 物理范围检查（例如切变、SST、风速阈值）。  
3. 为 RFT 生成候选与奖励数据（官方一致性 + 事后误差 + 物理一致性）。

### 13.4 下一步“明确要补充的数据”清单（执行版）

| 数据 | 作用层 | 当前状态 | 下一步动作 | 完成标志 |
|---|---|---|---|---|
| IBTrACS 聚合表 | `verification_targets` | `READY` | 维持回退链路并跟踪来源版本 | B-deck 缺失样本可稳定回退 |
| GOES 特征表 | `now_inputs` 证据层 | `PARTIAL` | 先按 `invalid_qc`/`no_image` 双原因分层，再做定向补洞与 QC 口径复核 | GOES 缺口和口径问题均可解释，不回退启发式修复 |
| ASCAT 风场表 | `now_inputs` 证据层 | `MISSING` | 建结构化风场摘要 | 可进入 prompt 的观测证据块 |
| Recon/VDM 表 | `now_inputs` 证据层 | `PARTIAL` | 已切换本地执行，默认补全 `REPNT2 REPNT3 AHONT1 AHOPN1` 并同步 canonical 文件 | 大西洋样本已有可用侦察证据行 |
| ERA5 原始变量归档 | `now_inputs` 数值层 | `PARTIAL` | 落盘原始网格并重算特征 | 特征可复算且可追溯 |
| EC 2022-2025 guidance | `guidance_inputs` | `PARTIAL` | 补齐缺失周期 | `EC-only` 样本覆盖提升 |
| IBTrACS 原始文件快照 | 数据治理 | `PARTIAL` | 补齐原始文件和版本指纹 | GroundTruth 聚合可重建 |

### 13.5 样本结构与现状（简版）
1. 当前样本契约仍为：
- `sample_meta / keys / input.now_inputs / input.guidance_inputs / output.official_outputs_noaa / quality_flags`

2. 最新样本文件：
- `data/interim/schema/dataset_v0_1_sample_preview_ec_single_source.json`
- 样本主键：`Atlantic_2020186N30289_2020-07-06T03:00:00Z_007`

3. 该样本当前状态：
- `guidance_inputs.multimodel_guidance_a_deck.status = available`
- `verification_targets.future_best_track_series.status = available`
- `verification_targets.groundtruth_source_policy.preferred_order = [atcf_b_deck, ibtracs_matched_groundtruth]`
- `prompt.now_inputs.observation_evidence_structured.status = available`（GOES 已注入）
- `guidance_inputs.multimodel_guidance_a_deck` 已移除 `model_ids` 与逐点 `models`，降低 token 占用
- 当前剩余缺块主要是 ASCAT 与 Recon 全量年份补齐，以及 GOES 的双类缺口（`invalid_qc` + `no_image`）

### 13.6 下一步决策（2026-03-09）
结论：**GOES 已完成修复与全量重跑，下一步进入“质量分层 + 新数据源补齐”阶段。**

执行建议：
1. GOES 保持严格 QC 口径，先按缺失原因做分层统计与定向诊断，再决定是否调整阈值；不回退启发式修复。  
2. 立即启动 ASCAT（优先）和 Recon（大西洋优先）结构化接入，补齐观测证据层的非卫星维度。  
3. 在批量建集时引入观测证据分层标签（`goes_available / goes_invalid_qc / goes_no_image / ascat_missing / recon_missing`），保证训练与评估可解释。

### 13.7 ASCAT/Recon 补全工作流（执行版，2026-03-11）
目标：按 GOES 既有范式补齐 `now_inputs` 的观测证据层；ASCAT 改为 Sherlock Slurm 主路径，Recon 保持本地直接执行（轻量文本解析）。

执行入口（新增文档）：
- `scripts/README_ascat_recon_pipeline.md`
- `slurm/run_ascat_sherlock_array.slurm`

核心流程：
1. 本地仅生成请求清单：
- `data/interim/ascat/ascat_request_manifest.csv`
- `data/interim/recon/recon_request_manifest.csv`
- Sherlock 若仅通过 `git clone`/`git pull` 获取项目，则需先把 `data/interim/ascat/ascat_request_manifest_full.csv` 纳入版本控制并推送

2. 远端（Sherlock Slurm 单一入口主路径，CDS JupyterLab 备选）完成 ASCAT 特征提取：
- 输入：Copernicus Marine ASCAT L3（NRT + Multi-year）+ 已提交到仓库的 `data/interim/ascat/ascat_request_manifest_full.csv`
- 输出：`data/interim/ascat/ascat_observation_features.csv`
- 汇总：`data/interim/ascat/ascat_observation_features_summary.json`

3. 本地完成 Recon 报文解析（默认路径）：
- 输入：NHC `archive/recon/<year>/` 文本目录（按需抓取 + 本地缓存）
- 执行：`scripts/run_recon_full_controlled.sh`
- 输出：`data/interim/recon/recon_observation_features_full.csv`
- 汇总：`data/interim/recon/recon_observation_features_full_summary.json`
- 限频口径（防止请求过频）：`--sleep-sec >= 0.05`、`--http-sleep-sec >= 0.08`、按年 `cooldown >= 1s`

4. 统一接入建集：
- 将 GOES/ASCAT/Recon 作为三路独立证据源合并；
- `status` 采用 `available / partial_available / missing_real_data`；
- 全程保留 `missing_reason + source_trace`，不丢样本。

5. 失败与覆盖策略：
- 按年份分块（`full_by_year`）+ 断点续跑；
- 不可得样本进入缺失分层，不阻塞主建集。

### 13.8 Recon 本地执行进展（2026-03-18 更新）
1. 已完成改造：
- Recon 提取从“远端运行”改为“本地运行”，执行入口为 `scripts/run_recon_full_controlled.sh`；
- 增加请求限频参数：`--sleep-sec`（行级）、`--http-sleep-sec`（HTTP 级）；
- Recon 默认补全子目录为：`REPNT2 REPNT3 AHONT1 AHOPN1`。
- 若历史 yearly summary 只覆盖了部分子目录，`scripts/run_recon_full_controlled.sh` 会自动识别并对该年份重跑。
- 合并完成后会同步 canonical：
  - `data/interim/recon/recon_observation_features.csv`
  - `data/interim/recon/recon_observation_features_summary.json`

2. 本地连通性烟雾测试（已完成）：
- 输出：`data/interim/recon/recon_observation_features_smoke.csv`
- 汇总：`data/interim/recon/recon_observation_features_smoke_summary.json`
- 结果：`rows_written=20`, `available_rows=14`, `missing_rows=6`

3. Baseline 全量运行结果（`REPNT2/REPNT3` 口径，2016-2025）：
- 全量特征：`data/interim/recon/recon_observation_features_full.csv`
- 全量汇总：`data/interim/recon/recon_observation_features_full_summary.json`
- 汇总统计：`rows_written=6981`, `available_rows=3551`, `missing_rows=3430`

4. 当前最佳补全结果（截至 `2026-03-18`）：
- 已用 `scripts/run_recon_missing_secondary_fill.sh` 对 `AHONT1/AHOPN1` 做 targeted supplement，并将当前最佳结果同步到 canonical；
- 当前 canonical 汇总：`rows_written=6981`, `available_rows=3561`, `missing_rows=3420`, `promoted_rows_total=10`；
- 已完成 supplement 的年份：
  - `2016`: `448 -> 457`（`+9`）
  - `2017`: `413 -> 414`（`+1`）
- 当前 `data/interim/recon/recon_observation_features_full_summary.json` 会标记：
  - `years_using_supplemented_outputs=[2016, 2017]`
  - `years_using_baseline_outputs=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]`

5. Baseline 分年覆盖（`REPNT2/REPNT3` 口径）：
- 2016: `448/791`（`0.566372`）
- 2017: `413/745`（`0.554362`）
- 2018: `392/957`（`0.409613`）
- 2019: `424/673`（`0.630015`）
- 2020: `291/842`（`0.345606`）
- 2021: `355/731`（`0.485636`）
- 2022: `450/656`（`0.685976`）
- 2023: `425/855`（`0.497076`）
- 2024: `317/622`（`0.509646`）
- 2025: `36/109`（`0.330275`）

6. 后续动作：
- 补跑 `AHONT1/AHOPN1`，恢复 HDOB/Dropsonde 扩展特征覆盖；
- 当前默认 `max_candidates_per_request=80`，在 `2016` 本地验证下仅比 `120` 少 `2` 条可用样本，可作为全量补跑默认值；
- 若需进一步追求边际覆盖，可按需提高 `max_candidates_per_request`（如 80 -> 120）做局部增益补跑。
- 对已有 `REPNT2/REPNT3` 基线结果，优先使用 `scripts/run_recon_missing_secondary_fill.sh` 只补 `missing_real_data` 的请求，再合并生成 supplemented 全量产物，避免整年全量重跑。

---

## 14. 参考入口（官方/一手）

注：以下入口于 `2026-03-05` 检索确认可访问，后续如有迁移请在实现前再次校验。

1. ATCF aid_public：<https://ftp.nhc.noaa.gov/atcf/aid_public/>
2. ATCF btk：<https://ftp.nhc.noaa.gov/atcf/btk/>
3. ATCF archive：<https://ftp.nhc.noaa.gov/atcf/archive/>
4. ATCF ABR 格式：<https://science.nrlmry.navy.mil/atcf/docs/database/new/abrdeck.html>
5. CDS API 指南：<https://cds.climate.copernicus.eu/en/how-to-api>
6. ERA5 文档：<https://confluence.ecmwf.int/x/Oi1EDg>
7. AWS NOAA GFS（NODD）：<https://registry.opendata.aws/noaa-gfs-bdp-pds/>
8. AWS NOAA GOES：<https://registry.opendata.aws/noaa-goes/>
9. GEE ERA5 Hourly：<https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_HOURLY>
10. GEE NOAA GFS：<https://developers.google.com/earth-engine/datasets/catalog/NOAA_GFS0P25>
11. GEE OISST：<https://developers.google.com/earth-engine/datasets/catalog/NOAA_CDR_OISST_V2_1>
12. GEE GOES-19 MCMIPC：<https://developers.google.com/earth-engine/datasets/catalog/NOAA_GOES_19_MCMIPC>
13. IBTrACS（NCEI）：<https://www.ncei.noaa.gov/products/international-best-track-archive>
14. WeatherBench2 数据指南：<https://weatherbench2.readthedocs.io/en/latest/data-guide.html>
15. Copernicus Marine ASCAT L3 NRT：<https://data.marine.copernicus.eu/product/WIND_GLO_PHY_L3_NRT_012_002/description>
16. Copernicus Marine ASCAT L3 Multi-year：<https://data.marine.copernicus.eu/product/WIND_GLO_PHY_L3_MY_012_006/description>
17. NHC Recon archive：<https://www.nhc.noaa.gov/archive/recon/>
18. NHC Recon text products：<https://www.nhc.noaa.gov/recon.php>
