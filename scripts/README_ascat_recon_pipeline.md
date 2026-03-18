# ASCAT/Recon 提取流程（ASCAT 远端 + Recon 本地）

目标：补齐 `now_inputs.observation_evidence_structured` 中的 `ASCAT/Recon` 观测证据；ASCAT 保持远端提取，Recon 使用本地低频请求 + 本地解析流程。

## 0. 执行原则（与 GOES 保持一致）

1. ASCAT 维持远端提取（CDS JupyterLab），Recon 允许本地下载并解析文本报文。
2. Recon 本地执行必须开启限频参数，避免请求过于密集。
3. 本地最终保留轻量产物：`manifest.csv`、`features.csv`、`summary.json`。

---

## 1. 统一任务清单（本地控制面）

已实现脚本：`scripts/build_obs_request_manifest.py`

输入：
- 本地 NOAA 时次（`noaa/*/*/*/forecast_advisory/*.txt`）
- `data/interim/atcf/storm_id_crosswalk.csv`

输出：
- `data/interim/ascat/ascat_request_manifest.csv`
- `data/interim/recon/recon_request_manifest.csv`

推荐字段（与 GOES manifest 对齐）：
- `request_id`
- `storm_id`
- `atcf_storm_id`
- `storm_id_match_status`
- `basin`
- `storm_name`
- `advisory_no`
- `issue_time_utc`
- `lat`
- `lon`
- `source_file`

可选过滤字段：
- `is_recon_candidate`（默认 Atlantic + East Pacific）
- `priority_tag`（如 `p1_obs_fill`）

快速命令：
```bash
python3 scripts/build_obs_request_manifest.py \
  --year-start 2016 \
  --year-end 2025 \
  --ascat-out-csv data/interim/ascat/ascat_request_manifest.csv \
  --recon-out-csv data/interim/recon/recon_request_manifest.csv
```

---

## 2. ASCAT 远端提取（Sherlock Slurm 主路径，CDS JupyterLab 备选）

已实现脚本：`scripts/extract_ascat_features_remote.py`

运行位置：
- Sherlock Slurm（全量主路径，推荐）
- CDS JupyterLab（小样本/备选）

上游数据（Copernicus Marine，2026-03-05 可用）：
- NRT: `WIND_GLO_PHY_L3_NRT_012_002`
- Multi-year: `WIND_GLO_PHY_L3_MY_012_006`
- 典型 ASCAT 数据集 ID（示例）：
  - `cmems_obs-wind_glo_phy-ascat-metop_a-l3-pt1h`
  - `cmems_obs-wind_glo_phy-ascat-metop_b-l3-pt1h`
  - `cmems_obs-wind_glo_phy-ascat-metop_c-l3-pt1h`

推荐提取策略：
1. 对每条 `request_id` 取时间窗 `issue_time_utc ± 3h`（可放宽到 `±6h`）。
2. 空间上以风暴中心构建 `inner=200km`、`outer=500km` 同心窗口。
3. 在远端完成栅格到特征聚合，仅导出标量。

推荐输出：
- `data/interim/ascat/ascat_observation_features.csv`
- `data/interim/ascat/ascat_observation_features_summary.json`

推荐字段：
- 标识：`request_id`, `storm_id`, `issue_time_utc`, `obs_time_utc`
- 状态：`ascat_status`, `missing_reason`, `qc_has_data`
- 时间匹配：`obs_offset_minutes`, `obs_offset_abs_minutes`
- 数据源：`ascat_dataset_id`, `ascat_platform`
- 统计特征：
  - `wind_mean_inner_kt`, `wind_p90_inner_kt`, `wind_max_inner_kt`
  - `wind_mean_ring_kt`, `wind_p90_ring_kt`, `wind_max_ring_kt`
  - `wind_area_ge34kt_inner_km2`, `wind_area_ge50kt_inner_km2`
  - `valid_cell_count`

快速命令：
```bash
python3 scripts/extract_ascat_features_remote.py \
  --manifest-csv data/interim/ascat/ascat_request_manifest.csv \
  --out-csv data/interim/ascat/ascat_observation_features.csv \
  --summary-json data/interim/ascat/ascat_observation_features_summary.json \
  --only-with-storm-id
```

Sherlock 提交入口（单一 `slurm` 脚本）：
```bash
sbatch --export=ALL,PROJECT_DIR=/scratch/users/$USER/Cyclone_next,ENV_PREFIX=/scratch/users/$USER/conda/envs/cyclone,ASCAT_CREDENTIALS_FILE=$HOME/.config/copernicusmarine/credentials,YEAR_START=2016,YEAR_END=2025 \
  slurm/run_ascat_sherlock_array.slurm
```

说明：
1. `slurm/run_ascat_sherlock_array.slurm` 现在是单一入口脚本，一个作业内完成：
- 生成 ASCAT full manifest
- 按年切分 manifest
- 逐年请求/提取 ASCAT 特征
- merge yearly outputs
- 同步 canonical 文件
2. 脚本内部仍按年 checkpoint，因此中断后可重提同一个 `slurm` 脚本继续跑。
3. 最终会同步：
- `data/interim/ascat/ascat_observation_features_full.csv`
- `data/interim/ascat/ascat_observation_features_full_summary.json`
- `data/interim/ascat/ascat_observation_features.csv`
- `data/interim/ascat/ascat_observation_features_summary.json`
4. 建议优先使用 `ASCAT_CREDENTIALS_FILE` 或已配置的 Copernicus Marine 默认认证；若必须使用明文环境变量，可在 `sbatch --export` 中传 `ASCAT_USERNAME` / `ASCAT_PASSWORD`。

---

## 3. Recon 本地提取（主路径）

已实现脚本：`scripts/extract_recon_features_remote.py`

运行位置：
- 本地终端（推荐）

上游数据（NHC 官方）：
- 归档根目录：`https://www.nhc.noaa.gov/archive/recon/`
- 年目录示例：`https://www.nhc.noaa.gov/archive/recon/2024/`
- 实况入口：`https://www.nhc.noaa.gov/text/refresh/MIATCXAT2+shtml/`
- VDM 入口：`https://www.nhc.noaa.gov/text/refresh/MIATCXAT3+shtml/`
- Dropsonde 入口：`https://www.nhc.noaa.gov/text/refresh/MIATCDAT3+shtml/`

推荐提取策略：
1. 仅对 `is_recon_candidate=1` 的请求跑 Recon。
2. 时间窗建议 `issue_time_utc - 12h` 到 `issue_time_utc + 3h`。
3. 本地按需下载并解析报文，提取结构化观测。
4. 若同窗多条报文，优先“时间最近 + 质量标记优先”。
5. 为避免请求过频，建议同时设置行级与 HTTP 级 sleep。

推荐输出：
- `data/interim/recon/recon_observation_features.csv`
- `data/interim/recon/recon_observation_features_summary.json`

推荐字段：
- 标识：`request_id`, `storm_id`, `issue_time_utc`, `recon_obs_time_utc`
- 状态：`recon_status`, `missing_reason`, `message_count`
- 报文来源：`recon_message_type`（`VDM/HDOB/DROPSONDE`）, `source_file`
- 关键特征：
  - `vdm_min_slp_mb`
  - `vdm_max_flight_level_wind_kt`
  - `vdm_center_lat`, `vdm_center_lon`
  - `hdob_max_sfmr_wind_kt`
  - `hdob_max_flight_level_wind_kt`
  - `dropsonde_min_slp_mb`

快速命令：
```bash
python3 scripts/extract_recon_features_remote.py \
  --manifest-csv data/interim/recon/recon_request_manifest.csv \
  --out-csv data/interim/recon/recon_observation_features.csv \
  --summary-json data/interim/recon/recon_observation_features_summary.json \
  --only-with-storm-id \
  --subdir REPNT2 \
  --subdir REPNT3 \
  --subdir AHONT1 \
  --subdir AHOPN1 \
  --sleep-sec 0.10 \
  --http-sleep-sec 0.08 \
  --max-candidates-per-request 80
```

---

## 4. GEE 协同路径（可选）

当需要与 GOES/ERA5 做统一时空对齐时：
1. 将 `ascat_observation_features.csv`、`recon_observation_features.csv` 上传为 GEE `FeatureCollection` 资产。
2. 在 GEE 侧做 `request_id`/时空最近邻 join。
3. 导出统一观测证据表（CSV）供本地建集。

说明：
- 该路径仅做“特征表融合”，不要求在 GEE 侧拉 ASCAT/Recon 原始报文。

---

## 5. 年度分块与断点续跑

已实现控制脚本：
- `scripts/run_ascat_full_controlled.sh`
- `scripts/run_recon_full_controlled.sh`
- `scripts/run_recon_missing_secondary_fill.sh`

目录契约（对齐 GOES）：
- `data/interim/ascat/full_by_year/manifests/`
- `data/interim/ascat/full_by_year/features/`
- `data/interim/ascat/full_by_year/summaries/`
- `data/interim/recon/full_by_year/manifests/`
- `data/interim/recon/full_by_year/features/`
- `data/interim/recon/full_by_year/summaries/`
- `data/interim/recon/supplement_secondary_fill/manifests/`
- `data/interim/recon/supplement_secondary_fill/features/`
- `data/interim/recon/supplement_secondary_fill/merged_features/`
- `data/interim/recon/supplement_secondary_fill/merged_summaries/`

控制策略：
1. 先按年拆分 manifest。
2. 每年单独跑提取并写 summary。
3. 年度产物存在即跳过，支持断点续跑。
4. 最后做全量 merge，输出 `*_full.csv` 与 `*_full_summary.json`。
5. 当 baseline 只有 `REPNT2/REPNT3` 时，优先使用 `scripts/run_recon_missing_secondary_fill.sh` 对 `missing_real_data` 行补跑 `AHONT1/AHOPN1`，避免整年全量重跑。

Recon 缺失补全入口：
```bash
RECON_YEAR_START=2016 \
RECON_YEAR_END=2025 \
RECON_SECONDARY_SUBDIRS="AHONT1 AHOPN1" \
RECON_MAX_CANDIDATES=80 \
PROMOTE_TO_CANONICAL=0 \
bash scripts/run_recon_missing_secondary_fill.sh
```

---

## 6. 本地与远端职责边界

本地（控制面 + Recon 计算面）：
1. 生成 `manifest`。
2. 本地执行 Recon 报文抓取与解析（带限频参数）。
3. 提交 ASCAT 远端作业（CDS/GEE）并下载结果。
4. 触发样本构建脚本。

远端（ASCAT 计算面）：
1. 请求 ASCAT 原始数据。
2. 重采样、聚合、质量控制。
3. 输出轻量特征表与统计摘要。

---

## 7. 建集接入建议（与当前代码衔接）

当前 `scripts/build_dataset_sample_preview_v0_1.py` 已有占位块：
- `goes_ascat_recon_structured_obs`

建议演进为三路状态并行：
1. `goes_structured_obs`
2. `ascat_structured_obs`
3. `recon_structured_obs`

合并规则：
1. 每路独立 `status/missing_reason/source_trace`。
2. `observation_evidence_structured.status` 使用：
- `available`：任一路可用；
- `partial_available`：至少一路可用且至少一路缺失；
- `missing_real_data`：三路全缺失。

---

## 8. 最小可执行顺序（建议）

1. 先落地 ASCAT（覆盖范围广、盆地通用）；
2. 再落地 Recon（Atlantic/East Pacific 优先）；
3. 最后做统一观测证据 merge，并接入批量建集；
4. 保留缺失分层标签：`goes_missing/ascat_missing/recon_missing`，不阻塞主建集。

---

## 9. 手动执行入口（ASCAT 远端 + Recon 本地）

ASCAT（CDS JupyterLab）：
1. 小样本连通性检查（推荐先跑）：
```bash
bash scripts/run_obs_smoke_cds_manual.sh 2020 50
```
2. 分年/全量运行：
```bash
bash scripts/run_obs_full_cds_manual.sh 2016 2025 1
```

Recon（本地终端，默认补全版）：
```bash
RECON_PREFETCH_FIRST=0 \
RECON_YEAR_START=2016 \
RECON_YEAR_END=2025 \
RECON_SLEEP_SEC=0.10 \
RECON_HTTP_SLEEP_SEC=0.08 \
bash scripts/run_recon_full_controlled.sh
```

---

## 10. 本地 Recon 进展（2026-03-12）

1. 烟雾测试完成：
- `data/interim/recon/recon_observation_features_smoke.csv`
- `rows_written=20`, `available_rows=14`, `missing_rows=6`

2. 全量合并结果（2016-2025）：
- `data/interim/recon/recon_observation_features_full.csv`
- `data/interim/recon/recon_observation_features_full_summary.json`
- 汇总统计：`rows_written=6981`, `available_rows=3551`, `missing_rows=3430`
 - 运行脚本现会自动同步 canonical：
 - `data/interim/recon/recon_observation_features.csv`
 - `data/interim/recon/recon_observation_features_summary.json`

3. 分年覆盖（`REPNT2/REPNT3`）：
- 2016: `0.566372`
- 2017: `0.554362`
- 2018: `0.409613`
- 2019: `0.630015`
- 2020: `0.345606`
- 2021: `0.485636`
- 2022: `0.685976`
- 2023: `0.497076`
- 2024: `0.509646`
- 2025: `0.330275`

4. 默认补全策略：
- `scripts/run_recon_full_controlled.sh` 现默认使用 `REPNT2 REPNT3 AHONT1 AHOPN1`
- 默认 `max_candidates_per_request=80`，在 `2016` 本地验证中仅比 `120` 少 `2` 条可用样本，但显著降低单请求候选报文数
- 若历史 yearly summary 只覆盖了部分子目录，脚本会自动识别并对该年份重跑

5. 进一步补跑建议（用于提升覆盖与特征维度）：
```bash
RECON_PREFETCH_FIRST=0 \
RECON_CACHE_ONLY=0 \
RECON_SUBDIRS="REPNT2 REPNT3 AHONT1 AHOPN1" \
RECON_YEAR_START=2016 \
RECON_YEAR_END=2025 \
RECON_SLEEP_SEC=0.05 \
RECON_HTTP_SLEEP_SEC=0.08 \
RECON_MAX_RETRIES=1 \
RECON_HTTP_TIMEOUT_SEC=8 \
RECON_MAX_CANDIDATES=120 \
bash scripts/run_recon_full_controlled.sh
```
