# 台风预报数据集数据提升行动计划（CDS JupyterLab 约束版）

> 版本：v2.0  
> 更新时间：2026-03-02  
> 适用环境：CDS JupyterLab 服务器 + 当前仓库 `TianGong-AI-Cyclone`

## 1. Review 结论与改造原则

### 1.1 对原计划的关键问题复盘
1. 原计划依赖 AWS/FTP/GEE 多平台，和“当前可用算力为 CDS JupyterLab”不一致，落地成本高。  
2. 目标描述偏宏大，但缺少可执行的数据契约、验收指标、失败回放策略。  
3. 与现有代码目录和脚本衔接不够明确，容易出现“计划可讲、工程不可跑”。

### 1.2 本版改造原则
1. **单平台优先**：第一阶段只依赖 CDS 可访问的数据与算力。  
2. **增量注入**：保留现有脚本链路和产物，只做新增特征抽取与对齐。  
3. **小特征回传**：不搬运大体积源数据，只回传标量/短序列特征。  
4. **可追踪可回放**：每个新增特征都保留时间偏差、空间窗口、覆盖标记。  
5. **先可用再扩展**：先完成 P0/P1 特征，后续再评估跨平台补数。

## 2. 当前基线（与仓库代码对齐）

### 2.1 现有主流程（保持不变）
1. `generate_nc_urls.py` 生成 `output/nc_file_urls.csv`（或 `*_new.csv`）。  
2. `src/environment_extractor/cli.py` 执行下载/追踪/环境提取，输出 `track_single/*.csv` 与 `final_single_output/*.json`。  
3. `src/process_environmental_outputs.py` 过滤不可信几何字段，产出 `data/final_single_output_trusted` 与 `data/cds_output_trusted`。  
4. `src/prepare_forecast_samples.py` 聚合多模式与环境信息，产出 `preprocessed_data/matched_samples.jsonl`。  
5. `src/generate_forecast_dataset.py` 组装 SFT/训练样本。

### 2.2 当前数据痛点（聚焦台风预报）
1. 环境指标已提取，但缺少统一的“数值诊断层”（可直接进入推理）。  
2. 多模式差异存在，但缺少标准化 spread 指标（按 lead time 输出）。  
3. 观测侧证据不足（卫星/海洋耦合代理信号不稳定）。

## 3. 数据补充范围（CDS-only）

### 3.1 优先级矩阵

| 优先级 | 数据来源（CDS可取） | 新增特征 | 价值 | 预期难度 |
| --- | --- | --- | --- | --- |
| P0 | 现有轨迹 + ERA5派生 | `track_spread_24/48/72h`、`intensity_spread_24/48/72h`、`persistence_12/24h` | 直接提升不确定性表达 | 低 |
| P0 | ERA5 pressure/single level | `vws_200_850`、`rh_700_500`、`div200`、`sst_local`、`thetae_gradient` | 形成强度变化因果特征 | 中 |
| P1 | CDS海洋/卫星相关产品（按目录可用性） | `sla_positive_flag`、`sst_anomaly`、`surface_wind_max_obs` | 增强 RI 触发证据 | 中-高 |
| P2 | 外部平台（暂缓） | SHIPS/ATCF/GEE 特征 | 提升上限但增加依赖 | 高 |

> 执行约束：本版本以 P0+P1 为交付目标，P2 不纳入本轮里程碑。

### 3.2 新增特征命名与口径（统一规范）
1. 强度环境：`vws_200_850_500km`、`rh_mid_500km`、`div200_500km`、`sst_2deg_mean`。  
2. 海洋热力：`sst_anom_2deg`、`sla_positive_flag`（可选）。  
3. 轨迹与置信度：`track_spread_km_{24,48,72}`、`wind_spread_kt_{24,48,72}`。  
4. 演变趋势：`delta_intensity_12h`、`delta_intensity_24h`、`motion_speed_12h`。  
5. 所有特征以 lead time 为主键，不直接拼大段原始场数据。

## 4. 数据契约（必须执行）

### 4.1 单条特征记录结构

```json
{
  "storm_id": "2021046N06142",
  "model": "GFS",
  "init_time": "2021-02-14T12:00:00",
  "target_time": "2021-02-15T12:00:00",
  "lead_hour": 24,
  "feature_name": "vws_200_850_500km",
  "feature_value": 11.8,
  "feature_unit": "kt",
  "source_dataset": "era5-pressure-levels",
  "source_time": "2021-02-15T12:00:00",
  "delta_t_hours": 0.0,
  "spatial_window": "radius_km=500",
  "coverage_flag": 1,
  "quality_flag": "ok"
}
```

### 4.2 对齐规则
1. 时间锚点：以轨迹点时间为准（T0/T0+6/...）。  
2. 时间容差：默认 `|delta_t_hours| <= 3`；超过阈值标记 `coverage_flag=0`。  
3. 空间窗口：与当前提取逻辑一致（500km 或 2°），写入 `spatial_window`。  
4. 缺测策略：保留记录并设 `coverage_flag=0`，严禁静默丢样本。

## 5. 实施路线（4周）

### Phase A（Week 1）：P0 特征落地与缓存
1. 新建增量缓存目录：`data/enrichment_cache/raw`、`data/enrichment_cache/aligned`。  
2. 产出 `features_p0.parquet`（或 JSONL），包含所有 P0 特征与元数据。  
3. 对每个 `storm_id + init_time + lead_hour` 生成唯一键，支持断点续跑。  
4. 输出覆盖率日报：按特征和 lead time 统计 `coverage_flag`。

### Phase B（Week 2-3）：P1 特征接入（CDS内可得即接）
1. 在 CDS JupyterLab 完成目录探测与最小样例抽取（先 1 个台风、3 个起报时次）。  
2. 可用则接入 `sla_positive_flag` 与 `sst_anomaly`；不可用则保留占位字段。  
3. 与 P0 使用同一数据契约，避免后续合并逻辑分叉。

### Phase C（Week 3-4）：样本组装与训练前验证
1. 在 `prepare_forecast_samples.py` 增加可选输入 `--enrichment-file`（新增特征文件）。  
2. 在输出样本中新增 `enrichment_timeline` 字段，不改动原字段语义。  
3. 在 `generate_forecast_dataset.py` 新增“环境诊断摘要 + spread 摘要”文本块。  
4. 形成 A/B 数据集：`baseline` 与 `enriched`，进入同一训练评估流程。

## 6. 质量门禁与验收标准

### 6.1 数据层验收
1. `coverage_flag=1` 占比：`T0-72h` 区间内 >= 80%。  
2. 时间偏差中位数：`median(|delta_t_hours|) <= 1.5`。  
3. 关键特征物理范围检查通过率 >= 99%（超界样本写入 `bad_cases.csv`）。  
4. 不可信几何字段清理规则持续生效（沿用 `process_environmental_outputs.py`）。

### 6.2 训练样本层验收
1. Prompt 增量文本长度控制在每样本 +120 至 +220 词。  
2. `model_spread` 描述与数值一致率 100%。  
3. 随机抽检 200 条样本，确保“因果链可读且可追溯到数值字段”。

### 6.3 业务效果验收（离线）
1. 强度变化描述（增强/减弱/持平）与真值趋势一致率提升。  
2. 对“高分歧场景”的不确定性措辞准确率提升。  
3. 失败案例可回放：能定位到源特征、源时间、空间窗口。

## 7. CDS JupyterLab 运行规范

1. 按月份分批处理，避免单次任务超时。  
2. 每批任务写入检查点（已完成 `storm_id + init_time` 清单）。  
3. 中间大文件仅临时保留，最终只回传 `parquet/jsonl` 特征缓存。  
4. 建议每批次结束产出 `run_report.md`（耗时、覆盖率、失败类型）。

## 8. 本周可直接执行的最小闭环（建议）

1. 先做 P0，不等待新数据源，1-2 天可产出第一版 `features_p0`。  
2. 完成 `prepare_forecast_samples.py` 的可选特征注入接口。  
3. 生成 `baseline vs enriched` 两套 `matched_samples.jsonl`。  
4. 抽样检查 50 条 Prompt，确认新增字段可解释后再全量重建。

## 9. 风险与回退

1. CDS目录权限或产品可得性不足：P1 字段置空但保留契约，不阻塞主流程。  
2. 特征覆盖率低于阈值：只启用覆盖率高的特征进入 Prompt。  
3. Prompt 过长导致训练退化：限制每类特征最多展示 2-3 个关键量。  
4. 任一新增模块异常：可完全回退到当前 baseline 数据链路。

---

### 结论
该版本将“台风预报数据集增强”收敛为**CDS 单平台可执行**方案：先做 P0/P1 高价值特征，维持现有代码主干稳定，通过统一数据契约与覆盖率门禁实现可持续迭代。
