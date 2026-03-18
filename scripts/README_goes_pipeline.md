# GOES 高性能提取流程（CDS JupyterLab + GEE）

目标：补齐 `now_inputs.observation_evidence_structured`，并把数据传输控制在“仅传特征表”。

## 1. 生成请求清单（本地快速）

```bash
python3 scripts/build_goes_request_manifest.py \
  --year-start 2016 \
  --year-end 2025 \
  --out-csv data/interim/goes/goes_request_manifest.csv
```

产物：`data/interim/goes/goes_request_manifest.csv`

说明：
- 每行包含 `issue_time_utc + storm center(lat/lon)`，直接来自 NOAA `forecast_advisory`。
- 自动使用 `data/interim/atcf/storm_id_crosswalk.csv` 回填 `storm_id`。

## 2. 在 GEE 端提取 GOES 结构化特征（核心）

```bash
python3 scripts/extract_goes_features_gee.py \
  --manifest-csv data/interim/goes/goes_request_manifest.csv \
  --out-csv data/interim/goes/goes_observation_features.csv \
  --summary-json data/interim/goes/goes_observation_features_summary.json \
  --batch-size 200 \
  --window-before-min 90 \
  --window-after-min 30 \
  --inner-radius-km 200 \
  --outer-radius-km 500 \
  --scale-m 4000 \
  --cold-threshold-k 235 \
  --only-with-storm-id
```

产物：
- `data/interim/goes/goes_observation_features.csv`
- `data/interim/goes/goes_observation_features_summary.json`

说明：
- 计算都在 GEE 端执行（`reduceRegion`），下载仅为 CSV 特征表。
- 默认数据源：`NOAA/GOES/16|17|18|19/MCMIPC`（自动校验可用集合）。

## 3. 接入样本构建

```bash
python3 scripts/build_dataset_sample_preview_v0_1.py
```

行为变化：
- 脚本会自动读取 `data/interim/goes/goes_observation_features.csv`。
- 若匹配成功，`prompt.now_inputs.observation_evidence_structured.status=available`。
- 若未匹配或超出 `±3h`，保留 `missing_real_data` 并写明 `missing_reason`。

## 4. 性能建议

- 优先在 CDS JupyterLab 运行脚本，但把重计算留在 GEE（本流程已做到）。
- `--batch-size` 建议 100-300，避免请求过大触发超时。
- 先用 `--max-rows 500` 小批验证，再全量跑。
- 数据传输瓶颈主要在结果 CSV 下载，不建议下载原始 GOES 栅格到本地。
