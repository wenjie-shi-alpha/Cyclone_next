# Dataset V2 Spec

本次数据重构在不修改训练代码的前提下，把现有旧版 raw 样本提升为一个统一的 `canonical v2`，再从同一套 canonical 记录导出多个训练视图。

## 目标

- `canonical v2` 先把 `结果 + 过程 + 解释` 的统一样本结构定住。
- `forecast-only` 继续输出与当前 strict SFT / GRPO 兼容的 JSONL 文件。
- `diagnostic-only` 与 `reasoning-only` 从同一 canonical record 派生，不再各自维护独立样本键。

## 目录结构

重构脚本默认输出到一个新的数据根目录，例如：

```text
data/training_rebuilt_v2_YYYYMMDD_HHMMSS/
  canonical_v2/
    train.jsonl
    val.jsonl
    test.jsonl
    unassigned.jsonl
    schema.json
    build_report.json
  views/
    forecast_only/
      train.jsonl
      val.jsonl
      test.jsonl
      rl_train.jsonl
      rl_val.jsonl
      rl_test.jsonl
      test_anonymous.jsonl
      test_structured_only.jsonl
      test_perturbation.jsonl
      report.json
    reasoning_only/
      train.jsonl
      val.jsonl
      test.jsonl
      report.json
    diagnostic_only/
      train.jsonl
      val.jsonl
      test.jsonl
      report.json
  sft_train.jsonl
  sft_val.jsonl
  sft_test.jsonl
  rl_train.jsonl
  rl_val.jsonl
  rl_test.jsonl
  sft_reasoning_train.jsonl
  sft_reasoning_val.jsonl
  sft_reasoning_test.jsonl
  sft_diagnostic_train.jsonl
  sft_diagnostic_val.jsonl
  sft_diagnostic_test.jsonl
  format_report.json
  dataset_ready_report.json
```

根目录保留 `sft_*.jsonl` / `rl_*.jsonl` 兼容文件，后续只需要切数据路径，不需要先改训练代码。

## Canonical V2 样本

每条 canonical 记录至少包含：

- `sample_id`, `storm_id`, `basin`, `issue_time`, `lead_times`, `source_split`
- `time_anchor_complete`, `input_window_spec`
- `inputs.observation_context`
- `inputs.environment_context`
- `inputs.model_guidance`
- `inputs.historical_track_context`
- `targets.official_forecast_table`
- `targets.verification_target`
- `targets.reasoning_text`, `targets.risk_text`
- `diagnostics.*`
- `flags.*`
- `metadata.*`

其中：

- `targets.official_forecast_table` 保持 strict forecast table 文本，便于直接导出 forecast view。
- `targets.reasoning_sections` 保留旧数据里的分段说明，避免信息丢失。
- `diagnostics.*` 目前是 `heuristic_v2` 弱监督结果，来源于结构化环境场、模式分歧和 discussion 文本信号。
- `metadata.quality_flags` 显式记录缺测、QC、潜在泄漏告警和 diagnostics 来源。

## 视图定义

### `forecast_only`

- 输入：与当前 strict forecast prompt 保持一致
- 输出：严格 `Official forecast:` 表格
- 兼容：直接生成根目录 `sft_*.jsonl` 与 `rl_*.jsonl`

### `reasoning_only`

- 输入：原始 context + fixed official forecast
- 输出：解释性文本，不重复 forecast table

### `diagnostic_only`

- 输入：与 forecast-only 相同
- 输出：固定 schema 的 JSON 对象，包含全部 `diagnostics.*` 键

## 构建命令

如果旧版 raw 数据已经存在，直接运行：

```bash
python3 scripts/rebuild_dataset_v2.py \
  --legacy-raw-dir data/training_canonical_20260410_203416/raw \
  --output-dir data/training_rebuilt_v2_20260414_guidancefix
```

如果不显式传 `--legacy-raw-dir`，脚本会自动选择最新的 `data/training_canonical_*/raw`。

## 当前实现边界

- 训练代码未改。
- `forecast_only` 继续复用现有 `dataset_formatter.py` 的 strict formatter，降低回归风险。
- `diagnostics` 已有可导出管线，但标签质量目前属于可追踪的弱监督，不应当视为最终专家标注。
