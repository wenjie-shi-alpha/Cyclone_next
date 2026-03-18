# Step 1：NOAA 预报文档解构与预报信息结构化清单

## 0. 目标与范围
本文件用于完成 `idea.md` 中第 1 步任务：
1. 解构 NOAA 气旋预报文档；
2. 提炼预报推理中的高频信息与推理结构；
3. 形成“预报时刻可获知信息”的详尽清单；
4. 严格区分“当下现状数据（now）”与“预测数据（forecast）”。

语料范围（本仓库本地数据）：
- 根目录：`noaa/`
- 文件总量：`71,160`
- `forecast_discussion`：`18,810`
- `forecast_advisory`：`18,505`
- `public_advisory`：`17,273`
- `wind_speed_probabilities`：`16,571`

说明：
- 统计使用大小写不敏感关键词匹配，结果用于识别高频信息和结构，不等同于严格 NLP 语义计数。
- 不同年代模板存在格式差异（老档案与近年文档），因此部分标题/字段覆盖率不是 100%。

---

## 1. NOAA 四类文档的功能分工（预报链路视角）

### 1.1 `forecast_discussion`（核心推理文本）
作用：记录预报员“为什么这样报”的推理过程。  
核心内容：
- 现状诊断：当前强度、位置、运动、结构；
- 证据来源：卫星、Dvorak、侦察机、散射计、雷达、地面/船舶等；
- 环境机理：切变、干空气、海温、脊/槽、引导气流；
- 模式评估：模型分歧、共识、趋势；
- 官方决策：轨迹和强度的调整理由；
- 不确定性与风险沟通（含 Key Messages）。

### 1.2 `forecast_advisory`（结构化官方预报产品）
作用：给出可机器解析的“官方预报数值主干”。  
核心内容：
- 当前中心位置、移动、最低气压、最大风；
- 风圈半径（34/50/64 kt，按四象限）；
- 各时效预报位置和强度；
- 扩展时效（96h/120h）与典型误差说明；
- 下一次公告时间。

### 1.3 `public_advisory`（公众沟通产品）
作用：面向公众和应急体系，强调“影响与行动”。  
核心内容：
- 当前概况摘要（位置、风速、气压、移动）；
- 警报状态（watches/warnings）；
- 对未来走势的简化描述；
- 影响描述（风、雨、风暴增水、龙卷等）；
- 下一次通告时间。

### 1.4 `wind_speed_probabilities`（概率化风险产品）
作用：给出地点级别、阈值级别（34/50/64kt）的发生概率。  
核心内容：
- 分时段起始概率 OP；
- 到指定时效累计概率 CP；
- 多地点概率表（12~120h）。

---

## 2. 高频信息：基于语料的统计证据

## 2.1 `forecast_discussion` 高频关键词（N=18,810）
频率定义：包含该关键词的文件数 / 18,810。

| 关键词 | 文件数 | 覆盖率 | 含义 |
|---|---:|---:|---|
| `shear` | 12,866 | 68.4% | 强度变化关键环境因子 |
| `ridge` | 11,336 | 60.3% | 轨迹引导核心因子 |
| `satellite` | 11,041 | 58.7% | 最常见观测证据来源 |
| `consensus` | 9,214 | 49.0% | 轨迹/强度决策常用锚点 |
| `trough` | 7,694 | 40.9% | 转向、再加速等关键机制 |
| `intensity forecast` | 7,577 | 40.3% | 强度预测显式段落 |
| `dvorak` | 7,287 | 38.7% | 传统卫星强度估计方法 |
| `track forecast` | 6,327 | 33.6% | 轨迹预测显式段落 |
| `extratropical` | 2,860 | 15.2% | 温带转化常见终末路径 |
| `key messages` | 2,651 | 14.1% | 风险沟通模块（较新模板） |
| `landfall` | 2,035 | 10.8% | 登陆相关叙述 |

结论：
- 轨迹与强度推理都强依赖“环境场解释 + 模式共识对比”；
- `shear/ridge/trough` 是推理骨架中的最高频机理词；
- `satellite + dvorak` 是初始强度估计主证据链；
- 风险沟通在近年增强（`key messages`）。

## 2.2 观测与模式信息高频项（`forecast_discussion`）

观测证据相关：

| 关键词 | 文件数 | 覆盖率 |
|---|---:|---:|
| `microwave` | 3,563 | 18.9% |
| `ASCAT` | 2,636 | 14.0% |
| `hurricane hunter` | 1,935 | 10.3% |
| `scatterometer` | 1,383 | 7.4% |
| `flight-level` | 1,359 | 7.2% |
| `SFMR` | 1,178 | 6.3% |

模式与指导相关：

| 关键词 | 文件数 | 覆盖率 |
|---|---:|---:|
| `GFS` | 5,191 | 27.6% |
| `SHIPS` | 5,159 | 27.4% |
| `ECMWF` | 4,218 | 22.4% |
| `UKMET` | 2,348 | 12.5% |
| `HCCA` | 1,961 | 10.4% |
| `HWRF` | 1,930 | 10.3% |
| `LGEM` | 1,493 | 7.9% |
| `HAFS` | 163 | 0.9% |

结论：
- 预报员在文中经常点名模型并解释“为何贴近/偏离某一指导”；
- 模式并非直接输出答案，而是“证据之一”，需结合环境机理和实时观测修正。

## 2.3 其他三类文档的结构稳定度

`forecast_advisory`（N=18,505）：
- `CENTER LOCATED NEAR`：18,490（99.9%）
- `PRESENT MOVEMENT`：18,500（100.0%）
- `MAX SUSTAINED WINDS`：18,499（100.0%）
- `FORECAST VALID`：18,489（99.9%）
- `NEXT ADVISORY AT`：17,530（94.7%）

`public_advisory`（N=17,273）：
- `MAXIMUM SUSTAINED WINDS`：17,244（99.8%）
- `WATCHES AND WARNINGS`：13,219（76.5%）
- `HAZARDS AFFECTING LAND`：12,000（69.5%）
- `KEY MESSAGES`：1,762（10.2%）

`wind_speed_probabilities`（N=16,571）：
- `FORECAST HOUR`：14,074（84.9%）
- `64 KT`：14,074（84.9%）
- `WIND SPEED PROBABILITY TABLE`：13,006（78.5%）

结论：
- `forecast_advisory` 最适合做结构化监督标签（稳定、字段固定）；
- `forecast_discussion` 最适合做推理链数据（非结构化但信息密度高）；
- `public_advisory` 与 `wind_speed_probabilities` 适合补充“风险表达”和“概率标签”。

---

## 3. NOAA 预报专家的典型推理结构（可转化为 reasoning graph）

可抽象为 6 段链路：

1. **现状初始化（Now Analysis）**
- 确定当前中心、强度、移动方向和速度；
- 汇总可用观测证据，给出当前最可信状态估计。

2. **环境机理诊断（Mechanism Diagnosis）**
- 判断引导气流（脊/槽/西风带等）对轨迹的影响；
- 判断切变、海温、干空气、外流、陆地作用对强度的影响。

3. **模式指导比较（Guidance Intercomparison）**
- 对比不同轨迹/强度模型及共识；
- 识别分歧来源（时效、环流结构、环境场不确定性）。

4. **官方轨迹决策（Track Decision）**
- 给出 12~120h 的官方路径；
- 明确“与上一报相比”的调整方向与原因（north/south/faster/slower 等）。

5. **官方强度决策（Intensity/Phase Decision）**
- 给出各时效最大风和相态演变（TS/HU/post-trop/extratrop/dissipated）；
- 显式表达强化/减弱机理与转化条件。

6. **不确定性与风险沟通（Uncertainty + Hazard Messaging）**
- 提示模型离散度、典型误差、不要过度聚焦单一路径；
- 转译为公众风险语言（风暴潮、强风、强降雨、洪水、龙卷等）。

### 3.1 结构化“推理图”建议（节点级）

`证据输入层`
- 观测证据：satellite, dvorak, recon, radar, scatterometer, surface
- 环境证据：shear, SST, moisture, ridge/trough, outflow
- 模式证据：track guidance, intensity guidance, consensus/spread

`诊断层`
- current_state_estimation（现状最佳估计）
- track_mechanism_diagnosis（轨迹机理解释）
- intensity_mechanism_diagnosis（强度机理解释）
- uncertainty_assessment（不确定性评估）

`决策输出层`
- official_track_forecast（12~120h）
- official_intensity_forecast（12~120h）
- phase_transition_forecast（extratrop/post-trop/dissipated）
- hazard_statement（公众风险与行动建议）

---

## 4. 预报时刻信息分层：现状输入 vs 未来可得输入 vs 官方输出

## 4.1 严格分类规则

**A. 现状输入（Now Inputs）**
- 有效时间 `<=` 预报发布时间（issue time）；
- 来自实时观测、近实时分析、前一时次实际状态；
- 可用于“初始化状态估计”。

**B. 未来可得输入（Guidance Inputs）**
- 在 `issue time` 可被预报员获取；
- 有效时间 `>` 预报发布时间；
- 主要是模式/统计-动力指导及其诊断衍生量（如未来切变、未来引导气流、共识/离散度）。

**C. 专家输出（Official Outputs）**
- 由预报员综合 A+B 后形成；
- 包括官方轨迹/强度/相态/风圈、官方概率产品、公众风险通告、推理解释文本；
- 这部分是“预报结果”，不是“当前时次预报前可得输入”。

**D. 跨时次复用说明**
- 当前时次的 `official outputs` 会在下一时次成为可参考的历史输入（如与上一报对比）。
- “过去 6~12h 趋势描述”仍归 `现状输入`（属于 now 分析背景）。

## 4.2 现状输入清单（Now Data Inventory）

| 模块 | 数据项 | 典型来源文档 | 建议字段 | 时间属性 | 分类 |
|---|---|---|---|---|---|
| 基础元信息 | 风暴 ID、名称、盆地、公告号 | 四类文档头部 | `storm_id,name,basin,advisory_no` | issue 时刻 | 现状输入 |
| 基础元信息 | 预报发布时间（UTC/本地） | 四类文档头部 | `issue_time_utc,issue_time_local` | issue 时刻 | 现状输入 |
| 当前位置 | 中心经纬度 | `forecast_advisory`,`public_advisory` | `current_lat,current_lon` | issue 时刻 | 现状输入 |
| 当前位置 | 定位误差半径 | `forecast_advisory` | `position_accuracy_nm` | issue 时刻 | 现状输入 |
| 当前运动 | 运动方向与速度 | `forecast_advisory`,`public_advisory`,`discussion` | `motion_dir_deg,motion_speed_kt` | issue 时刻 | 现状输入 |
| 当前强度 | 最大持续风速 | 四类文档（值可能有 mph/kt 双单位） | `vmax_kt,vmax_mph` | issue 时刻 | 现状输入 |
| 当前强度 | 最低中心气压 | `forecast_advisory`,`public_advisory` | `pmin_mb` | issue 时刻 | 现状输入 |
| 当前结构 | 风圈半径（34/50/64kt，四象限） | `forecast_advisory` | `r34_ne/r34_se/r34_sw/r34_nw` 等 | issue 时刻 | 现状输入 |
| 当前结构 | 海况半径（如 4 m seas） | `forecast_advisory` | `seas4m_ne/...` | issue 时刻 | 现状输入 |
| 观测证据 | 卫星组织、云顶温度、眼墙结构 | `forecast_discussion` | `satellite_structure_text` | issue 近时刻 | 现状输入 |
| 观测证据 | Dvorak 主/客观估计 | `forecast_discussion` | `dvorak_tafb,dvorak_sab,dvorak_obj` | issue 近时刻 | 现状输入 |
| 观测证据 | 侦察机（flight-level, SFMR, dropsonde） | `forecast_discussion` | `recon_obs[]` | issue 近时刻 | 现状输入 |
| 观测证据 | 雷达、散射计、微波、船舶/浮标观测 | `forecast_discussion` | `radar_obs,scat_obs,mw_obs,ship_buoy_obs` | issue 近时刻 | 现状输入 |
| 环境现状 | 垂直风切变 | `forecast_discussion` | `shear_now_kt` 或 `shear_now_desc` | issue 近时刻 | 现状输入 |
| 环境现状 | SST/暖水区/OHC（若有） | `forecast_discussion` | `sst_now_c,ohc_now_desc` | issue 近时刻 | 现状输入 |
| 环境现状 | 干空气/湿度/外流 | `forecast_discussion` | `dry_air_now,humidity_now,outflow_now` | issue 近时刻 | 现状输入 |
| 环境现状 | 脊、槽、西风带、引导流场形势 | `forecast_discussion` | `synoptic_now_text` | issue 近时刻 | 现状输入 |
| 当前风险状态 | 现有 watch/warning 状态 | `public_advisory` | `warnings_current[]` | issue 时刻 | 现状输入 |
| 历史变化 | 与上一报对比（增强/减弱/转向） | `forecast_discussion` | `delta_from_prev` | 相对上一时次 | 现状输入 |

## 4.3 未来可得输入清单（Guidance Inputs Inventory）

| 模块 | 数据项 | 典型来源文档/数据源 | 建议字段 | 时间属性 | 分类 |
|---|---|---|---|---|---|
| 模式轨迹指导 | GFS/ECMWF/UKMET/HAFS 等路径 | `forecast_discussion`（叙述）+ A-deck（结构化） | `model_track_fcst[model][h]` | issue+12h ... +120h | 未来可得输入 |
| 模式强度指导 | SHIPS/LGEM/HWRF/HAFS/HCCA 等 | `forecast_discussion` + A-deck/统计指导文件 | `model_intensity_fcst[model][h]` | issue+12h ... +120h | 未来可得输入 |
| 共识与离散度 | consensus/ensemble spread/envelope | `forecast_discussion` + 模式集合 | `consensus_fcst,spread_metrics` | 未来时效 | 未来可得输入 |
| 未来环境场预报 | 未来切变、SST、湿度、外流、脊槽演变 | `forecast_discussion`（解释文本）+ NWP 场 | `env_guidance[h]` | 未来时效 | 未来可得输入 |
| 登陆窗口先验 | 由模式轨迹推断的登陆时空窗口 | `forecast_discussion` + 模式路径 | `landfall_window_guidance` | 未来时段 | 未来可得输入 |
| 风险先验 | 基于模式场的降雨/风暴潮/大风先验 | `forecast_discussion` + 外部影响模型 | `hazard_guidance[]` | 未来时段 | 未来可得输入 |
| 历史误差统计 | 不同 lead time 典型 track/intensity error | `forecast_advisory` 误差说明 + 历史统计 | `error_climatology` | 与时效相关 | 未来可得输入 |

## 4.4 专家输出清单（Official Outputs Inventory）

| 模块 | 数据项 | 典型来源文档 | 建议字段 | 时间属性 | 分类 |
|---|---|---|---|---|---|
| 官方轨迹预报 | 12/24/36/48/72/96/120h 位置 | `forecast_advisory`,`forecast_discussion`表格 | `track_fcst[h].lat/lon` | issue+12h ... +120h | 专家输出 |
| 官方强度预报 | 各时效最大风/阵风 | `forecast_advisory`,`forecast_discussion` | `intensity_fcst[h].vmax_kt/gust_kt` | issue+12h ... +120h | 专家输出 |
| 官方相态预报 | TS/HU/Post-trop/Extratrop/Dissipated | 同上 | `phase_fcst[h]` | 各未来时效 | 专家输出 |
| 官方风圈预报 | 各时效 34/50/64kt 风圈四象限 | `forecast_advisory` | `wind_radii_fcst[h]` | 各未来时效 | 专家输出 |
| 官方海况预报 | 未来海况半径（若给出） | `forecast_advisory` | `seas_fcst[h]` | 各未来时效 | 专家输出 |
| 官方概率产品 | 地点级 34/50/64kt OP/CP 概率 | `wind_speed_probabilities` | `wind_prob[loc][h][thr]` | 12~120h | 专家输出 |
| 公众风险产品 | watch/warning、影响描述、Key Messages | `public_advisory` | `warnings_fcst,hazard_fcst,key_messages` | 未来时段 | 专家输出 |
| 决策解释文本 | 轨迹/强度/不确定性解释 | `forecast_discussion` | `track_reasoning_text,intensity_reasoning_text,uncertainty_text` | 面向未来 | 专家输出 |

## 4.5 混淆项与判别建议

1. **模型输出（guidance）**
- 在发布时间“可获得”，但内容是未来时效；训练集中应归 `未来可得输入`。

2. **官方预报与输入的关系**
- 官方轨迹/强度/概率产品是本时次 `专家输出`，不是本时次预报前输入。
- 但它们会在下一时次变成可参考历史信息。

3. **未来环境变化来源**
- 讨论文本里的“未来切变变化、脊槽演变、海温影响”等，主要来自数值模式未来场及其诊断产品。
- 预报员会在模型输出基础上加入主观修正，因此文本是“模式指导 + 专家判断”的综合表达。

4. **官方预报表（INIT 行）**
- `INIT` 行是当前分析值，归 `现状输入`；
- `12H` 及之后是未来有效时间，但在数据角色上属于本时次 `专家输出`。

5. **叙述中的“已经发生变化”与“将会发生变化”**
- 过去/现在完成式归 `现状输入`；
- `will/expected/forecast` 引导句需再区分：若是模型依据归 `未来可得输入`，若是官方结论归 `专家输出`。

---

## 5. 建议的数据组织结构（供后续步骤直接使用）

建议采用三层主结构 + 一层跨时次缓存：

1. `now_state`（纯现状）
- 当前中心、强度、结构、观测证据、环境现状、当前警报状态。

2. `guidance_inputs`（预报时刻可见的未来输入）
- 多模式轨迹/强度指导、共识、离散度、未来环境趋势信息。

3. `official_outputs`（专家最终决策）
- 官方轨迹/强度/相态/风圈表；
- 概率产品（地点-阈值-时效）；
- 风险与不确定性沟通文本。

4. `prev_cycle_context`（可选）
- 上一报官方输出及与当前报的差值，用于学习预报员“修正动作”。

对应样例（简化）：

```json
{
  "storm_meta": {
    "storm_id": "AL042025",
    "name": "DEXTER",
    "issue_time_utc": "2025-08-06T09:00:00Z",
    "advisory_no": 10
  },
  "now_state": {
    "center": {"lat": 39.4, "lon": -59.9},
    "motion": {"dir_deg": 60, "speed_kt": 11},
    "intensity": {"vmax_kt": 40, "pmin_mb": 1003},
    "wind_radii": {"34kt": {"ne": 70, "se": 80, "sw": 20, "nw": 60}},
    "evidence": {
      "satellite": "...",
      "scatterometer": "...",
      "dvorak": "...",
      "recon": "..."
    },
    "environment_now": {
      "shear": "strong",
      "synoptic": "southern side of mid-level westerlies"
    }
  },
  "guidance_inputs": {
    "model_track_guidance": {},
    "model_intensity_guidance": {},
    "environment_guidance": {"shear_fcst": "...", "synoptic_fcst": "..."},
    "consensus_aids": ["HCCA", "TVCN"],
    "guidance_spread_text": "..."
  },
  "official_outputs": {
    "track_intensity_table": [
      {"tau_h": 12, "lat": 39.8, "lon": -57.7, "vmax_kt": 45},
      {"tau_h": 24, "lat": 40.6, "lon": -54.1, "vmax_kt": 50}
    ],
    "phase_transition": [{"tau_h": 36, "phase": "POST-TROP/EXTRATROP"}],
    "wind_probabilities": {},
    "hazard_messages": [],
    "decision_rationale_text": "..."
  }
}
```

---

## 6. 本阶段可直接交付的结论（用于 Step 2 对接数据源）

1. **高频推理骨架已明确**：`现状估计 -> 环境机理 -> 模式比较 -> 官方轨迹/强度决策 -> 风险沟通`。  
2. **字段级“现状输入/未来可得输入/专家输出”边界已可执行**：可直接用于标注规范、特征抽取规范和数据表设计。  
3. **四类文档的任务分工清晰**：
- `forecast_discussion`：推理链与决策解释；
- `forecast_advisory`：结构化主标签（轨迹/强度/风圈）；
- `public_advisory`：公众影响表达；
- `wind_speed_probabilities`：概率监督信号。  
4. **下一步（Step 2）可直接做**：把本清单逐项映射到 A-deck/B-deck、ERA5、卫星和再分析数据字段，建立“字段-数据源-脚本”对照表。
