# Unsloth / TRL 训练骨架（SFT -> GRPO）

当前主训练输入目录：
- `data/training_canonical_20260410_203416`

当前代码入口：
- `scripts/train_sft.py`
- `scripts/train_grpo.py`
- `scripts/train_sft_then_grpo.py`
- `scripts/download_modelscope_model.sh`

当前配置：
- `configs/training/sft_gemma_4_e4b_unsloth.yaml`
- `configs/training/grpo_gemma_4_e4b_unsloth.yaml`

## 设计原则

1. 不再复用旧项目的 prompt/response、regex 抽取链、随机 split 逻辑。
2. SFT 直接读取新的 chat 格式 `sft_*.jsonl`。
3. GRPO 直接读取新的 `rl_*.jsonl`，奖励函数只使用结构化 `verification`。
4. GRPO 默认从 SFT 保存出的 LoRA adapter 继续训练。
5. 整个训练栈围绕 `Unsloth + TRL`，但导入做懒加载，便于先把代码结构落稳。

## 目录结构

`cyclone_training/`：
- `config.py`：YAML 配置加载
- `datasets.py`：SFT / RL 数据加载与 chat 渲染
- `modeling.py`：Unsloth 模型与 LoRA adapter 装配
- `rewards.py`：结构化 GRPO reward
- `sft.py`：SFT 训练入口
- `grpo.py`：GRPO 训练入口
- `pipeline.py`：SFT 后自动接 GRPO

## 运行方式

SFT：

```bash
python scripts/train_sft.py --config configs/training/sft_gemma_4_e4b_unsloth.yaml
```

GRPO：

```bash
python scripts/train_grpo.py --config configs/training/grpo_gemma_4_e4b_unsloth.yaml
```

串行跑 SFT -> GRPO：

```bash
python scripts/train_sft_then_grpo.py \
  --sft-config configs/training/sft_gemma_4_e4b_unsloth.yaml \
  --grpo-config configs/training/grpo_gemma_4_e4b_unsloth.yaml
```

如果 `grpo` 配置里的 `adapter_init_path` 为空，pipeline 会自动把 SFT 产出的
`final_adapter` 作为 GRPO 初始 adapter。

## 当前 reward 逻辑

GRPO 奖励不再依赖旧项目的自由文本 regex 提取链，而是：
1. 解析生成结果里的 `Official forecast` 行；
2. 将生成结果严格对齐到目标 forecast slot；
3. 在这些 slot 上用 `verification.future_best_track` 计算 truth-facing 的 track / intensity 奖励；
4. 若输出不满足 strict forecast schema，则直接不给 reward。

## 下一步

1. 选定实际基座模型。
2. 激活 `.venv` 并安装 `requirements-training.txt`。
3. 用 `scripts/download_modelscope_model.sh` 下载 `google/gemma-4-E4B-it`。
4. 先跑一轮小样本 smoke SFT。
5. 再用 smoke adapter 接一轮小步数 GRPO，检查 reward 分布和生成质量。
