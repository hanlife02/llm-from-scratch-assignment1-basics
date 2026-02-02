# 实验执行说明（Assignment 1 需要跑的实验）

本文件只覆盖需要运行实验或训练的部分，理论推导/资源估算类问题不在此列。

## 0. 统一前置步骤

1. 下载数据（TinyStories + OWT）：按 `README.md` 的命令执行。
2. 训练 BPE（已实现脚本，默认有进度条）：
   - TinyStories：`uv run python cs336_basics/scripts/train_bpe.py`
   - OWT：`uv run python cs336_basics/scripts/train_bpe.py --dataset owt`
3. 统计 tokenizer 实验数据：
   - `uv run python cs336_basics/scripts/tokenizer_stats.py`
4. 将训练/验证集 token 化并保存为 `uint16`：
   - 建议实现脚本 `cs336_basics/scripts/tokenize_dataset.py`，读取文本、用对应 tokenizer 编码、将结果按顺序写入 `np.memmap` 或 `.npy`。
   - 目标输出：`data/processed/tinystories_train.npy`、`data/processed/tinystories_valid.npy`、`data/processed/owt_train.npy`、`data/processed/owt_valid.npy`。

## 1. 训练脚本（training_together）

需要实现一个基础训练脚本（建议路径：`cs336_basics/scripts/train_lm.py`），至少支持：
- 模型配置：`d_model`, `num_layers`, `num_heads`, `d_ff`, `context_length`, `rope_theta` 等
- 数据加载：使用 `cs336_basics/data.py` 的 `get_batch` 或基于 `.npy` 的随机采样
- 优化器/调度：AdamW + cosine schedule（已实现函数）
- 梯度裁剪：已实现 `gradient_clipping`
- 验证集评估：固定步数评估并记录 loss
- checkpoint：已实现 `save_checkpoint`/`load_checkpoint`

建议 CLI 形态（你可以按需实现，但尽量保持简单）：
```
uv run python cs336_basics/scripts/train_lm.py \
  --dataset tinystories \
  --train-path data/processed/tinystories_train.npy \
  --valid-path data/processed/tinystories_valid.npy \
  --batch-size 64 --context-length 256 --max-steps 20000 \
  --lr 3e-4 --warmup-iters 1000 --cosine-iters 20000 \
  --save-dir artifacts/checkpoints --run-name base
```

## 2. 学习率调参（learning_rate_tuning）

目标：比较 1e1、1e2、1e3（或你作业要求的范围）：
- 基于同一训练脚本，仅修改 `--lr`。
- 每次训练记录验证集 loss 曲线，选择最佳。
- 输出：哪一个学习率最稳定/最佳。

## 3. batch size 实验（batch_size_experiment）

- 固定模型与学习率，改变 `batch_size`（如 32/64/128）。
- 记录验证集 loss 变化与吞吐量（step/s）。
- 输出：性能与稳定性的对比结论。

## 4. 生成文本（generate）

需要实现解码函数（temperature + top-p），建议文件：
- `cs336_basics/decoding.py`：实现 `sample_next_token` 与 `generate`
- 在训练脚本中新增 `--generate` 模式，载入 checkpoint 生成 ≥256 tokens

## 5. 消融实验

建议统一在模型构造中加开关参数（训练脚本传参）：

- Remove RMSNorm（layer_norm_ablation）：
  - 加 `--no-rmsnorm` 选项，跳过 RMSNorm。
- Post-norm（pre_norm_ablation）：
  - 加 `--post-norm` 选项，将 RMSNorm 放在残差之后。
- NoPE（no_pos_emb）：
  - 加 `--no-rope` 选项，在注意力中跳过 RoPE。
- SwiGLU vs SiLU（swiglu_ablation）：
  - 加 `--ffn-type silu|swiglu` 选项，切换 FFN 结构。

每个消融实验单独训练一次，并比较验证集 loss 与训练稳定性。

## 6. OWT 实验（main_experiment）

- 用 OWT 训练集与验证集跑一遍完整训练。
- 记录验证集 loss，并与 TinyStories 对比。

## 7. Leaderboard 实验

- 在 1.5 小时 H100 限制内，调模型/超参/批量大小以达成目标 loss。
- 记录最终配置、训练时长、验证 loss，并提交结果。

## 8. 实验日志（experiment_log）

最简实现：训练脚本里把每个 step 的 loss、lr、step time 写到 `artifacts/logs/{run}.csv`。
可选：接入 wandb（需登录），记录超参、曲线与模型 artifact。

---

如果你希望我直接把 `train_lm.py` 或 `tokenize_dataset.py` 也补齐，我可以继续在本机生成代码版本。  
