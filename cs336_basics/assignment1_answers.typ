#import "../template.typ": *

= Problem (unicode1): Understanding Unicode

(a) chr(0) 返回 Unicode 的空字符 NUL（U+0000）。

(b) repr(chr(0)) 显示为转义序列 "\x00"，而直接打印时不可见。

(c) 它会作为字符串中的一个不可见字符存在并影响长度/索引，拼接后打印看起来像“空了一格”，但实际包含了该字符。

= Problem (unicode2): Unicode Encodings

(a) UTF-8 对 ASCII 兼容且英文常见字符通常 1 字节/字符；UTF-16/UTF-32 至少 2/4 字节，带来大量 0 字节与端序问题，字节级 BPE 处理成本更高。

(b)
```shell
>>> decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
    decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))
  File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
```
中文是多字节字符，逐字节解码会把一个字符拆开导致解码失败或产生错误输出。

(c) 示例：`[0xC0, 0x80]` 是 UTF-8 的过长编码序列，标准明确禁止，无法解码为有效字符。

= Problem (train_bpe): BPE Tokenizer Training

已在 `cs336_basics/bpe.py` 实现 `train_bpe`，支持 `input_path`、`vocab_size`、`special_tokens`，先按特殊 token 分割并用正则预分词，再通过增量更新的 pair 计数完成合并，返回 `vocab` 和 `merges`。

= Problem (train_bpe_tinystories): BPE Training on TinyStories

(a) 训练统计来自 `artifacts/bpe/tinystories_train_stats.json`：耗时【待填】小时，峰值内存【待填】GB，最长 token 为【待填】（【待填】字节），语义上看起来像【待填】，整体合理/不合理【待填】。

(b) Profiling 显示最耗时部分为【待填：预分词 regex/合并更新/IO】。

= Problem (train_bpe_expts_owt): BPE Training on OpenWebText

(a) 最长 token 来自 `artifacts/bpe/owt_train_stats.json`：token【待填】（【待填】字节），语义上看起来像【待填】，是否合理【待填】。

(b) TinyStories 分词器更偏向短句与儿童叙事高频片段；OpenWebText 分词器覆盖更杂的新闻/网页片段，长 token 和罕见子串更多（用统计结果补充【待填】）。

= Problem (tokenizer): Implementing the tokenizer

已在 `cs336_basics/tokenizer.py` 实现 `BPETokenizer`，包含 `__init__`、`from_files`、`encode`、`encode_iterable`、`decode`，并支持将 `special_tokens` 追加进词表；测试接口已接入 `tests/adapters.py`。

= Problem (tokenizer_experiments): Experiments with tokenizers

(a) 压缩比来自 `artifacts/bpe/tokenizer_stats.json`：TinyStories tokenizer on TinyStories 为【待填】bytes/token，OWT tokenizer on OWT 为【待填】bytes/token。

(b) 用 TinyStories 分词器处理 OWT 时压缩比变为【待填】bytes/token，token 更碎/数量更多，反映词表覆盖不足【待填补一句观察】。

(c) 吞吐量来自 `tokenizer_stats.json`：TinyStories tokenizer 为【待填】bytes/s，OWT tokenizer 为【待填】bytes/s；则处理 825GB Pile 约需【待填】小时（按 `825e9 / 吞吐量` 估算）。

(d) uint16 可表示 0–65535 的词表索引，覆盖 10K/32K 词表且内存占用比 int32 少一半，适合大规模序列存储。

= Problem (linear): Implementing the linear module

已在 `cs336_basics/model.py` 实现无 bias 线性层，使用 `torch.matmul(in_features, weights.t())` 支持批量输入。

= Problem (embedding): Implement the embedding module

已在 `cs336_basics/model.py` 实现 `weights[token_ids]` 索引式 embedding。

= Problem (rmsnorm): Root Mean Square Layer Normalization

已在 `cs336_basics/model.py` 实现 RMSNorm：`x / sqrt(mean(x^2) + eps)` 后乘以权重。

= Problem (positionwise_feedforward): Implement the position-wise feed-forward network

已在 `cs336_basics/model.py` 实现 SwiGLU：`silu(xW1) * (xW3)` 后接 `W2` 投影。

= Problem (rope): Implement RoPE

已在 `cs336_basics/model.py` 实现 RoPE：按 `theta` 生成频率，偶数维做旋转变换并保持形状不变。

= Problem (softmax): Implement softmax

已在 `cs336_basics/nn_utils.py` 实现稳定 softmax：先减去 max 再 exponentiate。

= Problem (scaled_dot_product_attention): Implement scaled dot-product attention

已在 `cs336_basics/model.py` 实现 `QK^T/sqrt(d_k)`，支持 mask，softmax 后与 `V` 相乘。

= Problem (multihead_self_attention): Implement causal multi-head self-attention

已在 `cs336_basics/model.py` 实现因果多头注意力：批量 QKV 投影、分头、因果 mask、合并后输出投影。

= Problem (transformer_block): Implement the Transformer block

已在 `cs336_basics/model.py` 实现 pre-norm Transformer block：RMSNorm + 因果 MHA（含 RoPE）+ 残差 + SwiGLU FFN + 残差。

= Problem (transformer_lm): Implementing the Transformer LM

已在 `cs336_basics/model.py` 实现 Transformer LM：token embedding、堆叠 block、最终 RMSNorm、lm head 线性投影输出 logits。

= Problem (transformer_accounting): Transformer LM resource accounting

(a) 参数量与显存占用需按 handout 公式计算，GPT-2 XL 数值填写为【待填】。

(b) 前向传播矩阵乘法与 FLOPs 统计按层拆解填写为【待填】。

(c) FLOPs 主要集中在注意力投影与 FFN 的大矩阵乘法，具体占比【待填】。

(d) 不同 GPT-2 规模的 FLOPs 分布对比【待填】。

(e) context length 增加使注意力部分 FLOPs 近似按 O(n^2) 增长，具体数值【待填】。

= Problem (cross_entropy): Implement Cross entropy

已在 `cs336_basics/nn_utils.py` 实现交叉熵：`logsumexp - target_logit` 的均值形式。

= Problem (learning_rate_tuning): Tuning the learning rate

未在本机运行实验，需在 GPU 上分别测试 1e1/1e2/1e3 并填入结果【待填】。

= Problem (adamw): Implement AdamW

已在 `cs336_basics/optim.py` 返回 `torch.optim.AdamW` 作为实现。

= Problem (adamwAccounting): Resource accounting for training with AdamW

(a) AdamW 峰值内存公式与数值【待填】。

(b) 80GB 显存下 GPT-2 XL 最大 batch size【待填】。

(c) AdamW 单步 FLOPs【待填】。

(d) 在 A100 上训练 GPT-2 XL 时长估计【待填】。

= Problem (learning_rate_schedule): Implement cosine learning rate schedule with warmup

已在 `cs336_basics/optim.py` 实现线性 warmup + 余弦退火调度。

= Problem (gradient_clipping): Implement gradient clipping

已在 `cs336_basics/nn_utils.py` 使用 `clip_grad_norm_` 实现梯度裁剪。

= Problem (data_loading): Implement data loading

已在 `cs336_basics/data.py` 实现 `get_batch`，随机采样起点并返回 `x` 与右移 `y`。

= Problem (checkpointing): Implement model checkpointing

已在 `cs336_basics/serialization.py` 实现 `save_checkpoint` 与 `load_checkpoint`，保存/恢复模型、优化器与迭代数。

= Problem (training_together): Put it together

训练脚本尚未在本仓库实现，需在 GPU 上编写完整训练循环并补充结果【待填】。

= Problem (decoding): Decoding

未实现温度缩放与 top-p 采样解码函数，需补充实现与结果【待填】。

= Problem (experiment_log): Experiment logging

未实现实验跟踪基础设施，需补充实现与结果【待填】。

= Problem (learning_rate): Tune the learning rate

未运行学习率搜索实验，需在 GPU 上完成并填入最终损失与结论【待填】。

= Problem (batch_size_experiment): Batch size variations

未运行 batch size 实验，需在 GPU 上完成并填入结果【待填】。

= Problem (generate): Generate text

未生成文本样例，需在训练完成后生成 ≥256 tokens 并填入结果【待填】。

= Problem (layer_norm_ablation): Remove RMSNorm and train

未做消融实验，需在 GPU 上完成并填入观察结论【待填】。

= Problem (pre_norm_ablation): Implement post-norm and train

未做 post-norm 对比实验，需在 GPU 上完成并填入结果【待填】。

= Problem (no_pos_emb): Implement NoPE

未做 NoPE 消融实验，需在 GPU 上完成并填入结果【待填】。

= Problem (swiglu_ablation): SwiGLU vs. SiLU

未做 SwiGLU/SiLU 对比实验，需在 GPU 上完成并填入结果【待填】。

= Problem (main_experiment): Experiment on OWT

未完成 OWT 训练实验，需在 GPU 上运行并填入结果【待填】。

= Problem (leaderboard): Leaderboard

未进行排行榜实验提交，需按要求优化并填入结果【待填】。
