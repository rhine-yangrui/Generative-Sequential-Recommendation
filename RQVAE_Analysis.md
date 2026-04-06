# RQ-VAE 实现分析

对比对象：本仓库 `embedding/rqvae.py` + `embedding/generate_rqvae_ids.py` vs. TIGER 论文（Rajput et al., NeurIPS 2023）与公开复现 `XiaoLongtaoo/TIGER`（位于 `/mnt/TIGER`，下文简称「参考实现」）。

## 一、当前训练结果速读

```
K_LEVELS = [4, 16, 256]
最优 unique_rate = 41.2%
冲突组数 = 2568，涉及 9157 / 12101 个 item
最大冲突组 = 24（c3 容量 512）
recon ≈ 2e-4，rq ≈ 1e-4
所有层 codebook usage ≈ 100%
```

表面上「码本全用上了、loss 很低」，但 **unique_rate 41%** 是一个非常糟糕的数字。TIGER 原文与参考实现在 Beauty 上的冲突率通常 < 10%（unique_rate > 90%）。也就是说：**码本没有崩塌，但语义粒度根本不够把 12101 个 item 区分开**。这是本实现最关键、也是最容易忽视的问题。

## 二、关键问题（按严重程度）

### 1. 码本尺寸 [4, 16, 256] 在数学上就不可能不冲突 ★★★★★

`4 × 16 × 256 = 16384` 个组合，看上去比 12101 多。但 RQ-VAE 是**残差量化**：第一层把全空间硬切成 4 块，每块平均 ≈ 3025 个 item，要被后面 `16 × 256 = 4096` 个桶装下；只要某一个 L0 cell 略大一点，或后两层不是完美均匀，就一定大量碰撞。

参考实现 / TIGER 论文里 Beauty 用的是 `[256, 256, 256]`（≈ 16M 组合），即使 c3 仍然要做碰撞解决，但碰撞规模远小于现在的 2568 组。

`improve_generative_prompt.md` 里坚持的 `4/16/256` 语义层级，其实是把 TIGER 论文中 *appendix 里某些数据集（比如 ML-1M）的较小码本* 误移植到了 Beauty 上。**这个选择和数据集规模不匹配。**

**修复方向（按代价由低到高）：**

1. 直接切到 `[256, 256, 256]`，对齐参考实现，是最稳的一步；c3 仍然保留作为最后的兜底。
2. 如果还想保留「顶层 = 粗类别」的可解释性，可以试 `[32, 256, 256]` 或 `[16, 256, 256]`，但要重新评估冲突率。
3. `[4, 16, 256]` 应明确定性为 *不可用* 的设定，文档不再宣称它是「目标层级」。

### 2. RQ loss 在层间是 **求和** 而不是 **平均** ★★★

```python
# rqvae.py
rq_loss = rq_loss + (residual.detach() - e).pow(2).mean()
rq_loss = rq_loss + BETA * (residual - e.detach()).pow(2).mean()
```

```python
# 参考实现 rq.py
mean_losses = torch.stack(all_losses).mean()
```

后果：本实现里 rq_loss 数值是参考实现的 ~3 倍，等价于把 `quant_loss_weight` 偷偷设成了 3。这会让 encoder 过度迁就 codebook，而 recon_loss 又很容易降到 1e-4，于是 encoder 学到一个**几乎所有 item 都挤在一起的低方差潜空间**（recon 已经够好，没动力把它们拉开）。这是为什么码本「全用上」却「区分不出来」的次要原因之一。

**修复**：把 rq_loss 改成各层平均，或显式加 `quant_loss_weight` 参数对齐参考实现。

### 3. Encoder warmup 的副作用 ★★★

参考实现 / TIGER 没有「先无量化训 50 epoch」这一步。它们用 `kmeans_init` 在第一次 forward 时直接对 encoder 当时的输出做 k-means 初始化，然后端到端联合训练。

本实现的 warmup 在 L2-normalized 输入 + 无 BN + 仅 MLP 的情况下，会让 encoder 收敛到一个 *recon 已经几乎为 0* 的近似恒等映射。这意味着：

- k-means init 之后的潜空间方差很小，所有 item 已经挤在一起；
- 进入主训练阶段时 codebook 想要重新「拉开」item 已经晚了，recon_loss 没有梯度信号去推 encoder。

观察到的「recon 从第 1 epoch 就是 2e-4 几乎不动」正是这个症状。

**修复**：要么取消 warmup 直接 joint train（推荐，与参考实现一致），要么把 warmup 缩短到 5–10 epoch 并加一点正则（dropout / weight decay 增大）。

### 4. 残差更新使用 `e.detach()` 而非 straight-through 值 ★★

```python
# 本实现
residual = residual - e.detach()
```

```python
# 参考实现
x_res, loss, indices = quantizer(residual, use_sk=use_sk)   # x_res 已是 straight-through
residual = residual - x_res
```

两者前向数值相同；区别在反向传播：参考实现的写法让残差链路能把梯度回传到 encoder，本实现等于在每一层之间切断了梯度。这使得**只有第一层的量化误差才会真正训练 encoder**，更深层的 codebook 训练信号完全是孤立的。这是比 (2) 更隐蔽的次要原因。

**修复**：保留 `quantize_level` 内部已有的 straight-through `e_st`，用 `residual = residual - e_st`（或像参考实现一样在外层单独维护 `x_res`）。

### 5. 训练 epoch 太短 ★

参考实现默认 3000 epoch，本实现 50 + 300。对 12k item 的小数据集，1k+ epoch 是常态。但在 (1)(2)(3)(4) 没修之前，多训也救不回来。

### 6. Sinkhorn 配置 ★

本实现 `[0.0, 0.003, 0.003]`，参考实现 `[0.0, 0.0, 0.003]`（只在最后一层启用）。两种都见过，不是错，但与参考实现略有出入，做对比时记得对齐。

## 三、`generate_rqvae_ids.py` 是否合理？

整体合理。要点：

- ✅ 从最优 checkpoint 加载、`use_sk=False` 走 argmin，正确。
- ✅ 用 c3 做 collision-resolution，最大组 24 < 容量 512，安全。
- ⚠️ c3 = 0 的语义不再「无冲突」——只要属于碰撞组就会被分配 0..N-1，这点要在 tokenizer / inference 端确认是否一致。
- ⚠️ 当前 c3 实际只用到 0..23，`tokenizer.K_LEVELS[3] = 512` 浪费了大量 vocab。可以等 RQ-VAE 修好后把这个数字调到 max_group + buffer（例如 32 或 64）以减小 GPT-2 词表。

## 四、`improve_generative_prompt.md` 是否还有用？

**结论：基本完成，建议归档/删除，仅保留一条遗留 TODO。**

逐项核对其 Acceptance Criteria：

| 条目 | 状态 |
|---|---|
| 1. `semantic_ids_rqvae.npy` 成功生成 | ✅ |
| 2. 每个 item 都有唯一 ID | ✅（12101 / 12101） |
| 3. `train.py` 端到端跑通 | ✅（依据 README / Progress） |
| 4. `evaluate.py` 端到端跑通 | ✅ |
| 5. 重跑不依赖幸运 seed | ⚠️ 部分——目前结果稳定（loss/usage 都收敛），但**质量差**（unique_rate 41%），并不是 seed 问题 |

「Recommended Next Tasks」里大部分项目也已经做完：

- 训练与生成已拆分（`rqvae.py` + `generate_rqvae_ids.py`）✅
- 用 collision 类指标选 checkpoint（`unique_rate` = 1 - collision_rate）✅
- AdamW + cosine LR + grad clip ✅
- 4-token tokenizer / 受限 beam search / Recall@10 早停 ✅（依据 CLAUDE.md）

**唯一仍未解决的，是这份文档自己反复强调的「不要照搬码本默认值，坚持 4/16/256」——而本次分析的结论恰恰相反：4/16/256 就是当前糟糕结果的根本原因。**

所以这份 prompt 不仅过时，而且其核心指令是错的，留着会误导后续工作。建议替换为一条简短的遗留 TODO。

## 五、建议的下一步动作（优先级排序）

1. **改 K_LEVELS 为 `[256, 256, 256]`**（同时改 `model/tokenizer.py`、`generate_rqvae_ids.py`、`COLLISION_K`、文档）。
2. 把 RQ loss 改成层间平均，或显式加 `quant_loss_weight=1.0`。
3. 把残差更新改为 straight-through 值 (`residual = residual - e_st`)。
4. 取消 warmup（或大幅缩短），与参考实现对齐 joint training + lazy k-means init。
5. 训练 1500–3000 epoch，每 50 epoch 评一次 collision_rate，保存最优。
6. 重新跑 `train.py` / `evaluate.py`，对比 SASRec baseline 与「random ID」消融。
7. 收敛后再把 `c3` 的 vocab 容量从 512 下调到 max_group + buffer。

## 六、文档清理建议

- `improve_generative_prompt.md`：删除或缩成一条「遗留 TODO：码本尺寸 + RQ loss 聚合 + 残差 straight-through，参见 RQVAE_Analysis.md」。
- `README.md`：确认 K_LEVELS / 文件名描述与代码一致（CLAUDE.md 已经要求这点）。
- `Progress.md`：保留作为实验台账，把本次 41% unique_rate 的结果与 K_LEVELS 设定一起记录在案，避免后续误以为是 seed 问题。
- `CLAUDE.md`：本次结论不影响其规则，无需改动。
