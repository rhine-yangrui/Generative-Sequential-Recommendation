# Experiment Progress

评估协议：**All-rank Recall@K / NDCG@K**，Leave-one-out，与 TIGER (NeurIPS 2023) 一致。
数据集：Amazon Beauty（22,363 users，12,101 items）。

---

## 固定配置（所有实验共用）

| 组件 | 配置 |
|------|------|
| 数据集 | Amazon Beauty 5-core，leave-one-out split |
| 训练数据增强 | 滑动窗口（每个位置都生成样本，~5-10x） |
| 生成式模型 | GPT-2 decoder-only，从头训练 |
| 模型参数 | n_embd=256，n_layer=4，n_head=4，~3.5M params |
| 优化器 | AdamW，lr=1e-3，weight_decay=0.01 |
| 调度器 | CosineAnnealingLR，eta_min=1e-5 |
| Early stopping | patience=10（每 3 epoch 检查 val loss） |
| Batch size | 128 |
| Max history | 50 items |
| Beam width（评估） | 50（除非特别注明） |

---

## 结果汇总

| # | Semantic ID 方法 | Embedding 模型 | 码本结构 | Epochs | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | 备注 |
|---|----------------|---------------|---------|--------|----------|--------|-----------|---------|------|
| REF | SASRec（TIGER 论文） | — | — | — | 0.0387 | 0.0249 | 0.0605 | 0.0318 | 论文参考值 |
| REF | TIGER（原论文） | — | RQ-VAE | — | 0.0454 | 0.0321 | 0.0648 | 0.0384 | 论文参考值 |
| E1 | K-means + 距离排序 | qwen2:7b (3584d) | 4/64/256 | 30 | 0.0197 | 0.0122 | 0.0322 | 0.0162 | beam=50 |
| E1b | 同上 | 同上 | 同上 | 同上 | 0.0198 | 0.0122 | 0.0330 | 0.0165 | beam=100，提升微乎其微 |
| E2 | K-means + 随机 ID（消融） | — | 4/64/256 | TBD | — | — | — | — | 对照组 |
| E3 | SASRec baseline | — | — | TBD | — | — | — | — | 判别式基线 |
| E4 | RQ-VAE | nomic-embed-text (768d) | 4/16/256 | TBD | — | — | — | — | 计划中 |

---

## 各实验详情

### E1：K-means + 距离排序（qwen2:7b）

**Semantic ID 构建**
- Embedding：qwen2:7b（Ollama 本地），3584 维，~36 min（2 workers）
- 码本：4/64/256，层次化 MiniBatchKMeans
  - L1：全量 k-means，K=4
  - L2：每个 L1 子集，K=64（保证子簇 ≤ 256 items）
  - L3：按到 L2 簇心欧氏距离排序赋序号（零冲突）
- 唯一 Semantic ID：12,101/12,101（零冲突）
- cluster purity（c1 与 L2 品类对齐度）：~70.60%（实测较低）

**训练**
- Epochs：30（CosineAnnealingLR 跑完，val_loss 仍在缓慢下降）
- 最终 val_loss：2.9042，train_loss：2.7110
- 收敛状态：LR 已降至 eta_min=1e-5，继续训练收益极小

**结果**
- beam=50：Recall@10=0.0322，NDCG@10=0.0162
- beam=100：Recall@10=0.0330，NDCG@10=0.0165（beam 不是瓶颈）

**分析**
- 与 TIGER 论文 SASRec 基线（0.0605）相比差约 1 倍
- beam_width 加倍几乎无提升，说明模型生成的候选根本不包含正确 item 的 Semantic ID
- 根本原因：qwen2:7b 作为生成式 LLM，其 hidden state 不如专门训练的 embedding 模型（如 nomic-embed-text）在语义聚类上表现好，导致 cluster purity 低，Semantic ID 语义结构弱

---

### E2：消融实验（随机 ID）

> 待填写

---

### E3：SASRec baseline

> 待填写

---

### E4：RQ-VAE（计划中）

**动机**：k-means 是 post-hoc 聚类，无法端到端优化码本。RQ-VAE 通过重建 loss + commitment loss 联合训练 encoder/codebook/decoder，码本主动学到对语义分组有用的结构，cluster purity 更高，冲突率更低。这是 TIGER 原文的核心贡献。

**计划配置**
- Embedding 输入：nomic-embed-text（768 维，cluster purity 91.65%）
- 码本结构：4/16/256（RQ-VAE 无子簇溢出问题，可回到原文结构）
- 架构：Encoder MLP → 3 层残差量化 → Decoder MLP
- 训练：standalone，先训 RQ-VAE 得到 semantic_ids.npy，再训推荐模型

> 结果待填写
