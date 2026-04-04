# LLM-Enhanced Sequential Recommendation：项目计划

> **写给协作者的说明**
> 这份文档是给之后接手这个项目的 Claude 看的，请完整阅读后再开始工作。
> 项目目标：在四天内完成一个贴近工业实践的推荐系统项目，核心思路来自 RLMRec（WWW 2024），
> 用对比学习对齐 LLM 语义空间与协同过滤空间，作为求职算法实习的简历项目。
>
> **执行人：** Rhine（孟杨瑞），UCLA ECE Master 准研究生，M3 Mac Air 24GB
> **训练环境：** Google Colab（T4/A100）
> **本地环境：** M3 Mac，用于 LLM embedding 推理（Ollama）和数据处理

---

## 一、项目背景与动机

### 问题出发点

传统序列推荐（如 SASRec）依赖 item ID 的 embedding，优点是协同过滤信号强、训练快，缺点是：
- **冷启动问题**：新 item 没有交互历史，ID embedding 随机初始化，效果差
- **语义鸿沟**：两个相似内容的 item（如同类商品），ID embedding 可能毫无关联

纯 LLM 推荐（直接用语义向量做最近邻）又有另一个问题：
- **缺乏协同信号**：LLM 不知道"买了 A 的人也买了 B"这类集体行为规律
- **推理成本高**：大模型在线推理延迟不可接受

**工业界的解法**（字节、阿里 2024 年的核心方向）：把 ID embedding 和 LLM semantic embedding **对齐融合**，两者互补。

### 对应论文

**RLMRec**: "Representation Learning with Large Language Models for Recommendation"
- 发表于 WWW 2024
- 作者团队：香港大学 + 华为诺亚方舟实验室
- 核心贡献：用对比学习（InfoNCE）把 LLM 表示空间和 CF 表示空间对齐
- 论文链接：https://arxiv.org/abs/2310.15950
- 官方代码：https://github.com/HKUDS/RLMRec

> **注意**：我们不是照搬 RLMRec 的代码，而是理解其思路后自己实现关键组件，
> 用 Qwen2.5 替换原论文的 GPT-3.5/text-embedding-ada，使其更贴近国内工业实践。

---

## 二、技术方案

### 整体架构

```
用户历史行为序列 [i₁, i₂, ..., iₙ]
        │
        ▼
┌─────────────────┐        ┌──────────────────────┐
│   SASRec 骨架   │        │  Qwen2.5-7B (离线)   │
│  (ID Embedding) │        │  Item 语义 Embedding  │
│                 │        │  (预先提取，存为文件)  │
└────────┬────────┘        └──────────┬───────────┘
         │                            │
         │      ┌─────────────────────┘
         │      │
         ▼      ▼
   ┌─────────────────┐
   │  InfoNCE 对齐损失 │  ← 核心创新点：让同一 item 的
   │                 │    ID emb 和 LLM emb 在向量空间对齐
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │   融合预测层     │  ← concat 或加权求和
   └────────┬────────┘
            │
            ▼
      推荐结果 HR@10 / NDCG@10
```

### 为什么用 InfoNCE（对比学习）做对齐

InfoNCE loss 的直觉：把同一个 item 的 ID embedding 和 LLM embedding 当作"正样本对"，
其他 item 当作"负样本"，训练目标是让正样本对在 embedding 空间里更靠近：

```python
def alignment_loss(id_emb, llm_emb, temperature=0.1):
    # id_emb:  [B, D]，CF 模型输出的 item 表示
    # llm_emb: [B, D]，Qwen 提取的语义向量（预先存好，训练时直接读）
    id_norm  = F.normalize(id_emb,  dim=-1)
    llm_norm = F.normalize(llm_emb, dim=-1)
    # [B, B] 相似度矩阵
    sim = torch.matmul(id_norm, llm_norm.T) / temperature
    # 对角线是正样本（同一 item），非对角线是负样本
    labels = torch.arange(sim.size(0)).to(sim.device)
    loss = F.cross_entropy(sim, labels)
    return loss
```

这个 loss 会迫使 SASRec 的 item embedding 学习到 LLM 捕捉的语义信息，
从而在保留协同过滤信号的同时获得语义泛化能力。

### 为什么选 Qwen2.5-7B

- 国内阿里出品，ModelScope 直接下载，无需翻墙
- M3 24GB Mac 可以本地跑推理（通过 Ollama）
- 7B 模型的 embedding 质量显著优于小模型
- 面试阿里系公司时有天然的话题关联性

---

## 三、数据集

### 选择 Amazon Beauty

**原因：**
1. 数据量适中（~52,000 条交互，~1,210 个 item），Colab 上训练快
2. 每个 item 有丰富的文本描述（title + description + category），适合 LLM 理解
3. RLMRec 原论文用的就是这个数据集，便于和论文结果直接对比
4. SNAP 官网可直接下载，无需任何账号

**下载地址：**
```
http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz
http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz
```

### 数据处理流程

1. **过滤**：保留交互数 ≥ 5 的用户和 item（5-core 设定，和论文一致）
2. **排序**：按时间戳升序排列每个用户的交互序列
3. **划分**：Leave-one-out——最后一个 item 作测试，倒数第二个作验证，其余训练
4. **序列截断**：用户序列超过 50 个 item 时，取最近 50 个

---

## 四、Day-by-Day 执行计划

---

### Day 1：数据处理 + SASRec 基线

**目标：** 在 Amazon Beauty 上跑通 SASRec，记录 HR@10 和 NDCG@10 作为基线。

**为什么先跑基线：** 基线是整个实验的锚点。如果基线数字和论文差太多，
说明数据处理或训练有问题，要在加入 LLM 组件之前先修正，否则后续实验全部失效。

#### Step 1.1：数据下载与预处理

```python
# data_process.py
import json, gzip, pickle
from collections import defaultdict

def load_reviews(path):
    """读取评论文件，提取 (用户, item, 时间戳)"""
    data = []
    with gzip.open(path, 'rb') as f:
        for line in f:
            d = json.loads(line)
            data.append((d['reviewerID'], d['asin'], d['unixReviewTime']))
    return data

def build_dataset(reviews, min_interactions=5):
    """
    构建用户序列数据集
    - 过滤低频用户和 item（5-core）
    - 按时间排序
    - 生成 item2id 映射（从 1 开始，0 留给 padding）
    """
    # 统计交互频次
    user_cnt = defaultdict(int)
    item_cnt = defaultdict(int)
    for u, i, t in reviews:
        user_cnt[u] += 1
        item_cnt[i] += 1

    # 过滤
    filtered = [(u, i, t) for u, i, t in reviews
                if user_cnt[u] >= min_interactions and item_cnt[i] >= min_interactions]

    # 构建映射
    items = sorted(set(i for _, i, _ in filtered))
    item2id = {item: idx+1 for idx, item in enumerate(items)}  # 0 = padding

    # 按用户分组并排序
    user_seqs = defaultdict(list)
    for u, i, t in sorted(filtered, key=lambda x: x[2]):
        user_seqs[u].append(item2id[i])

    return dict(user_seqs), item2id

# 运行
reviews = load_reviews('reviews_Beauty_5.json.gz')
user_seqs, item2id = build_dataset(reviews)
pickle.dump({'user_seqs': user_seqs, 'item2id': item2id}, open('beauty_data.pkl', 'wb'))
print(f"用户数: {len(user_seqs)}, item 数: {len(item2id)}")
# 预期输出：用户数: ~22,363，item 数: ~12,101
```

#### Step 1.2：SASRec 实现要点

参考实现：https://github.com/pmixer/SASRec.pytorch（干净，易读）

关键超参数（和论文保持一致）：

```python
config = {
    'maxlen': 50,          # 序列最大长度
    'hidden_units': 64,    # embedding 维度
    'num_blocks': 2,       # Transformer 层数
    'num_heads': 1,        # 注意力头数
    'dropout_rate': 0.2,   # Dropout
    'lr': 0.001,           # 学习率
    'batch_size': 128,
    'num_epochs': 200,
    'l2_emb': 0.0,
}
```

**为什么这些超参数：** 这是 SASRec 原论文在 Amazon Beauty 上的设定，
保持一致才能和原论文的数字对比，否则基线数字没有参考价值。

#### Step 1.3：评估函数

```python
def evaluate(model, test_data, item_num, k=10):
    """
    Leave-one-out 评估，负采样 99 个
    对每个测试用户：从所有未交互 item 中随机采 99 个负样本
    在 100 个候选（1 正 + 99 负）里排序，计算 HR@K 和 NDCG@K
    """
    HR, NDCG = [], []
    for user, seq, target in test_data:
        # 负采样
        neg_samples = random.sample(all_items - interacted_items[user], 99)
        candidates = [target] + neg_samples  # 100 个候选

        scores = model.predict(seq, candidates)  # [100]
        rank = scores.argsort(descending=True).tolist().index(0) + 1  # target 的排名

        HR.append(1 if rank <= k else 0)
        NDCG.append(1 / math.log2(rank + 1) if rank <= k else 0)

    return np.mean(HR), np.mean(NDCG)
```

**Day 1 结束标准：**
- SASRec 在 Amazon Beauty 上 HR@10 ≈ 0.57，NDCG@10 ≈ 0.33（参考 RLMRec 论文 Table 2）
- 如果差距超过 5%，检查数据处理是否有问题

---

### Day 2：Qwen2.5 语义 Embedding 提取

**目标：** 为每个 item 提取语义向量，存为文件供后续训练使用。

**为什么离线提取：** LLM 推理很慢，如果每次训练迭代都调用 LLM，
训练会慢几十倍。正确做法是离线一次性提取所有 item 的 embedding 存好，
训练时直接从文件加载，这也是工业界的标准做法。

#### Step 2.1：Prompt 设计

item 的文本信息来自 meta_Beauty.json.gz 里的 title + description + categories。
Prompt 设计要让 LLM 专注于提取语义特征，而不是做推荐：

```python
def build_item_prompt(item_meta):
    title = item_meta.get('title', '')
    description = item_meta.get('description', '')
    categories = ' > '.join(item_meta.get('categories', [[]])[0])

    prompt = f"""Product Information:
Title: {title}
Category: {categories}
Description: {description[:300]}  # 截断避免超出上下文长度

Please represent the semantic meaning of this product for retrieval."""

    return prompt
```

**为什么加最后一句话：** 研究表明显式说明"用于检索"会让 LLM 输出更适合 embedding 的表示，
而不是对话式的回答。这是 E5 和 BGE 系列 embedding 模型的标准做法。

#### Step 2.2：本地用 Ollama 提取

```bash
# 先安装并拉取模型
brew install ollama
ollama pull qwen2.5:7b
```

```python
# extract_embeddings.py
import ollama, pickle, numpy as np
from tqdm import tqdm

def extract_item_embeddings(item_metas, item2id):
    embeddings = {}

    for asin, meta in tqdm(item_metas.items()):
        if asin not in item2id:
            continue

        prompt = build_item_prompt(meta)

        try:
            response = ollama.embeddings(model='qwen2.5:7b', prompt=prompt)
            emb = np.array(response['embedding'])  # shape: (3584,)
            embeddings[item2id[asin]] = emb
        except Exception as e:
            print(f"Failed for {asin}: {e}")
            continue

    return embeddings

item_embeddings = extract_item_embeddings(item_metas, item2id)
np.save('item_llm_embeddings.npy', item_embeddings)
print(f"提取完成，共 {len(item_embeddings)} 个 item")
```

**M3 24GB 性能预估：** Qwen2.5-7B embedding 推理，约 50ms/item，
Amazon Beauty 有约 12,000 个 item，总时间约 10 分钟，完全可接受。

#### Step 2.3：降维（可选但建议做）

Qwen2.5-7B 的 embedding 维度是 3584，而 SASRec 的 hidden_units 是 64，
维度差距过大直接对齐效果不好，建议用 PCA 或一个线性投影层降到 64 维：

```python
# 方案 A：PCA（简单，不需要训练）
from sklearn.decomposition import PCA
emb_matrix = np.stack(list(item_embeddings.values()))
pca = PCA(n_components=64)
reduced = pca.fit_transform(emb_matrix)

# 方案 B：可学习的线性投影层（更灵活，推荐）
# 在模型里加一个 nn.Linear(3584, 64)，和主模型一起训练
self.llm_proj = nn.Linear(3584, hidden_units)
```

**推荐用方案 B**，因为投影层可以在对齐训练中学习到最适合任务的降维方式。

**Day 2 结束标准：**
- item_llm_embeddings.npy 文件存在，shape 为 (num_items, 3584)
- 随机抽查几个 item，语义相近的 item embedding 余弦相似度 > 0.7

---

### Day 3：对齐训练 + 融合模型

**目标：** 把 LLM embedding 对齐到 SASRec 的 ID embedding 空间，训练融合模型。

#### Step 3.1：修改 SASRec 加入对齐损失

```python
class SASRecWithAlignment(nn.Module):
    def __init__(self, config, llm_embeddings):
        super().__init__()
        # 原 SASRec 组件
        self.item_emb = nn.Embedding(config['item_num']+1, config['hidden_units'], padding_idx=0)
        self.pos_emb  = nn.Embedding(config['maxlen'], config['hidden_units'])
        self.transformer_blocks = nn.ModuleList([...])  # 同原始 SASRec

        # 新增：LLM embedding 投影层
        self.llm_emb_matrix = torch.FloatTensor(llm_embeddings)  # 预加载，不参与梯度
        self.llm_proj = nn.Linear(3584, config['hidden_units'])   # 可训练投影

        self.temperature = 0.1  # InfoNCE 温度参数，越小对比越"硬"

    def alignment_loss(self, item_ids):
        """
        对一个 batch 的 item，计算 ID embedding 和 LLM embedding 的 InfoNCE 对齐损失

        为什么在 batch 内做对比：
        batch 内的其他 item 天然就是负样本，不需要额外采样，
        这是 SimCLR/MoCo 的标准做法，简单且有效。
        """
        # ID embedding
        id_emb = self.item_emb(item_ids)                    # [B, D]

        # LLM embedding（经过投影）
        raw_llm = self.llm_emb_matrix[item_ids]             # [B, 3584]
        llm_emb = self.llm_proj(raw_llm.to(id_emb.device)) # [B, D]

        # L2 归一化
        id_norm  = F.normalize(id_emb,  dim=-1)
        llm_norm = F.normalize(llm_emb, dim=-1)

        # [B, B] 相似度矩阵
        sim = torch.matmul(id_norm, llm_norm.T) / self.temperature

        # 对角线是正样本
        labels = torch.arange(sim.size(0)).to(sim.device)
        return F.cross_entropy(sim, labels)

    def forward(self, seq, pos, neg):
        """
        训练时的前向传播
        总损失 = 推荐损失（BPR）+ λ * 对齐损失

        λ 是超参数，建议从 0.1 开始调，
        太大会导致模型忘记协同过滤信号，太小对齐效果弱。
        """
        # 原 SASRec 的 BPR 损失
        seq_output = self.sasrec_forward(seq)  # [B, D]
        pos_emb = self.item_emb(pos)
        neg_emb = self.item_emb(neg)
        rec_loss = self.bpr_loss(seq_output, pos_emb, neg_emb)

        # 对齐损失（只对 positive item 做对齐）
        align_loss = self.alignment_loss(pos)

        return rec_loss + self.lambda_align * align_loss
```

#### Step 3.2：融合策略

预测时，把 SASRec 的用户表示和目标 item 的 LLM embedding 都用起来：

```python
def predict(self, seq, candidates):
    seq_output = self.sasrec_forward(seq)  # [D]，用户表示

    # ID-based 得分
    cand_id_emb  = self.item_emb(candidates)   # [C, D]
    score_id     = torch.matmul(seq_output, cand_id_emb.T)  # [C]

    # LLM-based 得分（用投影后的 LLM embedding）
    cand_llm_raw = self.llm_emb_matrix[candidates]          # [C, 3584]
    cand_llm_emb = self.llm_proj(cand_llm_raw)              # [C, D]
    score_llm    = torch.matmul(seq_output, cand_llm_emb.T) # [C]

    # 加权融合（alpha 是超参数，建议从 0.5 开始）
    score = (1 - self.alpha) * score_id + self.alpha * score_llm
    return score
```

#### Step 3.3：超参数搜索

需要调的超参数只有两个，做一个简单的 grid search：

```
lambda_align ∈ {0.05, 0.1, 0.2}    # 对齐损失权重
alpha        ∈ {0.3, 0.5, 0.7}     # 预测时 LLM 得分权重
temperature  = 0.1                   # 固定不动
```

**在 Colab 上训练，总训练时间预估 2-3 小时（T4），可以并行跑多组。**

**Day 3 结束标准：**
- 融合模型的 HR@10 > SASRec 基线
- 有一组超参数组合效果最好，记录下来

---

### Day 4：消融实验 + GitHub 整理

**目标：** 跑完所有对比实验，整理 GitHub，写好 README。

#### Step 4.1：消融实验矩阵

| 实验 | 目的 |
|------|------|
| SASRec（纯 ID） | 基线，协同过滤能做到多好 |
| LLM emb + KNN（不训练）| 纯语义，没有协同信号 |
| SASRec + concat（无对齐损失） | 验证"对齐损失"是否有效，还是只靠多了维度 |
| **SASRec + LLM 对齐（你的方法）** | 最终方案 |

**为什么要跑"concat 无对齐"这个 ablation：**
如果你直接拼接但不加 InfoNCE loss，效果比对齐版本差，
就证明对比学习对齐这个设计是有实质贡献的，而不只是维度更多带来的提升。
这个消融实验是整个项目最重要的实验设计。

#### Step 4.2：结果分析与可视化

```python
# 用 t-SNE 可视化对齐效果（选 10 个 item 类别，看对齐前后的 embedding 分布）
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 对齐前：ID embedding 和 LLM embedding 分布很散，没有对应关系
# 对齐后：同一 item 的两种 embedding 应该在空间上更接近

# 这个图是面试时很好的展示素材
```

#### Step 4.3：GitHub 结构

```
Recommendation-Retrieval/
├── README.md                    # 项目介绍，包含结果表格和架构图
├── PROJECT_PLAN.md              # 本文件
├── data/
│   ├── data_process.py          # 数据预处理
│   └── beauty_data.pkl          # 处理后的数据（gitignore 原始数据）
├── model/
│   ├── sasrec.py                # 原始 SASRec 实现
│   └── sasrec_align.py          # 加入 LLM 对齐的版本
├── embedding/
│   ├── extract_embeddings.py    # Qwen 提取 embedding
│   └── item_llm_embeddings.npy  # 存储的 embedding（可 gitignore 大文件）
├── train.py                     # 训练入口
├── evaluate.py                  # 评估函数
├── notebooks/
│   └── analysis.ipynb           # 消融实验结果可视化
└── requirements.txt
```

#### Step 4.4：README 关键内容

README 里要包含：
1. 一句话项目介绍
2. 架构图（可以用 ASCII 画，参考本文件开头的图）
3. 实验结果表格（四行四列，见 Day 4 消融矩阵）
4. 快速复现指令（三条命令能跑起来）
5. 参考论文

---

## 五、面试叙事

做完之后，在面试中这样介绍：

> "我在 Amazon Beauty 数据集上复现并改进了 RLMRec（WWW 2024）的核心思路。
> 用 Qwen2.5-7B 离线提取 item 的语义 embedding，设计了基于 InfoNCE 的对比学习损失，
> 把 SASRec 的 ID embedding 空间和 LLM 语义空间对齐，
> 最终 HR@10 比 SASRec 基线提升了 X%，比纯 LLM embedding 方案提升了 Y%。
> 消融实验验证了对比学习对齐这个设计的有效性，而不只是维度增加带来的提升。
> 这个方向和字节的 HLLM、阿里的 LLM4Rec 工作在同一个技术路线上，
> 代表了当前工业界 LLM 增强推荐的主流做法。"

---

## 六、参考资料

| 资源 | 链接 | 用途 |
|------|------|------|
| RLMRec 论文 | https://arxiv.org/abs/2310.15950 | 理解核心思路 |
| RLMRec 代码 | https://github.com/HKUDS/RLMRec | 参考，不直接用 |
| SASRec 代码 | https://github.com/pmixer/SASRec.pytorch | 基线骨架 |
| Amazon Beauty | http://snap.stanford.edu/data/amazon/ | 数据集 |
| Ollama 文档 | https://ollama.com | 本地 LLM 推理 |
| HLLM 论文 | https://arxiv.org/abs/2409.12740 | 了解工业方向 |
| OneRec 论文 | 快手 2025，搜索 "OneRec Kuaishou" | 了解前沿探索 |

---

## 七、给协作 Claude 的注意事项

1. **不要改变评估协议**：必须用 leave-one-out + 99 负采样，否则结果没有可比性
2. **基线必须先跑通**：Day 1 的 SASRec 数字要和论文对上再往下走
3. **LLM embedding 离线提取**：不要在训练循环里调用 Ollama，会慢死
4. **超参数 lambda_align 很敏感**：如果效果不好先检查这个
5. **Rhine 的背景**：有华为双塔+对比学习实习经验，InfoNCE 这块不需要过度解释
6. **时间限制**：四天，不要追求完美，跑出四行消融表格就达标了

---

*文档版本：v1.0 | 创建日期：2026-04-04*
