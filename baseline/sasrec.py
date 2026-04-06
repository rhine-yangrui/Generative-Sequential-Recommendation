"""
SASRec：Self-Attentive Sequential Recommendation（ICDM 2018）
判别式基线模型，用于和生成式推荐对比。

核心思路：
  - 用户历史序列 → Transformer 编码 → 得到用户表示向量
  - 用户向量和所有候选 item 的 embedding 做点积打分
  - 取 Top-K 作为推荐结果

与生成式模型的根本区别：
  - SASRec 用随机初始化的整数 ID embedding，无语义信息
  - 本项目生成式模型用语义 ID，天然编码了 item 内容
"""

import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(self, num_items, hidden_size=64, num_layers=2,
                 num_heads=1, dropout=0.2, maxlen=50):
        """
        Args:
            num_items:   item 总数（不含 padding，从 1 开始）
            hidden_size: embedding 维度
            num_layers:  Transformer block 层数
            num_heads:   Multi-head attention 头数
            dropout:     dropout 比例
            maxlen:      用户历史序列最大长度
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.maxlen      = maxlen

        # item embedding：index 0 留给 padding
        self.item_emb = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        # 位置 embedding：最多 maxlen 个位置
        self.pos_emb  = nn.Embedding(maxlen, hidden_size)

        self.emb_dropout = nn.Dropout(dropout)
        self.layer_norm  = nn.LayerNorm(hidden_size, eps=1e-8)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, item_seq):
        """
        Args:
            item_seq: (B, L) item ID 序列，0 为 padding

        Returns:
            seq_emb: (B, hidden_size) 序列最后一个有效位置的表示
        """
        B, L = item_seq.shape

        # 位置编码
        positions = torch.arange(L, device=item_seq.device).unsqueeze(0).expand(B, -1)
        x = self.item_emb(item_seq) + self.pos_emb(positions)
        x = self.emb_dropout(self.layer_norm(x))

        # 因果掩码：只能看到当前及之前的位置
        causal_mask = torch.triu(
            torch.ones(L, L, device=item_seq.device, dtype=torch.bool), diagonal=1)

        # 不使用 src_key_padding_mask：当某行真实 item 很少时，
        # pad 位置会出现 "所有 key 都被 mask" 的情况，softmax 全 -inf → NaN，
        # 并通过下一层 attention 的 0*NaN 渗到最后一位。
        # SASRec 原版做法：靠 padding_idx=0 让 pad 位置 embedding 为 0，
        # pad 位置产生的 representation 是噪声但不影响最后一位的读出。
        x = self.transformer(x, mask=causal_mask, is_causal=True)

        # 左对齐后，最后一个位置始终是最近的有效 item
        seq_emb = x[:, -1, :]  # (B, hidden_size)
        return seq_emb

    def predict(self, item_seq, candidate_ids):
        """
        给候选 item 打分。

        Args:
            item_seq:      (B, L) 历史序列
            candidate_ids: (B, C) 候选 item ID，C 个候选

        Returns:
            scores: (B, C) 每个候选的得分
        """
        seq_emb    = self.forward(item_seq)                        # (B, H)
        cand_emb   = self.item_emb(candidate_ids)                  # (B, C, H)
        scores     = torch.bmm(cand_emb, seq_emb.unsqueeze(-1)).squeeze(-1)  # (B, C)
        return scores
