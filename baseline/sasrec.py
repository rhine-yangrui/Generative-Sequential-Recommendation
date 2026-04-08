"""
SASRec — Self-Attentive Sequential Recommendation (ICDM 2018).

A discriminative baseline used for comparison against the generative model.
The user history is encoded by a Transformer with a causal mask, then a dot
product against the item embedding table produces ranking scores.
"""

import torch
import torch.nn as nn


class SASRec(nn.Module):
    def __init__(self, num_items, hidden_size=64, num_layers=2,
                 num_heads=1, dropout=0.2, maxlen=50):
        super().__init__()
        self.hidden_size = hidden_size
        self.maxlen      = maxlen

        # index 0 is reserved for padding
        self.item_emb = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.pos_emb  = nn.Embedding(maxlen, hidden_size)

        self.emb_dropout = nn.Dropout(dropout)
        self.layer_norm  = nn.LayerNorm(hidden_size, eps=1e-8)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN — more stable to train
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, item_seq):
        """
        Args:
            item_seq: (B, L) item ids, 0 for padding (left-padded).

        Returns:
            seq_emb: (B, hidden_size) — representation of the latest position.
        """
        B, L = item_seq.shape

        positions = torch.arange(L, device=item_seq.device).unsqueeze(0).expand(B, -1)
        x = self.item_emb(item_seq) + self.pos_emb(positions)
        x = self.emb_dropout(self.layer_norm(x))

        causal_mask = torch.triu(
            torch.ones(L, L, device=item_seq.device, dtype=torch.bool), diagonal=1)

        # We deliberately do not use src_key_padding_mask: when a row has very
        # few real items, every key for some query gets masked, the softmax
        # becomes all -inf and produces NaNs. Relying on padding_idx=0 keeps
        # pad-position embeddings as zero vectors and the readout is taken from
        # the last (always-valid) position.
        x = self.transformer(x, mask=causal_mask, is_causal=True)

        seq_emb = x[:, -1, :]   # left-padded: last position is the most recent item
        return seq_emb

    def predict(self, item_seq, candidate_ids):
        """Score a batch of candidate items against the encoded history."""
        seq_emb  = self.forward(item_seq)                                  # (B, H)
        cand_emb = self.item_emb(candidate_ids)                            # (B, C, H)
        scores   = torch.bmm(cand_emb, seq_emb.unsqueeze(-1)).squeeze(-1)  # (B, C)
        return scores
