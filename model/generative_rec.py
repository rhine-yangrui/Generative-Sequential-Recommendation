"""
生成式推荐模型：基于 GPT-2 架构，从头训练（不加载预训练权重）。

输入：用户历史行为的 Semantic ID token 序列
      [BOS, c1¹, c2¹, c3¹, c4¹, EOS, c1², c2², c3², c4², EOS, ...]
输出：下一个 item 的 Semantic ID token 序列
"""

from transformers import GPT2Config, GPT2LMHeadModel
from model.tokenizer import VOCAB_SIZE, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


def build_model(n_embd=256, n_layer=4, n_head=4):
    """
    构建轻量 GPT-2 推荐模型。

    参数规模约 10M，适合 Colab T4 训练（~1-2 小时）。
    如果 Colab 提供 A100，可以把 n_embd 调大到 512，n_layer 调到 6。

    Args:
        n_embd:  token embedding 维度（也是每层隐状态的维度）
        n_layer: Transformer block 层数
        n_head:  Multi-head attention 的头数，必须能整除 n_embd

    Returns:
        GPT2LMHeadModel，未初始化权重（随机）
    """
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_positions=512,         # 1 BOS + 5*maxlen items (4 codes + EOS); maxlen=20 → ~106
        bos_token_id=BOS_TOKEN,
        eos_token_id=EOS_TOKEN,
        pad_token_id=PAD_TOKEN,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(config)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    import torch

    model = build_model()
    print(f'模型参数量: {count_parameters(model) / 1e6:.1f}M')
    print(f'词表大小:   {model.config.vocab_size}')
    print(f'最大序列长度: {model.config.n_positions}')
    print()

    # 构造一个假的 batch 验证 forward pass
    batch_size = 4
    seq_len = 26   # BOS + 5 items × 5 tokens(4 codes + EOS) = 26
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    input_ids[:, 0] = BOS_TOKEN  # 第一个 token 是 BOS

    # labels 和 input_ids 相同，GPT-2 内部自动做 shift（预测下一个 token）
    outputs = model(input_ids=input_ids, labels=input_ids)
    print(f'Forward pass 成功')
    print(f'  输入 shape: {input_ids.shape}')
    print(f'  Loss: {outputs.loss.item():.4f}  (随机初始化，期望 ≈ ln({VOCAB_SIZE}) = {__import__("math").log(VOCAB_SIZE):.2f})')
    print(f'  Logits shape: {outputs.logits.shape}  (batch, seq_len, vocab_size)')
