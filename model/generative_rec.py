"""
生成式推荐模型：T5 encoder-decoder，从头训练（不加载预训练权重）。

对齐 TIGER (NeurIPS 2023) 参考实现 (../TIGER/model/main.py)。

输入：encoder 收到拍平的历史 Semantic ID token 序列
        [c1¹, c2¹, c3¹, c4¹, c1², c2², c3², c4², ..., PAD, PAD]
输出：decoder 自回归生成下一个 item 的 4 个 Semantic ID token
"""

from transformers import T5Config, T5ForConditionalGeneration
from model.tokenizer import VOCAB_SIZE, PAD_TOKEN, K_LEVELS


def build_model(d_model=128, d_ff=1024, num_layers=4,
                num_heads=6, d_kv=64, dropout=0.1):
    """
    构建轻量 T5 推荐模型，对齐 TIGER 论文配置。

    默认参数与 TIGER 参考实现一致：
      d_model=128, d_ff=1024, num_layers=4 (enc) + 4 (dec),
      num_heads=6, d_kv=64, dropout=0.1, feed_forward_proj='relu'

    Returns:
        T5ForConditionalGeneration，未初始化权重（随机）
    """
    config = T5Config(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_decoder_layers=num_layers,
        num_heads=num_heads,
        d_kv=d_kv,
        dropout_rate=dropout,
        feed_forward_proj='relu',
        pad_token_id=PAD_TOKEN,
        eos_token_id=PAD_TOKEN,            # 不使用 EOS，但 T5Config 必须给一个值
        decoder_start_token_id=PAD_TOKEN,
    )
    return T5ForConditionalGeneration(config)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    import torch

    model = build_model()
    print(f'模型参数量: {count_parameters(model) / 1e6:.1f}M')
    print(f'词表大小:   {model.config.vocab_size}')

    # 构造假 batch 验证 forward pass
    batch_size = 4
    enc_len = 20 * len(K_LEVELS)   # maxlen=20 items × 4 tokens
    dec_len = len(K_LEVELS)        # 目标 4 tokens
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, enc_len))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, VOCAB_SIZE, (batch_size, dec_len))

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print(f'Forward pass 成功')
    print(f'  encoder 输入 shape: {input_ids.shape}')
    print(f'  decoder labels shape: {labels.shape}')
    print(f'  Loss: {outputs.loss.item():.4f}')
    print(f'  Logits shape: {outputs.logits.shape}')
