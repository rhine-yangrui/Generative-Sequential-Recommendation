"""
Generative recommender: a small T5 encoder-decoder trained from scratch.

Encoder input  : flattened Semantic ID tokens of the user history
Decoder output : 4 Semantic ID tokens of the next item, generated autoregressively
"""

from transformers import T5Config, T5ForConditionalGeneration

from model.tokenizer import K_LEVELS, PAD_TOKEN, VOCAB_SIZE


def build_model(d_model=128, d_ff=1024, num_layers=4,
                num_heads=6, d_kv=64, dropout=0.1):
    """
    Build the lightweight T5 used in this project (~4.6M parameters).
    All weights are randomly initialised; no pretrained checkpoint is loaded.
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
        eos_token_id=PAD_TOKEN,            # EOS unused; T5Config still requires a value
        decoder_start_token_id=PAD_TOKEN,
    )
    return T5ForConditionalGeneration(config)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    import torch

    model = build_model()
    print(f'Parameters: {count_parameters(model) / 1e6:.1f}M')
    print(f'Vocab size: {model.config.vocab_size}')

    # Dummy forward pass to sanity-check shapes.
    batch_size = 4
    enc_len = 20 * len(K_LEVELS)   # maxlen=20 items × 4 tokens
    dec_len = len(K_LEVELS)
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, enc_len))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, VOCAB_SIZE, (batch_size, dec_len))

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print('Forward pass OK')
    print(f'  encoder input shape : {input_ids.shape}')
    print(f'  decoder labels shape: {labels.shape}')
    print(f'  loss : {outputs.loss.item():.4f}')
    print(f'  logits shape : {outputs.logits.shape}')
