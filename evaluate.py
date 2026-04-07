"""
评估脚本：All-rank 协议，与 TIGER (NeurIPS 2023) 评估方式一致。

对每个测试用户，beam search 直接生成 top-K 推荐，
不做负采样，直接计算 Recall@K 和 NDCG@K。

与旧版 99 负采样协议的区别：
  旧协议（99-neg HR@K）：beam search 结果和 100 个候选取交集，
    对生成模型不公平（beam 未命中 target 时 target 被随机追加）。
  新协议（all-rank Recall@K）：beam search 结果直接看 target 是否在内，
    天然匹配生成模型的工作方式，与 TIGER 论文完全一致。

用法：
    python evaluate.py
"""

import os
import math
import pickle
import numpy as np
import torch
from tqdm import tqdm

from model.generative_rec import build_model
from model.inference import build_reverse_index, predict_topk_batch

K_LIST     = [5, 10]    # 与 TIGER 论文一致：Recall@5, @10, NDCG@5, @10
BEAM_WIDTH = 50         # beam search 宽度，越大越准但越慢
BATCH_SIZE = 256        # 批量 beam search（A100 40GB 够；OOM 就降到 128）
ACTIVE_SEMANTIC_IDS = 'semantic_ids_rqvae_3kep.npy'


def compute_metrics(recommended_items, target, k_list):
    """
    All-rank 评估指标。

    Args:
        recommended_items: beam search 生成的推荐列表（已按分数排序）
        target: 真实目标 item_id
        k_list: K 值列表

    Returns:
        dict, {k: {'Recall': 0/1, 'NDCG': float}}
    """
    results = {}
    for k in k_list:
        topk = recommended_items[:k]
        if target in topk:
            rank = topk.index(target) + 1  # 从 1 开始
            results[k] = {'Recall': 1, 'NDCG': 1 / math.log2(rank + 1)}
        else:
            results[k] = {'Recall': 0, 'NDCG': 0.0}
    return results


def evaluate(model, test_seqs, semantic_ids, sid_to_item, sid_array,
             item_id_list, device, k_list=K_LIST,
             beam_width=BEAM_WIDTH, batch_size=BATCH_SIZE):
    """
    All-rank 评估：批量化 beam search，不做负采样。
    """
    model.eval()
    metrics = {k: {'Recall': [], 'NDCG': []} for k in k_list}
    max_k   = max(k_list)

    histories_buf = []
    targets_buf   = []
    pbar = tqdm(test_seqs.items(), desc='Evaluating')

    def _flush():
        if not histories_buf:
            return
        results = predict_topk_batch(
            model, histories_buf, semantic_ids, sid_to_item,
            sid_array, item_id_list,
            k=max_k, beam_width=beam_width, device=device,
        )
        for recs, target in zip(results, targets_buf):
            result = compute_metrics(recs, target, k_list)
            for k in k_list:
                metrics[k]['Recall'].append(result[k]['Recall'])
                metrics[k]['NDCG'].append(result[k]['NDCG'])
        histories_buf.clear()
        targets_buf.clear()

    for user, full_seq in pbar:
        histories_buf.append(full_seq[:-1])
        targets_buf.append(full_seq[-1])
        if len(histories_buf) >= batch_size:
            _flush()
    _flush()

    return {k: {m: np.mean(v) for m, v in mv.items()} for k, mv in metrics.items()}


def print_results(summary, model_name='Our Model'):
    print(f'\n{"="*60}')
    print(f'  {model_name}')
    print(f'  评估协议: All-rank (与 TIGER 论文一致)')
    print(f'{"="*60}')
    print(f'  {"K":>4}  {"Recall@K":>10}  {"NDCG@K":>10}')
    print(f'  {"-"*35}')
    for k in sorted(summary.keys()):
        print(f'  {k:>4}  {summary[k]["Recall"]:>10.4f}  {summary[k]["NDCG"]:>10.4f}')
    print(f'{"="*60}')

    # TIGER 论文参考数字（all-rank，Amazon Beauty）
    tiger_ref = {
        'SASRec': {5: (0.0387, 0.0249), 10: (0.0605, 0.0318)},
        'TIGER':  {5: (0.0454, 0.0321), 10: (0.0648, 0.0384)},
    }
    print(f'\n  TIGER 论文参考（all-rank，Amazon Beauty）:')
    print(f'  {"Model":<10}  {"Recall@5":>10}  {"NDCG@5":>10}  {"Recall@10":>10}  {"NDCG@10":>10}')
    for name, ref in tiger_ref.items():
        r5, n5   = ref.get(5,  (0, 0))
        r10, n10 = ref.get(10, (0, 0))
        print(f'  {name:<10}  {r5:>10.4f}  {n5:>10.4f}  {r10:>10.4f}  {n10:>10.4f}')


if __name__ == '__main__':
    np.random.seed(42)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'使用设备: {device}')

    # 加载数据
    data         = pickle.load(open(os.path.join(base_dir, 'data/beauty_data.pkl'), 'rb'))
    # Active Semantic ID file: RQ-VAE on top of nomic embeddings.
    semantic_ids = np.load(os.path.join(base_dir, 'embedding', ACTIVE_SEMANTIC_IDS),
                           allow_pickle=True).item()

    # 加载模型（可通过命令行参数指定 checkpoint：python evaluate.py checkpoints/xxx.pt）
    import sys
    ckpt_rel = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/best_model_t5_200ep.pt'
    ckpt_path = os.path.join(base_dir, ckpt_rel)
    model = build_model().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f'模型加载成功: {ckpt_rel}')

    # 构建反向索引
    sid_to_item, sid_array, item_id_list = build_reverse_index(semantic_ids)

    # 评估
    print(f'\n开始评估（测试集用户数: {len(data["test"])}）...')
    summary = evaluate(
        model,
        test_seqs    = data['test'],
        semantic_ids = semantic_ids,
        sid_to_item  = sid_to_item,
        sid_array    = sid_array,
        item_id_list = item_id_list,
        device       = device,
        beam_width   = BEAM_WIDTH,
    )

    print_results(summary, model_name='Generative Rec (Semantic ID + GPT-2)')
