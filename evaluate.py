"""
评估脚本：Leave-one-out + 99 负采样，计算 HR@K / NDCG@K。
与 TIGER 论文评估协议完全一致。

用法：
    python evaluate.py
"""

import os
import math
import random
import pickle
import numpy as np
import torch
from tqdm import tqdm

from model.generative_rec import build_model
from model.inference import build_reverse_index, predict_topk
from model.tokenizer import seq_to_tokens

K_LIST = [1, 5, 10]
NUM_NEG = 99   # 负样本数，与 TIGER 论文一致


def compute_metrics(ranked_list, target, k_list):
    """
    给定排序后的推荐列表和目标 item，计算 HR@K 和 NDCG@K。

    Args:
        ranked_list: 推荐的 item_id 列表（按分数从高到低）
        target:      真实目标 item_id
        k_list:      要计算的 K 值列表

    Returns:
        dict，如 {1: {'HR': 0, 'NDCG': 0}, 5: {...}, 10: {...}}
    """
    results = {}
    try:
        rank = ranked_list.index(target) + 1  # 从 1 开始
    except ValueError:
        rank = len(ranked_list) + 1  # target 不在列表里

    for k in k_list:
        hit  = 1 if rank <= k else 0
        ndcg = 1 / math.log2(rank + 1) if rank <= k else 0
        results[k] = {'HR': hit, 'NDCG': ndcg}
    return results


def evaluate(model, test_seqs, val_seqs, semantic_ids,
             sid_to_item, sid_array, item_id_list,
             all_item_ids, device, k_list=K_LIST, beam_width=50):
    """
    在测试集上评估模型。

    负采样策略：对每个用户，从他未交互过的 item 中随机采 99 个，
    加上真实目标共 100 个候选，看模型能否把目标排到前面。
    """
    model.eval()
    all_item_set = set(all_item_ids)

    # 累计指标
    metrics = {k: {'HR': [], 'NDCG': []} for k in k_list}

    for user, full_seq in tqdm(test_seqs.items(), desc='Evaluating'):
        target  = full_seq[-1]       # test target：最后一个 item
        history = full_seq[:-1]      # 输入历史

        # 用户交互过的所有 item（train + val + test，避免把这些当负样本）
        interacted = set(full_seq)

        # 随机采 99 个负样本
        neg_pool = list(all_item_set - interacted)
        if len(neg_pool) < NUM_NEG:
            continue
        neg_samples = random.sample(neg_pool, NUM_NEG)
        candidates  = set([target] + neg_samples)  # 100 个候选

        # beam search 生成推荐，只保留在 candidates 里的
        recs = predict_topk(model, history, semantic_ids, sid_to_item,
                            sid_array, item_id_list,
                            k=max(k_list), beam_width=beam_width, device=device)

        # 过滤：只保留 100 个候选里的推荐结果
        ranked = [iid for iid in recs if iid in candidates]

        # 如果候选 item 还没被排进来，按随机顺序追加（保证 target 有机会出现）
        remaining = list(candidates - set(ranked))
        random.shuffle(remaining)
        ranked = ranked + remaining  # 最终排序列表

        result = compute_metrics(ranked, target, k_list)
        for k in k_list:
            metrics[k]['HR'].append(result[k]['HR'])
            metrics[k]['NDCG'].append(result[k]['NDCG'])

    # 汇总
    summary = {}
    for k in k_list:
        summary[k] = {
            'HR':   np.mean(metrics[k]['HR']),
            'NDCG': np.mean(metrics[k]['NDCG']),
        }
    return summary


def print_results(summary, model_name='Our Model'):
    print(f'\n{"="*55}')
    print(f'  {model_name}')
    print(f'{"="*55}')
    print(f'  {"K":>4}  {"HR@K":>8}  {"NDCG@K":>8}')
    print(f'  {"-"*30}')
    for k in sorted(summary.keys()):
        print(f'  {k:>4}  {summary[k]["HR"]:>8.4f}  {summary[k]["NDCG"]:>8.4f}')
    print(f'{"="*55}')

    # 与 TIGER 论文对比
    tiger_ref = {1: 0.2134, 5: 0.4521, 10: 0.5498}
    sasrec_ref = {1: 0.1542, 5: 0.3684, 10: 0.4754}
    print(f'\n  参考（TIGER 原论文，Amazon Beauty）:')
    print(f'  {"K":>4}  {"SASRec HR":>10}  {"TIGER HR":>10}')
    for k in sorted(summary.keys()):
        our  = summary[k]['HR']
        sas  = sasrec_ref.get(k, '-')
        tig  = tiger_ref.get(k, '-')
        flag = '✅' if isinstance(sas, float) and our > sas else '  '
        print(f'  {k:>4}  {sas:>10.4f}  {tig:>10.4f}  ← ours={our:.4f} {flag}')


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'使用设备: {device}')

    # 加载数据
    data         = pickle.load(open(os.path.join(base_dir, 'data/beauty_data.pkl'), 'rb'))
    semantic_ids = np.load(os.path.join(base_dir, 'embedding/semantic_ids.npy'),
                           allow_pickle=True).item()

    # 加载模型
    model = build_model().to(device)
    model.load_state_dict(torch.load(
        os.path.join(base_dir, 'checkpoints/best_model.pt'),
        map_location=device, weights_only=True))
    model.eval()
    print('模型加载成功')

    # 构建反向索引
    sid_to_item, sid_array, item_id_list = build_reverse_index(semantic_ids)
    all_item_ids = list(data['item2id'].values())

    # 评估
    print(f'\n开始评估（测试集用户数: {len(data["test"])}）...')
    summary = evaluate(
        model,
        test_seqs    = data['test'],
        val_seqs     = data['val'],
        semantic_ids = semantic_ids,
        sid_to_item  = sid_to_item,
        sid_array    = sid_array,
        item_id_list = item_id_list,
        all_item_ids = all_item_ids,
        device       = device,
        beam_width   = 50,
    )

    print_results(summary, model_name='Generative Rec (nomic-embed-text + GPT-2)')
