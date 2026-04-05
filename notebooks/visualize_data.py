"""
数据可视化：展示 Amazon Beauty 数据集的处理过程与统计分布
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from collections import Counter, defaultdict
import os

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

data = pickle.load(open(os.path.join(os.path.dirname(__file__), '../data/beauty_data.pkl'), 'rb'))
train     = data['train']
val       = data['val']
test      = data['test']
item2id   = data['item2id']
item_metas = data['item_metas']

# ── 预计算 ──────────────────────────────────────────────────────────────────
train_seq_lens = [len(v)     for v in train.values()]
full_seq_lens  = [len(v)     for v in test.values()]   # 完整序列（含 val/test item）

item_interaction_cnt = Counter()
for seq in test.values():
    for item_id in seq:
        item_interaction_cnt[item_id] += 1

cat2_counts = Counter()
for meta in item_metas.values():
    cats = meta.get('categories', [[]])
    if cats and len(cats[0]) >= 2:
        cat2_counts[cats[0][1]] += 1

prices = []
for meta in item_metas.values():
    p = meta.get('price')
    if p and isinstance(p, (int, float)) and p > 0:
        prices.append(float(p))
prices = np.array(prices)

# 每个用户的 train / val target / test target 分布
train_len_per_user = [len(train[u])      for u in train]
val_target_pos     = [len(val[u]) - 1    for u in val]    # val target 在完整序列里的位置
# ────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor('#f8f9fa')
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

BLUE   = '#4C72B0'
GREEN  = '#55A868'
ORANGE = '#DD8452'
RED    = '#C44E52'
PURPLE = '#8172B2'

# ══════════════════════════════════════════════════════════════════════
# [0,0-2]  数据处理流程图（跨三列）
# ══════════════════════════════════════════════════════════════════════
ax_flow = fig.add_subplot(gs[0, :])
ax_flow.set_xlim(0, 10)
ax_flow.set_ylim(0, 1)
ax_flow.axis('off')
ax_flow.set_facecolor('#f8f9fa')

def draw_box(ax, x, y, w, h, text, color, fontsize=10):
    rect = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.05", linewidth=1.5,
        edgecolor='#555', facecolor=color, alpha=0.85, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white', zorder=4,
            wrap=True, multialignment='center')

def draw_arrow(ax, x1, x2, y=0.5):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle='->', color='#444', lw=2), zorder=5)

boxes = [
    (0.1,  0.25, 1.4, 0.5, 'Raw Data\n198,502 reviews\n22,363 users\n12,101 items',   '#5a6f8a'),
    (2.1,  0.25, 1.4, 0.5, '5-core Filter\n(already applied)\nAll users/items\n≥5 interactions', '#6b8e5e'),
    (4.1,  0.25, 1.4, 0.5, 'Sort by Time\n& Map to\nInteger IDs\n(1-indexed)',         BLUE),
    (6.1,  0.25, 1.4, 0.5, 'Leave-One-Out\nSplit\nper user',                           PURPLE),
    (8.1,  0.25, 1.4, 0.5, 'beauty_data\n.pkl\n✓ Ready',                              GREEN),
]
for (x, y, w, h, text, color) in boxes:
    draw_box(ax_flow, x, y, w, h, text, color, fontsize=9)
for i in range(len(boxes) - 1):
    draw_arrow(ax_flow, boxes[i][0] + boxes[i][2], boxes[i+1][0], y=0.5)

# Leave-one-out 说明标注
ax_flow.annotate(
    'train=[i₁…iₙ₋₂]  val=[i₁…iₙ₋₁]  test=[i₁…iₙ]',
    xy=(7.2, 0.25), xytext=(7.2, 0.07),
    fontsize=8.5, color='#555', ha='center',
    arrowprops=dict(arrowstyle='->', color='#aaa', lw=1))

ax_flow.set_title('Data Processing Pipeline', fontsize=14, fontweight='bold',
                  pad=6, color='#333')

# ══════════════════════════════════════════════════════════════════════
# [1,0]  用户序列长度分布
# ══════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[1, 0])
ax1.set_facecolor('#f0f4f8')
bins = range(3, min(max(full_seq_lens)+2, 61))
ax1.hist(full_seq_lens, bins=bins, color=BLUE, edgecolor='white', linewidth=0.5, alpha=0.9)
ax1.axvline(np.mean(full_seq_lens), color=RED, linestyle='--', lw=1.8, label=f'Mean={np.mean(full_seq_lens):.1f}')
ax1.axvline(np.median(full_seq_lens), color=ORANGE, linestyle='--', lw=1.8, label=f'Median={np.median(full_seq_lens):.1f}')
ax1.set_xlabel('Sequence Length (# items)', fontsize=10)
ax1.set_ylabel('# Users', fontsize=10)
ax1.set_title('User Sequence Length Distribution', fontsize=11, fontweight='bold')
ax1.set_xlim(3, 60)
ax1.legend(fontsize=9)
ax1.text(0.97, 0.95, f'Total: {len(full_seq_lens):,} users', transform=ax1.transAxes,
         ha='right', va='top', fontsize=9, color='#444')

# ══════════════════════════════════════════════════════════════════════
# [1,1]  Train / Val / Test 序列长度对比（箱线图）
# ══════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1, 1])
ax2.set_facecolor('#f0f4f8')
train_lens = [len(train[u]) for u in train]
val_lens   = [len(val[u])-1 for u in val]    # 训练部分长度（不含 target）
test_lens  = [len(test[u])-1 for u in test]  # 同上

bp = ax2.boxplot([train_lens, val_lens, test_lens],
                 labels=['Train\n(input seq)', 'Val\n(input seq)', 'Test\n(input seq)'],
                 patch_artist=True, widths=0.5,
                 medianprops=dict(color='white', linewidth=2))
colors_bp = [BLUE, GREEN, ORANGE]
for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax2.set_ylabel('Input Sequence Length', fontsize=10)
ax2.set_title('Train / Val / Test Split\n(Leave-One-Out)', fontsize=11, fontweight='bold')
ax2.text(0.5, -0.18,
    'Val target = last-2nd item\nTest target = last item',
    transform=ax2.transAxes, ha='center', fontsize=9, color='#666', style='italic')

# ══════════════════════════════════════════════════════════════════════
# [1,2]  item 交互次数分布（长尾）
# ══════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 2])
ax3.set_facecolor('#f0f4f8')
ic_vals = list(item_interaction_cnt.values())
bins3 = [5,10,15,20,30,50,75,100,150,200,500]
ax3.hist(ic_vals, bins=bins3, color=GREEN, edgecolor='white', linewidth=0.5, alpha=0.9)
ax3.set_xlabel('# Interactions per Item', fontsize=10)
ax3.set_ylabel('# Items', fontsize=10)
ax3.set_title('Item Popularity (Long-tail)', fontsize=11, fontweight='bold')
pct_tail = sum(1 for x in ic_vals if x <= 20) / len(ic_vals) * 100
ax3.text(0.97, 0.95, f'{pct_tail:.0f}% items ≤20 interactions\n(long tail)',
         transform=ax3.transAxes, ha='right', va='top', fontsize=9, color='#444')

# ══════════════════════════════════════════════════════════════════════
# [2,0]  品类分布（横向柱状图）
# ══════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[2, 0])
ax4.set_facecolor('#f0f4f8')
top_cats = cat2_counts.most_common(6)
cat_names = [c for c, _ in top_cats]
cat_cnts  = [n for _, n in top_cats]
bars = ax4.barh(cat_names[::-1], cat_cnts[::-1], color=BLUE, alpha=0.85, edgecolor='white')
for bar, cnt in zip(bars, cat_cnts[::-1]):
    ax4.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
             str(cnt), va='center', fontsize=9, color='#444')
ax4.set_xlabel('# Items', fontsize=10)
ax4.set_title('Item Category Distribution\n(Level 2)', fontsize=11, fontweight='bold')
ax4.set_xlim(0, max(cat_cnts) * 1.15)

# ══════════════════════════════════════════════════════════════════════
# [2,1]  价格分布
# ══════════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[2, 1])
ax5.set_facecolor('#f0f4f8')
prices_clip = prices[prices <= 100]
ax5.hist(prices_clip, bins=40, color=ORANGE, edgecolor='white', linewidth=0.5, alpha=0.9)
ax5.axvline(np.mean(prices_clip), color=RED, linestyle='--', lw=1.8, label=f'Mean=${np.mean(prices_clip):.1f}')
ax5.axvline(np.median(prices_clip), color=BLUE, linestyle='--', lw=1.8, label=f'Median=${np.median(prices_clip):.1f}')
ax5.set_xlabel('Price ($)', fontsize=10)
ax5.set_ylabel('# Items', fontsize=10)
ax5.set_title(f'Item Price Distribution (≤$100)\n{len(prices_clip)/len(prices)*100:.0f}% of items shown', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)

# ══════════════════════════════════════════════════════════════════════
# [2,2]  数据集关键数字汇总表
# ══════════════════════════════════════════════════════════════════════
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
ax6.set_facecolor('#f0f4f8')
table_data = [
    ['Metric', 'Value'],
    ['# Users',        f'{len(train):,}'],
    ['# Items',        f'{len(item2id):,}'],
    ['# Interactions', '198,502'],
    ['Min seq len',    str(min(full_seq_lens))],
    ['Max seq len',    str(max(full_seq_lens))],
    ['Mean seq len',   f'{np.mean(full_seq_lens):.1f}'],
    ['Median seq len', f'{np.median(full_seq_lens):.1f}'],
    ['Sparsity',       f'{1 - len([x for seq in test.values() for x in seq])/(len(train)*len(item2id)):.4f}'],
    ['Items w/ price', f'{len(prices):,} ({len(prices)/len(item2id)*100:.0f}%)'],
    ['Split method',   'Leave-one-out'],
]
tbl = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                loc='center', cellLoc='left')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1.15, 1.55)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#4C72B0')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#dce8f5')
    else:
        cell.set_facecolor('#f0f6fc')
    cell.set_edgecolor('#ccc')
ax6.set_title('Dataset Summary', fontsize=11, fontweight='bold', pad=8)

# ══════════════════════════════════════════════════════════════════════
# [3,0-2]  Leave-one-out 示意图（跨三列）
# ══════════════════════════════════════════════════════════════════════
ax_loo = fig.add_subplot(gs[3, :])
ax_loo.set_xlim(0, 10)
ax_loo.set_ylim(0, 1)
ax_loo.axis('off')
ax_loo.set_facecolor('#f8f9fa')
ax_loo.set_title('Leave-One-Out Split Illustration', fontsize=13, fontweight='bold', pad=4)

row_labels = ['Full sequence', 'Train input', 'Val input', 'Test input']
row_colors = ['#bbb', BLUE, GREEN, ORANGE]
item_example = ['i₁', 'i₂', 'i₃', 'i₄', 'i₅', 'i₆', 'i₇']
n = len(item_example)

for row_idx, (label, rcolor) in enumerate(zip(row_labels, row_colors)):
    y_center = 0.82 - row_idx * 0.22
    ax_loo.text(1.05, y_center, label, ha='right', va='center',
                fontsize=10, fontweight='bold', color=rcolor)
    for col_idx, item in enumerate(item_example):
        x = 1.2 + col_idx * 1.1
        # 决定此格子的颜色
        if row_idx == 0:
            fc, alpha, lw = '#999', 0.7, 1
            txt_color = 'white'
        elif row_idx == 1:  # train: 前 n-2 个 active，后 2 个灰
            if col_idx <= n - 3:
                fc, alpha, lw = BLUE, 0.85, 1.5
                txt_color = 'white'
            else:
                fc, alpha, lw = '#ddd', 0.4, 0.5
                txt_color = '#bbb'
        elif row_idx == 2:  # val: 前 n-1 个 active，最后 1 个是 target
            if col_idx <= n - 2:
                fc = GREEN if col_idx < n-1 else ORANGE
                alpha, lw = 0.85, 1.5
                txt_color = 'white'
            else:
                fc, alpha, lw = '#ddd', 0.4, 0.5
                txt_color = '#bbb'
        else:  # test: 所有 active，最后一个是 target
            fc = ORANGE if col_idx == n-1 else ORANGE
            if col_idx < n-1:
                fc = GREEN
            alpha, lw = 0.85, 1.5
            txt_color = 'white'

        rect = mpatches.FancyBboxPatch((x-0.42, y_center-0.08), 0.84, 0.16,
            boxstyle="round,pad=0.02", linewidth=lw,
            edgecolor='#555', facecolor=fc, alpha=alpha, zorder=3)
        ax_loo.add_patch(rect)
        ax_loo.text(x, y_center, item, ha='center', va='center',
                    fontsize=9.5, fontweight='bold', color=txt_color, zorder=4)

# 图例
legend_patches = [
    mpatches.Patch(color=BLUE, label='Train input (i₁ … iₙ₋₂)'),
    mpatches.Patch(color=GREEN, label='Val/Test input prefix'),
    mpatches.Patch(color=ORANGE, label='Prediction target (iₙ₋₁ for val, iₙ for test)'),
]
ax_loo.legend(handles=legend_patches, loc='lower center', fontsize=9,
              ncol=3, framealpha=0.85, bbox_to_anchor=(0.5, -0.05))

# 文字说明
ax_loo.text(5.0, 0.03,
    'Each user contributes one val sample and one test sample. '
    '99 random negative items are added during evaluation (matching TIGER paper).',
    ha='center', va='center', fontsize=9, color='#555', style='italic')

# ── 保存 ──────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), 'data_visualization.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'图已保存至: {out_path}')
plt.show()
