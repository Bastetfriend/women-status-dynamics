# -*- coding: utf-8 -*-
"""画性别权力振荡的示意图，插入 Word 文档。"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

def sigmoid(x, center=0.5, steepness=12):
    return 1 / (1 + np.exp(-steepness * (x - center)))

def make_half_cycle(n_points, peak, phase_a_frac=0.08, inflection_a=0.3, inflection_b=0.85):
    """构建一个半周期：Phase A (短倒S上升) + Phase B (长正S平台+下降)"""
    n_a = max(int(n_points * phase_a_frac), 10)
    n_b = n_points - n_a

    # Phase A: 倒S上升 (0 → peak)，拐点偏前
    x_a = np.linspace(0, 1, n_a)
    # 倒S = 先凹后凸: sigmoid with early inflection
    y_a = peak * sigmoid(x_a, center=inflection_a, steepness=15)
    # normalize to start at 0 and end at peak
    y_a = peak * (y_a - y_a[0]) / (y_a[-1] - y_a[0])

    # Phase B: 正S下降 (peak → 0)，长平台 + 最后快速下降
    x_b = np.linspace(0, 1, n_b)
    # 正S = 先凸后凹 for descent: stays near peak for long, then drops
    y_b = peak * (1 - sigmoid(x_b, center=inflection_b, steepness=12))
    # normalize
    y_b = peak * (y_b - y_b[-1]) / (y_b[0] - y_b[-1])

    return np.concatenate([y_a, y_b[1:]])  # avoid duplicate at junction

# ── 构建完整波形 ──
# 第一周期
T1 = 1000  # 第一周期的点数（代表几万年）
peak1 = 1.0

# Q1: 正半周期（母系），S > 0
q1 = make_half_cycle(T1 // 2, peak=peak1, phase_a_frac=0.06)

# Q2: 负半周期（父系），S < 0
q2 = -make_half_cycle(T1 // 2, peak=peak1 * 0.95, phase_a_frac=0.06)

cycle1 = np.concatenate([q1, q2[1:]])

# 第二周期（更短、更小）
T2 = int(T1 * 0.25)
peak2 = peak1 * 0.35

q3 = make_half_cycle(T2 // 2, peak=peak2, phase_a_frac=0.08)
q4 = -make_half_cycle(T2 // 2, peak=peak2 * 0.9, phase_a_frac=0.08)
cycle2 = np.concatenate([q3, q4[1:]])

# 第三周期（更更短）
T3 = int(T2 * 0.3)
peak3 = peak2 * 0.4
q5 = make_half_cycle(max(T3 // 2, 20), peak=peak3, phase_a_frac=0.1)
q6 = -make_half_cycle(max(T3 // 2, 20), peak=peak3 * 0.9, phase_a_frac=0.1)
cycle3 = np.concatenate([q5, q6[1:]])

# 拼接
full = np.concatenate([cycle1, cycle2[1:], cycle3[1:]])
t = np.arange(len(full))

# ── 画图 ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1.2])
fig.subplots_adjust(hspace=0.35)

# === 上图：完整波形 ===
ax1.plot(t, full, 'k-', linewidth=1.8)
ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')

# 标注区域
# Q1 区域
q1_end = len(q1)
q2_start = q1_end
q2_end = len(cycle1)

# Phase A/B 分界（Q1）
pa1_end = int(len(q1) * 0.06)
ax1.axvspan(0, pa1_end, alpha=0.25, color='#FF6B6B', zorder=0)
ax1.axvspan(pa1_end, q1_end, alpha=0.08, color='#FFB347', zorder=0)

# Phase A/B 分界（Q2）
pa2_start = q2_start
pa2_end = q2_start + int((q2_end - q2_start) * 0.06)
ax1.axvspan(pa2_start, pa2_end, alpha=0.25, color='#4ECDC4', zorder=0)
ax1.axvspan(pa2_end, q2_end, alpha=0.08, color='#95E1D3', zorder=0)

# 标注文字
ax1.annotate('Phase A\n(倒S·短)', xy=(pa1_end//2, peak1*0.5), fontsize=8,
            ha='center', color='#CC3333', fontweight='bold')
ax1.annotate('Phase B (正S·长平台)', xy=((pa1_end + q1_end)//2, peak1*0.7), fontsize=9,
            ha='center', color='#CC7722')
ax1.annotate('Phase A\n(倒S·短)', xy=((pa2_start+pa2_end)//2, -peak1*0.5), fontsize=8,
            ha='center', color='#228888', fontweight='bold')
ax1.annotate('Phase B (正S·长平台)', xy=((pa2_end + q2_end)//2, -peak1*0.6), fontsize=9,
            ha='center', color='#339966')

# Q1/Q2 标签
ax1.text(q1_end // 2, peak1 * 1.15, 'Q1: 母系阶段 (S > 0)', ha='center', fontsize=11,
        fontweight='bold', color='#CC3333')
ax1.text((q2_start + q2_end) // 2, -peak1 * 1.15, 'Q2: 父系阶段 (S < 0)', ha='center',
        fontsize=11, fontweight='bold', color='#228888')

# "我们在这" 标注
we_are_here = q2_end - int((q2_end - pa2_end) * 0.15)
ax1.annotate('← 我们在这\n   (Q2 Phase B 后段)',
            xy=(we_are_here, full[min(we_are_here, len(full)-1)]),
            xytext=(we_are_here + 80, -0.55),
            fontsize=10, fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

# 第二三周期标注
c2_start = len(cycle1)
c2_end = c2_start + len(cycle2) - 1
c3_start = c2_end
ax1.annotate('第二周期\n(更短·更小)', xy=((c2_start + c2_end)//2, peak2 * 1.1),
            fontsize=9, ha='center', color='#666666', style='italic')

# 阻尼包络线
envelope_t = [0, pa1_end, q1_end//2, q1_end, q2_start + (q2_end-q2_start)//3,
              q2_end, c2_start + len(cycle2)//4, c2_end, c3_start + T3//2, len(full)-1]
envelope_pos = [0, peak1*0.8, peak1, peak1*0.5, 0.1, 0, peak2*0.8, peak2*0.3, peak3*0.5, peak3*0.2]
envelope_neg = [0, -peak1*0.3, 0, -peak1*0.8, -peak1*0.95, 0, -peak2*0.3, -peak2*0.5, -peak3*0.3, -peak3*0.15]
# simplified envelope
env_t_simple = [0, q1_end//2, q2_end, (c2_start+c2_end)//2, len(full)-1]
env_pos = [peak1*0.05, peak1*1.05, peak1*0.02, peak2*1.05, peak3*0.6]
env_neg = [0, -peak1*0.02, -peak1*1.0, -peak2*0.5, -peak3*0.3]

ax1.fill_between([0, len(full)], [0.05]*2, [0.05]*2, alpha=0, color='gray')  # placeholder

# δ 线
delta = 0.08
ax1.axhline(y=delta, color='purple', linewidth=0.8, linestyle='--', alpha=0.5)
ax1.axhline(y=-delta, color='purple', linewidth=0.8, linestyle='--', alpha=0.5)
ax1.text(len(full) + 10, delta, 'δ > 0', fontsize=9, color='purple', va='center')
ax1.text(len(full) + 10, -delta, '-δ', fontsize=9, color='purple', va='center')

# 标记不可检验/可检验
ax1.axvspan(0, q1_end, alpha=0.03, color='gray')
ax1.axvspan(q2_start, q2_end, alpha=0.03, color='blue')
ax1.text(q1_end//2, -peak1*1.35, '← 史前·不可直接检验 →', ha='center', fontsize=9,
        color='gray', style='italic')
ax1.text((q2_start+q2_end)//2, peak1*1.35, '← 有文字记录·可检验 →', ha='center', fontsize=9,
        color='#2244AA', style='italic')

ax1.set_ylabel('S(t)\n正 = 权力流向女性\n负 = 权力流向男性', fontsize=10, rotation=0,
              labelpad=80, va='center')
ax1.set_xlabel('时间 t →  (周期递减：生产力加速)', fontsize=10)
ax1.set_title('性别权力振荡：阻尼啁啾非对称弛豫模型', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlim(-30, len(full) + 50)
ax1.set_ylim(-1.5, 1.5)
ax1.set_xticks([])
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

# === 下图：单个半周期放大 ===
# 只展示 Q1（母系半周期）
t_q1 = np.arange(len(q1))
ax2.plot(t_q1, q1, 'k-', linewidth=2)
ax2.axhline(y=0, color='gray', linewidth=0.5)

# Phase A 区域
ax2.axvspan(0, pa1_end, alpha=0.3, color='#FF6B6B')
ax2.axvspan(pa1_end, len(q1), alpha=0.1, color='#FFB347')

# 拐点标注
inflection_pos = int(pa1_end * 0.3)
ax2.plot(inflection_pos, q1[inflection_pos], 'ro', markersize=8, zorder=5)
ax2.annotate('拐点\n(爆发)', xy=(inflection_pos, q1[inflection_pos]),
            xytext=(inflection_pos + 30, q1[inflection_pos] - 0.2),
            fontsize=9, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red'))

# Phase 标注
ax2.text(pa1_end // 2, -0.15, 'Phase A\n倒S (短)', ha='center', fontsize=9,
        color='#CC3333', fontweight='bold')
ax2.text((pa1_end + len(q1)) // 2, peak1 * 0.85, 'Phase B: 正S (长平台)',
        ha='center', fontsize=10, color='#CC7722')

# 平台期标注
plateau_start = pa1_end + int((len(q1) - pa1_end) * 0.05)
plateau_end = pa1_end + int((len(q1) - pa1_end) * 0.75)
ax2.annotate('', xy=(plateau_start, peak1 * 0.97), xytext=(plateau_end, peak1 * 0.97),
            arrowprops=dict(arrowstyle='<->', color='#CC7722', lw=1.5))
ax2.text((plateau_start + plateau_end)//2, peak1 * 1.05, '思想顽固·地位几乎不变',
        ha='center', fontsize=8, color='#CC7722', style='italic')

ax2.set_ylabel('S(t)', fontsize=10)
ax2.set_xlabel('时间 →', fontsize=10)
ax2.set_title('放大：一个半周期内部结构（Phase A 短爆发 + Phase B 长平台）', fontsize=11, fontweight='bold')
ax2.set_xticks([])
ax2.set_ylim(-0.25, 1.2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

plt.savefig(r'data\_gender_oscillation.png',
           dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Figure saved!")
