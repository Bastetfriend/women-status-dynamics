"""
棘轮效应全量分析：所有国家所有 suppression 事件
- 反弹率 (rebound rate)
- 反弹幅度 (overshoot magnitude)
- 恢复时间 (recovery time)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 数据 ──

df = pd.read_csv(r"data\women-political-empowerment-index.csv")
df.columns = ['entity', 'code', 'year', 'wpei', 'region']
df = df.sort_values(['code', 'year'])

# ── 参数 ──

MIN_DROP = 0.03       # WPEI 最小跌幅
MIN_DURATION = 2      # peak→trough 最少年数
MIN_YEARS = 30        # 国家最少数据年数

# ── 检测 suppression 事件 ──

def detect_suppressions(years, wpei, min_drop=MIN_DROP, min_dur=MIN_DURATION):
    """
    检测所有 suppression-rebound 事件。

    算法：
    1. 扫描序列，维护 running_max
    2. 当 running_max - current >= min_drop，标记进入 suppression
    3. 找到 trough（局部最小值）
    4. 从 trough 向后找恢复
    """
    events = []
    n = len(years)
    i = 0

    while i < n:
        # 找局部 peak：向前走直到开始下降
        while i < n - 1 and wpei[i + 1] >= wpei[i]:
            i += 1

        if i >= n - 1:
            break

        peak_idx = i
        peak_val = wpei[peak_idx]
        peak_year = years[peak_idx]

        # 向后找 trough
        j = i + 1
        trough_idx = j
        trough_val = wpei[j]

        while j < n - 1:
            if wpei[j] < trough_val:
                trough_val = wpei[j]
                trough_idx = j
            # 如果已经回升超过 peak，这不是 suppression
            if wpei[j] > peak_val:
                break
            # 如果开始持续回升（连续3年上升），认为 trough 已找到
            if j >= trough_idx + 3 and all(wpei[j-k] > wpei[j-k-1] for k in range(3) if j-k-1 >= trough_idx):
                break
            j += 1

        drop = peak_val - trough_val
        duration = years[trough_idx] - peak_year

        if drop >= min_drop and duration >= min_dur:
            # 找反弹：trough 之后的最大值
            post_trough = wpei[trough_idx:]
            post_years = years[trough_idx:]

            if len(post_trough) > 1:
                max_after = np.max(post_trough)
                max_after_idx = np.argmax(post_trough)
                max_after_year = post_years[max_after_idx]

                rebounded = max_after > peak_val
                overshoot = max_after - peak_val if rebounded else None
                recovery_time = (max_after_year - years[trough_idx]) if rebounded else None

                # 是否仍在下跌中（trough 是最后几年）
                ongoing = (years[trough_idx] >= years[-1] - 3)

                events.append({
                    'peak_year': int(peak_year),
                    'peak_val': float(peak_val),
                    'trough_year': int(years[trough_idx]),
                    'trough_val': float(trough_val),
                    'drop': float(drop),
                    'duration': int(duration),
                    'rebounded': bool(rebounded),
                    'overshoot': float(overshoot) if overshoot is not None else None,
                    'recovery_time': int(recovery_time) if recovery_time is not None else None,
                    'max_after': float(max_after),
                    'max_after_year': int(max_after_year),
                    'ongoing': bool(ongoing),
                })

        # 移到 trough 之后继续扫描
        i = trough_idx + 1

    return events

# ── 主程序 ──

print("=" * 70)
print("  RATCHET EFFECT: FULL ANALYSIS")
print("  Suppression threshold: WPEI drop >= {:.2f}, duration >= {} years".format(MIN_DROP, MIN_DURATION))
print("=" * 70)

all_events = []
countries_with_events = 0
countries_scanned = 0

for code, grp in df.groupby('code'):
    if len(grp) < MIN_YEARS:
        continue
    countries_scanned += 1
    years = grp['year'].values
    wpei = grp['wpei'].values
    entity = grp['entity'].iloc[0]
    region = grp['region'].iloc[0]

    events = detect_suppressions(years, wpei)
    if events:
        countries_with_events += 1
        for e in events:
            e['code'] = code
            e['entity'] = entity
            e['region'] = region
            all_events.append(e)

events_df = pd.DataFrame(all_events)

# ── 存 CSV ──

csv_path = r"data\_vdem_ratchet_all.csv"
events_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

# ── 统计 ──

print(f"\n  Countries scanned: {countries_scanned}")
print(f"  Countries with >= 1 event: {countries_with_events}")
print(f"  Total suppression events: {len(events_df)}")

# 排除 ongoing（当前正在下跌，还没来得及反弹）
completed = events_df[~events_df['ongoing']]
ongoing = events_df[events_df['ongoing']]

print(f"  Completed events (not ongoing): {len(completed)}")
print(f"  Ongoing events (trough in last 3 years): {len(ongoing)}")

# ── 反弹率 ──

n_rebounded = completed['rebounded'].sum()
n_completed = len(completed)
rebound_rate = n_rebounded / n_completed * 100 if n_completed > 0 else 0

print(f"\n{'='*70}")
print(f"  REBOUND RATE (completed events only)")
print(f"{'='*70}")
print(f"  Rebounded (exceeded pre-drop level): {n_rebounded}/{n_completed} ({rebound_rate:.1f}%)")
print(f"  Not rebounded: {n_completed - n_rebounded}/{n_completed} ({100 - rebound_rate:.1f}%)")

# ── 反弹幅度 ──

rebounded_events = completed[completed['rebounded']]
if len(rebounded_events) > 0:
    print(f"\n{'='*70}")
    print(f"  OVERSHOOT MAGNITUDE (n={len(rebounded_events)})")
    print(f"{'='*70}")
    os_vals = rebounded_events['overshoot']
    print(f"  Mean overshoot:   {os_vals.mean():.4f}")
    print(f"  Median overshoot: {os_vals.median():.4f}")
    print(f"  Min overshoot:    {os_vals.min():.4f}")
    print(f"  Max overshoot:    {os_vals.max():.4f}")
    print(f"  Std:              {os_vals.std():.4f}")

    # 按跌幅大小分层
    print(f"\n  By drop severity:")
    for lo, hi, label in [(0.03, 0.05, 'Small (0.03-0.05)'),
                           (0.05, 0.10, 'Medium (0.05-0.10)'),
                           (0.10, 0.20, 'Large (0.10-0.20)'),
                           (0.20, 1.00, 'Severe (>0.20)')]:
        subset = rebounded_events[(rebounded_events['drop'] >= lo) & (rebounded_events['drop'] < hi)]
        if len(subset) > 0:
            print(f"    {label}: n={len(subset)}, "
                  f"mean overshoot={subset['overshoot'].mean():.4f}, "
                  f"mean recovery={subset['recovery_time'].mean():.1f} years")

# ── 恢复时间 ──

if len(rebounded_events) > 0:
    rt = rebounded_events['recovery_time']
    print(f"\n{'='*70}")
    print(f"  RECOVERY TIME (years from trough to exceed pre-drop)")
    print(f"{'='*70}")
    print(f"  Mean:   {rt.mean():.1f}")
    print(f"  Median: {rt.median():.1f}")
    print(f"  Min:    {rt.min()}")
    print(f"  Max:    {rt.max()}")

# ── 未反弹事件分析 ──

not_rebounded = completed[~completed['rebounded']]
if len(not_rebounded) > 0:
    print(f"\n{'='*70}")
    print(f"  NOT-REBOUNDED EVENTS (n={len(not_rebounded)})")
    print(f"{'='*70}")
    for _, e in not_rebounded.iterrows():
        deficit = e['peak_val'] - e['max_after']
        print(f"  {e['entity']:<25} {e['peak_year']}-{e['trough_year']} "
              f"drop={e['drop']:.3f} max_after={e['max_after']:.3f} "
              f"deficit={deficit:.3f}")

# ── Ongoing 事件（当前正在下跌的国家） ──

if len(ongoing) > 0:
    print(f"\n{'='*70}")
    print(f"  ONGOING SUPPRESSIONS (trough in last 3 years, n={len(ongoing)})")
    print(f"{'='*70}")
    ongoing_sorted = ongoing.sort_values('drop', ascending=False)
    print(f"  {'Country':<25} {'Peak':<6} {'Trough':<7} {'Drop':>6} {'Peak WPEI':>10}")
    for _, e in ongoing_sorted.head(30).iterrows():
        print(f"  {e['entity']:<25} {e['peak_year']:<6} {e['trough_year']:<7} "
              f"{e['drop']:>6.3f} {e['peak_val']:>10.3f}")

# ── 按地区 ──

print(f"\n{'='*70}")
print(f"  BY REGION (completed events)")
print(f"{'='*70}")
for region, grp in completed.groupby('region'):
    n_r = grp['rebounded'].sum()
    n_t = len(grp)
    rate = n_r / n_t * 100 if n_t > 0 else 0
    os_mean = grp[grp['rebounded']]['overshoot'].mean() if n_r > 0 else 0
    print(f"  {region:<20} events={n_t:>3}  rebounded={n_r:>3} ({rate:>5.1f}%)  "
          f"mean_overshoot={os_mean:.4f}")

# ── 画图 ──

fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

# 1. 反弹率汇总
ax = axes[0, 0]
labels = ['Rebounded\n(exceeded pre-drop)', 'Not rebounded', 'Ongoing\n(in progress)']
sizes = [n_rebounded, n_completed - n_rebounded, len(ongoing)]
colors = ['#2ca02c', '#d62728', '#ff7f0e']
bars = ax.bar(labels, sizes, color=colors, edgecolor='white')
for bar, size in zip(bars, sizes):
    pct = size / len(events_df) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{size}\n({pct:.0f}%)', ha='center', fontsize=10)
ax.set_ylabel('Number of events')
ax.set_title(f'Suppression Event Outcomes (n={len(events_df)})')

# 2. Overshoot 分布
ax = axes[0, 1]
if len(rebounded_events) > 0:
    ax.hist(rebounded_events['overshoot'], bins=30, color='#2ca02c',
            edgecolor='white', alpha=0.8)
    ax.axvline(rebounded_events['overshoot'].mean(), color='r', ls='--',
               label=f"Mean={rebounded_events['overshoot'].mean():.3f}")
    ax.axvline(rebounded_events['overshoot'].median(), color='orange', ls='--',
               label=f"Median={rebounded_events['overshoot'].median():.3f}")
    ax.set_xlabel('Overshoot (WPEI above pre-drop peak)')
    ax.set_ylabel('Count')
    ax.set_title(f'Rebound Overshoot Distribution (n={len(rebounded_events)})')
    ax.legend()

# 3. Drop vs Overshoot 散点图
ax = axes[1, 0]
if len(rebounded_events) > 0:
    ax.scatter(rebounded_events['drop'], rebounded_events['overshoot'],
               s=15, alpha=0.5, color='steelblue', edgecolors='k', linewidths=0.3)
    # 对角线：overshoot = drop（反弹等于跌幅）
    max_val = max(rebounded_events['drop'].max(), rebounded_events['overshoot'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.3, label='Overshoot = Drop')
    ax.set_xlabel('Drop magnitude (WPEI)')
    ax.set_ylabel('Overshoot magnitude (WPEI)')
    ax.set_title('Drop vs Overshoot')
    ax.legend()

# 4. Recovery time 分布
ax = axes[1, 1]
if len(rebounded_events) > 0:
    rt_clipped = rebounded_events['recovery_time'].clip(upper=80)
    ax.hist(rt_clipped, bins=30, color='#ff7f0e', edgecolor='white', alpha=0.8)
    ax.axvline(rebounded_events['recovery_time'].mean(), color='r', ls='--',
               label=f"Mean={rebounded_events['recovery_time'].mean():.0f} yr")
    ax.axvline(rebounded_events['recovery_time'].median(), color='orange', ls='--',
               label=f"Median={rebounded_events['recovery_time'].median():.0f} yr")
    ax.set_xlabel('Recovery time (years)')
    ax.set_ylabel('Count')
    ax.set_title('Time to Exceed Pre-Drop Level')
    ax.legend()

fig.suptitle(f'Ratchet Effect: {len(events_df)} Suppression Events across {countries_with_events} Countries',
             fontsize=13, fontweight='bold')
plt.savefig(r"data\_vdem_ratchet.png", dpi=200)
print(f"\nFigure saved: _vdem_ratchet.png")
plt.close()

# ── 论文用语句 ──

print(f"\n{'='*70}")
print(f"  PAPER-READY")
print(f"{'='*70}")
if n_completed > 0:
    print(f"  Across {countries_with_events} countries, we identified {len(events_df)} "
          f"suppression events (WPEI drop >= {MIN_DROP}, duration >= {MIN_DURATION} years).")
    print(f"  Of {n_completed} completed events, {n_rebounded} ({rebound_rate:.1f}%) "
          f"rebounded above the pre-suppression level")
    if len(rebounded_events) > 0:
        print(f"  (mean overshoot = {rebounded_events['overshoot'].mean():.4f}, "
              f"median recovery time = {rebounded_events['recovery_time'].median():.0f} years).")
    print(f"  {len(ongoing)} events are ongoing (trough within last 3 years).")
