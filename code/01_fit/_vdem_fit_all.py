"""
V-Dem S(t) 全量拟合：198 个国家 logistic + Richards
结果存 CSV + 汇总统计 + 拐点直方图
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')

# ── 数据 ──────────────────────────────────────────────

df = pd.read_csv(r"data\women-political-empowerment-index.csv")
df.columns = ['entity', 'code', 'year', 'wpei', 'region']
df['S'] = df['wpei'] - 1

# ── 拟合函数 ──────────────────────────────────────────

def logistic(t, S_floor, S_ceil, k, t_mid):
    return S_floor + (S_ceil - S_floor) / (1 + np.exp(-k * (t - t_mid)))

def richards(t, S_floor, S_ceil, k, t_mid, v):
    z = v * np.exp(-k * (t - t_mid))
    # 数值保护
    z = np.clip(z, 1e-15, 1e15)
    return S_floor + (S_ceil - S_floor) / (1 + z) ** (1 / v)

# ── 单国拟合 ──────────────────────────────────────────

def fit_one(code, entity, t, s):
    """返回 dict 或 None"""
    row = {
        'code': code, 'entity': entity,
        'year_start': int(t.min()), 'year_end': int(t.max()),
        'n_years': len(t),
        'S_now': s[-1],
        'S_range': s.max() - s.min(),
    }

    # 变化太小的跳过（几乎没动过）
    if row['S_range'] < 0.05:
        row['status'] = 'flat'
        return row

    # ── Logistic ──
    try:
        p0 = [s.min(), s.max(), 0.03, (t.min() + t.max()) / 2]
        bounds = ([s.min() - 0.5, -1.0, 0.0005, t.min() - 100],
                  [0.0, 0.0, 2.0, t.max() + 100])
        popt, _ = curve_fit(logistic, t, s, p0=p0, bounds=bounds, maxfev=30000)
        pred = logistic(t, *popt)
        ss_res = np.sum((s - pred) ** 2)
        ss_tot = np.sum((s - np.mean(s)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        row['log_R2'] = r2
        row['log_floor'] = popt[0]
        row['log_ceil'] = popt[1]
        row['log_k'] = popt[2]
        row['log_tmid'] = popt[3]
    except Exception:
        row['log_R2'] = np.nan
        row['status'] = 'log_fail'
        return row

    # ── Richards ──
    try:
        p0r = [popt[0], popt[1], popt[2], popt[3], 1.0]
        bounds_r = ([s.min() - 0.5, -1.0, 0.0005, t.min() - 100, 0.005],
                    [0.0, 0.0, 2.0, t.max() + 100, 20.0])
        poptr, _ = curve_fit(richards, t, s, p0=p0r, bounds=bounds_r, maxfev=30000)
        pred_r = richards(t, *poptr)
        ss_res = np.sum((s - pred_r) ** 2)
        r2r = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        row['ric_R2'] = r2r
        row['ric_floor'] = poptr[0]
        row['ric_ceil'] = poptr[1]
        row['ric_k'] = poptr[2]
        row['ric_tmid'] = poptr[3]
        row['ric_v'] = poptr[4]
        row['dR2'] = r2r - r2
        row['status'] = 'ok'
    except Exception:
        row['ric_R2'] = np.nan
        row['ric_v'] = np.nan
        row['status'] = 'ric_fail'

    return row

# ── 主程序 ──────────────────────────────────────────

MIN_YEARS = 30

all_codes = df.groupby('code').agg(
    entity=('entity', 'first'),
    n=('year', 'count')
).reset_index()
all_codes = all_codes[all_codes['n'] >= MIN_YEARS].sort_values('code')

print(f"Total countries with >= {MIN_YEARS} years data: {len(all_codes)}")
print("Fitting...\n")

rows = []
for _, info in all_codes.iterrows():
    code = info['code']
    entity = info['entity']
    sub = df[df['code'] == code].sort_values('year')
    t = sub['year'].values.astype(float)
    s = sub['S'].values
    row = fit_one(code, entity, t, s)
    if row:
        rows.append(row)
    # 进度
    if len(rows) % 20 == 0:
        print(f"  {len(rows)} done...")

results = pd.DataFrame(rows)

# ── 存 CSV ──────────────────────────────────────────

csv_path = r"data\_vdem_fit_all.csv"
results.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"\nSaved: {csv_path}")

# ── 统计 ──────────────────────────────────────────

ok = results[results['status'] == 'ok']
flat = results[results['status'] == 'flat']
fail = results[~results['status'].isin(['ok', 'flat'])]

print(f"\n{'='*60}")
print(f"  FITTING SUMMARY")
print(f"{'='*60}")
print(f"  Total countries:  {len(results)}")
print(f"  Successful (ok):  {len(ok)}")
print(f"  Flat (range<0.05): {len(flat)}")
print(f"  Failed:           {len(fail)}")

# R2 分布
print(f"\n  Logistic R2 distribution (n={len(ok)}):")
for thresh in [0.99, 0.95, 0.90, 0.80, 0.70]:
    n_above = (ok['log_R2'] >= thresh).sum()
    print(f"    R2 >= {thresh}: {n_above} ({n_above/len(ok)*100:.1f}%)")

print(f"\n  Richards R2 distribution (n={len(ok)}):")
for thresh in [0.99, 0.95, 0.90, 0.80, 0.70]:
    n_above = (ok['ric_R2'] >= thresh).sum()
    print(f"    R2 >= {thresh}: {n_above} ({n_above/len(ok)*100:.1f}%)")

# R2 改善
print(f"\n  Richards vs Logistic improvement:")
print(f"    Mean dR2: {ok['dR2'].mean():+.4f}")
print(f"    Median dR2: {ok['dR2'].median():+.4f}")
print(f"    Countries where Richards wins (dR2>0.005): {(ok['dR2'] > 0.005).sum()}")

# v 参数分布
print(f"\n  Richards v parameter distribution:")
v_ok = ok['ric_v'].dropna()
print(f"    Mean: {v_ok.mean():.2f}")
print(f"    Median: {v_ok.median():.2f}")
print(f"    Symmetric (|v-1|<0.1): {((v_ok - 1).abs() < 0.1).sum()}")
print(f"    Asymmetric v>1 (slow finish): {(v_ok > 1.1).sum()}")
print(f"    Asymmetric v<1 (slow start): {(v_ok < 0.9).sum()}")
print(f"    Extreme v>=10: {(v_ok >= 10).sum()}")

# t_mid 分布
print(f"\n  Inflection point (Richards t_mid) distribution:")
tmid_ok = ok['ric_tmid'].dropna()
print(f"    Mean: {tmid_ok.mean():.1f}")
print(f"    Median: {tmid_ok.median():.1f}")
print(f"    Earliest: {tmid_ok.min():.1f}")
print(f"    Latest: {tmid_ok.max():.1f}")

# S_ceil 分布 (预测极限)
print(f"\n  Predicted ceiling S_ceil (Richards):")
ceil_ok = ok['ric_ceil'].dropna()
print(f"    Mean: {ceil_ok.mean():.3f}")
print(f"    Closest to 0: {ceil_ok.max():.3f}")
print(f"    Furthest from 0: {ceil_ok.min():.3f}")

# ── TOP/BOTTOM 排名 ──────────────────────────────

print(f"\n{'='*60}")
print(f"  TOP 20 by Richards R2")
print(f"{'='*60}")
top20 = ok.nlargest(20, 'ric_R2')
print(f"{'Country':<25} {'Years':<12} {'R2_ric':>7} {'v':>6} {'t_mid':>7} {'S_ceil':>7} {'S_now':>7}")
for _, r in top20.iterrows():
    print(f"{r['entity']:<25} {r['year_start']}-{r['year_end']:<5} "
          f"{r['ric_R2']:>7.4f} {r['ric_v']:>6.2f} {r['ric_tmid']:>7.1f} "
          f"{r['ric_ceil']:>7.3f} {r['S_now']:>7.3f}")

print(f"\n{'='*60}")
print(f"  BOTTOM 10 by Richards R2 (among ok)")
print(f"{'='*60}")
bot10 = ok.nsmallest(10, 'ric_R2')
for _, r in bot10.iterrows():
    print(f"{r['entity']:<25} {r['year_start']}-{r['year_end']:<5} "
          f"{r['ric_R2']:>7.4f} {r['ric_v']:>6.2f} {r['ric_tmid']:>7.1f}")

# ── 画图 ──────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

# 1. R2 直方图
ax = axes[0, 0]
ax.hist(ok['ric_R2'], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0.95, color='r', ls='--', alpha=0.5, label='R2=0.95')
ax.axvline(0.90, color='orange', ls='--', alpha=0.5, label='R2=0.90')
ax.set_xlabel('Richards R2')
ax.set_ylabel('Count')
ax.set_title(f'R2 Distribution (n={len(ok)})')
ax.legend()

# 2. t_mid 直方图（拐点年份分布）
ax = axes[0, 1]
ax.hist(tmid_ok, bins=30, color='coral', edgecolor='white', alpha=0.8)
ax.set_xlabel('Inflection Year (t_mid)')
ax.set_ylabel('Count')
ax.set_title('Inflection Point Distribution')
# 标注历史节点
for yr, label in [(1918, 'WWI'), (1945, 'WWII'), (1970, '2nd Wave')]:
    ax.axvline(yr, color='gray', ls=':', alpha=0.4)
    ax.text(yr, ax.get_ylim()[1] * 0.9, label, fontsize=8, ha='center', rotation=90)

# 3. v 参数直方图
ax = axes[1, 0]
v_clipped = v_ok.clip(upper=20)
ax.hist(v_clipped, bins=40, color='forestgreen', edgecolor='white', alpha=0.8)
ax.axvline(1.0, color='r', ls='--', label='v=1 (symmetric)')
ax.set_xlabel('Richards v parameter')
ax.set_ylabel('Count')
ax.set_title('Asymmetry Distribution')
ax.legend()

# 4. S_now vs t_mid 散点图
ax = axes[1, 1]
sc = ax.scatter(ok['ric_tmid'], ok['S_now'], c=ok['ric_v'].clip(upper=15),
                cmap='coolwarm', s=20, alpha=0.7, edgecolors='k', linewidths=0.3)
ax.set_xlabel('Inflection Year (t_mid)')
ax.set_ylabel('S(now)')
ax.set_title('Current Status vs Inflection Timing')
ax.axhline(0, color='k', ls='-', alpha=0.2)
plt.colorbar(sc, ax=ax, label='v (asymmetry)')

# 标注关键国家
highlight = {'CHN': 'China', 'USA': 'USA', 'NOR': 'Norway', 'SAU': 'Saudi',
             'ISL': 'Iceland', 'FRA': 'France', 'JPN': 'Japan', 'SWE': 'Sweden',
             'KOR': 'S.Korea', 'FIN': 'Finland', 'GBR': 'UK'}
for code, label in highlight.items():
    row = ok[ok['code'] == code]
    if len(row) > 0:
        r = row.iloc[0]
        ax.annotate(label, (r['ric_tmid'], r['S_now']),
                    fontsize=7, ha='left', va='bottom',
                    xytext=(3, 3), textcoords='offset points')

fig.suptitle('V-Dem Phase B Sigmoid Fit: All Countries', fontsize=14, fontweight='bold')
plt.savefig(r"data\_vdem_fit_all.png", dpi=200)
print(f"\nFigure saved: _vdem_fit_all.png")
plt.close()

# ── 按地区汇总 ──────────────────────────────────────

print(f"\n{'='*60}")
print(f"  BY REGION")
print(f"{'='*60}")
ok_with_region = ok.merge(df[['code', 'region']].drop_duplicates(), on='code', how='left')
for region, grp in ok_with_region.groupby('region'):
    print(f"\n  {region} (n={len(grp)}):")
    print(f"    Mean R2: {grp['ric_R2'].mean():.3f}")
    print(f"    Mean t_mid: {grp['ric_tmid'].mean():.1f}")
    print(f"    Mean v: {grp['ric_v'].mean():.2f}")
    print(f"    Mean S_now: {grp['S_now'].mean():.3f}")
