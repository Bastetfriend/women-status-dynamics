# -*- coding: utf-8 -*-
"""全量排序：191国 S_ceil vs EDI，不分组，让数据说话"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = r'data'

params = pd.read_csv(f'{BASE}/_vdem_fit_all.csv')
params = params[params['status'] == 'ok'].copy()

edi = pd.read_csv(f'{BASE}/electoral-democracy-index.csv')
edi.columns = ['entity', 'code', 'year', 'edi', 'region']

# Per-country EDI: mean across all years + latest
edi_mean = edi.groupby('code')['edi'].mean().rename('edi_mean')
edi_latest = edi.sort_values('year').groupby('code').last()[['edi']].rename(columns={'edi': 'edi_latest'})

df = params.merge(edi_mean, on='code', how='left').merge(edi_latest, on='code', how='left')
df = df.sort_values('ric_ceil')

print('=' * 95)
print('  191 countries sorted by S_ceil (ascending) — NO grouping, NO labels')
print('=' * 95)
print(f"  {'#':>3} {'Code':<8} {'Entity':<28} {'S_ceil':>7} {'k':>6} {'EDI_mean':>9} {'EDI_now':>8}")
print('  ' + '─' * 85)
for i, (_, r) in enumerate(df.iterrows(), 1):
    em = f"{r['edi_mean']:.3f}" if pd.notna(r['edi_mean']) else '  N/A'
    el = f"{r['edi_latest']:.3f}" if pd.notna(r['edi_latest']) else '  N/A'
    print(f"  {i:>3} {r['code']:<8} {str(r['entity'])[:28]:<28} "
          f"{r['ric_ceil']:>+7.3f} {r['ric_k']:>6.3f} {em:>9} {el:>8}")

# ── Correlation ──
valid = df.dropna(subset=['edi_mean', 'ric_ceil'])
r_mean, p_mean = stats.pearsonr(valid['edi_mean'], valid['ric_ceil'])
sr_mean, sp_mean = stats.spearmanr(valid['edi_mean'], valid['ric_ceil'])

v2 = valid.dropna(subset=['edi_latest'])
r_lat, p_lat = stats.pearsonr(v2['edi_latest'], v2['ric_ceil'])

print(f'\n  ═══ CORRELATION: EDI vs S_ceil (全量, 无分组) ═══')
print(f'  Pearson  (EDI_mean):   r={r_mean:+.4f}, p={p_mean:.2e}, n={len(valid)}')
print(f'  Spearman (EDI_mean):   ρ={sr_mean:+.4f}, p={sp_mean:.2e}')
print(f'  Pearson  (EDI_latest): r={r_lat:+.4f}, p={p_lat:.2e}, n={len(v2)}')
print(f'  R² (EDI_mean):         {r_mean**2:.4f} = {r_mean**2*100:.1f}% variance explained')

# ── Bottom vs Top comparison ──
print(f'\n  ═══ S_ceil BOTTOM 20 vs TOP 20 ═══')
bot = df.head(20)
top = df.tail(20)
print(f'  Bottom 20:  EDI_mean median = {bot["edi_mean"].median():.3f}')
print(f'  Top 20:     EDI_mean median = {top["edi_mean"].median():.3f}')
u, p = stats.mannwhitneyu(bot['edi_mean'].dropna(), top['edi_mean'].dropna())
print(f'  Mann-Whitney: U={u:.0f}, p={p:.4f} {"***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""}')

# ── Scatter plot ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.scatter(valid['edi_mean'], valid['ric_ceil'], alpha=0.5, s=25, c='steelblue')
for _, r in valid.nsmallest(15, 'ric_ceil').iterrows():
    ax.annotate(r['code'], (r['edi_mean'], r['ric_ceil']),
                fontsize=7, alpha=0.8, ha='left')
z = np.polyfit(valid['edi_mean'], valid['ric_ceil'], 1)
xline = np.linspace(0, 0.85, 100)
ax.plot(xline, np.polyval(z, xline), 'r--', alpha=0.6, linewidth=1.5)
ax.set_xlabel('Electoral Democracy Index (mean over all years)', fontsize=11)
ax.set_ylabel('S_ceil (Richards ceiling)', fontsize=11)
ax.set_title(f'(a) EDI_mean vs S_ceil  (r={r_mean:+.3f}***)', fontsize=12)

ax = axes[1]
ax.scatter(v2['edi_latest'], v2['ric_ceil'], alpha=0.5, s=25, c='darkorange')
for _, r in v2.nsmallest(15, 'ric_ceil').iterrows():
    ax.annotate(r['code'], (r['edi_latest'], r['ric_ceil']),
                fontsize=7, alpha=0.8, ha='left')
z2 = np.polyfit(v2['edi_latest'], v2['ric_ceil'], 1)
ax.plot(xline, np.polyval(z2, xline), 'r--', alpha=0.6, linewidth=1.5)
ax.set_xlabel('Electoral Democracy Index (latest available)', fontsize=11)
ax.set_ylabel('S_ceil (Richards ceiling)', fontsize=11)
ax.set_title(f'(b) EDI_latest vs S_ceil  (r={r_lat:+.3f}***)', fontsize=12)

plt.tight_layout()
plt.savefig(f'{BASE}/_vdem_edi_vs_sceil.png', dpi=200, bbox_inches='tight')
print(f'\n  Saved: _vdem_edi_vs_sceil.png')
