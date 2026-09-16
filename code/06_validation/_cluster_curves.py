# -*- coding: utf-8 -*-
"""Pairwise JS-divergence clustering of Richards curves across 174 countries.

For each country, the Richards derivative dŜ/dt is normalized to a probability
density over time. Pairwise Jensen-Shannon divergence yields a distance matrix.
Hierarchical clustering then tests whether curve shape groups by external factors.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import spearmanr
import matplotlib
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load data
# ============================================================
diag = pd.read_csv(r'data\_richards_diagnostics.csv')
panel = pd.read_csv(r'data\_panel_S_MAC.csv')

# Build param dict
params = {}
for _, row in diag.iterrows():
    code = row['code']
    ps = {k: row[k] for k in ['S_floor','S_ceil','k','t_mid','v']}
    if not any(pd.isna(v) for v in ps.values()):
        ps['entity'] = row['entity']
        ps['n'] = row['n']
        ps['R2'] = row['R2']
        params[code] = ps

codes = sorted(params.keys())
print('Loaded: {} countries with valid Richards params'.format(len(codes)))

# ============================================================
# 2. Richards function and its derivative
# ============================================================
def richards(t, p):
    vc = max(p['v'], 0.01)
    z = vc * np.exp(-p['k'] * (t - p['t_mid']))
    z = np.clip(z, 1e-15, 1e15)
    base = np.clip(1.0 + z, 1e-15, 1e15)
    return p['S_floor'] + (p['S_ceil'] - p['S_floor']) / (base ** (1.0 / vc))

# Time grid: use each country's own observed range for distribution computation
# to avoid extrapolation artifacts. Then all on a common fine grid for pairwise comparison.
T_COMMON = np.linspace(1900, 2025, 126)  # 1-year steps, covers most data ranges

# ============================================================
# 3. Compute dS/dt distribution for each country
# ============================================================
print('Computing dS/dt distributions...')

# Use a finer grid for derivative computation
t_fine = np.linspace(1789, 2050, 262)  # extended range for tail behavior
dt = t_fine[1] - t_fine[0]

distributions = {}  # code -> np.array (probability mass at each t_fine point)

for code in codes:
    p = params[code]
    S = richards(t_fine, p)
    dS = np.gradient(S, dt)
    # Take absolute value? No — the curve is monotonic rising, so dS > 0 always.
    # But numerical issues or extreme params could cause tiny negatives.
    dS = np.maximum(dS, 0)
    total = dS.sum()
    if total > 1e-15:
        dist = dS / total
    else:
        dist = np.ones_like(dS) / len(dS)  # fallback, shouldn't happen
    distributions[code] = dist

print('  Done. Grid: {} points, dt={:.3f} yr'.format(len(t_fine), dt))

# ============================================================
# 4. Pairwise JS divergence -> distance matrix
# ============================================================
print('Computing pairwise JS divergence ({}×{} = {} pairs)...'.format(
    len(codes), len(codes), len(codes)*(len(codes)-1)//2))

n = len(codes)
dist_matrix = np.zeros((n, n))

for i in range(n):
    if (i+1) % 20 == 0:
        print('  row {}/{}...'.format(i+1, n))
    pi = distributions[codes[i]]
    for j in range(i+1, n):
        pj = distributions[codes[j]]
        # M = (P+Q)/2
        m = (pi + pj) / 2.0
        # JS = 0.5*KL(P||M) + 0.5*KL(Q||M)
        # KL with safe log
        eps = 1e-15
        kl_pm = np.sum(pi * np.log((pi + eps) / (m + eps)))
        kl_qm = np.sum(pj * np.log((pj + eps) / (m + eps)))
        js = 0.5 * kl_pm + 0.5 * kl_qm
        # Use sqrt(JS) as distance (metric, range [0, sqrt(ln 2)])
        dist_matrix[i, j] = np.sqrt(max(js, 0))
        dist_matrix[j, i] = dist_matrix[i, j]

print('  Done.')
print('  Distance range: [{:.6f}, {:.6f}]'.format(dist_matrix[np.triu_indices(n,1)].min(),
                                                    dist_matrix[np.triu_indices(n,1)].max()))

# ============================================================
# 5. Hierarchical clustering
# ============================================================
print('Clustering (Ward)...')
condensed = squareform(dist_matrix)
Z = linkage(condensed, method='ward')
print('  Done. {} leaves.'.format(len(Z)+1))

# ============================================================
# 6. Dendrogram
# ============================================================
print('Plotting dendrogram...')
fig, ax = plt.subplots(figsize=(40, 12))
labels = [params[c]['entity'] for c in codes]

dn = dendrogram(
    Z, labels=labels, leaf_font_size=6, ax=ax,
    color_threshold=0.7 * max(Z[:, 2]),
    above_threshold_color='#888888',
)

ax.set_title('Hierarchical Clustering of Richards Curves (JS-divergence, Ward linkage)\n{} countries'.format(n),
             fontsize=14, fontweight='bold')
ax.set_ylabel('Sqrt Jensen-Shannon distance', fontsize=12)
ax.tick_params(axis='x', labelsize=5)
plt.tight_layout()

png_path = r'data\_cluster_dendrogram.png'
fig.savefig(png_path, dpi=200, bbox_inches='tight')
plt.close()
print('  Saved: {}'.format(png_path))

# ============================================================
# 7. Export distance matrix + country list for external label analysis
# ============================================================
dist_df = pd.DataFrame(dist_matrix, index=codes, columns=codes)
csv_path = r'data\_js_distance_matrix.csv'
dist_df.to_csv(csv_path)
print('Distance matrix saved: {}'.format(csv_path))

# Also export country list with key params for label merging
info_rows = []
for code in codes:
    p = params[code]
    # Get observed year range
    ctry = panel[panel['code'] == code]
    yr_min = ctry['year'].min()
    yr_max = ctry['year'].max()
    info_rows.append({
        'code': code,
        'entity': p['entity'],
        'n_years': len(ctry),
        'year_start': int(yr_min) if not pd.isna(yr_min) else 0,
        'year_end': int(yr_max) if not pd.isna(yr_max) else 0,
        'S_floor': p['S_floor'],
        'S_ceil': p['S_ceil'],
        'k': p['k'],
        't_mid': p['t_mid'],
        'v': p['v'],
        'R2': p['R2'],
    })

info_df = pd.DataFrame(info_rows)
info_path = r'data\_cluster_country_info.csv'
info_df.to_csv(info_path, index=False, encoding='utf-8-sig')
print('Country info saved: {}'.format(info_path))

# ============================================================
# 8. Cluster stats
# ============================================================
print('\n=== CLUSTER STATS ===')
# Try cutting at different k
for k in [4, 6, 8, 10, 12]:
    clusters = fcluster(Z, k, criterion='maxclust')
    sizes = np.bincount(clusters)[1:]  # skip 0
    print('k={:2d}: sizes = {}'.format(k, sorted(sizes, reverse=True)))

# ============================================================
print('\nDone. Next: supply external labels (region, EDI tier, class proxy) to colour the dendrogram.')
print('Needed data columns per country: region, EDI, IDV, Gini, labor_share, polity_type')
