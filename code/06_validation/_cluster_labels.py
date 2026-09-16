# -*- coding: utf-8 -*-
"""Merge external labels onto the 191-country clustering, colour dendrogram, compute ARI."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load distance matrix + clustering
# ============================================================
dist_df = pd.read_csv(r'data\_js_distance_matrix.csv', index_col=0)
info_df = pd.read_csv(r'data\_cluster_country_info.csv')
codes = info_df['code'].tolist()
n = len(codes)
print('Loaded: {} countries, distance matrix {}×{}'.format(n, *dist_df.shape))

condensed = squareform(dist_df.values)
Z = linkage(condensed, method='ward')

# ============================================================
# 2. Load external labels
# ============================================================

# --- Region (OWID) ---
vdem = pd.read_csv(r'data\women-political-empowerment-index.csv')
region_map = vdem.groupby('Code')['World region according to OWID'].first().to_dict()
print('\nRegion: {} unique, {} codes'.format(len(set(region_map.values())), len(region_map)))

# --- EDI (latest) ---
edi_raw = pd.read_csv(r'data\electoral-democracy-index.csv')
# Take latest year per country
edi_latest = edi_raw.sort_values('Year').groupby('Code').last()
edi_map = edi_latest['Electoral democracy index'].to_dict()
print('EDI: {} codes'.format(len(edi_map)))

# --- IDV (Hofstede) ---
idv_df = pd.read_csv(r'data\_idv_edi_sceil_export.csv')
idv_map = dict(zip(idv_df['iso3'], idv_df['idv']))
print('IDV: {} codes'.format(len(idv_map)))

# --- Gini (SWIID, latest market) ---
gini_raw = pd.read_csv(r'data\swiid9_92_summary.csv')
# Take latest year per country
gini_latest = gini_raw.sort_values('year').groupby('country').last()
gini_map = gini_latest['gini_mkt'].to_dict()
print('Gini (market): {} countries (by name)'.format(len(gini_map)))

# --- Labor share ---
labsh_df = pd.read_csv(r'data\_class_laborshare_rank.csv')
labsh_map = dict(zip(labsh_df['code'], labsh_df['labsh_mean']))
print('Labor share: {} codes'.format(len(labsh_map)))

# --- Intergenerational mobility (BETA from GDIM) ---
gdim_raw = pd.read_csv(r'data\GDIM_2023_03.csv')
# Use BETA (intergenerational persistence): higher = less mobile
# Take most recent survey per country
gdim_recent = gdim_raw.sort_values('year').groupby('code').last()
beta_map = gdim_recent['BETA'].to_dict()
print('BETA (GDIM): {} codes'.format(len(beta_map)))

# --- GDP per capita (latest) ---
gdp_raw = pd.read_csv(r'data\gdp-per-capita-worldbank-constant-usd.csv')
# Wide format: Entity, Code, Year columns → take latest year per code
# Actually this might be wide format. Let me check.
print('GDP file columns: {}'.format(list(gdp_raw.columns)[:10]))

# ============================================================
# 3. Build merged label table
# ============================================================
label_rows = []
for code in codes:
    info = info_df[info_df['code'] == code].iloc[0]
    entity = info['entity']

    # Gini by entity name
    gini_val = None
    for name_key in [entity, entity.lower(), entity.upper()]:
        if name_key in gini_map:
            gini_val = gini_map[name_key]
            break

    row = {
        'code': code,
        'entity': entity,
        'region': region_map.get(code),
        'edi': edi_map.get(code),
        'idv': idv_map.get(code),
        'gini_mkt': gini_val,
        'labsh': labsh_map.get(code),
        'beta': beta_map.get(code),
    }
    label_rows.append(row)

labels_df = pd.DataFrame(label_rows)
print('\n--- Label coverage ---')
for col in ['region','edi','idv','gini_mkt','labsh','beta']:
    n_have = labels_df[col].notna().sum()
    print('  {}: {}/{} ({:.1f}%)'.format(col, n_have, len(labels_df), n_have/len(labels_df)*100))

# ============================================================
# 4. Categorize continuous variables into tiers
# ============================================================
def make_tiers(series, n_tiers=3, labels=None):
    """Split a continuous series into n_tiers quantile-based tiers."""
    valid = series.dropna().astype(float)
    if len(valid) < n_tiers + 1:  # need at least n_tiers+1 distinct values
        return pd.Series(['no_data'] * len(series), index=series.index)
    try:
        # Use pd.cut with quantile-based bins instead of pd.qcut
        bins = np.percentile(valid, np.linspace(0, 100, n_tiers + 1))
        # Ensure unique bins
        bins = np.unique(bins)
        if len(bins) < 2:
            return pd.Series(['no_data'] * len(series), index=series.index)
        n_actual = len(bins) - 1
        if labels is None:
            lbls = ['t{}'.format(i+1) for i in range(n_actual)]
        else:
            lbls = labels[:n_actual]
        tier_series = pd.cut(series.astype(float), bins=bins, labels=lbls, include_lowest=True)
        result = tier_series.astype(str).fillna('no_data')
        result[series.isna()] = 'no_data'
        return result
    except Exception:
        return pd.Series(['no_data'] * len(series), index=series.index)

labels_df['edi_tier'] = make_tiers(labels_df['edi'], 3, ['authoritarian','hybrid','democratic'])
labels_df['idv_tier'] = make_tiers(labels_df['idv'], 3, ['collectivist','mid','individualist'])
labels_df['gini_tier'] = make_tiers(labels_df['gini_mkt'], 3, ['low_inequality','mid','high_inequality'])
labels_df['labsh_tier'] = make_tiers(labels_df['labsh'], 3, ['low_labsh','mid','high_labsh'])
labels_df['beta_tier'] = make_tiers(labels_df['beta'], 3, ['high_mobility','mid','low_mobility'])

print('\n--- Tier distributions ---')
for col in ['edi_tier','idv_tier','gini_tier','labsh_tier','beta_tier']:
    print('  {}:'.format(col), labels_df[col].value_counts().to_dict())

# Save merged labels
labels_df.to_csv(r'data\_cluster_labels.csv',
                 index=False, encoding='utf-8-sig')
print('\nLabels saved: _cluster_labels.csv')

# ============================================================
# 5. Colour dendrogram by each external label
# ============================================================

# Color maps
cmaps = {
    'region': {
        'Asia': '#e41a1c', 'Europe': '#377eb8', 'Africa': '#4daf4a',
        'North America': '#984ea3', 'South America': '#ff7f00', 'Oceania': '#a65628',
    },
    'edi_tier': {
        'authoritarian': '#d7191c', 'hybrid': '#fdae61', 'democratic': '#2b83ba', 'no_data': '#cccccc',
    },
    'idv_tier': {
        'collectivist': '#d7191c', 'mid': '#fdae61', 'individualist': '#2b83ba', 'no_data': '#cccccc',
    },
    'gini_tier': {
        'low_inequality': '#2b83ba', 'mid': '#fdae61', 'high_inequality': '#d7191c', 'no_data': '#cccccc',
    },
    'labsh_tier': {
        'low_labsh': '#d7191c', 'mid': '#fdae61', 'high_labsh': '#2b83ba', 'no_data': '#cccccc',
    },
    'beta_tier': {
        'high_mobility': '#2b83ba', 'mid': '#fdae61', 'low_mobility': '#d7191c', 'no_data': '#cccccc',
    },
}

# Build colour lists ordered by dendrogram leaf order
# (We'll reconstruct dendrogram with leaf colours)
for label_col, cmap in cmaps.items():
    # Map code -> colour
    code_to_colour = {}
    for _, row in labels_df.iterrows():
        val = row[label_col]
        if pd.isna(val) or val not in cmap:
            code_to_colour[row['code']] = '#cccccc'
        else:
            code_to_colour[row['code']] = cmap[val]

    # Reorder colours to match dendrogram leaf order
    leaf_codes_order = []  # We'll extract from dendrogram
    # For now, build a standalone dendrogram

    fig, ax = plt.subplots(figsize=(40, 12))

    # Get leaf colours in dendrogram order
    dn = dendrogram(Z, labels=codes, leaf_font_size=5, ax=ax,
                    color_threshold=0.7*max(Z[:,2]),
                    above_threshold_color='#888888',
                    no_plot=False)
    # Set leaf colours from our map
    for label, (x, y) in zip(dn['ivl'], zip(dn['leaves'], [0]*len(dn['leaves']))):
        # Actually, let's use a different approach — colour the leaf labels
        pass

    plt.close()  # We'll just use the tree structure, rebuild properly below

# Build a clean dendrogram for each label
for label_col, cmap in cmaps.items():
    fig, ax = plt.subplots(figsize=(40, 12))

    # Get unique label values present
    label_vals = labels_df[label_col].fillna('no_data')

    # Create index mapping: code -> position in labels_df
    code_to_idx = {c: i for i, c in enumerate(codes)}

    # Build dendrogram and colour leaves
    dn = dendrogram(
        Z, labels=codes, leaf_font_size=5, ax=ax,
        color_threshold=0.7 * max(Z[:, 2]),
        above_threshold_color='#888888',
        no_plot=False,
        link_color_func=lambda k: '#888888',
    )

    # Colour the leaf labels
    ax2 = ax  # same axes
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        code = lbl.get_text()
        idx = code_to_idx.get(code)
        if idx is not None:
            val = label_vals.iloc[idx]
            colour = cmap.get(val, '#cccccc')
            lbl.set_color(colour)
            lbl.set_fontweight('bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=v) for v, c in cmap.items() if v in label_vals.values]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, title=label_col)

    ax.set_title('Hierarchical Clustering by {} — {} countries (JS-div, Ward)'.format(
        label_col, n), fontsize=14, fontweight='bold')
    ax.set_ylabel('Sqrt Jensen-Shannon distance', fontsize=12)
    plt.tight_layout()

    png_path = r'data\_cluster_dendrogram_{}.png'.format(label_col)
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close()
    print('Saved: {}'.format(png_path))

# ============================================================
# 6. ARI: adjusted Rand index quantifies agreement
# ============================================================
print('\n=== Adjusted Rand Index (clustering vs external labels) ===')
print('ARI = 0: random; ARI = 1: perfect match; ARI < 0: worse than random\n')

for k in [4, 5, 6, 7, 8]:
    clusters = fcluster(Z, k, criterion='maxclust')
    print('--- k={} ---'.format(k))
    for label_col in ['region','edi_tier','idv_tier','gini_tier','labsh_tier','beta_tier']:
        label_vals = labels_df[label_col].fillna('no_data')
        # Only use rows where label is available
        mask = label_vals != 'no_data'
        if mask.sum() < 10:
            print('  {:20s}: insufficient data ({})'.format(label_col, mask.sum()))
            continue
        ari = adjusted_rand_score(label_vals[mask], clusters[mask])
        print('  {:20s}: ARI = {:+.4f}  (n={})'.format(label_col, ari, mask.sum()))

print('\nDone.')
