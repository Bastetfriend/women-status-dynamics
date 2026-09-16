# -*- coding: utf-8 -*-
"""
M(E(t)) v3 — 中介变量分析（本地CSV版）
女性受教育率 / 性别比 / 女性劳动参与率 作为生产力→S(t)的中介通道
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings, os, glob as globmod
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = r"data"

# ═══════════════════════════════════════════════════════
#  1. Load WPEI + Richards
# ═══════════════════════════════════════════════════════

print("=" * 70)
print("  M(E(t)) v3 — Mediating Variables Analysis (local CSV)")
print("=" * 70)

wpei = pd.read_csv(f"{BASE}/women-political-empowerment-index.csv")
wpei.columns = ['entity', 'code', 'year', 'wpei', 'region']
wpei['S'] = wpei['wpei'] - 1

params = pd.read_csv(f"{BASE}/_vdem_fit_all.csv")
params = params[params['status'] == 'ok']
print(f"  WPEI: {wpei['code'].nunique()} countries | Richards: {len(params)}")

# ═══════════════════════════════════════════════════════
#  2. Read World Bank CSV files (wide → long)
# ═══════════════════════════════════════════════════════

WB_FILES = {
    'fem_labor':  ('API_SL.TLF.CACT.FE.ZS', 'Female labor participation (%)'),
    'fem_educ':   ('API_SE.SEC.ENRR.FE',     'Female secondary enrollment (%)'),
    'sex_ratio':  ('API_SP.POP.BRTH.MF',     'Sex ratio at birth (M/F)'),
}

def read_wb_csv(pattern, indicator_key):
    """Read World Bank wide CSV → long format DataFrame."""
    files = globmod.glob(f"{BASE}/{pattern}*.csv")
    # Skip metadata files
    files = [f for f in files if 'Metadata' not in f]
    if not files:
        print(f"    {indicator_key}: FILE NOT FOUND ({pattern})")
        return pd.DataFrame()

    df = pd.read_csv(files[0], skiprows=4)
    # Columns: Country Name, Country Code, Indicator Name, Indicator Code, 1960, 1961, ...
    id_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
    year_cols = [c for c in df.columns if c not in id_cols and c.strip().isdigit()]

    long = df.melt(id_vars=['Country Code'], value_vars=year_cols,
                   var_name='year', value_name='value')
    long = long.rename(columns={'Country Code': 'code'})
    long['year'] = long['year'].astype(int)
    long['value'] = pd.to_numeric(long['value'], errors='coerce')
    long = long.dropna(subset=['value'])
    long['indicator'] = indicator_key
    print(f"    {indicator_key}: {len(long)} values, {long['code'].nunique()} countries, "
          f"{long['year'].min()}-{long['year'].max()}")
    return long

print("\n  Loading World Bank CSV files...")
all_med = []
for key, (pattern, label) in WB_FILES.items():
    df = read_wb_csv(pattern, key)
    all_med.append(df)

med_df = pd.concat(all_med, ignore_index=True)
print(f"  Total mediator data: {len(med_df)} rows")

# Also load productive forces from v2 cache for comparison
PF_CACHE = f"{BASE}/_wb_indicators_cache.csv"
PRODUCTIVE = {
    'industry': 'Industry (% GDP)',
    'services': 'Services (% GDP)',
    'gdp_grow': 'GDP growth (%)',
}

pf_df = pd.DataFrame()
if os.path.exists(PF_CACHE):
    pf_raw = pd.read_csv(PF_CACHE)
    pf_sub = pf_raw[pf_raw['indicator'].isin(PRODUCTIVE.keys())].copy()
    # v2 cache has wb_code (alpha-2); need to map to alpha-3
    # But med_df already has alpha-3 codes — let's map pf_sub
    owid_e2c = {}
    for _, row in wpei.drop_duplicates('code').iterrows():
        owid_e2c[row['entity'].strip().lower()] = row['code']

    MANUAL = {
        'Korea, Rep.': 'KOR', "Korea, Dem. People's Rep.": 'PRK',
        'Russian Federation': 'RUS', 'Iran, Islamic Rep.': 'IRN',
        'Egypt, Arab Rep.': 'EGY', 'Venezuela, RB': 'VEN',
        'Turkiye': 'TUR', 'Lao PDR': 'LAO', 'Kyrgyz Republic': 'KGZ',
        'Slovak Republic': 'SVK', 'Czechia': 'CZE',
        'Congo, Rep.': 'COG', 'Congo, Dem. Rep.': 'COD',
        'Gambia, The': 'GMB', 'Bahamas, The': 'BHS',
        'Yemen, Rep.': 'YEM', 'Syrian Arab Republic': 'SYR',
        'Eswatini': 'SWZ', 'North Macedonia': 'MKD',
        'Myanmar': 'MMR', "Cote d'Ivoire": 'CIV',
    }

    wb2owid = {}
    for _, row in pf_sub.drop_duplicates('wb_code').iterrows():
        e, c = row['entity'], row['wb_code']
        if e in MANUAL: wb2owid[c] = MANUAL[e]; continue
        k = e.strip().lower()
        if k in owid_e2c: wb2owid[c] = owid_e2c[k]; continue
        for ok, oc in owid_e2c.items():
            if k in ok or ok in k: wb2owid[c] = oc; break

    pf_sub['code'] = pf_sub['wb_code'].map(wb2owid)
    pf_df = pf_sub.dropna(subset=['code'])[['code', 'year', 'value', 'indicator']]
    print(f"  Productive forces (from v2 cache): {len(pf_df)} rows")

# Combine and pivot
combined = pd.concat([med_df[['code', 'year', 'value', 'indicator']], pf_df], ignore_index=True)
pivot = combined.pivot_table(index=['code', 'year'], columns='indicator',
                              values='value', aggfunc='first').reset_index()

ALL_KEYS = list(WB_FILES.keys()) + list(PRODUCTIVE.keys())
ALL_LABELS = {k: WB_FILES[k][1] if k in WB_FILES else PRODUCTIVE[k] for k in ALL_KEYS}

print(f"\n  Pivoted: {len(pivot)} country-year rows")
for key in ALL_KEYS:
    if key in pivot.columns:
        print(f"    {key:>12}: {pivot[key].notna().sum()} values")

# ═══════════════════════════════════════════════════════
#  3. Richards residuals + merge
# ═══════════════════════════════════════════════════════

def richards(t, S_floor, S_ceil, k, t_mid, v):
    z = v * np.exp(-k * (t - t_mid))
    z = np.clip(z, 1e-15, 1e15)
    return S_floor + (S_ceil - S_floor) / (1 + z) ** (1 / v)

residuals = []
for _, p in params.iterrows():
    for _, row in wpei[wpei['code'] == p['code']].iterrows():
        s_pred = richards(row['year'], p['ric_floor'], p['ric_ceil'],
                          p['ric_k'], p['ric_tmid'], p['ric_v'])
        residuals.append({'code': p['code'], 'year': row['year'],
                          'residual': row['S'] - s_pred})

res_df = pd.DataFrame(residuals)
merged = res_df.merge(pivot, on=['code', 'year'], how='inner')
merged = merged.replace([np.inf, -np.inf], np.nan)

entity_map = wpei.drop_duplicates('code').set_index('code')['entity'].to_dict()
merged['entity'] = merged['code'].map(entity_map)

AVAIL_MED = [k for k in WB_FILES if k in merged.columns and merged[k].notna().sum() > 50]
AVAIL_PF = [k for k in PRODUCTIVE if k in merged.columns and merged[k].notna().sum() > 50]
print(f"\n  Merged: {len(merged)} rows, {merged['code'].nunique()} countries")
print(f"  Mediators: {AVAIL_MED} | Productive: {AVAIL_PF}")

# ═══════════════════════════════════════════════════════
#  4. TEST A: Residual ~ each indicator
# ═══════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  TEST A: Richards residual ~ indicator")
print("═" * 70)

results_a = {}
print(f"\n  {'Indicator':>15} {'Category':>10}  {'r':>8}  {'p':>10}  {'n':>6}")
print("  " + "─" * 60)

for key in AVAIL_MED + AVAIL_PF:
    sub = merged[['residual', key]].dropna()
    if len(sub) < 50: continue
    r, p = stats.pearsonr(sub['residual'], sub[key])
    cat = "MEDIATOR" if key in WB_FILES else "productive"
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"  {key:>15} {cat:>10}  {r:>+8.4f}  {p:>10.1e}  {len(sub):>6} {sig}")
    results_a[key] = {'r': r, 'p': p, 'n': len(sub), 'cat': cat}

# ═══════════════════════════════════════════════════════
#  5. TEST B: Within-country correlations
# ═══════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  TEST B: Within-country correlations (fixed effects)")
print("═" * 70)

within = {}
for key in AVAIL_MED + AVAIL_PF:
    crs = []
    for code, grp in merged.groupby('code'):
        sub = grp[['residual', key]].dropna()
        if len(sub) >= 10:
            r, p = stats.pearsonr(sub['residual'], sub[key])
            crs.append({'code': code, 'entity': grp['entity'].iloc[0], 'r': r, 'p': p})
    if not crs: continue
    cdf = pd.DataFrame(crs)
    within[key] = cdf
    cat = "MED" if key in WB_FILES else "pf"
    ppos = (cdf['r'] > 0).mean() * 100
    psigpos = ((cdf['p'] < 0.05) & (cdf['r'] > 0)).mean() * 100
    print(f"  {key:>12} [{cat:>3}]: mean_r={cdf['r'].mean():+.4f}, "
          f"{ppos:.0f}% pos, {psigpos:.0f}% sig+pos ({len(cdf)} countries)")

# ═══════════════════════════════════════════════════════
#  6. TEST C: Mediation chain — productive → mediators
# ═══════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  TEST C: Productive forces → mediators (mediation first leg)")
print("═" * 70)

for pf_key in AVAIL_PF:
    for med_key in AVAIL_MED:
        sub = merged[[pf_key, med_key]].dropna()
        if len(sub) < 100: continue
        r, p = stats.pearsonr(sub[pf_key], sub[med_key])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {pf_key:>12} → {med_key:<12}: r={r:+.4f}, p={p:.1e}, n={len(sub)} {sig}")

# ═══════════════════════════════════════════════════════
#  7. Outlier check
# ═══════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  OUTLIER CHECK — latest available values + within-country r")
print("═" * 70)

flagged = ['CHN', 'PRK', 'SAU', 'KOR', 'CUB', 'IRN', 'USA', 'NOR', 'RUS', 'JPN']
for code in flagged:
    sub = merged[merged['code'] == code]
    if len(sub) == 0:
        print(f"  {code}: no data")
        continue
    entity = sub['entity'].iloc[0]
    latest = sub.sort_values('year').iloc[-1]

    vals = []
    for k in AVAIL_MED:
        v = latest.get(k, np.nan)
        short = k.replace('fem_', 'f_').replace('sex_ratio', 'sx')
        vals.append(f"{short}={v:.1f}" if not np.isnan(v) else f"{short}=N/A")

    wcorrs = []
    for k in AVAIL_MED:
        s = sub[['residual', k]].dropna()
        if len(s) >= 5:
            r, p = stats.pearsonr(s['residual'], s[k])
            short = k.replace('fem_', 'f_').replace('sex_ratio', 'sx')
            sig = '*' if p < 0.05 else ''
            wcorrs.append(f"r({short})={r:+.3f}{sig}")

    print(f"  {code} {entity[:20]:<20} | {' | '.join(vals)}")
    if wcorrs:
        print(f"     {'':20} | {', '.join(wcorrs)}")

# ═══════════════════════════════════════════════════════
#  8. Visualize: bar chart comparison
# ═══════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) Effect size bar chart
ax = axes[0]
bar_keys = AVAIL_MED + AVAIL_PF
bar_r = [results_a[k]['r'] if k in results_a else 0 for k in bar_keys]
bar_colors = ['#e41a1c' if k in WB_FILES else '#377eb8' for k in bar_keys]
bar_labels = [ALL_LABELS.get(k, k).split('(')[0].strip() for k in bar_keys]

ax.bar(range(len(bar_keys)), bar_r, color=bar_colors, edgecolor='white')
ax.set_xticks(range(len(bar_keys)))
ax.set_xticklabels(bar_labels, rotation=25, ha='right', fontsize=9)
ax.set_ylabel('Pearson r with Richards residual')
ax.set_title('(a) Mediators (red) vs Productive Forces (blue)')
ax.axhline(0, color='gray', linewidth=0.5)
for i, (k, r) in enumerate(zip(bar_keys, bar_r)):
    p_val = results_a[k]['p'] if k in results_a else 1
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax.text(i, r + (0.005 if r >= 0 else -0.012), sig, ha='center', fontsize=9)

# (b) Within-country mean r comparison
ax = axes[1]
wbar_keys = [k for k in AVAIL_MED + AVAIL_PF if k in within]
wbar_r = [within[k]['r'].mean() for k in wbar_keys]
wbar_colors = ['#e41a1c' if k in WB_FILES else '#377eb8' for k in wbar_keys]
wbar_labels = [ALL_LABELS.get(k, k).split('(')[0].strip() for k in wbar_keys]

ax.bar(range(len(wbar_keys)), wbar_r, color=wbar_colors, edgecolor='white')
ax.set_xticks(range(len(wbar_keys)))
ax.set_xticklabels(wbar_labels, rotation=25, ha='right', fontsize=9)
ax.set_ylabel('Within-country mean Pearson r')
ax.set_title('(b) Within-country: Mediators (red) vs Productive Forces (blue)')
ax.axhline(0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig(f"{BASE}/_vdem_mediators.png", dpi=200, bbox_inches='tight')
print(f"\n  Saved: _vdem_mediators.png")

# ═══════════════════════════════════════════════════════
#  9. Summary
# ═══════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  SUMMARY")
print("═" * 70)

med_rs = [abs(results_a[k]['r']) for k in AVAIL_MED if k in results_a]
pf_rs = [abs(results_a[k]['r']) for k in AVAIL_PF if k in results_a]

if med_rs:
    print(f"\n  Mean |r| mediators:          {np.mean(med_rs):.4f}")
if pf_rs:
    print(f"  Mean |r| productive forces:  {np.mean(pf_rs):.4f}")
if med_rs and pf_rs:
    ratio = np.mean(med_rs) / (np.mean(pf_rs) + 1e-10)
    print(f"  Ratio (mediators/productive): {ratio:.2f}x")

print("\n  Mediation chain:")
print("  productive forces ──(TEST C)──→ female indicators ──(TEST A)──→ S(t) residual")
print("  productive forces ─────────────(TEST A, weak)────────────────→ S(t) residual")

print()
