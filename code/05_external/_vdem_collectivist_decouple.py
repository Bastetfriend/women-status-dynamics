# -*- coding: utf-8 -*-
"""
集体主义国家 vs 其他国家：经济-性别解耦对比
谢的直觉：集体主义国家显著压榨女性，解耦更严重
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
#  集体主义国家定义（冷战+现存社会主义）
# ═══════════════════════════════════════════════════════

# 两组：现存社会主义 + 前社会主义（苏东集团）
CURRENT_SOCIALIST = {
    'CHN': '🇨🇳 China',
    'PRK': '🇰🇵 North Korea',
    'CUB': '🇨🇺 Cuba',
    'VNM': '🇻🇳 Vietnam',
    'LAO': '🇱🇦 Laos',
}

FORMER_SOCIALIST = {
    'RUS': '🇷🇺 Russia',
    'POL': '🇵🇱 Poland',
    'CZE': '🇨🇿 Czechia',
    'SVK': '🇸🇰 Slovakia',
    'HUN': '🇭🇺 Hungary',
    'ROU': '🇷🇴 Romania',
    'BGR': '🇧🇬 Bulgaria',
    'ALB': '🇦🇱 Albania',
    'SRB': '🇷🇸 Serbia',
    'HRV': '🇭🇷 Croatia',
    'SVN': '🇸🇮 Slovenia',
    'BIH': '🇧🇦 Bosnia',
    'MKD': '🇲🇰 N. Macedonia',
    'MNE': '🇲🇪 Montenegro',
    'MNG': '🇲🇳 Mongolia',
    'UKR': '🇺🇦 Ukraine',
    'BLR': '🇧🇾 Belarus',
    'KAZ': '🇰🇿 Kazakhstan',
    'UZB': '🇺🇿 Uzbekistan',
    'TJK': '🇹🇯 Tajikistan',
    'KGZ': '🇰🇬 Kyrgyzstan',
    'TKM': '🇹🇲 Turkmenistan',
    'GEO': '🇬🇪 Georgia',
    'ARM': '🇦🇲 Armenia',
    'AZE': '🇦🇿 Azerbaijan',
    'EST': '🇪🇪 Estonia',
    'LVA': '🇱🇻 Latvia',
    'LTU': '🇱🇹 Lithuania',
    'MOZ': '🇲🇿 Mozambique',
    'AGO': '🇦🇴 Angola',
    'ETH': '🇪🇹 Ethiopia',
}

ALL_COLLECTIVIST = {**CURRENT_SOCIALIST, **FORMER_SOCIALIST}

print("=" * 70)
print("  集体主义 vs 其他国家：经济-性别解耦程度对比")
print("=" * 70)

# ═══════════════════════════════════════════════════════
#  Load data
# ═══════════════════════════════════════════════════════

wpei = pd.read_csv(f"{BASE}/women-political-empowerment-index.csv")
wpei.columns = ['entity', 'code', 'year', 'wpei', 'region']
wpei['S'] = wpei['wpei'] - 1

params = pd.read_csv(f"{BASE}/_vdem_fit_all.csv")
params = params[params['status'] == 'ok'].copy()

# Tag collectivist
params['group'] = params['code'].apply(
    lambda c: 'current_socialist' if c in CURRENT_SOCIALIST
    else 'former_socialist' if c in FORMER_SOCIALIST
    else 'other')
params['is_collectivist'] = params['code'].isin(ALL_COLLECTIVIST)

entity_map = wpei.drop_duplicates('code').set_index('code')['entity'].to_dict()
params['entity'] = params['code'].map(entity_map)

print(f"  Richards fits: {len(params)} countries")
print(f"  Current socialist: {(params['group']=='current_socialist').sum()}")
print(f"  Former socialist:  {(params['group']=='former_socialist').sum()}")
print(f"  Other:             {(params['group']=='other').sum()}")

# ═══════════════════════════════════════════════════════
#  1. Richards参数对比
# ═══════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  1. RICHARDS PARAMETERS BY GROUP")
print("═" * 70)

for col in ['ric_floor', 'ric_ceil', 'ric_k', 'ric_tmid', 'ric_v', 'ric_R2']:
    vals = {}
    for grp in ['current_socialist', 'former_socialist', 'other']:
        sub = params[params['group'] == grp][col].dropna()
        vals[grp] = sub

    cs = vals['current_socialist']
    fs = vals['former_socialist']
    ot = vals['other']

    # Mann-Whitney: collectivist vs other
    all_coll = pd.concat([cs, fs])
    if len(all_coll) > 3 and len(ot) > 3:
        u, p = stats.mannwhitneyu(all_coll, ot, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    else:
        p, sig = 1, ''

    print(f"\n  {col:>12}: current_soc={cs.median():.3f} (n={len(cs)}) | "
          f"former_soc={fs.median():.3f} (n={len(fs)}) | "
          f"other={ot.median():.3f} (n={len(ot)}) | MW p={p:.3f} {sig}")

# ═══════════════════════════════════════════════════════
#  2. 残差波动（解耦指标）
# ═══════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  2. RESIDUAL VOLATILITY (解耦程度)")
print("═" * 70)

def richards(t, S_floor, S_ceil, k, t_mid, v):
    z = v * np.exp(-k * (t - t_mid))
    z = np.clip(z, 1e-15, 1e15)
    return S_floor + (S_ceil - S_floor) / (1 + z) ** (1 / v)

country_stats = []
for _, p in params.iterrows():
    sub = wpei[wpei['code'] == p['code']].copy()
    if len(sub) < 10: continue
    sub['s_pred'] = richards(sub['year'], p['ric_floor'], p['ric_ceil'],
                              p['ric_k'], p['ric_tmid'], p['ric_v'])
    sub['residual'] = sub['S'] - sub['s_pred']

    country_stats.append({
        'code': p['code'],
        'entity': p.get('entity', p['code']),
        'group': p['group'],
        'is_collectivist': p['is_collectivist'],
        'res_std': sub['residual'].std(),
        'res_mad': np.median(np.abs(sub['residual'] - sub['residual'].median())),
        'res_range': sub['residual'].max() - sub['residual'].min(),
        'ric_R2': p['ric_R2'],
        'ric_ceil': p['ric_ceil'],
        'ric_floor': p['ric_floor'],
        'ric_tmid': p['ric_tmid'],
        'ric_v': p['ric_v'],
        'ceiling_gap': p['ric_ceil'] - sub['S'].iloc[-1],  # how far from ceiling
        'n_years': len(sub),
    })

cs_df = pd.DataFrame(country_stats)

for metric in ['res_std', 'res_mad', 'res_range']:
    coll = cs_df[cs_df['is_collectivist']][metric]
    other = cs_df[~cs_df['is_collectivist']][metric]
    u, p = stats.mannwhitneyu(coll, other, alternative='two-sided')
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    ratio = coll.median() / other.median()
    print(f"  {metric:>12}: collectivist={coll.median():.4f} | other={other.median():.4f} | "
          f"ratio={ratio:.2f}x | MW p={p:.3f} {sig}")

# ═══════════════════════════════════════════════════════
#  3. Ceiling（天花板）对比 —— 压榨指标
# ═══════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  3. CEILING & FLOOR — 压榨程度")
print("═" * 70)

# ric_ceil 接近0 = 天花板低 = 女性赋权上限被压住
# ric_floor 更低 = 起点更低
for metric, label in [('ric_ceil', '天花板(ceil)'), ('ric_floor', '地板(floor)'),
                       ('ceiling_gap', '距天花板差距')]:
    for grp, name in [('current_socialist', '现存社会主义'),
                       ('former_socialist', '前社会主义'),
                       ('other', '其他')]:
        sub = cs_df[cs_df['group'] == grp][metric]
        print(f"  {label:>16} | {name:>10}: median={sub.median():+.3f}, "
              f"mean={sub.mean():+.3f}, min={sub.min():+.3f}, max={sub.max():+.3f} (n={len(sub)})")

    coll = cs_df[cs_df['is_collectivist']][metric]
    other = cs_df[~cs_df['is_collectivist']][metric]
    u, p = stats.mannwhitneyu(coll, other, alternative='two-sided')
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"  {'MW test':>16} | collectivist vs other: p={p:.4f} {sig}")
    print()

# ═══════════════════════════════════════════════════════
#  4. Within-country economic-gender correlation
#     (from v3 merged data)
# ═══════════════════════════════════════════════════════

print("═" * 70)
print("  4. WITHIN-COUNTRY: economic indicator ~ residual")
print("     集体主义国家的经济-性别相关是否更弱/反向？")
print("═" * 70)

# Reload v3 merged data
WB_FILES = {
    'fem_labor':  ('API_SL.TLF.CACT.FE.ZS', 'Female labor participation (%)'),
    'fem_educ':   ('API_SE.SEC.ENRR.FE',     'Female secondary enrollment (%)'),
    'sex_ratio':  ('API_SP.POP.BRTH.MF',     'Sex ratio at birth (M/F)'),
}

def read_wb_csv(pattern, indicator_key):
    files = globmod.glob(f"{BASE}/{pattern}*.csv")
    files = [f for f in files if 'Metadata' not in f]
    if not files: return pd.DataFrame()
    df = pd.read_csv(files[0], skiprows=4)
    id_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
    year_cols = [c for c in df.columns if c not in id_cols and c.strip().isdigit()]
    long = df.melt(id_vars=['Country Code'], value_vars=year_cols,
                   var_name='year', value_name='value')
    long = long.rename(columns={'Country Code': 'code'})
    long['year'] = long['year'].astype(int)
    long['value'] = pd.to_numeric(long['value'], errors='coerce')
    long = long.dropna(subset=['value'])
    long['indicator'] = indicator_key
    return long

# WB uses alpha-3? Actually WB CSVs use alpha-3 country codes
print("\n  Loading mediator data...")
all_med = []
for key, (pattern, label) in WB_FILES.items():
    df = read_wb_csv(pattern, key)
    all_med.append(df)
med_df = pd.concat(all_med, ignore_index=True)

# Also load productive forces cache
PF_CACHE = f"{BASE}/_wb_indicators_cache.csv"
PRODUCTIVE = {'industry': 'Industry', 'services': 'Services', 'gdp_grow': 'GDP growth'}

pf_df = pd.DataFrame()
if os.path.exists(PF_CACHE):
    pf_raw = pd.read_csv(PF_CACHE)
    pf_sub = pf_raw[pf_raw['indicator'].isin(PRODUCTIVE.keys())].copy()
    # Map wb_code (alpha-2) to alpha-3
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

combined = pd.concat([med_df[['code', 'year', 'value', 'indicator']], pf_df], ignore_index=True)
pivot = combined.pivot_table(index=['code', 'year'], columns='indicator',
                              values='value', aggfunc='first').reset_index()

# Compute residuals
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
merged['group'] = merged['code'].apply(
    lambda c: 'collectivist' if c in ALL_COLLECTIVIST else 'other')

ALL_KEYS = list(WB_FILES.keys()) + list(PRODUCTIVE.keys())
AVAIL = [k for k in ALL_KEYS if k in merged.columns and merged[k].notna().sum() > 50]

# Within-country correlations by group
print(f"\n  {'Indicator':>12} | {'Collectivist mean_r':>20} | {'Other mean_r':>15} | {'Diff':>8} | MW test")
print("  " + "─" * 80)

within_data = {}
for key in AVAIL:
    for grp_label in ['collectivist', 'other']:
        crs = []
        subset = merged[merged['group'] == grp_label]
        for code, grp in subset.groupby('code'):
            sub = grp[['residual', key]].dropna()
            if len(sub) >= 8:
                r, p = stats.pearsonr(sub['residual'], sub[key])
                crs.append({'code': code, 'r': r, 'p': p})
        within_data[(key, grp_label)] = pd.DataFrame(crs) if crs else pd.DataFrame()

    coll_r = within_data.get((key, 'collectivist'), pd.DataFrame())
    other_r = within_data.get((key, 'other'), pd.DataFrame())

    if len(coll_r) > 3 and len(other_r) > 3:
        cm = coll_r['r'].mean()
        om = other_r['r'].mean()
        u, p = stats.mannwhitneyu(coll_r['r'], other_r['r'], alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {key:>12} | {cm:>+10.4f} (n={len(coll_r):>3}) | "
              f"{om:>+10.4f} (n={len(other_r):>3}) | {cm-om:>+8.4f} | p={p:.3f} {sig}")

# ═══════════════════════════════════════════════════════
#  5. 逐国详表：集体主义国家
# ═══════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  5. COLLECTIVIST COUNTRY DETAIL")
print("═" * 70)

print(f"\n  {'Code':<5} {'Country':<22} {'ceil':>6} {'floor':>6} {'R2':>5} "
      f"{'t_mid':>6} {'v':>5} {'res_std':>8} {'gap':>6}")
print("  " + "-" * 80)

for _, row in cs_df[cs_df['is_collectivist']].sort_values('group').iterrows():
    grp_tag = '*' if row['code'] in CURRENT_SOCIALIST else ' '
    print(f"  {row['code']:<5}{grp_tag} {str(row['entity'])[:20]:<22} "
          f"{row['ric_ceil']:>+6.3f} {row['ric_floor']:>+6.3f} {row['ric_R2']:>5.3f} "
          f"{row['ric_tmid']:>6.0f} {row['ric_v']:>5.2f} {row['res_std']:>8.4f} "
          f"{row['ceiling_gap']:>+6.3f}")

# Benchmark: median of "other" countries
ot = cs_df[~cs_df['is_collectivist']]
print(f"\n  {'':5}  {'── OTHER median ──':<22} "
      f"{ot['ric_ceil'].median():>+6.3f} {ot['ric_floor'].median():>+6.3f} {ot['ric_R2'].median():>5.3f} "
      f"{ot['ric_tmid'].median():>6.0f} {ot['ric_v'].median():>5.2f} {ot['res_std'].median():>8.4f} "
      f"{ot['ceiling_gap'].median():>+6.3f}")

# ═══════════════════════════════════════════════════════
#  6. FIGURE: 4-panel comparison
# ═══════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Ceiling comparison
ax = axes[0, 0]
for i, (grp, color, label) in enumerate([
    ('current_socialist', '#e41a1c', 'Current socialist'),
    ('former_socialist', '#ff7f00', 'Former socialist'),
    ('other', '#377eb8', 'Other')]):
    vals = cs_df[cs_df['group'] == grp]['ric_ceil']
    bp = ax.boxplot([vals], positions=[i], widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor=color, alpha=0.5),
                     medianprops=dict(color='black', linewidth=2))
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Current\nsocialist', 'Former\nsocialist', 'Other'], fontsize=10)
ax.set_ylabel('Richards ceiling (S)')
ax.set_title('(a) Gender empowerment ceiling', fontsize=12)
ax.axhline(0, color='gray', linewidth=0.5, ls='--')

# (b) Residual std comparison
ax = axes[0, 1]
for i, (grp, color, label) in enumerate([
    ('current_socialist', '#e41a1c', 'Current socialist'),
    ('former_socialist', '#ff7f00', 'Former socialist'),
    ('other', '#377eb8', 'Other')]):
    vals = cs_df[cs_df['group'] == grp]['res_std']
    bp = ax.boxplot([vals], positions=[i], widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor=color, alpha=0.5),
                     medianprops=dict(color='black', linewidth=2))
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['Current\nsocialist', 'Former\nsocialist', 'Other'], fontsize=10)
ax.set_ylabel('Residual σ (decoupling strength)')
ax.set_title('(b) Economic-gender decoupling', fontsize=12)

# (c) Within-country r for fem_labor by group — scatter
ax = axes[1, 0]
for grp_label, color, marker in [('collectivist', '#e41a1c', 's'),
                                   ('other', '#377eb8', 'o')]:
    crs = within_data.get(('fem_labor', grp_label), pd.DataFrame())
    if len(crs) > 0:
        ax.scatter(range(len(crs)), crs['r'].sort_values().values,
                   c=color, alpha=0.5, s=20, marker=marker, label=grp_label)
ax.axhline(0, color='gray', linewidth=0.5)
ax.set_ylabel('Within-country r (residual ~ fem_labor)')
ax.set_title('(c) Female labor ↔ gender empowerment residual', fontsize=12)
ax.legend(fontsize=9)

# (d) Ceiling vs floor: collectivist highlighted
ax = axes[1, 1]
other_sub = cs_df[~cs_df['is_collectivist']]
coll_sub = cs_df[cs_df['is_collectivist']]
ax.scatter(other_sub['ric_floor'], other_sub['ric_ceil'], c='#377eb8', alpha=0.3,
           s=15, label='Other')
ax.scatter(coll_sub['ric_floor'], coll_sub['ric_ceil'], c='#e41a1c', alpha=0.8,
           s=40, marker='s', label='Collectivist', edgecolors='black', linewidth=0.5)
# Label notable ones
for _, row in coll_sub.iterrows():
    if row['code'] in ['CHN', 'PRK', 'CUB', 'RUS', 'VNM']:
        ax.annotate(row['code'], (row['ric_floor'], row['ric_ceil']),
                    fontsize=8, fontweight='bold', color='#e41a1c',
                    xytext=(5, 5), textcoords='offset points')
ax.set_xlabel('Richards floor (S)')
ax.set_ylabel('Richards ceiling (S)')
ax.set_title('(d) Floor vs Ceiling: collectivist countries', fontsize=12)
ax.legend(fontsize=9)
ax.plot([-1, 0], [-1, 0], 'k--', alpha=0.2)  # diagonal

plt.suptitle('Collectivist vs Other Countries: Gender–Economy Decoupling',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{BASE}/_vdem_collectivist_decouple.png", dpi=200, bbox_inches='tight')
print(f"\n  Saved: _vdem_collectivist_decouple.png")

# ═══════════════════════════════════════════════════════
#  7. SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  SUMMARY")
print("═" * 70)

coll = cs_df[cs_df['is_collectivist']]
other = cs_df[~cs_df['is_collectivist']]

# Effect sizes
for metric, label in [('ric_ceil', 'Ceiling'), ('ric_floor', 'Floor'),
                       ('res_std', 'Residual σ'), ('ceiling_gap', 'Gap to ceiling')]:
    c_med = coll[metric].median()
    o_med = other[metric].median()
    # Cohen's d
    pooled_std = np.sqrt((coll[metric].var() * (len(coll)-1) +
                          other[metric].var() * (len(other)-1)) /
                         (len(coll) + len(other) - 2))
    d = (coll[metric].mean() - other[metric].mean()) / (pooled_std + 1e-10)
    u, p = stats.mannwhitneyu(coll[metric], other[metric], alternative='two-sided')
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"  {label:>16}: coll={c_med:+.3f} | other={o_med:+.3f} | "
          f"Cohen's d={d:+.3f} | p={p:.4f} {sig}")

print("\n  Done!")
