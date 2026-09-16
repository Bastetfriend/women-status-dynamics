# -*- coding: utf-8 -*-
"""
按 within-country 经济→性别相关程度排序全量191国
然后看这个相关程度跟EDI的关系
谢的逻辑：不是排天花板，是排"经济能不能推动性别"
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob as globmod, warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = r'data'

# ═══ Load all data ═══
params = pd.read_csv(f'{BASE}/_vdem_fit_all.csv')
params = params[params['status'] == 'ok'].copy()

wpei = pd.read_csv(f'{BASE}/women-political-empowerment-index.csv')
wpei.columns = ['entity', 'code', 'year', 'wpei', 'region']
wpei['S'] = wpei['wpei'] - 1

edi_raw = pd.read_csv(f'{BASE}/electoral-democracy-index.csv')
edi_raw.columns = ['entity', 'code', 'year', 'edi', 'region']
edi_mean = edi_raw.groupby('code')['edi'].mean().rename('edi_mean')
edi_latest = edi_raw.sort_values('year').groupby('code').last()[['edi']].rename(
    columns={'edi': 'edi_latest'})

# ═══ Load World Bank indicators ═══
WB = {
    'fem_labor': 'API_SL.TLF.CACT.FE.ZS',
    'fem_educ':  'API_SE.SEC.ENRR.FE',
    'sex_ratio': 'API_SP.POP.BRTH.MF',
}

def load_wb(pattern, key):
    files = [f for f in globmod.glob(f'{BASE}/{pattern}*.csv') if 'Metadata' not in f]
    if not files: return pd.DataFrame()
    df = pd.read_csv(files[0], skiprows=4)
    id_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
    ycols = [c for c in df.columns if c not in id_cols and c.strip().isdigit()]
    long = df.melt(id_vars=['Country Code'], value_vars=ycols,
                   var_name='year', value_name='value')
    long = long.rename(columns={'Country Code': 'code'})
    long['year'] = long['year'].astype(int)
    long['value'] = pd.to_numeric(long['value'], errors='coerce')
    long = long.dropna(subset=['value'])
    long = long.rename(columns={'value': key})
    return long[['code', 'year', key]]

wb_dfs = {k: load_wb(v, k) for k, v in WB.items()}

# Also load productive forces from v2 cache
pf_cache = f'{BASE}/_wb_indicators_cache.csv'
pf_keys = ['industry', 'services', 'gdp_grow']
pf_labels = {
    'industry': 'Industry %GDP',
    'services': 'Services %GDP',
    'gdp_grow': 'GDP growth',
    'fem_labor': 'Female labor',
    'fem_educ': 'Female education',
    'sex_ratio': 'Sex ratio at birth',
}

import os
if os.path.exists(pf_cache):
    pf_raw = pd.read_csv(pf_cache)
    # Map wb_code to OWID code
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
    for _, row in pf_raw.drop_duplicates('wb_code').iterrows():
        e, c = row['entity'], row['wb_code']
        if e in MANUAL: wb2owid[c] = MANUAL[e]; continue
        k = e.strip().lower()
        if k in owid_e2c: wb2owid[c] = owid_e2c[k]; continue
        for ok, oc in owid_e2c.items():
            if k in ok or ok in k: wb2owid[c] = oc; break
    pf_raw['code'] = pf_raw['wb_code'].map(wb2owid)
    for pk in pf_keys:
        sub = pf_raw[pf_raw['indicator'] == pk].dropna(subset=['code'])
        wb_dfs[pk] = sub[['code', 'year', 'value']].rename(columns={'value': pk})

# ═══ Richards residuals ═══
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

# ═══ Merge everything ═══
merged = res_df.copy()
for key, wdf in wb_dfs.items():
    merged = merged.merge(wdf, on=['code', 'year'], how='left')

entity_map = wpei.drop_duplicates('code').set_index('code')['entity'].to_dict()
ALL_KEYS = ['fem_labor', 'fem_educ', 'services', 'industry', 'gdp_grow']

# ═══ Per-country within-country correlations ═══
country_corrs = []
for code, grp in merged.groupby('code'):
    row = {'code': code, 'entity': entity_map.get(code, code)}
    has_any = False
    rs_for_avg = []
    for key in ALL_KEYS:
        sub = grp[['residual', key]].dropna()
        if len(sub) >= 8:
            r, p = stats.pearsonr(sub['residual'], sub[key])
            row[f'r_{key}'] = r
            row[f'p_{key}'] = p
            row[f'n_{key}'] = len(sub)
            has_any = True
            if key in ['fem_labor', 'fem_educ', 'services']:
                rs_for_avg.append(r)
        else:
            row[f'r_{key}'] = np.nan
            row[f'p_{key}'] = np.nan
            row[f'n_{key}'] = 0
    if rs_for_avg:
        row['r_avg_3'] = np.mean(rs_for_avg)
    else:
        row['r_avg_3'] = np.nan
    if has_any:
        country_corrs.append(row)

cc = pd.DataFrame(country_corrs)
cc = cc.merge(edi_mean, on='code', how='left')
cc = cc.merge(edi_latest, on='code', how='left')
cc = cc.merge(params[['code', 'ric_ceil', 'ric_k', 'S_now']], on='code', how='left')

# ═══ Sort by r_avg_3 (mean of fem_labor, fem_educ, services ~ residual) ═══
cc_sorted = cc.dropna(subset=['r_avg_3']).sort_values('r_avg_3')

print('=' * 100)
print('  191 countries sorted by MEAN r(fem_labor, fem_educ, services ~ residual)')
print('  = "经济/教育能不能推动性别转型"')
print('=' * 100)
print(f"  {'#':>3} {'Code':<6} {'Entity':<25} {'r_avg':>6} "
      f"{'r_flab':>7} {'r_fedu':>7} {'r_serv':>7} {'EDI':>6} {'S_now':>6}")
print('  ' + '─' * 90)

for i, (_, r) in enumerate(cc_sorted.iterrows(), 1):
    def fmt(v): return f'{v:+.3f}' if pd.notna(v) else '   N/A'
    edi = f'{r["edi_latest"]:.3f}' if pd.notna(r['edi_latest']) else ' N/A'
    sn = f'{r["S_now"]:+.3f}' if pd.notna(r['S_now']) else ' N/A'
    print(f"  {i:>3} {r['code']:<6} {str(r['entity'])[:25]:<25} {r['r_avg_3']:>+6.3f} "
          f"{fmt(r.get('r_fem_labor')):>7} {fmt(r.get('r_fem_educ')):>7} "
          f"{fmt(r.get('r_services')):>7} {edi:>6} {sn:>6}")

# ═══ Correlation: r_avg_3 vs EDI ═══
valid = cc_sorted.dropna(subset=['edi_latest', 'r_avg_3'])
if len(valid) > 10:
    r_corr, p_corr = stats.pearsonr(valid['edi_latest'], valid['r_avg_3'])
    sr_corr, sp_corr = stats.spearmanr(valid['edi_latest'], valid['r_avg_3'])

    print(f'\n  ═══ KEY: EDI vs "经济能否推动性别" ═══')
    print(f'  Pearson:  r={r_corr:+.4f}, p={p_corr:.2e}, n={len(valid)}')
    print(f'  Spearman: ρ={sr_corr:+.4f}, p={sp_corr:.2e}')
    print(f'  R² = {r_corr**2:.4f} = {r_corr**2*100:.1f}% variance explained')

    # Bottom vs Top quartile
    q1 = valid['r_avg_3'].quantile(0.25)
    q3 = valid['r_avg_3'].quantile(0.75)
    bot_q = valid[valid['r_avg_3'] <= q1]
    top_q = valid[valid['r_avg_3'] >= q3]
    print(f'\n  Bottom quartile (经济推不动): mean EDI = {bot_q["edi_latest"].mean():.3f}, n={len(bot_q)}')
    print(f'  Top quartile (经济推得动):    mean EDI = {top_q["edi_latest"].mean():.3f}, n={len(top_q)}')
    u, p = stats.mannwhitneyu(bot_q['edi_latest'].dropna(), top_q['edi_latest'].dropna())
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f'  Mann-Whitney: U={u:.0f}, p={p:.4f} {sig}')

# ═══ Scatter: EDI vs r_avg_3 ═══
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
v = valid.copy()
ax.scatter(v['edi_latest'], v['r_avg_3'], alpha=0.5, s=25, c='steelblue')
# Label extremes
for _, r in v.nsmallest(8, 'r_avg_3').iterrows():
    ax.annotate(r['code'], (r['edi_latest'], r['r_avg_3']), fontsize=7, alpha=0.8)
for _, r in v.nlargest(5, 'r_avg_3').iterrows():
    ax.annotate(r['code'], (r['edi_latest'], r['r_avg_3']), fontsize=7, alpha=0.8)
z = np.polyfit(v['edi_latest'], v['r_avg_3'], 1)
xline = np.linspace(0, 0.95, 100)
ax.plot(xline, np.polyval(z, xline), 'r--', alpha=0.6)
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax.set_xlabel('Electoral Democracy Index (latest)', fontsize=11)
ax.set_ylabel('Mean r(fem_labor, fem_educ, services ~ residual)', fontsize=11)
ax.set_title(f'EDI vs "Can economy drive gender transition?"  (r={r_corr:+.3f})', fontsize=11)

# Individual indicators
ax = axes[1]
colors = {'r_fem_labor': '#e41a1c', 'r_fem_educ': '#377eb8', 'r_services': '#4daf4a'}
labels_plot = {'r_fem_labor': 'Female labor', 'r_fem_educ': 'Female education', 'r_services': 'Services %GDP'}
for key, color in colors.items():
    sub = cc.dropna(subset=[key, 'edi_latest'])
    if len(sub) > 10:
        r_k, p_k = stats.pearsonr(sub['edi_latest'], sub[key])
        ax.scatter(sub['edi_latest'], sub[key], alpha=0.3, s=15, c=color,
                   label=f'{labels_plot[key]} (r={r_k:+.3f})')
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax.legend(fontsize=9)
ax.set_xlabel('Electoral Democracy Index (latest)', fontsize=11)
ax.set_ylabel('Within-country r(indicator ~ residual)', fontsize=11)
ax.set_title('Per-indicator: EDI vs transmission strength', fontsize=11)

plt.tight_layout()
plt.savefig(f'{BASE}/_vdem_edi_vs_transmission.png', dpi=200, bbox_inches='tight')
print(f'\n  Saved: _vdem_edi_vs_transmission.png')
