# -*- coding: utf-8 -*-
"""
直接比：经济指标 vs S(t)，不过Richards残差
然后按相关程度排序，看EDI的关系
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob as globmod, os, warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = r'data'

# ═══ Load ═══
wpei = pd.read_csv(f'{BASE}/women-political-empowerment-index.csv')
wpei.columns = ['entity', 'code', 'year', 'wpei', 'region']
wpei['S'] = wpei['wpei'] - 1

edi_raw = pd.read_csv(f'{BASE}/electoral-democracy-index.csv')
edi_raw.columns = ['entity', 'code', 'year', 'edi', 'region']
edi_latest = edi_raw.sort_values('year').groupby('code').last()[['edi']].rename(
    columns={'edi': 'edi_latest'})

# World Bank indicators
def load_wb(pattern, key):
    files = [f for f in globmod.glob(f'{BASE}/{pattern}*.csv') if 'Metadata' not in f]
    if not files: return pd.DataFrame()
    df = pd.read_csv(files[0], skiprows=4)
    id_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
    ycols = [c for c in df.columns if c not in id_cols and c.strip().isdigit()]
    long = df.melt(id_vars=['Country Code'], value_vars=ycols,
                   var_name='year', value_name=key)
    long = long.rename(columns={'Country Code': 'code'})
    long['year'] = long['year'].astype(int)
    long[key] = pd.to_numeric(long[key], errors='coerce')
    return long.dropna(subset=[key])[['code', 'year', key]]

wb = {}
wb['fem_labor'] = load_wb('API_SL.TLF.CACT.FE.ZS', 'fem_labor')
wb['fem_educ'] = load_wb('API_SE.SEC.ENRR.FE', 'fem_educ')
wb['sex_ratio'] = load_wb('API_SP.POP.BRTH.MF', 'sex_ratio')

# Productive forces from v2 cache
pf_cache = f'{BASE}/_wb_indicators_cache.csv'
if os.path.exists(pf_cache):
    pf_raw = pd.read_csv(pf_cache)
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
        'Gambia, The': 'GMB', 'Yemen, Rep.': 'YEM',
        'Syrian Arab Republic': 'SYR', 'Eswatini': 'SWZ',
        'North Macedonia': 'MKD', 'Myanmar': 'MMR', "Cote d'Ivoire": 'CIV',
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
    for pk in ['industry', 'services', 'gdp_grow']:
        sub = pf_raw[pf_raw['indicator'] == pk].dropna(subset=['code'])
        wb[pk] = sub[['code', 'year', 'value']].rename(columns={'value': pk})

# ═══ Merge S(t) with all indicators ═══
merged = wpei[['code', 'year', 'S', 'entity']].copy()
for key, wdf in wb.items():
    merged = merged.merge(wdf, on=['code', 'year'], how='left')

ALL_KEYS = ['fem_labor', 'fem_educ', 'services', 'industry']
LABELS = {
    'fem_labor': 'Female labor participation',
    'fem_educ': 'Female secondary education',
    'services': 'Services % GDP',
    'industry': 'Industry % GDP',
}

# ═══ Per-country: r(indicator, S(t)) — 直接，不过残差 ═══
rows = []
for code, grp in merged.groupby('code'):
    row = {'code': code, 'entity': grp['entity'].iloc[0]}
    rs = []
    for key in ALL_KEYS:
        sub = grp[['S', key]].dropna()
        if len(sub) >= 8:
            r, p = stats.pearsonr(sub['S'], sub[key])
            row[f'r_{key}'] = r
            row[f'p_{key}'] = p
            row[f'n_{key}'] = len(sub)
            if key in ['fem_labor', 'fem_educ', 'services']:
                rs.append(r)
        else:
            row[f'r_{key}'] = np.nan
    row['r_avg'] = np.mean(rs) if rs else np.nan
    rows.append(row)

cc = pd.DataFrame(rows)
cc = cc.merge(edi_latest, on='code', how='left')

# Filter to countries with at least r_avg
cc = cc.dropna(subset=['r_avg']).sort_values('r_avg')

print('=' * 100)
print('  r(经济指标, S(t)) — 直接相关，不过Richards残差')
print('  sorted by mean r(fem_labor, fem_educ, services ~ S)')
print('=' * 100)
print(f"  {'#':>3} {'Code':<6} {'Entity':<25} {'r_avg':>6} "
      f"{'r_flab':>7} {'r_fedu':>7} {'r_serv':>7} {'r_ind':>7} {'EDI':>6}")
print('  ' + '─' * 90)

for i, (_, r) in enumerate(cc.iterrows(), 1):
    def fmt(v): return f'{v:+.3f}' if pd.notna(v) else '   N/A'
    edi = f'{r["edi_latest"]:.3f}' if pd.notna(r['edi_latest']) else ' N/A'
    print(f"  {i:>3} {r['code']:<6} {str(r['entity'])[:25]:<25} {r['r_avg']:>+6.3f} "
          f"{fmt(r.get('r_fem_labor')):>7} {fmt(r.get('r_fem_educ')):>7} "
          f"{fmt(r.get('r_services')):>7} {fmt(r.get('r_industry')):>7} {edi:>6}")

# ═══ KEY TEST: EDI vs r_avg ═══
valid = cc.dropna(subset=['edi_latest', 'r_avg'])
r_corr, p_corr = stats.pearsonr(valid['edi_latest'], valid['r_avg'])
sr_corr, sp_corr = stats.spearmanr(valid['edi_latest'], valid['r_avg'])

print(f'\n  ═══ EDI vs r(经济→S(t)) 直接相关 ═══')
print(f'  Pearson:  r={r_corr:+.4f}, p={p_corr:.2e}, n={len(valid)}')
print(f'  Spearman: ρ={sr_corr:+.4f}, p={sp_corr:.2e}')

# Per indicator
print(f'\n  ═══ 分指标 ═══')
for key in ALL_KEYS:
    sub = cc.dropna(subset=[f'r_{key}', 'edi_latest'])
    if len(sub) > 10:
        r_k, p_k = stats.pearsonr(sub['edi_latest'], sub[f'r_{key}'])
        print(f'  EDI vs r({key:>12}~S): r={r_k:+.4f}, p={p_k:.2e}, n={len(sub)}')

# Bottom vs top quartile
q1 = valid['r_avg'].quantile(0.25)
q3 = valid['r_avg'].quantile(0.75)
bot = valid[valid['r_avg'] <= q1]
top = valid[valid['r_avg'] >= q3]
print(f'\n  Bottom quartile (弱传导): mean EDI = {bot["edi_latest"].mean():.3f}, n={len(bot)}')
print(f'  Top quartile (强传导):    mean EDI = {top["edi_latest"].mean():.3f}, n={len(top)}')
u, p = stats.mannwhitneyu(bot['edi_latest'].dropna(), top['edi_latest'].dropna())
sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
print(f'  Mann-Whitney: U={u:.0f}, p={p:.4f} {sig}')

# ═══ Scatter ═══
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(valid['edi_latest'], valid['r_avg'], alpha=0.5, s=25, c='steelblue')
for _, r in valid.nsmallest(8, 'r_avg').iterrows():
    ax.annotate(r['code'], (r['edi_latest'], r['r_avg']), fontsize=7, alpha=0.8)
for _, r in valid.nlargest(5, 'r_avg').iterrows():
    ax.annotate(r['code'], (r['edi_latest'], r['r_avg']), fontsize=7, alpha=0.8)
z = np.polyfit(valid['edi_latest'], valid['r_avg'], 1)
xline = np.linspace(0, 0.95, 100)
ax.plot(xline, np.polyval(z, xline), 'r--', alpha=0.6)
ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
ax.set_xlabel('Electoral Democracy Index (latest)', fontsize=12)
ax.set_ylabel('Mean r(fem_labor, fem_educ, services ~ S(t))', fontsize=12)
ax.set_title(f'EDI vs economic-gender transmission (r={r_corr:+.3f})', fontsize=13)
plt.tight_layout()
plt.savefig(f'{BASE}/_vdem_edi_vs_direct.png', dpi=200, bbox_inches='tight')
print(f'\n  Saved: _vdem_edi_vs_direct.png')
