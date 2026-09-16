# -*- coding: utf-8 -*-
"""Full recompute of IDVxEDI section: all numbers for 69-country sample.
   Output every number that appears in Methods/Section 2.2/Fig 4 legend."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
from scipy import stats
import statsmodels.api as sm
import os

BASE = r'data'
CSV = BASE + r'\实验输出_CSV'

# ═══ Load data ═══
rich = pd.read_csv(f'{CSV}/_richards_diagnostics.csv')
rich = rich[['code', 'entity', 'S_ceil']].dropna(subset=['S_ceil'])

hof = pd.read_csv(f'{CSV}/6-dimensions-for-website-2015-08-16.csv', sep=';')
hof['idv'] = pd.to_numeric(hof['idv'].replace('#NULL!', np.nan), errors='coerce')
HOF2ISO = {
    'ARG':'ARG','AUL':'AUS','AUT':'AUT','BAN':'BGD','BEL':'BEL','BRA':'BRA','BUL':'BGR',
    'CAN':'CAN','CHL':'CHL','CHI':'CHN','COL':'COL','COS':'CRI','CRO':'HRV','CZE':'CZE',
    'DEN':'DNK','ECA':'ECU','SAL':'SLV','EST':'EST','FIN':'FIN','FRA':'FRA','GER':'DEU',
    'GBR':'GBR','GRE':'GRC','GUA':'GTM','HOK':'HKG','HUN':'HUN','IND':'IND','IDO':'IDN',
    'IRA':'IRN','IRE':'IRL','ISR':'ISR','ITA':'ITA','JAM':'JAM','JPN':'JPN','KOR':'KOR',
    'LAT':'LVA','LIT':'LTU','LUX':'LUX','MAL':'MYS','MLT':'MLT','MEX':'MEX','MOR':'MAR',
    'NET':'NLD','NZL':'NZL','NOR':'NOR','PAK':'PAK','PAN':'PAN','PER':'PER','PHI':'PHL',
    'POL':'POL','POR':'PRT','ROM':'ROU','RUS':'RUS','SER':'SRB','SIN':'SGP','SLK':'SVK',
    'SLV':'SVN','SAF':'ZAF','SPA':'ESP','SUR':'SUR','SWE':'SWE','SWI':'CHE','TAI':'TWN',
    'THA':'THA','TRI':'TTO','TUR':'TUR','USA':'USA','URU':'URY','VEN':'VEN','VIE':'VNM',
}
hof['iso3'] = hof['ctr'].map(HOF2ISO)
idv_df = hof.dropna(subset=['idv','iso3'])[['iso3','country','idv']].copy()
idv_df['idv'] = idv_df['idv'].astype(int)

edi_raw = pd.read_csv(f'{CSV}/electoral-democracy-index.csv')
edi_raw.columns = ['entity','code','year','edi','region']
edi_latest = edi_raw.sort_values('year').groupby('code').last()[['edi']].reset_index()

# ═══ Merge: IDV + EDI + Richards S_ceil ═══
m = idv_df.merge(edi_latest, left_on='iso3', right_on='code', how='inner')
m = m.merge(rich[['code','S_ceil']], left_on='iso3', right_on='code', how='inner')
m = m.dropna(subset=['idv','edi','S_ceil'])

n_tot = len(m)
idv_med = m['idv'].median()
edi_med = m['edi'].median()

# Also load b (education→status slope) for the full analysis
# b is stored in the 中介分析 file
import glob as _glob
_bpath = _glob.glob(BASE + r'\**\*中介分析*教育效应排序*.xlsx', recursive=True)[0]
bdf = pd.read_excel(_bpath)
col_b = [c for c in bdf.columns if '教育' in c and '地位' in c][0]
# Merge b into m
m_b = m.merge(bdf[['国家代码', col_b]], left_on='iso3', right_on='国家代码', how='left')
m_b = m_b.dropna(subset=[col_b])
n_with_b = len(m_b)

print(f'IDV + EDI + S_ceil: {n_tot} countries')
print(f'IDV + EDI + S_ceil + b: {n_with_b} countries')
print(f'IDV median = {idv_med}')
print(f'EDI median = {edi_med:.4f}')

# ═══ Quadrants ═══
i_d = m[(m['idv'] > idv_med) & (m['edi'] > edi_med)]
c_a = m[(m['idv'] <= idv_med) & (m['edi'] <= edi_med)]
off = m[~m.index.isin(i_d.index) & ~m.index.isin(c_a.index)]
print(f'\nQuadrants (S_ceil sample, n={n_tot}):')
print(f'  I+D: n={len(i_d)}')
print(f'  C+A: n={len(c_a)}')
print(f'  Off-diag: n={len(off)}')

# ═══ Mann-Whitney tests (one-tailed) ═══
# off-diag vs C+A
b_off = m_b.loc[m_b.index.isin(off.index), col_b].dropna()
b_ca = m_b.loc[m_b.index.isin(c_a.index), col_b].dropna()
s_off = off['S_ceil']
s_ca = c_a['S_ceil']

# off-diag vs C+A: expecting off-diag > C+A
_, p_b_off_ca = stats.mannwhitneyu(b_off, b_ca, alternative='greater')
_, p_s_off_ca = stats.mannwhitneyu(s_off, s_ca, alternative='greater')

# I+D vs C+A
b_id = m_b.loc[m_b.index.isin(i_d.index), col_b].dropna()
s_id = i_d['S_ceil']
_, p_b_id_ca = stats.mannwhitneyu(b_id, b_ca, alternative='greater')
_, p_s_id_ca = stats.mannwhitneyu(s_id, s_ca, alternative='greater')

print(f'\nMann-Whitney (one-tailed):')
print(f'  off-diag vs C+A:  b p={p_b_off_ca:.4f}, S p={p_s_off_ca:.4f}')
print(f'  I+D vs C+A:       b p={p_b_id_ca:.4f}, S p={p_s_id_ca:.4f}')

# ═══ Continuous interaction models ═══
# Center predictors
m['idv_c'] = m['idv'] - m['idv'].mean()
m['edi_c'] = m['edi'] - m['edi'].mean()
m['idv_x_edi'] = m['idv_c'] * m['edi_c']

# Model for S_ceil
X_s = sm.add_constant(m[['idv_c', 'edi_c', 'idv_x_edi']])
ols_s_full = sm.OLS(m['S_ceil'], X_s).fit()

# EDI alone
ols_s_edi = sm.OLS(m['S_ceil'], sm.add_constant(m['edi'])).fit()

# IDV alone
ols_s_idv = sm.OLS(m['S_ceil'], sm.add_constant(m['idv'])).fit()

print(f'\nContinuous model: S_ceil ~ IDV + EDI + IDVxEDI (n={n_tot})')
print(f'  EDI alone R² = {ols_s_edi.rsquared:.4f} ({ols_s_edi.rsquared*100:.1f}%)')
print(f'  IDV alone R² = {ols_s_idv.rsquared:.4f}')
print(f'  Full model R² = {ols_s_full.rsquared:.4f}')
print(f'  ΔR² (add IDV+interaction) = {ols_s_full.rsquared - ols_s_edi.rsquared:.4f}')
# F-test for the increment
rss_edi = ols_s_edi.ssr
rss_full = ols_s_full.ssr
df_edi = ols_s_edi.df_resid
df_full = ols_s_full.df_resid
F_inc = ((rss_edi - rss_full) / (df_edi - df_full)) / (rss_full / df_full)
p_inc = 1 - stats.f.cdf(F_inc, df_edi - df_full, df_full)
print(f'  F({df_edi-df_full:.0f},{df_full:.0f}) = {F_inc:.3f}, p = {p_inc:.4f}')

print(f'\nFull model coefficients:')
for name in ['idv_c', 'edi_c', 'idv_x_edi']:
    idx = list(X_s.columns).index(name)
    beta = ols_s_full.params.iloc[idx]
    pval = ols_s_full.pvalues.iloc[idx]
    print(f'  β_{name} = {beta:.6f}, p = {pval:.4f}')

# Model for b
if n_with_b > 0:
    m_b2 = m_b.copy()
    m_b2['idv_c'] = m_b2['idv'] - m_b2['idv'].mean()
    m_b2['edi_c'] = m_b2['edi'] - m_b2['edi'].mean()
    m_b2['idv_x_edi'] = m_b2['idv_c'] * m_b2['edi_c']
    X_b = sm.add_constant(m_b2[['idv_c', 'edi_c', 'idv_x_edi']])
    ols_b_full = sm.OLS(m_b2[col_b], X_b).fit()

    # IDV alone, EDI alone
    ols_b_idv = sm.OLS(m_b2[col_b], sm.add_constant(m_b2['idv'])).fit()
    ols_b_edi = sm.OLS(m_b2[col_b], sm.add_constant(m_b2['edi'])).fit()

    print(f'\nContinuous model: b ~ IDV + EDI + IDVxEDI (n={n_with_b})')
    print(f'  IDV alone R² = {ols_b_idv.rsquared:.4f}')
    print(f'  EDI alone R² = {ols_b_edi.rsquared:.4f}')
    print(f'  Full model R² = {ols_b_full.rsquared:.4f}')
    for name in ['idv_c', 'edi_c', 'idv_x_edi']:
        idx = list(X_b.columns).index(name)
        beta = ols_b_full.params.iloc[idx]
        pval = ols_b_full.pvalues.iloc[idx]
        print(f'  β_{name} = {beta:.6f}, p = {pval:.4f}')

# ═══ All-191 Spearman ═══
mA = edi_latest.merge(rich, on='code', how='inner').dropna(subset=['edi','S_ceil'])
rho_all, p_all = stats.spearmanr(mA['edi'], mA['S_ceil'])
print(f'\nAll-{len(mA)} EDI x S_ceil:')
print(f'  Spearman ρ = {rho_all:.4f} (ρ² = {rho_all**2:.4f} = {rho_all**2*100:.1f}%)')
print(f'  p = {p_all:.2e}')

# ═══ Export for verification ═══
out = m[['iso3','country','idv','edi','S_ceil']].copy()
out.to_csv(BASE + r'\_idv_edi_sceil_export.csv', index=False)
print(f'\nExported {len(out)} rows to _idv_edi_sceil_export.csv')
