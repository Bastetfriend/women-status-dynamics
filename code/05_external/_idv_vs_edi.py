# -*- coding: utf-8 -*-
"""Hofstede IDV vs EDI，IDV vs b（教育→地位）"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from scipy import stats

BASE = r'data'

# ═══ 读 Hofstede ═══
hof = pd.read_csv(f'{BASE}/6-dimensions-for-website-2015-08-16.csv', sep=';')
hof['idv'] = pd.to_numeric(hof['idv'].replace('#NULL!', np.nan), errors='coerce')

# Hofstede code → ISO3 映射
HOF2ISO = {
    'ARG': 'ARG', 'AUL': 'AUS', 'AUT': 'AUT', 'BAN': 'BGD',
    'BEL': 'BEL', 'BRA': 'BRA', 'BUL': 'BGR', 'CAN': 'CAN',
    'CHL': 'CHL', 'CHI': 'CHN', 'COL': 'COL', 'COS': 'CRI',
    'CRO': 'HRV', 'CZE': 'CZE', 'DEN': 'DNK', 'ECA': 'ECU',
    'SAL': 'SLV', 'EST': 'EST', 'FIN': 'FIN', 'FRA': 'FRA',
    'GER': 'DEU', 'GBR': 'GBR', 'GRE': 'GRC', 'GUA': 'GTM',
    'HOK': 'HKG', 'HUN': 'HUN', 'IND': 'IND', 'IDO': 'IDN',
    'IRA': 'IRN', 'IRE': 'IRL', 'ISR': 'ISR', 'ITA': 'ITA',
    'JAM': 'JAM', 'JPN': 'JPN', 'KOR': 'KOR', 'LAT': 'LVA',
    'LIT': 'LTU', 'LUX': 'LUX', 'MAL': 'MYS', 'MLT': 'MLT',
    'MEX': 'MEX', 'MOR': 'MAR', 'NET': 'NLD', 'NZL': 'NZL',
    'PAK': 'PAK', 'PAN': 'PAN', 'PER': 'PER', 'PHI': 'PHL',
    'POL': 'POL', 'POR': 'PRT', 'ROM': 'ROU', 'RUS': 'RUS',
    'SER': 'SRB', 'SIN': 'SGP', 'SLK': 'SVK', 'SLV': 'SVN',
    'SPA': 'ESP', 'SUR': 'SUR', 'SWE': 'SWE', 'SWI': 'CHE',
    'THA': 'THA', 'TRI': 'TTO', 'TUR': 'TUR', 'USA': 'USA',
    'URU': 'URY', 'VEN': 'VEN', 'VIE': 'VNM', 'TAI': 'TWN',
}

hof['iso3'] = hof['ctr'].map(HOF2ISO)
idv_df = hof.dropna(subset=['idv', 'iso3'])[['iso3', 'country', 'idv']].copy()
idv_df['idv'] = idv_df['idv'].astype(int)
print(f'Hofstede IDV 有效国家：{len(idv_df)}')

# ═══ 读 EDI ═══
edi_raw = pd.read_csv(f'{BASE}/electoral-democracy-index.csv')
edi_raw.columns = ['entity', 'code', 'year', 'edi', 'region']
edi_latest = edi_raw.sort_values('year').groupby('code').last()[['edi']].reset_index()

# ═══ 读 b（教育→地位）═══
bdf = pd.read_excel(f'{BASE}/_中介分析_按教育效应排序.xlsx')
col_b = [c for c in bdf.columns if '教育' in c and '地位' in c][0]

# ═══ 读 S(t) 最新值 ═══
wpei = pd.read_csv(f'{BASE}/women-political-empowerment-index.csv')
wpei.columns = ['entity', 'code', 'year', 'wpei', 'region']
wpei['S'] = wpei['wpei'] - 1
s_latest = wpei.sort_values('year').groupby('code').last()[['S']].reset_index()

# ═══ 合并 ═══
m = idv_df.merge(edi_latest, left_on='iso3', right_on='code', how='left')
m = m.merge(bdf[['国家代码', col_b]], left_on='iso3', right_on='国家代码', how='left')
m = m.merge(s_latest, left_on='iso3', right_on='code', how='left', suffixes=('', '_s'))
m = m.dropna(subset=['edi'])

print(f'合并后有 EDI 的：{len(m)}')
print(f'合并后有 b 的：  {m[col_b].notna().sum()}')

# ═══ 1. IDV vs EDI ═══
v1 = m.dropna(subset=['idv', 'edi'])
r1, p1 = stats.pearsonr(v1['idv'], v1['edi'])
rho1, sp1 = stats.spearmanr(v1['idv'], v1['edi'])

print(f'\n{"="*70}')
print(f'  1. 个人主义 IDV vs 民主指数 EDI（{len(v1)} 国）')
print(f'{"="*70}')
print(f'  Pearson:  r = {r1:+.4f}, p = {p1:.2e}')
print(f'  Spearman: ρ = {rho1:+.4f}, p = {sp1:.2e}')

# 分层：IDV低 vs IDV高
idv_med = v1['idv'].median()
low_idv = v1[v1['idv'] <= idv_med]
high_idv = v1[v1['idv'] > idv_med]
print(f'\n  IDV 中位数 = {idv_med}')
print(f'  集体主义（IDV ≤ {idv_med:.0f}）：EDI 均值 = {low_idv["edi"].mean():.3f}，中位数 = {low_idv["edi"].median():.3f}，n={len(low_idv)}')
print(f'  个人主义（IDV > {idv_med:.0f}）：EDI 均值 = {high_idv["edi"].mean():.3f}，中位数 = {high_idv["edi"].median():.3f}，n={len(high_idv)}')
u, pu = stats.mannwhitneyu(low_idv['edi'], high_idv['edi'])
print(f'  Mann-Whitney: p = {pu:.2e}')

# ═══ 2. IDV vs b（教育→地位）═══
v2 = m.dropna(subset=['idv', col_b])
r2, p2 = stats.pearsonr(v2['idv'], v2[col_b])
rho2, sp2 = stats.spearmanr(v2['idv'], v2[col_b])

print(f'\n{"="*70}')
print(f'  2. 个人主义 IDV vs b（教育→地位）（{len(v2)} 国）')
print(f'{"="*70}')
print(f'  Pearson:  r = {r2:+.4f}, p = {p2:.4f}')
print(f'  Spearman: ρ = {rho2:+.4f}, p = {sp2:.4f}')

# b<0 vs b>=0 的 IDV 比较
neg_b = v2[v2[col_b] < 0]
pos_b = v2[v2[col_b] >= 0]
print(f'\n  b < 0（教育推不动）：IDV 均值 = {neg_b["idv"].mean():.1f}，中位数 = {neg_b["idv"].median():.0f}，n={len(neg_b)}')
print(f'  b ≥ 0（教育能推动）：IDV 均值 = {pos_b["idv"].mean():.1f}，中位数 = {pos_b["idv"].median():.0f}，n={len(pos_b)}')
if len(neg_b) > 3 and len(pos_b) > 3:
    u2, pu2 = stats.mannwhitneyu(neg_b['idv'], pos_b['idv'])
    print(f'  Mann-Whitney: p = {pu2:.4f}')

# ═══ 3. 全部国家按IDV排列，标注b和EDI ═══
v3 = m.dropna(subset=['idv']).sort_values('idv')
print(f'\n{"="*70}')
print(f'  全部国家按 IDV 从低到高（最集体主义 → 最个人主义）')
print(f'{"="*70}\n')

for i, (_, r) in enumerate(v3.iterrows(), 1):
    idv = int(r['idv'])
    edi = f'{r["edi"]:.3f}' if pd.notna(r['edi']) else 'N/A'
    b = f'{r[col_b]:+.4f}' if pd.notna(r[col_b]) else '  N/A'
    s = f'{r["S"]:.3f}' if pd.notna(r.get('S')) else 'N/A'
    name = str(r['country'])[:20]
    code = r['iso3']
    bval = r[col_b] if pd.notna(r[col_b]) else 0
    marker = ' ◄' if (pd.notna(r[col_b]) and bval < 0 and pd.notna(r.get('S')) and r['S'] < -0.217) else ''
    print(f'{i:>3} {code:<5} IDV={idv:>3}  EDI={edi}  b={b}  S={s}  {name}{marker}')
