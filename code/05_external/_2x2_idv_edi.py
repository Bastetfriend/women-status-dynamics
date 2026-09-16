# -*- coding: utf-8 -*-
"""2×2：集体主义(IDV) × 威权(EDI) 交互效应 on b 和 S"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from scipy import stats

BASE = r'data'

# ═══ 读 Hofstede ═══
hof = pd.read_csv(f'{BASE}/6-dimensions-for-website-2015-08-16.csv', sep=';')
hof['idv'] = pd.to_numeric(hof['idv'].replace('#NULL!', np.nan), errors='coerce')

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

# ═══ EDI ═══
edi_raw = pd.read_csv(f'{BASE}/electoral-democracy-index.csv')
edi_raw.columns = ['entity', 'code', 'year', 'edi', 'region']
edi_latest = edi_raw.sort_values('year').groupby('code').last()[['edi']].reset_index()

# ═══ b ═══
bdf = pd.read_excel(f'{BASE}/_中介分析_按教育效应排序.xlsx')
col_b = [c for c in bdf.columns if '教育' in c and '地位' in c][0]

# ═══ S(t) ═══
wpei = pd.read_csv(f'{BASE}/women-political-empowerment-index.csv')
wpei.columns = ['entity', 'code', 'year', 'wpei', 'region']
wpei['S'] = wpei['wpei'] - 1
s_latest = wpei.sort_values('year').groupby('code').last()[['S']].reset_index()

# ═══ 合并 ═══
m = idv_df.merge(edi_latest, left_on='iso3', right_on='code', how='left')
m = m.merge(bdf[['国家代码', col_b]], left_on='iso3', right_on='国家代码', how='left')
m = m.merge(s_latest, left_on='iso3', right_on='code', how='left', suffixes=('', '_s'))
m = m.dropna(subset=['idv', 'edi', col_b])

# ═══ 2×2 分组 ═══
# 用中位数切分
idv_cut = m['idv'].median()
edi_cut = m['edi'].median()
print(f'切分点：IDV 中位数 = {idv_cut:.0f}，EDI 中位数 = {edi_cut:.3f}')
print(f'总国家数：{len(m)}\n')

def label(row):
    coll = row['idv'] <= idv_cut
    auth = row['edi'] <= edi_cut
    if coll and auth:
        return '集体+威权'
    elif coll and not auth:
        return '集体+民主'
    elif not coll and auth:
        return '个人+威权'
    else:
        return '个人+民主'

m['组'] = m.apply(label, axis=1)

# 按理论预测排序：个人+民主 > 单因素 > 集体+威权
order = ['个人+民主', '个人+威权', '集体+民主', '集体+威权']

print(f'{"="*75}')
print(f'  2×2 交互：集体主义 × 威权 对 b（教育→地位）和 S（地位水平）的影响')
print(f'{"="*75}\n')
print(f'  {"组别":<12} {"n":>3}  {"b均值":>10}  {"b中位":>10}  {"S均值":>8}  {"S中位":>8}  {"b<0比例":>8}')
print(f'  {"-"*65}')

groups = {}
for g in order:
    sub = m[m['组'] == g]
    if len(sub) == 0:
        continue
    groups[g] = sub
    b_mean = sub[col_b].mean()
    b_med = sub[col_b].median()
    s_mean = sub['S'].mean()
    s_med = sub['S'].median()
    neg_pct = (sub[col_b] < 0).mean()
    print(f'  {g:<12} {len(sub):>3}  {b_mean:>+10.5f}  {b_med:>+10.5f}  {s_mean:>+8.4f}  {s_med:>+8.4f}  {neg_pct:>8.0%}')

# ═══ 统计检验 ═══
print(f'\n{"="*75}')
print(f'  统计检验')
print(f'{"="*75}')

# 有一个因素 vs 没有
has_none = groups.get('个人+民主', pd.DataFrame())
has_one = pd.concat([groups.get('个人+威权', pd.DataFrame()),
                     groups.get('集体+民主', pd.DataFrame())])
has_both = groups.get('集体+威权', pd.DataFrame())

if len(has_none) > 2 and len(has_one) > 2:
    u, p = stats.mannwhitneyu(has_none[col_b], has_one[col_b], alternative='greater')
    print(f'\n  个人+民主 vs 单因素（b值）：Mann-Whitney p = {p:.4f}')
    u, p = stats.mannwhitneyu(has_none['S'], has_one['S'], alternative='greater')
    print(f'  个人+民主 vs 单因素（S值）：Mann-Whitney p = {p:.4f}')

if len(has_one) > 2 and len(has_both) > 2:
    u, p = stats.mannwhitneyu(has_one[col_b], has_both[col_b], alternative='greater')
    print(f'\n  单因素 vs 集体+威权（b值）：Mann-Whitney p = {p:.4f}')
    u, p = stats.mannwhitneyu(has_one['S'], has_both['S'], alternative='greater')
    print(f'  单因素 vs 集体+威权（S值）：Mann-Whitney p = {p:.4f}')

if len(has_none) > 2 and len(has_both) > 2:
    u, p = stats.mannwhitneyu(has_none[col_b], has_both[col_b], alternative='greater')
    print(f'\n  个人+民主 vs 集体+威权（b值）：Mann-Whitney p = {p:.4f}')
    u, p = stats.mannwhitneyu(has_none['S'], has_both['S'], alternative='greater')
    print(f'  个人+民主 vs 集体+威权（S值）：Mann-Whitney p = {p:.4f}')

# ═══ 各象限国家列表 ═══
for g in order:
    sub = m[m['组'] == g].sort_values(col_b)
    print(f'\n{"─"*75}')
    print(f'  {g}（{len(sub)} 国）')
    print(f'{"─"*75}')
    for i, (_, r) in enumerate(sub.iterrows(), 1):
        b = r[col_b]
        s = r['S']
        idv = int(r['idv'])
        edi = r['edi']
        name = str(r['country'])[:20]
        print(f'  {i:>2} {r["iso3"]:<5} {name:<20} IDV={idv:>3} EDI={edi:.3f} b={b:+.5f} S={s:+.4f}')
