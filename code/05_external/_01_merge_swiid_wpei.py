# -*- coding: utf-8 -*-
"""阶级反相位分析 · 步骤1：SWIID 基尼 × WPEI(S=wpei-1) 对齐 merge + 覆盖诊断。
只读输入、输出合并面板 CSV + 诊断报告到终端。不跑统计（统计在步骤2）。
口径对齐：S=wpei-1（同 _idv_vs_edi.py 等）；社会主义分类用 _vdem_collectivist_decouple.py 正式版。"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import os
import numpy as np, pandas as pd

BASE = r'data'
OUT  = r'data'

# ---- 社会主义分类（版本二 · _vdem_collectivist_decouple.py 正式图表口径）----
CURRENT_SOC = {'CHN', 'PRK', 'CUB', 'VNM', 'LAO'}
FORMER_SOC = {'RUS', 'POL', 'CZE', 'SVK', 'HUN', 'ROU', 'BGR', 'ALB', 'SRB', 'HRV',
              'SVN', 'BIH', 'MKD', 'MNE', 'MNG', 'UKR', 'BLR', 'KAZ', 'UZB', 'TJK',
              'KGZ', 'TKM', 'GEO', 'ARM', 'AZE', 'EST', 'LVA', 'LTU', 'MOZ', 'AGO', 'ETH'}

# ---- 读 WPEI，S=wpei-1，剔除 OWID 区域聚合 ----
wpei = pd.read_csv(f'{BASE}/实验输出_CSV/women-political-empowerment-index.csv')
wpei.columns = ['entity', 'code', 'year', 'wpei', 'region']
wpei = wpei[wpei['code'].notna() & ~wpei['code'].astype(str).str.startswith('OWID')].copy()
wpei['S'] = wpei['wpei'] - 1
wpei['year'] = wpei['year'].astype(int)

# WPEI entity->code 映射（strip.lower），SWIID 大部分国名与 WPEI entity 同名，靠此自动匹配
e2c = {}
for _, r in wpei.drop_duplicates('code').iterrows():
    e2c[str(r['entity']).strip().lower()] = r['code']

# ---- 读 SWIID ----
sw = pd.read_csv(f'{BASE}/swiid9_92/swiid9_92_summary.csv')
sw['year'] = sw['year'].astype(int)

# SWIID country -> ISO3：先 MANUAL（处理 SWIID 与 WPEI 命名歧义），再 WPEI e2c 兜底
SWIID_MANUAL = {
    'Korea': 'KOR', 'Congo-Kinshasa': 'COD', 'Congo-Brazzaville': 'COG',
    'Czech Republic': 'CZE', 'Slovakia': 'SVK', 'Laos': 'LAO', 'Vietnam': 'VNM',
    'Russia': 'RUS', 'Timor-Leste': 'TLS', 'Yemen': 'YEM', 'Serbia': 'SRB',
    'North Macedonia': 'MKD', 'Eswatini': 'SWZ', 'Cape Verde': 'CPV', 'Brunei': 'BRN',
    'Myanmar': 'MMR', 'Hong Kong': 'HKG', 'Taiwan': 'TWN',
    'Palestinian Territories': 'PSE', "Côte d'Ivoire": 'CIV',
    'São Tomé and Príncipe': 'STP', 'St. Kitts and Nevis': 'KNA', 'St. Lucia': 'LCA',
    'St. Vincent and Grenadines': 'VCT', 'Bahamas': 'BHS', 'Gambia': 'GMB',
}
HIST = {'Czechoslovakia', 'Soviet Union', 'Yugoslavia'}  # 历史实体，无现代单一ISO，跳过并标注


def to_iso(country):
    c = str(country).strip()
    if c in HIST:
        return None
    if c in SWIID_MANUAL:
        return SWIID_MANUAL[c]
    return e2c.get(c.lower())


sw['iso3'] = sw['country'].map(to_iso)
unmapped = sorted(sw[sw['iso3'].isna() & ~sw['country'].isin(HIST)]['country'].unique())

# ---- Merge（inner：只保留 S 与 Gini 同国同年都有的行）----
S = wpei[['code', 'year', 'S']].rename(columns={'code': 'iso3'})
G = sw[sw['iso3'].notna()][['iso3', 'year', 'gini_mkt', 'gini_disp',
                            'gini_mkt_se', 'gini_disp_se']]
panel = S.merge(G, on=['iso3', 'year'], how='inner').sort_values(['iso3', 'year']).reset_index(drop=True)

# ---- 覆盖诊断 ----
print('=' * 64)
print('未映射的 SWIID 国家（需人工补映射）：')
print('  ', unmapped if unmapped else '（无，全部映射成功）')
print('  历史实体（有意跳过，无现代单一 ISO）：', sorted(HIST))
print('=' * 64)
n_country = panel['iso3'].nunique()
ov = panel.groupby('iso3').size()
print(f'合并面板：{n_country} 国，共 {len(panel)} 国×年观测')
print(f'每国 S∩Gini 重叠年数：中位 {ov.median():.0f}，最小 {ov.min()}，最大 {ov.max()}')
print(f'重叠≥15 年的国家数：{(ov >= 15).sum()}；≥20 年：{(ov >= 20).sum()}；≥30 年：{(ov >= 30).sum()}')
print('=' * 64)
print('社会主义国覆盖 · 现存 5 国：')
for iso in sorted(CURRENT_SOC):
    n = int(ov.get(iso, 0))
    yr = ''
    if n > 0:
        sub = panel[panel['iso3'] == iso]
        yr = f" ({sub['year'].min()}-{sub['year'].max()})"
    flag = '✓' if n > 0 else '✗ SWIID 无 Gini'
    print(f'  {iso}: {n} 年{yr}  {flag}')
covered_former = sorted(i for i in FORMER_SOC if ov.get(i, 0) > 0)
missing_former = sorted(i for i in FORMER_SOC if ov.get(i, 0) == 0)
print(f'前社会主义 {len(FORMER_SOC)} 国有 Gini 覆盖：{len(covered_former)}/{len(FORMER_SOC)}')
print(f'  缺 Gini：{missing_former if missing_former else "（无）"}')
print('=' * 64)

# ---- 输出合并面板 ----
os.makedirs(OUT, exist_ok=True)
outfp = f'{OUT}/_panel_S_gini.csv'
panel.to_csv(outfp, index=False, encoding='utf-8-sig')
print(f'已输出合并面板：{outfp}')
print(f'（列：iso3, year, S, gini_mkt, gini_disp, gini_mkt_se, gini_disp_se）')
