# -*- coding: utf-8 -*-
"""阶级×性别 · 横截面秩相关（照威权指数 EDI 法 _edi_fullsort.py 同一把尺子）。
把横轴"威权/EDI"换成"阶级矛盾代理·基尼"，纵轴不变：每国 Richards 拟合的
S_ceil(性别上限=ric_ceil) 与 k(爬升速率=ric_k)。一国一个点，不是时间序列。
横轴基尼每国取 gini_mean(跨年均值)+gini_latest(最新年)，market 主/disposable 稳健。
统计照抄：全样本按 S_ceil 升序拉通排(不分组)→Spearman 秩相关(+Pearson+R²)
        →S_ceil 最低20 vs 最高20 国的基尼中位数 Mann-Whitney。k 同样一套。
概念口径：阶级与性别平行/交替/互不决定(太阳月亮)，只描述规律不碰因果。
只读 _vdem_fit_all.csv + SWIID summary + WPEI(仅国名→ISO映射)；输出排序表+相关+图。"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = r'data'
OUT  = r'data'

# ── 纵轴：Richards 拟合 S_ceil + k（照 _edi_fullsort.py：status=ok）──
params = pd.read_csv(f'{BASE}/实验输出_CSV/_vdem_fit_all.csv')
params = params[params['status'] == 'ok'][['code', 'entity', 'ric_ceil', 'ric_k']].copy()

# ── 横轴：SWIID 基尼 → ISO3（口径同 _01_merge_swiid_wpei.py）──
wpei = pd.read_csv(f'{BASE}/实验输出_CSV/women-political-empowerment-index.csv')
wpei.columns = ['entity', 'code', 'year', 'wpei', 'region']
wpei = wpei[wpei['code'].notna() & ~wpei['code'].astype(str).str.startswith('OWID')]
e2c = {str(r['entity']).strip().lower(): r['code']
       for _, r in wpei.drop_duplicates('code').iterrows()}

sw = pd.read_csv(f'{BASE}/swiid9_92/swiid9_92_summary.csv')
sw['year'] = sw['year'].astype(int)
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
HIST = {'Czechoslovakia', 'Soviet Union', 'Yugoslavia'}


def to_iso(c):
    c = str(c).strip()
    if c in HIST:
        return None
    if c in SWIID_MANUAL:
        return SWIID_MANUAL[c]
    return e2c.get(c.lower())


sw['iso3'] = sw['country'].map(to_iso)
sw = sw[sw['iso3'].notna()].copy()


def per_country_gini(gcol):
    """每国基尼：跨年均值 + 最新年值。"""
    g = sw.dropna(subset=[gcol])
    gmean = g.groupby('iso3')[gcol].mean().rename('gini_mean')
    glat = g.sort_values('year').groupby('iso3').last()[gcol].rename('gini_latest')
    return pd.concat([gmean, glat], axis=1)


def corr_block(df, xcol, ycol, ylabel):
    v = df.dropna(subset=[xcol, ycol])
    r, p = stats.pearsonr(v[xcol], v[ycol])
    rho, sp = stats.spearmanr(v[xcol], v[ycol])
    print(f'  {ylabel:<14} vs {xcol:<11}:  Spearman ρ={rho:+.4f} (p={sp:.2e})  |  '
          f'Pearson r={r:+.4f} (p={p:.2e})  |  R²={r**2:.3f}  n={len(v)}')
    return rho, sp, r, p, len(v)


for gname, gcol in [('MARKET 基尼（主）', 'gini_mkt'), ('DISPOSABLE 基尼（稳健）', 'gini_disp')]:
    g = per_country_gini(gcol)
    df = params.merge(g, left_on='code', right_index=True, how='left').sort_values('ric_ceil')
    print('=' * 92)
    print(f'  {gname}：一国一点，按 S_ceil 升序，横轴=该国基尼（均值/最新）')
    print('=' * 92)

    if gcol == 'gini_mkt':
        print(f"  {'#':>3} {'Code':<6} {'Entity':<26} {'S_ceil':>7} {'k':>6} {'Gini_mean':>9} {'Gini_now':>8}")
        print('  ' + '─' * 82)
        for i, (_, r) in enumerate(df.iterrows(), 1):
            gm = f"{r['gini_mean']:.1f}" if pd.notna(r['gini_mean']) else '  N/A'
            gl = f"{r['gini_latest']:.1f}" if pd.notna(r['gini_latest']) else '  N/A'
            print(f"  {i:>3} {str(r['code']):<6} {str(r['entity'])[:26]:<26} "
                  f"{r['ric_ceil']:>+7.3f} {r['ric_k']:>6.3f} {gm:>9} {gl:>8}")
        df.to_csv(f'{OUT}/_class_gini_rank_market.csv', index=False, encoding='utf-8-sig')

    print(f'\n  ═══ 秩相关（阶级矛盾↑ = 基尼↑）═══')
    corr_block(df, 'gini_mean', 'ric_ceil', 'S_ceil(上限)')
    corr_block(df, 'gini_latest', 'ric_ceil', 'S_ceil(上限)')
    corr_block(df, 'gini_mean', 'ric_k', 'k(爬升速率)')
    corr_block(df, 'gini_latest', 'ric_k', 'k(爬升速率)')

    # ── S_ceil 最低20 vs 最高20 的基尼比较（df 已按 S_ceil 升序）──
    valid = df.dropna(subset=['gini_mean'])
    bot, top = valid.head(20), valid.tail(20)
    u, pu = stats.mannwhitneyu(bot['gini_mean'], top['gini_mean'])
    mark = "***" if pu < 0.001 else "**" if pu < 0.01 else "*" if pu < 0.05 else "n.s."
    print(f'\n  ═══ S_ceil 最低20 vs 最高20 国·基尼(均值) ═══')
    print(f'  最低20(性别上限最低)：基尼中位 = {bot["gini_mean"].median():.1f}  (n={len(bot)})')
    print(f'  最高20(性别上限最高)：基尼中位 = {top["gini_mean"].median():.1f}  (n={len(top)})')
    print(f'  Mann-Whitney U={u:.0f}, p={pu:.4f} {mark}')
    print()

# ── 图：market 基尼 × (S_ceil, k) 四联 ──
g = per_country_gini('gini_mkt')
df = params.merge(g, left_on='code', right_index=True, how='left')
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
panels = [('gini_mean', 'ric_ceil', '(a) Gini_mean vs S_ceil (ceiling)'),
          ('gini_latest', 'ric_ceil', '(b) Gini_latest vs S_ceil (ceiling)'),
          ('gini_mean', 'ric_k', '(c) Gini_mean vs k (rise rate)'),
          ('gini_latest', 'ric_k', '(d) Gini_latest vs k (rise rate)')]
for ax, (xc, yc, title) in zip(axes.ravel(), panels):
    v = df.dropna(subset=[xc, yc])
    rho, sp = stats.spearmanr(v[xc], v[yc])
    ax.scatter(v[xc], v[yc], alpha=0.5, s=22, c='crimson')
    z = np.polyfit(v[xc], v[yc], 1)
    xl = np.linspace(v[xc].min(), v[xc].max(), 100)
    ax.plot(xl, np.polyval(z, xl), 'k--', alpha=0.6, lw=1.3)
    ax.set_xlabel('market Gini (' + ('mean' if 'mean' in xc else 'latest') + ')')
    ax.set_ylabel('S_ceil (ceiling)' if yc == 'ric_ceil' else 'k (Richards rise rate)')
    ax.set_title(f'{title}   ρ={rho:+.3f}, p={sp:.1e}, n={len(v)}', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT}/_class_gini_rank.png', dpi=200, bbox_inches='tight')
print('已存：_class_gini_rank_market.csv、_class_gini_rank.png')
