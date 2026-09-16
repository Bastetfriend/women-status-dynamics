# -*- coding: utf-8 -*-
"""阶级×性别 · 阶级固化(代际教育流动 BETA) vs 性别上限/爬升 —— 横截面秩相关＋控发展＋控民主。
谢 2026-07-08 定(compact 后):在 labsh(初次分配公平度)之外，加第三条稳健轴【阶级固化】。
代理 = 世行 GDIM 2023 教育版代际流动的 BETA(子代教育对亲代教育回归系数=相对持续性=世袭强度)。
⚠️符号提醒:BETA 越高=越固化=阶级再生产越强=阶级矛盾越大→预期与 S_ceil 负相关
   (正好与 labsh 相反:labsh 高=分配公平=S_ceil 高=正;两者一致互证"分配越公→性别上限越高")。
口径锁定(实摸字段):parent=max、child=all、cohort=1980 → 每国一个 BETA(153国全覆盖);
   parent=avg(142国)当稳健。纵轴不变:每国 Richards 的 S_ceil(ric_ceil,上限) 与 k(ric_k,速率)。
统计照 _08:独立性诊断(BETA~GDPpc/~EDI)＋Spearman＋偏相关控 GDPpc/控 EDI/双控＋最低20vs最高20 MW;
   照 _09:社会主义读数＋民主/威权分组。只读 _vdem_fit_all.csv+GDIM_2023_03.csv+pwt110.xlsx+EDI;出表+图。"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = r'data'
OUT  = r'data'
PWT  = r'data\pwt110.xlsx'
GDIM = r'data\GDIM_2023_03.csv'

CURRENT_SOC = {'CHN', 'PRK', 'CUB', 'VNM', 'LAO'}
FORMER_SOC = {'RUS', 'POL', 'CZE', 'SVK', 'HUN', 'ROU', 'BGR', 'ALB', 'SRB', 'HRV',
              'SVN', 'BIH', 'MKD', 'MNE', 'MNG', 'UKR', 'BLR', 'KAZ', 'UZB', 'TJK',
              'KGZ', 'TKM', 'GEO', 'ARM', 'AZE', 'EST', 'LVA', 'LTU', 'MOZ', 'AGO', 'ETH'}

# ── 纵轴：Richards S_ceil + k（status=ok）──
params = pd.read_csv(f'{BASE}/实验输出_CSV/_vdem_fit_all.csv')
params = params[params['status'] == 'ok'][['code', 'entity', 'ric_ceil', 'ric_k']].copy()

# ── 横轴：GDIM BETA（阶级固化）主口径 parent=max/child=all/cohort=1980；avg 稳健 ──
g = pd.read_csv(GDIM)


def beta_by(parent):
    sub = g[(g['parent'] == parent) & (g['child'] == 'all') & (g['cohort'] == 1980)]
    sub = sub.dropna(subset=['BETA'])
    return sub.groupby('code')['BETA'].mean()   # 每国一行，mean 只为保险去重


beta_max = beta_by('max').rename('beta_max')
beta_avg = beta_by('avg').rename('beta_avg')

# ── 控制轴1：PWT GDP per capita（每国 mean → log）──
pwt = pd.read_excel(PWT, sheet_name='Data')
pwt['gdppc'] = pwt['cgdpo'] / pwt['pop']
gpc_m = pwt.dropna(subset=['gdppc']).groupby('countrycode')['gdppc'].mean().rename('gdppc_mean')

# ── 控制轴2：EDI（电子民主，每国 mean + latest）──
edi = pd.read_csv(f'{BASE}/实验输出_CSV/electoral-democracy-index.csv')
edi.columns = ['entity', 'code', 'year', 'edi', 'region']
edi = edi[edi['code'].notna()]
edi_m = edi.groupby('code')['edi'].mean().rename('edi_mean')
edi_l = edi.sort_values('year').groupby('code').last()['edi'].rename('edi_latest')

dev = pd.concat([beta_max, beta_avg, gpc_m], axis=1)
dev['lgdppc_mean'] = np.log(dev['gdppc_mean'])
df = (params.merge(dev, left_on='code', right_index=True, how='left')
            .merge(edi_m, left_on='code', right_index=True, how='left')
            .merge(edi_l, left_on='code', right_index=True, how='left'))
df['beta_pctile'] = df['beta_max'].rank(pct=True) * 100


def sp(d, x, y):
    v = d.dropna(subset=[x, y])
    if len(v) < 6:
        return np.nan, np.nan, len(v)
    rho, p = stats.spearmanr(v[x], v[y])
    return rho, p, len(v)


def partial_spearman(d, x, y, controls):
    """偏 Spearman：x,y,控制变量取秩→x、y 各自对控制变量回归取残差→残差相关。"""
    sub = d.dropna(subset=[x, y] + controls)
    if len(sub) < len(controls) + 5:
        return np.nan, np.nan, len(sub)
    R = sub[[x, y] + controls].rank()
    C = np.column_stack([np.ones(len(R))] + [R[c].values for c in controls])

    def resid(v):
        b, *_ = np.linalg.lstsq(C, v, rcond=None)
        return v - C @ b

    r, p = stats.pearsonr(resid(R[x].values), resid(R[y].values))
    return r, p, len(sub)


print('=' * 84)
print('  阶级固化 BETA = 代际教育持续性（越高=越世袭=阶级矛盾越大）')
print('  ⚠️预期方向：BETA↑(固化重) ↔ S_ceil↓(性别上限低)，即负相关 = 阶级固化压性别上限')
print('  样本：parent=max, child=all, cohort=1980')
print('=' * 84)

print('\n独立性诊断：BETA 自身跟 发展/民主 缠多紧（够不够格当独立阶级轴）')
for a, b in [('beta_max', 'lgdppc_mean'), ('beta_max', 'edi_mean'), ('edi_mean', 'lgdppc_mean')]:
    rho, p, n = sp(df, a, b)
    print(f'  {a:<12} ~ {b:<12}: Spearman ρ={rho:+.3f} (p={p:.2e}, n={n})')

for ylab, ycol in [('S_ceil(上限)', 'ric_ceil'), ('k(爬升速率)', 'ric_k')]:
    print('\n' + '=' * 84)
    print(f'  阶级固化 BETA  →  {ylab}   [原始 / 控GDPpc / 控EDI / 双控]')
    print('=' * 84)
    for xcol in ['beta_max', 'beta_avg']:
        rho, p, n = sp(df, xcol, ycol)
        pr_g, pp_g, _ = partial_spearman(df, xcol, ycol, ['lgdppc_mean'])
        pr_e, pp_e, _ = partial_spearman(df, xcol, ycol, ['edi_mean'])
        pr_b, pp_b, nb = partial_spearman(df, xcol, ycol, ['lgdppc_mean', 'edi_mean'])
        tag = '主口径 parent=max' if xcol == 'beta_max' else '稳健 parent=avg'
        print(f'  [{xcol}·{tag}]  n={n}')
        print(f'    原始         Spearman ρ={rho:+.4f} (p={p:.2e})')
        print(f'    控 GDPpc     偏ρ={pr_g:+.4f} (p={pp_g:.2e})')
        print(f'    控 EDI       偏ρ={pr_e:+.4f} (p={pp_e:.2e})')
        print(f'    控 GDPpc+EDI 偏ρ={pr_b:+.4f} (p={pp_b:.2e}, n={nb})')

# ── S_ceil 最低20 vs 最高20 的 BETA ──
d2 = df.dropna(subset=['beta_max']).sort_values('ric_ceil')
bot, top = d2.head(20), d2.tail(20)
u, pu = stats.mannwhitneyu(bot['beta_max'], top['beta_max'])
print('\n' + '=' * 84)
print(f'  S_ceil 最低20 vs 最高20 国·BETA(阶级固化)：'
      f'{bot["beta_max"].median():.3f} vs {top["beta_max"].median():.3f}  '
      f'Mann-Whitney U={u:.0f}, p={pu:.4f}')
print('  （若固化压上限成立：最低20的 BETA 应更高）')

# ── Part 2 · 社会主义国 BETA 读数 ──
print('\n' + '=' * 84)
print('  社会主义国 BETA 读数（BETA 越高=阶级越固化=矛盾越大）')
print('=' * 84)
print(f"  {'Code':<5} {'Entity':<20} {'BETA':>7} {'百分位':>6} {'S_ceil':>8} {'EDI_now':>8} {'GDPpc':>9}")
print('  ' + '─' * 74)
for code in sorted(CURRENT_SOC):
    r = df[df['code'] == code]
    if len(r) == 0 or pd.isna(r['beta_max'].iloc[0]):
        print(f"  {code:<5} {'(GDIM 无 BETA)':<20}")
        continue
    r = r.iloc[0]
    print(f"  {code:<5} {str(r['entity'])[:20]:<20} {r['beta_max']:>7.3f} {r['beta_pctile']:>5.0f}% "
          f"{r['ric_ceil']:>+8.3f} {r['edi_latest']:>8.3f} {r['gdppc_mean']:>9.0f}")
world_med = df['beta_max'].median()
cur = df[df['code'].isin(CURRENT_SOC)].dropna(subset=['beta_max'])
former = df[df['code'].isin(FORMER_SOC)].dropna(subset=['beta_max'])
rest = df[~df['code'].isin(CURRENT_SOC | FORMER_SOC)].dropna(subset=['beta_max'])
print(f'\n  全样本 BETA 中位 = {world_med:.3f}（越高越固化）')
print(f'  现存社会主义 (n={len(cur)}) BETA 中位 = {cur["beta_max"].median():.3f}')
print(f'  前社会主义   (n={len(former)}) BETA 中位 = {former["beta_max"].median():.3f}')
print(f'  其余各国     (n={len(rest)}) BETA 中位 = {rest["beta_max"].median():.3f}')

# ── Part 3 · 民主/威权分组，组内 BETA → S_ceil ──
print('\n' + '=' * 84)
print('  按 EDI(最新) 分组，组内 BETA → S_ceil（拆"阶级独立 vs 搭威权的车"）')
print('=' * 84)
thr = df['edi_latest'].median()
print(f'  分组阈值 = EDI_latest 中位 {thr:.3f}')
for name, sub in [('民主组 EDI≥中位', df[df['edi_latest'] >= thr]),
                  ('威权组 EDI<中位', df[df['edi_latest'] < thr])]:
    rho, p, n = sp(sub, 'beta_max', 'ric_ceil')
    pr, pp, _ = partial_spearman(sub, 'beta_max', 'ric_ceil', ['lgdppc_mean'])
    rk, pk, _ = sp(sub, 'beta_max', 'ric_k')
    print(f'  [{name}] n={n}')
    print(f'    BETA→S_ceil 原始 ρ={rho:+.4f} (p={p:.3f}) | 控GDPpc 偏ρ={pr:+.4f} (p={pp:.3f})')
    print(f'    BETA→k      原始 ρ={rk:+.4f} (p={pk:.3f})')

# ── 图：BETA vs (S_ceil, k)，点色=log GDPpc ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (yc, tt) in zip(axes, [('ric_ceil', 'S_ceil (ceiling)'), ('ric_k', 'k (rise rate)')]):
    v = df.dropna(subset=['beta_max', yc, 'lgdppc_mean'])
    rho, p, n = sp(v, 'beta_max', yc)
    scat = ax.scatter(v['beta_max'], v[yc], c=v['lgdppc_mean'], cmap='viridis', s=26, alpha=0.85)
    z = np.polyfit(v['beta_max'], v[yc], 1)
    xl = np.linspace(v['beta_max'].min(), v['beta_max'].max(), 100)
    ax.plot(xl, np.polyval(z, xl), 'r--', lw=1.3, alpha=0.7)
    ax.set_xlabel('intergenerational persistence BETA (rigidity, cohort1980)')
    ax.set_ylabel(tt)
    ax.set_title(f'{tt}   ρ={rho:+.3f}, p={p:.1e}, n={n}', fontsize=11)
    plt.colorbar(scat, ax=ax, label='log GDP per capita')
plt.tight_layout()
plt.savefig(f'{OUT}/_class_mobility_rank.png', dpi=200, bbox_inches='tight')
d2.to_csv(f'{OUT}/_class_mobility_rank.csv', index=False, encoding='utf-8-sig')
print('\n已存：_class_mobility_rank.csv、_class_mobility_rank.png')
