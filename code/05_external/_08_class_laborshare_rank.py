# -*- coding: utf-8 -*-
"""阶级×性别 · 初次分配公平度(劳动收入份额 labsh) vs 性别上限/爬升 —— 横截面秩相关＋控发展＋控民主。
谢 2026-07-08 定：基尼跨国=发展水平马甲、且与 EDI 缠绕(自己打自己)，弃用。改用【初次分配公平度】，
操作化=PWT labor share(labsh，再分配之前劳vs资的分配=生产关系本身，比值、较不吃发展、较独立于民主轴)。
纵轴不变：每国 Richards 的 S_ceil(ric_ceil,上限) 与 k(ric_k,爬升速率)。一国一点。
横轴 labsh 每国取 mean+latest。统计：Spearman 秩相关(+Pearson+R²)；关键=**偏相关控 GDP per capita、
控 EDI(电子民主)**——验"扣掉发展与民主后阶级还剩不剩独立信号"，防自己打自己。
只读 _vdem_fit_all.csv + pwt110.xlsx + electoral-democracy-index.csv；输出相关表+图。"""
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

# ── 纵轴：Richards S_ceil + k（status=ok）──
params = pd.read_csv(f'{BASE}/实验输出_CSV/_vdem_fit_all.csv')
params = params[params['status'] == 'ok'][['code', 'entity', 'ric_ceil', 'ric_k']].copy()

# ── 横轴：PWT labor share + GDP per capita（每国 mean+latest）──
pwt = pd.read_excel(PWT, sheet_name='Data')
pwt['gdppc'] = pwt['cgdpo'] / pwt['pop']          # 发展水平代理：产出侧实际GDP / 人口


def pc_mean_latest(col):
    g = pwt.dropna(subset=[col])
    m = g.groupby('countrycode')[col].mean()
    l = g.sort_values('year').groupby('countrycode').last()[col]
    return m, l


lab_m, lab_l = pc_mean_latest('labsh')
gpc_m, gpc_l = pc_mean_latest('gdppc')
dev = pd.DataFrame({'labsh_mean': lab_m, 'labsh_latest': lab_l,
                    'gdppc_mean': gpc_m, 'gdppc_latest': gpc_l})
dev['lgdppc_mean'] = np.log(dev['gdppc_mean'])
dev['lgdppc_latest'] = np.log(dev['gdppc_latest'])

# ── 控制轴2：EDI（电子民主，每国 mean）──
edi = pd.read_csv(f'{BASE}/实验输出_CSV/electoral-democracy-index.csv')
edi.columns = ['entity', 'code', 'year', 'edi', 'region']
edi_m = edi[edi['code'].notna()].groupby('code')['edi'].mean().rename('edi_mean')

df = (params.merge(dev, left_on='code', right_index=True, how='left')
            .merge(edi_m, left_on='code', right_index=True, how='left'))


def sp(d, x, y):
    v = d.dropna(subset=[x, y])
    rho, p = stats.spearmanr(v[x], v[y])
    return rho, p, len(v)


def partial_spearman(d, x, y, controls):
    """偏 Spearman：x,y,控制变量都取秩→x、y 各自对控制变量线性回归取残差→残差相关。"""
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


print('=' * 80)
print('  独立性诊断：labsh 自身跟 发展/民主 缠多紧（够不够格当独立阶级轴）')
print('=' * 80)
for a, b in [('labsh_mean', 'lgdppc_mean'), ('labsh_mean', 'edi_mean'),
             ('edi_mean', 'lgdppc_mean')]:
    rho, p, n = sp(df, a, b)
    print(f'  {a:<12} ~ {b:<12}: Spearman ρ={rho:+.3f} (p={p:.2e}, n={n})')

for ylab, ycol in [('S_ceil(上限)', 'ric_ceil'), ('k(爬升速率)', 'ric_k')]:
    print('\n' + '=' * 80)
    print(f'  labsh(初次分配公平度)  →  {ylab}   [原始 / 控GDPpc / 控EDI / 双控]')
    print('=' * 80)
    for xcol in ['labsh_mean', 'labsh_latest']:
        ctrl = 'lgdppc_mean' if 'mean' in xcol else 'lgdppc_latest'
        rho, p, n = sp(df, xcol, ycol)
        pr_g, pp_g, _ = partial_spearman(df, xcol, ycol, [ctrl])
        pr_e, pp_e, _ = partial_spearman(df, xcol, ycol, ['edi_mean'])
        pr_b, pp_b, nb = partial_spearman(df, xcol, ycol, [ctrl, 'edi_mean'])
        print(f'  [{xcol}]  n={n}')
        print(f'    原始         Spearman ρ={rho:+.4f} (p={p:.2e})')
        print(f'    控 GDPpc     偏ρ={pr_g:+.4f} (p={pp_g:.2e})')
        print(f'    控 EDI       偏ρ={pr_e:+.4f} (p={pp_e:.2e})')
        print(f'    控 GDPpc+EDI 偏ρ={pr_b:+.4f} (p={pp_b:.2e}, n={nb})')

# ── S_ceil 最低20 vs 最高20 的 labsh ──
d2 = df.dropna(subset=['labsh_mean']).sort_values('ric_ceil')
bot, top = d2.head(20), d2.tail(20)
u, pu = stats.mannwhitneyu(bot['labsh_mean'], top['labsh_mean'])
print('\n' + '=' * 80)
print(f'  S_ceil 最低20 vs 最高20 国·labsh(均值)：'
      f'{bot["labsh_mean"].median():.3f} vs {top["labsh_mean"].median():.3f}  '
      f'Mann-Whitney U={u:.0f}, p={pu:.4f}')

# ── 图：labsh vs (S_ceil, k)，点色=log GDPpc（显影发展混淆）──
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (yc, tt) in zip(axes, [('ric_ceil', 'S_ceil (ceiling)'), ('ric_k', 'k (rise rate)')]):
    v = df.dropna(subset=['labsh_mean', yc, 'lgdppc_mean'])
    rho, p, n = sp(v, 'labsh_mean', yc)
    scat = ax.scatter(v['labsh_mean'], v[yc], c=v['lgdppc_mean'], cmap='viridis', s=26, alpha=0.85)
    z = np.polyfit(v['labsh_mean'], v[yc], 1)
    xl = np.linspace(v['labsh_mean'].min(), v['labsh_mean'].max(), 100)
    ax.plot(xl, np.polyval(z, xl), 'r--', lw=1.3, alpha=0.7)
    ax.set_xlabel('labor share (labsh, mean)')
    ax.set_ylabel(tt)
    ax.set_title(f'{tt}   ρ={rho:+.3f}, p={p:.1e}, n={n}', fontsize=11)
    plt.colorbar(scat, ax=ax, label='log GDP per capita')
plt.tight_layout()
plt.savefig(f'{OUT}/_class_laborshare_rank.png', dpi=200, bbox_inches='tight')
d2.to_csv(f'{OUT}/_class_laborshare_rank.csv', index=False, encoding='utf-8-sig')
print('\n已存：_class_laborshare_rank.csv、_class_laborshare_rank.png')
