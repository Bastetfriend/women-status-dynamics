"""
模型比较：AIC/BIC 信息准则
5 个竞争模型 × 191 国

1. Linear:          S = a + bt                    (k=2)
2. Quadratic:       S = a + bt + ct^2             (k=3)
3. Piecewise-linear: S = a + b1*t + b2*max(0,t-tb) (k=4, 含断点)
4. Logistic:        4-param sigmoid               (k=4)
5. Richards:        5-param asymmetric sigmoid     (k=5)

AIC = n*ln(RSS/n) + 2k
BIC = n*ln(RSS/n) + k*ln(n)
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')

# ── 数据 ──

df = pd.read_csv(r"data\women-political-empowerment-index.csv")
df.columns = ['entity', 'code', 'year', 'wpei', 'region']
df['S'] = df['wpei'] - 1

# ── 模型定义 ──

def m_linear(t, a, b):
    return a + b * t

def m_quadratic(t, a, b, c):
    return a + b * t + c * t**2

def m_piecewise(t, a, b1, b2, tb):
    return a + b1 * t + b2 * np.maximum(0, t - tb)

def m_logistic(t, S_floor, S_ceil, k, t_mid):
    return S_floor + (S_ceil - S_floor) / (1 + np.exp(-k * (t - t_mid)))

def m_richards(t, S_floor, S_ceil, k, t_mid, v):
    z = v * np.exp(-k * (t - t_mid))
    z = np.clip(z, 1e-15, 1e15)
    return S_floor + (S_ceil - S_floor) / (1 + z) ** (1 / v)

# ── 信息准则 ──

def aic_bic(n, rss, k):
    if rss <= 0 or n <= k:
        return np.inf, np.inf
    ll = n * np.log(rss / n)  # proportional to -2*loglik (Gaussian)
    aic = ll + 2 * k
    bic = ll + k * np.log(n)
    return aic, bic

# ── 单国拟合 ──

def fit_all_models(code, entity, t_raw, s):
    n = len(t_raw)
    # 中心化时间（数值稳定性）
    t0 = t_raw.mean()
    t = t_raw - t0

    row = {'code': code, 'entity': entity, 'n': n}

    # ── 1. Linear (k=2) ──
    try:
        popt, _ = curve_fit(m_linear, t, s)
        rss = np.sum((s - m_linear(t, *popt))**2)
        r2 = 1 - rss / np.sum((s - s.mean())**2)
        a, b = aic_bic(n, rss, 2)
        row.update({'lin_R2': r2, 'lin_AIC': a, 'lin_BIC': b, 'lin_RSS': rss})
    except:
        row.update({'lin_R2': np.nan, 'lin_AIC': np.inf, 'lin_BIC': np.inf})

    # ── 2. Quadratic (k=3) ──
    try:
        popt, _ = curve_fit(m_quadratic, t, s)
        rss = np.sum((s - m_quadratic(t, *popt))**2)
        r2 = 1 - rss / np.sum((s - s.mean())**2)
        a, b = aic_bic(n, rss, 3)
        row.update({'quad_R2': r2, 'quad_AIC': a, 'quad_BIC': b, 'quad_RSS': rss})
    except:
        row.update({'quad_R2': np.nan, 'quad_AIC': np.inf, 'quad_BIC': np.inf})

    # ── 3. Piecewise-linear (k=4) ──
    try:
        # 网格搜索断点位置（用原始时间）
        best_rss_pw = np.inf
        best_popt_pw = None
        candidates = np.linspace(t.min() + 5, t.max() - 5, 50)
        for tb_try in candidates:
            try:
                p0 = [s[0], (s[-1]-s[0])/(t[-1]-t[0]), 0, tb_try]
                popt_pw, _ = curve_fit(m_piecewise, t, s, p0=p0,
                                       bounds=([-2, -0.1, -0.1, t.min()],
                                               [0.5, 0.1, 0.1, t.max()]),
                                       maxfev=5000)
                rss_pw = np.sum((s - m_piecewise(t, *popt_pw))**2)
                if rss_pw < best_rss_pw:
                    best_rss_pw = rss_pw
                    best_popt_pw = popt_pw
            except:
                pass
        if best_popt_pw is not None:
            r2 = 1 - best_rss_pw / np.sum((s - s.mean())**2)
            a, b = aic_bic(n, best_rss_pw, 4)
            row.update({'pw_R2': r2, 'pw_AIC': a, 'pw_BIC': b, 'pw_RSS': best_rss_pw})
        else:
            row.update({'pw_R2': np.nan, 'pw_AIC': np.inf, 'pw_BIC': np.inf})
    except:
        row.update({'pw_R2': np.nan, 'pw_AIC': np.inf, 'pw_BIC': np.inf})

    # ── 4. Logistic (k=4) ──
    try:
        # 用原始时间（非中心化）
        p0 = [s.min(), s.max(), 0.03, t0]
        bounds_l = ([s.min() - 0.5, -1.0, 0.0005, t_raw.min() - 100],
                    [0.0, 0.0, 2.0, t_raw.max() + 100])
        popt, _ = curve_fit(m_logistic, t_raw, s, p0=p0, bounds=bounds_l, maxfev=30000)
        rss = np.sum((s - m_logistic(t_raw, *popt))**2)
        r2 = 1 - rss / np.sum((s - s.mean())**2)
        a, b = aic_bic(n, rss, 4)
        row.update({'log_R2': r2, 'log_AIC': a, 'log_BIC': b, 'log_RSS': rss})
    except:
        row.update({'log_R2': np.nan, 'log_AIC': np.inf, 'log_BIC': np.inf})

    # ── 5. Richards (k=5) ──
    try:
        if not np.isnan(row.get('log_R2', np.nan)):
            p0r = list(popt) + [1.0]
        else:
            p0r = [s.min(), s.max(), 0.03, t0, 1.0]
        bounds_r = ([s.min() - 0.5, -1.0, 0.0005, t_raw.min() - 100, 0.005],
                    [0.0, 0.0, 2.0, t_raw.max() + 100, 20.0])
        poptr, _ = curve_fit(m_richards, t_raw, s, p0=p0r, bounds=bounds_r, maxfev=30000)
        rss = np.sum((s - m_richards(t_raw, *poptr))**2)
        r2 = 1 - rss / np.sum((s - s.mean())**2)
        a, b = aic_bic(n, rss, 5)
        row.update({'ric_R2': r2, 'ric_AIC': a, 'ric_BIC': b, 'ric_RSS': rss})
    except:
        row.update({'ric_R2': np.nan, 'ric_AIC': np.inf, 'ric_BIC': np.inf})

    # ── Winner ──
    models = ['lin', 'quad', 'pw', 'log', 'ric']
    labels = ['Linear', 'Quadratic', 'Piecewise', 'Logistic', 'Richards']
    aics = [row.get(f'{m}_AIC', np.inf) for m in models]
    bics = [row.get(f'{m}_BIC', np.inf) for m in models]
    row['aic_winner'] = labels[np.argmin(aics)]
    row['bic_winner'] = labels[np.argmin(bics)]
    row['aic_best'] = min(aics)
    row['bic_best'] = min(bics)

    # Delta AIC/BIC vs best simple (= min of lin/quad/pw)
    simple_aic = min(aics[:3])
    simple_bic = min(bics[:3])
    sig_aic = min(aics[3:])  # best sigmoid (logistic or richards)
    sig_bic = min(bics[3:])
    row['dAIC_sig_vs_simple'] = sig_aic - simple_aic  # negative = sigmoid wins
    row['dBIC_sig_vs_simple'] = sig_bic - simple_bic

    return row

# ── 主程序 ──

MIN_YEARS = 30
all_codes = df.groupby('code').agg(
    entity=('entity', 'first'), n=('year', 'count')
).reset_index()
all_codes = all_codes[all_codes['n'] >= MIN_YEARS].sort_values('code')

# 过滤 flat 国家
flat_codes = set()
for _, info in all_codes.iterrows():
    sub = df[df['code'] == info['code']]
    if sub['S'].max() - sub['S'].min() < 0.05:
        flat_codes.add(info['code'])
all_codes = all_codes[~all_codes['code'].isin(flat_codes)]

print(f"Fitting {len(all_codes)} countries (excluded {len(flat_codes)} flat)...")
print(f"5 models x {len(all_codes)} countries = {5*len(all_codes)} fits\n")

rows = []
for _, info in all_codes.iterrows():
    sub = df[df['code'] == info['code']].sort_values('year')
    t = sub['year'].values.astype(float)
    s = sub['S'].values
    row = fit_all_models(info['code'], info['entity'], t, s)
    rows.append(row)
    if len(rows) % 20 == 0:
        print(f"  {len(rows)} done...")

results = pd.DataFrame(rows)

# ── 存 CSV ──

csv_path = r"data\_vdem_model_comparison.csv"
results.to_csv(csv_path, index=False, encoding='utf-8-sig')

# ── 汇总 ──

print(f"\n{'='*70}")
print(f"  MODEL COMPARISON SUMMARY (n={len(results)})")
print(f"{'='*70}")

# AIC winner counts
print(f"\n  AIC Winner:")
for label, count in results['aic_winner'].value_counts().items():
    print(f"    {label:<12} {count:>4} ({count/len(results)*100:.1f}%)")

print(f"\n  BIC Winner:")
for label, count in results['bic_winner'].value_counts().items():
    print(f"    {label:<12} {count:>4} ({count/len(results)*100:.1f}%)")

# Sigmoid vs Simple
print(f"\n  Sigmoid (Logistic|Richards) vs Best Simple (Lin|Quad|PW):")
sig_wins_aic = (results['dAIC_sig_vs_simple'] < 0).sum()
sig_wins_bic = (results['dBIC_sig_vs_simple'] < 0).sum()
print(f"    Sigmoid wins AIC: {sig_wins_aic}/{len(results)} ({sig_wins_aic/len(results)*100:.1f}%)")
print(f"    Sigmoid wins BIC: {sig_wins_bic}/{len(results)} ({sig_wins_bic/len(results)*100:.1f}%)")
print(f"    Mean dAIC (sig-simple): {results['dAIC_sig_vs_simple'].mean():.1f}")
print(f"    Mean dBIC (sig-simple): {results['dBIC_sig_vs_simple'].mean():.1f}")

# Strong wins (dAIC < -10)
strong_aic = (results['dAIC_sig_vs_simple'] < -10).sum()
strong_bic = (results['dBIC_sig_vs_simple'] < -10).sum()
print(f"    Strong sigmoid win (dAIC<-10): {strong_aic} ({strong_aic/len(results)*100:.1f}%)")
print(f"    Strong sigmoid win (dBIC<-10): {strong_bic} ({strong_bic/len(results)*100:.1f}%)")

# R2 comparison table
print(f"\n  Mean R2 by model:")
for m, label in [('lin','Linear'), ('quad','Quadratic'), ('pw','Piecewise'),
                  ('log','Logistic'), ('ric','Richards')]:
    col = f'{m}_R2'
    vals = results[col].dropna()
    print(f"    {label:<12} mean={vals.mean():.4f}  median={vals.median():.4f}  "
          f">=0.95: {(vals>=0.95).sum():>3}  >=0.90: {(vals>=0.90).sum():>3}")

# ── 关键国家详细 ──

highlight = ['CHN', 'USA', 'NOR', 'FRA', 'SWE', 'JPN', 'GBR', 'ISL', 'FIN', 'KOR', 'SAU']
print(f"\n{'='*70}")
print(f"  KEY COUNTRIES DETAIL")
print(f"{'='*70}")
print(f"{'Country':<14} {'Lin_AIC':>8} {'Quad_AIC':>9} {'PW_AIC':>8} {'Log_AIC':>8} {'Ric_AIC':>8} {'Winner':<10} {'dAIC':>6}")
print("-" * 85)
for code in highlight:
    r = results[results['code'] == code]
    if len(r) == 0:
        continue
    r = r.iloc[0]
    print(f"{r['entity']:<14} {r['lin_AIC']:>8.1f} {r['quad_AIC']:>9.1f} {r['pw_AIC']:>8.1f} "
          f"{r['log_AIC']:>8.1f} {r['ric_AIC']:>8.1f} {r['aic_winner']:<10} {r['dAIC_sig_vs_simple']:>6.1f}")

# ── 画图 ──

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

# 1. AIC winner 饼图
ax = axes[0]
counts = results['aic_winner'].value_counts()
colors = {'Richards': '#d62728', 'Logistic': '#ff7f0e', 'Piecewise': '#2ca02c',
          'Quadratic': '#1f77b4', 'Linear': '#7f7f7f'}
ax.bar(counts.index, counts.values,
       color=[colors.get(x, 'gray') for x in counts.index])
ax.set_ylabel('Count')
ax.set_title(f'AIC Winner (n={len(results)})')
for i, (label, val) in enumerate(counts.items()):
    ax.text(i, val + 1, f'{val}\n({val/len(results)*100:.0f}%)', ha='center', fontsize=9)

# 2. dAIC 分布（sigmoid vs simple）
ax = axes[1]
daic = results['dAIC_sig_vs_simple']
ax.hist(daic, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0, color='r', ls='--', label='Break-even')
ax.axvline(-10, color='orange', ls='--', alpha=0.5, label='Strong win (dAIC=-10)')
ax.set_xlabel('dAIC (Sigmoid - Best Simple)')
ax.set_ylabel('Count')
ax.set_title('Sigmoid vs Simple: dAIC')
ax.legend(fontsize=8)
# 标注百分比
pct_left = (daic < 0).sum() / len(daic) * 100
ax.text(0.05, 0.95, f'Sigmoid wins:\n{pct_left:.0f}%', transform=ax.transAxes,
        fontsize=11, va='top', fontweight='bold', color='steelblue')

# 3. R2 箱线图
ax = axes[2]
r2_data = []
r2_labels = []
for m, label in [('lin','Linear\n(k=2)'), ('quad','Quadratic\n(k=3)'),
                  ('pw','Piecewise\n(k=4)'), ('log','Logistic\n(k=4)'),
                  ('ric','Richards\n(k=5)')]:
    vals = results[f'{m}_R2'].dropna().values
    r2_data.append(vals)
    r2_labels.append(label)
bp = ax.boxplot(r2_data, labels=r2_labels, patch_artist=True)
model_colors = ['#7f7f7f', '#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
for patch, color in zip(bp['boxes'], model_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax.set_ylabel('R2')
ax.set_title('R2 Distribution by Model')
ax.axhline(0.95, color='gray', ls=':', alpha=0.3)

fig.suptitle('Model Selection: AIC/BIC Comparison (191 Countries)', fontsize=13, fontweight='bold')
plt.savefig(r"data\_vdem_model_comparison.png", dpi=200)
print(f"\nFigure saved: _vdem_model_comparison.png")
plt.close()

# ── 论文用一句话 ──

print(f"\n{'='*70}")
print(f"  PAPER-READY STATEMENT")
print(f"{'='*70}")
print(f"  Across {len(results)} countries/territories with >=30 years of data,")
print(f"  the Richards generalized logistic was selected as the best model")
print(f"  by AIC in {(results['aic_winner']=='Richards').sum()} cases ({(results['aic_winner']=='Richards').sum()/len(results)*100:.1f}%)")
print(f"  and by BIC in {(results['bic_winner']=='Richards').sum()} cases ({(results['bic_winner']=='Richards').sum()/len(results)*100:.1f}%).")
sig_total_aic = ((results['aic_winner']=='Richards') | (results['aic_winner']=='Logistic')).sum()
sig_total_bic = ((results['bic_winner']=='Richards') | (results['bic_winner']=='Logistic')).sum()
print(f"  Sigmoid family (Logistic+Richards) wins AIC in {sig_total_aic} ({sig_total_aic/len(results)*100:.1f}%)")
print(f"  and BIC in {sig_total_bic} ({sig_total_bic/len(results)*100:.1f}%) of cases.")
print(f"  Mean dAIC(sigmoid-simple) = {results['dAIC_sig_vs_simple'].mean():.1f}")
print(f"  Mean dBIC(sigmoid-simple) = {results['dBIC_sig_vs_simple'].mean():.1f}")
