"""
V-Dem S(t) 正式拟合：多国 logistic + Richards 曲线
S(t) = WPEI - 1，负半轴，Phase B 正S上升

两个模型对比：
1. Standard logistic（对称sigmoid）
2. Richards / Generalized logistic（不对称sigmoid，多一个形状参数 v）
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 数据 ──────────────────────────────────────────────

df = pd.read_csv(r"data\women-political-empowerment-index.csv")
df.columns = ['entity', 'code', 'year', 'wpei', 'region']
df['S'] = df['wpei'] - 1  # 变换到负半轴

# ── 国家选择 ──────────────────────────────────────────

COUNTRIES = {
    'CHN': 'China',
    'NOR': 'Norway',
    'FRA': 'France',
    'SWE': 'Sweden',
    'JPN': 'Japan',
    'SAU': 'Saudi Arabia',
    'ISL': 'Iceland',
    'GBR': 'United Kingdom',
    'USA': 'United States',
    'FIN': 'Finland',
    'KOR': 'South Korea',  # 东亚对照
}

# ── 拟合函数 ──────────────────────────────────────────

def logistic(t, S_floor, S_ceil, k, t_mid):
    """标准 logistic sigmoid（对称）"""
    return S_floor + (S_ceil - S_floor) / (1 + np.exp(-k * (t - t_mid)))

def richards(t, S_floor, S_ceil, k, t_mid, v):
    """Richards / Generalized logistic（不对称）
    v > 1: 上半段比下半段慢（拐点偏下）
    v < 1: 下半段比上半段慢（拐点偏上）
    v = 1: 退化为标准 logistic
    """
    return S_floor + (S_ceil - S_floor) / (1 + v * np.exp(-k * (t - t_mid))) ** (1 / v)

# ── 拟合 ──────────────────────────────────────────────

def fit_country(code, name, min_years=30):
    """对单个国家做 logistic + Richards 拟合"""
    sub = df[df['code'] == code].sort_values('year')
    if len(sub) < min_years:
        print(f"  {name} ({code}): only {len(sub)} years, skipping")
        return None

    t = sub['year'].values.astype(float)
    s = sub['S'].values

    result = {'code': code, 'name': name, 'years': f"{t.min():.0f}-{t.max():.0f}", 'n': len(t)}

    # ── Logistic ──
    try:
        p0_log = [s.min(), s.max(), 0.03, (t.min() + t.max()) / 2]
        bounds_log = ([s.min() - 0.5, -1.0, 0.001, t.min() - 50],
                      [0.0, 0.01, 1.0, t.max() + 50])
        popt_l, pcov_l = curve_fit(logistic, t, s, p0=p0_log, bounds=bounds_log, maxfev=20000)
        s_pred_l = logistic(t, *popt_l)
        ss_res = np.sum((s - s_pred_l) ** 2)
        ss_tot = np.sum((s - np.mean(s)) ** 2)
        r2_l = 1 - ss_res / ss_tot
        result['logistic'] = {
            'params': {'S_floor': popt_l[0], 'S_ceil': popt_l[1],
                       'k': popt_l[2], 't_mid': popt_l[3]},
            'R2': r2_l, 'pred': s_pred_l
        }
    except Exception as e:
        print(f"  {name} logistic fit failed: {e}")
        result['logistic'] = None

    # ── Richards ──
    try:
        # 用 logistic 结果作为初始值
        if result.get('logistic'):
            lp = result['logistic']['params']
            p0_r = [lp['S_floor'], lp['S_ceil'], lp['k'], lp['t_mid'], 1.0]
        else:
            p0_r = [s.min(), s.max(), 0.03, (t.min() + t.max()) / 2, 1.0]

        bounds_r = ([s.min() - 0.5, -1.0, 0.001, t.min() - 50, 0.01],
                    [0.0, 0.01, 1.0, t.max() + 50, 20.0])
        popt_r, pcov_r = curve_fit(richards, t, s, p0=p0_r, bounds=bounds_r, maxfev=20000)
        s_pred_r = richards(t, *popt_r)
        ss_res = np.sum((s - s_pred_r) ** 2)
        ss_tot = np.sum((s - np.mean(s)) ** 2)
        r2_r = 1 - ss_res / ss_tot
        result['richards'] = {
            'params': {'S_floor': popt_r[0], 'S_ceil': popt_r[1],
                       'k': popt_r[2], 't_mid': popt_r[3], 'v': popt_r[4]},
            'R2': r2_r, 'pred': s_pred_r
        }
    except Exception as e:
        print(f"  {name} Richards fit failed: {e}")
        result['richards'] = None

    result['t'] = t
    result['s'] = s
    return result

# ── 主程序 ──────────────────────────────────────────

print("=" * 70)
print("  V-Dem S(t) Formal Curve Fitting")
print("  S(t) = WPEI - 1, Phase B sigmoid")
print("=" * 70)

results = {}
for code, name in COUNTRIES.items():
    print(f"\n{'─' * 50}")
    print(f"  {name} ({code})")
    r = fit_country(code, name)
    if r:
        results[code] = r
        if r.get('logistic'):
            lp = r['logistic']['params']
            print(f"  Logistic:  R2={r['logistic']['R2']:.4f}  "
                  f"floor={lp['S_floor']:.3f}  ceil={lp['S_ceil']:.3f}  "
                  f"k={lp['k']:.4f}  t_mid={lp['t_mid']:.1f}")
        if r.get('richards'):
            rp = r['richards']['params']
            print(f"  Richards:  R2={r['richards']['R2']:.4f}  "
                  f"floor={rp['S_floor']:.3f}  ceil={rp['S_ceil']:.3f}  "
                  f"k={rp['k']:.4f}  t_mid={rp['t_mid']:.1f}  v={rp['v']:.3f}")
            if r.get('logistic'):
                delta_r2 = r['richards']['R2'] - r['logistic']['R2']
                print(f"  Richards vs Logistic: dR2={delta_r2:+.6f}  "
                      f"v={'asymmetric' if abs(rp['v'] - 1.0) > 0.1 else 'symmetric (~logistic)'}")

# ── 汇总表 ──────────────────────────────────────────

print("\n" + "=" * 70)
print("  SUMMARY TABLE")
print("=" * 70)
print(f"{'Country':<18} {'Years':<12} {'n':>4}  {'R2_log':>7} {'R2_ric':>7} {'v':>6} {'t_mid':>7} {'k':>7} {'S_floor':>8} {'S_ceil':>7}")
print("─" * 100)

for code in COUNTRIES:
    if code not in results:
        continue
    r = results[code]
    rl = r.get('logistic')
    rr = r.get('richards')
    r2l = f"{rl['R2']:.4f}" if rl else "  FAIL"
    r2r = f"{rr['R2']:.4f}" if rr else "  FAIL"
    v = f"{rr['params']['v']:.2f}" if rr else "  -"
    t_mid = f"{rr['params']['t_mid']:.1f}" if rr else (f"{rl['params']['t_mid']:.1f}" if rl else "  -")
    k = f"{rr['params']['k']:.4f}" if rr else (f"{rl['params']['k']:.4f}" if rl else "  -")
    s_floor = f"{rr['params']['S_floor']:.3f}" if rr else (f"{rl['params']['S_floor']:.3f}" if rl else "  -")
    s_ceil = f"{rr['params']['S_ceil']:.3f}" if rr else (f"{rl['params']['S_ceil']:.3f}" if rl else "  -")
    print(f"{r['name']:<18} {r['years']:<12} {r['n']:>4}  {r2l:>7} {r2r:>7} {v:>6} {t_mid:>7} {k:>7} {s_floor:>8} {s_ceil:>7}")

# ── 画图 ──────────────────────────────────────────

n_countries = len(results)
ncols = 3
nrows = (n_countries + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True)
axes = axes.flatten()

for i, (code, r) in enumerate(results.items()):
    ax = axes[i]
    ax.scatter(r['t'], r['s'], s=4, alpha=0.5, color='gray', label='V-Dem data')

    if r.get('logistic'):
        t_smooth = np.linspace(r['t'].min(), r['t'].max(), 500)
        lp = r['logistic']['params']
        ax.plot(t_smooth, logistic(t_smooth, **lp), 'b-', lw=1.5,
                label=f"Logistic R2={r['logistic']['R2']:.3f}")

    if r.get('richards'):
        rp = r['richards']['params']
        ax.plot(t_smooth, richards(t_smooth, **rp), 'r--', lw=1.5,
                label=f"Richards R2={r['richards']['R2']:.3f}, v={rp['v']:.2f}")
        # 标记拐点
        ax.axvline(rp['t_mid'], color='r', ls=':', alpha=0.3)

    ax.axhline(0, color='k', ls='-', alpha=0.2)
    ax.set_title(f"{r['name']} ({r['years']})", fontsize=11)
    ax.set_ylabel("S(t) = WPEI − 1")
    ax.legend(fontsize=7, loc='lower right')
    ax.set_ylim(-1.05, 0.05)

# 隐藏多余子图
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Phase B Sigmoid Fit: S(t) = WPEI − 1", fontsize=14, fontweight='bold')
plt.savefig(r"data\_vdem_fit.png", dpi=200)
print(f"\nFigure saved: _vdem_fit.png")
plt.close()

# ── 拐点排序（论文用） ──────────────────────────────

print("\n" + "=" * 70)
print("  INFLECTION POINT RANKING (Richards t_mid)")
print("=" * 70)
ranked = []
for code, r in results.items():
    rr = r.get('richards') or r.get('logistic')
    if rr:
        ranked.append((r['name'], rr['params']['t_mid'], rr['R2']))
ranked.sort(key=lambda x: x[1])
for name, tmid, r2 in ranked:
    print(f"  {name:<18} t_mid = {tmid:.1f}  (R2 = {r2:.4f})")
