# -*- coding: utf-8 -*-
"""两套 Richards 拟合参数对比 + 关键国重跑验证 —— 纯只读，不存/不改任何文件。
A 套 = _vdem_fit_all.csv（论文 Phase B 主拟合，列 ric_*）
B 套 = _richards_diagnostics.csv（带 SE/CI/DW 的诊断拟合，列 S_floor/S_ceil/k/t_mid/v）
目的：用程序逐参数算两套差异（治我肉眼数 CSV 数错的病），并对关键国用「当前脚本设置」
重跑 Richards，看重跑结果跟哪套一致 = 哪套是现行代码真值。
"""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

BASE = r'data'

# ---- 拟合函数（与两个脚本逐字一致）----
def logistic(t, S_floor, S_ceil, k, t_mid):
    return S_floor + (S_ceil - S_floor) / (1 + np.exp(-k * (t - t_mid)))

def richards(t, S_floor, S_ceil, k, t_mid, v):
    z = v * np.exp(-k * (t - t_mid))
    z = np.clip(z, 1e-15, 1e15)
    return S_floor + (S_ceil - S_floor) / (1 + z) ** (1 / v)

# ---- 当前脚本设置重跑（照抄 _vdem_fit_all.py / _richards_diagnostics.py 的 Richards 段）----
def refit(t, s):
    p0 = [s.min(), s.max(), 0.03, (t.min() + t.max()) / 2]
    bnds = ([s.min() - 0.5, -1.0, 0.0005, t.min() - 100],
            [0.0, 0.0, 2.0, t.max() + 100])
    plog, _ = curve_fit(logistic, t, s, p0=p0, bounds=bnds, maxfev=30000)
    p0r = [plog[0], plog[1], plog[2], plog[3], 1.0]
    bnds_r = ([s.min() - 0.5, -1.0, 0.0005, t.min() - 100, 0.005],
              [0.0, 0.0, 2.0, t.max() + 100, 20.0])
    pr, _ = curve_fit(richards, t, s, p0=p0r, bounds=bnds_r, maxfev=30000)
    pred = richards(t, *pr)
    r2 = 1 - np.sum((s - pred) ** 2) / np.sum((s - s.mean()) ** 2)
    return pr, r2

# ---- 数据 ----
raw = pd.read_csv(f'{BASE}/women-political-empowerment-index.csv')
raw.columns = ['entity', 'code', 'year', 'wpei', 'region']
raw['S'] = raw['wpei'] - 1

A = pd.read_csv(f'{BASE}/_vdem_fit_all.csv')
B = pd.read_csv(f'{BASE}/_richards_diagnostics.csv')
A = A[A['status'] == 'ok'].copy()
B = B[B['status'] == 'ok'].copy()
m = A.merge(B, on='code', suffixes=('_fa', '_dg'))
print(f'A 套(fit_all) ok={len(A)}；B 套(diagnostics) ok={len(B)}；两套都 ok 的交集={len(m)}\n')

# ---- 逐参数差异（A vs B）----
PAIRS = [('floor', 'ric_floor', 'S_floor'),
         ('ceil',  'ric_ceil',  'S_ceil'),
         ('k',     'ric_k',     'k'),
         ('t_mid', 'ric_tmid',  't_mid'),
         ('v',     'ric_v',     'v'),
         ('R2',    'ric_R2',    'R2')]
print('=' * 78)
print('  两套参数差异 |A−B|（A=fit_all, B=diagnostics）')
print('=' * 78)
print(f'{"参数":<7}{"中位|Δ|":>12}{"最大|Δ|":>12}{"最差国":>8}{"  |Δ|>0.01 国数":>16}')
for name, ca, cb in PAIRS:
    d = (m[ca] - m[cb]).abs()
    worst = m.loc[d.idxmax(), 'code']
    thr = 0.01 if name != 't_mid' else 1.0   # t_mid 单位是年，阈值放宽到 1 年
    print(f'{name:<7}{d.median():>12.4f}{d.max():>12.4f}{worst:>8}{(d > thr).sum():>14} 国')

# ---- v 越界 = 版本不同步铁证 ----
print('\n' + '=' * 78)
print('  v 是否越界（当前代码 bounds 上界=20；越界 ⟹ 该 CSV 来自旧版脚本）')
print('=' * 78)
print(f'  A 套 ric_v > 20 的国家数 : {(m["ric_v"] > 20).sum()}（最大 {m["ric_v"].max():.2f}）')
print(f'  B 套   v   > 20 的国家数 : {(m["v"] > 20).sum()}（最大 {m["v"].max():.2f}）')
print(f'  A 套 v 撞下界 0.005 国数 : {(m["ric_v"] <= 0.0051).sum()}')
print(f'  B 套 v 撞下界 0.005 国数 : {(m["v"] <= 0.0051).sum()}')

# ---- R2 哪套系统更高 ----
print('\n' + '=' * 78)
print('  R² 对比')
print('=' * 78)
print(f'  A 套 mean R²={m["ric_R2"].mean():.4f}  median={m["ric_R2"].median():.4f}')
print(f'  B 套 mean R²={m["R2"].mean():.4f}  median={m["R2"].median():.4f}')
print(f'  A 严格>B 的国家数={ (m["ric_R2"] > m["R2"] + 1e-6).sum() }；'
      f'B 严格>A={ (m["R2"] > m["ric_R2"] + 1e-6).sum() }；'
      f'近乎相等(|Δ|<1e-4)={ ((m["ric_R2"]-m["R2"]).abs() < 1e-4).sum() }')

# ---- 关键国并排 + 重跑验证 ----
KEY = ['CHN', 'USA', 'AFG', 'AGO', 'NOR', 'SWE', 'FRA', 'JPN', 'KOR', 'ISL']
print('\n' + '=' * 78)
print('  关键国：A套 / B套 / 当前代码重跑  三方并排')
print('=' * 78)
for code in KEY:
    r = m[m['code'] == code]
    if len(r) == 0:
        print(f'\n{code}: 不在两套交集'); continue
    r = r.iloc[0]
    sub = raw[raw['code'] == code].sort_values('year')
    t = sub['year'].values.astype(float); s = sub['S'].values
    try:
        pr, r2r = refit(t, s)
        re_str = (f'floor={pr[0]:+.3f} ceil={pr[1]:+.3f} k={pr[2]:.4f} '
                  f'tmid={pr[3]:.1f} v={pr[4]:.3f} R²={r2r:.4f}')
    except Exception as e:
        re_str = f'重跑失败 {e}'; pr = None
    print(f'\n{code} ({r["entity_fa"]}):')
    print(f'  A套(fit_all)   floor={r["ric_floor"]:+.3f} ceil={r["ric_ceil"]:+.3f} '
          f'k={r["ric_k"]:.4f} tmid={r["ric_tmid"]:.1f} v={r["ric_v"]:.3f} R²={r["ric_R2"]:.4f}')
    print(f'  B套(diagnost.) floor={r["S_floor"]:+.3f} ceil={r["S_ceil"]:+.3f} '
          f'k={r["k"]:.4f} tmid={r["t_mid"]:.1f} v={r["v"]:.3f} R²={r["R2"]:.4f}')
    print(f'  重跑(现行代码) {re_str}')
    if pr is not None:
        # 用 (floor,ceil,k,tmid) 欧氏距离判重跑更接近谁（v 不可靠，单列出）
        va = np.array([r['ric_floor'], r['ric_ceil'], r['ric_k'], r['ric_tmid']])
        vb = np.array([r['S_floor'], r['S_ceil'], r['k'], r['t_mid']])
        vr = np.array([pr[0], pr[1], pr[2], pr[3]])
        # tmid 量纲大，标准化：tmid 差除以 10
        scale = np.array([1, 1, 1, 0.1])
        dA = np.sqrt(np.sum(((vr - va) * scale) ** 2))
        dB = np.sqrt(np.sum(((vr - vb) * scale) ** 2))
        who = 'A套' if dA < dB else ('B套' if dB < dA else '两套等距')
        print(f'  → 重跑(形状参数)更接近：{who}  (距A={dA:.4f}, 距B={dB:.4f})')

print('\n（本脚本只读：读 3 个现成 CSV + 内存重跑关键国拟合，未保存/未修改任何文件。）')
