# -*- coding: utf-8 -*-
"""
反演加速度 a(t)：把 a(t) 参数化为分段常数，用线性最小二乘解出形状
关系：L_obs = Δ * ∫_{T_start}^{ric_tmid} ∫_{T_start}^{t} a(s) ds dt
"""
import csv
import numpy as np

DELTA = 479.84
ADV = {'PRT', 'ESP', 'NLD', 'GBR', 'FRA'}
EX = {'ISR', 'CPV', 'MOZ', 'STP', 'USA', 'AUS', 'ETH', 'TLS', 'CUB', 'PRK'}

ns = {}
src = open(r'data\_tstart_acceleration.py', encoding='utf-8').read().split('def main()')[0]
exec(src, ns)
TS = ns['T_START']

rows = list(csv.DictReader(open(r'data\_vdem_fit_all.csv', encoding='utf-8-sig')))

# 收集 (T_start, ric_tmid, L)
pairs = []
for r in rows:
    c = r['code']
    if c.startswith('OWID_') or r['status'] == 'flat' or c in ADV or c not in TS or c in EX:
        continue
    try:
        tmid = float(r['ric_tmid'])
    except:
        continue
    ts = TS[c][0]
    t = tmid - ts
    if t <= 0:
        continue
    L = DELTA - t
    pairs.append((ts, tmid, L, c))

pairs.sort()
T0 = np.array([p[0] for p in pairs], float)
R = np.array([p[1] for p in pairs], float)
Lobs = np.array([p[2] for p in pairs], float)
n = len(pairs)
print('样本数', n, 'T_start 范围 [%d, %d]' % (T0.min(), T0.max()))

# 分段常数 a(t)：段边界
seg_edges = np.arange(1700, 2011, 25.0)  # 1700..2010 每25年
n_seg = len(seg_edges) - 1
print('分段边界', seg_edges.astype(int), '共 %d 段' % n_seg)

def kernel_seg(T0v, Rv, A, B):
    """∫_{T0}^{R} max(0, min(t,B)-max(T0,A)) dt，向量化"""
    lo = np.maximum(T0v, A)
    hi = np.minimum(Rv, B)
    # 分段线性积分
    inner = np.clip(hi - lo, 0, None)  # 长度
    # 直接数值：用解析
    out = np.zeros_like(T0v, float)
    # case 全部覆盖：g(t)=clip(min(t,B)-max(T0,A),0,∞)
    # ∫ = 对 t in [max(T0,A), min(R,B)] 的 (t - max(T0,A)) dt + (B-max(T0,A))*max(0, R-B) 当 A<=T0
    # 统一：∫_{T0}^{R} clip(min(t,B)-lo0,0) dt, lo0=max(T0,A)
    # 当 t<=B: min(t,B)=t, g=t-lo0 (t>=lo0)
    # 当 t>B: g=B-lo0
    # 数值积分更稳
    for i in range(len(T0v)):
        t0, rv = T0v[i], Rv[i]
        if rv <= t0:
            out[i] = 0
            continue
        lo0 = max(t0, A)
        if B <= t0:
            out[i] = 0
            continue
        # 分段：t in [max(t0,A), min(rv,B)]: g=t-lo0 ; t in [min(rv,B), rv]: g=B-lo0 (若 B<rv)
        t1 = min(rv, B)
        if t1 > lo0:
            out[i] += 0.5 * (t1 - lo0) ** 2  # ∫ (t-lo0) dt
        if B < rv and B > lo0:
            out[i] += (B - lo0) * (rv - B)
    return out * DELTA

# 构建 M 矩阵
M = np.zeros((n, n_seg))
for k in range(n_seg):
    A, B = seg_edges[k], seg_edges[k + 1]
    M[:, k] = kernel_seg(T0, R, A, B)

# 线性最小二乘 + 岭正则化
for ridge in [0.0, 1e-3, 1e-2, 1e-1, 1.0]:
    MtM = M.T @ M + ridge * np.eye(n_seg)
    MtL = M.T @ Lobs
    a = np.linalg.solve(MtM, MtL)
    Lpred = M @ a
    ss_res = np.sum((Lobs - Lpred) ** 2)
    ss_tot = np.sum((Lobs - Lobs.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print('\nridge=%.3f  R2=%.4f' % (ridge, r2))
    print('段  年份范围      a(t) 加速度值')
    for k in range(n_seg):
        print('  %d  %d-%d   %.6f' % (k, seg_edges[k], seg_edges[k+1], a[k]))

# 也输出解析预测拐点 vs 观测，看残差
ridge = 1e-1
a = np.linalg.solve(M.T @ M + ridge * np.eye(n_seg), M.T @ Lobs)
Lpred = M @ a
print('\n=== 残差最大的 10 个国家（ridge=0.1）===')
resid = Lobs - Lpred
idx = np.argsort(-np.abs(resid))[:10]
for i in idx:
    print('%s  T_start=%d  L观测=%.1f  L预测=%.1f  残差=%.1f' % (pairs[i][3], T0[i], Lobs[i], Lpred[i], resid[i]))
