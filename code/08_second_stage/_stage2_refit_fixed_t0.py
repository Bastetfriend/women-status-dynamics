# -*- coding: utf-8 -*-
"""
第二段：重拟合"从 0 开始"的 a(t)，t0 固定 = 1482.64（对称性外推自洽点）
三种形式（都满足 a(1482.64)=0）：
  1) 指数减常数  a(t)=A*(e^{r(t-t0)} - 1)
  2) 幂律        a(t)=A*(t-t0)^p
  3) 线性        a(t)=A*(t-t0)
反演方程：R - T_start + DELTA*I(R) - DELTA = 0
只拟合形状参数，比较 SSE / 模型拐点 sd / 拟合出的 a(1953.82) 与 a(1985)
基准：旧函数 a(t)=5e-8*e^{0.035(t-1700)} 在同数据的 SSE
"""
import csv
import numpy as np
from scipy.optimize import brentq, minimize

DELTA = 479.84
ADV = {'PRT', 'ESP', 'NLD', 'GBR', 'FRA'}
EX = {'ISR', 'CPV', 'MOZ', 'STP', 'USA', 'AUS', 'ETH', 'TLS', 'CUB', 'PRK'}
T0_FIX = 1482.64          # 对称性外推：2*1953.82 - 2425
CHN_TMID = 1953.82
CHN_ACC = 3.607e-4        # 旧函数 a(1953.82)

ns = {}
src = open(r'data\_tstart_acceleration.py', encoding='utf-8').read().split('def main()')[0]
exec(src, ns)
TS = ns['T_START']

rows = list(csv.DictReader(open(r'data\_vdem_fit_all.csv', encoding='utf-8-sig')))
pairs = []
for r in rows:
    c = r['code']
    if c.startswith('OWID_') or r.get('status') == 'flat' or c in ADV or c not in TS or c in EX:
        continue
    try:
        tmid = float(r['ric_tmid'])
    except Exception:
        continue
    ts = TS[c][0]
    if tmid - ts <= 0:
        continue
    pairs.append((ts, tmid, c))

T0 = np.array([p[0] for p in pairs], float)
Robs = np.array([p[1] for p in pairs], float)
n = len(pairs)
print('样本数', n, '  T_start 范围 [%.0f, %.0f]' % (T0.min(), T0.max()))
print('观测 ric_tmid 均值 %.1f 标准差 %.1f' % (Robs.mean(), Robs.std()))
print()

# ---------- 解析积分 I(R) ----------
def I_exp(A, r, T0v, Rv):
    U1, U2 = T0v - T0_FIX, Rv - T0_FIX
    if U1 < 0:
        U1 = 0.0
    e1, e2 = np.exp(r * U1), np.exp(r * U2)
    part = (Rv - T0_FIX) * (e2 - e1) / r - (U2 * e2 / r - e2 / (r * r)) + (U1 * e1 / r - e1 / (r * r))
    rect = (Rv - T0v) ** 2 / 2.0
    return A * (part - rect)

def I_pow(A, p, T0v, Rv):
    U1, U2 = T0v - T0_FIX, Rv - T0_FIX
    if U1 < 0:
        U1 = 0.0
    pu, pv = p + 1.0, p + 2.0
    part = (Rv - T0_FIX) * (U2 ** pu / pu) - U2 ** pv / pv - (Rv - T0_FIX) * (U1 ** pu / pu) + U1 ** pv / pv
    return A * part

def I_lin(A, T0v, Rv):
    U1, U2 = T0v - T0_FIX, Rv - T0_FIX
    if U1 < 0:
        U1 = 0.0
    part = (Rv - T0_FIX) * U2 ** 2 / 2.0 - U2 ** 3 / 3.0 - (Rv - T0_FIX) * U1 ** 2 / 2.0 + U1 ** 3 / 3.0
    return A * part

# 旧函数（基准）
def I_old(T0v, Rv):
    a0, r0 = 5e-8, 0.035
    return a0 / (r0 * r0) * (np.exp(r0 * (Rv - 1700)) - np.exp(r0 * (T0v - 1700)) * (r0 * (Rv - T0v) + 1))

def solve_R(I_func):
    out = np.zeros(n)
    for i in range(n):
        t0s = T0[i]
        lo = t0s + 0.001
        hi = t0s + DELTA + 1500
        f = lambda R: R - t0s + DELTA * I_func(t0s, R) - DELTA
        for _ in range(30):
            if f(hi) >= 0:
                break
            hi += 1000
        out[i] = brentq(f, lo, hi, xtol=1e-5)
    return out

def sse_of(Rpred):
    return float(np.sum((Rpred - Robs) ** 2))

def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)

# ---------- 基准：旧函数 ----------
R_old = solve_R(I_old)
print('=== 基准：旧函数 a=5e-8*e^{0.035(t-1700)}（从 5e-8 起，非 0）===')
print('  SSE=%.0f  R_pred 均值=%.1f sd=%.1f' % (sse_of(R_old), R_old.mean(), R_old.std()))
print()

# ---------- 拟合 ----------
def fit_exp(x):
    lgA, r = x
    A = 10.0 ** lgA
    return sse_of(solve_R(lambda t0s, Rv: I_exp(A, r, t0s, Rv)))

def fit_pow(x):
    lgA, lgp = x
    A, p = 10.0 ** lgA, np.exp(lgp)
    return sse_of(solve_R(lambda t0s, Rv: I_pow(A, p, t0s, Rv)))

def fit_lin(x):
    lgA = x[0]
    A = 10.0 ** lgA
    return sse_of(solve_R(lambda t0s, Rv: I_lin(A, t0s, Rv)))

results = []

# 指数减常数：拟合 (logA, r)
best = None
for st in [(-11.0, 0.035), (-12.0, 0.03), (-10.0, 0.04), (-11.5, 0.045)]:
    try:
        res = minimize(fit_exp, st, method='Nelder-Mead',
                       options={'maxiter': 4000, 'xatol': 1e-8, 'fatol': 1e-4})
        if best is None or res.fun < best[0]:
            best = (res.fun, res.x)
    except Exception as e:
        print('  起点 %s 失败 %s' % (st, repr(e)[:60]))
if best:
    sse, x = best
    A, r = 10.0 ** x[0], x[1]
    Rp = solve_R(lambda t0s, Rv: I_exp(A, r, t0s, Rv))
    results.append(('指数减常数', sse, Rp, 'A=%.3e r=%.4f' % (A, r), A, r, None))
    print('指数减常数  SSE=%.0f  R_pred sd=%.1f  A=%.3e r=%.4f' % (sse, Rp.std(), A, r))

# 幂律：拟合 (logA, logp)
best = None
for st in [(-10.0, np.log(1.5)), (-10.0, np.log(2.0)), (-11.0, np.log(1.0)), (-9.0, np.log(2.5))]:
    try:
        res = minimize(fit_pow, st, method='Nelder-Mead',
                       options={'maxiter': 4000, 'xatol': 1e-8, 'fatol': 1e-4})
        if best is None or res.fun < best[0]:
            best = (res.fun, res.x)
    except Exception as e:
        print('  起点 %s 失败 %s' % (st, repr(e)[:60]))
if best:
    sse, x = best
    A, p = 10.0 ** x[0], np.exp(x[1])
    Rp = solve_R(lambda t0s, Rv: I_pow(A, p, t0s, Rv))
    results.append(('幂律', sse, Rp, 'A=%.3e p=%.3f' % (A, p), A, None, p))
    print('幂律        SSE=%.0f  R_pred sd=%.1f  A=%.3e p=%.3f' % (sse, Rp.std(), A, p))

# 线性：拟合 logA
best = None
for st in [(-6.0,), (-6.5,), (-5.5,)]:
    try:
        res = minimize(fit_lin, st, method='Nelder-Mead',
                       options={'maxiter': 4000, 'xatol': 1e-8, 'fatol': 1e-4})
        if best is None or res.fun < best[0]:
            best = (res.fun, res.x)
    except Exception as e:
        print('  起点 %s 失败 %s' % (st, repr(e)[:60]))
if best:
    sse, x = best
    A = 10.0 ** x[0]
    Rp = solve_R(lambda t0s, Rv: I_lin(A, t0s, Rv))
    results.append(('线性', sse, Rp, 'A=%.3e' % A, A, None, None))
    print('线性        SSE=%.0f  R_pred sd=%.1f  A=%.3e' % (sse, Rp.std(), A))

print()
print('=== 拟合出的 a(t) 在关键年份的值（对比剧变值）===')
print('旧函数参考：a(1953.82)=%.3e  a(1985)=%.3e  中位剧变=1.13e-3' %
      (5e-8 * np.exp(0.035 * (CHN_TMID - 1700)), 5e-8 * np.exp(0.035 * (1985 - 1700))))
for name, sse, Rp, lab, A, r, p in results:
    if r is not None:
        a1953 = A * (np.exp(r * (CHN_TMID - T0_FIX)) - 1)
        a1985 = A * (np.exp(r * (1985 - T0_FIX)) - 1)
    elif p is not None:
        a1953 = A * (CHN_TMID - T0_FIX) ** p
        a1985 = A * (1985 - T0_FIX) ** p
    else:
        a1953 = A * (CHN_TMID - T0_FIX)
        a1985 = A * (1985 - T0_FIX)
    print('%s  %s' % (name, lab))
    print('    a(1953.82)=%.3e  a(1985)=%.3e' % (a1953, a1985))
