# -*- coding: utf-8 -*-
"""
提前量 lead vs T_start 的多种函数拟合
反推加速度 a(t) 的形状
"""
import csv
import numpy as np
from scipy.optimize import curve_fit

DELTA = 479.84
ADVANCED = {'PRT', 'ESP', 'NLD', 'GBR', 'FRA'}

# 复用 T_start 映射
import importlib.util
spec = importlib.util.spec_from_file_location("ts", r"data\_tstart_acceleration.py")
# 直接 import 会跑 main，改用 exec 拿 T_START
ns = {}
src = open(r"data\_tstart_acceleration.py", encoding='utf-8').read()
src = src.split("def main()")[0]  # 只拿 T_START 定义部分
exec(src, ns)
T_START = ns['T_START']

rows = list(csv.DictReader(open(r'data\_vdem_fit_all.csv', encoding='utf-8-sig')))

data = []  # (T_start, lead, code, v)
for r in rows:
    code = r['code']
    if code.startswith('OWID_') or r['status'] == 'flat' or code in ADVANCED:
        continue
    if code not in T_START:
        continue
    try:
        tmid = float(r['ric_tmid']); v = float(r['ric_v'])
    except:
        continue
    ts = T_START[code][0]
    lead = ts + DELTA - tmid
    data.append((ts, lead, code, v))

data.sort()
ts = np.array([d[0] for d in data], float)
lead = np.array([d[1] for d in data], float)

# 离群值：以色列(1948建国) 提前量493，葡殖民三小国负提前量
print('原始数据点数', len(data))
print('lead 范围 [%.1f, %.1f]' % (lead.min(), lead.max()))
print()

# 定义拟合函数
def f_linear(T, a, b):
    return a * T + b

def f_quad(T, a, b, c):
    return a * T**2 + b * T + c

def f_exp(T, A, r, T0):
    return A * np.exp(r * (T - T0))

def f_logistic(T, L, k, T0):
    return L / (1.0 + np.exp(-k * (T - T0)))

def f_power(T, a, b, T0):
    return a * (T - T0) ** b

def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot

fits = [
    ('线性 lead=a*T+b', f_linear, [0.3, -200]),
    ('二次 lead=aT^2+bT+c', f_quad, [0.0005, -1.5, 1200]),
    ('指数 lead=A*exp(r*(T-T0))', f_exp, [50, 0.008, 1400]),
    ('逻辑斯蒂S型 lead=L/(1+exp(-k(T-T0)))', f_logistic, [400, 0.01, 1800]),
    ('幂律 lead=a*(T-T0)^b', f_power, [0.0001, 2.5, 1400]),
]

print('=== 全数据拟合（%d 点）===' % len(ts))
for name, fn, p0 in fits:
    try:
        popt, _ = curve_fit(fn, ts, lead, p0=p0, maxfev=20000)
        yhat = fn(ts, *popt)
        print('%-40s R2=%.4f  参数=%s' % (name, r2(lead, yhat), np.round(popt, 5)))
    except Exception as e:
        print('%-40s 失败 %s' % (name, repr(e)[:60]))

print()
# 剔除离群值后拟合
# 离群：以色列(ISR) 1948、葡殖民三小国(CPV/MOZ/STP 负提前量)、美国/澳大利亚(欧洲移民无封建)
EXCLUDE = {'ISR', 'CPV', 'MOZ', 'STP', 'USA', 'AUS', 'ETH', 'TLS', 'CUB', 'PRK'}
mask = np.array([d[2] not in EXCLUDE for d in data])
ts2, lead2 = ts[mask], lead[mask]
print('=== 剔除离群后（%d 点，去掉 ISR/葡殖民三小国/美澳/埃塞/东帝汶/古巴/朝鲜）===' % ts2.size)
for name, fn, p0 in fits:
    try:
        popt, _ = curve_fit(fn, ts2, lead2, p0=p0, maxfev=20000)
        yhat = fn(ts2, *popt)
        print('%-40s R2=%.4f  参数=%s' % (name, r2(lead2, yhat), np.round(popt, 5)))
    except Exception as e:
        print('%-40s 失败 %s' % (name, repr(e)[:60]))

print()
# 分桶均值（每50年一桶）看形状
print('=== 分桶均值（看 lead 随 T_start 的形状）===')
bins = {}
for t, l in zip(ts, lead):
    b = int(t // 50) * 50
    bins.setdefault(b, []).append(l)
for b in sorted(bins):
    arr = np.array(bins[b])
    print('%d-%d: n=%2d  lead均值=%.1f  范围[%.1f,%.1f]' % (b, b+49, len(arr), arr.mean(), arr.min(), arr.max()))
