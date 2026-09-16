# -*- coding: utf-8 -*-
"""
全球数据：用新函数 a(t)=A*(e^{r(t-t0)}-1)（指数减常数，从 0 开始）
重算 157 国拐点处加速度的剧变区间 + 全球主流化年份
"""
import csv
import numpy as np

A = 5.937e-18
R = 0.0676
T0FIX = 1482.64
CHN_TMID = 1953.82
ADV = {'PRT', 'ESP', 'NLD', 'GBR', 'FRA'}
EX = {'ISR', 'CPV', 'MOZ', 'STP', 'USA', 'AUS', 'ETH', 'TLS', 'CUB', 'PRK'}

ns = {}
src = open(r'data\_tstart_acceleration.py', encoding='utf-8').read().split('def main()')[0]
exec(src, ns)
TS = ns['T_START']

rows = list(csv.DictReader(open(r'data\_vdem_fit_all.csv', encoding='utf-8-sig')))
tmid_list = []
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
    tmid_list.append(tmid)

tmid = np.array(tmid_list)
acc = A * (np.exp(R * (tmid - T0FIX)) - 1.0)

print('样本数', len(tmid))
print()
print('=== 全球拐点分布（数据本身，与新函数无关）===')
print('ric_tmid: 均值 %.1f  中位 %.1f  sd %.1f  范围 [%.1f, %.1f]' %
      (tmid.mean(), np.median(tmid), tmid.std(), tmid.min(), tmid.max()))
print()
print('=== 全球剧变区间（新函数 a(t)=%.3e*(e^{%.4f(t-%.2f)}-1)）===' % (A, R, T0FIX))
print('拐点处加速度 a(ric_tmid) 分布：')
print('  中位 %.3e  均值 %.3e  sd %.3e' % (np.median(acc), acc.mean(), acc.std()))
print('  范围 [%.3e, %.3e]' % (acc.min(), acc.max()))
print()
print('分位数（a 值 -> 对应年份 -> 从 0 涨到该值年数）:')
for p in [10, 25, 50, 75, 90]:
    av = np.percentile(acc, p)
    yr = T0FIX + np.log(1.0 + av / A) / R
    yrs = yr - T0FIX
    print('  p%2d  a=%.3e   年份 %.1f   从0涨到该值 %.1f 年' % (p, av, yr, yrs))

lo, hi = tmid.mean() - tmid.std(), tmid.mean() + tmid.std()
alo = A * (np.exp(R * (lo - T0FIX)) - 1)
ahi = A * (np.exp(R * (hi - T0FIX)) - 1)
print()
print('拐点集中区间 [%.1f, %.1f] (mean±1sd) 对应加速度 [%.3e, %.3e]' % (lo, hi, alo, ahi))

print()
print('=== 中国对照 ===')
a_chn = A * (np.exp(R * (CHN_TMID - T0FIX)) - 1)
yrs_chn = CHN_TMID - T0FIX
print('a(1953.82)=%.3e  从0涨到该值 %.1f 年' % (a_chn, yrs_chn))

print()
print('=== 全球主流化年份（对称性外推：T_end = 2*t_mid - t0）===')
for lab, tv in [('拐点均值', tmid.mean()), ('拐点中位', np.median(tmid))]:
    tend = 2 * tv - T0FIX
    print('  用%s %.1f -> 主流化 %.1f（第二段耗时 %.1f 年）' % (lab, tv, tend, tend - tv))
print('  （中国参考：t_mid=1953.82 -> 主流化 2425）')
