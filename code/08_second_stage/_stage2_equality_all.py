# -*- coding: utf-8 -*-
"""
两个剧变值分别代入，算全部 172 国的平等年份（2026-09-07 谢确认方案）
剧变值 1（第一次出现新体系）：T1=1795, V1=a(1795), tau1=从0涨到V1
剧变值 2（集中出现新体系）：T2=1953, V2=a(1953), tau2=从0涨到V2
每个国家平等 = 该国废除封建时间 + tau（tau1 或 tau2）
"""
import csv
import numpy as np
from _stage2_feudal_end2 import FEUDAL_END, A, R, T0FIX, acc_at, years_from_zero, ADVANCED

T1 = 1795.0   # 先进国废除封建中位
T2 = 1878.0   # 全部国家废除封建中位（新标准：完全殖民=殖民年，半封建=革命年，排除未稳定国家）

V1 = acc_at(T1)
V2 = acc_at(T2)
tau1 = years_from_zero(V1)
tau2 = years_from_zero(V2)

print('=== 两个剧变值（方案确认版） ===')
print('剧变值1 第一次出现 T1=%d  V1=a(%.0f)=%.3e  tau1=%.1f 年' % (T1, T1, V1, tau1))
print('剧变值2 集中出现 T2=%d  V2=a(%.0f)=%.3e  tau2=%.1f 年' % (T2, T2, V2, tau2))
print()

# 每个国家两个平等年份
rows = []
for code, yr in FEUDAL_END.items():
    eq1 = yr + tau1
    eq2 = yr + tau2
    is_adv = code in ADVANCED
    rows.append((code, yr, is_adv, eq1, eq2))

rows.sort(key=lambda x: x[1])  # 按废除封建时间排序

# 分布统计
eq1_arr = np.array([r[3] for r in rows])
eq2_arr = np.array([r[4] for r in rows])
yr_arr = np.array([r[1] for r in rows])

print('=== 平等年份分布（%d 国） ===' % len(rows))
print('%-30s %10s %10s' % ('', '用剧变值1(tau1)', '用剧变值2(tau2)'))
print('%-30s %10.1f %10.1f' % ('最早平等', eq1_arr.min(), eq2_arr.min()))
print('%-30s %10.1f %10.1f' % ('中位平等', np.median(eq1_arr), np.median(eq2_arr)))
print('%-30s %10.1f %10.1f' % ('最晚平等', eq1_arr.max(), eq2_arr.max()))
print()

# 最早 15 国（按废除封建时间最早）
print('=== 最早废除封建的 15 国（+ 各自两个平等年份） ===')
print('%-6s %10s %10s %10s' % ('code', '废除封建', '平等(tau1)', '平等(tau2)'))
for code, yr, is_adv, eq1, eq2 in rows[:15]:
    tag = ' [先进]' if is_adv else ''
    print('%-6s %10d %10.1f %10.1f%s' % (code, yr, eq1, eq2, tag))

print()
print('=== 最晚废除封建的 10 国 ===')
for code, yr, is_adv, eq1, eq2 in rows[-10:]:
    tag = ' [先进]' if is_adv else ''
    print('%-6s %10d %10.1f %10.1f%s' % (code, yr, eq1, eq2, tag))

# 写完整表到 CSV
out = r'data\_stage2_equality_all.csv'
with open(out, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['code', 'feudal_end', 'is_advanced', 'equality_tau1', 'equality_tau2'])
    for code, yr, is_adv, eq1, eq2 in rows:
        w.writerow([code, yr, int(is_adv), round(eq1, 1), round(eq2, 1)])
print()
print('完整表已写：', out)
