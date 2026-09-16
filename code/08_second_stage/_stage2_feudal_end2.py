# -*- coding: utf-8 -*-
"""
第二段起点 = 建立稳定资本主义/社会主义（没反复），谢 2026-09-07 纠正
关键：完全殖民的国家，殖民者带来资本主义 = 已经废除封建、建立资本主义，
和有没有独立没关系；殖民结束也继续资本主义（像香港印度），不回归封建。
所以完全殖民的国家「废除封建」= 殖民年（= T_START 殖民接触年），不是独立年。
半殖民地半封建（中国/土耳其等）= 封建还在，直到革命才废除（革命年）。
"""
import csv
import numpy as np

A = 5.937e-18
R = 0.0676
T0FIX = 1482.64
ADVANCED = {'PRT', 'ESP', 'NLD', 'GBR', 'FRA'}

ns = {}
src = open(r'data\_tstart_acceleration.py', encoding='utf-8').read().split('def main()')[0]
exec(src, ns)
TS = ns['T_START']

# 半殖民地半封建国家：废除封建 = 革命年（覆盖 T_START 的"被迫开国"年）
SEMI_FEUDAL = {
    # 东亚
    'CHN': 1949, 'JPN': 1868, 'KOR': 1945, 'PRK': 1948, 'MNG': 1921,
    # 东南亚
    'THA': 1932,
    # 南亚
    'NPL': 1951, 'AFG': 1919,
    # 中东
    'IRN': 1925, 'TUR': 1923, 'EGY': 1952, 'SYR': 1946, 'IRQ': 1932,
    'YEM': 1962, 'JOR': 1946, 'LBN': 1943,
    # 俄国 + 中亚（十月革命废除封建）
    'RUS': 1917, 'UKR': 1917, 'BLR': 1917,
    'KAZ': 1917, 'UZB': 1917, 'TKM': 1917, 'KGZ': 1917, 'TJK': 1917,
    'AZE': 1917, 'ARM': 1917, 'GEO': 1917,
    # 东欧（1848 革命失败的，废除封建更晚）
    'POL': 1918, 'HUN': 1867, 'CZE': 1918, 'SVK': 1918, 'AUT': 1867,
    'HRV': 1918, 'SVN': 1918, 'MDA': 1918, 'LVA': 1918, 'LTU': 1918, 'EST': 1918,
    # 巴尔干（独立年=废除奥斯曼封建）
    'SRB': 1878, 'BGR': 1878, 'ROU': 1877, 'GRC': 1821, 'ALB': 1912, 'MNE': 1878,
    'MKD': 1918, 'BIH': 1918,
    # 海湾君主国：君主制未废（特殊标记，先剔除）
    # 'SAU','KWT','QAT','BHR','ARE','OMN' 不在下面填，单独剔除
}

# 欧洲自身革命（先进国 + 西欧，自身资产阶级革命/宪政）
EUROPE_REVOL = {
    'GBR': 1688, 'FRA': 1789, 'NLD': 1795, 'ESP': 1812, 'PRT': 1820,
    'BEL': 1830, 'DEU': 1871, 'ITA': 1861, 'CHE': 1848, 'IRL': 1922,
    'DNK': 1849, 'NOR': 1814, 'SWE': 1809, 'FIN': 1917, 'ISL': 1944,
    'USA': 1776, 'CAN': 1867, 'AUS': 1901, 'NZL': 1907,  # 移民殖民地=自身建立资本主义
}

# 无封建概念 / 君主制未废除 / 还没稳定建立资本主义社会主义（战乱·奴隶制·独裁）（剔除）
EXCLUDE = {'SAU', 'KWT', 'QAT', 'BHR', 'ARE', 'OMN', 'ISR', 'CPV', 'STP',
           'AFG', 'YEM', 'SDN', 'SSD', 'SOM', 'SYR', 'LBY', 'COD', 'CAF', 'IRQ', 'HTI', 'MMR',
           'PRK', 'ERI', 'TKM', 'GNQ', 'SWZ', 'COM', 'GNB', 'NER', 'MLI', 'BFA', 'TCD', 'GIN', 'GMB'}

# 构建映射：默认完全殖民国家 = T_START（殖民接触年），再覆盖半封建和欧洲
FEUDAL_END = {}
for code, (yr, ev, reg) in TS.items():
    if code in ADVANCED or code in EXCLUDE:
        continue
    FEUDAL_END[code] = yr  # 完全殖民 = T_START 殖民接触年

FEUDAL_END.update(SEMI_FEUDAL)   # 半封建覆盖为革命年
FEUDAL_END.update(EUROPE_REVOL)  # 欧洲覆盖为自身革命年
for c in EXCLUDE:                 # 最后剔除未稳定/无封建国家（覆盖 update 后仍在的）
    FEUDAL_END.pop(c, None)


def acc_at(t):
    return A * (np.exp(R * (t - T0FIX)) - 1.0)


def years_from_zero(v):
    return np.log(1.0 + v / A) / R


def main():
    years_all = np.array(sorted(FEUDAL_END.values()))
    adv_years = sorted([FEUDAL_END[c] for c in ADVANCED if c in FEUDAL_END])
    print('=== 废除封建年份（新标准：完全殖民=殖民年，半封建=革命年）===')
    print('覆盖 %d 国' % len(years_all))
    print('全部：min=%d  median=%.1f  max=%d' % (years_all.min(), np.median(years_all), years_all.max()))
    print('先进国（西欧）废除封建：%s  中位=%.1f' % (adv_years, np.median(adv_years)))
    print()

    T1 = np.median(adv_years)
    T2 = np.median(years_all)
    V1 = acc_at(T1)
    V2 = acc_at(T2)
    tau1 = years_from_zero(V1)
    tau2 = years_from_zero(V2)
    print('T1(先进国中位)=%.1f  V1=%.3e  tau1=%.1f 年' % (T1, V1, tau1))
    print('T2(全部中位)=%.1f  V2=%.3e  tau2=%.1f 年' % (T2, V2, tau2))
    print()

    # 中国自洽
    print('=== 中国（1949）自洽校验 ===')
    eq_cn = 1949 + tau2
    print('中国 1949 + tau2=%.1f = %.1f 年，对照锚点 2425 差 %.1f 年' % (tau2, eq_cn, abs(2425 - eq_cn)))

    # 最早平等
    print()
    print('=== 最早平等 = T1 + tau ===')
    print('用第一次剧变值：%.1f + %.1f = %.1f 年' % (T1, tau1, T1 + tau1))
    print('用集中剧变值：%.1f + %.1f = %.1f 年' % (T1, tau2, T1 + tau2))


if __name__ == '__main__':
    main()
