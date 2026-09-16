"""
Item 12: Compute k_A/k_B distribution from piecewise-linear slopes.
Replace "one to two orders of magnitude" with actual median and IQR.

Approach:
- For each country, fit two-segment piecewise-linear regression around t_mid (Richards inflection)
- Phase A slope = linear rate of S(t) increase during ascent (pre-t_mid)
- Phase B slope = linear rate of S(t) increase during approach to ceiling (post-t_mid)
- k_A/k_B = ratio of ascent rate to approach rate
- Report distribution for countries where piecewise-linear is a reasonable model
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import csv
import numpy as np
from scipy import stats

# ── Load WPEI data ──
wpei_path = r'data\women-political-empowerment-index.csv'
years = None
country_data = {}  # code -> {years: [], values: []}

with open(wpei_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    # Header: "Entity", "Code", "Year", "WPEI" or similar
    # Let's figure out the columns
    entity_idx = header.index('Entity') if 'Entity' in header else 0
    code_idx = header.index('Code') if 'Code' in header else 1
    year_idx = header.index('Year') if 'Year' in header else 2
    wpei_col = 3  # usually the 4th column

    for row in reader:
        if len(row) < 4:
            continue
        code = row[code_idx].strip()
        try:
            year = float(row[year_idx])
            wpei = float(row[wpei_col])
        except (ValueError, IndexError):
            continue
        if code not in country_data:
            country_data[code] = {'years': [], 'wpei': []}
        country_data[code]['years'].append(year)
        country_data[code]['wpei'].append(wpei)

print(f"Loaded WPEI data for {len(country_data)} countries")

# ── Load Richards t_mid for each country ──
richards_path = r'data\_richards_diagnostics.csv'
tmid = {}  # code -> t_mid
with open(richards_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    # Find column indices
    code_idx = 0  # ﻿code
    tmid_idx = header.index('t_mid')
    for row in reader:
        code = row[code_idx].strip().lstrip('﻿')
        try:
            tmid[code] = float(row[tmid_idx])
        except (ValueError, IndexError):
            continue

print(f"Loaded t_mid for {len(tmid)} countries")

# ── Load AIC comparison to identify piecewise-linear winner countries ──
aic_path = r'data\_vdem_model_comparison.csv'
pw_winners = set()
with open(aic_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    code_idx = 0
    winner_idx = header.index('aic_winner')
    for row in reader:
        code = row[code_idx].strip().lstrip('﻿')
        if row[winner_idx].strip() == 'Piecewise':
            pw_winners.add(code)

print(f"Piecewise-linear AIC winners: {len(pw_winners)} countries")

# ── Compute piecewise-linear slopes for each country ──
results = []

for code in country_data:
    if code not in tmid:
        continue

    years_arr = np.array(country_data[code]['years'])
    wpei_arr = np.array(country_data[code]['wpei'])
    # S(t) = WPEI - 1
    s_arr = wpei_arr - 1.0
    t_mid = tmid[code]

    # Mask: Phase A (pre-t_mid), Phase B (post-t_mid)
    mask_a = years_arr <= t_mid
    mask_b = years_arr >= t_mid

    # Need at least 5 points in each phase
    if mask_a.sum() < 5 or mask_b.sum() < 5:
        continue

    # Phase A: linear fit S = a_A * t + b_A
    t_a = years_arr[mask_a]
    s_a = s_arr[mask_a]
    slope_a, intercept_a, r_a, p_a, se_a = stats.linregress(t_a, s_a)

    # Phase B: linear fit S = a_B * t + b_B
    t_b = years_arr[mask_b]
    s_b = s_arr[mask_b]
    slope_b, intercept_b, r_b, p_b, se_b = stats.linregress(t_b, s_b)

    # k_A = slope of Phase A (should be positive for ascent)
    # k_B = slope of Phase B (should be positive but smaller, approaching ceiling)
    k_A = slope_a
    k_B = slope_b

    # Only include if both slopes are positive (meaningful ascent and approach)
    if k_A <= 0 or k_B <= 0:
        continue

    ratio = k_A / k_B

    results.append({
        'code': code,
        'entity': code,  # will be filled later
        'n_a': mask_a.sum(),
        'n_b': mask_b.sum(),
        't_mid': t_mid,
        'k_A': k_A,
        'k_B': k_B,
        'ratio': ratio,
        'is_pw_winner': code in pw_winners,
        'r2_a': r_a**2,
        'r2_b': r_b**2,
    })

print(f"Countries with valid piecewise-linear slopes: {len(results)}")

# ── Analysis ──
ratios_all = np.array([r['ratio'] for r in results])
ratios_pw = np.array([r['ratio'] for r in results if r['is_pw_winner']])

print("\n=== ALL COUNTRIES (n={}) ===".format(len(ratios_all)))
print(f"  k_A/k_B median: {np.median(ratios_all):.1f}")
print(f"  k_A/k_B IQR: {np.percentile(ratios_all, 25):.1f} – {np.percentile(ratios_all, 75):.1f}")
print(f"  k_A/k_B range: {np.min(ratios_all):.1f} – {np.max(ratios_all):.1f}")
print(f"  k_A/k_B geometric mean: {stats.gmean(ratios_all):.1f}")
print(f"  k_A median: {np.median([r['k_A'] for r in results]):.6f}")
print(f"  k_B median: {np.median([r['k_B'] for r in results]):.6f}")

print(f"\n=== PIECEWISE-LINEAR AIC WINNERS (n={len(ratios_pw)}) ===")
print(f"  k_A/k_B median: {np.median(ratios_pw):.1f}")
print(f"  k_A/k_B IQR: {np.percentile(ratios_pw, 25):.1f} – {np.percentile(ratios_pw, 75):.1f}")
print(f"  k_A/k_B range: {np.min(ratios_pw):.1f} – {np.max(ratios_pw):.1f}")
print(f"  k_A/k_B geometric mean: {stats.gmean(ratios_pw):.1f}")
print(f"  k_A median: {np.median([r['k_A'] for r in results if r['is_pw_winner']]):.6f}")
print(f"  k_B median: {np.median([r['k_B'] for r in results if r['is_pw_winner']]):.6f}")

# ── Also compute from Richards k values ──
print("\n=== FROM RICHARDS k PARAMETER ===")
richards_k = []
for code, k_val in tmid.items():
    if code in country_data and code in [r['code'] for r in results]:
        richards_k.append(k_val)
richards_k = np.array(richards_k)
print(f"  Richards k: median={np.median(richards_k):.4f}, IQR={np.percentile(richards_k,25):.4f}–{np.percentile(richards_k,75):.4f}")

# ── Summary for paper ──
print("\n=== FOR PAPER (§3.2 or Methods) ===")
med = np.median(ratios_pw)
q25 = np.percentile(ratios_pw, 25)
q75 = np.percentile(ratios_pw, 75)
print(f"  median k_A/k_B ≈ {med:.0f}, IQR {q25:.0f}–{q75:.0f} (PW-winner countries, n={len(ratios_pw)})")
med_all = np.median(ratios_all)
q25_all = np.percentile(ratios_all, 25)
q75_all = np.percentile(ratios_all, 75)
print(f"  median k_A/k_B ≈ {med_all:.0f}, IQR {q25_all:.0f}–{q75_all:.0f} (all countries, n={len(ratios_all)})")

# ── Export full results ──
out_path = r'data\_ka_kb_ratios.csv'
with open(out_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['code', 't_mid', 'n_a', 'n_b', 'k_A', 'k_B', 'ratio', 'r2_a', 'r2_b', 'is_pw_winner'])
    for r in sorted(results, key=lambda x: x['ratio'], reverse=True):
        writer.writerow([
            r['code'], r['t_mid'], r['n_a'], r['n_b'],
            r['k_A'], r['k_B'], r['ratio'],
            r['r2_a'], r['r2_b'],
            1 if r['is_pw_winner'] else 0
        ])
print(f"\nFull results exported to: {out_path}")
