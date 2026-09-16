"""
Compute Chile Pinochet case study: WPEI decline during dictatorship
and recovery after democratic restoration (1990).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# ── Load WPEI ──
wpei = pd.read_csv(r'实验输出_CSV\women-political-empowerment-index.csv')
chile = wpei[wpei['Code'] == 'CHL'].copy()
chile = chile.sort_values('Year')
wpei_col = "Women's Political Empowerment Index"
chile['S'] = chile[wpei_col] - 1.0  # S(t) = WPEI - 1

print("=== Chile WPEI data ===")
print(f"Years: {chile['Year'].min()}–{chile['Year'].max()}, {len(chile)} obs")
print()

# ── Pinochet period: 1973–1990 ──
# Pre-coup: use data up to 1972 to fit a sigmoid
# Then see how far WPEI fell during dictatorship
# Then track recovery after 1990

pre_coup = chile[chile['Year'] <= 1972]
dictatorship = chile[(chile['Year'] >= 1973) & (chile['Year'] <= 1990)]
post_dict = chile[chile['Year'] >= 1990]

print(f"Pre-coup (≤1972): {len(pre_coup)} obs, S range: {pre_coup['S'].min():.4f} to {pre_coup['S'].max():.4f}")
print(f"Dictatorship (1973–1990): {len(dictatorship)} obs")
print(f"Post-dictatorship (≥1990): {len(post_dict)} obs")
print()

# ── Find the key data points ──
# Peak before coup
peak_pre = pre_coup[pre_coup['S'] == pre_coup['S'].max()]
print(f"Pre-coup peak: {peak_pre['Year'].values[0]}, S = {peak_pre['S'].values[0]:.4f} (WPEI = {peak_pre[wpei_col].values[0]:.4f})")

# Trough during or after dictatorship
trough = dictatorship[dictatorship['S'] == dictatorship['S'].min()]
print(f"Dictatorship trough: {trough['Year'].values[0]}, S = {trough['S'].values[0]:.4f} (WPEI = {trough[wpei_col].values[0]:.4f})")
delta = peak_pre['S'].values[0] - trough['S'].values[0]
print(f"Δ (peak → trough): {delta:.4f} WPEI units")
print()

# ── Full Chile sigmoid fit for comparison ──
def richards(t, S_floor, S_ceil, k, t_mid, v):
    return S_floor + (S_ceil - S_floor) / (1 + v * np.exp(-k * (t - t_mid)))**(1/v)

# Fit to all Chile data
t_data = chile['Year'].values
S_data = chile['S'].values
try:
    popt, _ = curve_fit(richards, t_data, S_data,
                        p0=[-0.5, 0.5, 0.05, 1950, 1.0],
                        maxfev=10000, bounds=([-2, -1, 0.001, 1800, 0.1], [1, 2, 0.5, 2020, 10]))
    S_fit = richards(t_data, *popt)
    print(f"Chile Richards fit: S_floor={popt[0]:.4f}, S_ceil={popt[1]:.4f}, k={popt[2]:.4f}, t_mid={popt[3]:.1f}, v={popt[4]:.2f}")
    r2 = 1 - np.sum((S_data - S_fit)**2) / np.sum((S_data - S_data.mean())**2)
    print(f"R² = {r2:.4f}")
    print()

    # ── Deviation during dictatorship ──
    dict_years = dictatorship['Year'].values
    dict_S = dictatorship['S'].values
    dict_fit = richards(dict_years, *popt)
    dict_deviation = dict_S - dict_fit
    max_dev = np.min(dict_deviation)  # most negative = biggest suppression
    max_dev_year = dict_years[np.argmin(dict_deviation)]
    print(f"Max negative deviation from sigmoid during dictatorship: {max_dev:.4f} at {max_dev_year}")
    print(f"Mean deviation during dictatorship: {np.mean(dict_deviation):.4f}")
    print()

    # ── Recovery after 1990 ──
    # How many years after 1990 until S(t) is within, say, 0.02 of the sigmoid?
    post_years = post_dict['Year'].values
    post_S = post_dict['S'].values
    post_fit = richards(post_years, *popt)
    post_dev = np.abs(post_S - post_fit)

    threshold = 0.02
    recovered_idx = np.where(post_dev < threshold)[0]
    if len(recovered_idx) > 0:
        recovery_year = post_years[recovered_idx[0]]
        recovery_time = recovery_year - 1990
        print(f"Recovery: within {threshold:.2f} of sigmoid by {recovery_year} ({recovery_time} years after democratic restoration)")
    else:
        # Find minimum deviation year
        min_dev_idx = np.argmin(post_dev)
        print(f"Minimum post-1990 deviation: {post_dev[min_dev_idx]:.4f} at {post_years[min_dev_idx]} (not yet within {threshold})")

    # ── Key numbers for the paper ──
    print()
    print("=== NUMBERS FOR PAPER ===")
    print(f"Peak year: {peak_pre['Year'].values[0]}, S = {peak_pre['S'].values[0]:.3f}")
    print(f"Trough year: {trough['Year'].values[0]}, S = {trough['S'].values[0]:.3f}")
    print(f"Decline (Δ): {delta:.3f} ({(delta/0.024):.1f}σ of WPEI annual change)")
    print(f"Recovery to within {threshold:.2f} of sigmoid: {recovery_year} ({recovery_time} years post-1990)")
    print(f"Max deviation from sigmoid during dictatorship: {max_dev:.3f}")

    # ── Annual data for context ──
    print()
    print("=== Key years ===")
    key_years = [1970, 1972, 1973, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
    for y in key_years:
        row = chile[chile['Year'] == y]
        if len(row) > 0:
            S_val = row['S'].values[0]
            fit_val = richards(y, *popt)
            print(f"  {y}: S = {S_val:.4f}, fit = {fit_val:.4f}, dev = {S_val-fit_val:.4f}")

except Exception as e:
    print(f"Fit failed: {e}")
