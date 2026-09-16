# -*- coding: utf-8 -*-
"""
Oscillation evidence v3 — leave-one-window blind extrapolation + decadal residual spectrum.

Key change from v2:
  - V2 used GLOBAL sigmoid fit (all years) → trend absorbed the events → overshoot vanished.
  - V3 fits sigmoid on PRE-PEAK data ONLY for each event, then extrapolates blindly forward.
    If the ratchet is a real restoring force, post-trough S should systematically exceed
    the blind extrapolation (which never saw the event). Surrogate data should not.

Exp 1: Blind-extrapolation excess — real vs surrogate per-event.
Exp 2: Decadal median residual spectrum (10-year bins, median per bin).
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, ttest_1samp, wilcoxon
from scipy.signal import lombscargle
import warnings
warnings.filterwarnings('ignore')

BASE = r'data'
SEED = 20260710
np.random.seed(SEED)
MIN_YEARS = 30
MIN_PRE_PEAK = 15   # need at least this many years before peak to fit sigmoid


def richards(t, S_floor, S_ceil, k, t_mid, v):
    z = v * np.exp(-k * (t - t_mid))
    z = np.clip(z, 1e-15, 1e15)
    return S_floor + (S_ceil - S_floor) / (1 + z) ** (1 / v)


def fit_richards(t, s):
    """Fit Richards to (t, s). Returns popt or None."""
    if len(t) < MIN_YEARS or s.max() - s.min() < 0.05:
        return None
    try:
        # Logistic seed
        p0_log = [s.min(), s.max(), 0.03, (t.min() + t.max()) / 2]
        bounds_log = ([s.min() - 0.5, -1.0, 0.0005, t.min() - 100],
                      [0.0, 0.0, 2.0, t.max() + 100])
        popt_log, _ = curve_fit(
            lambda t_, a, b, c, d: a + (b - a) / (1 + np.exp(-c * (t_ - d))),
            t, s, p0=p0_log, bounds=bounds_log, maxfev=30000)
        # Richards
        p0r = [popt_log[0], popt_log[1], popt_log[2], popt_log[3], 1.0]
        bounds_r = ([s.min() - 0.5, -1.0, 0.0005, t.min() - 100, 0.005],
                    [0.0, 0.0, 2.0, t.max() + 100, 20.0])
        poptr, _ = curve_fit(richards, t, s, p0=p0r, bounds=bounds_r, maxfev=30000)
        return poptr
    except:
        return None


def fit_richards_blind(t, s):
    """Fit Richards with relaxed bounds (for short pre-peak windows)."""
    if len(t) < MIN_PRE_PEAK:
        return None
    try:
        p0_log = [s.min(), max(s.max(), s.min() + 0.1), 0.03, (t.min() + t.max()) / 2]
        bounds_log = ([s.min() - 0.5, s.min() - 0.01, 0.0005, t.min() - 100],
                      [0.5, 1.0, 2.0, t.max() + 100])
        popt_log, _ = curve_fit(
            lambda t_, a, b, c, d: a + (b - a) / (1 + np.exp(-c * (t_ - d))),
            t, s, p0=p0_log, bounds=bounds_log, maxfev=30000)
        p0r = [popt_log[0], popt_log[1], popt_log[2], popt_log[3], 1.0]
        bounds_r = ([s.min() - 0.5, s.min() - 0.01, 0.0005, t.min() - 100, 0.005],
                    [0.5, 1.0, 2.0, t.max() + 100, 20.0])
        poptr, _ = curve_fit(richards, t, s, p0=p0r, bounds=bounds_r, maxfev=30000)
        return poptr
    except:
        return None


def detect_suppressions(years, wpei, min_drop=0.03, min_dur=2):
    events = []
    n = len(years)
    i = 0
    while i < n:
        while i < n - 1 and wpei[i + 1] >= wpei[i]:
            i += 1
        if i >= n - 1:
            break
        peak_idx = i
        peak_val = wpei[peak_idx]
        peak_year = years[peak_idx]
        j = i + 1
        trough_idx = j
        trough_val = wpei[j]
        while j < n - 1:
            if wpei[j] < trough_val:
                trough_val = wpei[j]
                trough_idx = j
            if wpei[j] > peak_val:
                break
            if j >= trough_idx + 3 and all(wpei[j - k] > wpei[j - k - 1] for k in range(3) if j - k - 1 >= trough_idx):
                break
            j += 1
        drop = peak_val - trough_val
        duration = years[trough_idx] - peak_year
        if drop >= min_drop and duration >= min_dur:
            post_trough = wpei[trough_idx:]
            post_years = years[trough_idx:]
            if len(post_trough) > 1:
                max_after = np.max(post_trough)
                max_after_idx = np.argmax(post_trough)
                max_after_year = post_years[max_after_idx]
                rebounded = max_after > peak_val
                overshoot = max_after - peak_val if rebounded else None
                ongoing = (years[trough_idx] >= years[-1] - 3)
                events.append({
                    'peak_year': int(peak_year), 'peak_val': float(peak_val),
                    'trough_year': int(years[trough_idx]), 'trough_val': float(trough_val),
                    'drop': float(drop), 'duration': int(duration),
                    'rebounded': bool(rebounded), 'overshoot': overshoot,
                    'max_after': float(max_after), 'max_after_year': int(max_after_year),
                    'ongoing': bool(ongoing),
                })
        i = trough_idx + 1
    return events


# ═══════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════
df = pd.read_csv(f'{BASE}/women-political-empowerment-index.csv')
df.columns = ['entity', 'code', 'year', 'wpei', 'region']
df['S'] = df['wpei'] - 1

ratchet = pd.read_csv(f'{BASE}/_vdem_ratchet_all.csv')
completed = ratchet[~ratchet['ongoing']].copy()
print(f"Completed events total: {len(completed)}")

# Build country series lookup
country_data = {}
for code, grp in df.groupby('code'):
    sub = grp.sort_values('year')
    country_data[code] = (sub['year'].values.astype(float), sub['S'].values)

# ═══════════════════════════════════════════════════
# EXP 1: LEAVE-ONE-WINDOW BLIND EXTRAPOLATION
# ═══════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  EXPERIMENT 1: BLIND EXTRAPOLATION (fit on pre-peak data only)")
print(f"{'='*60}")

# For each event, fit sigmoid on data BEFORE peak_year, extrapolate forward
blind_results = []  # per-event excess

for _, ev in completed.iterrows():
    code = ev['code']
    if code not in country_data:
        continue
    t_all, s_all = country_data[code]
    peak_yr = int(ev['peak_year'])
    trough_yr = int(ev['trough_year'])

    # Only use pre-peak data for fitting
    mask_pre = t_all < peak_yr
    if mask_pre.sum() < MIN_PRE_PEAK:
        continue
    t_pre = t_all[mask_pre]
    s_pre = s_all[mask_pre]

    popt_blind = fit_richards_blind(t_pre, s_pre)
    if popt_blind is None:
        continue

    # Blind extrapolation over the full observed range
    s_blind = richards(t_all, *popt_blind)

    # Post-trough excess: S_actual - S_blind, averaged over [trough+1, trough+25]
    mask_post = (t_all > trough_yr) & (t_all <= trough_yr + 25)
    if mask_post.sum() < 3:
        continue

    post_excess = s_all[mask_post] - s_blind[mask_post]
    mean_excess = float(np.mean(post_excess))

    # Also: max S after trough vs blind fit at that time
    idx_max = np.argmax(s_all[mask_post])
    t_max = t_all[mask_post][idx_max]
    s_max = s_all[mask_post][idx_max]
    s_blind_at_max = richards(np.array([t_max]), *popt_blind)[0]
    peak_excess = float(s_max - s_blind_at_max)

    # Pre-peak residual std (for later surrogate calibration)
    resid_pre = s_pre - richards(t_pre, *popt_blind)
    resid_std = float(np.std(resid_pre))

    blind_results.append({
        'code': code,
        'entity': ev['entity'],
        'peak_year': peak_yr,
        'trough_year': trough_yr,
        'drop': float(ev['drop']),
        'mean_excess': mean_excess,
        'peak_excess': peak_excess,
        'n_pre': int(mask_pre.sum()),
        'n_post': int(mask_post.sum()),
        'resid_std': resid_std,
        'rebounded': bool(ev['rebounded']),
        'overshoot_raw': ev['overshoot'],
    })

blind_df = pd.DataFrame(blind_results)
print(f"  Events with valid blind fit: {len(blind_df)}")

mean_excess = blind_df['mean_excess'].values
peak_excess = blind_df['peak_excess'].values

print(f"\n  Mean post-trough excess (S_actual − S_blind):")
print(f"    mean = {mean_excess.mean():.4f}, median = {np.median(mean_excess):.4f}")
print(f"    std = {mean_excess.std():.4f}, se = {mean_excess.std()/np.sqrt(len(mean_excess)):.4f}")

t_stat, p_val = ttest_1samp(mean_excess, 0)
print(f"    t-test vs 0: t = {t_stat:.3f}, p = {p_val:.6f} (one-tailed: {p_val/2:.6f})")
print(f"    > 0 in {(mean_excess > 0).mean()*100:.1f}% of events")

# Peak excess
print(f"\n  Peak post-trough excess (max S − blind_fit at max):")
print(f"    mean = {peak_excess.mean():.4f}, median = {np.median(peak_excess):.4f}")
t_stat_p, p_val_p = ttest_1samp(peak_excess, 0)
print(f"    t-test vs 0: t = {t_stat_p:.3f}, p = {p_val_p:.6f} (one-tailed: {p_val_p/2:.6f})")
print(f"    > 0 in {(peak_excess > 0).mean()*100:.1f}% of events")

# By drop severity
print(f"\n  By drop severity:")
for lo, hi, label in [(0.03, 0.05, 'Small (0.03-0.05)'),
                       (0.05, 0.10, 'Medium (0.05-0.10)'),
                       (0.10, 0.20, 'Large (0.10-0.20)'),
                       (0.20, 1.00, 'Severe (>0.20)')]:
    sub = blind_df[(blind_df['drop'] >= lo) & (blind_df['drop'] < hi)]
    if len(sub) > 0:
        print(f"    {label}: n={len(sub)}, mean_excess={sub['mean_excess'].mean():.4f}, "
              f">0 in {(sub['mean_excess'] > 0).mean()*100:.0f}%")

# ═══════════════════════════════════════════════════
# EXP 1b: SURROGATE — blind extrapolation on AR(1) surrogate events
# ═══════════════════════════════════════════════════
print(f"\n\n{'='*60}")
print(f"  EXPERIMENT 1b: SURROGATE BLIND EXTRAPOLATION")
print(f"{'='*60}")

# For each country, generate AR(1) surrogates, run detection, compute blind excess
# Use the blind-fit residuals from the FULL series (global sigmoid) for AR(1) params
# Then on surrogate, re-detect events, and for each detected event, do the same blind fit

# First, fit global sigmoid for each country to get benchmark residuals
print("  Fitting global sigmoids for AR(1) calibration...")
country_global = {}  # code → (t, s, s_fit, resid, phi, sigma)
for code, (t, s) in country_data.items():
    if len(t) < MIN_YEARS:
        continue
    popt = fit_richards(t, s)
    if popt is None:
        continue
    s_fit = richards(t, *popt)
    resid = s - s_fit
    if len(resid) < 3:
        continue
    ar1_r, ar1_l = resid[1:], resid[:-1]
    mask = ~(np.isnan(ar1_r) | np.isnan(ar1_l))
    if mask.sum() < 3:
        continue
    phi = np.corrcoef(ar1_l[mask], ar1_r[mask])[0, 1]
    sigma = np.std(resid)
    country_global[code] = (t, s, s_fit, resid, phi, sigma)

print(f"  Countries with global fit: {len(country_global)}")

# Now generate surrogates and compute blind excess
N_SURR_PER_COUNTRY = 15
surr_blind_excess = []
surr_event_count = []

for code, (t, s, s_fit, resid, phi, sigma) in country_global.items():
    for _ in range(N_SURR_PER_COUNTRY):
        # AR(1) noise
        noise = np.zeros(len(t))
        noise[0] = np.random.randn() * sigma
        for k in range(1, len(t)):
            noise[k] = phi * noise[k - 1] + np.random.randn() * sigma * np.sqrt(1 - phi**2)
        s_surr = s_fit + noise
        wpei_surr = s_surr + 1

        surr_events = detect_suppressions(t, wpei_surr)
        surr_completed = [e for e in surr_events if not e['ongoing']]
        surr_event_count.append(len(surr_completed))

        for se in surr_completed:
            peak_yr = int(se['peak_year'])
            trough_yr = int(se['trough_year'])

            # Blind fit on pre-peak surrogate data
            mask_pre = t < peak_yr
            if mask_pre.sum() < MIN_PRE_PEAK:
                continue
            t_pre = t[mask_pre]
            s_pre = s_surr[mask_pre]
            popt_blind = fit_richards_blind(t_pre, s_pre)
            if popt_blind is None:
                continue

            s_blind = richards(t, *popt_blind)
            mask_post = (t > trough_yr) & (t <= trough_yr + 25)
            if mask_post.sum() < 3:
                continue
            post_excess = s_surr[mask_post] - s_blind[mask_post]
            mean_ex = float(np.mean(post_excess))
            if not np.isnan(mean_ex):
                surr_blind_excess.append(mean_ex)

surr_excess = np.array(surr_blind_excess)
surr_event_count = np.array(surr_event_count)

print(f"  Surrogate events with valid blind fit: {len(surr_excess)}")
print(f"  Surrogate events per country: mean = {surr_event_count.mean():.1f}")
print(f"\n  Surrogate mean post-trough excess:")
print(f"    mean = {surr_excess.mean():.4f}, median = {np.median(surr_excess):.4f}")
print(f"    > 0 in {(surr_excess > 0).mean()*100:.1f}%")

# Mann-Whitney: real > surrogate?
stat_mw, p_mw = mannwhitneyu(mean_excess, surr_excess, alternative='greater')
print(f"\n  Mann-Whitney U (real excess > surrogate excess):")
print(f"    U = {stat_mw:.0f}, p = {p_mw:.6f}")
print(f"    Real mean excess  = {mean_excess.mean():.4f}")
print(f"    Surr mean excess  = {surr_excess.mean():.4f}")
print(f"    Δ (real − surr)   = {mean_excess.mean() - surr_excess.mean():.4f}")

# Cohen's d
pooled_std = np.sqrt((np.var(mean_excess) + np.var(surr_excess)) / 2)
cohens_d = (mean_excess.mean() - surr_excess.mean()) / pooled_std if pooled_std > 0 else 0
print(f"    Cohen's d          = {cohens_d:.3f}")

# ═══════════════════════════════════════════════════
# EXP 2: DECADAL MEDIAN RESIDUAL SPECTRUM
# ═══════════════════════════════════════════════════
print(f"\n\n{'='*60}")
print(f"  EXPERIMENT 2: DECADAL MEDIAN RESIDUAL SPECTRUM")
print(f"{'='*60}")

# For each country: fit global sigmoid, bin residuals by decade, take median
# Then Lomb-Scargle on the decadal series

decadal_power_real = []
decadal_power_surr = []

# Common frequency grid for decadal data (periods from ~20 to ~200 yr)
# With ~6 decades per country, Nyquist ≈ 0.5/(10 yr) = 0.05 yr⁻¹ → period_min ≈ 20 yr
freqs_decadal = np.linspace(0.005, 0.05, 150)  # periods 20-200 yr

for code, (t, s) in country_data.items():
    if len(t) < MIN_YEARS:
        continue
    # Global sigmoid
    popt = fit_richards(t, s)
    if popt is None:
        continue
    s_fit = richards(t, *popt)
    resid = s - s_fit

    # Bin by decade, take median
    decade_bins = {}
    for yr, r in zip(t, resid):
        dec = int(yr // 10) * 10
        decade_bins.setdefault(dec, []).append(r)

    dec_years = np.array(sorted(decade_bins.keys()))
    dec_resid = np.array([np.median(decade_bins[d]) for d in dec_years])

    if len(dec_years) < 4:  # need at least 4 decades for meaningful spectrum
        continue

    # Lomb-Scargle on decadal series
    if np.std(dec_resid) < 1e-10:
        continue
    pgram = lombscargle(dec_years.astype(float), dec_resid, freqs_decadal * 2 * np.pi, normalize='power')
    decadal_power_real.append(pgram)

    # One surrogate: AR(1) on decadal residuals
    if len(dec_resid) >= 4:
        ar1_r, ar1_l = dec_resid[1:], dec_resid[:-1]
        mask = ~(np.isnan(ar1_r) | np.isnan(ar1_l))
        if mask.sum() >= 3:
            phi_d = np.corrcoef(ar1_l[mask], ar1_r[mask])[0, 1]
            sigma_d = np.std(dec_resid)
            noise_d = np.zeros(len(dec_years))
            noise_d[0] = np.random.randn() * sigma_d
            for k in range(1, len(dec_years)):
                noise_d[k] = phi_d * noise_d[k - 1] + np.random.randn() * sigma_d * np.sqrt(1 - phi_d**2)
            if np.std(noise_d) >= 1e-10:
                pgram_s = lombscargle(dec_years.astype(float), noise_d, freqs_decadal * 2 * np.pi, normalize='power')
                decadal_power_surr.append(pgram_s)

decadal_real = np.array(decadal_power_real)
decadal_surr = np.array(decadal_power_surr)
mean_decadal_real = np.mean(decadal_real, axis=0)
mean_decadal_surr = np.mean(decadal_surr, axis=0)
periods_decadal = 1 / freqs_decadal

print(f"  Countries with decadal series: {len(decadal_real)}")
print(f"  Surrogate decadal series: {len(decadal_surr)}")

# Peak
idx_peak_d = np.argmax(mean_decadal_real)
print(f"\n  Peak period: {periods_decadal[idx_peak_d]:.1f} yr")
print(f"  Peak power:  {mean_decadal_real[idx_peak_d]:.4f} (surr: {mean_decadal_surr[idx_peak_d]:.4f})")

# Power ratio in low-frequency band
mask_low_d = freqs_decadal < 0.025  # periods > 40 yr
ratio_low = np.mean(mean_decadal_real[mask_low_d]) / np.mean(mean_decadal_surr[mask_low_d])
print(f"  Low-freq (>40yr) power ratio real/surr: {ratio_low:.2f}x")

# Top 3 peaks
top3_idx = np.argsort(mean_decadal_real)[-3:][::-1]
print(f"\n  Top 3 spectral peaks:")
for rank, idx in enumerate(top3_idx):
    print(f"    #{rank+1}: {periods_decadal[idx]:.1f} yr, power={mean_decadal_real[idx]:.4f} "
          f"(surr={mean_decadal_surr[idx]:.4f}, ratio={mean_decadal_real[idx]/mean_decadal_surr[idx]:.2f}x)")

# ═══════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════
# Blind extrapolation results
blind_df.to_csv(f'{BASE}/_blind_extrapolation.csv', index=False, encoding='utf-8-sig')
print(f"\nSaved: _blind_extrapolation.csv")

# Surrogate comparison summary
surr_summary = pd.DataFrame({
    'source': ['real'] * len(mean_excess) + ['surrogate'] * len(surr_excess),
    'mean_excess': np.concatenate([mean_excess, surr_excess]),
})
surr_summary.to_csv(f'{BASE}/_blind_excess_real_vs_surr.csv', index=False, encoding='utf-8-sig')
print(f"Saved: _blind_excess_real_vs_surr.csv")

# Decadal spectrum
spec_d = pd.DataFrame({
    'frequency': freqs_decadal,
    'period_years': periods_decadal,
    'power_real': mean_decadal_real,
    'power_surrogate': mean_decadal_surr,
})
spec_d.to_csv(f'{BASE}/_decadal_residual_spectrum.csv', index=False, encoding='utf-8-sig')
print(f"Saved: _decadal_residual_spectrum.csv")

# ═══════════════════════════════════════════════════
# PAPER-READY SUMMARY
# ═══════════════════════════════════════════════════
print(f"\n\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"\n  BLIND EXTRAPOLATION (leave-one-window):")
print(f"    {len(blind_df)} events with valid pre-peak fits")
print(f"    Mean post-trough excess: {mean_excess.mean():.4f} ± {mean_excess.std()/np.sqrt(len(mean_excess)):.4f}")
print(f"    t-test vs 0: t={t_stat:.2f}, p_one-tailed={p_val/2:.6f}")
print(f"    Excess > 0: {(mean_excess > 0).mean()*100:.1f}%")
print(f"    vs surrogate: MW U={stat_mw:.0f}, p={p_mw:.6f}, d={cohens_d:.3f}")

print(f"\n  DECADAL MEDIAN RESIDUAL SPECTRUM:")
print(f"    Peak: {periods_decadal[idx_peak_d]:.1f} yr")
print(f"    Low-freq power ratio: {ratio_low:.2f}x")

print(f"\nDone.")
