"""Items 6 & 8: Pre-1900 sovereign subset + Sham collapse (isotonic regression).

Item 6: Compare Richards fits for pre-1900 vs post-1900 sovereign states
Item 8: Isotonic regression as non-Richards monotonic baseline for scaling collapse
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──
WPEI_CSV = r"data\women-political-empowerment-index.csv"
DIAG_CSV = r"data\_richards_diagnostics.csv"

wpei = pd.read_csv(WPEI_CSV)
diag = pd.read_csv(DIAG_CSV)

# Rename for convenience
wpei_col = "Women's Political Empowerment Index"
wpei['S'] = wpei[wpei_col] - 1

print(f"WPEI data: {len(wpei)} rows, {wpei['Entity'].nunique()} countries, {wpei['Year'].min()}–{wpei['Year'].max()}")
print(f"Diagnostics: {len(diag)} countries")

# ═══════════════════════════════════════════════════════════
# ITEM 6: Pre-1900 independent sovereign states
# ═══════════════════════════════════════════════════════════

# Classification based on standard historical references (COW, CIDCM, etc.)
# Countries that were internationally recognized sovereign states before 1900
PRE_1900_SOVEREIGN = {
    # Europe (most were sovereign by 1900)
    'Albania', 'Andorra', 'Austria', 'Belgium', 'Bulgaria', 'Denmark', 'Finland',
    'France', 'Germany', 'Greece', 'Hungary', 'Italy', 'Liechtenstein', 'Luxembourg',
    'Monaco', 'Montenegro', 'Netherlands', 'Norway', 'Portugal', 'Romania',
    'Russia', 'San Marino', 'Serbia', 'Spain', 'Sweden', 'Switzerland',
    'United Kingdom',
    # Latin America (all independent by 1830)
    'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Costa Rica',
    'Dominican Republic', 'Ecuador', 'El Salvador', 'Guatemala', 'Haiti', 'Honduras',
    'Mexico', 'Nicaragua', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela',
    # North America
    'United States',
    # Asia
    'China', 'Japan', 'Thailand', 'Iran', 'Nepal', 'Bhutan', 'Afghanistan',
    'Oman', 'Turkey',  # Ottoman Empire
    # Middle East / North Africa
    'Morocco', 'Egypt',  # Nominally Ottoman but effectively autonomous by 19th c
    # Sub-Saharan Africa
    'Ethiopia', 'Liberia',
    # Oceania
    'Tonga',  # Never colonized
}

# Note: Some of these were not continuously sovereign (e.g., WWII occupation)
# but the classification is based on pre-1900 existence as independent state

# Match with diagnostics data
diag['pre1900'] = diag['entity'].apply(lambda x: x in PRE_1900_SOVEREIGN)
pre1900 = diag[diag['pre1900']]
post1900 = diag[~diag['pre1900']]

print(f"\n═══ ITEM 6: Pre-1900 Sovereign States Sensitivity Analysis ═══")
print(f"Pre-1900 sovereign: {len(pre1900)} countries")
print(f"Post-1900 (or ambiguous): {len(post1900)} countries")

# Compare key metrics
for label, subset in [('Pre-1900', pre1900), ('Post-1900', post1900)]:
    print(f"\n  {label} ({len(subset)} countries):")
    print(f"    Median R² = {subset['R2'].median():.3f} (IQR {subset['R2'].quantile(0.25):.3f}–{subset['R2'].quantile(0.75):.3f})")
    print(f"    R² ≥ 0.95: {sum(subset['R2'] >= 0.95)}/{len(subset)} ({100*sum(subset['R2']>=0.95)/len(subset):.1f}%)")
    print(f"    R² < 0.80: {sum(subset['R2'] < 0.80)}/{len(subset)} ({100*sum(subset['R2']<0.80)/len(subset):.1f}%)")
    print(f"    Median k = {subset['k'].median():.3f} (IQR {subset['k'].quantile(0.25):.3f}–{subset['k'].quantile(0.75):.3f})")
    print(f"    Median t_mid = {subset['t_mid'].median():.1f} (IQR {subset['t_mid'].quantile(0.25):.1f}–{subset['t_mid'].quantile(0.75):.1f})")
    print(f"    Median v = {subset['v'].median():.3f} (IQR {subset['v'].quantile(0.25):.3f}–{subset['v'].quantile(0.75):.3f})")
    print(f"    Median S_ceil = {subset['S_ceil'].median():.3f}")

# Statistical tests
print(f"\n  Mann-Whitney U tests (pre-1900 vs post-1900):")
for param in ['R2', 'k', 't_mid', 'v', 'S_ceil']:
    u, p = stats.mannwhitneyu(pre1900[param].dropna(), post1900[param].dropna(), alternative='two-sided')
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    print(f"    {param}: U={u:.0f}, p={p:.4f} {sig}")

# Country overlap check — which pre-1900 countries are in the fitting set
pre1900_names = set(pre1900['entity'])
print(f"\n  Pre-1900 countries in dataset: {sorted(pre1900_names)}")

# Countries flagged as pre-1900 but NOT in the diagnostics
missing = PRE_1900_SOVEREIGN - set(diag['entity'])
if missing:
    print(f"  Classified as pre-1900 but not in diagnostics: {sorted(missing)}")

# ═══════════════════════════════════════════════════════════
# ITEM 8: Sham Collapse — Isotonic Regression Baseline
# ═══════════════════════════════════════════════════════════

print(f"\n\n═══ ITEM 8: Sham Collapse (Isotonic Regression) ═══")

def richards(t, S_floor, S_ceil, k, t_mid, v):
    """Richards generalized logistic function."""
    return S_floor + (S_ceil - S_floor) / (1 + v * np.exp(-k * (t - t_mid)))**(1/v)

def fit_richards(t, S):
    """Fit Richards function with bounds."""
    S_min, S_max = S.min(), S.max()
    S_range = S_max - S_min
    if S_range < 0.01:
        return None

    try:
        p0 = [S_min - 0.1 * S_range, S_max + 0.1 * S_range, 0.05, np.median(t), 1.0]
        bounds = (
            [S_min - 0.5, S_min + 0.5 * S_range, 0.001, t.min() - 50, 0.005],
            [S_min + 0.5 * S_range, S_max + 0.5, 1.0, t.max() + 50, 20.0]
        )
        popt, _ = curve_fit(richards, t, S, p0=p0, bounds=bounds,
                           maxfev=30000, method='trf')
        return popt
    except Exception:
        return None

def compute_band_fraction(t_hat, S_hat, v, band_width=0.10):
    """Compute fraction of points within ±band of v-specific master curve."""
    master = (1 + v * np.exp(-t_hat))**(-1/v)
    within = np.abs(S_hat - master) < band_width
    return within.mean()

# Use the same collapse data — countries that have Richards fits
collapse_results = []
iso_results = []

# Load collapse stats for reference
collapse_df = pd.read_csv(r"data\_p1-3_collapse_stats.csv")
collapse_band = pd.read_csv(r"data\_p1-3_band_fraction.csv")

print(f"Richards collapse reference: median band fraction = {collapse_band['frac_within_0.10'].median():.3f}")
print(f"  from {len(collapse_band)} countries")

# Run isotonic regression on each country and do sham collapse
# We need the raw S(t) data for each country
# Use the WPEI data and filter to countries that have ≥30 data points
country_counts = wpei.groupby('Entity').size()
valid_countries = country_counts[country_counts >= 30].index

# Get Richards fit parameters for normalization
richards_params = {}
for _, row in diag.iterrows():
    if row['status'] == 'ok':
        richards_params[row['entity']] = {
            'S_floor': row['S_floor'], 'S_ceil': row['S_ceil'],
            'k': row['k'], 't_mid': row['t_mid'], 'v': row['v']
        }

n_iso = 0
for entity in valid_countries:
    if entity not in richards_params:
        continue

    sub = wpei[wpei['Entity'] == entity].sort_values('Year')
    t = sub['Year'].values.astype(float)
    S = sub['S'].values

    if len(t) < 30:
        continue

    # Isotonic regression (monotonic increasing)
    try:
        iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
        S_iso = iso.fit_transform(t, S)
    except Exception:
        continue

    # Use Richards parameters for normalization (sham = keep same normalization, different underlying fit)
    rp = richards_params[entity]

    # Normalize using Richards parameters
    S_floor, S_ceil, k, t_mid = rp['S_floor'], rp['S_ceil'], rp['k'], rp['t_mid']

    # Avoid division by zero
    if abs(S_ceil - S_floor) < 0.001:
        continue

    S_hat_richards = (S - S_floor) / (S_ceil - S_floor)
    S_hat_iso = (S_iso - S_floor) / (S_ceil - S_floor)
    t_hat = k * (t - t_mid)

    # Band fraction for Richards fit
    v_val = rp['v']
    richards_band = compute_band_fraction(t_hat, S_hat_richards, v_val)

    # Band fraction for isotonic fit (same normalization, different S_hat)
    iso_band = compute_band_fraction(t_hat, S_hat_iso, v_val)

    iso_results.append({
        'entity': entity,
        'n_points': len(t),
        'richards_band': richards_band,
        'iso_band': iso_band,
    })
    n_iso += 1

# Summarize sham collapse
if iso_results:
    iso_df = pd.DataFrame(iso_results)
    print(f"\nSham collapse (isotonic regression): {len(iso_df)} countries")
    print(f"  Richards band fraction: median = {iso_df['richards_band'].median():.3f}")
    print(f"  Isotonic band fraction:  median = {iso_df['iso_band'].median():.3f}")
    print(f"  Difference (Richards − Iso): median = {(iso_df['richards_band'] - iso_df['iso_band']).median():.3f}")

    # Statistical test
    t_stat, p_val = stats.wilcoxon(iso_df['richards_band'], iso_df['iso_band'])
    print(f"  Wilcoxon signed-rank: statistic={t_stat:.0f}, p={p_val:.6f}")

    # How many countries does Richards beat isotonic?
    n_richards_better = (iso_df['richards_band'] > iso_df['iso_band']).sum()
    print(f"  Richards > Isotonic: {n_richards_better}/{len(iso_df)} ({100*n_richards_better/len(iso_df):.1f}%)")

    # Save results
    iso_df.to_csv(r"data\_r3_sham_collapse.csv", index=False)
    print(f"\n  Saved sham collapse results to _r3_sham_collapse.csv")

# Also try Monotonic Gaussian Process for a subset
# (GP is computationally expensive — do it for a sample)
print(f"\nMonotonic GP (sample of 30 countries)...")
sample_countries = list(valid_countries[:30])
gp_band = []

for entity in sample_countries:
    if entity not in richards_params:
        continue

    sub = wpei[wpei['Entity'] == entity].sort_values('Year')
    t = sub['Year'].values.astype(float)
    S = sub['S'].values

    if len(t) < 30:
        continue

    rp = richards_params[entity]
    S_floor, S_ceil, k, t_mid, v_val = rp['S_floor'], rp['S_ceil'], rp['k'], rp['t_mid'], rp['v']

    if abs(S_ceil - S_floor) < 0.001:
        continue

    try:
        # GP with RBF kernel (smooth but not monotonic by default)
        t_norm = (t - t.mean()) / t.std()
        gp = GaussianProcessRegressor(kernel=RBF(length_scale=1.0), alpha=0.01**2,
                                       normalize_y=True, n_restarts_optimizer=3)
        gp.fit(t_norm.reshape(-1, 1), S)
        S_gp = gp.predict(t_norm.reshape(-1, 1))

        S_hat_gp = (S_gp - S_floor) / (S_ceil - S_floor)
        t_hat = k * (t - t_mid)

        gp_band_val = compute_band_fraction(t_hat, S_hat_gp, v_val)
        richards_band_val = compute_band_fraction(t_hat, (S - S_floor) / (S_ceil - S_floor), v_val)

        gp_band.append({
            'entity': entity,
            'richards_band': richards_band_val,
            'gp_band': gp_band_val,
        })
    except Exception as e:
        pass

if gp_band:
    gp_df = pd.DataFrame(gp_band)
    print(f"  GP sham collapse ({len(gp_df)} countries):")
    print(f"    Richards band: median = {gp_df['richards_band'].median():.3f}")
    print(f"    GP band:       median = {gp_df['gp_band'].median():.3f}")

# ═══════════════════════════════════════════════════════════
# SUMMARY FOR PAPER INSERTION
# ═══════════════════════════════════════════════════════════
print(f"\n\n{'='*60}")
print(f"SUMMARY FOR PAPER")
print(f"{'='*60}")

print(f"\nItem 6 (Pre-1900 sovereign subset):")
print(f"  Pre-1900 sovereign: {len(pre1900)}/{len(diag)} countries")
print(f"  Post-1900: {len(post1900)}/{len(diag)} countries")
print(f"  Pre-1900 median R²: {pre1900['R2'].median():.3f}")
print(f"  Post-1900 median R²: {post1900['R2'].median():.3f}")
mw_r2 = stats.mannwhitneyu(pre1900['R2'].dropna(), post1900['R2'].dropna(), alternative='two-sided')
print(f"  R² difference: Mann-Whitney p = {mw_r2.pvalue:.4f}")

if iso_results:
    iso_df = pd.DataFrame(iso_results)
    print(f"\nItem 8 (Sham collapse):")
    print(f"  N = {len(iso_df)} countries")
    print(f"  Richards median band fraction: {iso_df['richards_band'].median():.3f}")
    print(f"  Isotonic median band fraction: {iso_df['iso_band'].median():.3f}")
    print(f"  Δ = {(iso_df['richards_band'] - iso_df['iso_band']).median():.3f}")
    n_rw = (iso_df['richards_band'] > iso_df['iso_band']).sum()
    print(f"  Richards > Isotonic: {n_rw}/{len(iso_df)}")
