"""Item 7: Richards fitting on V-Dem three sub-dimensions.

Women's Political Empowerment Index (WPEI) = three sub-indices:
  1. Women's civil liberties (v2x_gencl)
  2. Women's civil society participation (v2x_gencs)
  3. Women's political participation (v2x_genpp)

For each, independently fit Richards and compare to the composite WPEI results.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

CSV_DIR = r"data"

SUB_DIMS = {
    'Civil Liberties': 'women-civil-liberties-index.csv',
    'Civil Society Participation': 'women-civil-society-participation-index.csv',
    'Political Participation': 'women-political-participation-index.csv',
}

WPEI_CSV = f"{CSV_DIR}\\women-political-empowerment-index.csv"

def richards(t, S_floor, S_ceil, k, t_mid, v):
    return S_floor + (S_ceil - S_floor) / (1 + v * np.exp(-k * (t - t_mid)))**(1/v)

def fit_richards(t, S):
    """Fit Richards with bounds, return (popt, R², status)."""
    S_min, S_max = np.nanmin(S), np.nanmax(S)
    S_range = S_max - S_min
    if S_range < 0.01:
        return None, None, 'range_too_small'

    try:
        p0 = [S_min - 0.1*S_range, S_max + 0.1*S_range, 0.05, np.nanmedian(t), 1.0]
        bounds = (
            [S_min - 0.5, S_min + 0.5*S_range, 0.001, np.nanmin(t) - 50, 0.005],
            [S_min + 0.5*S_range, S_max + 0.5, 1.0, np.nanmax(t) + 50, 20.0]
        )
        popt, _ = curve_fit(richards, t, S, p0=p0, bounds=bounds,
                           maxfev=30000, method='trf')
        S_fit = richards(t, *popt)
        ss_res = np.nansum((S - S_fit)**2)
        ss_tot = np.nansum((S - np.nanmean(S))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        return popt, r2, 'ok'
    except Exception:
        return None, None, 'fit_failed'

def find_val_col(df):
    for c in df.columns:
        if c not in ['Entity', 'Code', 'Year', 'World region according to OWID', 'Continent']:
            return c
    return None

# ── Load WPEI diagnostics as baseline ──
wpei_diag = pd.read_csv(r"data\_richards_diagnostics.csv")
wpei_ok = wpei_diag[wpei_diag['status'] == 'ok']
print(f"WPEI baseline: {len(wpei_ok)} countries with median R² = {wpei_ok['R2'].median():.3f}")

# ── Run for each sub-dimension ──
all_results = {}

for dim_label, fname in SUB_DIMS.items():
    print(f"\n{'='*60}")
    print(f"  {dim_label}")
    print(f"{'='*60}")

    df = pd.read_csv(f"{CSV_DIR}\\{fname}")
    val_col = find_val_col(df)
    print(f"  Indicator: {val_col}")
    print(f"  Rows: {len(df)}, Countries: {df['Entity'].nunique()}")

    # Shift to S = value - 1 if values are in [0,1] range (matching WPEI convention)
    vals = df[val_col].dropna()
    if vals.max() <= 1.0 and vals.min() >= 0:
        df['S'] = df[val_col] - 1.0
    else:
        # Already in a different range — normalize to [-1, 0] for consistency
        df['S'] = df[val_col] / df[val_col].max() - 1.0 if df[val_col].max() > 0 else df[val_col]

    results = []
    entities = sorted(df['Entity'].unique())
    n_total = 0
    n_ok = 0

    for entity in entities:
        sub = df[df['Entity'] == entity].sort_values('Year').dropna(subset=['S'])
        t = sub['Year'].values.astype(float)
        S = sub['S'].values

        if len(t) < 12:
            continue
        n_total += 1

        # Remove outliers — values beyond 5 IQR from median
        S_med = np.nanmedian(S)
        S_iqr = np.nanpercentile(S, 75) - np.nanpercentile(S, 25)
        mask = np.abs(S - S_med) < 5 * S_iqr
        if mask.sum() < 12:
            continue

        popt, r2, status = fit_richards(t[mask], S[mask])

        if popt is not None and r2 is not None:
            results.append({
                'entity': entity,
                'n_points': mask.sum(),
                'S_floor': popt[0], 'S_ceil': popt[1],
                'k': popt[2], 't_mid': popt[3], 'v': popt[4],
                'R2': r2, 'status': status,
            })
            if status == 'ok':
                n_ok += 1

    res_df = pd.DataFrame(results)
    ok_df = res_df[res_df['status'] == 'ok']

    print(f"  Fitted: {len(res_df)} countries")
    print(f"  OK (converged): {len(ok_df)} countries")
    if len(ok_df):
        print(f"  Median R² = {ok_df['R2'].median():.3f} (IQR {ok_df['R2'].quantile(0.25):.3f}–{ok_df['R2'].quantile(0.75):.3f})")
        print(f"  R² ≥ 0.95: {sum(ok_df['R2']>=0.95)}/{len(ok_df)} ({100*sum(ok_df['R2']>=0.95)/len(ok_df):.1f}%)")
        print(f"  R² ≥ 0.80: {sum(ok_df['R2']>=0.80)}/{len(ok_df)} ({100*sum(ok_df['R2']>=0.80)/len(ok_df):.1f}%)")
        print(f"  Median k = {ok_df['k'].median():.4f}")
        print(f"  Median t_mid = {ok_df['t_mid'].median():.1f}")
        print(f"  Median v = {ok_df['v'].median():.3f}")
        # v boundary issues
        v_lo = sum(ok_df['v'] <= 0.01)
        v_hi = sum(ok_df['v'] >= 19.9)
        print(f"  v at boundary: {v_lo} at v≈0, {v_hi} at v≈20 ({100*(v_lo+v_hi)/len(ok_df):.1f}%)")

    all_results[dim_label] = res_df
    res_df.to_csv(f"{CSV_DIR}\\..\\_r3_subdim_{dim_label.replace(' ','_').lower()}.csv", index=False)

# ── Cross-dimension comparison ──
print(f"\n\n{'='*60}")
print(f"  CROSS-DIMENSION COMPARISON")
print(f"{'='*60}")
print(f"  {'Dimension':<35} {'N_ok':>6} {'Median R²':>10} {'R²≥0.80':>8} {'v@boundary%':>12}")
print(f"  {'─'*35} {'─'*6} {'─'*10} {'─'*8} {'─'*12}")
print(f"  {'WPEI (composite)':<35} {len(wpei_ok):>6} {wpei_ok['R2'].median():>10.3f} {f'{sum(wpei_ok.R2>=0.80)}/{len(wpei_ok)}':>8} {'—':>12}")

for dim_label, res_df in all_results.items():
    ok = res_df[res_df['status'] == 'ok']
    if len(ok):
        v_boundary = 100*(sum(ok['v']<=0.01) + sum(ok['v']>=19.9))/len(ok)
        print(f"  {dim_label:<35} {len(ok):>6} {ok['R2'].median():>10.3f} {f'{sum(ok.R2>=0.80)}/{len(ok)}':>8} {v_boundary:>11.1f}%")

# ── Country-level: do all three sub-dimensions converge for the same countries? ──
print(f"\n\n{'='*60}")
print(f"  CONCORDANCE: Countries where all 3 sub-dims achieve R²≥0.80")
print(f"{'='*60}")

r2_threshold = 0.80
country_r2 = {}
for dim_label, res_df in all_results.items():
    ok = res_df[res_df['status'] == 'ok']
    for _, row in ok.iterrows():
        if row['entity'] not in country_r2:
            country_r2[row['entity']] = {}
        country_r2[row['entity']][dim_label] = row['R2']

n_all3 = 0
n_any = 0
for entity, dims in country_r2.items():
    if len(dims) == 3:
        if all(v >= r2_threshold for v in dims.values()):
            n_all3 += 1
        if any(v >= r2_threshold for v in dims.values()):
            n_any += 1

n_with_3 = sum(1 for d in country_r2.values() if len(d) == 3)
print(f"  Countries with all 3 sub-dimensions fitted: {n_with_3}")
print(f"  R²≥0.80 in all 3: {n_all3}/{n_with_3} ({100*n_all3/n_with_3:.1f}%)")
print(f"  R²≥0.80 in at least 1: {n_any}/{n_with_3} ({100*n_any/n_with_3:.1f}%)")

# Countries where WPEI fits well but sub-dimensions don't
wpei_good = set(wpei_ok[wpei_ok['R2'] >= 0.80]['entity'])
subdim_good_all3 = {e for e, d in country_r2.items() if len(d)==3 and all(v>=0.80 for v in d.values())}
wpei_ok_but_subdim_fail = wpei_good - subdim_good_all3
print(f"\n  WPEI R²≥0.80 but NOT all 3 sub-dims R²≥0.80: {len(wpei_ok_but_subdim_fail)} countries")
if len(wpei_ok_but_subdim_fail) <= 20 and len(wpei_ok_but_subdim_fail) > 0:
    print(f"  Examples: {sorted(list(wpei_ok_but_subdim_fail))[:15]}")

print(f"\nDone. Results saved to _r3_subdim_*.csv")
