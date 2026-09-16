# -*- coding: utf-8 -*-
"""
Richards parameter diagnostics for 191 countries.
Captures pcov from curve_fit, computes:
- Parameter standard errors and 95% CI
- Residual Durbin-Watson statistic
- RMSE
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import shapiro
import warnings
warnings.filterwarnings('ignore')

BASE = r'data'

df = pd.read_csv(f'{BASE}/women-political-empowerment-index.csv')
df.columns = ['entity', 'code', 'year', 'wpei', 'region']
df['S'] = df['wpei'] - 1

def richards(t, S_floor, S_ceil, k, t_mid, v):
    z = v * np.exp(-k * (t - t_mid))
    z = np.clip(z, 1e-15, 1e15)
    return S_floor + (S_ceil - S_floor) / (1 + z) ** (1 / v)

def logistic(t, S_floor, S_ceil, k, t_mid):
    return S_floor + (S_ceil - S_floor) / (1 + np.exp(-k * (t - t_mid)))

def durbin_watson(residuals):
    """Durbin-Watson statistic: 2.0 = no autocorrelation."""
    diff = np.diff(residuals)
    return np.sum(diff**2) / np.sum(residuals**2)

PARAM_NAMES = ['S_floor', 'S_ceil', 'k', 't_mid', 'v']
MIN_YEARS = 30

all_codes = df.groupby('code').agg(
    entity=('entity', 'first'),
    n=('year', 'count')
).reset_index()
all_codes = all_codes[all_codes['n'] >= MIN_YEARS].sort_values('code')

print(f"Fitting {len(all_codes)} countries with parameter diagnostics...")

rows = []
for idx, info in all_codes.iterrows():
    code = info['code']
    entity = info['entity']
    sub = df[df['code'] == code].sort_values('year')
    t = sub['year'].values.astype(float)
    s = sub['S'].values

    row = {'code': code, 'entity': entity, 'n': len(t)}

    if s.max() - s.min() < 0.05:
        row['status'] = 'flat'
        rows.append(row)
        continue

    # Logistic first (for initial parameters)
    try:
        p0 = [s.min(), s.max(), 0.03, (t.min() + t.max()) / 2]
        bounds = ([s.min() - 0.5, -1.0, 0.0005, t.min() - 100],
                  [0.0, 0.0, 2.0, t.max() + 100])
        popt_log, _ = curve_fit(logistic, t, s, p0=p0, bounds=bounds, maxfev=30000)
    except:
        row['status'] = 'log_fail'
        rows.append(row)
        continue

    # Richards with pcov
    try:
        p0r = [popt_log[0], popt_log[1], popt_log[2], popt_log[3], 1.0]
        bounds_r = ([s.min() - 0.5, -1.0, 0.0005, t.min() - 100, 0.005],
                    [0.0, 0.0, 2.0, t.max() + 100, 20.0])
        poptr, pcov = curve_fit(richards, t, s, p0=p0r, bounds=bounds_r, maxfev=30000)

        pred = richards(t, *poptr)
        resid = s - pred
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((s - s.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(ss_res / len(t))

        # Parameter SEs from pcov
        if np.any(np.isinf(pcov)):
            se = np.full(5, np.nan)
            pcov_valid = False
        else:
            se = np.sqrt(np.diag(pcov))
            pcov_valid = True

        # Store parameters and SEs
        for j, name in enumerate(PARAM_NAMES):
            row[name] = poptr[j]
            row[f'{name}_SE'] = se[j]
            if pcov_valid and se[j] > 0:
                row[f'{name}_CI_lo'] = poptr[j] - 1.96 * se[j]
                row[f'{name}_CI_hi'] = poptr[j] + 1.96 * se[j]
                row[f'{name}_CV'] = abs(se[j] / poptr[j]) if abs(poptr[j]) > 1e-10 else np.nan
            else:
                row[f'{name}_CI_lo'] = np.nan
                row[f'{name}_CI_hi'] = np.nan
                row[f'{name}_CV'] = np.nan

        # Residual diagnostics
        row['R2'] = r2
        row['RMSE'] = rmse
        row['DW'] = durbin_watson(resid)
        row['pcov_valid'] = pcov_valid
        row['status'] = 'ok'

        # Parameter correlations (off-diagonal)
        if pcov_valid:
            corr = pcov / np.outer(se, se)
            # Store max absolute off-diagonal correlation
            np.fill_diagonal(corr, 0)
            row['max_param_corr'] = np.max(np.abs(corr))

    except Exception as e:
        row['status'] = 'ric_fail'

    rows.append(row)
    if len(rows) % 50 == 0:
        print(f"  {len(rows)} done...")

results = pd.DataFrame(rows)
ok = results[results['status'] == 'ok']

print(f"\n{'='*70}")
print(f"  RESULTS: {len(ok)} OK / {len(results)} total")
print(f"{'='*70}")

# ═══ Summary statistics ═══
print(f"\n  === R² distribution ===")
for thresh in [0.99, 0.95, 0.90, 0.80]:
    n = (ok['R2'] >= thresh).sum()
    print(f"    R² >= {thresh}: {n} ({n/len(ok)*100:.1f}%)")

print(f"\n  === RMSE ===")
print(f"    Median: {ok['RMSE'].median():.4f}")
print(f"    Mean:   {ok['RMSE'].mean():.4f}")
print(f"    Max:    {ok['RMSE'].max():.4f}")

print(f"\n  === Durbin-Watson (2.0 = no autocorrelation) ===")
print(f"    Median: {ok['DW'].median():.3f}")
print(f"    Mean:   {ok['DW'].mean():.3f}")
print(f"    [0.5-1.5] (positive autocorr): {((ok['DW'] >= 0.5) & (ok['DW'] <= 1.5)).sum()}")
print(f"    [1.5-2.5] (acceptable):        {((ok['DW'] >= 1.5) & (ok['DW'] <= 2.5)).sum()}")
print(f"    <0.5 (strong autocorr):         {(ok['DW'] < 0.5).sum()}")

print(f"\n  === Parameter SE (median across countries) ===")
for name in PARAM_NAMES:
    se_col = f'{name}_SE'
    cv_col = f'{name}_CV'
    if se_col in ok.columns:
        valid = ok[ok[se_col].notna()]
        print(f"    {name:8s}: median SE = {valid[se_col].median():.4f}, "
              f"median CV = {valid[cv_col].median():.3f}")

print(f"\n  === pcov validity ===")
pcov_ok = ok['pcov_valid'].sum()
print(f"    Valid pcov: {pcov_ok}/{len(ok)} ({pcov_ok/len(ok)*100:.1f}%)")

if 'max_param_corr' in ok.columns:
    valid_corr = ok[ok['max_param_corr'].notna()]
    print(f"\n  === Max parameter correlation (off-diagonal) ===")
    print(f"    Median: {valid_corr['max_param_corr'].median():.3f}")
    print(f"    > 0.95: {(valid_corr['max_param_corr'] > 0.95).sum()} countries")
    print(f"    > 0.99: {(valid_corr['max_param_corr'] > 0.99).sum()} countries")

# Key countries
print(f"\n  === Key countries ===")
key_codes = ['CHN', 'USA', 'NOR', 'FRA', 'KOR', 'JPN', 'SWE', 'ISL']
for code in key_codes:
    r = ok[ok['code'] == code]
    if len(r) == 0:
        continue
    r = r.iloc[0]
    print(f"\n  {code} ({r.get('entity', '')[:20]}): R²={r['R2']:.4f}, RMSE={r['RMSE']:.4f}, DW={r['DW']:.3f}")
    for name in PARAM_NAMES:
        val = r[name]
        se_val = r.get(f'{name}_SE', np.nan)
        ci_lo = r.get(f'{name}_CI_lo', np.nan)
        ci_hi = r.get(f'{name}_CI_hi', np.nan)
        if pd.notna(se_val):
            print(f"    {name:8s} = {val:+.4f} ± {se_val:.4f}  95%CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")

# Save
out_path = f'{BASE}/_richards_diagnostics.csv'
results.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\nSaved: {out_path}")
