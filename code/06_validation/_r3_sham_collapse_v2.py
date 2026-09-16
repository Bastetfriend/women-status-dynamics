"""Recompute sham collapse with FAIR comparison: Richards fitted values vs isotonic fitted values."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

WPEI_CSV = r"data\women-political-empowerment-index.csv"
DIAG_CSV = r"data\_richards_diagnostics.csv"

wpei = pd.read_csv(WPEI_CSV)
diag = pd.read_csv(DIAG_CSV)
wpei_col = "Women's Political Empowerment Index"
wpei['S'] = wpei[wpei_col] - 1

def richards(t, S_floor, S_ceil, k, t_mid, v):
    return S_floor + (S_ceil - S_floor) / (1 + v * np.exp(-k * (t - t_mid)))**(1/v)

def compute_band_fraction(t_hat, S_hat, v, band_width=0.10):
    master = (1 + v * np.exp(-t_hat))**(-1/v)
    within = np.abs(S_hat - master) < band_width
    return within.mean()

# Get Richards params
richards_params = {}
for _, row in diag.iterrows():
    if row['status'] == 'ok':
        richards_params[row['entity']] = {
            'S_floor': row['S_floor'], 'S_ceil': row['S_ceil'],
            'k': row['k'], 't_mid': row['t_mid'], 'v': row['v']
        }

country_counts = wpei.groupby('Entity').size()
valid_countries = country_counts[country_counts >= 30].index

results = []
for entity in valid_countries:
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

    t_hat = k * (t - t_mid)

    # 1. Richards FITTED values
    S_rich = richards(t, S_floor, S_ceil, k, t_mid, v_val)
    S_hat_rich = (S_rich - S_floor) / (S_ceil - S_floor)
    rich_band = compute_band_fraction(t_hat, S_hat_rich, v_val)

    # 2. Raw data
    S_hat_raw = (S - S_floor) / (S_ceil - S_floor)
    raw_band = compute_band_fraction(t_hat, S_hat_raw, v_val)

    # 3. Isotonic regression (monotonic, ~N degrees of freedom)
    try:
        iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
        S_iso = iso.fit_transform(t, S)
        S_hat_iso = (S_iso - S_floor) / (S_ceil - S_floor)
        iso_band = compute_band_fraction(t_hat, S_hat_iso, v_val)
    except Exception:
        iso_band = np.nan

    # 4. Logistic (v=1) fitted values — the simplest Richards special case
    S_log = richards(t, S_floor, S_ceil, k, t_mid, 1.0)
    S_hat_log = (S_log - S_floor) / (S_ceil - S_floor)
    log_band = compute_band_fraction(t_hat, S_hat_log, 1.0)

    results.append({
        'entity': entity, 'n': len(t),
        'rich_fit_band': rich_band,
        'raw_band': raw_band,
        'iso_band': iso_band,
        'log_band': log_band,
    })

df = pd.DataFrame(results)
df = df.dropna(subset=['iso_band'])

print(f'Fair sham collapse comparison: {len(df)} countries\n')

print('━' * 60)
print('BAND FRACTION (fraction of points within ±0.10 of master curve)')
print('━' * 60)
print(f'  Richards FIT values:           median = {df["rich_fit_band"].median():.4f}')
print(f'  Raw data (original collapse):   median = {df["raw_band"].median():.4f}')
print(f'  Isotonic regression (sham):     median = {df["iso_band"].median():.4f}')
print(f'  Logistic v=1 (simple baseline): median = {df["log_band"].median():.4f}')
print()

# Key comparison 1: Richards fit vs Isotonic
print('━' * 60)
print('COMPARISON 1: Richards fitted values vs isotonic regression')
print('━' * 60)
diff_ri = df['iso_band'] - df['rich_fit_band']
print(f'  Iso − Richards fit: median Δ = {diff_ri.median():.4f}')
w_ri = stats.wilcoxon(df['iso_band'], df['rich_fit_band'])
print(f'  Wilcoxon signed-rank p = {w_ri.pvalue:.6f}')
n_iso_better = (df['iso_band'] > df['rich_fit_band']).sum()
print(f'  Isotonic > Richards: {n_iso_better}/{len(df)} ({100*n_iso_better/len(df):.1f}%)')
n_rich_better = (df['rich_fit_band'] > df['iso_band']).sum()
print(f'  Richards > Isotonic: {n_rich_better}/{len(df)} ({100*n_rich_better/len(df):.1f}%)')

# Key comparison 2: Richards fit vs Logistic (v=1)
print()
print('━' * 60)
print('COMPARISON 2: Richards (v free) vs Logistic (v=1) fitted values')
print('━' * 60)
diff_rl = df['rich_fit_band'] - df['log_band']
print(f'  Richards − Logistic: median Δ = {diff_rl.median():.4f}')
w_rl = stats.wilcoxon(df['rich_fit_band'], df['log_band'])
print(f'  Wilcoxon signed-rank p = {w_rl.pvalue:.6f}')
n_rich_better2 = (df['rich_fit_band'] > df['log_band']).sum()
print(f'  Richards > Logistic: {n_rich_better2}/{len(df)} ({100*n_rich_better2/len(df):.1f}%)')

# How much of the gap does Richards close?
print()
print('━' * 60)
print('INTERPRETATION')
print('━' * 60)
# Isotonic regression has ~N degrees of freedom and achieves near-perfect fit
# The Richards function has ~4 effective parameters
# How much does Richards capture of the feasible band fraction?
gap = df['iso_band'].median() - df['log_band'].median()  # total gap: worst (logistic) to best (isotonic)
closed = df['rich_fit_band'].median() - df['log_band'].median()  # how much Richards improves over logistic
if gap > 0:
    print(f'  Logistic → Isotonic gap: {gap:.4f}')
    print(f'  Richards improvement over logistic: {closed:.4f}')
    print(f'  Fraction of gap closed by Richards: {closed/gap:.2%}')
else:
    print(f'  Gap is zero or negative — isotonic and logistic are similar')

print()
# Save
df.to_csv(r'data\_r3_sham_collapse_v2.csv', index=False)
print('Saved to _r3_sham_collapse_v2.csv')
