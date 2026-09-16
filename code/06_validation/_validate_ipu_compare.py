"""Compare IPU/WB fits with WPEI fits and generate paper-ready summary."""
import sys, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
from scipy.stats import spearmanr

DIR = r'data'

# Load IPU results
ipu = pd.read_csv(f'{DIR}/_ipu_wb_richards_fits.csv')
print(f'IPU fits: {len(ipu)} countries')
print(f'  median R² = {ipu["r2"].median():.4f}')
print(f'  mean R² = {ipu["r2"].mean():.4f}')
print(f'  >=0.80: {(ipu["r2"] >= 0.80).sum()}/{len(ipu)} ({(ipu["r2"] >= 0.80).mean()*100:.1f}%)')
print(f'  >=0.90: {(ipu["r2"] >= 0.90).sum()}/{len(ipu)} ({(ipu["r2"] >= 0.90).mean()*100:.1f}%)')
print(f'  >=0.95: {(ipu["r2"] >= 0.95).sum()}/{len(ipu)} ({(ipu["r2"] >= 0.95).mean()*100:.1f}%)')
yr0 = ipu["year_range"].iloc[0].split("-")[0]
yr1 = ipu["year_range"].iloc[-1].split("-")[1]
print(f'  Year range: {yr0}-{yr1}')
print(f'  Mean n_points: {ipu["n_points"].mean():.1f}')

# Load WPEI data (fits from vdem_fit_all)
wpei = pd.read_csv(f'{DIR}/_vdem_fit_all.csv')
print(f'\nWPEI fits: {len(wpei)} countries')
print(f'  median R² (Richards) = {wpei["ric_R2"].median():.4f}')
print(f'  >=0.80: {(wpei["ric_R2"] >= 0.80).sum()}/{len(wpei)}')
print(f'  >=0.90: {(wpei["ric_R2"] >= 0.90).sum()}/{len(wpei)}')
print(f'  >=0.95: {(wpei["ric_R2"] >= 0.95).sum()}/{len(wpei)}')

# ── Match countries between datasets ──
# IPU uses World Bank names, WPEI uses OWID names
# Try matching on name similarity
matched = []
for _, irow in ipu.iterrows():
    name = irow['country'].lower().strip()
    # Try direct match
    match = wpei[wpei['entity'].str.lower().str.strip() == name]
    if len(match) == 0:
        # Try partial match
        match = wpei[wpei['entity'].str.lower().str.contains(name[:10], na=False)]
    if len(match) > 0:
        wrow = match.iloc[0]
        matched.append({
            'ipu_country': irow['country'],
            'wpei_country': wrow['entity'],
            'ipu_r2': irow['r2'],
            'wpei_r2': wrow['ric_R2'],
            'ipu_tmid': irow['t_mid'],
            'wpei_tmid': wrow['ric_tmid'],
            'ipu_n': irow['n_points'],
            'wpei_n': wrow['n_years'],
        })

df_m = pd.DataFrame(matched)
print(f'\nMatched countries: {len(df_m)}')

if len(df_m) > 10:
    # Spearman correlation between IPU and WPEI R²
    rho, p = spearmanr(df_m['ipu_r2'], df_m['wpei_r2'])
    print(f'ρ(IPU_R², WPEI_R²) = {rho:.3f} (p={p:.4f})')

    # For countries with IPU R² >= 0.90, what's the WPEI R²?
    high_ipu = df_m[df_m['ipu_r2'] >= 0.90]
    print(f'\nCountries with IPU R² >= 0.90: {len(high_ipu)}')
    print(f'  Their median WPEI R²: {high_ipu["wpei_r2"].median():.4f}')
    print(f'  Their median WPEI t_mid: {high_ipu["wpei_tmid"].median():.0f}')

# ── Key comparison: IPU vs WPEI for late-transition countries ──
# Countries with t_mid >= 1990 should have most of their transition in the IPU window
if len(df_m) > 10:
    late = df_m[df_m['wpei_tmid'] >= 1980]
    print(f'\nCountries with WPEI t_mid >= 1980: {len(late)}')
    if len(late) > 0:
        print(f'  IPU median R² = {late["ipu_r2"].median():.4f}')
        print(f'  WPEI median R² = {late["wpei_r2"].median():.4f}')

# ── Best validation: IPU-only high-R² countries ──
top20 = ipu.nlargest(20, 'r2')
print(f'\nTop 20 IPU R² countries:')
for _, r in top20.iterrows():
    print(f'  {r["country"]:30s}  n={r["n_points"]:3.0f}  R²={r["r2"]:.4f}  t_mid={r["t_mid"]:.0f}')

print('\nDone.')
