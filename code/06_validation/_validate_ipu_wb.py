"""IPU/WB women's parliamentary seats Richards validation.
Compares R² between WPEI and independent WB SG.GEN.PARL.ZS data.
"""
import sys, requests, json, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ── 1. Download World Bank SG.GEN.PARL.ZS ──
print("Downloading World Bank SG.GEN.PARL.ZS...")
records = []
page = 1
while True:
    url = f'https://api.worldbank.org/v2/country/all/indicator/SG.GEN.PARL.ZS?format=json&per_page=1000&page={page}'
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f'  Page {page}: status {r.status_code}, stopping.')
            break
        data = r.json()
        if len(data) < 2 or not data[1]:
            break
        for item in data[1]:
            if item['value'] is not None:
                records.append({
                    'country': item['country']['value'],
                    'iso3': item['country']['id'],
                    'year': int(item['date']),
                    'seats_pct': float(item['value']) / 100.0  # normalize to 0-1 like WPEI
                })
        print(f'  Page {page}: {len(data[1])} records, total so far {len(records)}')
        page += 1
        time.sleep(0.3)  # rate limit
    except Exception as e:
        print(f'  Error page {page}: {e}')
        break

df = pd.DataFrame(records)
print(f'\nDownloaded {len(df)} records, {df["country"].nunique()} entities, {df["year"].min()}-{df["year"].max()}')

# ── 2. Load WPEI data ──
wpei = pd.read_csv(r'data\women-political-empowerment-index.csv')
wpei_col = "Women's Political Empowerment Index"
# WPEI is already 0-1, but paper uses S = WPEI - 1. We'll fit on S+1 = WPEI.
# For IPU data, it's seats_pct (0-1). Let's transform both to the same scale.
# Paper uses S(t) = WPEI - 1, range [-1, 0] for erosion phase.
# For comparison, let's fit Richards on the raw 0-1 scale for both.

# ── 3. Richards function ──
def richards(t, S_floor, S_ceil, k, t_mid, v):
    return S_floor + (S_ceil - S_floor) / (1 + v * np.exp(-k * (t - t_mid)))**(1/v)

# ── 4. Fit for WB data ──
WB_DIR = r'data'
results = []
MIN_POINTS = 12  # need at least 12 data points for a 5-parameter fit

for country in sorted(df['country'].unique()):
    cdf = df[df['country'] == country].sort_values('year')
    if len(cdf) < MIN_POINTS:
        continue

    t = cdf['year'].values.astype(float)
    y = cdf['seats_pct'].values

    # Skip if data is too flat (no variation)
    if np.std(y) < 0.005:
        continue

    # Initial guess
    t0, t1 = t.min(), t.max()
    y0, y1 = y[0], y[-1]
    # Heuristic bounds
    try:
        popt, _ = curve_fit(
            richards, t, y,
            p0=[0.0, max(0.3, y1), 0.1, (t0 + t1) / 2, 1.0],
            bounds=([-0.1, 0.0, 0.001, t0 - 50, 0.005],
                    [0.3, 0.8, 1.0, t1 + 50, 20.0]),
            maxfev=5000
        )
        y_fit = richards(t, *popt)
        ss_res = np.sum((y - y_fit)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        results.append({
            'country': country,
            'iso3': cdf['iso3'].iloc[0],
            'n_points': len(cdf),
            'year_range': f'{int(t0)}-{int(t1)}',
            'r2': r2,
            'S_floor': popt[0],
            'S_ceil': popt[1],
            'k': popt[2],
            't_mid': popt[3],
            'v': popt[4],
        })
    except Exception:
        pass

df_r = pd.DataFrame(results)
print(f'\nWB fits: {len(df_r)} countries converged')
print(f'R² stats: mean={df_r["r2"].mean():.3f}, median={df_r["r2"].median():.3f}')
print(f'R² >= 0.80: {(df_r["r2"] >= 0.80).sum()}/{len(df_r)} ({(df_r["r2"] >= 0.80).mean()*100:.1f}%)')
print(f'R² >= 0.90: {(df_r["r2"] >= 0.90).sum()}/{len(df_r)} ({(df_r["r2"] >= 0.90).mean()*100:.1f}%)')

# Compare with WPEI R² distribution on the SAME countries
# Load WPEI fits
wpei_fits = pd.read_csv(f'{WB_DIR}/_vdem_fit_all.csv')
print(f'\nWPEI reference: median R² = {wpei_fits["r_squared"].median():.3f} (all 191 countries)')

# Save
df_r.to_csv(f'{WB_DIR}/_ipu_wb_richards_fits.csv', index=False)
print(f'\nSaved to {WB_DIR}/_ipu_wb_richards_fits.csv')

# Print top 20 countries by R² for spot-check
print('\nTop 20 by R²:')
for _, row in df_r.nlargest(20, 'r2').iterrows():
    print(f'  {row["country"]:30s}  n={row["n_points"]:3d}  R²={row["r2"]:.4f}  t_mid={row["t_mid"]:.0f}')
