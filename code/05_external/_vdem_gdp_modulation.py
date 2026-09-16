# -*- coding: utf-8 -*-
"""
M(E(t)) 经济调制验证
Richards拟合残差 vs GDP per capita growth → 验证经济周期是否调制S(t)
"""

import numpy as np
import pandas as pd
import json
import urllib.request
import urllib.error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = r"data"

# ═══════════════════════════════════════════════════════
#  1. Load WPEI data
# ═══════════════════════════════════════════════════════

print("=" * 65)
print("  M(E(t)) Economic Modulation Verification")
print("=" * 65)

wpei = pd.read_csv(f"{BASE}/women-political-empowerment-index.csv")
wpei.columns = ['entity', 'code', 'year', 'wpei', 'region']
wpei['S'] = wpei['wpei'] - 1
print(f"\n  WPEI: {len(wpei)} rows, {wpei['code'].nunique()} countries")

# ═══════════════════════════════════════════════════════
#  2. Load Richards fit parameters
# ═══════════════════════════════════════════════════════

params = pd.read_csv(f"{BASE}/_vdem_fit_all.csv")
params = params[params['status'] == 'ok']
print(f"  Richards fits: {len(params)} countries")

# ═══════════════════════════════════════════════════════
#  3. Download GDP per capita growth from World Bank API
# ═══════════════════════════════════════════════════════

GDP_FILE = f"{BASE}/_gdp_growth_worldbank.csv"

def download_gdp():
    """Download GDP per capita growth (annual %) from World Bank API."""
    print("\n  Downloading GDP per capita growth from World Bank API...")

    all_data = []
    page = 1
    total_pages = 999

    while page <= total_pages:
        url = (
            f"https://api.worldbank.org/v2/country/all/indicator/NY.GDP.PCAP.KD.ZG"
            f"?format=json&per_page=10000&date=1960:2024&page={page}"
        )
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp = urllib.request.urlopen(req, timeout=30)
            raw = json.loads(resp.read())

            if len(raw) < 2:
                print(f"    Page {page}: no data")
                break

            meta = raw[0]
            total_pages = meta.get('pages', 1)
            records = raw[1]

            for r in records:
                if r.get('value') is not None:
                    all_data.append({
                        'code': r['country']['id'],
                        'entity': r['country']['value'],
                        'year': int(r['date']),
                        'gdp_growth': float(r['value']),
                    })

            print(f"    Page {page}/{total_pages}: {len(records)} records")
            page += 1

        except Exception as e:
            print(f"    Error on page {page}: {e}")
            break

    gdp = pd.DataFrame(all_data)
    gdp.to_csv(GDP_FILE, index=False)
    print(f"  Saved: {len(gdp)} GDP records → {GDP_FILE}")
    return gdp

# Try loading cached; download if missing
try:
    gdp = pd.read_csv(GDP_FILE)
    print(f"  GDP data (cached): {len(gdp)} rows, {gdp['code'].nunique()} countries")
except FileNotFoundError:
    gdp = download_gdp()

print(f"  GDP years: {gdp['year'].min()}–{gdp['year'].max()}")

# ── Convert World Bank alpha-2 → OWID alpha-3 codes ──

# Build entity → OWID code mapping
owid_entity_to_code = {}
for _, row in wpei.drop_duplicates('code').iterrows():
    owid_entity_to_code[row['entity'].strip().lower()] = row['code']

# Manual overrides for common World Bank ↔ OWID name mismatches
MANUAL_WB_TO_OWID = {
    'Korea, Rep.': 'KOR', 'Korea, Dem. People\'s Rep.': 'PRK',
    'Russian Federation': 'RUS', 'Iran, Islamic Rep.': 'IRN',
    'Egypt, Arab Rep.': 'EGY', 'Venezuela, RB': 'VEN',
    'Turkiye': 'TUR', 'Lao PDR': 'LAO', 'Kyrgyz Republic': 'KGZ',
    'Slovak Republic': 'SVK', 'Czech Republic': 'CZE', 'Czechia': 'CZE',
    'Congo, Rep.': 'COG', 'Congo, Dem. Rep.': 'COD',
    'Gambia, The': 'GMB', 'Bahamas, The': 'BHS',
    'Yemen, Rep.': 'YEM', 'Syrian Arab Republic': 'SYR',
    'West Bank and Gaza': 'PSE', 'Micronesia, Fed. Sts.': 'FSM',
    'St. Lucia': 'LCA', 'St. Vincent and the Grenadines': 'VCT',
    'St. Kitts and Nevis': 'KNA', 'Cabo Verde': 'CPV',
    'Cote d\'Ivoire': 'CIV', "Côte d'Ivoire": 'CIV',
    'Eswatini': 'SWZ', 'North Macedonia': 'MKD',
    'Myanmar': 'MMR', 'Brunei Darussalam': 'BRN',
    'Hong Kong SAR, China': 'HKG', 'Macao SAR, China': 'MAC',
}

wb_code_to_owid = {}
for _, row in gdp.drop_duplicates('code').iterrows():
    wb_entity = row['entity']
    wb_code = row['code']

    # 1. Try manual override
    if wb_entity in MANUAL_WB_TO_OWID:
        wb_code_to_owid[wb_code] = MANUAL_WB_TO_OWID[wb_entity]
        continue

    # 2. Try exact match on entity name (case-insensitive)
    key = wb_entity.strip().lower()
    if key in owid_entity_to_code:
        wb_code_to_owid[wb_code] = owid_entity_to_code[key]
        continue

    # 3. Try substring match
    for owid_key, owid_code in owid_entity_to_code.items():
        if key in owid_key or owid_key in key:
            wb_code_to_owid[wb_code] = owid_code
            break

gdp['code_owid'] = gdp['code'].map(wb_code_to_owid)
matched = gdp['code_owid'].notna().sum()
print(f"\n  Code mapping: {len(wb_code_to_owid)} WB→OWID matches ({matched}/{len(gdp)} rows mapped)")
gdp = gdp.dropna(subset=['code_owid'])
gdp['code'] = gdp['code_owid']
gdp = gdp.drop(columns=['code_owid'])

# ═══════════════════════════════════════════════════════
#  4. Compute Richards predicted S(t) + residuals
# ═══════════════════════════════════════════════════════

def richards(t, S_floor, S_ceil, k, t_mid, v):
    z = v * np.exp(-k * (t - t_mid))
    z = np.clip(z, 1e-15, 1e15)
    return S_floor + (S_ceil - S_floor) / (1 + z) ** (1 / v)

print("\n  Computing Richards residuals...")

residuals = []
for _, p in params.iterrows():
    code = p['code']
    country_data = wpei[wpei['code'] == code].copy()

    for _, row in country_data.iterrows():
        t = row['year']
        s_actual = row['S']
        s_pred = richards(t, p['ric_floor'], p['ric_ceil'], p['ric_k'], p['ric_tmid'], p['ric_v'])
        residuals.append({
            'code': code,
            'entity': row['entity'],
            'year': t,
            'S_actual': s_actual,
            'S_predicted': s_pred,
            'residual': s_actual - s_pred,
        })

res_df = pd.DataFrame(residuals)
print(f"  Residuals: {len(res_df)} country-year observations")

# ═══════════════════════════════════════════════════════
#  5. Merge residuals with GDP growth
# ═══════════════════════════════════════════════════════

merged = res_df.merge(gdp[['code', 'year', 'gdp_growth']], on=['code', 'year'], how='inner')
print(f"  Merged (residual + GDP): {len(merged)} observations, {merged['code'].nunique()} countries")

# Drop NaN/inf
merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=['residual', 'gdp_growth'])
print(f"  After cleaning: {len(merged)} observations")

# ═══════════════════════════════════════════════════════
#  6. Overall correlation
# ═══════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("  OVERALL CORRELATION")
print("─" * 50)

r_pearson, p_pearson = stats.pearsonr(merged['gdp_growth'], merged['residual'])
r_spearman, p_spearman = stats.spearmanr(merged['gdp_growth'], merged['residual'])

print(f"  Pearson  r = {r_pearson:+.4f}, p = {p_pearson:.2e}")
print(f"  Spearman ρ = {r_spearman:+.4f}, p = {p_spearman:.2e}")

# ═══════════════════════════════════════════════════════
#  7. Within-country correlations
# ═══════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("  WITHIN-COUNTRY CORRELATIONS")
print("─" * 50)

country_corrs = []
for code, grp in merged.groupby('code'):
    if len(grp) >= 15:  # Need enough years for meaningful correlation
        r, p = stats.pearsonr(grp['gdp_growth'], grp['residual'])
        country_corrs.append({
            'code': code,
            'entity': grp['entity'].iloc[0],
            'n_years': len(grp),
            'pearson_r': r,
            'p_value': p,
        })

cc = pd.DataFrame(country_corrs)
print(f"  Countries with ≥15 years: {len(cc)}")
print(f"  Mean within-country r:    {cc['pearson_r'].mean():+.4f}")
print(f"  Median within-country r:  {cc['pearson_r'].median():+.4f}")
print(f"  % positive r:             {(cc['pearson_r'] > 0).mean()*100:.1f}%")
print(f"  % significant (p<0.05):   {(cc['p_value'] < 0.05).mean()*100:.1f}%")
print(f"  % sig & positive:         {((cc['p_value'] < 0.05) & (cc['pearson_r'] > 0)).mean()*100:.1f}%")

# Top/bottom
print(f"\n  Strongest POSITIVE correlations:")
for _, row in cc.nlargest(5, 'pearson_r').iterrows():
    sig = '*' if row['p_value'] < 0.05 else ''
    print(f"    {row['entity']:<30} r={row['pearson_r']:+.3f} (n={row['n_years']}){sig}")

print(f"\n  Strongest NEGATIVE correlations:")
for _, row in cc.nsmallest(5, 'pearson_r').iterrows():
    sig = '*' if row['p_value'] < 0.05 else ''
    print(f"    {row['entity']:<30} r={row['pearson_r']:+.3f} (n={row['n_years']}){sig}")

# ═══════════════════════════════════════════════════════
#  8. Lag analysis: does GDP change LEAD S(t) residual?
# ═══════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("  LAG ANALYSIS (GDP leads residual by 0–5 years)")
print("─" * 50)

lag_results = []
for lag in range(0, 6):
    # Shift GDP growth backwards by `lag` years
    # i.e., residual(t) ~ gdp_growth(t - lag)
    lag_merged = res_df.copy()
    gdp_shifted = gdp.copy()
    gdp_shifted['year'] = gdp_shifted['year'] + lag
    lag_merged = lag_merged.merge(gdp_shifted[['code', 'year', 'gdp_growth']], on=['code', 'year'], how='inner')
    lag_merged = lag_merged.replace([np.inf, -np.inf], np.nan).dropna(subset=['residual', 'gdp_growth'])

    if len(lag_merged) > 100:
        r, p = stats.pearsonr(lag_merged['gdp_growth'], lag_merged['residual'])
        lag_results.append({'lag': lag, 'r': r, 'p': p, 'n': len(lag_merged)})
        print(f"  Lag {lag}y: r = {r:+.4f}, p = {p:.2e}, n = {len(lag_merged)}")

# ═══════════════════════════════════════════════════════
#  9. By transition phase: stronger in mid-transition?
# ═══════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("  BY TRANSITION PHASE (sigmoid derivative)")
print("─" * 50)

# Compute derivative of Richards at each point
def richards_derivative(t, S_floor, S_ceil, k, t_mid, v):
    z = v * np.exp(-k * (t - t_mid))
    z = np.clip(z, 1e-15, 1e15)
    base = (1 + z) ** (1/v)
    return (S_ceil - S_floor) * k * np.exp(-k * (t - t_mid)) / ((1 + z) ** (1/v + 1))

for _, p in params.iterrows():
    code = p['code']
    mask = merged['code'] == code
    if mask.any():
        t_vals = merged.loc[mask, 'year'].values
        deriv = np.abs(richards_derivative(t_vals, p['ric_floor'], p['ric_ceil'],
                                            p['ric_k'], p['ric_tmid'], p['ric_v']))
        merged.loc[mask, 'abs_derivative'] = deriv

# Terciles by derivative
if 'abs_derivative' in merged.columns:
    merged['deriv_tercile'] = pd.qcut(merged['abs_derivative'], 3, labels=['slow', 'mid', 'fast'])

    for tercile in ['slow', 'mid', 'fast']:
        sub = merged[merged['deriv_tercile'] == tercile]
        r, p = stats.pearsonr(sub['gdp_growth'], sub['residual'])
        print(f"  {tercile:>5} transition: r = {r:+.4f}, p = {p:.2e}, n = {len(sub)}")

# ═══════════════════════════════════════════════════════
#  10. Visualize
# ═══════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('M(E(t)) Economic Modulation Verification', fontsize=14, fontweight='bold')

# (a) Scatter: GDP growth vs residual
ax = axes[0, 0]
ax.scatter(merged['gdp_growth'], merged['residual'], alpha=0.05, s=3, color='steelblue')
# Binned means
bins = np.percentile(merged['gdp_growth'].dropna(), np.arange(0, 101, 5))
merged['gdp_bin'] = pd.cut(merged['gdp_growth'], bins=bins)
binned = merged.groupby('gdp_bin', observed=True)['residual'].agg(['mean', 'sem']).dropna()
bin_centers = [(iv.left + iv.right)/2 for iv in binned.index]
ax.errorbar(bin_centers, binned['mean'], yerr=1.96*binned['sem'],
            color='red', linewidth=2, capsize=3, label='Binned mean ± 95% CI')
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('GDP per capita growth (%)')
ax.set_ylabel('Richards residual (S actual − S predicted)')
ax.set_title(f'(a) Overall: r={r_pearson:+.3f}, p={p_pearson:.1e}')
ax.legend(fontsize=8)
ax.set_xlim(-30, 30)

# (b) Within-country correlation histogram
ax = axes[0, 1]
ax.hist(cc['pearson_r'], bins=40, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', linewidth=1.5, linestyle='--')
ax.axvline(cc['pearson_r'].mean(), color='orange', linewidth=2, label=f"Mean = {cc['pearson_r'].mean():+.3f}")
ax.set_xlabel('Within-country Pearson r')
ax.set_ylabel('Number of countries')
ax.set_title(f'(b) Within-country correlations (n={len(cc)})')
ax.legend(fontsize=8)

# (c) Lag analysis
ax = axes[1, 0]
if lag_results:
    lr = pd.DataFrame(lag_results)
    ax.bar(lr['lag'], lr['r'], color='steelblue', edgecolor='white')
    ax.set_xlabel('Lag (years, GDP leads)')
    ax.set_ylabel('Pearson r')
    ax.set_title('(c) Lag analysis: GDP growth → S(t) residual')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    for _, row in lr.iterrows():
        sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*' if row['p'] < 0.05 else ''
        ax.text(row['lag'], row['r'] + 0.001, sig, ha='center', fontsize=10)

# (d) Example countries: residual vs GDP growth time series
ax = axes[1, 1]
examples = ['CHN', 'USA', 'NOR', 'JPN']
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
for i, code in enumerate(examples):
    sub = merged[merged['code'] == code].sort_values('year')
    if len(sub) > 10:
        entity = sub['entity'].iloc[0]
        # Normalize both to z-scores for overlay
        res_z = (sub['residual'] - sub['residual'].mean()) / (sub['residual'].std() + 1e-10)
        gdp_z = (sub['gdp_growth'] - sub['gdp_growth'].mean()) / (sub['gdp_growth'].std() + 1e-10)
        ax.plot(sub['year'], res_z.rolling(3, center=True).mean(),
                color=colors[i], linewidth=1.5, label=f'{entity} (residual)')
        ax.plot(sub['year'], gdp_z.rolling(3, center=True).mean(),
                color=colors[i], linewidth=1, linestyle='--', alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Z-score (3-year rolling mean)')
ax.set_title('(d) Example countries: S residual (solid) vs GDP growth (dashed)')
ax.legend(fontsize=7, ncol=2)
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

plt.tight_layout()
plt.savefig(f"{BASE}/_vdem_gdp_modulation.png", dpi=200, bbox_inches='tight')
print(f"\n  Saved: _vdem_gdp_modulation.png")

# ═══════════════════════════════════════════════════════
#  11. Summary
# ═══════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)
print(f"""
  Data: {merged['code'].nunique()} countries, {len(merged)} country-year observations
  Period: {merged['year'].min()}–{merged['year'].max()}

  Overall:
    Pearson  r = {r_pearson:+.4f}  (p = {p_pearson:.2e})
    Spearman ρ = {r_spearman:+.4f}  (p = {p_spearman:.2e})

  Within-country (n ≥ 15 years):
    {len(cc)} countries
    Mean r = {cc['pearson_r'].mean():+.4f}
    {(cc['pearson_r'] > 0).mean()*100:.0f}% positive
    {(cc['p_value'] < 0.05).mean()*100:.0f}% significant (p < 0.05)

  Interpretation:
    Positive r → GDP growth associated with S above Richards trend
    = economic prosperity modulates gender empowerment upward
    = M(E(t)) term is empirically supported
""")

# Save within-country correlation table
cc.to_csv(f"{BASE}/_vdem_gdp_country_correlations.csv", index=False)
print(f"  Saved: _vdem_gdp_country_correlations.csv")
