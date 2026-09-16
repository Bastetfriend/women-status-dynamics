# -*- coding: utf-8 -*-
"""
M(E(t)) 经济调制验证 v2 — 生产力综合指标
不只看GDP增长，看城市化/工业化/第三产业/教育/R&D/高科技出口
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
import warnings, os, time
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = r"data"

# ═══════════════════════════════════════════════════════
#  1. Load WPEI + Richards fit
# ═══════════════════════════════════════════════════════

print("=" * 70)
print("  M(E(t)) Productive Forces Modulation — Comprehensive Verification")
print("=" * 70)

wpei = pd.read_csv(f"{BASE}/women-political-empowerment-index.csv")
wpei.columns = ['entity', 'code', 'year', 'wpei', 'region']
wpei['S'] = wpei['wpei'] - 1

params = pd.read_csv(f"{BASE}/_vdem_fit_all.csv")
params = params[params['status'] == 'ok']
print(f"  WPEI: {wpei['code'].nunique()} countries | Richards fits: {len(params)}")

# ═══════════════════════════════════════════════════════
#  2. World Bank indicators to download
# ═══════════════════════════════════════════════════════

INDICATORS = {
    'urban':    ('SP.URB.TOTL.IN.ZS', 'Urbanization (% population)'),
    'industry': ('NV.IND.TOTL.ZS',    'Industry value added (% GDP)'),
    'services': ('NV.SRV.TOTL.ZS',    'Services value added (% GDP)'),
    'educ':     ('SE.SEC.ENRR',        'Secondary school enrollment (% gross)'),
    'rnd':      ('GB.XPD.RSDV.GD.ZS', 'R&D expenditure (% GDP)'),
    'hitech':   ('TX.VAL.TECH.MF.ZS', 'High-tech exports (% manuf.)'),
    'gdp_grow': ('NY.GDP.PCAP.KD.ZG', 'GDP per capita growth (%)'),
}

# ═══════════════════════════════════════════════════════
#  3. Download from World Bank API
# ═══════════════════════════════════════════════════════

CACHE_FILE = f"{BASE}/_wb_indicators_cache.csv"

def download_indicator(wb_code, label):
    """Download one indicator from World Bank API, all countries 1960-2024."""
    all_data = []
    page = 1
    total_pages = 999

    while page <= total_pages:
        url = (
            f"https://api.worldbank.org/v2/country/all/indicator/{wb_code}"
            f"?format=json&per_page=10000&date=1960:2024&page={page}"
        )
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            resp = urllib.request.urlopen(req, timeout=60)
            raw = json.loads(resp.read())
            if len(raw) < 2:
                break
            meta = raw[0]
            total_pages = meta.get('pages', 1)
            for r in raw[1]:
                if r.get('value') is not None:
                    all_data.append({
                        'wb_code': r['country']['id'],
                        'entity': r['country']['value'],
                        'year': int(r['date']),
                        'value': float(r['value']),
                    })
            page += 1
        except Exception as e:
            print(f"      Error page {page}: {e}")
            break
        time.sleep(0.3)  # be polite to API

    return all_data

def download_all():
    all_rows = []
    for key, (wb_code, label) in INDICATORS.items():
        print(f"    Downloading {key} ({wb_code})...", end="", flush=True)
        data = download_indicator(wb_code, label)
        for d in data:
            d['indicator'] = key
        all_rows.extend(data)
        print(f" {len(data)} records")

    df = pd.DataFrame(all_rows)
    df.to_csv(CACHE_FILE, index=False)
    print(f"  Total: {len(df)} records → {CACHE_FILE}")
    return df

if os.path.exists(CACHE_FILE):
    wb = pd.read_csv(CACHE_FILE)
    print(f"  WB indicators (cached): {len(wb)} rows")
    cached_indicators = set(wb['indicator'].unique())
    needed = set(INDICATORS.keys())
    missing = needed - cached_indicators
    if missing:
        print(f"  Missing indicators: {missing} — skipping (use what we have)")
else:
    print("\n  Downloading World Bank indicators...")
    wb = download_all()

# ═══════════════════════════════════════════════════════
#  4. Code mapping: WB alpha-2 → OWID alpha-3
# ═══════════════════════════════════════════════════════

owid_entity_to_code = {}
for _, row in wpei.drop_duplicates('code').iterrows():
    owid_entity_to_code[row['entity'].strip().lower()] = row['code']

MANUAL_WB_TO_OWID = {
    'Korea, Rep.': 'KOR', "Korea, Dem. People's Rep.": 'PRK',
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
    "Cote d'Ivoire": 'CIV', "Côte d'Ivoire": 'CIV',
    'Eswatini': 'SWZ', 'North Macedonia': 'MKD',
    'Myanmar': 'MMR', 'Brunei Darussalam': 'BRN',
    'Hong Kong SAR, China': 'HKG', 'Macao SAR, China': 'MAC',
}

wb_code_to_owid = {}
for _, row in wb.drop_duplicates('wb_code').iterrows():
    wb_entity = row['entity']
    wb_code = row['wb_code']
    if wb_entity in MANUAL_WB_TO_OWID:
        wb_code_to_owid[wb_code] = MANUAL_WB_TO_OWID[wb_entity]
        continue
    key = wb_entity.strip().lower()
    if key in owid_entity_to_code:
        wb_code_to_owid[wb_code] = owid_entity_to_code[key]
        continue
    for owid_key, owid_code in owid_entity_to_code.items():
        if key in owid_key or owid_key in key:
            wb_code_to_owid[wb_code] = owid_code
            break

wb['code'] = wb['wb_code'].map(wb_code_to_owid)
wb = wb.dropna(subset=['code'])
print(f"  Mapped: {wb['code'].nunique()} countries")

# ═══════════════════════════════════════════════════════
#  5. Pivot: (code, year) → one column per indicator
# ═══════════════════════════════════════════════════════

wb_pivot = wb.pivot_table(index=['code', 'year'], columns='indicator',
                          values='value', aggfunc='first').reset_index()
print(f"  Pivoted: {len(wb_pivot)} country-year rows")
for key in INDICATORS:
    if key in wb_pivot.columns:
        n_valid = wb_pivot[key].notna().sum()
        print(f"    {key:>10}: {n_valid} values")
    else:
        print(f"    {key:>10}: NO DATA (download failed)")


# ═══════════════════════════════════════════════════════
#  6. Compute Richards residuals
# ═══════════════════════════════════════════════════════

def richards(t, S_floor, S_ceil, k, t_mid, v):
    z = v * np.exp(-k * (t - t_mid))
    z = np.clip(z, 1e-15, 1e15)
    return S_floor + (S_ceil - S_floor) / (1 + z) ** (1 / v)

print("\n  Computing Richards residuals...")
residuals = []
for _, p in params.iterrows():
    code = p['code']
    country_data = wpei[wpei['code'] == code]
    for _, row in country_data.iterrows():
        t = row['year']
        s_actual = row['S']
        s_pred = richards(t, p['ric_floor'], p['ric_ceil'], p['ric_k'], p['ric_tmid'], p['ric_v'])
        residuals.append({'code': code, 'year': t,
                          'S_actual': s_actual, 'S_predicted': s_pred,
                          'residual': s_actual - s_pred})

res_df = pd.DataFrame(residuals)

# ═══════════════════════════════════════════════════════
#  7. Merge residuals with indicators
# ═══════════════════════════════════════════════════════

merged = res_df.merge(wb_pivot, on=['code', 'year'], how='inner')
merged = merged.replace([np.inf, -np.inf], np.nan)
print(f"  Merged: {len(merged)} rows, {merged['code'].nunique()} countries")

# Track which indicators actually have data
AVAIL = [k for k in INDICATORS if k in merged.columns and merged[k].notna().sum() > 50]
print(f"  Available indicators: {AVAIL}")

# Also merge Richards params for cross-sectional analysis
merged = merged.merge(params[['code', 'ric_tmid', 'ric_v', 'ric_k', 'ric_ceil', 'ric_R2']],
                       on='code', how='left')

# ═══════════════════════════════════════════════════════
#  8. TEST 1: Residual ~ indicator (panel correlation)
# ═══════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  TEST 1: Richards residual ~ productive forces indicators")
print("═" * 70)

test1_results = {}
for key, (_, label) in INDICATORS.items():
    sub = merged[['residual', key]].dropna() if key in merged.columns else pd.DataFrame()
    if len(sub) < 100:
        print(f"  {key:>10}: too few data points ({len(sub)})")
        continue
    r, p = stats.pearsonr(sub['residual'], sub[key])
    rho, sp = stats.spearmanr(sub['residual'], sub[key])
    test1_results[key] = {'r': r, 'p': p, 'rho': rho, 'sp': sp, 'n': len(sub), 'label': label}
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"  {key:>10}: r={r:+.4f} (p={p:.1e}) ρ={rho:+.4f} n={len(sub)} {sig}")

# ═══════════════════════════════════════════════════════
#  9. TEST 2: Within-country correlation (fixed effects)
# ═══════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  TEST 2: Within-country correlations (fixed effects)")
print("═" * 70)

within_results = {}
for key, (_, label) in INDICATORS.items():
    country_rs = []
    if key not in merged.columns:
        continue
    for code, grp in merged.groupby('code'):
        sub = grp[['residual', key]].dropna()
        if len(sub) >= 10:
            r, p = stats.pearsonr(sub['residual'], sub[key])
            country_rs.append({'code': code, 'r': r, 'p': p, 'n': len(sub)})

    if not country_rs:
        continue

    cdf = pd.DataFrame(country_rs)
    pct_pos = (cdf['r'] > 0).mean() * 100
    pct_sig = (cdf['p'] < 0.05).mean() * 100
    pct_sig_pos = ((cdf['p'] < 0.05) & (cdf['r'] > 0)).mean() * 100
    within_results[key] = cdf

    print(f"\n  {key} ({label}):")
    print(f"    Countries: {len(cdf)}")
    print(f"    Mean r:    {cdf['r'].mean():+.4f}")
    print(f"    Median r:  {cdf['r'].median():+.4f}")
    print(f"    % positive:        {pct_pos:.1f}%")
    print(f"    % sig (p<0.05):    {pct_sig:.1f}%")
    print(f"    % sig & positive:  {pct_sig_pos:.1f}%")

# ═══════════════════════════════════════════════════════
#  10. TEST 3: Cross-sectional — t_mid ~ modernization level
# ═══════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  TEST 3: Sigmoid timing (t_mid) ~ modernization level")
print("  (Do more modernized countries transition earlier?)")
print("═" * 70)

# For each country, take the EARLIEST available value of each indicator
# (proxy for starting modernization level)
# AND the value at t_mid (level when gender transition inflection happened)

cross_section = []
for _, p in params.iterrows():
    code = p['code']
    t_mid = p['ric_tmid']
    row = {'code': code, 'entity': p['entity'], 'ric_tmid': t_mid,
           'ric_v': p['ric_v'], 'ric_k': p['ric_k'], 'ric_ceil': p['ric_ceil']}

    country_wb = wb_pivot[wb_pivot['code'] == code].sort_values('year')
    if len(country_wb) == 0:
        continue

    # Value nearest to t_mid
    for key in INDICATORS:
        if key not in wb_pivot.columns:
            continue
        sub = country_wb[['year', key]].dropna() if key in country_wb.columns else pd.DataFrame()
        if len(sub) == 0:
            continue
        # Closest year to t_mid
        idx_closest = (sub['year'] - t_mid).abs().idxmin()
        row[f'{key}_at_tmid'] = sub.loc[idx_closest, key]
        row[f'{key}_at_tmid_year'] = sub.loc[idx_closest, 'year']
        # Mean value (overall level)
        row[f'{key}_mean'] = country_wb[key].mean()

    cross_section.append(row)

cs = pd.DataFrame(cross_section)
print(f"\n  Countries with cross-section data: {len(cs)}")

for key, (_, label) in INDICATORS.items():
    col = f'{key}_at_tmid'
    if col not in cs.columns:
        continue
    sub = cs[['ric_tmid', col]].dropna()
    if len(sub) < 20:
        continue
    r, p = stats.pearsonr(sub['ric_tmid'], sub[col])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"  t_mid ~ {key:>10} (at t_mid): r={r:+.4f}, p={p:.1e}, n={len(sub)} {sig}")

    # Also test: v ~ indicator
    col_v = f'{key}_at_tmid'
    sub_v = cs[['ric_v', col_v]].dropna()
    if len(sub_v) >= 20:
        r_v, p_v = stats.pearsonr(sub_v['ric_v'], sub_v[col_v])
        sig_v = '***' if p_v < 0.001 else '**' if p_v < 0.01 else '*' if p_v < 0.05 else ''
        print(f"     v  ~ {key:>10} (at t_mid): r={r_v:+.4f}, p={p_v:.1e}, n={len(sub_v)} {sig_v}")

# ═══════════════════════════════════════════════════════
#  11. Flag interesting outliers (China, DPRK, etc.)
# ═══════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  OUTLIER CHECK: Revolutionary vs. gradual path")
print("═" * 70)

flagged = ['CHN', 'PRK', 'CUB', 'VNM', 'RUS', 'SAU', 'IRN', 'KOR', 'USA', 'NOR']
for code in flagged:
    row = cs[cs['code'] == code]
    if len(row) == 0:
        print(f"  {code}: no data")
        continue
    r = row.iloc[0]
    urban = r.get('urban_at_tmid', np.nan)
    indust = r.get('industry_at_tmid', np.nan)
    serv = r.get('services_at_tmid', np.nan)
    educ = r.get('educ_at_tmid', np.nan)
    print(f"  {code} ({r['entity'][:20]:>20}): t_mid={r['ric_tmid']:.0f}, v={r['ric_v']:.1f}"
          f"  | urban={urban:.0f}%" if not np.isnan(urban) else f"  | urban=N/A",
          f" industry={indust:.0f}%" if not np.isnan(indust) else " industry=N/A",
          f" services={serv:.0f}%" if not np.isnan(serv) else " services=N/A",
          f" educ={educ:.0f}%" if not np.isnan(educ) else " educ=N/A")

# ═══════════════════════════════════════════════════════
#  12. Visualize
# ═══════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('M(E(t)) Productive Forces Modulation Verification', fontsize=14, fontweight='bold')

plot_indicators = [k for k in ['urban', 'industry', 'services', 'educ', 'rnd', 'hitech'] if k in AVAIL]
# Pad to 6 for grid layout
while len(plot_indicators) < 6:
    plot_indicators.append(None)
titles_map = {'urban': '(a) Urbanization', 'industry': '(b) Industrialization',
              'services': '(c) Services sector', 'educ': '(d) Education',
              'rnd': '(e) R&D expenditure', 'hitech': '(f) High-tech exports'}

for i in range(6):
    key = plot_indicators[i]
    ax = axes[i // 3, i % 3]

    if key is None:
        ax.set_visible(False)
        continue

    title = titles_map.get(key, key)
    ax = axes[i // 3, i % 3]

    sub = merged[['residual', key]].dropna()
    if len(sub) < 50:
        ax.text(0.5, 0.5, f'{key}\nInsufficient data\n(n={len(sub)})',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title(title)
        continue

    ax.scatter(sub[key], sub['residual'], alpha=0.03, s=3, color='steelblue')

    # Binned means
    try:
        bins = np.percentile(sub[key].dropna(), np.arange(0, 101, 5))
        bins = np.unique(bins)
        if len(bins) >= 3:
            sub_copy = sub.copy()
            sub_copy['bin'] = pd.cut(sub_copy[key], bins=bins)
            binned = sub_copy.groupby('bin', observed=True)['residual'].agg(['mean', 'sem']).dropna()
            bin_centers = [(iv.left + iv.right)/2 for iv in binned.index]
            ax.errorbar(bin_centers, binned['mean'], yerr=1.96*binned['sem'],
                        color='red', linewidth=2, capsize=3)
    except Exception:
        pass

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    if key in test1_results:
        r_val = test1_results[key]['r']
        p_val = test1_results[key]['p']
        ax.set_title(f"{title}: r={r_val:+.3f}, p={p_val:.1e}")
    else:
        ax.set_title(title)

    ax.set_xlabel(INDICATORS[key][1])
    ax.set_ylabel('Richards residual')

plt.tight_layout()
plt.savefig(f"{BASE}/_vdem_productive_forces.png", dpi=200, bbox_inches='tight')
print(f"\n  Saved: _vdem_productive_forces.png")

# ── Cross-section figure: t_mid vs modernization ──

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle('Sigmoid Timing (t_mid) vs. Modernization Level', fontsize=13, fontweight='bold')

for i, (key, title) in enumerate(zip(['urban', 'industry', 'educ'],
                                      ['Urbanization at t_mid', 'Industry at t_mid', 'Education at t_mid'])):
    ax = axes2[i]
    col = f'{key}_at_tmid'
    if col not in cs.columns:
        ax.text(0.5, 0.5, f'{key}\nNo data', transform=ax.transAxes, ha='center', fontsize=12)
        ax.set_title(title)
        continue
    sub = cs[['ric_tmid', col, 'code', 'entity']].dropna()
    if len(sub) < 20:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
        continue

    ax.scatter(sub[col], sub['ric_tmid'], alpha=0.4, s=20, color='steelblue')

    # Highlight flagged countries
    for code in ['CHN', 'KOR', 'USA', 'NOR', 'SAU', 'CUB']:
        pt = sub[sub['code'] == code]
        if len(pt):
            ax.scatter(pt[col].values[0], pt['ric_tmid'].values[0],
                       s=60, color='red', zorder=5, edgecolors='black')
            ax.annotate(code, (pt[col].values[0], pt['ric_tmid'].values[0]),
                        fontsize=7, ha='left', va='bottom')

    r, p = stats.pearsonr(sub[col], sub['ric_tmid'])
    ax.set_xlabel(f'{INDICATORS[key][1]} at t_mid')
    ax.set_ylabel('t_mid (inflection year)')
    ax.set_title(f'{title}\nr={r:+.3f}, p={p:.1e}')

plt.tight_layout()
plt.savefig(f"{BASE}/_vdem_tmid_vs_modernization.png", dpi=200, bbox_inches='tight')
print(f"  Saved: _vdem_tmid_vs_modernization.png")

# ═══════════════════════════════════════════════════════
#  13. Summary
# ═══════════════════════════════════════════════════════

print("\n" + "═" * 70)
print("  SUMMARY")
print("═" * 70)

print("\n  Test 1 — Residual ~ indicator (overall):")
for key in AVAIL:
    if key in test1_results:
        t = test1_results[key]
        sig = '***' if t['p'] < 0.001 else '**' if t['p'] < 0.01 else '*' if t['p'] < 0.05 else 'ns'
        print(f"    {key:>10}: r={t['r']:+.4f} {sig:>4}  (n={t['n']})")

print("\n  Test 2 — Within-country mean r:")
for key in AVAIL:
    if key in within_results:
        cdf = within_results[key]
        pct_pos = (cdf['r'] > 0).mean() * 100
        print(f"    {key:>10}: mean_r={cdf['r'].mean():+.4f}, {pct_pos:.0f}% positive ({len(cdf)} countries)")

print("\n  Test 3 — t_mid ~ indicator (cross-section):")
for key in ['urban', 'industry', 'services', 'educ']:
    col = f'{key}_at_tmid'
    if col not in cs.columns:
        continue
    sub = cs[['ric_tmid', col]].dropna()
    if len(sub) >= 20:
        r, p = stats.pearsonr(sub['ric_tmid'], sub[col])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        print(f"    t_mid ~ {key:>10}: r={r:+.4f} {sig:>4}")

print()
