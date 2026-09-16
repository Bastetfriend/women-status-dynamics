# -*- coding: utf-8 -*-
"""
Bootstrap mediation: ln(GDP) -> fem_educ -> S(t)
Panel FE + time trend + block bootstrap by country (Preacher & Hayes approach)
Replaces Sobel test with bootstrap 95% CI for indirect effect
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from scipy import stats
import glob as globmod, os, warnings
warnings.filterwarnings('ignore')

BASE = r'data'
N_BOOT = 5000
SEED = 42

# ═══ Load data (same as _mediation_panel_cn.py) ═══
wpei = pd.read_csv(f'{BASE}/women-political-empowerment-index.csv')
wpei.columns = ['entity', 'code', 'year', 'wpei', 'region']
wpei['S'] = wpei['wpei'] - 1

def load_wb(pattern, key):
    files = [f for f in globmod.glob(f'{BASE}/{pattern}*.csv') if 'Metadata' not in f]
    if not files: return pd.DataFrame()
    df = pd.read_csv(files[0], skiprows=4)
    id_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
    ycols = [c for c in df.columns if c not in id_cols and c.strip().isdigit()]
    long = df.melt(id_vars=['Country Code'], value_vars=ycols,
                   var_name='year', value_name=key)
    long = long.rename(columns={'Country Code': 'code'})
    long['year'] = long['year'].astype(int)
    long[key] = pd.to_numeric(long[key], errors='coerce')
    return long.dropna(subset=[key])[['code', 'year', key]]

fem_educ = load_wb('API_SE.SEC.ENRR.FE', 'fem_educ')
gdp_raw = load_wb('API_NY.GDP.MKTP.CD', 'gdp')

# ═══ Merge ═══
merged = wpei[['code', 'year', 'S', 'entity']].copy()
merged = merged.merge(fem_educ, on=['code', 'year'], how='left')
merged = merged.merge(gdp_raw, on=['code', 'year'], how='left')
merged['ln_gdp'] = np.log(merged['gdp'].clip(lower=1))
merged.loc[merged['gdp'].isna(), 'ln_gdp'] = np.nan
merged = merged.sort_values(['code', 'year'])

# ═══ Panel setup ═══
panel = merged[['code', 'year', 'S', 'fem_educ', 'ln_gdp', 'entity']].dropna().copy()
counts = panel.groupby('code').size()
panel = panel[panel['code'].isin(counts[counts >= 8].index)].copy()

n_countries = panel['code'].nunique()
n_obs = len(panel)
country_codes = panel['code'].unique()

print('=' * 70)
print('  Bootstrap mediation: ln(GDP) -> fem_educ -> S(t)')
print(f'  Panel FE + time trend, block bootstrap by country')
print(f'  {n_countries} countries, {n_obs} obs, {N_BOOT} bootstrap iterations')
print('=' * 70)

# ═══ Helper functions ═══
def demean(df, cols, group='code'):
    out = df.copy()
    for c in cols:
        out[c] = out[c] - out.groupby(group)[c].transform('mean')
    return out

def ols_cluster(y, X, clusters):
    n = len(y)
    X_aug = np.column_stack([np.ones(n), X])
    k = X_aug.shape[1]
    beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    resid = y - X_aug @ beta
    sse = np.sum(resid ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - sse / sst if sst > 0 else 0
    unique_c = np.unique(clusters)
    G = len(unique_c)
    meat = np.zeros((k, k))
    for g in unique_c:
        idx = clusters == g
        X_g = X_aug[idx]
        e_g = resid[idx].reshape(-1, 1)
        score = (X_g * e_g).sum(axis=0).reshape(-1, 1)
        meat += score @ score.T
    bread = np.linalg.inv(X_aug.T @ X_aug)
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    V = correction * bread @ meat @ bread
    se = np.sqrt(np.diag(V))
    t_stat = beta / se
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=G - 1))
    return {'beta': beta, 'se': se, 't': t_stat, 'p': p_val, 'r2': r2, 'n': n, 'G': G}

def run_mediation(data):
    """Run full mediation analysis on a panel dataset. Returns (a, b, indirect, c)."""
    dm = demean(data, ['S', 'fem_educ', 'ln_gdp', 'year'])
    y_S = dm['S'].values
    y_educ = dm['fem_educ'].values
    x_gdp = dm['ln_gdp'].values
    x_year = dm['year'].values
    clusters = data['code'].values

    # Path a: GDP -> educ
    res_a = ols_cluster(y_educ, np.column_stack([x_gdp, x_year]), clusters)
    a = res_a['beta'][1]
    se_a = res_a['se'][1]
    p_a = res_a['p'][1]

    # Path c: GDP -> S (total)
    res_c = ols_cluster(y_S, np.column_stack([x_gdp, x_year]), clusters)
    c = res_c['beta'][1]

    # Path c' + b: GDP + educ -> S
    res_cb = ols_cluster(y_S, np.column_stack([x_gdp, y_educ, x_year]), clusters)
    c_prime = res_cb['beta'][1]
    b = res_cb['beta'][2]
    se_b = res_cb['se'][2]
    p_b = res_cb['p'][2]

    indirect = a * b
    return a, b, indirect, c, se_a, se_b, p_a, p_b, c_prime, res_a, res_c, res_cb

# ═══ Original analysis (verify matches paper) ═══
a, b, indirect, c, se_a, se_b, p_a, p_b, c_prime, res_a, res_c, res_cb = run_mediation(panel)

sig = lambda p: '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else '' if p<.1 else ''

sobel_se = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
sobel_z = indirect / sobel_se if sobel_se > 0 else 0
sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
prop = indirect / c if abs(c) > 1e-10 else float('nan')

print(f'\n  === Original (Sobel) results ===')
print(f'  Path a: ln(GDP)->educ   a={a:+.4f}, p={p_a:.4f}')
print(f'  Path b: educ->S         b={b:+.6f}, p={p_b:.4f}')
print(f'  Path c: total           c={c:+.6f}')
print(f'  Path c\': direct         c\'={c_prime:+.6f}')
print(f'  Indirect a*b = {indirect:+.6f}')
print(f'  Sobel z = {sobel_z:+.3f}, p = {sobel_p:.4f}')
print(f'  % mediated = {prop:.1%}')

# ═══ Block bootstrap ═══
print(f'\n  Running {N_BOOT} block bootstrap iterations...')
rng = np.random.RandomState(SEED)

# Pre-index: store each country's data as a dict for fast resampling
country_data = {}
for code in country_codes:
    country_data[code] = panel[panel['code'] == code].copy()

boot_indirect = np.zeros(N_BOOT)
boot_a = np.zeros(N_BOOT)
boot_b = np.zeros(N_BOOT)
n_fail = 0

for i in range(N_BOOT):
    # Resample countries with replacement
    boot_codes = rng.choice(country_codes, size=len(country_codes), replace=True)

    # Build bootstrap panel (rename codes to avoid demeaning issues with duplicates)
    frames = []
    for j, code in enumerate(boot_codes):
        df_c = country_data[code].copy()
        df_c['code'] = f'{code}_{j}'  # unique ID for demeaning
        frames.append(df_c)
    boot_panel = pd.concat(frames, ignore_index=True)

    try:
        dm = demean(boot_panel, ['S', 'fem_educ', 'ln_gdp', 'year'])
        y_S = dm['S'].values
        y_educ = dm['fem_educ'].values
        x_gdp = dm['ln_gdp'].values
        x_year = dm['year'].values
        clusters = boot_panel['code'].values

        # Path a
        X_a = np.column_stack([np.ones(len(y_educ)), x_gdp, x_year])
        beta_a = np.linalg.lstsq(X_a, y_educ, rcond=None)[0]
        a_i = beta_a[1]

        # Path b (controlling GDP and year)
        X_cb = np.column_stack([np.ones(len(y_S)), x_gdp, y_educ, x_year])
        beta_cb = np.linalg.lstsq(X_cb, y_S, rcond=None)[0]
        b_i = beta_cb[2]

        boot_indirect[i] = a_i * b_i
        boot_a[i] = a_i
        boot_b[i] = b_i
    except:
        boot_indirect[i] = np.nan
        n_fail += 1

    if (i + 1) % 1000 == 0:
        print(f'    {i+1}/{N_BOOT} done...')

# Remove failed iterations
valid = ~np.isnan(boot_indirect)
boot_indirect_valid = boot_indirect[valid]
n_valid = len(boot_indirect_valid)

print(f'\n  Bootstrap complete: {n_valid} valid / {n_fail} failed')

# ═══ Bootstrap CI ═══
# Percentile method (Preacher & Hayes standard)
ci_lo = np.percentile(boot_indirect_valid, 2.5)
ci_hi = np.percentile(boot_indirect_valid, 97.5)
boot_mean = np.mean(boot_indirect_valid)
boot_se = np.std(boot_indirect_valid, ddof=1)

# Bias-corrected (BC) CI
z0 = stats.norm.ppf(np.mean(boot_indirect_valid < indirect))
alpha_lo = stats.norm.cdf(2 * z0 + stats.norm.ppf(0.025))
alpha_hi = stats.norm.cdf(2 * z0 + stats.norm.ppf(0.975))
bc_ci_lo = np.percentile(boot_indirect_valid, alpha_lo * 100)
bc_ci_hi = np.percentile(boot_indirect_valid, alpha_hi * 100)

# Significance: CI does NOT include 0
sig_percentile = (ci_lo > 0) or (ci_hi < 0)
sig_bc = (bc_ci_lo > 0) or (bc_ci_hi < 0)

print(f'\n  ═══ Bootstrap results ({N_BOOT} iterations) ═══')
print(f'  Original indirect effect:   {indirect:+.6f}')
print(f'  Bootstrap mean:             {boot_mean:+.6f}')
print(f'  Bootstrap SE:               {boot_se:.6f}')
print(f'')
print(f'  Percentile 95% CI:          [{ci_lo:+.6f}, {ci_hi:+.6f}]')
print(f'    Contains 0? {"YES -> NOT significant" if not sig_percentile else "NO -> SIGNIFICANT"}')
print(f'')
print(f'  Bias-corrected 95% CI:      [{bc_ci_lo:+.6f}, {bc_ci_hi:+.6f}]')
print(f'    Contains 0? {"YES -> NOT significant" if not sig_bc else "NO -> SIGNIFICANT"}')
print(f'')
print(f'  % mediated:                 {prop:.1%}')

print(f'\n  ═══ Comparison: Sobel vs Bootstrap ═══')
print(f'  Sobel:     z={sobel_z:+.3f}, p={sobel_p:.4f}  {"SIG" if sobel_p < 0.05 else "NS"}')
print(f'  Bootstrap: 95% CI [{ci_lo:+.6f}, {ci_hi:+.6f}]  {"SIG" if sig_percentile else "NS"}')
print(f'  BC Boot:   95% CI [{bc_ci_lo:+.6f}, {bc_ci_hi:+.6f}]  {"SIG" if sig_bc else "NS"}')

print(f'\n  === Paper text suggestion ===')
if not sig_percentile:
    print(f'  "Bootstrap中介检验（Preacher & Hayes, 2008; 5000次block重抽样）：')
    print(f'   间接效应a × b = {indirect:+.6f}，95% CI [{ci_lo:+.6f}, {ci_hi:+.6f}]，')
    print(f'   置信区间包含零，中介效应不显著。中介比例{prop:.1%}。"')
else:
    print(f'  "Bootstrap中介检验（Preacher & Hayes, 2008; 5000次block重抽样）：')
    print(f'   间接效应a × b = {indirect:+.6f}，95% CI [{ci_lo:+.6f}, {ci_hi:+.6f}]，')
    print(f'   置信区间不包含零，中介效应显著。中介比例{prop:.1%}。"')

print('\n  Done.')
