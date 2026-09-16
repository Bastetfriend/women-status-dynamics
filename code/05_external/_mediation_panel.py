# -*- coding: utf-8 -*-
"""
正式中介分析：面板固定效应 + 时间趋势
水平值，不差分，不用残差
国家固定效应 (within estimator) + 时间趋势控制common trend
Cluster SE by country
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from scipy import stats
import glob as globmod, os, warnings
warnings.filterwarnings('ignore')

BASE = r'data'

# ═══ Load ═══
wpei = pd.read_csv(f'{BASE}/women-political-empowerment-index.csv')
wpei.columns = ['entity', 'code', 'year', 'wpei', 'region']
wpei['S'] = wpei['wpei'] - 1

edi_raw = pd.read_csv(f'{BASE}/electoral-democracy-index.csv')
edi_raw.columns = ['entity', 'code', 'year', 'edi', 'region']
edi_latest = edi_raw.sort_values('year').groupby('code').last()[['edi']].rename(
    columns={'edi': 'edi_latest'})

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

econ_dfs = {}
pf_cache = f'{BASE}/_wb_indicators_cache.csv'
if os.path.exists(pf_cache):
    pf_raw = pd.read_csv(pf_cache)
    owid_e2c = {}
    for _, row in wpei.drop_duplicates('code').iterrows():
        owid_e2c[row['entity'].strip().lower()] = row['code']
    MANUAL = {
        'Korea, Rep.': 'KOR', "Korea, Dem. People's Rep.": 'PRK',
        'Russian Federation': 'RUS', 'Iran, Islamic Rep.': 'IRN',
        'Egypt, Arab Rep.': 'EGY', 'Venezuela, RB': 'VEN',
        'Turkiye': 'TUR', 'Lao PDR': 'LAO', 'Kyrgyz Republic': 'KGZ',
        'Slovak Republic': 'SVK', 'Czechia': 'CZE',
        'Congo, Rep.': 'COG', 'Congo, Dem. Rep.': 'COD',
        'Gambia, The': 'GMB', 'Yemen, Rep.': 'YEM',
        'Syrian Arab Republic': 'SYR', 'Eswatini': 'SWZ',
        'North Macedonia': 'MKD', 'Myanmar': 'MMR', "Cote d'Ivoire": 'CIV',
    }
    wb2owid = {}
    for _, row in pf_raw.drop_duplicates('wb_code').iterrows():
        e, c = row['entity'], row['wb_code']
        if e in MANUAL: wb2owid[c] = MANUAL[e]; continue
        k = e.strip().lower()
        if k in owid_e2c: wb2owid[c] = owid_e2c[k]; continue
        for ok, oc in owid_e2c.items():
            if k in ok or ok in k: wb2owid[c] = oc; break
    pf_raw['code'] = pf_raw['wb_code'].map(wb2owid)
    for pk in ['industry', 'services']:
        sub = pf_raw[pf_raw['indicator'] == pk].dropna(subset=['code'])
        econ_dfs[pk] = sub[['code', 'year', 'value']].rename(columns={'value': pk})

# ═══ Merge ═══
merged = wpei[['code', 'year', 'S', 'entity']].copy()
merged = merged.merge(fem_educ, on=['code', 'year'], how='left')
for key, edf in econ_dfs.items():
    merged = merged.merge(edf, on=['code', 'year'], how='left')
merged = merged.sort_values(['code', 'year'])

# ═══ Within estimator (country FE via demeaning) ═══
def demean(df, cols, group='code'):
    """Demean columns within groups = country fixed effects."""
    out = df.copy()
    for c in cols:
        out[c] = out[c] - out.groupby(group)[c].transform('mean')
    return out

# ═══ OLS with cluster SE ═══
def ols_cluster(y, X, clusters):
    """OLS with cluster-robust standard errors."""
    n, k = X.shape
    # Add intercept (should be ~0 after demeaning, but keep for correctness)
    X_aug = np.column_stack([np.ones(n), X])
    k_aug = X_aug.shape[1]

    # OLS
    beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    resid = y - X_aug @ beta
    sse = np.sum(resid ** 2)
    r2 = 1 - sse / np.sum((y - y.mean()) ** 2) if np.var(y) > 0 else 0

    # Cluster-robust variance (Cameron & Miller)
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    meat = np.zeros((k_aug, k_aug))
    for g in unique_clusters:
        idx = clusters == g
        X_g = X_aug[idx]
        e_g = resid[idx].reshape(-1, 1)
        score_g = (X_g * e_g).sum(axis=0).reshape(-1, 1)
        meat += score_g @ score_g.T

    bread = np.linalg.inv(X_aug.T @ X_aug)
    # Small-sample correction: G/(G-1) * (n-1)/(n-k)
    correction = (G / (G - 1)) * ((n - 1) / (n - k_aug))
    V = correction * bread @ meat @ bread
    se = np.sqrt(np.diag(V))

    # t-stats with G-1 df (cluster)
    t_stat = beta / se
    p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=G - 1))

    return {
        'beta': beta, 'se': se, 't': t_stat, 'p': p_val,
        'r2': r2, 'n': n, 'n_clusters': G,
        'resid': resid
    }

# ═══ Run panel mediation for each economic variable ═══
print('=' * 80)
print('  面板中介分析：水平值 + 国家固定效应 + 时间趋势')
print('  经济 → fem_educ → S(t)')
print('  Cluster-robust SE by country')
print('=' * 80)

for econ_var in ['services', 'industry']:
    # Prepare panel: drop rows missing any variable
    panel = merged[['code', 'year', 'S', 'fem_educ', econ_var]].dropna().copy()

    # Filter: keep countries with >= 8 observations
    counts = panel.groupby('code').size()
    valid_codes = counts[counts >= 8].index
    panel = panel[panel['code'].isin(valid_codes)].copy()

    n_countries = panel['code'].nunique()
    n_obs = len(panel)

    # Demean (within estimator = country FE)
    dm_cols = ['S', 'fem_educ', econ_var, 'year']
    panel_dm = demean(panel, dm_cols)

    y_S = panel_dm['S'].values
    y_educ = panel_dm['fem_educ'].values
    x_econ = panel_dm[econ_var].values
    x_year = panel_dm['year'].values
    clusters = panel['code'].values  # original codes for clustering

    print(f'\n{"═" * 70}')
    print(f'  {econ_var.upper()}  |  {n_countries} countries, {n_obs} obs')
    print(f'{"═" * 70}')

    # ─── Path a: econ → fem_educ (controlling year trend) ───
    X_a = np.column_stack([x_econ, x_year])
    res_a = ols_cluster(y_educ, X_a, clusters)
    a = res_a['beta'][1]  # econ coefficient (index 0=intercept, 1=econ, 2=year)
    se_a = res_a['se'][1]
    p_a = res_a['p'][1]

    print(f'\n  Path a: {econ_var} → fem_educ')
    print(f'    a = {a:+.6f}, SE = {se_a:.6f}, t = {res_a["t"][1]:+.3f}, '
          f'p = {p_a:.4f} {"***" if p_a<.001 else "**" if p_a<.01 else "*" if p_a<.05 else ""}')
    print(f'    year trend = {res_a["beta"][2]:+.6f}, p = {res_a["p"][2]:.4f}')
    print(f'    R² (within) = {res_a["r2"]:.4f}')

    # ─── Path c: econ → S (total effect, controlling year) ───
    X_c = np.column_stack([x_econ, x_year])
    res_c = ols_cluster(y_S, X_c, clusters)
    c = res_c['beta'][1]
    se_c = res_c['se'][1]
    p_c = res_c['p'][1]

    print(f'\n  Path c: {econ_var} → S (total)')
    print(f'    c = {c:+.6f}, SE = {se_c:.6f}, t = {res_c["t"][1]:+.3f}, '
          f'p = {p_c:.4f} {"***" if p_c<.001 else "**" if p_c<.01 else "*" if p_c<.05 else ""}')
    print(f'    year trend = {res_c["beta"][2]:+.6f}, p = {res_c["p"][2]:.4f}')
    print(f'    R² (within) = {res_c["r2"]:.4f}')

    # ─── Path c' + b: econ + educ → S (controlling year) ───
    X_cb = np.column_stack([x_econ, y_educ, x_year])
    res_cb = ols_cluster(y_S, X_cb, clusters)
    c_prime = res_cb['beta'][1]  # econ direct
    b = res_cb['beta'][2]        # educ → S controlling econ
    se_b = res_cb['se'][2]
    p_b = res_cb['p'][2]
    p_cprime = res_cb['p'][1]

    print(f'\n  Path c\' + b: {econ_var} + fem_educ → S')
    print(f'    c\' (direct) = {c_prime:+.6f}, SE = {res_cb["se"][1]:.6f}, '
          f'p = {p_cprime:.4f} {"***" if p_cprime<.001 else "**" if p_cprime<.01 else "*" if p_cprime<.05 else ""}')
    print(f'    b (educ→S)  = {b:+.6f}, SE = {se_b:.6f}, '
          f'p = {p_b:.4f} {"***" if p_b<.001 else "**" if p_b<.01 else "*" if p_b<.05 else ""}')
    print(f'    year trend  = {res_cb["beta"][3]:+.6f}, p = {res_cb["p"][3]:.4f}')
    print(f'    R² (within) = {res_cb["r2"]:.4f}')

    # ─── Indirect effect ───
    indirect = a * b
    sobel_se = np.sqrt(b**2 * se_a**2 + a**2 * se_b**2)
    sobel_z = indirect / sobel_se if sobel_se > 0 else 0
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
    prop_med = indirect / c if abs(c) > 1e-10 else float('nan')

    print(f'\n  ─── Mediation summary ───')
    print(f'    a × b (indirect) = {indirect:+.6f}')
    print(f'    c (total)        = {c:+.6f}')
    print(f'    c\' (direct)      = {c_prime:+.6f}')
    print(f'    c - c\'           = {c - c_prime:+.6f}')
    print(f'    Sobel z = {sobel_z:+.3f}, p = {sobel_p:.4f} '
          f'{"***" if sobel_p<.001 else "**" if sobel_p<.01 else "*" if sobel_p<.05 else ""}')
    if not np.isnan(prop_med):
        print(f'    % mediated = {prop_med:.1%}')

    # ─── Per-country indirect effect (for xlsx ranking) ───
    country_rows = []
    for code, grp in panel.groupby('code'):
        if len(grp) < 8:
            continue
        entity = merged[merged['code'] == code]['entity'].iloc[0]
        g = grp.copy()
        for col in ['S', 'fem_educ', econ_var, 'year']:
            g[col] = g[col] - g[col].mean()

        y_s = g['S'].values
        y_e = g['fem_educ'].values
        x_ec = g[econ_var].values
        x_yr = g['year'].values

        try:
            # Path a
            Xa = np.column_stack([np.ones(len(g)), x_ec, x_yr])
            ba = np.linalg.lstsq(Xa, y_e, rcond=None)[0]
            # Path c
            Xc = np.column_stack([np.ones(len(g)), x_ec, x_yr])
            bc = np.linalg.lstsq(Xc, y_s, rcond=None)[0]
            # Path c' + b
            Xcb = np.column_stack([np.ones(len(g)), x_ec, y_e, x_yr])
            bcb = np.linalg.lstsq(Xcb, y_s, rcond=None)[0]

            a_i = ba[1]
            c_i = bc[1]
            cprime_i = bcb[1]
            b_i = bcb[2]
            indirect_i = a_i * b_i
            prop_i = indirect_i / c_i if abs(c_i) > 1e-10 else np.nan

            country_rows.append({
                'code': code, 'entity': entity, 'n': len(grp),
                'a': a_i, 'b': b_i, 'c': c_i, 'c_prime': cprime_i,
                'indirect': indirect_i, 'prop_mediated': prop_i,
            })
        except:
            pass

    cdf = pd.DataFrame(country_rows)
    cdf = cdf.merge(edi_latest, on='code', how='left')
    cdf = cdf.sort_values('indirect', ascending=False)

    # Save xlsx
    out_path = f'{BASE}/_mediation_panel_{econ_var}.xlsx'
    cdf.to_excel(out_path, index=False, float_format='%.4f')
    print(f'\n  Per-country xlsx: {out_path} ({len(cdf)} countries)')

    # Top/bottom 10
    print(f'\n  ═══ Indirect 最强 10 国 ═══')
    for i, (_, r) in enumerate(cdf.head(10).iterrows(), 1):
        edi = f'{r["edi_latest"]:.3f}' if pd.notna(r['edi_latest']) else 'N/A'
        print(f'  {i:>2} {r["code"]:<6} {str(r["entity"])[:22]:<22} '
              f'indirect={r["indirect"]:+.5f} a={r["a"]:+.4f} b={r["b"]:+.4f} '
              f'c={r["c"]:+.5f} EDI={edi}')

    print(f'\n  ═══ Indirect 最弱 10 国 ═══')
    for i, (_, r) in enumerate(cdf.tail(10).iterrows(), 1):
        edi = f'{r["edi_latest"]:.3f}' if pd.notna(r['edi_latest']) else 'N/A'
        print(f'  {i:>2} {r["code"]:<6} {str(r["entity"])[:22]:<22} '
              f'indirect={r["indirect"]:+.5f} a={r["a"]:+.4f} b={r["b"]:+.4f} '
              f'c={r["c"]:+.5f} EDI={edi}')

    # EDI correlation with indirect
    valid_e = cdf.dropna(subset=['edi_latest', 'indirect'])
    if len(valid_e) > 10:
        re, pe = stats.pearsonr(valid_e['edi_latest'], valid_e['indirect'])
        print(f'\n  EDI vs country indirect: r={re:+.4f}, p={pe:.4f}')

print('\n  Done.')
