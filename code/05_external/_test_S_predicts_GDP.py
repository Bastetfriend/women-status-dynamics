# -*- coding: utf-8 -*-
"""Reverse causal test v2: match by entity name, then run S→GDP prediction."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
from scipy import stats

# ---- Load ----
panel = pd.read_csv(r'data\_panel_S_MAC.csv')
diag = pd.read_csv(r'data\_richards_diagnostics.csv')
gdp = pd.read_csv(r'data\_gdp_growth_worldbank.csv')

# Build ISO3→entity mapping from diagnostics
code_to_entity = dict(zip(diag['code'], diag['entity']))

# Build entity→gdp mapping: standardize names
def clean_name(s):
    s = str(s).strip().lower()
    s = s.replace(', the','').replace('the ','')
    s = s.replace('dem. rep.', 'democratic republic').replace('rep.', 'republic')
    s = s.replace('dem. people\'s rep.', 'democratic people\'s republic')
    s = s.replace('st. ', 'saint ').replace('sao tome and principe', 'sao tome & principe')
    return s

gdp_entities = gdp[['entity']].drop_duplicates()
gdp_entities['clean'] = gdp_entities['entity'].apply(clean_name)
gdp['clean_entity'] = gdp['entity'].apply(clean_name)

# Match panel codes to GDP via entity name
diag_ents = diag[['code','entity']].drop_duplicates()
diag_ents['clean'] = diag_ents['entity'].apply(clean_name)

# Join
matched = diag_ents.merge(gdp_entities, on='clean', how='inner')
print('Matched entities: {} / {} diag codes -> {} GDP rows'.format(
    len(matched), len(diag_ents), len(matched)))

# Build code→gdp_entity mapping
code_to_gdp_entity = dict(zip(matched['code'], matched['entity_y']))

# Now merge panel with GDP
panel_with_entity = panel.copy()
panel_with_entity['gdp_entity'] = panel_with_entity['code'].map(code_to_gdp_entity)
panel_with_entity = panel_with_entity.dropna(subset=['gdp_entity'])

merged = panel_with_entity.merge(
    gdp[['entity','year','gdp_growth']],
    left_on=['gdp_entity','year'], right_on=['entity','year'], how='inner'
)
# Drop the duplicate entity column from gdp side
merged = merged.drop(columns=['entity'])
print('Merged panel+GDP: {} rows, {} unique codes'.format(len(merged), merged['code'].nunique()))

# ---- Richards ----
diag_params = {}
for _, row in diag.iterrows():
    if not any(pd.isna(row[c]) for c in ['S_floor','S_ceil','k','t_mid','v']):
        diag_params[row['code']] = {
            'S_floor': row['S_floor'], 'S_ceil': row['S_ceil'],
            'k': row['k'], 't_mid': row['t_mid'], 'v': row['v']
        }

def richards(t, p):
    vc = max(p['v'], 0.01)
    z = vc * np.exp(-p['k'] * (t - p['t_mid']))
    z = np.clip(z, 1e-10, 1e10)
    base = np.clip(1.0 + z, 1e-10, 1e10)
    return p['S_floor'] + (p['S_ceil'] - p['S_floor']) / (base ** (1.0 / vc))

# Compute S_hat and diffs
rows = []
for code, grp in merged.groupby('code'):
    if code not in diag_params:
        continue
    grp = grp.sort_values('year').copy()
    p = diag_params[code]
    grp['S_hat'] = richards(grp['year'].values, p)
    grp['dS_hat'] = np.diff(grp['S_hat'], prepend=np.nan)
    grp['dgdp'] = np.diff(grp['gdp_growth'], prepend=np.nan)
    rows.append(grp)

df = pd.concat(rows, ignore_index=True)
df = df.dropna(subset=['S_hat','dS_hat','gdp_growth'])
print('Final: {} rows, {} countries'.format(len(df), df['code'].nunique()))

# ===============================================================
# TEST 1: Level — Ŝ(t) → gdp_growth
# ===============================================================
print('\n' + '='*60)
print('TEST 1: gdp_growth ~ S_hat  (level)')
print('='*60)

level_r2s = []
for code, grp in df.groupby('code'):
    if len(grp) < 10: continue
    if np.std(grp['S_hat']) < 1e-10: continue
    slope, intercept, r, p, se = stats.linregress(grp['S_hat'], grp['gdp_growth'])
    level_r2s.append({'code':code,'n':len(grp),'r2':r**2,'slope':slope,'p':p})

lr = pd.DataFrame(level_r2s)
print('N >= 10yr: {}'.format(len(lr)))
print('  median R² = {:.4f}'.format(lr['r2'].median()))
print('  mean   R² = {:.4f}'.format(lr['r2'].mean()))
print('  R²<0.05: {}/{} ({:.1f}%)'.format((lr['r2']<0.05).sum(), len(lr), (lr['r2']<0.05).sum()/len(lr)*100))
print('  p<0.05:  {}/{} ({:.1f}%)'.format((lr['p']<0.05).sum(), len(lr), (lr['p']<0.05).sum()/len(lr)*100))

sl, il, rl, pl, sel = stats.linregress(df['S_hat'], df['gdp_growth'])
print('  Pooled: slope={:.4f}  R²={:.4f}  p={:.4f}'.format(sl, rl**2, pl))

# ===============================================================
# TEST 2: ΔŜ → gdp_growth
# ===============================================================
print('\n' + '='*60)
print('TEST 2: gdp_growth ~ dS_hat  (first-diff S)')
print('='*60)

diff_r2s = []
for code, grp in df.groupby('code'):
    g = grp.dropna(subset=['dS_hat'])
    if len(g) < 10: continue
    if np.std(g['dS_hat']) < 1e-10: continue
    slope, intercept, r, p, se = stats.linregress(g['dS_hat'], g['gdp_growth'])
    diff_r2s.append({'code':code,'n':len(g),'r2':r**2,'slope':slope,'p':p})

dr = pd.DataFrame(diff_r2s)
print('N >= 10yr: {}'.format(len(dr)))
print('  median R² = {:.4f}'.format(dr['r2'].median()))
print('  mean   R² = {:.4f}'.format(dr['r2'].mean()))
print('  R²<0.05: {}/{} ({:.1f}%)'.format((dr['r2']<0.05).sum(), len(dr), (dr['r2']<0.05).sum()/len(dr)*100))
print('  p<0.05:  {}/{} ({:.1f}%)'.format((dr['p']<0.05).sum(), len(dr), (dr['p']<0.05).sum()/len(dr)*100))

gcl = df.dropna(subset=['dS_hat'])
sd, id_, rd, pd_, sed = stats.linregress(gcl['dS_hat'], gcl['gdp_growth'])
print('  Pooled: slope={:.4f}  R²={:.4f}  p={:.4f}'.format(sd, rd**2, pd_))

# ===============================================================
# BONUS: ΔŜ → Δ(gdp_growth)
# ===============================================================
print('\n' + '='*60)
print('BONUS: dgdp_growth ~ dS_hat  (double-diff)')
print('='*60)

dd_r2s = []
for code, grp in df.groupby('code'):
    g = grp.dropna(subset=['dS_hat','dgdp'])
    if len(g) < 10: continue
    if np.std(g['dS_hat']) < 1e-10: continue
    slope, intercept, r, p, se = stats.linregress(g['dS_hat'], g['dgdp'])
    dd_r2s.append({'code':code,'n':len(g),'r2':r**2,'slope':slope,'p':p})

ddr = pd.DataFrame(dd_r2s)
print('N >= 10yr: {}'.format(len(ddr)))
print('  median R² = {:.4f}'.format(ddr['r2'].median()))
print('  mean   R² = {:.4f}'.format(ddr['r2'].mean()))
print('  R²<0.05: {}/{} ({:.1f}%)'.format((ddr['r2']<0.05).sum(), len(ddr), (ddr['r2']<0.05).sum()/len(ddr)*100))
print('  p<0.05:  {}/{} ({:.1f}%)'.format((ddr['p']<0.05).sum(), len(ddr), (ddr['p']<0.05).sum()/len(ddr)*100))

gdd = df.dropna(subset=['dS_hat','dgdp'])
sd2, id2, rd2, pd2, se2 = stats.linregress(gdd['dS_hat'], gdd['dgdp'])
print('  Pooled: slope={:.4f}  R²={:.4f}  p={:.4f}'.format(sd2, rd2**2, pd2))

# ===============================================================
print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print('  Level         (S_hat → gdp_growth):     median R² = {:.4f}  mean R² = {:.4f}'.format(lr['r2'].median(), lr['r2'].mean()))
print('  First-diff    (dS_hat → gdp_growth):     median R² = {:.4f}  mean R² = {:.4f}'.format(dr['r2'].median(), dr['r2'].mean()))
print('  Double-diff   (dS_hat → dgdp_growth):    median R² = {:.4f}  mean R² = {:.4f}'.format(ddr['r2'].median(), ddr['r2'].mean()))
