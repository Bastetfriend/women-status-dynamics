# -*- coding: utf-8 -*-
"""Recompute EDI vs S_ceil statistics from raw data.
   Three samples: (A) all 191, (B) 63 with IDV, (C) the 68-country Table 4 sample.
   Report Spearman rho + OLS R^2 separately for each."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
from scipy import stats
import statsmodels.api as sm

BASE = r'data'
CSV = BASE + r'\实验输出_CSV'

# ═══ Load data ═══
# Richards S_ceil
rich = pd.read_csv(f'{CSV}/_richards_diagnostics.csv')
rich = rich[['code', 'entity', 'S_ceil']].dropna(subset=['S_ceil'])
print(f'Richards S_ceil: {len(rich)} countries')

# EDI (latest year)
edi_raw = pd.read_csv(f'{CSV}/electoral-democracy-index.csv')
edi_raw.columns = ['entity', 'code', 'year', 'edi', 'region']
edi_latest = edi_raw.sort_values('year').groupby('code').last()[['edi']].reset_index()
print(f'EDI: {len(edi_latest)} countries')

# Hofstede IDV
hof = pd.read_csv(f'{CSV}/6-dimensions-for-website-2015-08-16.csv', sep=';')
hof['idv'] = pd.to_numeric(hof['idv'].replace('#NULL!', np.nan), errors='coerce')
HOF2ISO = {
    'ARG':'ARG','AUL':'AUS','AUT':'AUT','BAN':'BGD','BEL':'BEL','BRA':'BRA','BUL':'BGR',
    'CAN':'CAN','CHL':'CHL','CHI':'CHN','COL':'COL','COS':'CRI','CRO':'HRV','CZE':'CZE',
    'DEN':'DNK','ECA':'ECU','SAL':'SLV','EST':'EST','FIN':'FIN','FRA':'FRA','GER':'DEU',
    'GBR':'GBR','GRE':'GRC','GUA':'GTM','HOK':'HKG','HUN':'HUN','IND':'IND','IDO':'IDN',
    'IRA':'IRN','IRE':'IRL','ISR':'ISR','ITA':'ITA','JAM':'JAM','JPN':'JPN','KOR':'KOR',
    'LAT':'LVA','LIT':'LTU','LUX':'LUX','MAL':'MYS','MLT':'MLT','MEX':'MEX','MOR':'MAR',
    'NET':'NLD','NZL':'NZL','NOR':'NOR','PAK':'PAK','PAN':'PAN','PER':'PER','PHI':'PHL',
    'POL':'POL','POR':'PRT','ROM':'ROU','RUS':'RUS','SER':'SRB','SIN':'SGP','SLK':'SVK',
    'SLV':'SVN','SAF':'ZAF','SPA':'ESP','SUR':'SUR','SWE':'SWE','SWI':'CHE','TAI':'TWN',
    'THA':'THA','TRI':'TTO','TUR':'TUR','USA':'USA','URU':'URY','VEN':'VEN','VIE':'VNM',
}
hof['iso3'] = hof['ctr'].map(HOF2ISO)
idv_df = hof.dropna(subset=['idv','iso3'])[['iso3','country','idv']].copy()
idv_df['idv'] = idv_df['idv'].astype(int)
print(f'IDV: {len(idv_df)} countries')

# ═══ Sample A: ALL 191 (EDI x Richards S_ceil, no IDV filter) ═══
mA = edi_latest.merge(rich, on='code', how='inner')
mA = mA.dropna(subset=['edi', 'S_ceil'])
rho_A, pA = stats.spearmanr(mA['edi'], mA['S_ceil'])
ols_A = sm.OLS(mA['S_ceil'], sm.add_constant(mA['edi'])).fit()

print(f'\n{"="*60}')
print(f'SAMPLE A: ALL {len(mA)} countries with EDI + Richards S_ceil')
print(f'{"="*60}')
print(f'Spearman ρ  = {rho_A:.4f}  (ρ² = {rho_A**2:.4f} = {rho_A**2*100:.1f}%)')
print(f'OLS R²      = {ols_A.rsquared:.4f}  ({ols_A.rsquared*100:.1f}%)')
print(f'OLS β_EDI   = {ols_A.params[1]:.4f}, p = {ols_A.pvalues[1]:.2e}')

# ═══ Sample B: 63 with IDV (IDV + EDI + Richards S_ceil) ═══
mB = idv_df.merge(edi_latest, left_on='iso3', right_on='code', how='inner')
mB = mB.merge(rich[['code','S_ceil']], left_on='iso3', right_on='code', how='inner')
mB = mB.dropna(subset=['idv','edi','S_ceil'])
rho_B, pB = stats.spearmanr(mB['edi'], mB['S_ceil'])
ols_B = sm.OLS(mB['S_ceil'], sm.add_constant(mB['edi'])).fit()

print(f'\n{"="*60}')
print(f'SAMPLE B: {len(mB)} countries with IDV + EDI + Richards S_ceil')
print(f'{"="*60}')
print(f'IDV median = {mB["idv"].median():.0f}')
print(f'EDI median = {mB["edi"].median():.4f}')
print(f'Spearman ρ  = {rho_B:.4f}  (ρ² = {rho_B**2:.4f} = {rho_B**2*100:.1f}%)')
print(f'OLS R²      = {ols_B.rsquared:.4f}  ({ols_B.rsquared*100:.1f}%)')
print(f'OLS β_EDI   = {ols_B.params[1]:.4f}, p = {ols_B.pvalues[1]:.2e}')

# ═══ Sample C: try to reproduce the 68-country Table 4 sample ═══
# The 68 comes from IDV + EDI + b (education→status slope) not from S_ceil.
# Let's check: merge IDV + EDI (no Richards filter) to see how many
mC_base = idv_df.merge(edi_latest, left_on='iso3', right_on='code', how='inner')
mC_base = mC_base.dropna(subset=['idv', 'edi'])
print(f'\n{"="*60}')
print(f'SAMPLE C (base): {len(mC_base)} countries with IDV + EDI (no S_ceil filter)')
print(f'{"="*60}')
print(f'IDV median = {mC_base["idv"].median():.0f}')
print(f'EDI median = {mC_base["edi"].median():.4f}')

# Now add S from latest WPEI (not Richards S_ceil)
wpei = pd.read_csv(f'{CSV}/women-political-empowerment-index.csv')
wpei.columns = ['entity', 'code', 'year', 'wpei', 'region']
wpei['S'] = wpei['wpei'] - 1
s_latest = wpei.sort_values('year').groupby('code').last()[['S']].reset_index()

mC = mC_base.merge(s_latest, left_on='iso3', right_on='code', how='inner')
mC = mC.dropna(subset=['S'])
rho_C, pC = stats.spearmanr(mC['edi'], mC['S'])
ols_C = sm.OLS(mC['S'], sm.add_constant(mC['edi'])).fit()

print(f'SAMPLE C: {len(mC)} countries with IDV + EDI + latest WPEI S')
print(f'Spearman ρ  = {rho_C:.4f}  (ρ² = {rho_C**2:.4f} = {rho_C**2*100:.1f}%)')
print(f'OLS R²      = {ols_C.rsquared:.4f}  ({ols_C.rsquared*100:.1f}%)')
print(f'OLS β_EDI   = {ols_C.params[1]:.4f}, p = {ols_C.pvalues[1]:.2e}')

# Also check: EDI vs latest WPEI S for all 191
mD = edi_latest.merge(s_latest, on='code', how='inner')
mD = mD.dropna(subset=['edi', 'S'])
rho_D, pD = stats.spearmanr(mD['edi'], mD['S'])
ols_D = sm.OLS(mD['S'], sm.add_constant(mD['edi'])).fit()
print(f'\n{"="*60}')
print(f'SAMPLE D: ALL {len(mD)} countries with EDI + latest WPEI S (no IDV filter)')
print(f'{"="*60}')
print(f'Spearman ρ  = {rho_D:.4f}  (ρ² = {rho_D**2:.4f} = {rho_D**2*100:.1f}%)')
print(f'OLS R²      = {ols_D.rsquared:.4f}  ({ols_D.rsquared*100:.1f}%)')
print(f'OLS β_EDI   = {ols_D.params[1]:.4f}, p = {ols_D.pvalues[1]:.2e}')

# ═══ Summary ═══
print(f'\n{"="*60}')
print('SUMMARY TABLE')
print(f'{"="*60}')
print(f'{"Sample":<30} {"n":>5} {"Spearman ρ":>12} {"ρ²":>8} {"OLS R²":>10}')
print(f'{"-"*65}')
for label, n, rho, r2 in [
    ('A: All 191, S_ceil (Richards)', len(mA), rho_A, ols_A.rsquared),
    ('B: 63 IDV, S_ceil (Richards)', len(mB), rho_B, ols_B.rsquared),
    ('C: IDV+EDI, S (latest WPEI)', len(mC), rho_C, ols_C.rsquared),
    ('D: All 191, S (latest WPEI)', len(mD), rho_D, ols_D.rsquared),
]:
    print(f'{label:<30} {n:>5} {rho:>12.4f} {rho**2:>7.3f} {r2:>10.4f}')

print(f'\nPaper reports:')
print(f'  Section 2.2:          rho ~ +0.44 (R2 ~ 21%)')
print(f'  Fig 4 legend:         R2 = 0.453')
print(f'  Methods (old):        EDI alone explains 45.3% (R2 = 0.453)')
