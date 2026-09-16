"""
Item: export the piecewise-linear BREAKPOINT YEAR for the 53 AIC winners.

_vdem_model_comparison.py fitted the two-segment piecewise model but only stored
pw_R2 / pw_AIC / pw_BIC / pw_RSS -- the breakpoint parameter tb was never written out.
This script re-runs exactly that fit (same function, same bounds, same 50-point grid)
for the 53 winner countries and exports the breakpoint in calendar years.

Grid + bounds + centering are copied verbatim from _vdem_model_comparison.py so the
recovered breakpoints are the same ones the AIC comparison selected.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

SRC = r"data\women-political-empowerment-index.csv"
CMP = r"data\_vdem_model_comparison.csv"
OUT = r"data\_pw_breakpoints.csv"

df = pd.read_csv(SRC)
df.columns = ['entity', 'code', 'year', 'wpei', 'region']
df['S'] = df['wpei'] - 1.0


def m_piecewise(t, a, b1, b2, tb):
    return a + b1 * t + b2 * np.maximum(0, t - tb)


# ── same sample filter as the model comparison (>=30 yrs, non-flat) ──
MIN_YEARS = 30
agg = df.groupby('code').agg(entity=('entity', 'first'), n=('year', 'count')).reset_index()
agg = agg[agg['n'] >= MIN_YEARS]
flat = set()
for _, info in agg.iterrows():
    sub = df[df['code'] == info['code']]
    if sub['S'].max() - sub['S'].min() < 0.05:
        flat.add(info['code'])
agg = agg[~agg['code'].isin(flat)]

cmp_df = pd.read_csv(CMP, encoding='utf-8-sig')
pw_codes = set(cmp_df[cmp_df['aic_winner'] == 'Piecewise']['code'])
print(f"piecewise winners in comparison csv: {len(pw_codes)}")

rows = []
for _, info in agg.iterrows():
    if info['code'] not in pw_codes:
        continue
    sub = df[df['code'] == info['code']].sort_values('year')
    t_raw = sub['year'].values.astype(float)
    s = sub['S'].values

    t0 = t_raw.mean()
    t = t_raw - t0

    best_rss, best_popt = np.inf, None
    for tb_try in np.linspace(t.min() + 5, t.max() - 5, 50):
        try:
            p0 = [s[0], (s[-1] - s[0]) / (t[-1] - t[0]), 0, tb_try]
            popt, _ = curve_fit(m_piecewise, t, s, p0=p0,
                                bounds=([-2, -0.1, -0.1, t.min()],
                                        [0.5, 0.1, 0.1, t.max()]),
                                maxfev=5000)
            rss = np.sum((s - m_piecewise(t, *popt)) ** 2)
            if rss < best_rss:
                best_rss, best_popt = rss, popt
        except Exception:
            pass

    if best_popt is None:
        continue

    tb_raw = best_popt[3] + t0          # undo the centering
    r2 = 1 - best_rss / np.sum((s - s.mean()) ** 2)
    rows.append({
        'code': info['code'], 'entity': info['entity'], 'n': len(t_raw),
        'year_min': int(t_raw.min()), 'year_max': int(t_raw.max()),
        'breakpoint': round(tb_raw, 1), 'pw_R2': round(r2, 4),
    })

out = pd.DataFrame(rows).sort_values('breakpoint')
out.to_csv(OUT, index=False, encoding='utf-8-sig')
print(f"\nwrote {len(out)} rows -> {OUT}\n")
print(out.to_string(index=False))
