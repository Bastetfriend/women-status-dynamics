# -*- coding: utf-8 -*-
"""Estimate, from the INDEPENDENT marriage-age series alone, whether the RISE in
   female age-at-first-marriage decelerates within the 1960-2019 window.
   Rationale (load-bearing, to be vetted by Xie): if S(t) and marriage age m(t)
   share the damped oscillation, dm/dt is proportional to the current amplitude A(t);
   A(t) ~ A0 then implies the rise is steady (not concave) within the window.
   Per country (>=MIN points): fit linear vs quadratic in centered time, compare by
   AIC, and test the quadratic curvature c (OLS t-stat). Classify each country as
   supporting A~A0 (no significant deceleration) or a proximate-rejection signal
   (significant concave deceleration). No paper/docx touched; prints only. (2026-06-30)"""
import sys, csv
import numpy as np
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

SMAM = r"data\_smam_female_worldbank.csv"

# ---- load marriage-age series ----
by_c = defaultdict(list)  # code -> list of (year, value)
with open(SMAM, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        by_c[row['code']].append((int(row['year']), float(row['smam_fe'])))
for c in by_c:
    by_c[c].sort()

def fit_with_se(t, y, deg):
    """OLS polynomial fit; return (coeffs, AIC, se_top) where se_top is the SE of
       the highest-order coefficient. t is assumed already centered."""
    n = len(y)
    X = np.vander(t, deg + 1, increasing=True)  # cols: 1, t, t^2, ...
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    k = deg + 1
    # guard rss=0
    rss = max(rss, 1e-12)
    aic = n * np.log(rss / n) + 2 * k
    # SE of top coef
    dof = n - k
    if dof > 0:
        sigma2 = rss / dof
        XtX_inv = np.linalg.inv(X.T @ X)
        se_top = float(np.sqrt(sigma2 * XtX_inv[-1, -1]))
    else:
        se_top = float('inf')
    return beta, aic, se_top

def analyze(MIN):
    support, reject, accel, flat_or_decl, total = [], [], [], [], 0
    rows = []
    for code, series in by_c.items():
        if len(series) < MIN:
            continue
        years = np.array([yr for yr, _ in series], float)
        y = np.array([v for _, v in series], float)
        tc = years - years.mean()  # center
        # linear
        bl, aic_lin, se_b = fit_with_se(tc, y, 1)
        b = bl[1]                      # rise rate
        t_b = b / se_b if se_b > 0 else 0.0
        # quadratic
        bq, aic_quad, se_c = fit_with_se(tc, y, 2)
        c = bq[2]                      # curvature
        t_c = c / se_c if se_c not in (0, float('inf')) else 0.0
        total += 1
        rising = (b > 0 and t_b > 2)   # marriage age genuinely rising
        quad_wins = (aic_quad < aic_lin - 2)
        sig_concave = (c < 0 and t_c < -2)   # significant deceleration
        sig_convex = (c > 0 and t_c > 2)     # significant acceleration
        if sig_concave and quad_wins:
            cls = 'REJECT(decel)'
            reject.append(code)
        elif rising:
            cls = 'SUPPORT(steady/accel)'
            support.append(code)
            if sig_convex:
                accel.append(code)
        else:
            cls = 'flat/declining'
            flat_or_decl.append(code)
        rows.append((code, len(series), round(b, 4), round(t_b, 1),
                     round(c, 5), round(t_c, 1), cls))
    return support, reject, accel, flat_or_decl, total, rows

for MIN in (8, 5, 12):
    sup, rej, acc, flat, tot, rows = analyze(MIN)
    print(f"\n========== MIN={MIN} time points  (N={tot} countries) ==========")
    print(f"  SUPPORT A~A0 (rising, not decelerating): {len(sup)} "
          f"({100*len(sup)/tot:.1f}%)  [of which significantly ACCELerating: {len(acc)}]")
    print(f"  REJECT signal (rising but significantly DECELerating, quad wins): "
          f"{len(rej)} ({100*len(rej)/tot:.1f}%)")
    print(f"  flat / declining marriage age: {len(flat)} ({100*len(flat)/tot:.1f}%)")
    if rej:
        print(f"  deceleration countries: {sorted(rej)}")

# detailed table for the main MIN=8 cut
print("\n\n===== per-country detail (MIN=8): code, n, b(rate/yr), t_b, c(curv), t_c, class =====")
sup, rej, acc, flat, tot, rows = analyze(8)
for r in sorted(rows, key=lambda x: x[0]):
    print(f"  {r[0]}  n={r[1]:2d}  b={r[2]:+.4f} (t={r[3]:+.1f})  "
          f"c={r[4]:+.5f} (t={r[5]:+.1f})  {r[6]}")

# pooled: stack all >=8-pt countries, demeaned per country, common quadratic in centered t
print("\n===== POOLED curvature (all MIN=8 countries, per-country demeaned) =====")
T, Y = [], []
for code, series in by_c.items():
    if len(series) < 8:
        continue
    years = np.array([yr for yr, _ in series], float)
    y = np.array([v for _, v in series], float)
    tc = years - years.mean()
    yc = y - y.mean()
    T.append(tc); Y.append(yc)
T = np.concatenate(T); Y = np.concatenate(Y)
Xp = np.vander(T, 3, increasing=True)
bp, *_ = np.linalg.lstsq(Xp, Y, rcond=None)
resid = Y - Xp @ bp
dof = len(Y) - 3
sigma2 = float(resid @ resid) / dof
se = np.sqrt(np.diag(sigma2 * np.linalg.inv(Xp.T @ Xp)))
print(f"  pooled rise rate  b = {bp[1]:+.4f} yr/yr  (t = {bp[1]/se[1]:+.1f})")
print(f"  pooled curvature  c = {bp[2]:+.6f}        (t = {bp[2]/se[2]:+.1f})")
print("  (c not significantly < 0  => no pooled deceleration => independent support for A~A0)")
