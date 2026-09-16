# -*- coding: utf-8 -*-
"""Layer 2: disentangle within-cycle SATURATION from cross-cycle AMPLITUDE DAMPING.
   Using each country's Richards sigmoid for S(t) (from _richards_diagnostics.csv),
   compute the sigmoid PROGRESS p(year) for every marriage-age observation, keep only
   MID-CURVE years (p in [LO,HI]: away from the ceiling, where any deceleration points
   to damping rather than saturation), and re-estimate the rise curvature there.
   If the deceleration that showed up in Layer 1 vanishes mid-curve -> it was saturation
   (i), so A(t)~A0 holds; if it persists mid-curve -> genuine damping (ii), trouble.
   Read-only; prints only. (2026-06-30)"""
import sys, csv
import numpy as np
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

DIAG = r"data\_richards_diagnostics.csv"
SMAM = r"data\_smam_female_worldbank.csv"

# Layer-1 deceleration set (for the before/after comparison)
L1_DECEL = {'AUS','AUT','BEN','BGD','CAN','DOM','ECU','EGY','FIN','HKG','IDN','IRQ',
            'ISL','JOR','KWT','MAR','MMR','NLD','NOR','PER','SVN','SWE','ZAF','ZWE'}
NORDIC = {'FIN','SWE','NOR','ISL','DNK'}

# ---- sigmoid params per country (status ok only) ----
par = {}
with open(DIAG, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if row.get('status') != 'ok':
            continue
        try:
            fl, ce = float(row['S_floor']), float(row['S_ceil'])
            k, tm, v = float(row['k']), float(row['t_mid']), float(row['v'])
        except (TypeError, ValueError, KeyError):
            continue
        if v <= 0 or (ce - fl) == 0:
            continue
        par[row['code']] = (fl, ce, k, tm, v)
print(f"countries with usable Richards fit: {len(par)}")

def progress(t, k, tmid, v):
    """Richards progress p(t) in (0,1): 0=just launched, 1=at ceiling. Numerically safe."""
    z = np.clip(-k * (t - tmid), -700.0, 700.0)
    with np.errstate(over='ignore', invalid='ignore'):
        ve = v * np.exp(z)
    logbase = np.where(np.isinf(ve), z + np.log(v), np.log1p(ve))
    logp = np.clip(-(1.0 / v) * logbase, -700.0, 0.0)
    return np.exp(logp)

# ---- marriage-age series ----
by_c = defaultdict(list)
with open(SMAM, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        by_c[row['code']].append((int(row['year']), float(row['smam_fe'])))
for c in by_c:
    by_c[c].sort()

def fit_curv(t, y):
    """centered quadratic; return b(rate), t_b, c(curv), t_c, quad_wins_vs_linear."""
    tc = t - t.mean()
    n = len(y)
    def ols(deg):
        X = np.vander(tc, deg + 1, increasing=True)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        rss = max(float(resid @ resid), 1e-12)
        aic = n * np.log(rss / n) + 2 * (deg + 1)
        dof = n - (deg + 1)
        if dof > 0:
            se = np.sqrt((rss / dof) * np.diag(np.linalg.inv(X.T @ X)))
        else:
            se = np.full(deg + 1, np.inf)
        return beta, aic, se
    bl, aic_l, sel = ols(1)
    bq, aic_q, seq = ols(2)
    b = bl[1]; t_b = b / sel[1] if sel[1] > 0 else 0.0
    c = bq[2]; t_c = c / seq[2] if np.isfinite(seq[2]) and seq[2] > 0 else 0.0
    return b, t_b, c, t_c, (aic_q < aic_l - 2)

def run(LO, HI, MINSEG):
    sup = rej = flat = 0
    kept_countries = []
    decel_resolved = []   # L1-decel countries whose deceleration vanished mid-curve
    decel_persist = []    # L1-decel countries still decelerating mid-curve
    decel_dropped = []    # L1-decel countries with too few mid-curve points
    T_all, Y_all = [], []
    nordic_report = []
    for code, series in by_c.items():
        if code not in par:
            continue
        fl, ce, k, tm, v = par[code]
        yrs = np.array([yr for yr, _ in series], float)
        vals = np.array([vv for _, vv in series], float)
        p = progress(yrs, k, tm, v)
        m = (p >= LO) & (p <= HI)
        n_mid = int(m.sum())
        if code in NORDIC:
            nordic_report.append((code, len(series), n_mid,
                                  round(float(p.min()), 2), round(float(p.max()), 2)))
        if n_mid < MINSEG:
            if code in L1_DECEL:
                decel_dropped.append(code)
            continue
        ym, vm = yrs[m], vals[m]
        b, t_b, c, t_c, qw = fit_curv(ym, vm)
        kept_countries.append(code)
        # pooled accumulation (demeaned)
        T_all.append(ym - ym.mean()); Y_all.append(vm - vm.mean())
        rising = (b > 0 and t_b > 2)
        sig_concave = (c < 0 and t_c < -2)
        if sig_concave and qw:
            rej += 1
            if code in L1_DECEL:
                decel_persist.append(code)
        elif rising:
            sup += 1
            if code in L1_DECEL:
                decel_resolved.append(code)
        else:
            flat += 1
            if code in L1_DECEL:
                decel_resolved.append(code)  # no longer a significant decel
    tot = sup + rej + flat
    # pooled curvature
    if T_all:
        T = np.concatenate(T_all); Y = np.concatenate(Y_all)
        X = np.vander(T, 3, increasing=True)
        bp, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ bp
        dof = len(Y) - 3
        se = np.sqrt((float(resid @ resid) / dof) * np.diag(np.linalg.inv(X.T @ X)))
        pooled = (bp[1], bp[1] / se[1], bp[2], bp[2] / se[2])
    else:
        pooled = (float('nan'),) * 4
    return dict(LO=LO, HI=HI, MINSEG=MINSEG, tot=tot, sup=sup, rej=rej, flat=flat,
                pooled=pooled, resolved=sorted(decel_resolved),
                persist=sorted(decel_persist), dropped=sorted(decel_dropped),
                nordic=sorted(nordic_report))

print("\nLayer-1 had 24 deceleration countries; testing whether they survive MID-CURVE.\n")
for LO, HI, MINSEG in [(0.15, 0.75, 6), (0.10, 0.80, 6), (0.20, 0.70, 5)]:
    r = run(LO, HI, MINSEG)
    b, tb, c, tc = r['pooled']
    print(f"========== MID-CURVE p in [{LO},{HI}], >= {MINSEG} mid pts  "
          f"(N={r['tot']} countries) ==========")
    print(f"  SUPPORT A~A0 (rising, not decel): {r['sup']} "
          f"({100*r['sup']/r['tot']:.1f}%)" if r['tot'] else "  (no countries)")
    print(f"  REJECT (still decel mid-curve)  : {r['rej']} "
          f"({100*r['rej']/r['tot']:.1f}%)")
    print(f"  flat/declining                  : {r['flat']} "
          f"({100*r['flat']/r['tot']:.1f}%)")
    print(f"  POOLED mid-curve: rise b={b:+.4f} (t={tb:+.1f}), "
          f"curvature c={c:+.6f} (t={tc:+.1f})")
    print(f"  of Layer-1's 24 decel countries -> resolved(no longer decel): "
          f"{len(r['resolved'])}, still persist: {len(r['persist'])}, "
          f"dropped(too few mid pts): {len(r['dropped'])}")
    if r['persist']:
        print(f"    STILL DECELERATING mid-curve (genuine-damping candidates): {r['persist']}")
    print()

# Nordic close-up at the main cut
print("===== Nordic close-up (main cut p in [0.15,0.75]) =====")
print("  code: n_total_obs, n_mid_curve_pts, p_min, p_max")
r = run(0.15, 0.75, 6)
for code, ntot, nmid, pmin, pmax in r['nordic']:
    note = "(mostly past ceiling -> saturation)" if nmid < 6 else ""
    print(f"  {code}: {ntot} obs, {nmid} mid-curve, p={pmin}..{pmax}  {note}")
