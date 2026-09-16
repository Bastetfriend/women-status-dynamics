"""
计算每个 AIC winner 组的 ΔAIC median
ΔAIC = best_sigmoid_AIC − winner_model_AIC
"""
import pandas as pd
import numpy as np

results = pd.read_csv(r"data\_vdem_model_comparison.csv")

# best sigmoid AIC per country
results['sig_best_AIC'] = results[['log_AIC','ric_AIC']].min(axis=1)

model_aic_cols = {
    'Richards': 'ric_AIC',
    'Logistic': 'log_AIC',
    'Piecewise': 'pw_AIC',
    'Quadratic': 'quad_AIC',
    'Linear': 'lin_AIC'
}

k_vals = {'Linear': 2, 'Quadratic': 3, 'Piecewise': 4, 'Logistic': 4, 'Richards': 5}
order = ['Linear', 'Quadratic', 'Piecewise', 'Logistic', 'Richards']

print("=" * 70)
print("  DELTA-AIC BY WINNER GROUP")
print("  dAIC = best_sigmoid_AIC - winner_model_AIC")
print("  Negative = sigmoid better; Positive = winner better than sigmoid")
print("=" * 70)
print()

table_rows = {}
for winner, grp in results.groupby('aic_winner'):
    n = len(grp)
    pct = n / len(results) * 100
    model_col = model_aic_cols[winner]
    daic = grp['sig_best_AIC'] - grp[model_col]

    table_rows[winner] = {
        'n': n, 'pct': pct,
        'median': daic.median(), 'mean': daic.mean(),
        'q25': daic.quantile(0.25), 'q75': daic.quantile(0.75),
        'min': daic.min(), 'max': daic.max(),
    }

    print(f"  {winner} (n={n}, {pct:.1f}%):")
    print(f"    dAIC median = {daic.median():.1f}")
    print(f"    dAIC mean   = {daic.mean():.1f}")
    print(f"    dAIC [Q25, Q75] = [{daic.quantile(0.25):.1f}, {daic.quantile(0.75):.1f}]")
    print(f"    dAIC min/max = [{daic.min():.1f}, {daic.max():.1f}]")
    print()

# Piecewise winners: how decisive?
pw = results[results['aic_winner'] == 'Piecewise']
if len(pw) > 0:
    daic_pw = pw['sig_best_AIC'] - pw['pw_AIC']
    print("=" * 70)
    print("  PIECEWISE WINNERS: HOW DECISIVE?")
    print("=" * 70)
    for thr in [2, 5, 10, 20]:
        n_close = (daic_pw < thr).sum()
        print(f"    Sigmoid within {thr} AIC: {n_close}/{len(pw)} ({n_close/len(pw)*100:.1f}%)")
    print()

# Paper-ready table
print("=" * 70)
print("  PAPER TABLE")
print("=" * 70)
header = f"{'Model':<12} {'k':>2} {'AIC wins':>9} {'%':>6} {'dAIC med':>9} {'IQR':>18}"
print(header)
print("-" * len(header))
for model in order:
    k = k_vals[model]
    if model in table_rows:
        r = table_rows[model]
        iqr = f"[{r['q25']:.1f}, {r['q75']:.1f}]"
        print(f"{model:<12} {k:>2} {r['n']:>9} {r['pct']:>5.1f}% {r['median']:>9.1f} {iqr:>18}")
    else:
        print(f"{model:<12} {k:>2}         0   0.0%       n/a                n/a")
