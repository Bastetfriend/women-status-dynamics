# -*- coding: utf-8 -*-
"""Conservative estimate: sigmoid dominance when piecewise-linear is excluded.
   Best sigmoid (Richards|Logistic) vs. best unstructured (Linear|Quadratic|Quartic|Spline).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

# Load both comparison files
m5 = pd.read_csv(r'data\_vdem_model_comparison.csv')
k5 = pd.read_csv(r'data\_vdem_5param_comparison.csv')

# Merge on code
df = m5.merge(k5[['code', 'quartic_AIC', 'spline_AIC']], on='code', how='left')

# Best sigmoid (piecewise excluded from both sides)
df['best_sigmoid_AIC'] = df[['ric_AIC', 'log_AIC']].min(axis=1)

# Best unstructured (no piecewise, no sigmoid): linear, quadratic, quartic, spline
df['best_unstructured_AIC'] = df[['lin_AIC', 'quad_AIC', 'quartic_AIC', 'spline_AIC']].min(axis=1)

# ΔAIC: positive = sigmoid wins, negative = unstructured wins
df['dAIC_sigmoid_vs_unstructured'] = df['best_unstructured_AIC'] - df['best_sigmoid_AIC']

# Filter out inf values
valid = df[(np.isfinite(df['best_sigmoid_AIC'])) & (np.isfinite(df['best_unstructured_AIC']))]
print(f"Valid countries: {len(valid)}/{len(df)}")

print(f"\n{'='*70}")
print(f"  SIGMOID vs UNSTRUCTURED (piecewise-linear EXCLUDED)")
print(f"{'='*70}")

# Win counts
sig_wins = (valid['dAIC_sigmoid_vs_unstructured'] > 0).sum()
uns_wins = (valid['dAIC_sigmoid_vs_unstructured'] < 0).sum()
ties = (valid['dAIC_sigmoid_vs_unstructured'] == 0).sum()

print(f"\n  Sigmoid wins:  {sig_wins}/{len(valid)} ({sig_wins/len(valid)*100:.1f}%)")
print(f"  Unstructured wins: {uns_wins}/{len(valid)} ({uns_wins/len(valid)*100:.1f}%)")
print(f"  Ties:            {ties}")

d = valid['dAIC_sigmoid_vs_unstructured']
print(f"\n  ΔAIC (unstructured − sigmoid):")
print(f"    Mean:   {d.mean():.1f}")
print(f"    Median: {d.median():.1f}")
print(f"    Std:    {d.std():.1f}")
print(f"    Min:    {d.min():.1f}")
print(f"    Max:    {d.max():.1f}")

# Strong wins (|ΔAIC| > 10)
strong_sig = (d > 10).sum()
strong_uns = (d < -10).sum()
print(f"\n  Strong sigmoid win  (ΔAIC > +10):  {strong_sig} ({strong_sig/len(valid)*100:.1f}%)")
print(f"  Strong unstructured  (ΔAIC < −10):  {strong_uns} ({strong_uns/len(valid)*100:.1f}%)")

# Very strong (|ΔAIC| > 2, conventional threshold)
moderate_sig = (d > 2).sum()
print(f"  ΔAIC > +2 (conventional threshold): {moderate_sig} ({moderate_sig/len(valid)*100:.1f}%)")

# Distribution
print(f"\n  ΔAIC distribution:")
bins = [-np.inf, -20, -10, -2, 0, 2, 10, 20, np.inf]
labels = ['< −20', '−20 to −10', '−10 to −2', '−2 to 0', '0 to +2', '+2 to +10', '+10 to +20', '> +20']
for i in range(len(bins)-1):
    cnt = ((d > bins[i]) & (d <= bins[i+1])).sum()
    bar = '█' * (cnt // 5)
    print(f"    {labels[i]:<14} {cnt:>4} ({cnt/len(valid)*100:5.1f}%) {bar}")

# Which unstructured model wins most when sigmoid loses?
uns_winners = valid[valid['dAIC_sigmoid_vs_unstructured'] <= 0].copy()
if len(uns_winners) > 0:
    print(f"\n  When unstructured wins (n={len(uns_winners)}), which model?")
    for m, label in [('lin_AIC','Linear'), ('quad_AIC','Quadratic'),
                      ('quartic_AIC','Quartic'), ('spline_AIC','Spline')]:
        # Count where this model has the minimum AIC among unstructured
        best = uns_winners[['lin_AIC','quad_AIC','quartic_AIC','spline_AIC']].min(axis=1)
        wins = (uns_winners[m] == best).sum()
        print(f"    {label:<12} {wins:>3} ({wins/len(uns_winners)*100:.1f}%)")

# Paper-ready statement
print(f"\n{'='*70}")
print(f"  PAPER-READY STATEMENT")
print(f"{'='*70}")
print(f"  Excluding the piecewise-linear model (which carries its own structural")
print(f"  claim of a regime break), the sigmoid family outperforms all purely")
print(f"  unstructured smooth alternatives (linear, quadratic, quartic polynomial,")
print(f"  cubic B-spline) in {sig_wins} of {len(valid)} countries ({sig_wins/len(valid)*100:.1f}%).")
print(f"  Mean ΔAIC = {d.mean():.1f}, median = {d.median():.1f}.")
print(f"  Strong sigmoid dominance (ΔAIC > 10) in {strong_sig} countries")
print(f"  ({strong_sig/len(valid)*100:.1f}%).")
