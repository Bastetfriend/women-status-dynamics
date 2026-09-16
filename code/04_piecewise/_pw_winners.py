"""
列出53个分段线性AIC赢家 + 按dAIC排序
"""
import pandas as pd

results = pd.read_csv(r"data\_vdem_model_comparison.csv")
pw = results[results['aic_winner'] == 'Piecewise'].sort_values('dAIC_sig_vs_simple', ascending=False)

print(f"Piecewise-linear AIC winners: {len(pw)} countries\n")
header = f"{'Entity':<35} {'Code':<5} {'dAIC':>7} {'PW_R2':>7} {'Ric_R2':>7}"
print(header)
print("-" * len(header))
for _, r in pw.iterrows():
    entity = str(r['entity'])[:34]
    code = str(r['code'])
    daic = r['dAIC_sig_vs_simple']
    pw_r2 = r['pw_R2'] if pd.notna(r['pw_R2']) else 0
    ric_r2 = r['ric_R2'] if pd.notna(r['ric_R2']) else 0
    print(f"{entity:<35} {code:<5} {daic:>7.1f} {pw_r2:>7.3f} {ric_r2:>7.3f}")
