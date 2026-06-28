import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, shapiro, levene
from itertools import combinations

data = {
    'A': [0.78, 0.80, 0.77, 0.81, 0.79, 0.82, 0.78, 0.80],
    'B': [0.81, 0.83, 0.82, 0.84, 0.82, 0.85, 0.81, 0.83],
    'C': [0.80, 0.82, 0.79, 0.83, 0.81, 0.84, 0.80, 0.81],
    'D': [0.84, 0.85, 0.83, 0.86, 0.85, 0.87, 0.84, 0.86],
}
df = pd.DataFrame(data, index=[f'Fold {i}' for i in range(1, 9)])

for col in df.columns:
    stat, p = shapiro(df[col])
    print(f"Shapiro-Wilk {col}: W={stat:.4f}, p={p:.4f}")

lstat, lp = levene(df['A'], df['B'], df['C'], df['D'])
print(f"Levene: W={lstat:.4f}, p={lp:.4f}")

fstat, fp = friedmanchisquare(df['A'], df['B'], df['C'], df['D'])
print(f"Friedman: Chi2={fstat:.4f}, p={fp:.6f}")

n, k = 8, 4
q_alpha = 2.569  
CD = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
avg_ranks = df.rank(axis=1).mean()
print(f"CD = {CD:.4f}")
for m1, m2 in combinations(['A','B','C','D'], 2):
    diff = abs(avg_ranks[m1] - avg_ranks[m2])
    sig = "SIGNIFIKAN" if diff > CD else "tidak signifikan"
    print(f"  {m1} vs {m2}: |diff|={diff:.2f} → {sig}")

wstat_DA, wp_DA = wilcoxon(df['D'], df['A'])
print(f"Wilcoxon D vs A: W={wstat_DA}, p={wp_DA:.6f}")

wstat_DB, wp_DB = wilcoxon(df['D'], df['B'])
print(f"Wilcoxon D vs B: W={wstat_DB}, p={wp_DB:.6f}")