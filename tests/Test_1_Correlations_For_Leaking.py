from scipy.stats import pointbiserialr
#tests Correlations > 0.5 or < -0.5 are strong and may indicate risk of leakage.
#Very small p-values mean the correlation is statistically significant.

for col in numerical_cols:
    corr, p_value = pointbiserialr(credit_data[col], credit_data['is_default'])
    print(f"{col}: correlation = {corr:.3f}, p-value = {p_value:.3g}")