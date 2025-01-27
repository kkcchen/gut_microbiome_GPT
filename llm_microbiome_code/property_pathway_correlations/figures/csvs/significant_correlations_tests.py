import pandas as pd
import numpy as np
from scipy.stats import epps_singleton_2samp, ks_2samp, anderson_ksamp


contextual_correlations = pd.read_csv("property_pathway_dict_allsig_contextual.txt", sep="\t")['corr_val'].values
glove_correlations = pd.read_csv("property_pathway_dict_allsig_glove.txt", sep="\t")['corr_val'].values


# Flatten the correlation matrices
contextual_flat = contextual_correlations.flatten()
glove_flat = glove_correlations.flatten()

# Filter out any correlations below 0.02 in absolute value
contextual_flat = contextual_flat[np.abs(contextual_flat) >= 0.02]
glove_flat = glove_flat[np.abs(glove_flat) >= 0.02]

print(np.mean(np.abs(contextual_flat)))
print(np.mean(np.abs(glove_flat)))

# Compute the median and 25th and 75th percentiles for contextual correlations
contextual_median = np.median(np.abs(contextual_flat))
contextual_25th_percentile = np.percentile(np.abs(contextual_flat), 25)
contextual_75th_percentile = np.percentile(np.abs(contextual_flat), 75)
contextual_99th_percentile = np.percentile(np.abs(contextual_flat), 99)

print(f"Contextual Correlations - Median: {contextual_median}, 25th Percentile: {contextual_25th_percentile}, 75th Percentile: {contextual_75th_percentile}, 99th Percentile: {contextual_99th_percentile}")

# Compute the median and 25th 75th and 99th percentiles for absolute glove correlations
glove_abs_median = np.median(np.abs(glove_flat))
glove_abs_25th_percentile = np.percentile(np.abs(glove_flat), 25)
glove_abs_75th_percentile = np.percentile(np.abs(glove_flat), 75)
glove_abs_99th_percentile = np.percentile(np.abs(glove_flat), 99)

print(f"Absolute Glove Correlations - Median: {glove_abs_median}, 25th Percentile: {glove_abs_25th_percentile}, 75th Percentile: {glove_abs_75th_percentile}, 99th Percentile: {glove_abs_99th_percentile}")

from scipy.stats import expon

# Fit exponential distributions to the absolute values of glove and contextual correlations
contextual_flat_abs = np.abs(contextual_flat)
glove_flat_abs = np.abs(glove_flat)

contextual_expon_params = expon.fit(contextual_flat_abs)
glove_expon_params = expon.fit(glove_flat_abs)

print(f"Exponential fit parameters for absolute contextual correlations: {contextual_expon_params}")
print(f"Exponential fit parameters for absolute glove correlations: {glove_expon_params}")





# Perform two-sample Kolmogorov-Smirnov test
ks_statistic, p_value = ks_2samp(contextual_flat, glove_flat)

print(f"Kolmogorov-Smirnov test results:")
print(f"KS statistic: {ks_statistic}")
print(f"p-value: {p_value}")

# Perform one-sided Kolmogorov-Smirnov test with the null hypothesis that glove correlations are greater than contextual correlations
ks_statistic_one_sided, p_value_one_sided = ks_2samp(np.abs(contextual_flat), np.abs(glove_flat), alternative='less')

print(f"One-sided Kolmogorov-Smirnov test results (H0: glove correlations > contextual correlations):")
print(f"KS statistic: {ks_statistic_one_sided}")
print(f"p-value: {p_value_one_sided}")

if p_value_one_sided < 0.05:
    print("The null hypothesis that glove correlations are greater than contextual correlations is rejected (p < 0.05)")
else:
    print("There is not enough evidence to reject the null hypothesis that glove correlations are greater than contextual correlations")


# Perform one-sided Kolmogorov-Smirnov test with the null hypothesis that glove correlations are less than contextual correlations
ks_statistic_one_sided, p_value_one_sided = ks_2samp(np.abs(contextual_flat), np.abs(glove_flat), alternative='greater')

print(f"One-sided Kolmogorov-Smirnov test results (H0: glove correlations < contextual correlations):")
print(f"KS statistic: {ks_statistic_one_sided}")
print(f"p-value: {p_value_one_sided}")

if p_value_one_sided < 0.05:
    print("The null hypothesis that glove correlations are less than contextual correlations is rejected (p < 0.05)")
else:
    print("There is not enough evidence to reject the null hypothesis that glove correlations are less than contextual correlations")


if p_value < 0.05:
    print("The distributions are significantly different (p < 0.05)")
else:
    print("There is not enough evidence to conclude that the distributions are different")

# Perform Epps-Singleton test
es_statistic, es_p_value = epps_singleton_2samp(contextual_flat, glove_flat)

print(f"\nEpps-Singleton test results:")
print(f"ES statistic: {es_statistic}")
print(f"p-value: {es_p_value}")

if es_p_value < 0.05:
    print("The distributions are significantly different (p < 0.05)")
else:
    print("There is not enough evidence to conclude that the distributions are different")

# Perform Anderson-Darling test
result = anderson_ksamp([contextual_flat, glove_flat])
ad_statistic, ad_p_value = result.statistic, result.pvalue

print(f"\nAnderson-Darling test results:")
print(f"AD statistic: {ad_statistic}")
print(f"p-value: {ad_p_value}")

if ad_p_value < 0.05:
    print("The distributions are significantly different (p < 0.05)")
else:
    print("There is not enough evidence to conclude that the distributions are different")

print(contextual_correlations.shape, glove_correlations.shape)
