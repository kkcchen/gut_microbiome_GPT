import pandas as pd
import matplotlib.pyplot as plt

contextual_correlations = pd.read_csv("property_pathway_dict_allsig_contextual.txt", sep="\t")['corr_val'].values
glove_correlations = pd.read_csv("property_pathway_dict_allsig_glove.txt", sep="\t")['corr_val'].values

# Create normalized histograms of the two correlation arrays
plt.hist(contextual_correlations, bins=50, density=True, alpha=0.7, label='Contextual')
plt.hist(glove_correlations, bins=50, density=True, alpha=0.7, label='GloVe')
plt.legend(loc='upper right')
plt.xlabel("Correlation strength")
plt.ylabel("Percent of correlations")
plt.savefig("../new_hists/correlation_histograms.pdf", bbox_inches='tight')
plt.close()

