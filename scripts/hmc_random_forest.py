import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from itertools import product
from joblib import Parallel, delayed
from scipy.stats import binomtest
import json
import os
from tqdm import tqdm

# Generate a random feature matrix X (shape: N x K)
X = np.load("/home/kevin/Desktop/gut_microbiome/dataset/hmc/electra/random_forest/taxonomy_table.npy")

# get labels
with open("/home/kevin/Desktop/gut_microbiome/dataset/hmc/electra/random_forest/labels.json") as f:
    labels_list = json.load(f)
f.close()

label_dict = {
    'Australia/New Zealand': 0,
    'Central and Southern Asia': 1,
    'Eastern and South-Eastern Asia': 2,
    'Europe and Northern America': 3,
    'Latin America and the Caribbean': 4,
    'Northern Africa and Western Asia': 5,
    'Sub-Saharan Africa': 6,
    'unknown': 7,
}
Y = np.array([label_dict[l] for l in labels_list])
# (array([0, 1, 2, 3, 4, 5, 6, 7]),
#  array([ 2953,  1075,  8797, 51500,   427,  1215,  1778, 14037]))

# filter out the data that has unknown location data
known_idx = np.argwhere(Y != 7).squeeze(1)
X = X[known_idx]
Y = Y[known_idx]
# -----  67745 samples, 4680 taxons
# ----- class distribution:[ 2953,  1075,  8797, 51500,   427,  1215,  1778]

# temporary: save the dataset without unknowns
# save_dir = "/home/kevin/Desktop/gut_microbiome/dataset/hmc/electra/random_forest/without_unknown"
# np.save(os.path.join(save_dir, "taxonomy_table.npy"), X)
# np.save(os.path.join(save_dir, "labels.npy"), Y)

# here we train the random forest on abundance data with per sample linear normalization
X = X[:, :, 1]
# do filtering same as paper prevalence threshold = 0.01, abundance threshold=0.05
prevalence_threshold = 0.01
abundance_threshold = 0.05
# prevalence threshold
tax_prevalence = np.mean(X > 0, axis=0)
mask_taxons = tax_prevalence > prevalence_threshold
X_filtered = X[:, mask_taxons]
# abundance threshold
tax_abun = np.mean(X, axis=0)
mask_taxons = tax_abun > abundance_threshold
X_filtered = X[:, mask_taxons]
# ----- 1178 taxa remained
# now normalize
min_X = np.expand_dims(np.min(X_filtered, axis=1), 1)
max_X = np.expand_dims(np.max(X_filtered, axis=1), 1)
X_filtered = (X_filtered - min_X) / (max_X - min_X + 1e6)

# grid search
min_samples_leaf_options = [1, 5, 10]
max_samples_options = [1.0, 0.75, 0.5]
max_features_options = [90, 120, 150, 180, 210, 240]
n_estimators_options = [800, 1000, 2000]
# min_samples_leaf_options = [1]
# max_samples_options = [1.0]
# max_features_options = [90]
# n_estimators_options = [100, 200, 300]
params_list = list(product(min_samples_leaf_options, max_samples_options, max_features_options, n_estimators_options))
print(f"Total parameter combinations to evaluate: {len(params_list)}")

x_train, x_test, y_train, y_test = train_test_split(X_filtered, Y, test_size=0.2, random_state=42)
no_info_class = 3
n_total = len(y_test)
n_majority = np.sum(y_test == no_info_class)
no_info_rate = n_majority / n_total

def evaluate_random_forest(params):
    min_samples_leaf, max_samples, max_features, n_estimators = params
    if max_features > x_train.shape[1]:
        return None
    model = RandomForestClassifier(
        min_samples_leaf=min_samples_leaf,
        max_samples=max_samples,
        max_features=max_features,
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1  # Use all threads for each model fit
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    n_correct = np.sum(y_pred == y_test)
    binomtest_ret = binomtest(n_correct, n_total, p=no_info_rate, alternative='greater')
    weighted_f1_score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    return binomtest_ret.pvalue, weighted_f1_score, params


# --------- final inference
for i in range(7):
    print(f"Starting Random Forest classifier on region {i}")
    Y_binary = (Y == i).astype(np.int64)

    # Split the data into training (80%) and testing (20%) sets.
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, Y_binary, test_size=0.2, random_state=42)
    # Initialize the RandomForestClassifier.
    rf_model = RandomForestClassifier(
        min_samples_leaf=1,
        max_samples=1.0,
        max_features=240,
        n_estimators=800,
        random_state=42,
        n_jobs=-1  # Use all threads for each model fit
    )
    # Train the model on the training data.
    rf_model.fit(X_train, y_train)
    # Predict the labels for the test set.
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(x_test)[:, 1]
    # Evaluate the model's accuracy on the test set.
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)
    auc = roc_auc_score(y_test, y_prob)
    print("AUC:", auc)
