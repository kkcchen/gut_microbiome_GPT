import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import json
import os

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

# temporary: save the dataset without unknowns
# save_dir = "/home/kevin/Desktop/gut_microbiome/dataset/hmc/electra/random_forest/without_unknown"
# np.save(os.path.join(save_dir, "taxonomy_table.npy"), X)
# np.save(os.path.join(save_dir, "labels.npy"), Y)

# here we train the random forest on abundance data with per sample linear normalization
X = X[:, :, 1]
min_X = np.expand_dims(np.min(X, axis=1), 1)
max_X = np.expand_dims(np.max(X, axis=1), 1)
X = (X - min_X) / (max_X - min_X + 1e6)

# instead of doing random forest over all categories, we follow the HMC paper and do per category one-vs-all classification
for i in range(7):
    print(f"Starting Random Forest classifier on region {i}")
    Y_binary = (Y == i).astype(np.int64)

    # Split the data into training (80%) and testing (20%) sets.
    X_train, X_test, y_train, y_test = train_test_split(X, Y_binary, test_size=0.2, random_state=42)

    # Initialize the RandomForestClassifier.
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)

    # Train the model on the training data.

    rf_model.fit(X_train, y_train)

    # Predict the labels for the test set.
    y_pred = rf_model.predict(X_test)

    # Evaluate the model's accuracy on the test set.
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)
    auc = roc_auc_score(y_test, y_pred)
    print("AUC:", auc)
