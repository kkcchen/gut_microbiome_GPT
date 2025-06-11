import numpy as np
import argparse
from skbio.stats.composition import clr, multi_replace, closure
import os

def compute_prevalence_abundance(X):
    prevalence = np.mean(X > 0, axis=0)
    abundance = np.mean(X, axis=0)
    return prevalence, abundance

def preprocess_clr(X):
    # Replace zeros using multiplicative replacement
    X_replaced = multi_replace(X)
    # Apply closure to normalize counts to proportions
    X_closed = closure(X_replaced)
    # Apply CLR transformation
    return clr(X_closed)

def main():
    parser = argparse.ArgumentParser(description="Filter and CLR-transform a taxa abundance matrix using scikit-bio.")
    parser.add_argument("--output-dir", type=str, required=True, help="Dir to output .npy file")
    parser.add_argument("--train-input", type=str, required=True, help="Path to train .npy file (samples, taxa, 2)")
    parser.add_argument("--test-input", type=str, required=True, help="Path to test .npy file (samples, taxa, 2)")
    parser.add_argument("--prevalence-threshold", type=float, default=0.01, help="Minimum prevalence threshold")
    parser.add_argument("--abundance-threshold", type=float, default=0.05, help="Minimum abundance threshold")
    args = parser.parse_args()

    # Load the data
    X_train = np.load(args.train_input)
    X_test = np.load(args.test_input)
    X = np.concatenate((X_train, X_test), axis=0)
    
    # Compute prevalence and abundance
    prevalence, abundance = compute_prevalence_abundance(X[:, :, 1])

    # Filter taxa
    keep_mask = (prevalence >= args.prevalence_threshold) & (abundance >= args.abundance_threshold)
    X_filtered = X[:, keep_mask, :]

    if X_filtered.shape[1] == 0:
        raise ValueError("No taxa passed the filtering thresholds.")

    # CLR transform using skbio
    X_filtered[:, :, 1] = preprocess_clr(X_filtered[:, :, 1])

    # Separate train and test data
    train_filtered = X_filtered[:X_train.shape[0], :, :]
    test_filtered = X_filtered[X_train.shape[0]:, :, :]

    # Save train and test data separately
    train_output_path = os.path.join(args.output_dir, "raw_encoded_train.npy")
    test_output_path = os.path.join(args.output_dir, "raw_encoded_train.npy")

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(train_output_path, train_filtered)
    np.save(test_output_path, test_filtered)

    print(f"Saved CLR-transformed train data to {train_output_path} with shape {train_filtered.shape}")
    print(f"Saved CLR-transformed test data to {test_output_path} with shape {test_filtered.shape}")

if __name__ == "__main__":
    main()
