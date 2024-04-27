import numpy as np
import pandas as pd
import json
import argparse

def generate_data(n_samples, n_features, lam, random_seed):
    np.random.seed(random_seed)

    # Generate random features
    X = np.random.randn(n_samples, n_features)

    # True coefficients for the linear relationship
    true_coefficients = np.random.normal(0, 1/np.sqrt(lam), n_features)

    # Generate target variable with added noise
    y = X.dot(true_coefficients) + np.random.randn(n_samples)

    return X, y, true_coefficients

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_filename', type=str, default='dataset.csv',
                        help="Name of the file where the generated data should be saved")
    parser.add_argument('--info_filename', type=str, default='dataset_info.json',
                        help="Name of the file where true coefficients and lambda value are saved")
    parser.add_argument('--n_samples', type=int, default=1000, 
                        help="Number of samples in data")
    parser.add_argument('--n_features', type=int, default=10, 
                        help="Number of dimensions of each data point")
    parser.add_argument('--lam', type=float, default=0.5, 
                        help="Inverse of variance of Gaussian prior on weight vector")
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    # Generate dataset
    X, y, true_coefficients = generate_data(args.n_samples, args.n_features, args.lam, args.seed)

    # Display some information about the generated dataset
    print("Generated dataset shape:", X.shape)
    print("True coefficients:", true_coefficients)

    with open(args.info_filename, "w") as file:
        json.dump({"true_coefficients": true_coefficients.tolist(), "lambda": args.lam}, file, indent=2)

    # Create a DataFrame with column names
    column_names = [f"feature_{i}" for i in range(1, args.n_features + 1)]

    df_X = pd.DataFrame(X, columns=column_names)
    df_y = pd.DataFrame(y, columns=["label"])

    # Combine X and y DataFrames
    df = pd.concat([df_X, df_y], axis=1)

    # Save to CSV
    df.to_csv(args.dataset_filename, index=False)
