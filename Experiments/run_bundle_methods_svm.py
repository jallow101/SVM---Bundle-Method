import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.datasets import make_classification

from Functions.solver import bundle_svm_solver

def load_iris_dataset(path="Datasets/iris.csv", binary=True):
    df = pd.read_csv(path)
    X = df.iloc[:, 1:-1].values  # Skip ID column
    y_raw = df.iloc[:, -1].values

    if binary:
        class_labels = np.unique(y_raw)[:2]
        mask = np.isin(y_raw, class_labels)
        X = X[mask]
        y_raw = y_raw[mask]
        y = np.where(y_raw == class_labels[0], -1, 1)
    else:
        raise NotImplementedError("Multiclass not supported.")

    return X, y

def generate_data(n_samples=200, n_features=2, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_informative=n_features,
        n_clusters_per_class=1,
        flip_y=0.05,
        class_sep=1.5,
        random_state=random_state
    )
    y = 2 * y - 1  # convert to {-1, +1}
    return X, y

def run_experiment(use_dataset=True, dataset_path="Datasets/iris.csv",
                   degree=2, C=1.0, mu_0=1.0, tol=1e-4, n_samples=200):

    # Step 1: Load data
    if use_dataset and os.path.exists(dataset_path):
        X, y = load_iris_dataset(dataset_path)
        print(f"✅ Loaded Iris dataset: {X.shape}")
    else:
        X, y = generate_data(n_samples=n_samples)
        print(f"🧪 Generated synthetic dataset: {X.shape}")

    # Step 2: Normalize + expand
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X_scaled)
    print(f"✅ Polynomial feature shape: {X_poly.shape}")

    # Step 3: Run bundle solver
    w_opt, b_opt, history = bundle_svm_solver(X_poly, y, C=C, mu_0=mu_0, tol=tol)

    # Step 4: Plot convergence
    plt.plot(history, marker='o')
    plt.title(f"Objective Convergence (Degree {degree})")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n🎯 Final Results")
    print("Bias (b):", b_opt)
    print("Weight shape:", w_opt.shape)
    print("Final objective:", history[-1])

if __name__ == "__main__":
    # Set use_dataset=False to run on generated data
    run_experiment(use_dataset=True)
