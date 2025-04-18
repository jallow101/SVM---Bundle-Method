import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import sys
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Functions.solver import bundle_svm_solver


def load_iris_dataset(path="Datasets/adult.csv", binary=True):
    df = pd.read_csv(path)
    #df = df.sample(20000, random_state=42)  # Sample 1000 rows for faster processing

    if path == "Datasets/adult.csv":
        # Separate target
        y_raw = df["income"].values
        X = df.drop("income", axis=1)

        # Define categorical columns
        categorical_cols = ['Status', 'edu', 'mstatus', 'job', 'status2', 'ethni', 'country', 'gender']
        numerical_cols = [col for col in X.columns if col not in categorical_cols]

        # Apply OneHotEncoding to categorical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", numerical_cols),
                ("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ]
        )

        X_encoded = preprocessor.fit_transform(X)

    else:
        X = df.iloc[:, 1:-1].values  # Skip ID column
        y_raw = df.iloc[:, -1].values
        X_encoded = X

    # Binary label conversion
    if binary:
        class_labels = np.unique(y_raw)[:2]
        mask = np.isin(y_raw, class_labels)
        X_encoded = X_encoded[mask]
        y_raw = y_raw[mask]
        y = np.where(y_raw == class_labels[0], -1, 1)
    else:
        raise NotImplementedError("Multiclass not supported.")

    return X_encoded, y

def preprocess(X, y, degree=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X_scaled)
    return X_poly, y

def run_bundle(X_poly, y, strategy="fixed", alpha=None, C=1.0, mu_0=1.0, tol=1e-4):
    start = time.time()
    w, b, history = bundle_svm_solver(
        X_poly, y,
        C=C,
        mu_0=mu_0,
        tol=tol,
        step_size_strategy=strategy,
        alpha=alpha
    )
    runtime = time.time() - start
    acc = accuracy_score(y, np.sign(X_poly @ w + b))
    final_obj = history[-1]["f"] if isinstance(history[-1], dict) and "f" in history[-1] else None
    return {
        "accuracy": acc,
        "runtime": runtime,
        "iterations": len(history),
        "objective": final_obj,
        "history": history
    }

def main():
    # Load and preprocess data
    degree = 2
    X, y = load_iris_dataset()
    X_poly, y = preprocess(X, y, degree)

    # Compare optimal alpha = 1.0 vs default fixed (α_k = 2 / (2 + k))
    print("Running: Optimal alpha (1.0)")
    opt_result = run_bundle(X_poly, y, strategy="fixed", alpha=1.0101)

    print("Running: Default fixed step size")
    fixed_result = run_bundle(X_poly, y, strategy="fixed", alpha=None)  # None triggers classic α_k

    # Plotting 2x2 comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Convergence
    opt_obj = [h["f"] for h in opt_result["history"] if "f" in h]
    fixed_obj = [h["f"] for h in fixed_result["history"] if "f" in h]

    axes[0, 0].plot(opt_obj, marker='o', label="Alpha = 1.0")
    axes[0, 0].plot(fixed_obj, marker='x', label="Fixed α_k = 2 / (2 + k)")
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title("Convergence")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Objective (log scale)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Iterations
    axes[0, 1].bar(["Alpha=1.0", "Fixed α_k"], [opt_result["iterations"], fixed_result["iterations"]],
                   color=["#1f77b4", "#ff7f0e"])
    axes[0, 1].set_title("# Iterations")
    axes[0, 1].set_ylabel("Iterations")

    # 3. Runtime
    axes[1, 0].bar(["Alpha=1.0", "Fixed α_k"], [opt_result["runtime"], fixed_result["runtime"]],
                   color=["#1f77b4", "#ff7f0e"])
    axes[1, 0].set_title("Execution Time")
    axes[1, 0].set_ylabel("Time (s)")

    # 4. Final Objective
    axes[1, 1].bar(["Alpha=1.0", "Fixed α_k"], [opt_result["objective"], fixed_result["objective"]],
                   color=["#1f77b4", "#ff7f0e"])
    axes[1, 1].set_title("Final Objective Value")
    axes[1, 1].set_ylabel("Objective Value")

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/compare_optimal_alpha_vs_fixed.png", dpi=300)
    print("Comparison plot saved to results/compare_optimal_alpha_vs_fixed.png")
    plt.show()

if __name__ == "__main__":
    main()
