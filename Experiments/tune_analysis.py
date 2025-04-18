import numpy as np
import pandas as pd
import os
import sys
import time
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Functions.solver import bundle_svm_solver
from Functions.ampl_model import run_ampl_svm


def load_iris_dataset(path="Datasets/iris.csv"):
    df = pd.read_csv(path)
    X = df.iloc[:, 1:-1].values
    y_raw = df.iloc[:, -1].values
    class_labels = np.unique(y_raw)[:2]
    mask = np.isin(y_raw, class_labels)
    X = X[mask]
    y_raw = y_raw[mask]
    y = np.where(y_raw == class_labels[0], -1, 1)
    return X, y

def preprocess(X, y, degree):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X_scaled)
    return X_poly, y

def run_bundle(X_poly, y, C, mu_0, tol, step_size_strategy="fixed"):
    start = time.time()
    w, b, history = bundle_svm_solver(X_poly, y, C=C, mu_0=mu_0, tol=tol, step_size_strategy=step_size_strategy)
    runtime = time.time() - start
    acc = accuracy_score(y, np.sign(X_poly @ w + b))
    return acc, runtime, history


def main():
    degrees = [1, 2, 3]
    C_values = [0.1, 1.0, 10.0]
    mu_0_values = [0.1, 1.0, 10.0]
    tol_values = [1e-2, 1e-3, 1e-4]

    results = []

    X, y = load_iris_dataset()

    for degree, C, mu_0, tol in itertools.product(degrees, C_values, mu_0_values, tol_values):
        print(f"Running: degree={degree}, C={C}, mu_0={mu_0}, tol={tol}")
        X_poly, y_proc = preprocess(X, y, degree)

        try:
            acc, runtime, history = run_bundle(X_poly, y_proc, C, mu_0, tol)
            results.append({
                "degree": degree,
                "C": C,
                "mu_0": mu_0,
                "tol": tol,
                "accuracy": acc,
                "runtime": runtime,
                "iterations": len(history)
            })
        except Exception as e:
            print(f"Error with config degree={degree}, C={C}, mu_0={mu_0}, tol={tol}: {e}")
            results.append({
                "degree": degree,
                "C": C,
                "mu_0": mu_0,
                "tol": tol,
                "accuracy": None,
                "runtime": None,
                "iterations": None
            })

    df = pd.DataFrame(results)
    df.to_csv("results/tuned_bundle_results.csv", index=False)
    print("\n=== Hyperparameter Tuning Results ===")
    print(df)

    # Optional: plot top accuracy configs
    top = df.sort_values("accuracy", ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    plt.barh([f"{row.degree},{row.C},{row.mu_0},{row.tol}" for _, row in top.iterrows()],
             top["accuracy"])
    plt.xlabel("Accuracy")
    plt.title("Top 10 Bundle Configurations")
    plt.tight_layout()
    plt.savefig("results/top_tuned_bundle_configs.png")
    plt.show()

if __name__ == "__main__":
    main()
