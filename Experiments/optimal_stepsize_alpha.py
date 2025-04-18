import numpy as np
import os
import sys
import pandas as pd
import time
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

def run_bundle(X_poly, y, C, mu_0, tol, alpha_val):
    start = time.time()
    w, b, history = bundle_svm_solver(
        X_poly, y, C=C, mu_0=mu_0, tol=tol,
        step_size_strategy="fixed", alpha=alpha_val
    )
    runtime = time.time() - start
    acc = accuracy_score(y, np.sign(X_poly @ w + b))
    final_obj = history[-1]["f"] if isinstance(history[-1], dict) and "f" in history[-1] else None
    return acc, runtime, final_obj, len(history), history

def main():
    # Alpha range: 100 values from 0.1 to 10 (log scale)
    alpha_vals = np.logspace(-1, 1, 100)
    alpha_vals = [round(float(a), 4) for a in alpha_vals]

    degree = 2
    C = 1.0
    mu_0 = 1.0
    tol = 1e-4

    X, y = load_iris_dataset()
    X_poly, y = preprocess(X, y, degree)

    results = []
    histories = []

    for alpha in alpha_vals:
        print(f"Running for alpha = {alpha}")
        try:
            acc, runtime, obj_val, iterations, history = run_bundle(X_poly, y, C, mu_0, tol, alpha)
            results.append({
                "alpha": alpha,
                "accuracy": acc,
                "runtime": runtime,
                "final_objective": obj_val,
                "iterations": iterations
            })
            histories.append((alpha, history))
        except Exception as e:
            print(f"Alpha {alpha} failed: {e}")
            results.append({
                "alpha": alpha,
                "accuracy": None,
                "runtime": None,
                "final_objective": None,
                "iterations": None
            })
            histories.append((alpha, None))

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/alpha_tuning_fixed_step.csv", index=False)
    print("\n=== Alpha Tuning Results ===")
    print(df)

    # Plot 2x2 summary grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(df["alpha"], df["accuracy"], marker="o", color='blue')
    axes[0, 0].set_xlabel("Alpha")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Accuracy vs. Alpha")
    axes[0, 0].grid(True)

    axes[0, 1].plot(df["alpha"], df["runtime"], marker="s", color='orange')
    axes[0, 1].set_xlabel("Alpha")
    axes[0, 1].set_ylabel("Runtime (s)")
    axes[0, 1].set_title("Runtime vs. Alpha")
    axes[0, 1].grid(True)

    axes[1, 0].plot(df["alpha"], df["iterations"], marker="^", color='green')
    axes[1, 0].set_xlabel("Alpha")
    axes[1, 0].set_ylabel("Iterations")
    axes[1, 0].set_title("Iterations vs. Alpha")
    axes[1, 0].grid(True)

    axes[1, 1].plot(df["alpha"], df["final_objective"], marker="x", color='red')
    axes[1, 1].set_xlabel("Alpha")
    axes[1, 1].set_ylabel("Final Objective Value")
    axes[1, 1].set_title("Final Objective vs. Alpha")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("results/alpha_tuning_summary_2x2.png", dpi=300)
    print("Full summary plot saved to results/alpha_tuning_summary_2x2.png")
    plt.show()

    # Plot convergence for top 5 alpha values (by accuracy)
    os.makedirs("results/convergence_plots", exist_ok=True)
    top_alphas = df.sort_values(by="accuracy", ascending=False).dropna().head(5)["alpha"].tolist()

    for alpha, hist in histories:
        if alpha in top_alphas and hist is not None:
            obj_vals = [h["f"] for h in hist if "f" in h]
            plt.figure()
            plt.plot(obj_vals, marker='o')
            plt.yscale('log')
            plt.xlabel("Iteration")
            plt.ylabel("Objective Value (log scale)")
            plt.title(f"Convergence for alpha = {alpha}")
            filename = f"results/convergence_plots/convergence_alpha_{alpha}.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=300)
            print(f"Saved convergence plot for alpha = {alpha} → {filename}")
            plt.close()

if __name__ == "__main__":
    main()
