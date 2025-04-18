import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Functions.solver import bundle_svm_solver


def load_iris_dataset(path="Datasets/iris.csv"):
    import pandas as pd
    df = pd.read_csv(path)
    X = df.iloc[:, 1:-1].values
    y_raw = df.iloc[:, -1].values
    class_labels = np.unique(y_raw)[:2]
    mask = np.isin(y_raw, class_labels)
    X = X[mask]
    y_raw = y_raw[mask]
    y = np.where(y_raw == class_labels[0], -1, 1)
    return X, y

def preprocess(X, y, degree=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X_scaled)
    return X_poly, y

def get_convergence_history(X_poly, y, alpha, C=1.0, mu_0=1.0, tol=1e-4):
    w, b, history = bundle_svm_solver(
        X_poly, y, C=C, mu_0=mu_0, tol=tol,
        step_size_strategy="fixed", alpha=alpha
    )
    return history

def plot_convergence_curves(histories, labels, save_path="results/convergence_overlay.png"):
    plt.figure(figsize=(10, 6))

    for hist, label in zip(histories, labels):
        objectives = [entry["f"] for entry in hist if "f" in entry]
        plt.plot(objectives, marker='', label=f"{label}")

    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value (log scale)")
    plt.title("Convergence Behavior Comparison")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Convergence comparison plot saved to {save_path}")
    plt.show()

def main():
    # Configurations to compare
    alpha_values = [0.1, 0.25, 0.5, 1.0,1.01, 1.012, 0.11, 0.117, 0.1,1.3,1.5, 1.7, 2.0, 2.2]  # Select your range
    degree = 2
    C = 1.0
    mu_0 = 1.0
    tol = 1e-4

    X, y = load_iris_dataset()
    X_poly, y = preprocess(X, y, degree)

    histories = []
    labels = []

    for alpha in alpha_values:
        print(f"Running for alpha = {alpha}")
        try:
            history = get_convergence_history(X_poly, y, alpha, C, mu_0, tol)
            histories.append(history)
            labels.append(f"alpha={alpha}")
        except Exception as e:
            print(f"Failed for alpha={alpha}: {e}")

    plot_convergence_curves(histories, labels)

if __name__ == "__main__":
    main()
