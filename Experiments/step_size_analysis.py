import numpy as np
import pandas as pd
import time
import os
import sys
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Functions.solver import bundle_svm_solver
from Functions.ampl_model import run_ampl_svm
from Functions.plotting import plot_convergence

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

def preprocess(X, y, degree=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly.fit_transform(X_scaled)
    return X_scaled, X_poly, y

def run_bundle_step_strategy(X_poly, y, C, mu_0, tol, strategy):
    start = time.time()
    w, b, history = bundle_svm_solver(X_poly, y, C=C, mu_0=mu_0, tol=tol, step_size_strategy=strategy)
    duration = time.time() - start
    acc = accuracy_score(y, np.sign(X_poly @ w + b))
    return acc, duration, history

def run_svc(X_scaled, y, degree, C):
    start = time.time()
    clf = SVC(kernel="poly", degree=degree, C=C, coef0=1, gamma="auto")
    clf.fit(X_scaled, y)
    acc = accuracy_score(y, clf.predict(X_scaled))
    duration = time.time() - start
    return acc, duration

def run_ampl(X_poly, y):
    try:
        start = time.time()
        w, b = run_ampl_svm()
        duration = time.time() - start
        acc = accuracy_score(y, np.sign(X_poly @ w + b))
        return acc, duration
    except Exception as e:
        print("AMPL run failed:", e)
        return None, None

def main():
    degree = 2
    C = 1.0
    mu_0 = 1.0
    tol = 1e-4
    strategies = ["fixed", "line_search"]

    X, y = load_iris_dataset()
    X_scaled, X_poly, y = preprocess(X, y, degree)

    # SVC and AMPL once
    svc_acc, svc_time = run_svc(X_scaled, y, degree, C)
    ampl_acc, ampl_time = run_ampl(X_poly, y)

    results = []

    for strategy in strategies:
        acc, runtime, history = run_bundle_step_strategy(X_poly, y, C, mu_0, tol, strategy)
        results.append({
            "step_size_strategy": strategy,
            "bundle_acc": acc,
            "bundle_time": runtime,
            "svc_acc": svc_acc,
            "svc_time": svc_time,
            "ampl_acc": ampl_acc,
            "ampl_time": ampl_time
        })

        # Optional: plot convergence for each strategy
        plot_convergence(history, method_label=f"Bundle - {strategy}", degree=degree, strategy=strategy)

    df = pd.DataFrame(results)
    print("\n=== Step Size Strategy Analysis ===")
    print(df)

    # Accuracy plot
    plt.figure()
    plt.bar(df["step_size_strategy"], df["bundle_acc"], label="Bundle Accuracy")
    plt.axhline(y=svc_acc, color='g', linestyle='--', label="SVC Accuracy")
    plt.axhline(y=ampl_acc, color='r', linestyle='--', label="AMPL Accuracy")
    plt.title("Accuracy vs. Step Size Strategy")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/accuracy_vs_step_size.png")
    plt.show()

    # Time plot
    plt.figure()
    plt.bar(df["step_size_strategy"], df["bundle_time"], label="Bundle Time")
    plt.axhline(y=svc_time, color='g', linestyle='--', label="SVC Time")
    if ampl_time:
        plt.axhline(y=ampl_time, color='r', linestyle='--', label="AMPL Time")
    plt.title("Runtime vs. Step Size Strategy")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/runtime_vs_step_size.png")
    plt.show()

    df.to_csv("results/step_size_strategy_analysis.csv", index=False)

if __name__ == "__main__":
    main()
