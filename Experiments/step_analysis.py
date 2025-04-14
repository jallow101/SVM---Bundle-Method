import numpy as np
import sys 
import os 
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_classification
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions.solver import bundle_svm_solver

def generate_data(n_samples=300, n_features=15, seed=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_informative=n_features,
        n_clusters_per_class=1,
        flip_y=0.05,
        class_sep=1.5,
        random_state=seed
    )
    y = 2 * y - 1
    return X, y

def plot_convergence_comparison(histories, labels):
    plt.figure(figsize=(8, 5))
    for hist, label in zip(histories, labels):
        objectives = [entry["f"] for entry in hist]
        plt.plot(objectives, label=label)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Objective (log scale)")
    plt.title("Convergence Comparison of Step-Size Strategies")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    plt.savefig("results/convergence_step_size_comparison.png")
    plt.show()

def plot_time_bar(times, labels):
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, times, color=["#1f77b4", "#ff7f0e"])
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time by Step-Size Strategy")
    for bar, val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.001, f"{val:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig("results/time_step_size_comparison.png")
    plt.show()

def run():
    X, y = generate_data()
    strategies = ["fixed", "line_search"]
    histories = []
    times = []

    for strategy in strategies:
        print(f"\nRunning strategy: {strategy}")
        start = time.time()
        _, _, history = bundle_svm_solver(X, y, step_size_strategy=strategy)
        elapsed = time.time() - start
        histories.append(history)
        times.append(elapsed)

    plot_convergence_comparison(histories, strategies)
    plot_time_bar(times, strategies)

if __name__ == "__main__":
    run()
