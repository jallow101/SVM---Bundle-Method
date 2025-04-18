import numpy as np
import matplotlib.pyplot as plt
import os

def plot_convergence_behavior(history, method_label="Solver", save_path=None):
    """
    Plot convergence behavior: objective values and convergence metric vs iterations.

    Parameters:
    - history: List or np.array of objective values per iteration.
    - method_label: Label for the optimization method.
    - save_path: If provided, saves the figure to this path.
    """
    history = np.array(history)
    iterations = np.arange(1, len(history) + 1)
    convergence_metric = np.abs(np.diff(history, prepend=history[0]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Objective Value vs Iteration (log-linear scale)
    axes[0].semilogy(iterations, history, marker='o', label=method_label)
    axes[0].set_title(f"{method_label} – Objective Value")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Objective Value (log scale)")
    axes[0].grid(True, which='both')
    axes[0].legend()

    # Plot 2: Convergence Metric (Change in Obj) vs Iteration (log-linear scale)
    axes[1].semilogy(iterations, convergence_metric, marker='x', color='orange', label="Change in Objective")
    axes[1].set_title(f"{method_label} – Convergence Behavior")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Δ Objective Value (log scale)")
    axes[1].grid(True, which='both')
    axes[1].legend()

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved convergence plot to {save_path}")
    else:
        plt.show()
