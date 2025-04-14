import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def plot_convergence(history, method_label="Bundle Method", degree=2, strategy="line_search"):
    """
    Plot convergence curve for Bundle Method.

    Parameters:
    - history: list of dicts with keys "f", "step_norm"
    - method_label: name to show on plot
    - degree: polynomial kernel degree
    - strategy: step size strategy ("line_search" or "fixed")
    """
    objectives = [entry["f"] for entry in history]
    step_norms = [entry["step_norm"] for entry in history]

    plt.figure(figsize=(8, 5))

    # Plot objective values
    plt.plot(objectives, marker='o', label="Objective Value $f(x_k)$", linewidth=2)

    # Optionally, add step norm (dual insight)
    #plt.plot(step_norms, marker='x', linestyle='--', label=r"Step Norm $\|x_{k+1} - \bar{x}_k\|$", linewidth=1.5)

    plt.yscale('log')
    plt.xlabel("Iterazione $k$", fontsize=12)
    plt.ylabel("Valore (log scale)", fontsize=12)
    plt.title(f"Convergenza ({method_label}, grado={degree})", fontsize=13)
    plt.grid(True, which="both", linestyle='--', linewidth=0.6)
    plt.legend()
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    filename = f"results/convergenza_{strategy}_grado{degree}.png"
    plt.savefig(filename, dpi=300)
    print(f" Convergence plot saved to {filename}")
    plt.show()


def plot_accuracy_comparison(bundle_acc, sklearn_acc, ampl_acc, save_path="results/accuracy_comparison.png"):
    methods = ["Bundle Method", "sklearn SVC", "AMPL"]
    accuracies = [bundle_acc, sklearn_acc, ampl_acc]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(methods, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylim(0.2, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison Across Methods")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval*100:.2f}%", ha='center')

    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f" Accuracy comparison plot saved to {save_path}")
    plt.show()

def plot_time_comparison(bundle_time, sklearn_time, ampl_time, save_path="results/time_comparison.png"):
    methods = ["Bundle Method", "sklearn SVC", "AMPL"]
    times = [bundle_time, sklearn_time, ampl_time]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(methods, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel("Time (seconds)")
    plt.title("Training Time Comparison")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f"{yval:.4f}s", ha='center')

    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"  Time comparison plot saved to {save_path}")
    plt.show()

def plot_objective_comparison(bundle_obj, sklearn_obj, ampl_obj, save_path="results/objective_comparison.png"):
    methods = ["Bundle Method", "sklearn SVC", "AMPL"]
    values = [bundle_obj, sklearn_obj, ampl_obj]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(methods, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.ylabel("Objective Value")
    plt.title("Final Objective Value Comparison")

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", va="bottom")

    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f" Objective comparison saved to {save_path}")
    plt.show()
