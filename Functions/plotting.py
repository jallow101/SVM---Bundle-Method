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

    #os.makedirs("results", exist_ok=True)
    #filename = f"results/convergenza_{strategy}_grado{degree}.png"
    #plt.savefig(filename, dpi=300)
   # print(f" Convergence plot saved to {filename}")
    plt.show()
