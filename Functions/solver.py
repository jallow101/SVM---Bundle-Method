import numpy as np
from Functions.subgradient import compute_subgradient


def bundle_svm_solver(X, y, C=1.0, mu_0=1.0, tol=1e-4, step_size_strategy="fixed"):
    """
    Simplified subgradient-based solver with step-size strategies.

    Returns:
        w, b: final model parameters
        history: list of {"f", "step_norm"} per iteration
    """
    max_iter = 1000
    w = np.zeros(X.shape[1])
    b = 0.0
    history = []

    for k in range(max_iter):
        f_val, grad_w, grad_b = compute_subgradient(w, b, X, y, C)

        # Step size
        if step_size_strategy == "fixed":
            alpha = 2 / (2 + k)
        elif step_size_strategy == "line_search":
            alpha = 1.0  # Placeholder – you can tune or implement actual line search
        else:
            raise ValueError("Invalid step_size_strategy")

        # Compute update step
        w_new = w - alpha * grad_w
        b_new = b - alpha * grad_b

        # Step norm
        step_vec = np.concatenate([w_new - w, [b_new - b]])
        step_norm = np.linalg.norm(step_vec)

        # Update
        w, b = w_new, b_new

        # Log
        history.append({
            "f": f_val,
            "step_norm": step_norm,
            "mu": mu_0,  # Not actually used here
            "serious": True  # Always true in this simplified method
        })

        if step_norm < tol:
            print(f" Converged in {k+1} iterations (‖step‖ < tol)")
            break

    return w, b, history