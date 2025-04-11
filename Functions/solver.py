import numpy as np
from Functions.subgradient import compute_subgradient


def bundle_svm_solver(X, y, C=1.0, mu_0=1.0, tol=1e-4, step_size_strategy="fixed"):
    """
    Simplified subgradient-based solver with step-size strategies.

    Returns:
        w, b: final model parameters
        history: list of {"f", "step_norm"} per iteration
    """
    max_iter = 800
    w = np.zeros(X.shape[1])
    b = 0.0
    history = []

    for k in range(max_iter):
        f_val, grad_w, grad_b = compute_subgradient(w, b, X, y, C)

        # Step size
        if step_size_strategy == "fixed":
            alpha = 2 / (2 + k)
        elif step_size_strategy == "line_search":
            # Backtracking line search
            alpha = 1.0
            beta = 0.5
            while True:
                w_temp = w - alpha * grad_w
                b_temp = b - alpha * grad_b
                step_vec = np.concatenate([w_temp - w, [b_temp - b]])
                f_temp, _, _ = compute_subgradient(w_temp, b_temp, X, y, C)
                if f_temp <= f_val - 0.5 * alpha * np.linalg.norm(step_vec)**2:
                    break
                alpha *= beta
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