import numpy as np
from Functions.subgradient import compute_subgradient
from Functions.subproblem import solve_bundle_subproblem
from Functions.step_update import update_stability_center


def bundle_svm_solver(X, y, C=1.0, mu_0=1.0, tol=1e-4, max_bundle_size=5, max_iter=100, step_size_strategy="fixed", alpha=None):
    """
    Full proximal bundle method for SVM optimization (primal).

    Returns:
        w, b: final model parameters
        history: list of dicts with convergence data
    """
    D = X.shape[1]
    w = np.zeros(D)
    b = 0.0
    mu = mu_0
    w_bar = w.copy()
    b_bar = b
    history = []
    bundle = []

    f_val, grad_w, grad_b = compute_subgradient(w, b, X, y, C)
    bundle.append((w.copy(), b, f_val, grad_w.copy(), grad_b))

    for k in range(max_iter):
        # Solve subproblem using bundle
        try:
            w_trial, b_trial = solve_bundle_subproblem(bundle, mu, w_bar, b_bar)
        except RuntimeError as e:
            print("Subproblem failed:", e)
            break

        f_trial, grad_w_trial, grad_b_trial = compute_subgradient(w_trial, b_trial, X, y, C)

        
        if step_size_strategy == "fixed":
            if alpha is None:
                alpha = 2 / (2 + k)
                print(f"Step size (α): {alpha:.4f}")
            else:
                alpha = alpha
                print(f"Step size (α): {alpha:.4f}")
        elif step_size_strategy == "line_search":
            # Line search strategy
            alpha = 1.0
            beta = 0.5
            c = 1e-4
            grad_vec = np.concatenate([grad_w_trial, [grad_b_trial]])
            f_curr = history[-1]['f'] if history else f_val

            while True:
                w_new = w + alpha * (w_trial - w)
                b_new = b + alpha * (b_trial - b)
                f_new, _, _ = compute_subgradient(w_new, b_new, X, y, C)
                step_vec = np.concatenate([w_new - w, [b_new - b]])
                if f_new <= f_curr - c * alpha * np.dot(grad_vec, step_vec):
                    break
                alpha *= beta
        
        # Final update
        w_new = w + alpha * (w_trial - w)
        b_new = b + alpha * (b_trial - b)

        f_new, grad_w_new, grad_b_new = compute_subgradient(w_new, b_new, X, y, C)

        # Check for serious step
        f_old = history[-1]['f'] if history else f_val
        is_serious, w_bar, b_bar, mu = update_stability_center(
            f_new, f_old, w_new, b_new, w_bar, b_bar, mu
        )

        # Log step
        step_norm = np.linalg.norm(np.concatenate([w_new - w, [b_new - b]]))
        history.append({
            "f": f_new,
            "step_norm": step_norm,
            "mu": mu,
            "serious": is_serious
        })

        # Update current point
        w, b = w_new, b_new

        # Add to bundle
        bundle.append((w.copy(), b, f_new, grad_w_new.copy(), grad_b_new))
        if len(bundle) > max_bundle_size:
            bundle.pop(0)

        # Stopping condition
        if step_norm < tol:
            print(f"Converged in {k+1} iterations (‖step‖ < tol)")
            break

    return w, b, history
