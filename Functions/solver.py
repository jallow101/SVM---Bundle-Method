import numpy as np
from Functions.subgradient import compute_subgradient
from Functions.subproblem import solve_bundle_subproblem
from Functions.step_update import update_stability_center

def bundle_svm_solver(X, y, C=1.0, mu_0=1.0, tol=1e-4, max_iter=100, max_bundle_size=10):
    """
    Solves the primal SVM using a Proximal Bundle Method.

    Parameters:
        X : ndarray (n_samples, D)
            Feature matrix.
        y : ndarray (n_samples,)
            Labels (+1 or -1).
        C : float
            Regularization parameter.
        mu_0 : float
            Initial proximal parameter.
        tol : float
            Step norm stopping criterion (ε).
        max_iter : int
            Maximum iterations.
        max_bundle_size : int
            Maximum size of bundle (default=10 for performance).

    Returns:
        w_bar, b_bar : Final stability center (solution)
        history : List of dicts containing iteration metrics
    """
    n_samples, D = X.shape

    # Initialization
    w = np.zeros(D)
    b = 0.0
    mu_k = mu_0

    # First subgradient
    f_val, grad_w, grad_b = compute_subgradient(w, b, X, y, C)
    bundle = [(w.copy(), b, f_val, grad_w.copy(), grad_b)]

    w_bar, b_bar = w.copy(), b
    f_bar = f_val

    history = [{
        "f": f_val,
        "mu": mu_k,
        "step_norm": 0.0,
        "serious": True
    }]

    for k in range(max_iter):
        # Solve QP subproblem (cutting-plane model + proximal term)
        w_new, b_new = solve_bundle_subproblem(bundle, mu_k, w_bar, b_bar)

        # Evaluate function and subgradient at new candidate
        f_new, grad_w_new, grad_b_new = compute_subgradient(w_new, b_new, X, y, C)

        # Decide serious/null step, update center
        is_serious, w_bar, b_bar, mu_k = update_stability_center(
            f_new, f_bar,
            w_new, b_new,
            w_bar, b_bar,
            mu_k
        )

        # Update best value if serious step taken
        if is_serious:
            f_bar = f_new

        # Add to bundle
        bundle.append((w_new.copy(), b_new, f_new, grad_w_new.copy(), grad_b_new))
        if len(bundle) > max_bundle_size:
            bundle.pop(0)

        # Compute step norm
        step_vec = np.concatenate([w_new - w_bar, [b_new - b_bar]])
        step_norm = np.linalg.norm(step_vec)

        # Log iteration info
        history.append({
            "f": f_new,
            "mu": mu_k,
            "step_norm": step_norm,
            "serious": is_serious
        })

        # Check convergence
        if step_norm < tol:
            print(f"✅ Converged in {k+1} iterations (‖step‖ < ε).")
            break

    return w_bar, b_bar, history


