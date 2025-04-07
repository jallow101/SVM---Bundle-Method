import numpy as np
from Functions.subgradient import compute_subgradient
from Functions.cutting_plane import build_cutting_plane_model
from Functions.subproblem import solve_bundle_subproblem
from Functions.step_update import update_stability_center
from Functions.convergence import check_convergence

def bundle_svm_solver(X, y, C=1.0, mu_0=1.0, tol=1e-4, max_iter=100):
    """
    Solves the primal SVM using a proximal bundle method.

    Parameters:
        X : ndarray of shape (n_samples, D)
            Feature matrix (each row is phi(x_i)).
        y : ndarray of shape (n_samples,)
            Labels (+1 or -1).
        C : float
            Regularization parameter.
        mu_0 : float
            Initial proximal parameter.
        tol : float
            Stopping tolerance on subgradient norm.
        max_iter : int
            Maximum number of iterations.

    Returns:
        w, b : Final weight vector and bias
        history : List of objective values per iteration
    """
    n_samples, D = X.shape

    # Initialize
    w = np.zeros(D)
    b = 0.0
    mu_k = mu_0

    # Initial subgradient
    f_val, grad_w, grad_b = compute_subgradient(w, b, X, y, C)
    bundle = [(w.copy(), b, f_val, grad_w.copy(), grad_b)]
    w_bar, b_bar = w.copy(), b
    f_bar = f_val
    history = [f_val]

    for k in range(max_iter):
        # Solve bundle subproblem
        w_new, b_new = solve_bundle_subproblem(bundle, mu_k, w_bar, b_bar)

        # Evaluate function and subgradient at new point
        f_new, grad_w_new, grad_b_new = compute_subgradient(w_new, b_new, X, y, C)

        # Step decision
        is_serious, w_bar, b_bar, mu_k = update_stability_center(
            f_new, f_bar,
            w_new, b_new,
            w_bar, b_bar,
            mu_k
        )

        # Update f_bar if serious
        if is_serious:
            f_bar = f_new

        # Add new point to bundle
        bundle.append((w_new.copy(), b_new, f_new, grad_w_new.copy(), grad_b_new))
        history.append(f_new)

        # Check convergence
        if check_convergence(grad_w_new, grad_b_new, tol):
            print(f"Converged in {k+1} iterations.")
            break

    return w_bar, b_bar, history
