import cvxpy as cp
import numpy as np

def solve_bundle_subproblem(bundle, mu, w_bar, b_bar):
    """
    Solves the proximal bundle subproblem.

    Parameters:
        bundle : list of tuples
            Each element is (w_j, b_j, f_j, grad_w_j, grad_b_j)
        mu : float
            Proximal regularization parameter.
        w_bar : ndarray of shape (D,)
            Current stability center for w.
        b_bar : float
            Current stability center for b.

    Returns:
        w_opt : ndarray
            Updated weight vector.
        b_opt : float
            Updated bias.
    """
    D = len(w_bar)
    w = cp.Variable(D)
    b = cp.Variable()
    xi = cp.Variable()

    constraints = []
    for (w_j, b_j, f_j, grad_w_j, grad_b_j) in bundle:
        affine_expr = f_j + grad_w_j @ (w - w_j) + grad_b_j * (b - b_j)
        constraints.append(xi >= affine_expr)

    objective = cp.Minimize(xi + (mu / 2) * (cp.sum_squares(w - w_bar) + cp.square(b - b_bar)))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    return w.value, b.value
