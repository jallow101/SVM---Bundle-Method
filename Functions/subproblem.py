import cvxpy as cp
import numpy as np
from Functions.ampl_model import write_ampl_data, run_ampl_solver


def solve_bundle_subproblem(bundle, mu, w_bar, b_bar, verbose=False):
    """
    Solves the proximal bundle subproblem using cvxpy, with fallback to SCS if ECOS is not available.

    Parameters:
        bundle : list of tuples
            Each element is (w_j, b_j, f_j, grad_w_j, grad_b_j)
        mu : float
            Proximal regularization parameter.
        w_bar : ndarray
            Current stability center (weight vector).
        b_bar : float
            Current stability center (bias).
        verbose : bool
            If True, prints solver output.

    Returns:
        w_opt : ndarray
            Updated weight vector.
        b_opt : float
            Updated bias.

    Raises:
        RuntimeError: if all solvers fail to return a solution.
    """
    D = len(w_bar)
    w = cp.Variable(D)
    b = cp.Variable()
    xi = cp.Variable()

    constraints = []
    for (w_j, b_j, f_j, grad_w_j, grad_b_j) in bundle:
        affine_expr = f_j + grad_w_j @ (w - w_j) + grad_b_j * (b - b_j)
        constraints.append(xi >= affine_expr)

    objective = cp.Minimize(
        xi + (mu / 2) * (cp.sum_squares(w - w_bar) + cp.square(b - b_bar))
    )

    problem = cp.Problem(objective, constraints)

    # Try ECOS first, fallback to SCS
    solvers = ['ECOS', 'SCS']
    for solver in solvers:
        try:
            problem.solve(solver=solver, verbose=verbose)
            if w.value is not None and b.value is not None:
                return w.value, b.value
            else:
                print(f"Warning: Solver {solver} failed to return solution.")
        except cp.SolverError:
            print(f"Solver {solver} not available or failed.")

    raise RuntimeError("All solvers failed. Install ECOS or check problem setup.")
