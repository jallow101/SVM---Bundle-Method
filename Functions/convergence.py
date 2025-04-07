import numpy as np

def check_convergence(grad_w, grad_b, tol=1e-4):
    """
    Check convergence based on the norm of the subgradient.

    Parameters:
        grad_w : ndarray
            Subgradient with respect to w
        grad_b : float
            Subgradient with respect to b
        tol : float
            Convergence tolerance

    Returns:
        bool : True if norm is below tol
    """
    grad_norm = np.sqrt(np.linalg.norm(grad_w)**2 + grad_b**2)
    return grad_norm < tol
