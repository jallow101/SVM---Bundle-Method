import numpy as np

def compute_subgradient(w, b, X, y, C):
    """
    Compute the subgradient and objective value for the primal SVM at (w, b).
    
    Parameters:
        w : ndarray of shape (D,)
            Current weight vector.
        b : float
            Current bias term.
        X : ndarray of shape (n_samples, D)
            Feature matrix (each row is phi(x_i)).
        y : ndarray of shape (n_samples,)
            Labels (+1 or -1).
        C : float
            Regularization parameter.

    Returns:
        f_val : float
            Objective value at (w, b).
        grad_w : ndarray
            Subgradient with respect to w.
        grad_b : float
            Subgradient with respect to b.
    """
    n_samples = X.shape[0]
    margins = y * (X @ w + b)
    indicator = margins < 1  # where hinge loss is active

    # Hinge loss values
    hinge_losses = np.maximum(0, 1 - margins)
    loss_term = C * np.sum(hinge_losses)
    
    # Objective value
    reg_term = 0.5 * np.dot(w, w)
    f_val = reg_term + loss_term

    # Subgradients
    grad_w = w - C * (X[indicator].T @ (y[indicator]))
    grad_b = -C * np.sum(y[indicator])

    return f_val, grad_w, grad_b
