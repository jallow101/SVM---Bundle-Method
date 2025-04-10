import numpy as np
import cvxpy as cp
from Functions.subgradient import compute_subgradient  # Adjust according to your file structure


# Existing bundle method, modified to include step_size_strategy argument
def bundle_svm_solver(X, y, C=1.0, mu_0=1.0, tol=1e-4, step_size_strategy="fixed"):
    """
    Solve the primal SVM optimization problem using the Bundle method.
    
    Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Target labels.
        C (float): Regularization parameter.
        mu_0 (float): Initial step size or regularization parameter.
        tol (float): Tolerance for convergence.
        step_size_strategy (str): The step size strategy ('fixed' or 'line_search').
    
    Returns:
        w_opt (ndarray): Optimal weight vector.
        b_opt (float): Optimal bias.
        history (list): History of the objective function values.
    """
    # Initialize parameters
    w = np.zeros(X.shape[1])
    b = 0
    history = []

    # Bundle method parameters
    bundle = []  # A list of tuples (w_j, b_j, f_j, grad_w_j, grad_b_j)
    mu = mu_0
    max_iter = 1000
    k = 0

    # Iterate until convergence or max iterations
    while k < max_iter:
        # Compute the subgradient and objective function (similar to existing code)
        f_new, grad_w_new, grad_b_new = compute_subgradient(w, b, X, y, C)
        
        # Apply the chosen step-size strategy
        if step_size_strategy == "fixed":
            # Fixed step-size: α = 2 / (2 + k)
            alpha = 2 / (2 + k)
        elif step_size_strategy == "line_search":
            # Implementing line search step-size can be more complex
            # You could implement a simple line search or use a fixed small value for now
            # Example: line search can try different values of alpha and choose the best
            alpha = 1.0  # Placeholder for line search; should be replaced with actual search logic.
        else:
            raise ValueError("Unknown step_size_strategy. Use 'fixed' or 'line_search'.")

        # Update weights and bias based on step size
        w = w - alpha * grad_w_new
        b = b - alpha * grad_b_new

        # Update bundle with current solution and subgradient
        bundle.append((w, b, f_new, grad_w_new, grad_b_new))
        
        # Record the objective function value for convergence tracking
        history.append(f_new)

        # Check convergence (simple tolerance check)
        if len(history) > 1 and np.abs(history[-1] - history[-2]) < tol:
            break
        
        # Increment iteration count
        k += 1

    return w, b, history
