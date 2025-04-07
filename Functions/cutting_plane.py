def build_cutting_plane_model(bundle):
    """
    Constructs the cutting-plane approximation of the objective from the bundle.

    Parameters:
        bundle : list of tuples
            Each element is (w_j, b_j, f_j, grad_w_j, grad_b_j)

    Returns:
        planes : list of callable functions
            Each function takes (w, b) and returns the value of one cutting-plane.
    """
    planes = []

    for (w_j, b_j, f_j, grad_w_j, grad_b_j) in bundle:
        def plane_fn(w, b, w_j=w_j, b_j=b_j, f_j=f_j, grad_w_j=grad_w_j, grad_b_j=grad_b_j):
            delta_w = w - w_j
            delta_b = b - b_j
            linear_term = grad_w_j @ delta_w + grad_b_j * delta_b
            return f_j + linear_term
        planes.append(plane_fn)

    return planes
