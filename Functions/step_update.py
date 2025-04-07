def update_stability_center(
    f_new, f_old,
    w_new, b_new,
    w_old, b_old,
    mu_k, epsilon=1e-4, gamma=2.0
):
    """
    Decide serious vs. null step and update stability center and proximal parameter.

    Parameters:
        f_new : float
            Objective value at (w_new, b_new)
        f_old : float
            Objective value at current stability center (w_old, b_old)
        w_new, b_new : ndarray, float
            Candidate new point
        w_old, b_old : ndarray, float
            Current stability center
        mu_k : float
            Current proximal parameter
        epsilon : float
            Tolerance for serious step condition
        gamma : float
            Proximal update multiplier

    Returns:
        is_serious_step : bool
            Whether a serious step was taken
        w_bar_new, b_bar_new : updated stability center
        mu_new : updated proximal parameter
    """
    if f_new < f_old - epsilon:
        # Serious step
        return True, w_new, b_new, mu_k * gamma
    else:
        # Null step
        return False, w_old, b_old, mu_k / gamma
