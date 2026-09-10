"""
Source best-response solver for the M/M/1 queueing model.
"""
import numpy as np

def best_response_mm1(mu_row: np.ndarray, p: np.ndarray, lam_i: float, flow_tol: float = 1e-12, max_iter: int = 320) -> np.ndarray:
    """
    Exact M/M/1 source best response using activation thresholds.

    Solves min sum_j x_j/(mu_j-x_j) + p_j x_j subject to x>=0 and sum_j x_j = lam_i.
    The multiplier search starts at the minimum activation threshold.

    Raises ValueError for invalid inputs (same checks as the reference
    notebook's best_response_mm1) and RuntimeError if the bisection cannot
    meet the conservation tolerance.
    """
    mu = np.asarray(mu_row, dtype=float)
    p = np.asarray(p, dtype=float)
    lam_i = float(lam_i)

    if mu.ndim != 1 or p.shape != mu.shape:
        raise ValueError("mu_row and p must be one-dimensional arrays of equal size")
    if np.any(~np.isfinite(mu)) or np.any(mu <= 0):
        raise ValueError("all access capacities must be finite and positive")
    if np.any(~np.isfinite(p)):
        raise ValueError("all prices must be finite")
    if not np.isfinite(lam_i):
        raise ValueError("source demand must be finite")
    if lam_i < 0 or lam_i >= mu.sum():
        raise ValueError("source demand must lie in [0, sum(mu_row))")
    if flow_tol <= 0 or not np.isfinite(flow_tol):
        raise ValueError("flow_tol must be finite and strictly positive")

    if lam_i == 0:
        return np.zeros_like(mu)

    # Activation thresholds: M_ij = C_ij(0) + p_j = 1/mu_ij + p_j
    thresholds = p + 1.0 / mu
    unique_thresholds = np.unique(np.sort(thresholds))

    def allocation_from_pivot(delta, pivot):
        # q = mu * (alpha - p)
        # alpha = pivot + delta
        q = mu * ((pivot - p) + delta)
        active = q > 1.0
        x = np.zeros_like(mu)
        if np.any(active):
            # x = mu * (1 - 1/sqrt(q))
            # Notebook uses: -mu * np.expm1(-0.5 * np.log(q))
            x[active] = -mu[active] * np.expm1(-0.5 * np.log(q[active]))
        return np.clip(x, 0.0, np.nextafter(mu, 0.0))

    # Bracket the root
    pivot = float(unique_thresholds[0])
    lo = 0.0
    hi = None
    for next_threshold in unique_thresholds[1:]:
        candidate_hi = float(next_threshold - pivot)
        if allocation_from_pivot(candidate_hi, pivot).sum() >= lam_i:
            hi = candidate_hi
            break
        pivot = float(next_threshold)

    if hi is None:
        hi = 1.0
        for _ in range(1024):
            if allocation_from_pivot(hi, pivot).sum() >= lam_i:
                break
            hi *= 2.0
            if not np.isfinite(hi):
                raise RuntimeError("failed to bracket the source best response")
        else:
            raise RuntimeError("failed to bracket the source best response")
    else:
        hi = float(hi)

    tolerance = flow_tol * max(1.0, abs(lam_i))
    best_x = None
    best_error = np.inf

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        x_mid = allocation_from_pivot(mid, pivot)
        residual = float(x_mid.sum() - lam_i)
        error = abs(residual)

        if error < best_error:
            best_x, best_error = x_mid, error

        if error <= tolerance:
            return x_mid

        if mid == lo or mid == hi:
            break

        if residual >= 0.0:
            hi = mid
        else:
            lo = mid

    if best_x is None or best_error > tolerance:
        raise RuntimeError(
            f"source best response did not meet conservation tolerance: "
            f"residual={best_error:.3e}, tolerance={tolerance:.3e}"
        )

    return best_x
