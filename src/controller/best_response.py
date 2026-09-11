"""Threshold best response (eq. 16)."""
import numpy as np

def best_response_mm1(mu_row: np.ndarray, p: np.ndarray, lam_i: float, flow_tol: float = 1e-12, max_iter: int = 320) -> np.ndarray:
    """
    min sum_j x_j/(mu_j - x_j) + p_j x_j s.t. x >= 0, sum x = lam_i, by bisection
    on the multiplier from the smallest activation threshold. ValueError on bad
    input, RuntimeError if the bisection can't meet the tolerance.
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

    # activation thresholds 1/mu_j + p_j
    thresholds = p + 1.0 / mu
    unique_thresholds = np.unique(np.sort(thresholds))

    def allocation_from_pivot(delta, pivot):
        # x = mu (1 - 1/sqrt(q)) with q = mu (alpha - p), alpha = pivot + delta; written as the notebook does
        q = mu * ((pivot - p) + delta)
        active = q > 1.0
        x = np.zeros_like(mu)
        if np.any(active):
            x[active] = -mu[active] * np.expm1(-0.5 * np.log(q[active]))
        return np.clip(x, 0.0, np.nextafter(mu, 0.0))

    # bracket the root
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

def best_response_available(mu_row: np.ndarray, p: np.ndarray, lam_i: float, **kwargs) -> np.ndarray:
    """Best response over the routes that exist (mu_row > 0); missing routes get 0."""
    mu_row = np.asarray(mu_row, dtype=float)
    available = mu_row > 0
    if available.all():
        return best_response_mm1(mu_row, p, lam_i, **kwargs)
    x = np.zeros_like(mu_row)
    x[available] = best_response_mm1(mu_row[available], np.asarray(p, dtype=float)[available], lam_i, **kwargs)
    return x
