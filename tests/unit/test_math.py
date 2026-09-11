import numpy as np
import pytest
from src.model.marginal_costs import mm1_marginal_cost, mm1_marginal_cost_vectorized
from src.controller.best_response import best_response_mm1

def test_mm1_marginal_cost():
    # Case: mu=10, x=5 -> C = 10/(10-5)^2 = 10/25 = 0.4
    assert mm1_marginal_cost(5.0, 10.0) == pytest.approx(0.4)
    # Case: mu=10, x=9 -> C = 10/1^2 = 10.0
    assert mm1_marginal_cost(9.0, 10.0) == pytest.approx(10.0)

def test_best_response_simple():
    # Case: 2 brokers, identical mu, identical p
    mu = np.array([10.0, 10.0])
    p = np.array([1.0, 1.0])
    lam_i = 4.0
    # Should split equally: [2.0, 2.0]
    res = best_response_mm1(mu, p, lam_i)
    assert np.allclose(res, [2.0, 2.0])

def test_best_response_asymmetric():
    # Case: Broker 1 is much cheaper (low price)
    mu = np.array([10.0, 10.0])
    p = np.array([0.1, 10.0])
    lam_i = 2.0
    # Should heavily favor broker 0
    res = best_response_mm1(mu, p, lam_i)
    assert res[0] > res[1]
    assert np.sum(res) == pytest.approx(lam_i)

def test_best_response_capacity_limit():
    # Case: One broker is very small
    mu = np.array([1.0, 10.0])
    p = np.array([0.0, 0.0])
    lam_i = 5.0
    # Must put most traffic on broker 1 because broker 0 capacity is 1.0
    res = best_response_mm1(mu, p, lam_i)
    assert res[0] < 1.0
    assert res[1] > 4.0
    assert np.sum(res) == pytest.approx(lam_i)

@pytest.mark.parametrize("mu, p, lam, match", [
    (np.array([[1.0, 2.0]]), np.array([0.0, 0.0]), 1.0, "one-dimensional"),
    (np.array([1.0, 2.0]), np.array([0.0]), 1.0, "one-dimensional"),
    (np.array([1.0, 0.0]), np.array([0.0, 0.0]), 0.5, "finite and positive"),
    (np.array([1.0, np.inf]), np.array([0.0, 0.0]), 0.5, "finite and positive"),
    (np.array([1.0, 2.0]), np.array([0.0, np.nan]), 0.5, "prices must be finite"),
    (np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.inf, "demand must be finite"),
    (np.array([1.0, 2.0]), np.array([0.0, 0.0]), -0.1, r"\[0, sum\(mu_row\)\)"),
    (np.array([1.0, 2.0]), np.array([0.0, 0.0]), 3.0, r"\[0, sum\(mu_row\)\)"),
])
def test_best_response_rejects_invalid_inputs(mu, p, lam, match):
    with pytest.raises(ValueError, match=match):
        best_response_mm1(mu, p, lam)

def test_best_response_rejects_bad_tolerance():
    with pytest.raises(ValueError, match="flow_tol"):
        best_response_mm1(np.array([1.0, 2.0]), np.zeros(2), 1.0, flow_tol=0.0)
