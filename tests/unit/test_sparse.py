"""Sparse access topologies (some source->broker links do not exist)."""
import numpy as np
import pytest

from src.controller.feasibility import random_feasible_routing, transportation_feasibility

def test_notebook_sparse_feasibility_regression():
    # Reference notebook sparse_feasibility_regression(): both sources can only
    # reach SN1 (capacity 1) with total demand 2. Aggregate capacity (2) would
    # suggest feasibility; the masked LP must reject it.
    with pytest.raises(ValueError):
        transportation_feasibility(np.array([1.0, 1.0]), np.ones((2, 2)), np.ones(2),
                                   route_mask=np.array([[1, 0], [1, 0]], dtype=bool), margin=1e-9)
    feasible = transportation_feasibility(np.array([1.0, 1.0]), 2.0 * np.ones((2, 2)), np.array([1.2, 1.2]),
                                          route_mask=np.ones((2, 2), dtype=bool), margin=1e-9)
    assert np.max(np.abs(feasible.sum(axis=1) - 1.0)) <= 1e-10
    assert np.max(np.maximum(feasible.sum(axis=0) - 1.2, 0.0)) <= 1e-10

def test_masked_routes_carry_no_flow():
    mask = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
    mu_links = np.where(mask, 5.0, 0.0)          # default mask = mu_links > 0
    L = transportation_feasibility(np.array([3.0, 3.0]), mu_links, np.array([4.0, 4.0, 4.0]))
    assert np.all(L[~mask] == 0.0)
    assert L.sum(axis=1) == pytest.approx([3.0, 3.0])
    R = random_feasible_routing(np.array([3.0, 3.0]), mu_links, np.array([4.0, 4.0, 4.0]), seed=1)
    assert np.all(R[~mask] == 0.0) and R.sum(axis=1) == pytest.approx([3.0, 3.0])
