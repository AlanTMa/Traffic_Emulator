import pytest
from src.model.config import controller_mode

@pytest.mark.parametrize("sim_cfg, expected", [
    ({"controller_mode": "static_algorithm1"}, "static_algorithm1"),
    ({"controller_mode": "windowed_stochastic"}, "windowed_stochastic"),
    ({}, "windowed_stochastic"),
    # Legacy simulation.mode values; 'asynchronous' never selected an async controller
    ({"mode": "static"}, "static_algorithm1"),
    ({"mode": "synchronous"}, "windowed_stochastic"),
    ({"mode": "asynchronous"}, "windowed_stochastic"),
])
def test_controller_mode_resolution(sim_cfg, expected):
    assert controller_mode({"simulation": sim_cfg}) == expected

def test_unknown_controller_mode_rejected():
    with pytest.raises(ValueError):
        controller_mode({"simulation": {"controller_mode": "async"}})
