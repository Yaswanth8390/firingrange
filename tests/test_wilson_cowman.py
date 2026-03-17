"""Tests for the Wilson-Cowan neural population model."""

import numpy as np
import pytest
from net_models.wilson_cowan import WilsonCowanModel, sigmoid


def test_sigmoid_zero():
    """Test sigmoid(0) == 0.5."""
    assert sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_bounded():
    """Test sigmoid output is bounded between 0 and 1."""
    x = np.linspace(-10, 10, 100)
    y = sigmoid(x)
    assert np.all(y > 0) and np.all(y < 1)


def test_wc_default_init():
    """Test WilsonCowanModel initializes with correct default parameters."""
    model = WilsonCowanModel()
    assert model.tau_e == 10.0
    assert model.tau_i == 10.0
    assert model.w_ee == 12.0
    assert model.w_ei == 4.0
    assert model.w_ie == 13.0
    assert model.w_ii == 11.0


def test_wc_custom_init(wc_model_fixture):
    """Test WilsonCowanModel initializes with custom parameters."""
    model = wc_model_fixture(tau_e=5.0, w_ee=10.0)
    assert model.tau_e == 5.0
    assert model.w_ee == 10.0


def test_wc_invalid_tau():
    """Test WilsonCowanModel raises ValueError for non-positive time constants."""
    with pytest.raises(ValueError, match="Time constants must be positive"):
        WilsonCowanModel(tau_e=0.0)


def test_wc_invalid_weights():
    """Test WilsonCowanModel raises ValueError for negative weights."""
    with pytest.raises(ValueError, match="Coupling weights must be non-negative"):
        WilsonCowanModel(w_ee=-1.0)


def test_wc_simulate_output_shape(wc_model):
    """Test simulate returns arrays of correct shape."""
    times, E, I = wc_model.simulate(dt=0.1, T=500.0)
    n_steps = int(500.0 / 0.1)
    assert len(times) == n_steps
    assert len(E) == n_steps
    assert len(I) == n_steps


def test_wc_simulate_activity_bounded(wc_simulation):
    """Test E and I activity stays within reasonable bounds."""
    _, E, I = wc_simulation
    assert np.all(E >= 0)
    assert np.all(I >= 0)


def test_wc_simulate_reaches_steady_state(wc_simulation):
    """Test that activity stabilizes toward end of simulation."""
    _, E, I = wc_simulation
    # check last 10% of simulation has low variance
    tail = int(len(E) * 0.9)
    assert np.std(E[tail:]) < 0.01
    assert np.std(I[tail:]) < 0.01


def test_wc_simulate_invalid_dt(wc_model):
    """Test simulate raises ValueError for non-positive dt."""
    with pytest.raises(ValueError, match="dt must be positive"):
        wc_model.simulate(dt=0.0)


def test_wc_simulate_invalid_T(wc_model):
    """Test simulate raises ValueError for non-positive T."""
    with pytest.raises(ValueError, match="T must be positive"):
        wc_model.simulate(T=0.0)


def test_wc_simulate_invalid_initial_E(wc_model):
    """Test simulate raises ValueError for E0 out of range."""
    with pytest.raises(ValueError, match="E0 must be between 0 and 1"):
        wc_model.simulate(E0=1.5)


def test_wc_simulate_array_input(wc_model):
    """Test simulate accepts array external input."""
    n_steps = int(500.0 / 0.1)
    P = np.full(n_steps, 1.25)
    times, E, I = wc_model.simulate(P=P, dt=0.1, T=500.0)
    assert len(times) == n_steps


def test_wc_simulate_array_wrong_length(wc_model):
    """Test simulate raises ValueError for wrong length array input."""
    with pytest.raises(ValueError):
        wc_model.simulate(P=np.array([1.0, 2.0]), dt=0.1, T=500.0)


def test_wc_zero_input_decays(wc_model):
    """Test activity decays to low values with zero external input."""
    _, E, I = wc_model.simulate(P=0.0, Q=0.0, E0=0.5, I0=0.5, dt=0.1, T=500.0)
    assert E[-1] < 0.1
    assert I[-1] < 0.1