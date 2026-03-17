"""Tests for the Hodgkin-Huxley neuron model."""

import numpy as np
import pytest
from net_models.hh import HHNeuron


def test_hh_default_init():
    """Test HHNeuron initializes with correct default parameters."""
    neuron = HHNeuron()
    assert neuron.C_m == 1.0
    assert neuron.gNa == 120.0
    assert neuron.gK == 36.0
    assert neuron.gL == 0.3
    assert neuron.E_Na == 50.0
    assert neuron.E_K == -77.0
    assert neuron.E_L == -54.4


def test_hh_custom_init(hh_neuron_fixture):
    """Test HHNeuron initializes with custom parameters."""
    neuron = hh_neuron_fixture(C_m=2.0, gNa=100.0)
    assert neuron.C_m == 2.0
    assert neuron.gNa == 100.0


def test_hh_invalid_capacitance():
    """Test HHNeuron raises ValueError for non-positive capacitance."""
    with pytest.raises(ValueError, match="C_m must be positive"):
        HHNeuron(C_m=0.0)


def test_hh_invalid_conductance():
    """Test HHNeuron raises ValueError for negative conductance."""
    with pytest.raises(ValueError, match="Conductances must be non-negative"):
        HHNeuron(gNa=-1.0)


def test_hh_simulate_output_shape(hh_neuron):
    """Test simulate returns arrays of correct shape."""
    times, V, m, h, n, spike_times = hh_neuron.simulate(I_ext=10.0, dt=0.01, T=100.0)
    n_steps = int(100.0 / 0.01)
    assert len(times) == n_steps
    assert len(V) == n_steps
    assert len(m) == n_steps
    assert len(h) == n_steps
    assert len(n) == n_steps


def test_hh_simulate_no_current(hh_neuron):
    """Test no spikes fired with zero external current."""
    _, _, _, _, _, spike_times = hh_neuron.simulate(I_ext=0.0, dt=0.01, T=100.0)
    assert len(spike_times) == 0


def test_hh_simulate_produces_spikes(hh_neuron):
    """Test spikes are produced with sufficient current."""
    _, _, _, _, _, spike_times = hh_neuron.simulate(I_ext=10.0, dt=0.01, T=100.0)
    assert len(spike_times) > 0


def test_hh_gating_variables_bounded(hh_simulation):
    """Test gating variables m, h, n stay within [0, 1]."""
    _, _, m, h, n, _ = hh_simulation
    assert np.all(m >= 0) and np.all(m <= 1)
    assert np.all(h >= 0) and np.all(h <= 1)
    assert np.all(n >= 0) and np.all(n <= 1)


def test_hh_simulate_invalid_dt(hh_neuron):
    """Test simulate raises ValueError for non-positive dt."""
    with pytest.raises(ValueError, match="dt must be positive"):
        hh_neuron.simulate(I_ext=10.0, dt=0.0)


def test_hh_simulate_invalid_T(hh_neuron):
    """Test simulate raises ValueError for non-positive T."""
    with pytest.raises(ValueError, match="T must be positive"):
        hh_neuron.simulate(I_ext=10.0, T=0.0)


def test_hh_simulate_array_current(hh_neuron):
    """Test simulate accepts array current input."""
    n_steps = int(100.0 / 0.01)
    I_ext = np.full(n_steps, 10.0)
    times, V, m, h, n, spike_times = hh_neuron.simulate(I_ext=I_ext, dt=0.01, T=100.0)
    assert len(times) == n_steps


def test_hh_simulate_array_current_wrong_length(hh_neuron):
    """Test simulate raises ValueError for wrong length array current."""
    with pytest.raises(ValueError):
        hh_neuron.simulate(I_ext=np.array([1.0, 2.0]), dt=0.01, T=100.0)