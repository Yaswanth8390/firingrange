"""Shared pytest fixtures for firingrange test suite."""

import pytest
from net_models.hh import HHNeuron
from net_models.wilson_cowan import WilsonCowanModel


@pytest.fixture
def hh_neuron():
    """Return a default HHNeuron instance."""
    return HHNeuron()


@pytest.fixture
def hh_neuron_fixture():
    """Return a factory for HHNeuron with custom parameters."""
    def _hh_neuron_fixture(C_m=1.0, gNa=120.0, gK=36.0, gL=0.3,
                            E_Na=50.0, E_K=-77.0, E_L=-54.4):
        return HHNeuron(C_m=C_m, gNa=gNa, gK=gK, gL=gL,
                        E_Na=E_Na, E_K=E_K, E_L=E_L)
    return _hh_neuron_fixture


@pytest.fixture
def hh_simulation(hh_neuron):
    """Return a default HH simulation result with constant current."""
    times, V, m, h, n, spike_times = hh_neuron.simulate(I_ext=10.0, dt=0.01, T=100.0)
    return times, V, m, h, n, spike_times


@pytest.fixture
def wc_model():
    """Return a default WilsonCowanModel instance."""
    return WilsonCowanModel()


@pytest.fixture
def wc_model_fixture():
    """Return a factory for WilsonCowanModel with custom parameters."""
    def _wc_model_fixture(tau_e=10.0, tau_i=10.0, w_ee=12.0,
                           w_ei=4.0, w_ie=13.0, w_ii=11.0):
        return WilsonCowanModel(tau_e=tau_e, tau_i=tau_i, w_ee=w_ee,
                                 w_ei=w_ei, w_ie=w_ie, w_ii=w_ii)
    return _wc_model_fixture


@pytest.fixture
def wc_simulation(wc_model):
    """Return a default Wilson-Cowan simulation result."""
    times, E, I = wc_model.simulate(P=1.25, Q=0.0, dt=0.1, T=500.0)
    return times, E, I