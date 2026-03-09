"""Hodgkin-Huxley (HH) neuron model.

The HH model describes action potential generation using voltage-gated
ion channels:

    C * dV/dt = I_ext - I_Na - I_K - I_L

where:
    I_Na = gNa * m^3 * h * (V - E_Na)   sodium current
    I_K  = gK  * n^4 * (V - E_K)        potassium current
    I_L  = gL  * (V - E_L)              leak current

Gating variables m, h, n follow:
    dx/dt = alpha_x(V) * (1 - x) - beta_x(V) * x
"""

import numpy as np


class HHNeuron:
    """Hodgkin-Huxley neuron.

    Parameters
    ----------
    C_m : float
        Membrane capacitance (uF/cm^2). Default: 1.0
    gNa : float
        Maximum sodium conductance (mS/cm^2). Default: 120.0
    gK : float
        Maximum potassium conductance (mS/cm^2). Default: 36.0
    gL : float
        Leak conductance (mS/cm^2). Default: 0.3
    E_Na : float
        Sodium reversal potential (mV). Default: 50.0
    E_K : float
        Potassium reversal potential (mV). Default: -77.0
    E_L : float
        Leak reversal potential (mV). Default: -54.4
    """

    def __init__(
        self,
        C_m=1.0,
        gNa=120.0,
        gK=36.0,
        gL=0.3,
        E_Na=50.0,
        E_K=-77.0,
        E_L=-54.4,
    ):
        if C_m <= 0:
            raise ValueError("C_m must be positive")
        if any(g < 0 for g in [gNa, gK, gL]):
            raise ValueError("Conductances must be non-negative")

        self.C_m = C_m
        self.gNa = gNa
        self.gK = gK
        self.gL = gL
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L

    def _alpha_m(self, V):
        return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))

    def _beta_m(self, V):
        return 4.0 * np.exp(-(V + 65.0) / 18.0)

    def _alpha_h(self, V):
        return 0.07 * np.exp(-(V + 65.0) / 20.0)

    def _beta_h(self, V):
        return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    def _alpha_n(self, V):
        return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))

    def _beta_n(self, V):
        return 0.125 * np.exp(-(V + 65.0) / 80.0)

    def simulate(self, I_ext, dt=0.01, T=100.0):
        """Simulate the Hodgkin-Huxley neuron.

        Parameters
        ----------
        I_ext : float or array-like
            External current (uA/cm^2). If float, constant current.
            If array, must match number of timesteps.
        dt : float
            Time step (ms). Default: 0.01
        T : float
            Total simulation time (ms). Default: 100.0

        Returns
        -------
        times : np.ndarray
            Simulation time points (ms).
        V : np.ndarray
            Membrane potential at each time point (mV).
        spike_times : list
            Times at which spikes occurred (ms).
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        if T <= 0:
            raise ValueError("T must be positive")

        n_steps = int(T / dt)
        times = np.arange(n_steps) * dt

        if np.isscalar(I_ext):
            I = np.full(n_steps, I_ext)
        else:
            I = np.asarray(I_ext)
            if len(I) != n_steps:
                raise ValueError(
                    f"I_ext length {len(I)} does not match "
                    f"n_steps {n_steps} for T={T}, dt={dt}"
                )

        # initialize arrays
        V = np.zeros(n_steps)
        m = np.zeros(n_steps)
        h = np.zeros(n_steps)
        n = np.zeros(n_steps)

        # initial conditions at resting potential
        V_rest = -65.0
        V[0] = V_rest
        m[0] = self._alpha_m(V_rest) / (self._alpha_m(V_rest) + self._beta_m(V_rest))
        h[0] = self._alpha_h(V_rest) / (self._alpha_h(V_rest) + self._beta_h(V_rest))
        n[0] = self._alpha_n(V_rest) / (self._alpha_n(V_rest) + self._beta_n(V_rest))

        spike_times = []
        spike_threshold = 0.0

        for i in range(1, n_steps):
            # gating variable updates
            dm = (self._alpha_m(V[i-1]) * (1 - m[i-1]) - self._beta_m(V[i-1]) * m[i-1]) * dt
            dh = (self._alpha_h(V[i-1]) * (1 - h[i-1]) - self._beta_h(V[i-1]) * h[i-1]) * dt
            dn = (self._alpha_n(V[i-1]) * (1 - n[i-1]) - self._beta_n(V[i-1]) * n[i-1]) * dt

            m[i] = m[i-1] + dm
            h[i] = h[i-1] + dh
            n[i] = n[i-1] + dn

            # currents
            I_Na = self.gNa * m[i]**3 * h[i] * (V[i-1] - self.E_Na)
            I_K = self.gK * n[i]**4 * (V[i-1] - self.E_K)
            I_L = self.gL * (V[i-1] - self.E_L)

            # voltage update
            dV = (I[i-1] - I_Na - I_K - I_L) / self.C_m
            V[i] = V[i-1] + dV * dt

            # spike detection
            if V[i-1] < spike_threshold <= V[i]:
                spike_times.append(times[i])

        return times, V, m, h, n, spike_times

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    neuron = HHNeuron()
    times, V, m, h, n, spikes = neuron.simulate(I_ext=100.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # membrane potential
    ax1.plot(times, V, color='blue')
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title("Hodgkin-Huxley Neuron Simulation")

    # gating variables
    ax2.plot(times, m, label='m (Na act)', color='red')
    ax2.plot(times, h, label='h (Na inact)', color='green')
    ax2.plot(times, n, label='n (K act)', color='orange')
    ax2.set_ylabel("Gating Variable")
    ax2.set_xlabel("Time (ms)")
    ax2.legend()

    plt.tight_layout()
    plt.show()