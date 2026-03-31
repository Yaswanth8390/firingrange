"""Wilson-Cowan population model.

The Wilson-Cowan model describes the dynamics of interacting
excitatory (E) and inhibitory (I) neural populations:

    tau_e * dE/dt = -E + S(w_ee*E - w_ei*I + P)
    tau_i * dI/dt = -I + S(w_ie*E - w_ii*I + Q)

where S(x) = 1 / (1 + exp(-x)) is the sigmoid activation function,
P and Q are external inputs to E and I populations respectively.
"""

import numpy as np


def sigmoid(x):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-x))


class WilsonCowanModel:
    """Wilson-Cowan neural population model.

    Parameters
    ----------
    tau_e : float
        Time constant of excitatory population (ms). Default: 10.0
    tau_i : float
        Time constant of inhibitory population (ms). Default: 10.0
    w_ee : float
        Excitatory-to-excitatory coupling weight. Default: 12.0
    w_ei : float
        Inhibitory-to-excitatory coupling weight. Default: 4.0
    w_ie : float
        Excitatory-to-inhibitory coupling weight. Default: 13.0
    w_ii : float
        Inhibitory-to-inhibitory coupling weight. Default: 11.0
    """

    def __init__(
        self,
        tau_e=10.0,
        tau_i=10.0,
        w_ee=12.0,
        w_ei=4.0,
        w_ie=13.0,
        w_ii=11.0,
    ):
        if tau_e <= 0 or tau_i <= 0:
            raise ValueError("Time constants must be positive")
        if any(w < 0 for w in [w_ee, w_ei, w_ie, w_ii]):
            raise ValueError("Coupling weights must be non-negative")

        self.tau_e = tau_e
        self.tau_i = tau_i
        self.w_ee = w_ee
        self.w_ei = w_ei
        self.w_ie = w_ie
        self.w_ii = w_ii

    def simulate(self, P=1.25, Q=0.0, E0=0.1, I0=0.05, dt=0.1, T=500.0):
        """Simulate the Wilson-Cowan model.

        Parameters
        ----------
        P : float or array-like
            External input to excitatory population. Default: 1.25
        Q : float or array-like
            External input to inhibitory population. Default: 0.0
        E0 : float
            Initial excitatory activity. Default: 0.1
        I0 : float
            Initial inhibitory activity. Default: 0.05
        dt : float
            Time step (ms). Default: 0.1
        T : float
            Total simulation time (ms). Default: 500.0

        Returns
        -------
        times : np.ndarray
            Simulation time points (ms).
        E : np.ndarray
            Excitatory population activity at each time point.
        I : np.ndarray
            Inhibitory population activity at each time point.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        if T <= 0:
            raise ValueError("T must be positive")
        if not (0.0 <= E0 <= 1.0):
            raise ValueError("E0 must be between 0 and 1")
        if not (0.0 <= I0 <= 1.0):
            raise ValueError("I0 must be between 0 and 1")

        n_steps = int(T / dt)
        times = np.arange(n_steps) * dt

        if np.isscalar(P):
            P_arr = np.full(n_steps, P)
        else:
            P_arr = np.asarray(P)
            if len(P_arr) != n_steps:
                raise ValueError(
                    f"P length {len(P_arr)} does not match n_steps {n_steps}"
                )

        if np.isscalar(Q):
            Q_arr = np.full(n_steps, Q)
        else:
            Q_arr = np.asarray(Q)
            if len(Q_arr) != n_steps:
                raise ValueError(
                    f"Q length {len(Q_arr)} does not match n_steps {n_steps}"
                )

        E = np.zeros(n_steps)
        I = np.zeros(n_steps)
        E[0] = E0
        I[0] = I0

        for i in range(1, n_steps):
            dE = (-E[i-1] + sigmoid(
                self.w_ee * E[i-1] - self.w_ei * I[i-1] + P_arr[i-1]
            )) / self.tau_e
            dI = (-I[i-1] + sigmoid(
                self.w_ie * E[i-1] - self.w_ii * I[i-1] + Q_arr[i-1]
            )) / self.tau_i

            E[i] = E[i-1] + dE * dt
            I[i] = I[i-1] + dI * dt

        return times, E, I
    if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Initialize model with default parameters
    model = WilsonCowanModel(
        tau_e=10.0, tau_i=10.0,
        w_ee=12.0, w_ei=4.0,
        w_ie=13.0, w_ii=11.0
    )

    # Run simulation
    times, E, I = model.simulate(P=1.25, Q=0.0, T=500.0)

    # Plot results
    plt.figure(figsize=(10, 4))
    plt.plot(times, E, label='Excitatory (E)', color='red')
    plt.plot(times, I, label='Inhibitory (I)', color='blue')
    plt.xlabel('Time (ms)')
    plt.ylabel('Population Activity')
    plt.title('Wilson-Cowan Neural Population Dynamics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
