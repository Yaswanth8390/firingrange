# firingrange

A computational neuroscience project exploring how neurons fire and respond to input currents, implementing two foundational models of neural dynamics.

## Models

### Hodgkin-Huxley Model
The Hodgkin-Huxley model describes action potential generation through voltage-gated ion channels. It models the flow of sodium (Na⁺) and potassium (K⁺) ions across the neuronal membrane using gating variables (m, h, n) that control channel opening and closing. This model accurately captures the shape and timing of biological action potentials.

### Wilson-Cowan Model
The Wilson-Cowan model describes the dynamics of interacting excitatory (E) and inhibitory (I) neural populations:
```
tau_e * dE/dt = -E + S(w_ee*E - w_ei*I + P)
tau_i * dI/dt = -I + S(w_ie*E - w_ii*I + Q)
```

where S(x) is the sigmoid activation function, and P, Q are external inputs. The balance between excitation and inhibition produces oscillatory dynamics relevant to understanding EEG/MEG signals — the same signals studied in HNN-Core.

## Features
- Simulate neuron firing under different input currents
- Visualize membrane potential over time
- Track gating variable dynamics (m, h, n) for Hodgkin-Huxley
- Observe excitatory/inhibitory population dynamics for Wilson-Cowan
- Configurable coupling weights and time constants
- pytest test suite for model validation

## Installation
```bash
git clone https://github.com/Yaswanth8390/firingrange
cd firingrange
pip install -r requirements.txt
```

## Usage

### Wilson-Cowan Model
```python
from wilson_cowan import WilsonCowanModel

model = WilsonCowanModel(
    tau_e=10.0, tau_i=10.0,
    w_ee=12.0, w_ei=4.0,
    w_ie=13.0, w_ii=11.0
)

times, E, I = model.simulate(P=1.25, Q=0.0, T=500.0)
```

Or run directly:
```bash
python wilson_cowan.py
```

### Hodgkin-Huxley Model
```bash
python hodgkin_huxley.py
```

## Running Tests
```bash
pytest tests/
```

## Background
## Background
This project was built to develop a solid foundation in computational neuroscience by implementing fundamental neural models from scratch.
