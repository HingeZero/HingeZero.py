# HingeZero

HingeZero treats the neighbourhood around zero as an active equilibrium region, not a passive dead zone.

The system uses deterministic dual-tanh associative dynamics for stabilised memory recall under noisy or corrupted inputs.

---

## Core Update Rule

```python
h = W @ x

φ(h) = tanh(h) + α·tanh(2h)

x(t+1) = (1 − λ)x(t) + ε·φ(h)
