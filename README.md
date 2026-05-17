
HingeZero treats the neighbourhood around zero as an active equilibrium region, not a passive dead zone.

The core operation uses dual tanh coupling:

H(x, y) = tanh(α(x - z₀)) · tanh(α(y + z₀))

This creates a bounded nonlinear hinge around zero, allowing opposing states to meet, cancel, hold tension, and resolve into stability.

Minimal use:

stable = hingezero_update(query, memory)
