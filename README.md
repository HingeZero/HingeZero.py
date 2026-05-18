# HingeZero

HingeZero treats the neighbourhood around zero as an active equilibrium region, not a passive dead zone.

This implementation is the **scalable production-style version** of HingeZero.

Unlike the minimal validation core, this version avoids dense `N × N` Hebbian matrices and instead uses:

- Direct memory-bank storage
- Top-K retrieval
- Local candidate refinement
- Deterministic HingeZero dynamics
- Noise-robust associative recall

---

# Core HingeZero Operator

```text
φ(h) = tanh(h) + α·tanh(2h)
```

The hinge operator is applied during recall refinement.

---

# Scalable Recall Pipeline

```text
Memory Bank (P × N)

        ↓

Candidate Search
(cosine / dot)

        ↓

Top-K Selection

        ↓

Local Retrieval Field

        ↓

HingeZero Refinement

        ↓

Final Recall
```

This avoids building:

```text
W ∈ R^(N×N)
```

allowing larger memory banks.

---

# Recall Update Rule

```text
x(t+1) = (1 − λ)x(t) + ε·φ(h)
```

where:

```text
h = local retrieval field
```

and:

```text
φ(h) = tanh(h) + α·tanh(2h)
```

---

# Example

```python
import numpy as np

rng = np.random.default_rng(0)

N = 512
P = 100000

memory = rng.choice(
    [-1.0, 1.0],
    size=(P, N)
)

hz.fit(memory)

target = memory[0]

query = target + rng.normal(
    0,
    0.5,
    size=N
)

result = hz.recall(
    query,
    top_k=64
)

print(result.index)
print(result.similarity)
```

---

# Design Goals

HingeZero is designed for:

- Stable recall under noise
- Corrupted input recovery
- Large memory banks
- Low precision systems
- Deterministic retrieval
- Quantised memory experiments

---

# Validation Notes

Minimal validation implementation:

```text
h = W @ x

φ(h) = tanh(h) + α·tanh(2h)

x(t+1) = (1−λ)x(t) + ε·φ(h)
```

Scalable implementation:

```text
Memory bank → Top-K → HingeZero refinement
```

Both remain deterministic.

---

# Current Status

Current repository contains:

✓ Scalable memory-bank implementation

✓ Production-style retrieval

✓ Deterministic refinement

✓ Large-scale testing support

✓ Noise validation framework

---

## Public Description

HingeZero is a deterministic associative-memory stabilisation layer designed for robust recall under noisy or corrupted inputs.

The scalable implementation avoids dense `N × N` weight matrices and instead performs:

Memory Bank → Candidate Search → Top-K → HingeZero Refinement

This enables larger memory banks while preserving deterministic recall behaviour.

---

Founder:

David Duffy  
HingeZero Ltd
