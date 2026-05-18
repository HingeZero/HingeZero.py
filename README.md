# HingeZero

HingeZero treats the neighbourhood around zero as an active equilibrium region, not a passive dead zone.

This implementation is the scalable production-style version of HingeZero.

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

---

# Scalable Recall Pipeline

```text
Memory Bank (P × N)
        ↓
Candidate Search
        ↓
Top-K Selection
        ↓
Local Retrieval Field
        ↓
HingeZero Refinement
        ↓
Final Recall
```

Avoids:

```text
W ∈ R^(N×N)
```

---

# Recall Update

```text
x(t+1) = (1−λ)x(t) + ε·φ(h)
```

---

# Example

```python
import numpy as np
from hingezero import HingeZeroMemory

rng = np.random.default_rng(0)

N = 512
P = 10000

memory = rng.choice(
    [-1.0, 1.0],
    size=(P, N)
)

hz = HingeZeroMemory(
    metric="cosine"
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

print("index:", result.index)
print("similarity:", result.similarity)
print("margin:", result.margin)
```

---

# Design Goals

- Stable recall under noise
- Corrupted input recovery
- Large memory banks
- Deterministic retrieval
- Quantised memory experiments

---

# Validation

Minimal:

```text
h = W @ x
φ(h)=tanh(h)+α·tanh(2h)
x(t+1)=(1−λ)x(t)+εφ(h)
```

Scalable:

```text
Memory → Search → Top-K → HZ refinement
```

---

Public description:

HingeZero is a deterministic associative-memory stabilisation layer designed for robust recall under noisy or corrupted inputs.

The scalable implementation avoids dense matrices and performs:

Memory Bank → Candidate Search → Top-K → HingeZero Refinement

---

David Duffy  
HingeZero Ltd
