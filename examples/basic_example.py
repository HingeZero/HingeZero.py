"""Basic example of HingeZero memory usage."""

import numpy as np
from hingezero import HingeZeroMemory


def main():
    """Run basic example."""
    # Set up random number generator
    rng = np.random.default_rng(0)

    # Configuration
    N = 512  # Dimension
    P = 10000  # Number of patterns
    top_k = 64
    trials = 10
    noise_sigma = 0.5

    print("=== HingeZero Basic Example ===")
    print(f"Dimension: {N}")
    print(f"Memory size: {P:,} patterns")
    print(f"Top-K: {top_k}")
    print(f"Noise level: σ={noise_sigma}\n")

    # Create random memory bank (binary patterns)
    print("Creating memory bank...")
    memory = rng.choice([-1.0, 1.0], size=(P, N)).astype(np.float32)

    # Initialize and fit HingeZero
    print("Fitting HingeZero memory...")
    hz = HingeZeroMemory(metric="cosine")
    hz.fit(memory)

    # Run recall tests
    print(f"\nRunning {trials} recall trials:\n")
    correct = 0
    margins = []

    for trial in range(trials):
        # Select random target pattern
        target_idx = int(rng.integers(0, P))
        target = memory[target_idx]

        # Create noisy query
        query = target + rng.normal(0.0, noise_sigma, size=N).astype(np.float32)

        # Perform recall
        result = hz.recall(
            query,
            top_k=top_k,
            steps=12,
            alpha=0.25,
            eps=0.10,
            lam=0.02,
            beta=8.0,
            normalize=True,
        )

        # Check correctness
        hit = result.index == target_idx
        correct += int(hit)
        margins.append(result.margin)

        print(
            f"Trial {trial + 1:2d}: "
            f"target={target_idx:5d} | "
            f"retrieved={result.index:5d} | "
            f"hit={'✓' if hit else '✗'} | "
            f"margin={result.margin:.6f} | "
            f"similarity={result.similarity:.4f}"
        )

    # Print results
    accuracy = correct / trials
    mean_margin = np.mean(margins)
    print(f"\n=== Results ===")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{trials})")
    print(f"Mean margin: {mean_margin:.6f}")
    print(f"Memory usage: {memory.nbytes / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
