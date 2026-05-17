import numpy as np

class HingeZeroMemory:
    """
    HingeZero (HZE) Associative Memory Model
    
    A compact associative memory system designed to converge toward a correct 
    stored pattern under noise using a smooth, deterministic dual-tanh hinge operator 
    [span_0](start_span)[span_1](start_span)and a damped integration recall loop[span_0](end_span)[span_1](end_span).
    """
    def __init__(self, num_nodes: int, alpha: float = 0.5, epsilon: float = 0.1, lambda_val: float = 0.05):
        self.N = num_nodes
        [span_2](start_span)self.alpha = alpha       # Controls the hinge strength (\alpha > 0)[span_2](end_span)
        [span_3](start_span)self.epsilon = epsilon   # Step size (\epsilon)[span_3](end_span)
        [span_4](start_span)[span_5](start_span)self.lambda_val = lambda_val  # Leak/damping term (\lambda)[span_4](end_span)[span_5](end_span)
        
        # [span_6](start_span)Initialize the weight matrix[span_6](end_span)
        self.W = np.zeros((self.N, self.N), dtype=np.float32)
        
    def train(self, patterns: np.ndarray):
        """
        [span_7](start_span)1) Pattern Storage (Hebbian Memory)[span_7](end_span)
        [span_8](start_span)Builds a symmetric weight matrix from P stored patterns[span_8](end_span):
        [span_9](start_span)W = (1/N) * sum( xi^mu * (xi^mu)^T ) with diag(W) = 0[span_9](end_span)
        """
        patterns = np.asarray(patterns, dtype=np.float32)
        P = len(patterns)
        
        # Reset matrix
        self.W = np.zeros((self.N, self.N), dtype=np.float32)
        
        # [span_10](start_span)Outer product accumulation[span_10](end_span)
        for p in patterns:
            self.W += np.outer(p, p)
            
        # [span_11](start_span)Normalize by the number of nodes N[span_11](end_span)
        self.W /= self.N
        
        # [span_12](start_span)Enforce the strict Null-Action Diagonal constraint: diag(W) = 0[span_12](end_span)
        np.fill_diagonal(self.W, 0.0)

    def _phi(self, h: np.ndarray) -> np.ndarray:
        """
        [span_13](start_span)3) HingeZero Mathematical Operator (Dual-Tanh Hinge)[span_13](end_span)
        [span_14](start_span)Element-wise smooth, deterministic nonlinearity applied to the retrieval field[span_14](end_span):
        [span_15](start_span)phi(h)_i = tanh(h_i) + \alpha * tanh(2 * h_i)[span_15](end_span)
        """
        return np.tanh(h) + self.alpha * np.tanh(2.0 * h)

    def recall(self, probe: np.ndarray, steps: int = 50, tolerance: float = 1e-5) -> np.ndarray:
        """
        [span_16](start_span)2) Retrieval Field & 4) Recall Dynamics (Damped Integration)[span_16](end_span)
        [span_17](start_span)Iterative continuous state update[span_17](end_span):
        [span_18](start_span)x_{t+1} = (1 - \lambda)x_t + \epsilon * phi(h_t)[span_18](end_span)
        """
        x = np.array(probe, dtype=np.float32)
        
        for _ in range(steps):
            x_old = np.copy(x)
            
            # [span_19](start_span)h_t = W * x_t[span_19](end_span)
            h = np.dot(self.W, x)
            
            # [span_20](start_span)Apply smooth dual-tanh nonlinearity[span_20](end_span)
            phi_h = self._phi(h)
            
            # [span_21](start_span)Damped integration step[span_21](end_span)
            x = (1.0 - self.lambda_val) * x + self.epsilon * phi_h
            
            # Check for convergence (smooth state stabilizes)
            if np.linalg.norm(x - x_old) < tolerance:
                break
                
        return x

    def calculate_recall_metric(self, final_state: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        [span_22](start_span)6) Recall Metric[span_22](end_span)
        [span_23](start_span)Measures recall quality using cosine similarity to the ground-truth pattern[span_23](end_span):
        [span_24](start_span)cos(x_t, \xi^v)[span_24](end_span)
        """
        dot_product = np.dot(final_state, ground_truth)
        norm_x = np.linalg.norm(final_state)
        norm_gt = np.linalg.norm(ground_truth)
        
        if norm_x == 0 or norm_gt == 0:
            return 0.0
            
        return float(dot_product / (norm_x * norm_gt))

# ==========================================
# VERIFICATION PIPELINE (Matches One-Pager)
# ==========================================
if __name__ == "__main__":
    print("Executing HingeZero One-Pager Verification Loop...")
    
    N = 100  # Number of nodes
    hz = HingeZeroMemory(num_nodes=N, alpha=0.5, epsilon=0.1, lambda_val=0.05)
    
    # [span_25](start_span)5) Stored pattern initialization \xi^\mu[span_25](end_span)
    np.random.seed(42)
    pattern_1 = np.random.choice([-1.0, 1.0], size=N)
    pattern_2 = np.random.choice([-1.0, 1.0], size=N)
    
    hz.train([pattern_1, pattern_2])
    
    # [span_26](start_span)5) Noisy probe initialization: x_0 = \xi^v + \sigma * \eta[span_26](end_span)
    sigma = 0.4
    [span_27](start_span)noise = np.random.normal(0, 1, size=N)  # \eta ~ N(0, I)[span_27](end_span)
    noisy_probe = pattern_1 + sigma * noise
    
    # [span_28](start_span)[span_29](start_span)Run the continuous recall loop[span_28](end_span)[span_29](end_span)
    final_state = hz.recall(noisy_probe, steps=100)
    
    # [span_30](start_span)6) Calculate the final Cosine Similarity metric[span_30](end_span)
    similarity = hz.calculate_recall_metric(final_state, pattern_1)
    
    [span_31](start_span)print(f"-> Setup: Continuous Damped Integration Loop[span_31](end_span)")
    print(f"-> Injected Noise Variance (\sigma): {sigma}")
    print(f"-> Final Convergence Cosine Similarity: {similarity:.4f}")
