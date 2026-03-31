# Performance and Scalability

ENM calculations are computationally intensive, especially for large systems. ElasNetMT aims to be efficient for systems up to several thousand nodes.

## Optimization Strategy

### Vectorization
Loops in Python are strictly prohibited for core mathematical operations. The following must be **vectorized** using NumPy/SciPy:
- **Hessian Matrix Construction:** Avoid triple/quadruple loops. Use `molsysmt` distance matrix utilities or `np.einsum` to build the Hessian.
- **B-factor Scaling:** Use vectorized linear algebra for fitting modeled to experimental B-factors.

### Lazy Solver
Diagonalization (`np.linalg.eigh`) is the main bottleneck. We follow a "calculate once, reuse many" approach:
1.  Check if `self._eigenvalues` exists before re-calculating.
2.  If parameters like `cutoff` or `selection` change, reset the internal state (`self._eigenvalues = None`).

## Scalability Guidelines (Current Benchmark)

| Model | Max Nodes (Standard PC) | Recommendation |
| :--- | :--- | :--- |
| **GNM** | ~15,000 nodes | Suitable for massive complexes. |
| **ANM** | ~5,000 nodes | Hessian grows at $3N \times 3N$. Memory usage is significant. |

For systems larger than these limits, we encourage the use of **Coarse-Grained (CG) selections** (e.g., residues as single nodes) instead of full alpha-carbon models.

## Memory Management
Ensure that large matrices (Kirchhoff, Hessian, Inverse) are deleted or set to `None` if the model is re-parameterized to avoid memory leaks.
