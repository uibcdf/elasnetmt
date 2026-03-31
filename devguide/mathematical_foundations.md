# Mathematical Foundations

ElasNetMT implements analytical solutions for protein dynamics based on the spectral analysis of contact-based matrices.

## 1. Gaussian Network Model (GNM)

The dynamics are governed by the **Kirchhoff Matrix** ($\Gamma$):

-   $\Gamma_{ij} = -1$ if atoms $i$ and $j$ are within the cutoff distance ($R_{ij} < R_c$).
-   $\Gamma_{ii} = \text{degree of node } i$ (number of contacts).
-   $\Gamma_{ij} = 0$ otherwise.

Fluctuations are derived from the pseudo-inverse of $\Gamma$:
$\langle \Delta R_i \cdot \Delta R_j \rangle = \frac{3k_BT}{\gamma} (\Gamma^{-1})_{ij}$

## 2. Anisotropic Network Model (ANM)

The dynamics are governed by the **Hessian Matrix** ($H$), a $3N \times 3N$ matrix:

-   The elements of $H$ are the second derivatives of the harmonic potential $V$.
-   Unlike GNM, ANM provides directional vectors for each mode of vibration.

## 3. Spectral Decomposition

Both matrices are diagonalized:
$M = V \Lambda V^T$

-   **Eigenvalues ($\Lambda$):** Represent the inverse of the variance (square of the frequencies).
-   **Eigenvectors ($V$):** Represent the shape of the normal modes.

The first 6 modes in ANM (and the first in GNM) correspond to rigid body motions (zero eigenvalues) and are discarded in physical analysis.
