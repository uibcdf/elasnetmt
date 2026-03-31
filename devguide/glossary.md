# Glossary

To ensure clarity in the code and documentation, ElasNetMT uses the following definitions:

| Term | Definition |
| :--- | :--- |
| **Node** | A discrete point in the network representing a molecular component (usually a Carbon-alpha or a residue's center of mass). |
| **Spring (or Contact)** | A harmonic constraint between two nodes within the cutoff distance. |
| **Kirchhoff Matrix** | The adjacency matrix used in GNM to describe node connectivity and isotropic fluctuations. |
| **Hessian Matrix** | The second-derivative matrix of the potential energy used in ANM to describe anisotropic (directional) motions. |
| **Normal Mode** | An intrinsic pattern of vibration derived from the eigenvectors of the Kirchhoff or Hessian matrix. |
| **Collectivity** | A measure of how many nodes are involved in a specific normal mode (global vs. local motion). |
| **Cutoff ($R_c$)** | The maximum distance between two nodes to consider them physically connected by a spring. |
| **Scaling Factor ($\gamma$)** | The force constant that scales the magnitude of fluctuations to match experimental data (B-factors). |
