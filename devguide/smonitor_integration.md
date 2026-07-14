# SMonitor Integration

ElastNetMT follows the **SMonitor** standard for diagnostics and telemetry.

## Diagnostic Codes

| Code | Level | Title | Description / Hint |
| :--- | :--- | :--- | :--- |
| `ENM-W001` | WARNING | Isolated Nodes | Some nodes have no contacts. Increase the `cutoff`. |
| `ENM-W005` | WARNING | Low Connectivity | Average degree is < 4.0. Network might be unstable. |
| `ENM-W010` | WARNING | Low Correlation | Theoretical B-factors have poor correlation with experimental ones. |
| `ENM-W015` | WARNING | Small Spectral Gap | Small difference between rigid and vibrational modes. Check constraints. |
| `ENM-E020` | ERROR | Singular Matrix | System is mechanically disconnected (singular Kirchhoff/Hessian). |
| `ENM-E030` | ERROR | Negative Eigenvalues | System not at a local minimum (unstable). Minimize structure. |

## Telemetry Signals

- **`elastnetmt.model.selection` (DEBUG):** Reports `n_nodes` and degree distribution (mean/std).
- **`elastnetmt.model.spectral_stats` (DEBUG):** Reports max rigid eigenvalue and spectral gap.
- **`elastnetmt.model.make_model` (INFO):** Reports engine used, node count, and calculation time.
