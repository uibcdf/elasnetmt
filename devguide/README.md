# ElasNetMT Developer Guide

Welcome to the **ElasNetMT** developer guide. This document serves as the entry point for understanding the technical philosophy, architecture, and standards of the library within the **MolSysSuite** ecosystem.

## Vision

ElasNetMT is the dynamic engine of the MolSysSuite. While **MolSysMT** provides the structural foundation, ElasNetMT breathes life into those structures by studying their intrinsic dynamics through Elastic Network Models (ENM). Our goal is to provide a physically rigorous yet computationally efficient framework to explore protein flexibility, vibrational modes, and conformational changes.

## Guiding Principles

- **Physical Rigor:** Every algorithm must be grounded in established ENM theory (Gaussian and Anisotropic models).
- **Interoperability:** Seamlessly consume objects from `molsysmt` and produce visual states for `molsysviewer`.
- **Dimensional Consistency:** All physical quantities must be handled via `pyunitwizard`.
- **Structured Diagnostics:** All warnings and errors must follow the `smonitor` standard.

## Installation for Development

To install ElasNetMT for development, it is recommended to use the following command to avoid dependency conflicts:

```bash
pip install --no-deps --editable .
```

- [Architecture](architecture.md): Class hierarchy and data flow.
- [Roadmap](roadmap.md): Technical milestones toward version 1.0.0.
- [Mathematical Foundations](mathematical_foundations.md): Kirchhoff and Hessian matrices, spectral analysis.
- [SMonitor Integration](smonitor_integration.md): Diagnostic codes and signals.
- [Units and Conventions](units_and_conventions.md): Physical quantities and force constants.
- [Visualization Protocols](visualization_protocols.md): Rendering springs and normal modes.
- [Testing Strategy](testing_strategy.md): Tiers of verification (Smoke, Physical, Regression).
