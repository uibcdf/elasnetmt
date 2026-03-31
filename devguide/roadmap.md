# ElasNetMT Roadmap

This roadmap outlines the evolution of **ElasNetMT** from its current state to a high-performance, AI-ready dynamics engine for the **MolSysSuite**.

## Phase 1: Structural Foundations & Suite Alignment
*Status: COMPLETED*

- [x] **Inheritance Refactoring:** `ElasticNetworkModel` base class unifies `GNM` and `ANM`.
- [x] **Contact Map Centralization:** Moved to `_private/contacts.py` with unit-aware normalization.
- [x] **Suite Integration:** 
    - Full `@arg_digest()` and `@dep_digest()` coverage.
    - Integrated `lindelint` for full-atom trajectory generation.
- [x] **SMonitor Diagnostics:** Deep instrumentation for network integrity and spectral health.
- [x] **Tiered Testing:** Basic `smoke` and `integration` tests implemented and passing.

## Phase 2: Performance & Physics Refinement
*Status: IN PROGRESS*

- [x] **Hessian Vectorization:** Implemented Level 1 (NumPy) and Level 2 (Numba).
- [x] **GPU Acceleration:** Level 3 (CuPy) implemented for spectral decomposition.
- [ ] **Lazy Evaluation Engine:** Refine dormant state triggers (Mostly done in Phase 1).
- [ ] **B-Factor Engine:** Vectorize scaling factor calculations in GNM.

## Phase 3: Drug Discovery & Pocket Dynamics
*Focus: Integration with TopoMT and PharmacophoreMT.*

- [ ] **Ligand Perturbation:** Implement `model.add_ligand()` to add non-protein nodes to the Hessian matrix.
- [ ] **Pocket Breathing:** Develop tools to track pocket volume (TopoMT) along mode trajectories.
- [ ] **Dynamic Pharmacophores:** Define tolerance ellipsoids for pharmacophoric points based on local flexibility.
- [ ] **AI-Binder Ensembles:** Create standard exporters for conformational ensembles compatible with BindCraft and RFdiffusion.

## Phase 4: Scaling & Advanced Biology
*Focus: Supramolecular Scale and AI-Readiness.*

- [ ] **Large-Scale Solvers:** Implement iterative solvers for systems with >10k nodes (Ribosomes, Capsids).
- [ ] **Conformational Morphing:** Implement transition path prediction between Apo and Holo states.
- [ ] **Elastic Descriptors:** Build the featurization engine to extract "mechanical fingerprints" for ML models.
- [ ] **Quaternary Assembly:** Develop mechanical complementarity metrics for predicting protein-protein association.

## Future Horizons
- **Mechanopharmacology:** Force-responsive drug design.
- **Environment-Aware ENM:** Temperature and pH-dependent elasticity.
- **QM/ENM Hybrids:** Quantum-level precision for ligand pockets.
- **Haptic Dynamics:** Real-time interactive manipulation in MolSysViewer.
