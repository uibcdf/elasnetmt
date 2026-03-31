# Testing Strategy

ElasNetMT follows a Tiered testing protocol to ensure code integrity and physical correctness.

## Tier 1: Smoke Tests (Unit)
- **Purpose:** Ensure the library imports and basic objects are created without crashing.
- **Scope:** 
  - GNM/ANM initialization with small peptides.
  - Basic mathematical operations (matrix building).

## Tier 2: Physical Validation (Functional)
- **Purpose:** Verify that the results are physically sound.
- **Scope:**
  - Correlation of modeled B-factors vs experimental B-factors for reference proteins (e.g., T4 Lysozyme).
  - Verification that the first 6 eigenvalues in ANM are zero (within numerical tolerance).

## Tier 3: Integration Tests
- **Purpose:** Test the interoperability with other MolSysSuite tools.
- **Scope:**
  - Loading systems via `molsysmt`.
  - Visualizing networks in `molsysviewer`.

## Regression Tests
- **Purpose:** Ensure that bug fixes stay fixed.
- **Scope:** Any edge case found during development (e.g., systems with multiple chains, isolated residues).

---

To run tests:
```bash
pytest tests/
```
To check coverage:
```bash
pytest --cov=elasnetmt
```
