# Environmental and External Factors

Beyond intrinsic dynamics, ElastNetMT aims to model how proteins respond to their physical and chemical surroundings.

## 1. External Force Response (Mechanostability)

Proteins in vivo are often subject to mechanical stress. We can predict the response to an external force vector $\mathbf{F}$ using:
$$\Delta \mathbf{R} = \mathbf{H}^{-1} \mathbf{F}$$

- **Applications:** Simulating single-molecule pulling experiments (AFM/Optical Tweezers) and predicting mechanostability in vascular or muscular proteins.

## 2. Membrane-Embedded Dynamics

The lipid bilayer acts as a flexible but confining environment.
- **Mechanism:** Add implicit elastic constraints (virtual springs) to residues identified as trans-membrane, anchoring them to a fluid-elastic plane.
- **Applications:** Studying the gating mechanisms of ion channels and the activation of GPCRs within a realistic membrane tension context.

## 3. Geometric Constraints and Tethering

Modeling proteins anchored to surfaces or scaffolds.
- **Mechanism:** Modify the Hessian by removing degrees of freedom for fixed nodes or adding high-stiffness "anchor" springs.
- **Applications:** Bio-sensor design, enzyme immobilization on industrial supports, and protein-origami stability.

## 4. Electro-Elastic Coupling

Since residues carry partial charges ($q_i$), external electric fields ($\mathbf{E}$) induce forces:
$$\mathbf{F}_i = q_i \mathbf{E}$$

- **Applications:** Understanding voltage-gated channel activation and the impact of environmental electromagnetic fields on enzymatic efficiency.

---

## Technical Integration Roadmap

- [ ] **`model.apply_force(force_vector)`**: Calculate linear response displacement.
- [ ] **`model.add_membrane_constraint(region)`**: Add implicit membrane elasticity.
- [ ] **`model.set_fixed_nodes(selection)`**: Support for tethered boundary conditions.
- [ ] **`model.apply_electric_field(field_vector)`**: Couple charge distribution with elastic response.
