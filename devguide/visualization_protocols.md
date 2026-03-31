# Visualization Protocols

ElasNetMT works in tandem with **MolSysViewer** to provide interactive visual representations of elastic networks.

## 1. Contact Map (Adjacency Matrix)
Rendered as a binary matrix (`matplotlib`). 
- **Style:** Black pixels for contacts, white for no-contacts.

## 2. Springs Protocol (Network View)
In the 3D viewer, the network is represented by cylinders between nodes.
- **Radius:** Default `0.2 angstroms`.
- **Color:** Hex `0x808080` (Gray).
- **Tag:** All springs should be tagged with `"network"` for easy scene management.

## 3. Normal Modes Protocol
Normal modes can be visualized as:
- **Arrows:** Vectors originating from nodes, pointing in the direction of the mode.
- **Vibrational Animation:** A generated trajectory along the mode (interpolated via `lindelint`).
- **Color Mapping:** Residues colored by their fluctuation amplitude (B-factor style).

## Reproducibility
Every view should be reproducible by saving the viewer state, capturing the model parameters used (cutoff, selection, and mode index).
