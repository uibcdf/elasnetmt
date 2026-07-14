# ElastNetMT Add-on Demo

This note provides the smallest reproducible flow to exercise the in-tree `molsysviewer_elastnetmt` add-on from this repository.

## Goal

Create a `MolSysView`, load a structure, and render the three MVP overlay families driven by ElastNetMT:

- contact network,
- ANM mode vectors,
- anisotropy ellipsoids.

## Minimal Demo

```python
from molsysviewer_elastnetmt.demo import build_demo_bundle

bundle = build_demo_bundle(
    "pdb_id:1tcd",
    mode_index=0,
    show_contact_network=True,
    show_mode_vectors=True,
    show_anisotropy_ellipsoids=True,
)

view = bundle["view"]
modes_section = bundle["modes_section"]
network_overlays_section = bundle["network_overlays_section"]
export_payload = bundle["export_payload"]

view
```

## Expected Result

- the view contains `elastnetmt:contacts`,
- the view contains `elastnetmt:mode:0`,
- the view contains `elastnetmt:anisotropy`,
- `modes_section` reports the active mode and model parameters,
- `export_payload` captures the reproducible state of the overlays.

## Why This Exists

This helper gives us a stable Python-side demonstration flow for the add-on. MolSysViewer already consumes the add-on workbench and export helpers through Python `entry` resolution, but the demo remains the most direct way to exercise the full MVP path from ElastNetMT while host-side UI presentation continues to mature.
