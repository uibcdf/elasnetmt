# Units and Conventions

Consistency in physical quantities is critical for reproducibility. ElastNetMT strictly uses **PyUnitWizard** for all inputs and outputs.

## Standard Units

Unless otherwise specified, ElastNetMT uses the following internal standards:
- **Distance:** Angstroms (Å)
- **Time:** Picoseconds (ps)
- **Mass:** AMU
- **Energy:** kcal/mol

## Physical Parameters

### Cutoff Distance
The default cutoff is typically `7 angstroms` for GNM and `12 angstroms` for ANM. Users can provide quantities in any unit (e.g., `0.7 nm`), but they must be standardized before calculation:

```python
from elastnetmt import pyunitwizard as puw
cutoff = puw.standardize(cutoff)
```

### Force Constant ($\gamma$)
The spring constant defaults to a value fitted to experimental B-factors if possible. It carries the dimension of `[mass]/[time]^2`.

## Selection Logic
Selection strings must follow the **MolSysMT** syntax (standard: `MolSysMT`). By default, models are built using `atom_name=="CA"` (alpha-carbons).
