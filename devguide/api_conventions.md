# API Conventions

To ensure consistency within the **MolSysSuite**, ElastNetMT follows a strict naming and behavioral convention for all public methods.

## Method Naming Prefixes

| Prefix | Intent | Example | Returns |
| :--- | :--- | :--- | :--- |
| `get_` | Retrieves calculated data or attributes. No side effects. | `get_eigenvalues()` | Arrays, Quantities, or Scalars. |
| `set_` | Modifies an internal parameter and resets dependent states. | `set_cutoff('10 A')` | `None`. |
| `show_` | Generates static 2D plots (usually Matplotlib). | `show_b_factors()` | `None` (shows plot) or `Figure`. |
| `view_` | Generates interactive 3D visualizations (MolSysViewer). | `view_mode(index=1)` | `MolSysView` (widget). |
| `calculate_`| Performs heavy computation and stores results internally. | `calculate_contacts()` | `None`. |
| `write_` | Exports data to external file formats. | `write_gnm_vectors()` | `None`. |

## Core Object Philosophy

1. **Lazy Evaluation:** Heavy computations (like matrix diagonalization) should **not** occur in the `__init__`. Instead, they should be triggered only when a `get_` or `view_` method requires the data.
2. **In-place vs. Copy:** Methods that modify the model (like changing the selection) should modify the object **in-place** to save memory, unless a `copy=True` argument is explicitly provided.
3. **Internal State:** Attributes meant for internal use only must be prefixed with an underscore (e.g., `self._kirchhoff_matrix`).

## Argument Handling

- All public methods must be decorated with `@arg_digest()` to ensure inputs are normalized.
- Selection strings must always follow the `MolSysMT` syntax by default.
- Cutoff and other physical distances must always be handled as `pyunitwizard` quantities.
