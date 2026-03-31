# Deprecation Policy

As ElasNetMT moves toward its first stable release (1.0.0), breaking changes are expected. To minimize friction for users, we follow the **MolSysSuite Deprecation Standard**.

## Protocol for Breaking Changes

1.  **Tagging:** Any function or method to be removed must be decorated with `elasnetmt_deprecation_warning`.
2.  **Grace Period:** Deprecated code must remain in the library for at least **two minor versions** before removal.
    - Example: If a method is deprecated in `0.5.0`, it must exist in `0.6.0` and can only be removed in `0.7.0`.
3.  **Documentation:** The deprecation message must include the **exact replacement method** and the version of removal.

```python
from elasnetmt._private.warnings import warn_deprecation

def old_method():
    warn_deprecation("old_method() is deprecated; use new_method() instead. Removal in v0.7.0.")
    return new_method()
```

## Policy During Pre-1.0.0 Phase
In the current stabilization phase, major internal refactorings (like the inheritance change) are considered **essential migrations**. We will attempt to keep backward compatibility but prioritize architectural health.
