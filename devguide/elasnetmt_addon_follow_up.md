# ElasNetMT Add-on Follow-up Note

This note records the state of the MolSysViewer add-on work after the initial MVP was implemented and pushed.

## Delivered

The following pieces are already delivered from the **ElasNetMT** repository:

- the in-tree package `molsysviewer_elasnetmt`,
- add-on lifecycle, runtime cache, demo helpers, workbench helpers, and export helpers,
- real rendering of:
  - contact links,
  - ANM mode vectors,
  - anisotropy ellipsoids,
- integration tests for the add-on MVP against `pdb_id:1tcd`,
- developer documentation for plan, demo, and roadmap state.

## Host Dependency Already Landed

The minimum corresponding host work has also been integrated in **MolSysViewer**:

- `entry` for add-on workbench sections and export helpers is now resolved to Python callables when available,
- runtime summaries are refreshed after add-on context actions,
- export messages include the materialized add-on runtime summary.

This means the add-on is no longer only declarative. The host already consumes the Python-side helper layer.

## What Remains Open

The most relevant open items are:

1. Improve the MolSysViewer workbench UI so enriched `runtime_payload` data is shown more explicitly.
2. Add broader end-to-end export and replay checks across both repositories.
3. Decide whether `molsysviewer_elasnetmt` remains in-tree or becomes a separate distribution.
4. Design a richer parameter-editing surface once the current Python-driven flow is considered stable.

## Current Recommendation

Do not widen scope yet. The correct next step is to polish host presentation and replay reliability before attempting a richer frontend or extraction into a separate package.
