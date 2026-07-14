# ElastNetMT Add-on Plan For MolSysViewer

This document defines the development plan for the **ElastNetMT** add-on that will expose elastic-network analysis inside **MolSysViewer**.

The add-on must be developed from the **ElastNetMT** repository. `molsysviewer` remains the host application and contract provider, but the product logic, runtime state, scientific adapters, and tests for the add-on belong here.

## Implementation Status

Current state as of April 7, 2026:

- the in-tree package `molsysviewer_elastnetmt` exists and is importable,
- the add-on exposes `AddonSpec`, lifecycle hooks, runtime state, demo helpers, workbench helpers, and export helpers,
- contact links, ANM mode vectors, and anisotropy ellipsoids are rendered from real ElastNetMT models,
- reproducible state is captured in runtime snapshots and export payloads,
- MolSysViewer already consumes the Python-side workbench and export helpers through `entry` resolution in the host.

The plan below is therefore no longer purely prospective. Phases 1 to 4 are implemented in a first usable form, and the remaining work is primarily refinement, UI integration, and extraction readiness.

## Product Goal

Provide a first-class interactive bridge between ElastNetMT models and MolSysViewer scenes so that a user can:

- build a GNM or ANM model from the current structure in the viewer,
- inspect the contact network as a 3D overlay,
- display normal mode vectors and anisotropy ellipsoids,
- export reproducible figures and viewer states tied to explicit model parameters.

## Current Constraints

The current MolSysViewer add-on host already supports:

- add-on discovery by Python module,
- workspaces, panels, workbench sections, and context actions,
- Python lifecycle hooks,
- overlay primitives for links, displacement vectors, and anisotropy ellipsoids.

The first ElastNetMT add-on release should therefore stay Python-driven. It should not depend on a rich custom frontend panel or on host-side extensions that do not yet exist.

## Repository Direction

The implementation should start in this repository as a local package named `molsysviewer_elastnetmt`.

This keeps three things aligned:

- the scientific logic stays close to `GaussianNetworkModel` and `AnisotropicNetworkModel`,
- regression tests can live next to ElastNetMT physics and API tests,
- the add-on can later be extracted into its own distribution without redesigning the runtime contract.

## MVP Scope

The minimum viable add-on should provide:

- one workspace: `elastnetmt` with title `Elastic Networks`,
- three panels: `model`, `modes`, and `figures`,
- two workbench sections: `modes` and `network-overlays`,
- two context actions: `show-contact-network` and `show-mode-vectors`,
- one export helper: `enm-figure`.

The MVP must support two scientific flows:

1. Build a GNM model from the structure currently loaded in MolSysViewer and render the contact network.
2. Build an ANM model and render one selected normal mode as displacement vectors.

Anisotropy ellipsoids should be included in the architecture from the start, even if they remain behind a second implementation step.

## Development Phases

## Phase 1: Add-on Skeleton In ElastNetMT

*Status: IMPLEMENTED*

Create a local package `molsysviewer_elastnetmt` with:

- `AddonSpec`,
- `AddonLifecycleSpec`,
- stable workspace, panel, and action identifiers,
- a minimal view runtime store,
- import-safe discovery entry points.

Deliverable:

- MolSysViewer can import and register `molsysviewer_elastnetmt` directly from this repository.

## Phase 2: Runtime And Scientific Adapters

*Status: IMPLEMENTED*

Add a runtime layer that:

- builds GNM and ANM models lazily from the current view structure,
- stores model parameters such as cutoff, node selection, and mode index,
- translates ElastNetMT outputs into MolSysViewer overlay payloads.

Initial adapters:

- contacts to links,
- modes to displacement vectors,
- anisotropy tensors to ellipsoids.

Deliverable:

- the add-on can compute and cache viewer-ready overlays without duplicating ENM logic inside lifecycle hooks.

## Phase 3: Viewer Actions And Reproducibility

*Status: IMPLEMENTED*

Implement context actions and panel/workbench summaries that allow the user to:

- trigger model construction,
- toggle overlays,
- switch the active mode,
- inspect the exact parameters used to create the current overlays.

All visible states must be reproducible from explicit parameters:

- model kind,
- cutoff,
- node selection,
- mode index,
- overlay tags.

Deliverable:

- the workbench becomes the control surface for the add-on even without a custom frontend application.

## Phase 4: Export Helpers

*Status: IMPLEMENTED*

Implement export helpers for standard ENM visual outputs:

- structure plus contact network,
- structure plus mode vectors,
- structure plus anisotropy ellipsoids.

Deliverable:

- users can create repeatable figures or saved scenes with the scientific parameters embedded in the exported state.

## Phase 5: Hardening And Extraction Readiness

*Status: IN PROGRESS*

Add regression coverage and package boundaries that make later extraction straightforward.

Required tests:

- discovery and registration tests,
- lifecycle tests,
- adapter tests for links, vectors, and ellipsoids,
- integration tests against a real structure such as `pdb_id:1tcd`.

Deliverable:

- the add-on can be kept in-tree or split into `molsysviewer-elastnetmt` with minimal churn.

## Remaining Technical Work

The main remaining tasks are now:

1. Improve host-side presentation in MolSysViewer so the enriched workbench payloads are rendered more explicitly in the panel UI.
2. Add broader regression coverage across both repositories for host consumption of add-on `entry` callables.
3. Decide whether `molsysviewer_elastnetmt` remains in-tree or is extracted into its own distribution.
4. Define a richer control surface for parameter editing from the viewer once the current Python-driven loop is considered stable.

## Non-Goals For The First Iteration

The first iteration should not attempt:

- a fully custom TypeScript frontend,
- real-time bidirectional manipulation from viewer dragging,
- editing the ENM graph manually inside the viewer,
- support for every ElastNetMT analysis feature before the overlay pipeline is stable.

## Immediate Next Steps

The next concrete steps should now be:

1. Polish the MolSysViewer workbench UI to display `runtime_payload` data from add-on workbench sections and export helpers.
2. Add end-to-end checks that the ElastNetMT add-on state survives HTML export and replay cleanly.
3. Prepare a split-ready package boundary if the add-on is going to become `molsysviewer-elastnetmt`.
4. Keep the current Python-side demo flow as the canonical smoke path while frontend integration evolves.
