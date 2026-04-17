from __future__ import annotations

from molsysviewer import (
    AddonContextActionSpec,
    AddonExportHelperSpec,
    AddonShapeProviderSpec,
    AddonLifecycleSpec,
    AddonPanelSpec,
    AddonSpec,
    AddonWorkbenchSectionSpec,
    AddonWorkspaceSpec,
)

from .adapters import render_anisotropy_ellipsoids, render_contact_network, render_mode_vectors
from .runtime import ensure_runtime, record_event, set_overlay_visibility
from .export import build_figure_export_payload
from .workbench import get_modes_section, get_network_overlays_section, get_runtime_snapshot


addon = AddonSpec(
    name="elasnetmt",
    package="elasnetmt",
    description="Elastic network analysis workspace for MolSysViewer.",
    workspaces=(
        AddonWorkspaceSpec(
            id="elasnetmt",
            title="Elastic Networks",
            entry_panel="modes",
            description="Workspace for GNM/ANM models and overlays.",
            order=40,
        ),
    ),
    panels=(
        AddonPanelSpec(
            id="model",
            title="Model",
            entry="molsysviewer_elasnetmt.panels.model",
            description="Model parameters and node selection.",
            order=10,
            widget_class="molsysviewer_elasnetmt.panels.model.ElasNetMTModelPanel",
        ),
        AddonPanelSpec(
            id="modes",
            title="Modes",
            entry="molsysviewer_elasnetmt.panels.modes",
            description="Normal mode selection and activation.",
            order=20,
            widget_class="molsysviewer_elasnetmt.panels.modes.ElasNetMTModesPanel",
        ),
        AddonPanelSpec(
            id="figures",
            title="Figures",
            entry="molsysviewer_elasnetmt.panels.figures",
            description="Export helpers and reproducible figure presets.",
            order=30,
            widget_class="molsysviewer_elasnetmt.panels.figures.ElasNetMTFiguresPanel",
        ),
    ),
    context_actions=(
        AddonContextActionSpec(
            id="show-contact-network",
            title="Show Contact Network",
            entry="molsysviewer_elasnetmt.context.show_contact_network",
            target_kinds=("structure",),
            group="elasnetmt",
            order=10,
        ),
        AddonContextActionSpec(
            id="show-mode-vectors",
            title="Show Mode Vectors",
            entry="molsysviewer_elasnetmt.context.show_mode_vectors",
            target_kinds=("structure",),
            group="elasnetmt",
            order=20,
        ),
        AddonContextActionSpec(
            id="show-anisotropy-ellipsoids",
            title="Show Anisotropy Ellipsoids",
            entry="molsysviewer_elasnetmt.context.show_anisotropy_ellipsoids",
            target_kinds=("structure",),
            group="elasnetmt",
            order=30,
        ),
    ),
    workbench_sections=(
        AddonWorkbenchSectionSpec(
            id="modes",
            title="Modes",
            entry="molsysviewer_elasnetmt.workbench.modes",
            target_panel="workbench",
            order=10,
        ),
        AddonWorkbenchSectionSpec(
            id="network-overlays",
            title="Network Overlays",
            entry="molsysviewer_elasnetmt.workbench.network_overlays",
            target_panel="workbench",
            order=20,
        ),
    ),
    shape_providers=(
        AddonShapeProviderSpec(
            id="contact-links",
            title="Contact Links",
            entry="molsysviewer_elasnetmt.shapes.contact_links",
            kinds=("link", "network"),
            order=10,
        ),
        AddonShapeProviderSpec(
            id="anisotropy-ellipsoids",
            title="Anisotropy Ellipsoids",
            entry="molsysviewer_elasnetmt.shapes.anisotropy_ellipsoids",
            kinds=("ellipsoid", "anisotropy"),
            order=20,
        ),
    ),
    export_helpers=(
        AddonExportHelperSpec(
            id="enm-figure",
            title="ENM Figure Export",
            entry="molsysviewer_elasnetmt.export.figure",
            formats=("png", "html"),
            order=10,
        ),
    ),
    meta={
        "domain": "elastic_networks",
        "repo": "elasnetmt",
        "status": "skeleton",
    },
)


def on_enable(view):
    runtime = ensure_runtime(view)
    runtime.enabled = True
    record_event(view, "enable", workspace=runtime.workspace)


def on_disable(view):
    runtime = ensure_runtime(view)
    runtime.enabled = False
    record_event(view, "disable", workspace=runtime.workspace)


def on_context_action(view, action_id, payload):
    runtime = ensure_runtime(view)
    runtime.last_context_action = {"action_id": action_id, "payload": dict(payload)}

    if action_id == "show-contact-network":
        render_contact_network(view)
    elif action_id == "show-mode-vectors":
        render_mode_vectors(view, mode_index=runtime.active_mode_index)
    elif action_id == "show-anisotropy-ellipsoids":
        render_anisotropy_ellipsoids(view)

    record_event(view, "context_action", action_id=action_id)


lifecycle = AddonLifecycleSpec(
    on_enable=on_enable,
    on_disable=on_disable,
    on_context_action=on_context_action,
)


def get_addon():
    return addon
