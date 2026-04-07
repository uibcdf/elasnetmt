from __future__ import annotations

from typing import Any

import molsysviewer as msv

from . import on_context_action, on_enable
from .export import build_figure_export_payload
from .workbench import get_modes_section, get_network_overlays_section


def build_demo_view(
    molecular_system: Any = "pdb_id:1tcd",
    *,
    mode_index: int = 0,
    show_contact_network: bool = True,
    show_mode_vectors: bool = True,
    show_anisotropy_ellipsoids: bool = False,
    debug_js: bool = False,
):
    view = msv.MolSysView(debug_js=debug_js)
    view.load(molecular_system)
    on_enable(view)
    view._elasnetmt_addon_runtime.active_mode_index = int(mode_index)

    if show_contact_network:
        on_context_action(
            view,
            "show-contact-network",
            {"addon": "elasnetmt", "addon_action_id": "show-contact-network"},
        )
    if show_mode_vectors:
        on_context_action(
            view,
            "show-mode-vectors",
            {"addon": "elasnetmt", "addon_action_id": "show-mode-vectors"},
        )
    if show_anisotropy_ellipsoids:
        on_context_action(
            view,
            "show-anisotropy-ellipsoids",
            {"addon": "elasnetmt", "addon_action_id": "show-anisotropy-ellipsoids"},
        )

    return view


def build_demo_bundle(
    molecular_system: Any = "pdb_id:1tcd",
    *,
    mode_index: int = 0,
    show_contact_network: bool = True,
    show_mode_vectors: bool = True,
    show_anisotropy_ellipsoids: bool = False,
    debug_js: bool = False,
) -> dict[str, Any]:
    view = build_demo_view(
        molecular_system,
        mode_index=mode_index,
        show_contact_network=show_contact_network,
        show_mode_vectors=show_mode_vectors,
        show_anisotropy_ellipsoids=show_anisotropy_ellipsoids,
        debug_js=debug_js,
    )
    return {
        "view": view,
        "modes_section": get_modes_section(view),
        "network_overlays_section": get_network_overlays_section(view),
        "export_payload": build_figure_export_payload(view),
    }
