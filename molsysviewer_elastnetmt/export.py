from __future__ import annotations

from typing import Any

from .workbench import get_runtime_snapshot


def build_figure_export_payload(view: Any) -> dict[str, Any]:
    snapshot = get_runtime_snapshot(view)
    return {
        "title": "ElastNetMT Figure Export",
        "format_options": ["png", "html"],
        "runtime": snapshot,
        "figure_recipe": {
            "workspace": snapshot["workspace"],
            "model_kind": snapshot["model_kind"],
            "cutoff": snapshot["cutoff"],
            "selection": snapshot["selection"],
            "active_mode_index": snapshot["active_mode_index"],
            "visible_overlays": list(snapshot["visible_overlays"]),
            "overlay_parameters": dict(snapshot["overlay_parameters"]),
        },
    }
