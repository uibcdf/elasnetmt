import importlib
import pathlib
import sys

import molsysmt as msm
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def test_molsysviewer_elasnetmt_module_exposes_valid_addon_contract():
    molsysviewer = pytest.importorskip("molsysviewer")
    module = importlib.import_module("molsysviewer_elasnetmt")

    assert isinstance(module.addon, molsysviewer.AddonSpec)
    assert module.addon.name == "elasnetmt"
    assert [item.id for item in module.addon.workspaces] == ["elasnetmt"]
    assert [item.id for item in module.addon.panels] == ["model", "modes", "figures"]
    assert [item.id for item in module.addon.context_actions] == [
        "show-contact-network",
        "show-mode-vectors",
        "show-anisotropy-ellipsoids",
    ]
    assert [item.id for item in module.addon.workbench_sections] == [
        "modes",
        "network-overlays",
    ]
    assert [item.id for item in module.addon.export_helpers] == ["enm-figure"]
    assert [item.id for item in module.addon.shape_providers] == [
        "contact-links",
        "anisotropy-ellipsoids",
    ]
    assert module.lifecycle.info() == {
        "has_on_enable": True,
        "has_on_disable": True,
        "has_on_context_action": True,
    }


def test_molsysviewer_elasnetmt_lifecycle_initializes_runtime_and_tracks_actions():
    molsysviewer = pytest.importorskip("molsysviewer")
    module = importlib.import_module("molsysviewer_elasnetmt")
    view = molsysviewer.MolSysView(debug_js=True)

    module.on_enable(view)
    runtime = view._elasnetmt_addon_runtime

    assert runtime.enabled is True
    assert runtime.workspace == "elasnetmt"
    assert runtime.visible_overlays == []
    assert runtime.event_log[-1]["event"] == "enable"

    view.load("pdb_id:1tcd")
    module.on_context_action(
        view,
        "show-contact-network",
        {"addon": "elasnetmt", "addon_action_id": "show-contact-network"},
    )
    assert "elasnetmt:contacts" in runtime.visible_overlays
    assert runtime.last_context_action["action_id"] == "show-contact-network"
    assert any(message.get("op") == "add_network_links" for message in view._message_history)

    module.on_context_action(
        view,
        "show-mode-vectors",
        {"addon": "elasnetmt", "addon_action_id": "show-mode-vectors"},
    )
    assert "elasnetmt:mode:0" in runtime.visible_overlays
    assert any(message.get("op") == "add_displacement_vectors" for message in view._message_history)
    assert runtime.event_log[-1]["event"] == "context_action"

    module.on_disable(view)
    assert runtime.enabled is False
    assert runtime.event_log[-1]["event"] == "disable"


def test_molsysviewer_elasnetmt_contact_adapter_builds_atom_pairs_and_renders_links():
    pytest.importorskip("molsysviewer")
    adapter_module = importlib.import_module("molsysviewer_elasnetmt.adapters.contacts")
    molsysviewer_module = importlib.import_module("molsysviewer")

    molecular_system = msm.convert("pdb_id:1tcd", to_form="molsysmt.MolSys")
    view = molsysviewer_module.MolSysView(debug_js=True)
    view.load(molecular_system)

    layer, model = adapter_module.render_contact_network(view, molecular_system=molecular_system)
    atom_pairs = adapter_module.build_contact_atom_pairs(model)

    assert layer.tag == "elasnetmt:contacts"
    assert model.n_nodes > 0
    assert len(atom_pairs) > 0
    assert all(len(pair) == 2 for pair in atom_pairs)
    assert view._message_history[-1]["op"] == "add_network_links"
    assert view._message_history[-1]["options"]["tag"] == "elasnetmt:contacts"
    assert len(view._message_history[-1]["options"]["atom_pairs"]) == len(atom_pairs)


def test_molsysviewer_elasnetmt_mode_adapter_builds_vectors_and_renders_displacements():
    pytest.importorskip("molsysviewer")
    adapter_module = importlib.import_module("molsysviewer_elasnetmt.adapters.modes")
    molsysviewer_module = importlib.import_module("molsysviewer")

    molecular_system = msm.convert("pdb_id:1tcd", to_form="molsysmt.MolSys")
    view = molsysviewer_module.MolSysView(debug_js=True)
    view.load(molecular_system)

    layer, model, vectors = adapter_module.render_mode_vectors(view, molecular_system=molecular_system, mode_index=0)

    assert layer.tag == "elasnetmt:mode:0"
    assert model.n_nodes > 0
    assert vectors.shape == (model.n_nodes, 3)
    assert view._message_history[-1]["op"] == "add_displacement_vectors"
    assert view._message_history[-1]["options"]["tag"] == "elasnetmt:mode:0"
    assert len(view._message_history[-1]["options"]["atom_indices"]) == model.n_nodes
    assert len(view._message_history[-1]["options"]["vectors"]) == model.n_nodes


def test_molsysviewer_elasnetmt_mode_adapter_respects_active_mode_and_reuses_cached_model():
    pytest.importorskip("molsysviewer")
    module = importlib.import_module("molsysviewer_elasnetmt")
    adapter_module = importlib.import_module("molsysviewer_elasnetmt.adapters.modes")
    molsysviewer_module = importlib.import_module("molsysviewer")

    molecular_system = msm.convert("pdb_id:1tcd", to_form="molsysmt.MolSys")
    view = molsysviewer_module.MolSysView(debug_js=True)
    view.load(molecular_system)
    module.on_enable(view)

    runtime = view._elasnetmt_addon_runtime
    runtime.active_mode_index = 1

    layer1, model1, vectors1 = adapter_module.render_mode_vectors(view, molecular_system=molecular_system)
    layer2, model2, vectors2 = adapter_module.render_mode_vectors(view, molecular_system=molecular_system, mode_index=2)

    assert layer1.tag == "elasnetmt:mode:1"
    assert layer2.tag == "elasnetmt:mode:2"
    assert model1 is model2
    assert runtime.active_mode_index == 2
    assert vectors1.shape == vectors2.shape == (model1.n_nodes, 3)
    assert np.any(np.not_equal(vectors1, vectors2))


def test_molsysviewer_elasnetmt_anisotropy_adapter_builds_ellipsoids():
    pytest.importorskip("molsysviewer")
    adapter_module = importlib.import_module("molsysviewer_elasnetmt.adapters.anisotropy")
    molsysviewer_module = importlib.import_module("molsysviewer")

    molecular_system = msm.convert("pdb_id:1tcd", to_form="molsysmt.MolSys")
    view = molsysviewer_module.MolSysView(debug_js=True)
    view.load(molecular_system)

    layer, model, eigenvalues, eigenvectors = adapter_module.render_anisotropy_ellipsoids(
        view,
        molecular_system=molecular_system,
        mode_count=10,
    )

    assert layer.tag == "elasnetmt:anisotropy"
    assert model.n_nodes > 0
    assert len(eigenvalues) == model.n_nodes
    assert len(eigenvectors) == model.n_nodes
    assert len(eigenvalues[0]) == 3
    assert len(eigenvectors[0]) == 3
    assert len(eigenvectors[0][0]) == 3
    assert view._message_history[-1]["op"] == "add_anisotropy_ellipsoids"
    assert view._message_history[-1]["options"]["tag"] == "elasnetmt:anisotropy"
    assert len(view._message_history[-1]["options"]["eigenvalues"]) == model.n_nodes


def test_molsysviewer_elasnetmt_workbench_and_export_helpers_report_reproducible_state():
    pytest.importorskip("molsysviewer")
    module = importlib.import_module("molsysviewer_elasnetmt")
    export_module = importlib.import_module("molsysviewer_elasnetmt.export")
    workbench_module = importlib.import_module("molsysviewer_elasnetmt.workbench")
    molsysviewer_module = importlib.import_module("molsysviewer")

    molecular_system = msm.convert("pdb_id:1tcd", to_form="molsysmt.MolSys")
    view = molsysviewer_module.MolSysView(debug_js=True)
    view.load(molecular_system)
    module.on_enable(view)

    module.on_context_action(view, "show-contact-network", {"addon": "elasnetmt", "addon_action_id": "show-contact-network"})
    module.on_context_action(view, "show-mode-vectors", {"addon": "elasnetmt", "addon_action_id": "show-mode-vectors"})
    module.on_context_action(view, "show-anisotropy-ellipsoids", {"addon": "elasnetmt", "addon_action_id": "show-anisotropy-ellipsoids"})

    modes_section = workbench_module.get_modes_section(view)
    overlays_section = workbench_module.get_network_overlays_section(view)
    export_payload = export_module.build_figure_export_payload(view)

    assert modes_section["title"] == "Modes"
    assert modes_section["item_title"] == "Mode 0"
    assert "cutoff=12 angstroms" in modes_section["item_subtitle"]
    assert overlays_section["title"] == "Network Overlays"
    assert "elasnetmt:contacts" in overlays_section["item_title"]
    assert export_payload["title"] == "ElasNetMT Figure Export"
    assert export_payload["figure_recipe"]["active_mode_index"] == 0
    assert "elasnetmt:contacts" in export_payload["figure_recipe"]["visible_overlays"]
    assert export_payload["figure_recipe"]["overlay_parameters"]["elasnetmt:anisotropy"]["kind"] == "anisotropy-ellipsoids"


def test_molsysviewer_elasnetmt_demo_bundle_builds_complete_mvp_state():
    pytest.importorskip("molsysviewer")
    demo_module = importlib.import_module("molsysviewer_elasnetmt.demo")

    bundle = demo_module.build_demo_bundle(
        "pdb_id:1tcd",
        mode_index=1,
        show_contact_network=True,
        show_mode_vectors=True,
        show_anisotropy_ellipsoids=True,
        debug_js=True,
    )

    view = bundle["view"]
    assert "elasnetmt:contacts" in bundle["network_overlays_section"]["snapshot"]["visible_overlays"]
    assert "elasnetmt:mode:1" in bundle["network_overlays_section"]["snapshot"]["visible_overlays"]
    assert "elasnetmt:anisotropy" in bundle["network_overlays_section"]["snapshot"]["visible_overlays"]
    assert bundle["modes_section"]["item_title"] == "Mode 1"
    assert bundle["export_payload"]["figure_recipe"]["active_mode_index"] == 1
    assert any(message.get("op") == "add_network_links" for message in view._message_history)
    assert any(message.get("op") == "add_displacement_vectors" for message in view._message_history)
    assert any(message.get("op") == "add_anisotropy_ellipsoids" for message in view._message_history)
