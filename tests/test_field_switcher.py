# segfix — fix instance segmentation of tree point clouds.
# Copyright (C) 2026 Tim Devereux, The University of Queensland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Switching the class field from the Point class box, the way CloudCompare
switches its active scalar field: any numeric per-point field can be the
one the class buttons edit and Colour by class shows, each field's value
names are remembered with the project, and a field with too many values
for a button each is coloured along a ramp.

Real panel, catalog and scene on a real (offscreen) canvas; skipped where
no canvas can be created.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication, QMessageBox  # noqa: E402

from segfix import app, workspace  # noqa: E402
from segfix.classes_ui import MAX_LISTED_VALUES  # noqa: E402

LEAF, WOOD, GROUND = 5, 4, 2


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def _project(tmp_path):
    """A project folder holding two trees with a classification (two
    values), a user_data (three) and an intensity (hundreds)."""
    import laspy

    rng = np.random.default_rng(0)
    coords = np.vstack([rng.random((60, 3)) + [k * 4.0, 0, 0] for k in (0, 1)])
    n = len(coords)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = [0.001] * 3
    header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    las = laspy.LasData(header)
    las.x, las.y, las.z = coords.T
    las.treeID = np.repeat([1, 2], 60).astype(np.int32)
    las.classification = np.where(coords[:, 2] < 0.5, WOOD, LEAF).astype(np.uint8)
    las.user_data = (np.arange(n) % 3).astype(np.uint8)
    las.intensity = (np.arange(n) * 97 % 2000).astype(np.uint16)
    source = tmp_path / "plot.las"
    las.write(str(source))
    return str(workspace.create_workspace(source, tmp_path / "proj"))


@pytest.fixture
def rig(tmp_path):
    try:
        from segfix.cloudview import CloudView

        view = CloudView()
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no vispy canvas available: {exc}")
    from segfix.model import PointCloud
    from segfix.scene_ui import SceneController
    from segfix.treecatalog import open_catalog
    from segfix.widgets import SegFixController, SegFixWidget

    path = _project(tmp_path)
    catalog = open_catalog(path, class_field="classification")
    empty = PointCloud(coords=np.empty((0, 3), np.float32),
                       labels=np.empty(0, np.int32))
    view.load_cloud(empty)
    seg = SegFixController(view, empty)
    panel = SegFixWidget(seg)
    scene = SceneController(view, catalog, seg)
    panel.c.class_fields = catalog.class_fields()
    app._use_classes(panel, catalog, {})
    panel.on_class_field_changed = (
        lambda field: app._switch_class_field(None, panel, catalog, scene, field)
    )
    panel.on_class_names_changed = (
        lambda names: app._remember_classes(catalog, catalog.class_field, names)
    )
    panel._rebuild_class_buttons()
    scene.load_tree(1)
    said = []
    view.on_status = said.append
    yield panel, seg, catalog, scene, said
    panel.deleteLater()


# -- what the project remembers ----------------------------------------------
def test_names_are_kept_per_field_and_old_projects_count_as_one_field():
    old = {"class_field": "semantic", "class_names": {"1": "leaf", "2": "wood"}}
    assert app._names_by_field(old) == {"semantic": {1: "leaf", 2: "wood"}}
    assert app._names_for_field(old, "semantic") == {1: "leaf", 2: "wood"}
    assert app._names_for_field(old, "other") == {}
    assert app._names_for_field(old, None) == {}

    new = {
        "class_field": "user_data",
        "class_names": {"0": "a"},
        "class_names_by_field": {"semantic": {"1": "leaf"}, "user_data": {"0": "a"}},
    }
    assert app._names_by_field(new) == {"semantic": {1: "leaf"}, "user_data": {0: "a"}}


def test_remembering_one_field_keeps_the_others_names(tmp_path):
    from segfix.treecatalog import open_catalog

    catalog = open_catalog(_project(tmp_path))
    app._remember_classes(catalog, "classification", {LEAF: "leaf", WOOD: "wood"})
    app._remember_classes(catalog, "user_data", {0: "zero"})
    saved = workspace.settings(catalog.path)
    assert saved["class_field"] == "user_data"
    assert saved["class_names"] == {"0": "zero"}
    assert saved["class_names_by_field"] == {
        "classification": {"4": "wood", "5": "leaf"},
        "user_data": {"0": "zero"},
    }


# -- the switch itself -------------------------------------------------------
def test_the_combo_offers_every_numeric_field_but_the_tree_id(rig):
    panel, _seg, catalog, _scene, _said = rig
    items = [panel.class_field_combo.itemText(i)
             for i in range(panel.class_field_combo.count())]
    assert items[0] == panel.NO_CLASS_FIELD
    assert {"classification", "user_data", "intensity"} <= set(items)
    assert "treeID" not in items
    assert panel.class_field_combo.currentText() == "classification"


def test_switching_fields_reads_the_new_one_and_remembers_each_fields_names(rig):
    panel, seg, catalog, scene, _said = rig
    # A LAS classification starts from the standard names.
    assert seg.class_names == {WOOD: "Medium vegetation", LEAF: "High vegetation"}
    seg.class_names[LEAF] = "leaf"
    panel.on_class_names_changed(seg.class_names)

    panel.class_field_combo.setCurrentText("user_data")
    assert catalog.class_field == "user_data"
    assert seg.class_field == "user_data"
    assert seg.class_ramp is None
    assert panel._class_codes == [0, 1, 2]
    assert set(np.unique(seg.cloud.classes)) <= {0, 1, 2}  # the scene reloaded
    assert workspace.settings(catalog.path)["class_field"] == "user_data"

    panel.class_field_combo.setCurrentText("classification")
    assert catalog.class_field == "classification"
    assert seg.class_names == {WOOD: "Medium vegetation", LEAF: "leaf"}
    assert [b.text().strip() for b in panel._class_btns] == [
        "Medium vegetation", "leaf"
    ]


def test_an_edit_in_one_field_saves_to_that_field_alone(rig, monkeypatch):
    import laspy

    panel, seg, catalog, scene, _said = rig
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Save),
    )
    seg.view.selected = {0, 1, 2}
    panel.on_set_class(GROUND)
    panel.class_field_combo.setCurrentText("user_data")  # saves, then switches
    assert catalog.class_field == "user_data"
    seg.view.selected = {0, 1}
    panel.on_set_class(2)
    panel.on_save()

    after = laspy.read(catalog.path)
    assert (np.asarray(after.classification) == GROUND).sum() == 3
    assert np.asarray(after.user_data)[:3].tolist() == [2, 2, 2]
    assert (np.asarray(after.user_data) == 2).sum() == 120 // 3 + 2


def test_keeping_unsaved_edits_refuses_the_switch_and_the_combo_follows(
        rig, monkeypatch):
    panel, seg, catalog, _scene, _said = rig
    seg.view.selected = {0}
    panel.on_set_class(GROUND)
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Cancel),
    )
    panel.class_field_combo.setCurrentText("user_data")
    assert catalog.class_field == "classification"
    assert panel.class_field_combo.currentText() == "classification"
    assert seg.cloud.classes[0] == GROUND  # the edit is still there


def test_a_many_valued_field_gets_a_ramp_not_a_button_each(rig):
    from segfix.viewer import ScalarRamp

    panel, seg, catalog, _scene, _said = rig
    panel.class_field_combo.setCurrentText("intensity")
    assert catalog.class_field == "intensity"
    assert len(np.unique(catalog.classes)) > MAX_LISTED_VALUES
    assert seg.class_ramp is not None
    assert seg.class_ramp[0] < seg.class_ramp[1]
    assert panel._class_btns == []
    assert not panel.class_scroll.isVisible()
    assert "Value…" in panel.class_info.text()
    panel.class_color_cb.setChecked(True)
    assert isinstance(seg.class_colours(), ScalarRamp)
    # Edits still work, and undo along with the rest.
    seg.view.selected = {0, 1}
    panel.on_set_class(1234)
    assert seg.cloud.classes[[0, 1]].tolist() == [1234, 1234]
    panel.on_undo()
    assert seg.cloud.classes[0] != 1234


def test_no_class_field_drops_classes_and_turns_colour_by_class_off(rig):
    panel, seg, catalog, _scene, _said = rig
    panel.class_color_cb.setChecked(True)
    panel.class_field_combo.setCurrentText(panel.NO_CLASS_FIELD)
    assert catalog.class_field is None and catalog.classes is None
    assert seg.class_field is None and seg.cloud.classes is None
    assert not panel.class_color_cb.isChecked()
    assert workspace.settings(catalog.path)["class_field"] is None
