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

"""The All Trees table finds a tree, shows how far the plot has got, and
marks a tree done without loading it.

A plot is a few hundred trees. The table used to be a scroll with one
gesture, the double-click; this is the filter box, the "To do only"
switch, the progress bar and the right-click menu that every file manager
and labelling tool has taught people to expect.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy")
from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from segfix import scene_ui  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def write_ply(path: Path, labels: np.ndarray) -> None:
    rng = np.random.default_rng(1)
    n = len(labels)
    rows = np.empty(n, dtype=np.dtype([("x", "<f4"), ("y", "<f4"),
                                       ("z", "<f4"), ("treeID", "<i4")]))
    coords = (rng.random((n, 3)) * 6).astype(np.float32)
    rows["x"], rows["y"], rows["z"] = coords.T
    rows["treeID"] = labels
    with open(path, "wb") as fh:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property int treeID\nend_header\n"
        )
        fh.write(header.encode())
        fh.write(rows.tobytes())


@pytest.fixture
def table(tmp_path):
    try:
        from segfix.cloudview import CloudView

        view = CloudView()
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no vispy canvas available: {exc}")
    from segfix.model import PointCloud
    from segfix.treecatalog import open_catalog
    from segfix.widgets import SegFixController, SegFixWidget

    path = tmp_path / "plot.ply"
    write_ply(path, np.repeat([1, 2, 12, 120, 0], 40).astype(np.int32))
    (tmp_path / "plot.ply.segfix.json").write_text(json.dumps({"done": [2]}))
    catalog = open_catalog(str(path))
    empty = PointCloud(coords=np.empty((0, 3), np.float32),
                       labels=np.empty(0, np.int32))
    view.load_cloud(empty)
    seg = SegFixController(view, empty)
    panel = SegFixWidget(seg)
    scene = scene_ui.SceneController(view, catalog, seg)
    widget = scene_ui.SceneWidget(scene)
    panel.on_done_changed = widget.refresh
    widget.on_mark_done = panel.mark_done
    yield widget, panel, path
    widget.deleteLater()
    panel.deleteLater()


def test_it_lists_every_tree_with_its_colour(table):
    widget, _panel, _path = table
    assert widget.visible_labels() == [1, 2, 12, 120]
    for row in range(widget.table.rowCount()):
        assert widget.table.item(row, 1).data(Qt.DecorationRole) is not None


def test_the_filter_matches_the_start_of_the_id(table):
    widget, _panel, _path = table
    widget.filter_edit.setText("12")
    assert widget.visible_labels() == [12, 120]
    assert widget.trees_box.title() == "All Trees (2 of 4 shown)"
    widget.filter_edit.clear()
    assert widget.visible_labels() == [1, 2, 12, 120]
    assert widget.trees_box.title() == "All Trees"


def test_to_do_only_hides_the_finished_trees(table):
    widget, _panel, _path = table
    widget.pending_only.setChecked(True)
    assert widget.visible_labels() == [1, 12, 120]


def test_enter_in_the_filter_lands_on_the_first_match(table):
    widget, _panel, _path = table
    widget.filter_edit.setText("12")
    widget.filter_edit.returnPressed.emit()
    assert widget._selected_label() == 12


def test_the_progress_bar_is_the_done_count(table):
    widget, _panel, _path = table
    assert (widget.progress.value(), widget.progress.maximum()) == (1, 4)


def test_a_tree_can_be_marked_done_without_loading_it(table):
    widget, _panel, path = table
    widget.set_done(120, True)
    assert widget._read_done() == {2, 120}
    assert widget.progress.value() == 2
    widget.set_done(2, False)
    assert widget._read_done() == {120}
    sidecar = json.loads(Path(f"{path}.segfix.json").read_text())
    assert sidecar["done"] == [120]


def test_marking_done_from_here_keeps_the_rest_of_the_sidecar(tmp_path):
    """The sidecar also holds the inventory links; a Done click must not
    wipe them."""
    path = tmp_path / "plot.ply"
    sidecar = Path(f"{path}.segfix.json")
    sidecar.write_text(json.dumps({"done": [3], "inventory": {"3": "S7"}}))
    assert scene_ui.update_done(str(path), 5, True) == str(sidecar)
    assert json.loads(sidecar.read_text()) == {
        "done": [3, 5], "inventory": {"3": "S7"},
    }


def test_the_queue_says_what_it_is_for_until_something_is_loaded(table):
    widget, panel, _path = table
    panel.tree_table.viewport().resize(300, 200)
    panel._place_tree_hint()
    assert panel.tree_hint.isVisibleTo(panel)
    widget.table.selectRow(0)
    widget.on_load_tree()
    assert panel.tree_table.rowCount() > 0
    assert not panel.tree_hint.isVisibleTo(panel)
