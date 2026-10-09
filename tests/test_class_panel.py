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

"""The Point class box: a button (and Ctrl+number) per class, colour by
class, and adding a class, on the real panel.

Real panel on a real (offscreen) vispy canvas; skipped where no canvas can
be created.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication  # noqa: E402

LEAF, WOOD = 1, 2


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def panel():
    """Two trees, each wood below and leaf above, with a named class field."""
    try:
        from segfix.cloudview import CloudView

        view = CloudView()
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no vispy canvas available: {exc}")
    from segfix.model import PointCloud
    from segfix.widgets import SegFixController, SegFixWidget

    rng = np.random.default_rng(0)
    coords = np.vstack([rng.random((40, 3)) + [k * 0.5, 0, 0] for k in (0, 1)])
    labels = np.repeat([1, 2], 40)
    classes = np.where(coords[:, 2] < 0.5, WOOD, LEAF)
    cloud = PointCloud(coords=coords.astype(np.float32), labels=labels,
                       classes=classes, class_field="classification")
    empty = PointCloud(coords=np.empty((0, 3), np.float32),
                       labels=np.empty(0, np.int32))
    view.load_cloud(empty)
    seg = SegFixController(view, empty)
    seg.class_field = "classification"
    seg.class_names = {LEAF: "leaf", WOOD: "wood"}
    seg.class_range = (0, 255)
    p = SegFixWidget(seg)
    view.load_cloud(cloud)
    seg.set_cloud(cloud)
    p._set_current(1, fly=False)
    said = []
    view.on_status = said.append
    yield p, seg, cloud, said
    p.deleteLater()


def test_each_named_class_gets_a_button_in_value_order(panel):
    p, _seg, _cloud, _said = panel
    assert p._class_codes == [LEAF, WOOD]
    assert [b.text().strip() for b in p._class_btns] == ["leaf", "wood"]
    assert "Ctrl+1" in p._class_btns[0].toolTip()


def test_ctrl_number_gives_the_selection_that_class(panel):
    p, seg, cloud, _said = panel
    leaves = np.flatnonzero((cloud.labels == 1) & (cloud.classes == LEAF))[:5]
    seg.view.selected = set(leaves.tolist())
    p.set_nth_class(2)  # wood
    assert set(cloud.classes[leaves]) == {WOOD}
    # Nothing else moved, and the tree IDs are untouched.
    assert (cloud.labels[leaves] == 1).all()
    assert (cloud.classes == WOOD).sum() == (np.asarray(cloud.coords)[:, 2] < 0.5).sum() + 5


def test_with_nothing_selected_a_class_goes_to_the_whole_current_tree(panel):
    p, seg, cloud, _said = panel
    seg.view.selected = set()
    p.on_set_class(LEAF)
    assert (cloud.classes[cloud.labels == 1] == LEAF).all()
    assert (cloud.classes[cloud.labels == 2] == WOOD).any()  # tree 2 untouched


def test_a_class_edit_undoes(panel):
    p, _seg, cloud, _said = panel
    before = cloud.classes.copy()
    p.on_set_class(WOOD)
    p.on_undo()
    np.testing.assert_array_equal(cloud.classes, before)


def test_colour_by_class_paints_each_class_its_colour(panel):
    from segfix.viewer import class_colors

    p, seg, cloud, _said = panel
    p.class_color_cb.setChecked(True)
    colours = class_colors(seg.class_names)
    fc = np.asarray(seg.view.face_color)
    leaf = np.flatnonzero(cloud.classes == LEAF)[0]
    wood = np.flatnonzero(cloud.classes == WOOD)[0]
    np.testing.assert_allclose(fc[leaf, :3], colours[LEAF], atol=1e-6)
    np.testing.assert_allclose(fc[wood, :3], colours[WOOD], atol=1e-6)

    # An edit recolours the points it changed, still by class.
    seg.view.selected = {int(leaf)}
    p.on_set_class(WOOD)
    np.testing.assert_allclose(
        np.asarray(seg.view.face_color)[leaf, :3], colours[WOOD], atol=1e-6
    )


def test_a_ctrl_number_past_the_last_class_says_so(panel):
    p, _seg, cloud, said = panel
    before = cloud.classes.copy()
    p.set_nth_class(4)
    assert "Only 2 classes" in said[-1]
    np.testing.assert_array_equal(cloud.classes, before)


def test_a_new_class_is_numbered_above_the_rest_and_takes_the_selection(
        panel, monkeypatch):
    from qtpy.QtWidgets import QInputDialog

    p, seg, cloud, _said = panel
    remembered = []
    p.on_class_names_changed = remembered.append
    monkeypatch.setattr(
        QInputDialog, "getText", staticmethod(lambda *a, **k: ("understorey", True))
    )
    seg.view.selected = {0, 1, 2}
    p.on_new_class()
    assert seg.class_names[3] == "understorey"
    assert remembered[-1][3] == "understorey"
    assert cloud.classes[[0, 1, 2]].tolist() == [3, 3, 3]
    assert p._class_codes == [LEAF, WOOD, 3]


def test_no_new_class_past_what_the_field_holds(panel):
    p, seg, _cloud, _said = panel
    seg.class_range = (0, 2)
    assert p.new_class_code() is None
    seg.class_range = (0, 3)
    assert p.new_class_code() == 3


def test_without_a_class_field_the_box_says_how_to_get_one(panel):
    p, seg, _cloud, _said = panel
    seg.class_field = None
    p._rebuild_class_buttons()
    assert p._class_btns == []
    assert not p.class_color_cb.isEnabled()
    assert "Set up" in p.class_info.text()


# -- the buttons fit their names ----------------------------------------------
ASPRS = {2: "Ground", 3: "Low vegetation", 4: "Medium vegetation",
         5: "High vegetation", 7: "Low point (noise)"}


def _columns_used(p) -> int:
    cols = set()
    for i in range(p.class_grid.count()):
        _row, col, _rs, _cs = p.class_grid.getItemPosition(i)
        cols.add(col)
    return len(cols)


def test_short_names_sit_two_to_a_row(panel):
    p, _seg, _cloud, _said = panel
    assert _columns_used(p) == 2


def test_long_names_get_a_column_each_rather_than_being_clipped(panel):
    """ASPRS names are twice the width of "leaf"; two to a row they were
    cut mid-word ("Medium ve", "Low point (") with nothing to say so."""
    p, seg, _cloud, _said = panel
    seg.class_names = dict(ASPRS)
    p._rebuild_class_buttons()
    assert _columns_used(p) == 1
    inner = p.OVERLAY_W - 2 * p._CLASS_BOX_MARGIN
    for btn in p._class_btns:
        assert p._class_button_width(btn) <= inner, btn.text()


def test_one_column_shows_a_row_more_so_ctrl_4_is_not_below_the_fold(panel):
    p, seg, _cloud, _said = panel
    seg.class_names = {k: ASPRS[k] for k in (2, 4, 5, 7)}
    p._rebuild_class_buttons()
    row_h = p._neighbour_row_height()
    assert p.class_scroll.height() >= 4 * row_h + 3 * 4
