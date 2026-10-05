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

"""What's shown follows an edit.

Regression coverage for a real bug: with H hiding unassigned and noise
points, points you then unassigned (or marked noise) stayed on screen --
visibility was only recomputed when a view toggle changed, never after an
edit. The same staleness left points visible after being moved into a
hidden tree.

Builds the real panel on a real (offscreen) vispy canvas; skipped where no
canvas can be created.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def panel():
    try:
        from segfix.cloudview import CloudView

        view = CloudView()
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no vispy canvas available: {exc}")
    from segfix.model import PointCloud
    from segfix.widgets import SegFixController, SegFixWidget

    rng = np.random.default_rng(0)
    coords = rng.random((60, 3)).astype(np.float32)
    labels = np.repeat([1, 2, 0], 20).astype(np.int32)  # tree 1, tree 2, ground
    empty = PointCloud(coords=np.empty((0, 3), np.float32),
                       labels=np.empty(0, np.int32))
    view.load_cloud(empty)
    seg = SegFixController(view, empty)
    p = SegFixWidget(seg)
    cloud = PointCloud(coords=coords, labels=labels)
    view.load_cloud(cloud)
    seg.set_cloud(cloud)
    p._set_current(1, fly=False)
    yield p, view
    p.deleteLater()


TREE1_HALF = np.arange(0, 10)


def test_points_unassigned_while_hidden_disappear_and_undo_brings_them_back(panel):
    p, view = panel
    p.show_unassigned.setChecked(False)          # H: hide unassigned + noise
    assert view.shown[TREE1_HALF].all()
    view.selected = set(TREE1_HALF.tolist())
    p.on_unassign()
    assert not view.shown[TREE1_HALF].any()
    p.on_undo()
    assert view.shown[TREE1_HALF].all()


def test_points_marked_noise_while_hidden_disappear(panel):
    p, view = panel
    p.show_unassigned.setChecked(False)
    view.selected = set(TREE1_HALF.tolist())
    p.on_noise()
    assert not view.shown[TREE1_HALF].any()


def test_points_moved_into_a_hidden_tree_disappear(panel):
    p, view = panel
    p._set_hidden(2, True)
    view.selected = set(TREE1_HALF.tolist())
    p.on_send_to_neighbour(2)
    assert not view.shown[TREE1_HALF].any()


# -- where the Point class box lives ------------------------------------------
def test_the_class_box_floats_over_the_canvas_by_default(panel):
    """Next to the points it labels — and a project with no class field
    never has to think about it at all."""
    p, view = panel
    assert not p.class_in_top_bar
    assert p._class_overlay.parent() is view.native


def test_asking_for_it_in_the_top_bar_docks_it(panel):
    p, _ = panel
    p._set_class_in_top_bar(True)
    assert p._class_overlay.parent() is p.top_bar
    # One box, moved — not a second copy to keep in step with the first.
    # (isHidden, not isVisible: the bar itself has no window here.)
    assert not p._class_overlay.isHidden()
    assert p._top_bar_row.indexOf(p._class_overlay) >= 0


def test_turning_it_off_floats_it_back_over_the_canvas(panel):
    p, view = panel
    p._set_class_in_top_bar(True)
    p._set_class_in_top_bar(False)
    assert p._class_overlay.parent() is view.native
    assert p._class_overlay.width() == p.OVERLAY_W


def test_the_docked_box_is_shorter_than_the_floating_one(panel):
    """The top bar grows to its tallest box, so the class list scrolls
    sooner there rather than pushing the canvas down."""
    p, _ = panel
    floating = p.class_scroll.height()
    p._set_class_in_top_bar(True)
    assert p.class_scroll.height() < floating


def test_the_class_buttons_still_work_from_the_bar(panel):
    """The same widgets moved, so the wiring has to come with them."""
    p, _ = panel
    p.c.class_field = "classification"
    p.c.class_names = {1: "leaf", 2: "wood"}
    p._rebuild_class_buttons()
    p._set_class_in_top_bar(True)
    assert [b.text().strip() for b in p._class_btns] == ["leaf", "wood"]
    assert p.class_color_cb.isEnabled()
