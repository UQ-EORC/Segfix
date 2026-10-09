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

"""Ctrl+click a point to make its tree the current tree.

The table finds a tree by number; this finds it by pointing, the way 3D
Forest's tree picking and CloudCompare's point picking do. A click only:
a Ctrl-drag still orbits, an armed tool keeps its clicks, and the ground
is nobody's tree.
"""

from __future__ import annotations

import os
import types

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
    labels = np.repeat([1, 2, 0], 20).astype(np.int32)
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


def _release(pos, press_pos, modifiers=("Control",), button=1):
    return types.SimpleNamespace(
        pos=pos, button=button, modifiers=modifiers,
        press_event=types.SimpleNamespace(pos=press_pos),
    )


def test_the_view_hands_the_clicked_point_to_the_panel(panel, monkeypatch):
    p, view = panel
    picked = []
    view.on_pick_tree = picked.append
    monkeypatch.setattr(view, "pick_point", lambda xy, radius=9.0: 25)
    view._on_release(_release((10.0, 10.0), (11.0, 12.0)))
    assert picked == [25]


def test_a_drag_is_not_a_click(panel, monkeypatch):
    p, view = panel
    picked = []
    view.on_pick_tree = picked.append
    monkeypatch.setattr(view, "pick_point", lambda xy, radius=9.0: 25)
    view._on_release(_release((10.0, 10.0), (40.0, 10.0)))
    assert picked == []


def test_without_ctrl_a_click_is_the_cameras(panel, monkeypatch):
    p, view = panel
    picked = []
    view.on_pick_tree = picked.append
    monkeypatch.setattr(view, "pick_point", lambda xy, radius=9.0: 25)
    view._on_release(_release((10.0, 10.0), (10.0, 10.0), modifiers=()))
    assert picked == []


def test_an_armed_tool_keeps_its_clicks(panel, monkeypatch):
    p, view = panel
    picked = []
    view.on_pick_tree = picked.append
    monkeypatch.setattr(view, "pick_point", lambda xy, radius=9.0: 25)
    p.lasso_btn.setChecked(True)  # takes the mouse from the camera
    view._on_release(_release((10.0, 10.0), (10.0, 10.0)))
    assert picked == []


def test_picking_a_point_makes_its_tree_current(panel):
    p, _view = panel
    p.on_pick_tree(25)  # a point of tree 2
    assert p.current == 2
    assert p._row_of(2) is not None
    assert p.tree_table.item(p._row_of(2), 1).isSelected()


def test_the_ground_is_nobodys_tree(panel):
    p, _view = panel
    p.on_pick_tree(45)  # an unassigned point
    assert p.current == 1


def test_picking_the_current_tree_changes_nothing(panel):
    p, view = panel
    view.select(np.arange(5))
    p.on_pick_tree(3)  # a point of tree 1, the current tree
    assert p.current == 1
    assert len(view.selected) == 5  # the selection survives
