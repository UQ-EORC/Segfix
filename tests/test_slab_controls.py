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

"""The cross-section slab is adjusted on the canvas, not in a popover.

It is drawn as a box with grips, as CloudCompare's is: an arrow on each
face along the axis moves that face, a square slides the whole slab, and
Ctrl+wheel nudges it without leaving the canvas or whatever tool is armed.
The sliders in the popover stay the one source of truth, so every route
ends in the same place.
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
    coords = rng.random((200, 3)).astype(np.float32) * [4, 4, 10]
    coords[0] = [0, 0, 0]
    coords[1] = [4, 4, 10]  # a known extent: z runs 0..10
    labels = np.ones(200, np.int32)
    empty = PointCloud(coords=np.empty((0, 3), np.float32),
                       labels=np.empty(0, np.int32))
    view.load_cloud(empty)
    seg = SegFixController(view, empty)
    p = SegFixWidget(seg)
    cloud = PointCloud(coords=coords, labels=labels)
    view.load_cloud(cloud)
    seg.set_cloud(cloud)
    p._set_current(1, fly=False)
    view.reset_view()
    yield p, view
    p.deleteLater()


def _wheel(dy, modifiers):
    return types.SimpleNamespace(delta=(0.0, dy), modifiers=modifiers,
                                 handled=False, blocked=False)


# -- the box --------------------------------------------------------------
def test_the_slab_is_drawn_only_while_the_section_is_on(panel):
    p, view = panel
    assert not view.slab.visible
    assert not p._slab_handles.visible
    p.cross_enable.setChecked(True)
    assert view.slab.visible
    assert p._slab_handles.visible
    p.cross_enable.setChecked(False)
    assert not view.slab.visible
    assert not p._slab_handles.visible


def test_the_grips_sit_on_the_slab_faces(panel):
    p, view = panel
    p.cross_enable.setChecked(True)
    p.set_slab_range(2.0, 5.0)
    lo, hi, mid = p._slab_handles._points
    assert lo[2] == pytest.approx(2.0)
    assert hi[2] == pytest.approx(5.0)
    assert mid[2] == pytest.approx(3.5)
    assert lo[0] == hi[0] == mid[0]  # centred across the cloud


def test_the_box_follows_the_sliders(panel):
    """Whichever way the slab is set, the box and the popover agree."""
    p, view = panel
    p.cross_enable.setChecked(True)
    p.cross_min_slider.setValue(300)
    p.cross_max_slider.setValue(600)
    lo, hi = p.slab_range()
    assert (lo, hi) == pytest.approx((3.0, 6.0), abs=0.02)
    assert p._slab_handles._points[0][2] == pytest.approx(lo)


# -- the wheel ------------------------------------------------------------
def test_ctrl_wheel_slides_the_slab_a_twentieth_of_its_thickness(panel):
    p, view = panel
    p.cross_enable.setChecked(True)
    p.set_slab_range(2.0, 4.0)
    view._on_wheel(_wheel(+1, ("Control",)))
    assert p.slab_range() == pytest.approx((2.1, 4.1), abs=0.02)
    view._on_wheel(_wheel(-1, ("Control",)))
    assert p.slab_range() == pytest.approx((2.0, 4.0), abs=0.02)


def test_ctrl_shift_wheel_changes_the_thickness_about_the_middle(panel):
    p, view = panel
    p.cross_enable.setChecked(True)
    p.set_slab_range(2.0, 4.0)
    view._on_wheel(_wheel(+1, ("Control", "Shift")))
    assert p.slab_range() == pytest.approx((1.9, 4.1), abs=0.02)


def test_the_wheel_is_taken_from_the_camera_only_with_ctrl(panel):
    p, view = panel
    p.cross_enable.setChecked(True)
    p.set_slab_range(2.0, 4.0)
    plain = _wheel(+1, ())
    view._on_wheel(plain)
    assert not plain.blocked and not plain.handled
    assert p.slab_range() == pytest.approx((2.0, 4.0), abs=0.02)
    held = _wheel(+1, ("Control",))
    view._on_wheel(held)
    assert held.blocked and held.handled


def test_sliding_stops_at_the_cloud_edge_without_thinning(panel):
    p, view = panel
    p.cross_enable.setChecked(True)
    p.set_slab_range(9.5, 10.0)
    view._on_wheel(_wheel(+1, ("Control",)))
    lo, hi = p.slab_range()
    assert hi == pytest.approx(10.0, abs=0.02)
    assert hi - lo == pytest.approx(0.5, abs=0.03)


def test_the_slab_never_thins_to_nothing(panel):
    p, view = panel
    p.cross_enable.setChecked(True)
    p.set_slab_range(5.0, 5.03)
    for _ in range(5):
        view._on_wheel(_wheel(-1, ("Control", "Shift")))
    lo, hi = p.slab_range()
    assert hi - lo >= p.SLAB_MIN_THICKNESS - 0.011  # slider resolution


def test_with_the_section_off_the_wheel_says_so(panel):
    p, view = panel
    said = []
    view.on_status = said.append
    view._on_wheel(_wheel(+1, ("Control",)))
    assert "Cross section" in said[-1]


# -- the grips ------------------------------------------------------------
def test_a_drag_is_read_along_the_axis_as_it_shows_on_screen(panel):
    """Looking from the front, Z runs up the screen: dragging a grip up by
    the pixels a metre spans moves the face a metre. Dragging sideways,
    across the axis, moves it nowhere."""
    p, view = panel
    p.cross_enable.setChecked(True)
    p.set_slab_range(2.0, 4.0)
    view.set_view("front")
    view.canvas.update()
    handles = p._slab_handles
    handles.place()
    d = handles._axis_px
    assert d is not None and abs(d[1]) > 1 and abs(d[0]) < 1e-3
    assert handles.metres_for(d[0], d[1]) == pytest.approx(1.0)
    assert handles.metres_for(50.0, 0.0) == pytest.approx(0.0, abs=1e-6)


def test_dragging_a_face_moves_only_that_face(panel):
    p, view = panel
    p.cross_enable.setChecked(True)
    p.set_slab_range(2.0, 4.0)
    p._on_slab_press()
    p._on_slab_drag("hi", 1.5)
    assert p.slab_range() == pytest.approx((2.0, 5.5), abs=0.02)
    p._on_slab_drag("hi", 2.0)  # cumulative from the press, not from last
    assert p.slab_range() == pytest.approx((2.0, 6.0), abs=0.02)
    p._on_slab_release()
    p._on_slab_press()
    p._on_slab_drag("lo", -1.0)
    assert p.slab_range() == pytest.approx((1.0, 6.0), abs=0.02)


def test_dragging_the_middle_slides_the_whole_slab(panel):
    p, view = panel
    p.cross_enable.setChecked(True)
    p.set_slab_range(2.0, 4.0)
    p._on_slab_press()
    p._on_slab_drag("mid", 3.0)
    assert p.slab_range() == pytest.approx((5.0, 7.0), abs=0.02)
    p._on_slab_drag("mid", 30.0)  # past the top: stops at the edge, same size
    assert p.slab_range() == pytest.approx((8.0, 10.0), abs=0.02)


def test_a_face_cannot_be_dragged_through_the_other(panel):
    p, view = panel
    p.cross_enable.setChecked(True)
    p.set_slab_range(2.0, 4.0)
    p._on_slab_press()
    p._on_slab_drag("lo", 5.0)
    lo, hi = p.slab_range()
    assert hi == pytest.approx(4.0, abs=0.02)
    assert hi - lo >= p.SLAB_MIN_THICKNESS - 0.011


def test_an_axis_pointing_at_the_camera_cannot_be_dragged(panel):
    """From the top, Z has no direction on screen; a drag means nothing."""
    p, view = panel
    p.cross_enable.setChecked(True)
    view.set_view("top")
    view.canvas.update()
    p._slab_handles.place()
    assert p._slab_handles.metres_for(10.0, 10.0) is None


def test_a_new_cloud_takes_the_box_with_it(panel):
    """The section resets on a tree load; the box must not stay behind."""
    from segfix.model import PointCloud

    p, view = panel
    p.cross_enable.setChecked(True)
    p.set_slab_range(2.0, 4.0)
    cloud = PointCloud(coords=np.zeros((0, 3), np.float32),
                       labels=np.zeros(0, np.int32))
    view.load_cloud(cloud)
    p.c.set_cloud(cloud)
    assert not view.slab.visible
    assert not p._slab_handles.visible
