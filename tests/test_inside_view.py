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

"""Standing inside the cloud: a perspective view from a chosen point.

An orthographic view of a dense plot is a wall — every stem at every
distance is the same size — so the inside view is how you see what is in
front of what. The things that have to hold: the camera goes back exactly
where it was when you step out, the moves that frame a whole tree take you
out rather than burying the camera in its trunk, and the lasso still selects
what you can see, which in perspective is a different projection and has a
behind-you case the orthographic one never had.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication  # noqa: E402

from segfix.cloudview import INSIDE_DISTANCE, CloudView  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def view():
    try:
        v = CloudView()
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no vispy canvas available: {exc}")
    rng = np.random.default_rng(0)
    coords = (rng.random((500, 3)) * [40.0, 40.0, 20.0]).astype(np.float32)
    v.canvas.size = (1200, 900)
    v._coords = coords
    v._shown = np.ones(len(coords), dtype=bool)
    v._face_color = np.ones((len(coords), 4), np.float32)
    v._selected = np.zeros(len(coords), dtype=bool)
    v.reset_view()
    yield v


def camera_state(view):
    cam = view.view.camera
    return cam.fov, cam.scale_factor, tuple(cam.center)


def test_the_view_starts_outside_and_orthographic(view):
    assert not view.inside_view
    assert view.view.camera.fov == 0.0


def test_stepping_inside_stands_at_the_point_in_perspective(view):
    view.enter_inside_view(origin=(10.0, 12.0, 3.0), fov=70.0)
    cam = view.view.camera
    assert view.inside_view
    assert cam.fov == 70.0
    assert tuple(cam.center) == (10.0, 12.0, 3.0)
    # The pivot sits just in front of the eye, so dragging turns the head
    # rather than circling the plot.
    assert cam.scale_factor == INSIDE_DISTANCE


def test_stepping_out_puts_the_camera_back_exactly(view):
    before = camera_state(view)
    view.enter_inside_view(origin=(5.0, 5.0, 2.0))
    view.leave_inside_view()
    assert camera_state(view) == before
    assert not view.inside_view


def test_moving_about_inside_keeps_the_original_to_return_to(view):
    """Double-clicking a new spot re-enters at that point; the way back out
    is still to where the camera was before any of it."""
    before = camera_state(view)
    view.enter_inside_view(origin=(5.0, 5.0, 2.0))
    view.enter_inside_view(origin=(20.0, 7.0, 4.0))
    assert tuple(view.view.camera.center) == (20.0, 7.0, 4.0)
    view.leave_inside_view()
    assert camera_state(view) == before


def test_leaving_when_already_outside_does_nothing(view):
    before = camera_state(view)
    view.leave_inside_view()
    assert camera_state(view) == before


def test_the_lens_can_be_changed_from_inside(view):
    view.enter_inside_view(origin=(1.0, 1.0, 1.0), fov=70.0)
    view.set_inside_fov(35.0)
    assert view.view.camera.fov == 35.0


def test_the_lens_setting_is_ignored_from_outside(view):
    """It would otherwise leave the camera in perspective with no way back
    through the toggle."""
    view.set_inside_fov(35.0)
    assert view.view.camera.fov == 0.0


# -- the moves that mean "stand back" ----------------------------------------
def test_flying_to_a_tree_steps_back_out(view):
    view.enter_inside_view(origin=(5.0, 5.0, 2.0))
    view.fly_to((20.0, 20.0, 10.0), span=8.0)
    assert not view.inside_view
    assert view.view.camera.fov == 0.0
    assert view.view.camera.scale_factor > INSIDE_DISTANCE


def test_resetting_the_view_steps_back_out(view):
    view.enter_inside_view(origin=(5.0, 5.0, 2.0))
    view.reset_view()
    assert not view.inside_view
    assert view.view.camera.fov == 0.0


# -- selecting from inside ---------------------------------------------------
def test_the_lasso_can_still_project_what_you_can_see(view):
    """The lasso projects the cloud to canvas pixels; in perspective that is
    a different matrix with a w-divide, and points behind the eye have to
    come back invalid rather than mirrored into the view in front."""
    view.enter_inside_view(origin=(20.0, 20.0, 10.0), fov=70.0)
    cam = view.view.camera
    cam.azimuth, cam.elevation = 0.0, 0.0

    xy, valid = view.project_to_canvas(view.coords)
    assert valid.any(), "nothing projected from inside the cloud"
    on_screen = valid & (xy[:, 0] > 0) & (xy[:, 0] < 1200)
    assert on_screen.any()

    # A point a long way behind the eye is not drawn in front of it.
    behind = np.array([[20.0, 20.0 + 1000.0, 10.0]], dtype=np.float32)
    ahead = np.array([[20.0, 20.0 - 1000.0, 10.0]], dtype=np.float32)
    _, behind_valid = view.project_to_canvas(behind)
    _, ahead_valid = view.project_to_canvas(ahead)
    assert behind_valid.tolist() != ahead_valid.tolist()


def test_distance_actually_changes_size_from_inside(view):
    """What the mode is for. Orthographic draws a pair of points a metre
    apart the same size wherever they are, which is why a dense plot reads
    as a wall; in perspective the far pair converges."""
    # Both pairs ahead of the eye at (20, 20, 10) looking along +y: one 5 m
    # away, one 35 m away, each a metre wide.
    near = np.array([[20.0, 25.0, 10.0], [21.0, 25.0, 10.0]], np.float32)
    far = np.array([[20.0, 55.0, 10.0], [21.0, 55.0, 10.0]], np.float32)

    def spread(points):
        xy, valid = view.project_to_canvas(points)
        assert valid.all()
        return abs(xy[1, 0] - xy[0, 0])

    cam = view.view.camera
    cam.azimuth, cam.elevation = 0.0, 0.0
    outside_near, outside_far = spread(near), spread(far)
    assert outside_near == pytest.approx(outside_far, rel=1e-3)

    view.enter_inside_view(origin=(20.0, 20.0, 10.0), fov=70.0)
    cam.azimuth, cam.elevation = 0.0, 0.0
    assert spread(near) > spread(far) * 3
