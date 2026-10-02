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

"""The Top / Front / Back / Left / Right / Bottom / 3D buttons under the point
size: each turns the camera to that side and leaves the pivot and zoom alone.

Real panel on a real (offscreen) vispy canvas; skipped where no canvas can be
created. What each view shows is checked by projecting the world axes: for
Top, +X runs right and +Y runs up the screen, and so on.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication, QToolButton  # noqa: E402


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

    view.canvas.size = (800, 800)
    empty = PointCloud(coords=np.empty((0, 3), np.float32),
                       labels=np.empty(0, np.int32))
    view.load_cloud(empty)
    seg = SegFixController(view, empty)
    p = SegFixWidget(seg)
    rng = np.random.default_rng(0)
    cloud = PointCloud(coords=rng.random((50, 3)).astype(np.float32),
                       labels=np.ones(50, np.int32))
    view.load_cloud(cloud)
    seg.set_cloud(cloud)
    yield p, view
    p.deleteLater()


def _screen_axes(view):
    """Where world +X, +Y and +Z point on screen: name -> (right, up)."""
    tr = view.view.scene.node_transform(view.canvas.scene)
    origin = tr.map([0, 0, 0])
    origin = origin[:2] / origin[3]
    out = {}
    for name, d in (("X", [1, 0, 0]), ("Y", [0, 1, 0]), ("Z", [0, 0, 1])):
        p = tr.map(d)
        dx, dy = p[:2] / p[3] - origin
        out[name] = (int(np.sign(round(dx, 3))), int(np.sign(round(-dy, 3))))
    return out


def _click(p, name):
    btn = p._point_size_overlay.findChild(QToolButton, f"view_{name}")
    assert btn is not None, f"no {name} button"
    btn.click()


# (right, up) on screen for each world axis; (0, 0) = straight into the screen.
EXPECTED = {
    "top": {"X": (1, 0), "Y": (0, 1), "Z": (0, 0)},
    "bottom": {"X": (1, 0), "Y": (0, -1), "Z": (0, 0)},
    "front": {"X": (1, 0), "Y": (0, 0), "Z": (0, 1)},
    "back": {"X": (-1, 0), "Y": (0, 0), "Z": (0, 1)},
    "left": {"X": (0, 0), "Y": (-1, 0), "Z": (0, 1)},
    "right": {"X": (0, 0), "Y": (1, 0), "Z": (0, 1)},
}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_each_button_shows_that_side(panel, name):
    p, view = panel
    _click(p, name)
    assert _screen_axes(view) == EXPECTED[name]


def test_views_keep_pivot_and_zoom_and_3d_restores_the_tilt(panel):
    p, view = panel
    cam = view.view.camera
    cam.center = (1.0, 2.0, 3.0)
    cam.scale_factor = 7.5
    start = (cam.azimuth, cam.elevation)
    for name in ("top", "left", "3d"):
        _click(p, name)
        assert tuple(cam.center) == (1.0, 2.0, 3.0)
        assert cam.scale_factor == 7.5
    assert (cam.azimuth, cam.elevation) == start


# -- the inside (bubble) view ------------------------------------------------
def test_the_inside_toggle_stands_the_camera_in_the_cloud(panel):
    p, view = panel
    view.view.camera.center = (2.0, 3.0, 1.0)  # as a double-click would
    p.inside_btn.setChecked(True)
    assert view.inside_view
    assert view.view.camera.fov == p.fov_spin.value()
    assert tuple(view.view.camera.center) == (2.0, 3.0, 1.0)
    p.inside_btn.setChecked(False)
    assert not view.inside_view
    assert view.view.camera.fov == 0.0


def test_the_fov_spinner_works_from_inside(panel):
    p, view = panel
    p.inside_btn.setChecked(True)
    p.fov_spin.setValue(45)
    assert view.view.camera.fov == 45.0


def test_the_toggle_pops_back_out_when_the_camera_does(panel):
    """Stepping through the queue flies the camera to frame a whole tree,
    which leaves the bubble — the button has to say so."""
    p, view = panel
    p._set_current(None, fly=False)
    p.inside_btn.setChecked(True)
    p._set_current(1, fly=True)  # the one tree this cloud has
    assert not view.inside_view
    assert not p.inside_btn.isChecked()


def test_the_toggle_refuses_an_empty_cloud(panel):
    from segfix.model import PointCloud

    p, view = panel
    empty = PointCloud(coords=np.empty((0, 3), np.float32),
                       labels=np.empty(0, np.int32))
    view.load_cloud(empty)
    p.c.set_cloud(empty)
    p.inside_btn.setChecked(True)
    assert not p.inside_btn.isChecked()
    assert not view.inside_view


def test_the_inside_toggle_looks_different_while_it_is_on(panel):
    """It changes how the whole canvas is projected, and a flat toolbutton
    looks the same inside or out — the FOV spinner follows it too, so the
    pair reads as one control."""
    p, _ = panel
    assert ":checked" in p.inside_btn.styleSheet()
    assert not p.fov_spin.isEnabled()
    p.inside_btn.setChecked(True)
    assert p.fov_spin.isEnabled()
    p.inside_btn.setChecked(False)
    assert not p.fov_spin.isEnabled()
