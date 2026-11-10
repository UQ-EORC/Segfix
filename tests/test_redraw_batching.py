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

"""One upload of the cloud per edit and per tree load, and the point spacing
measured off the GUI thread.

Every setter on the view sends all the shown points to the GPU again. An
edit recolours and then re-applies visibility; loading a tree draws it and
then re-applies the point size. Each used to pay for the cloud two or three
times over. These tests count the uploads on a real (offscreen) canvas, and
check that what ends up on screen is what it always was.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication  # noqa: E402

from segfix import analysis  # noqa: E402
from segfix.model import UNASSIGNED, PointCloud  # noqa: E402
from segfix.viewer import batched, colors_for_labels  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def _cloud(n_per_tree: int = 60, seed: int = 0) -> PointCloud:
    """Two trees side by side over a strip of unassigned ground."""
    rng = np.random.default_rng(seed)
    coords = np.vstack([
        rng.random((n_per_tree, 3)) + [0.0, 0, 1],
        rng.random((n_per_tree, 3)) + [0.8, 0, 1],
        rng.random((n_per_tree, 3)) * [1.8, 1, 0.1],
    ]).astype(np.float32)
    labels = np.repeat([1, 2, UNASSIGNED], n_per_tree).astype(np.int32)
    return PointCloud(coords=coords, labels=labels)


@pytest.fixture
def view():
    try:
        from segfix.cloudview import CloudView

        view = CloudView()
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no vispy canvas available: {exc}")
    return view


def _count_uploads(view) -> list:
    """Record every ``set_data`` on the point markers; returns the list the
    calls' keyword arguments are appended to."""
    calls = []
    real = view.markers.set_data

    def set_data(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    view.markers.set_data = set_data
    return calls


@pytest.fixture
def panel(view):
    from segfix.widgets import SegFixController, SegFixWidget

    cloud = _cloud()
    view.load_cloud(cloud)
    seg = SegFixController(view, cloud)
    p = SegFixWidget(seg)
    seg.set_cloud(cloud)
    p._set_current(1, fly=False)
    yield p, seg, cloud
    p.deleteLater()


# -- the batch itself ---------------------------------------------------------
def test_setters_in_a_batch_draw_once_when_it_ends(view):
    cloud = _cloud()
    view.load_cloud(cloud)
    uploads = _count_uploads(view)

    with view.batch():
        view.face_color = colors_for_labels(cloud.labels)
        view.size = 7.0
        view.shown = cloud.labels != UNASSIGNED
        view.select(np.arange(10))
        assert uploads == []  # nothing yet
    assert len(uploads) == 1
    # ...and it is the final state that was drawn.
    np.testing.assert_array_equal(
        uploads[0]["pos"], cloud.coords[cloud.labels != UNASSIGNED]
    )
    assert uploads[0]["size"] == 7.0
    assert view.highlight.visible


def test_outside_a_batch_every_setter_still_draws(view):
    cloud = _cloud()
    view.load_cloud(cloud)
    uploads = _count_uploads(view)
    view.size = 5.0
    view.shown = cloud.labels != UNASSIGNED
    assert len(uploads) == 2


def test_nested_batches_draw_once_at_the_outermost_exit(view):
    cloud = _cloud()
    view.load_cloud(cloud)
    uploads = _count_uploads(view)
    with view.batch():
        with view.batch():
            view.size = 4.0
        assert uploads == []
        view.size = 6.0
    assert len(uploads) == 1 and uploads[0]["size"] == 6.0


def test_a_batch_that_changes_nothing_draws_nothing(view):
    view.load_cloud(_cloud())
    uploads = _count_uploads(view)
    with view.batch():
        pass
    assert uploads == []


def test_a_batch_still_draws_if_the_block_raises(view):
    """The view must not be left showing the old cloud, nor stuck batching."""
    cloud = _cloud()
    view.load_cloud(cloud)
    uploads = _count_uploads(view)
    with pytest.raises(RuntimeError):
        with view.batch():
            view.size = 9.0
            raise RuntimeError("mid-edit")
    assert len(uploads) == 1
    view.size = 3.0  # not batching any more
    assert len(uploads) == 2


def test_batched_is_a_no_op_for_a_view_without_batches():
    class Bare:
        pass

    with batched(Bare()):
        pass  # must simply run


# -- everything shown: no copy through the mask --------------------------------
def test_a_fully_shown_cloud_is_uploaded_as_it_is(view):
    cloud = _cloud()
    view.load_cloud(cloud)
    uploads = _count_uploads(view)
    view.face_color = colors_for_labels(cloud.labels)
    np.testing.assert_array_equal(uploads[-1]["pos"], cloud.coords)
    np.testing.assert_array_equal(
        uploads[-1]["face_color"], colors_for_labels(cloud.labels)
    )
    # Hide some, and only the shown ones go up, colours in step with points.
    shown = cloud.labels == 1
    view.shown = shown
    np.testing.assert_array_equal(uploads[-1]["pos"], cloud.coords[shown])
    np.testing.assert_array_equal(
        uploads[-1]["face_color"], colors_for_labels(cloud.labels)[shown]
    )


# -- an edit -------------------------------------------------------------------
def test_an_edit_uploads_the_cloud_once(panel):
    p, seg, cloud = panel
    uploads = _count_uploads(seg.view)
    ground = np.flatnonzero(cloud.labels == UNASSIGNED)[:20]
    seg.view.select(ground)

    p.on_add()  # the selection joins tree 1

    assert (cloud.labels[ground] == 1).all()
    assert len(uploads) == 1
    # What is on screen is the edited cloud: new colours, selection gone.
    np.testing.assert_array_equal(
        seg.view.face_color, colors_for_labels(cloud.labels)
    )
    assert not seg.view.selected_mask.any()
    assert not seg.view.highlight.visible


def test_an_edit_that_hides_points_still_draws_them_gone_at_once(panel):
    """The visibility half of the edit is in the same single draw."""
    p, seg, cloud = panel
    p.show_unassigned.setChecked(False)
    uploads = _count_uploads(seg.view)
    mine = np.flatnonzero(cloud.labels == 1)[:15]
    seg.view.select(mine)

    p.on_unassign()

    assert len(uploads) == 1
    assert not seg.view.shown[mine].any()
    np.testing.assert_array_equal(
        uploads[0]["pos"], cloud.coords[seg.view.shown]
    )


def test_undo_uploads_once_too(panel):
    p, seg, cloud = panel
    seg.view.select(np.flatnonzero(cloud.labels == UNASSIGNED)[:5])
    p.on_add()
    uploads = _count_uploads(seg.view)
    p.on_undo()
    assert len(uploads) == 1
    np.testing.assert_array_equal(
        seg.view.face_color, colors_for_labels(cloud.labels)
    )


# -- loading a tree ---------------------------------------------------------------
def test_loading_a_tree_uploads_it_once(view, tmp_path):
    from segfix.scene_ui import SceneController
    from segfix.treecatalog import open_catalog
    from segfix.widgets import SegFixController, SegFixWidget
    from tests.test_core import _write_arbor_las

    cloud = _cloud()
    path = tmp_path / "plot.las"
    _write_arbor_las(path, cloud.coords.astype(np.float64), cloud.labels)
    empty = PointCloud(coords=np.empty((0, 3), np.float32),
                       labels=np.empty(0, np.int32))
    view.load_cloud(empty)
    seg = SegFixController(view, empty)
    p = SegFixWidget(seg)
    p.size_spin.setValue(6.0)  # not the size the scene loads at
    scene = SceneController(view, open_catalog(str(path)), seg)
    uploads = _count_uploads(view)

    scene.load_tree(1)

    assert len(uploads) == 1
    assert uploads[0]["size"] == 6.0  # the panel's size, in that one draw
    assert len(uploads[0]["pos"]) == seg.cloud.n_points
    assert p.current == 1
    p.deleteLater()


# -- point spacing, ahead of need ----------------------------------------------------
def _controller(view, cloud, monkeypatch, above=10):
    from segfix import widgets
    from segfix.widgets import SegFixController

    monkeypatch.setattr(widgets, "SPACING_PREFETCH_ABOVE", above)
    measured = []
    real = analysis.point_spacing

    def counting(coords, *args, **kwargs):
        measured.append(coords)
        return real(coords, *args, **kwargs)

    monkeypatch.setattr(analysis, "point_spacing", counting)
    view.load_cloud(cloud)
    seg = SegFixController(view, cloud)
    return seg, measured, real


def test_spacing_is_measured_on_a_worker_when_a_cloud_arrives(view, monkeypatch):
    cloud = _cloud(n_per_tree=400)
    seg, measured, real = _controller(view, cloud, monkeypatch)

    seg.set_cloud(cloud)
    assert seg._spacing_job is not None
    seg._spacing_job[1].join(timeout=30)

    # Collected, not measured again, and the number the GUI thread would
    # have got from the same points.
    assert seg.cluster_gap == real(view.coords) * seg.cluster_gap_factor
    assert len(measured) == 1 and measured[0] is view.coords
    assert seg._spacing_job is None
    assert seg.cluster_gap == seg._cluster_spacing * seg.cluster_gap_factor
    assert len(measured) == 1


def test_asking_before_the_worker_is_done_waits_for_it(view, monkeypatch):
    import threading

    cloud = _cloud(n_per_tree=400)
    seg, measured, real = _controller(view, cloud, monkeypatch)
    release = threading.Event()
    counting = analysis.point_spacing

    def slow(coords, *args, **kwargs):
        release.wait(timeout=30)
        return counting(coords, *args, **kwargs)

    monkeypatch.setattr(analysis, "point_spacing", slow)
    seg.set_cloud(cloud)
    threading.Timer(0.05, release.set).start()

    assert seg.cluster_gap == real(view.coords) * seg.cluster_gap_factor
    assert len(measured) == 1


def test_a_worker_for_other_points_is_ignored(view, monkeypatch):
    """The gap is for the points on screen: if the view was handed different
    ones after the measurement started, it is measured afresh."""
    cloud = _cloud(n_per_tree=400)
    seg, measured, real = _controller(view, cloud, monkeypatch)
    seg.set_cloud(cloud)
    seg._spacing_job[1].join(timeout=30)

    other = _cloud(n_per_tree=200, seed=5)
    other.coords *= 3.0
    view.load_cloud(other)

    assert seg.cluster_gap == real(view.coords) * seg.cluster_gap_factor
    assert measured[-1] is view.coords


def test_a_new_cloud_drops_the_old_spacing(view, monkeypatch):
    cloud = _cloud(n_per_tree=400)
    seg, _measured, real = _controller(view, cloud, monkeypatch)
    seg.set_cloud(cloud)
    first = seg.cluster_gap

    other = _cloud(n_per_tree=300, seed=9)
    other.coords *= 2.0
    view.load_cloud(other)
    seg.set_cloud(other)

    assert seg.cluster_gap == real(view.coords) * seg.cluster_gap_factor
    assert seg.cluster_gap != first


def test_no_worker_for_a_small_cloud_or_with_prefetch_off(view, monkeypatch):
    cloud = _cloud(n_per_tree=400)
    seg, measured, real = _controller(view, cloud, monkeypatch, above=10**9)
    seg.set_cloud(cloud)
    assert seg._spacing_job is None and measured == []
    assert seg.cluster_gap == real(view.coords) * seg.cluster_gap_factor

    seg2, measured2, _ = _controller(view, cloud, monkeypatch, above=10)
    seg2.cluster_prefetch = False
    seg2.set_cloud(cloud)
    assert seg2._spacing_job is None and measured2 == []
