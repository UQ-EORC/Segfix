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

"""What a tree ID of -1 means, and opening Sylva's output because of it.

segfix writes -1 for a point dismissed as noise. Sylva's `trees --segment`
writes it for every point no stem claimed, which is segfix's *unassigned*.
Read the wrong way round, a Sylva plot opens with its whole ground flagged
as noise: hidden by F, and missing from the grey points the workflow lassoes
back into trees. Nothing in the file tells the two apart, so it is asked
once per project and remembered, and whichever way it was read, -1 is what
goes back to the file.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from segfix import operations as ops
from segfix import workspace
from segfix.model import NOISE, UNASSIGNED
from segfix.treecatalog import open_catalog


def sylva_cloud(path, n=240, seed=0):
    """What `sylva trees --segment` leaves behind: tree_id from 1, -1 for
    everything no stem claimed."""
    laspy = pytest.importorskip("laspy")
    rng = np.random.default_rng(seed)
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [0.0, 0.0, 0.0]
    header.add_extra_dim(laspy.ExtraBytesParams(name="tree_id", type=np.int32))
    las = laspy.LasData(header)
    coords = rng.random((n, 3)) * 10
    las.x, las.y, las.z = coords.T
    labels = np.repeat([1, 2, -1], n // 3).astype(np.int32)
    las.tree_id = labels
    las.write(str(path))
    return labels


@pytest.fixture
def cloud(tmp_path):
    path = tmp_path / "plot_segmented.las"
    labels = sylva_cloud(path)
    return path, labels


# -- reading -----------------------------------------------------------------
def test_without_an_answer_minus_one_stays_segfix_noise(cloud):
    """Headless callers and every existing project keep the old reading."""
    path, _ = cloud
    catalog = open_catalog(str(path))
    assert int((catalog.labels == NOISE).sum()) == 80
    assert catalog.negative_means == "noise"


def test_told_it_is_unassigned_the_ground_comes_in_unassigned(cloud):
    path, _ = cloud
    catalog = open_catalog(str(path), negative_prompt=lambda n: "unassigned")
    assert int((catalog.labels == UNASSIGNED).sum()) == 80
    assert not (catalog.labels == NOISE).any()
    assert sorted(catalog.records) == [1, 2]


def test_the_prompt_is_told_how_many_points_it_is_about(cloud):
    path, _ = cloud
    asked = []
    open_catalog(str(path), negative_prompt=lambda n: asked.append(n) or "unassigned")
    assert asked == [80]


def test_a_cloud_with_no_negative_labels_is_never_asked(tmp_path):
    path = tmp_path / "plain.las"
    laspy = pytest.importorskip("laspy")
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    las = laspy.LasData(header)
    las.x = las.y = las.z = np.arange(30, dtype=float)
    las.treeID = np.repeat([0, 1, 2], 10).astype(np.int32)
    las.write(str(path))

    asked = []
    open_catalog(str(path), negative_prompt=lambda n: asked.append(n) or "noise")
    assert asked == []


def test_the_answer_is_asked_once_and_remembered(tmp_path, cloud):
    """It is a fact about the file, not a choice to revisit: a project that
    answered once opens the same way every time."""
    path, _ = cloud
    ws = tmp_path / "project"
    data = workspace.create_workspace(str(path), ws)

    asked = []
    first = open_catalog(str(data), negative_prompt=lambda n: asked.append(n) or "unassigned")
    assert first.negative_means == "unassigned"
    manifest = json.loads((ws / workspace.MANIFEST_NAME).read_text())
    assert manifest["settings"]["negative_means"] == "unassigned"

    again = open_catalog(str(data), negative_prompt=lambda n: asked.append(n) or "noise")
    assert again.negative_means == "unassigned"   # the saved answer wins
    assert len(asked) == 1


def test_the_answer_can_be_given_outright(cloud):
    path, _ = cloud
    catalog = open_catalog(str(path), negative_means="unassigned")
    assert int((catalog.labels == UNASSIGNED).sum()) == 80


# -- writing -----------------------------------------------------------------
def read_back(path):
    laspy = pytest.importorskip("laspy")
    return np.asarray(laspy.read(str(path)).tree_id)


def test_what_came_in_as_minus_one_goes_back_as_minus_one(cloud, tmp_path):
    """Sylva reads 0 as a tree numbered zero, so unassigned has to go home
    the way it arrived."""
    path, _ = cloud
    catalog = open_catalog(str(path), negative_prompt=lambda n: "unassigned")
    # Unassign a few of tree 1's points, as D would.
    cloud, global_idx = catalog.load([1], margin=0.0)
    ops.unassign(cloud, np.flatnonzero(cloud.labels == 1)[:10])
    catalog.apply(cloud, global_idx)
    catalog.save()

    written = read_back(path)
    assert int((written == -1).sum()) == 90
    assert not (written == 0).any()


def test_noise_marked_in_a_sylva_cloud_is_written_as_minus_one(cloud):
    """The file has one way to say "no tree". Which points were noise stays
    in segfix's own sidecar, as it already does for an unsigned column."""
    path, _ = cloud
    catalog = open_catalog(str(path), negative_prompt=lambda n: "unassigned")
    cloud, global_idx = catalog.load([2], margin=0.0)
    ops.mark_noise(cloud, np.flatnonzero(cloud.labels == 2)[:5])
    catalog.apply(cloud, global_idx)
    catalog.save()

    written = read_back(path)
    assert int((written == -1).sum()) == 85
    assert not (written == 0).any()


def test_a_segfix_cloud_still_round_trips_its_noise(cloud):
    path, _ = cloud
    catalog = open_catalog(str(path), negative_prompt=lambda n: "noise")
    cloud, global_idx = catalog.load([1], margin=0.0)
    ops.unassign(cloud, np.flatnonzero(cloud.labels == 1)[:10])
    catalog.apply(cloud, global_idx)
    catalog.save()

    written = read_back(path)
    assert int((written == 0).sum()) == 10    # unassigned stays 0
    assert int((written == -1).sum()) == 80   # the file's noise stays noise


def test_the_whole_round_trip_leaves_sylva_what_it_expects(cloud):
    """Open, fix, save, and read it back the way Sylva would: trees from 1,
    -1 for everything else, and no tree numbered zero."""
    path, labels_before = cloud
    catalog = open_catalog(str(path), negative_prompt=lambda n: "unassigned")
    cloud, global_idx = catalog.load([2], margin=50.0)   # tree 2 and the ground
    ops.reassign(cloud, np.flatnonzero(cloud.labels == UNASSIGNED)[:20], 2)
    catalog.apply(cloud, global_idx)
    catalog.save()

    written = read_back(path)
    assert set(np.unique(written).tolist()) == {-1, 1, 2}
    assert int((written == 2).sum()) == 100
    assert int((written == -1).sum()) == 60
    assert len(written) == len(labels_before)
