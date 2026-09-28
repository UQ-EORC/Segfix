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

"""Point classes: a second per-point field (leaf, wood, ground, ... as
numbers the user names), edited, undone and saved alongside the tree IDs.

The save has to keep the promise the tree labels make: only the class values
that changed are written, and every other byte of the file stays as it was —
including the flag bits that share a byte with a LAS 0-5 classification.
"""

import numpy as np
import pytest

from segfix import density, operations as ops
from segfix.model import PointCloud
from segfix.treecatalog import open_catalog

LEAF, WOOD, GROUND = 1, 2, 3


def _write_las(path, coords, tree_id, classes, fmt=6, synthetic=None):
    """XYZ, a ``treeID`` Extra-Bytes column and the standard classification."""
    import laspy

    header = laspy.LasHeader(version="1.4", point_format=fmt)
    header.offsets = [0.0, 0.0, 0.0]
    header.scales = [0.001, 0.001, 0.001]
    header.add_extra_dim(laspy.ExtraBytesParams(name="treeID", type=np.int32))
    las = laspy.LasData(header)
    las.x, las.y, las.z = coords[:, 0], coords[:, 1], coords[:, 2]
    las.treeID = np.asarray(tree_id, dtype=np.int32)
    las.classification = np.asarray(classes, dtype=np.uint8)
    if synthetic is not None:
        las.synthetic = np.asarray(synthetic, dtype=bool)
    las.write(str(path))


def _write_ply(path, coords, tree_id, classes, class_type="f4"):
    """A binary PLY with a ``treeID`` and a ``semantic`` class property."""
    from plyfile import PlyData, PlyElement

    data = np.empty(len(coords), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("treeID", "i4"), ("semantic", class_type),
    ])
    data["x"], data["y"], data["z"] = coords.T
    data["treeID"] = tree_id
    data["semantic"] = classes
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))


def _two_trees():
    """Tree 1 at the origin, tree 2 a metre along x: trunk then crown."""
    z = np.arange(10) * 0.2
    a = np.column_stack([np.zeros(10), np.zeros(10), z])
    b = np.column_stack([np.ones(10), np.zeros(10), z])
    coords = np.vstack([a, b])
    tid = np.repeat([1, 2], 10)
    cls = np.tile(np.repeat([WOOD, LEAF], 5), 2)
    return coords, tid, cls


# -- the model ---------------------------------------------------------------
def test_class_and_tree_edits_share_one_undo_history():
    cloud = PointCloud(coords=np.zeros((4, 3)), labels=[1, 1, 2, 2],
                       classes=[LEAF, LEAF, LEAF, LEAF])
    ops.reassign(cloud, [0], 2)
    ops.set_class(cloud, [1, 2], WOOD, "wood")
    assert cloud.labels.tolist() == [2, 1, 2, 2]
    assert cloud.classes.tolist() == [LEAF, WOOD, WOOD, LEAF]

    cloud.undo()  # the class edit, last made
    assert cloud.classes.tolist() == [LEAF] * 4
    assert cloud.labels.tolist() == [2, 1, 2, 2]
    cloud.undo()  # then the tree edit
    assert cloud.labels.tolist() == [1, 1, 2, 2]
    cloud.redo()
    cloud.redo()
    assert cloud.labels.tolist() == [2, 1, 2, 2]
    assert cloud.classes.tolist() == [LEAF, WOOD, WOOD, LEAF]


def test_a_class_edit_reports_only_the_points_it_changed():
    cloud = PointCloud(coords=np.zeros((3, 3)), labels=[1, 1, 1],
                       classes=[LEAF, WOOD, LEAF])
    assert cloud.set_classes([0, 1, 2], WOOD, "wood") == 2
    assert cloud.last_changed.tolist() == [0, 2]


def test_a_cloud_without_classes_refuses_a_class_edit():
    cloud = PointCloud(coords=np.zeros((1, 3)), labels=[1])
    with pytest.raises(ValueError, match="no point-class field"):
        cloud.set_classes([0], WOOD, "wood")


# -- choosing the field -------------------------------------------------------
def test_a_las_offers_its_classification_by_its_usual_name(tmp_path):
    coords, tid, cls = _two_trees()
    for fmt in (3, 6):
        path = tmp_path / f"fmt{fmt}.las"
        _write_las(path, coords, tid, cls, fmt=fmt)
        cat = open_catalog(str(path))
        fields = cat.class_fields()
        assert "classification" in fields
        # Neither the tree ID nor the flag bytes are a class.
        assert "treeID" not in fields
        assert not {"bit_fields", "classification_flags"} & set(fields)
        assert cat.class_values_in("classification").tolist() == [LEAF, WOOD]


def test_a_bit_packed_classification_only_holds_five_bits(tmp_path):
    coords, tid, cls = _two_trees()
    path = tmp_path / "fmt3.las"
    _write_las(path, coords, tid, cls, fmt=3)
    cat = open_catalog(str(path), class_field="classification")
    assert cat.class_range() == (0, 31)


def test_the_tree_field_cannot_be_the_class_field(tmp_path):
    coords, tid, cls = _two_trees()
    path = tmp_path / "plot.las"
    _write_las(path, coords, tid, cls)
    cat = open_catalog(str(path))
    with pytest.raises(ValueError, match="tree ID field"):
        cat.set_class_field("treeID")
    with pytest.raises(ValueError, match="no field"):
        cat.set_class_field("nonesuch")


# -- load, edit, save ----------------------------------------------------------
@pytest.mark.parametrize("fmt", [3, 6])
def test_a_las_class_edit_writes_only_the_changed_class_bits(tmp_path, fmt):
    import laspy

    coords, tid, cls = _two_trees()
    synthetic = np.zeros(len(coords), dtype=bool)
    synthetic[::3] = True  # flag bits sharing the byte, in format 3
    path = tmp_path / "plot.las"
    _write_las(path, coords, tid, cls, fmt=fmt, synthetic=synthetic)
    before = laspy.read(str(path))

    cat = open_catalog(str(path), class_field="classification")
    cloud, gidx = cat.load([2], margin=0.0)
    crown = np.flatnonzero((cloud.labels == 2) & (cloud.classes == LEAF))
    ops.set_class(cloud, crown, WOOD, "wood")
    cat.apply(cloud, gidx)
    assert cat.has_unsaved_edits()
    msg = cat.save()
    assert msg.startswith("Saved 5 reclassified")

    after = laspy.read(str(path))
    expected = np.asarray(before.classification).copy()
    expected[(np.asarray(before.treeID) == 2) & (expected == LEAF)] = WOOD
    assert np.asarray(after.classification).tolist() == expected.tolist()
    # Everything else about each point is exactly as it was.
    assert np.asarray(after.synthetic).tolist() == synthetic.tolist()
    assert np.asarray(after.treeID).tolist() == tid.tolist()
    np.testing.assert_array_equal(after.X, before.X)
    assert not cat.has_unsaved_edits()
    assert cat.save() == "Nothing changed since last save"


def test_tree_and_class_edits_save_together(tmp_path):
    import laspy

    coords, tid, cls = _two_trees()
    path = tmp_path / "plot.las"
    _write_las(path, coords, tid, cls)
    cat = open_catalog(str(path), class_field="classification")
    cloud, gidx = cat.load([1, 2], margin=0.0)
    ops.reassign(cloud, np.flatnonzero(cloud.labels == 2)[:2], 1)
    ops.set_class(cloud, np.flatnonzero(cloud.labels == 1)[:3], GROUND, "ground")
    cat.apply(cloud, gidx)
    msg = cat.save()
    assert "2 changed point(s) and 3 reclassified" in msg

    after = laspy.read(str(path))
    assert (np.asarray(after.treeID) == 1).sum() == 12
    assert (np.asarray(after.classification) == GROUND).sum() == 3


@pytest.mark.parametrize("class_type", ["f4", "u1", "i2"])
def test_a_ply_class_property_round_trips_in_its_own_type(tmp_path, class_type):
    from plyfile import PlyData

    coords, tid, cls = _two_trees()
    path = tmp_path / "plot.ply"
    _write_ply(path, coords, tid, cls, class_type)
    cat = open_catalog(str(path), class_field="semantic")
    assert cat.class_fields() == ["semantic"]

    cloud, gidx = cat.load([1], margin=0.0)
    assert "semantic" not in cloud.attributes  # it's the class, not an extra
    ops.set_class(cloud, np.flatnonzero(cloud.labels == 1), GROUND, "ground")
    cat.apply(cloud, gidx)
    cat.save()

    vertex = PlyData.read(str(path))["vertex"]
    assert vertex["semantic"].dtype == np.dtype(class_type)
    assert vertex["semantic"].tolist() == [GROUND] * 10 + cls[10:].tolist()


def test_the_class_field_cannot_change_under_unsaved_class_edits(tmp_path):
    coords, tid, cls = _two_trees()
    path = tmp_path / "plot.las"
    _write_las(path, coords, tid, cls)
    cat = open_catalog(str(path), class_field="classification")
    cloud, gidx = cat.load([1], margin=0.0)
    ops.set_class(cloud, [0], GROUND, "ground")
    cat.apply(cloud, gidx)
    with pytest.raises(ValueError, match="Save the class edits"):
        cat.set_class_field("user_data")
    cat.save()
    cat.set_class_field("user_data")
    assert cat.class_field == "user_data"


def test_without_a_class_field_nothing_about_classes_is_loaded(tmp_path):
    coords, tid, cls = _two_trees()
    path = tmp_path / "plot.las"
    _write_las(path, coords, tid, cls)
    cat = open_catalog(str(path))
    cloud, _gidx = cat.load([1], margin=0.0)
    assert cat.classes is None and cloud.classes is None
    assert not cat.has_unsaved_edits()


# -- a downsampled session -----------------------------------------------------
def _dense_classed_plot(tmp_path):
    """Two 5 mm-spaced trees, each wood below 1.5 cm and leaf above."""
    def grid(origin):
        g = np.stack(np.meshgrid(
            np.arange(30), np.arange(30), np.arange(6), indexing="ij"
        ), axis=-1).reshape(-1, 3) * 0.005
        return g + np.asarray(origin)

    a, b = grid((0, 0, 0)), grid((5.0, 0, 0))
    coords = np.vstack([a, b])
    tid = np.repeat([1, 2], [len(a), len(b)])
    cls = np.where(coords[:, 2] < 0.015, WOOD, LEAF)
    path = tmp_path / "dense.las"
    _write_las(path, coords, tid, cls)
    return str(path), coords, tid, cls


def test_a_downsampled_class_edit_reaches_every_point_of_that_tree_and_class(tmp_path):
    import laspy

    path, coords, tid, cls = _dense_classed_plot(tmp_path)
    cat = open_catalog(path, class_field="classification",
                       density_prompt=lambda *a: a[2])
    assert cat.is_decimated

    # Tree 2's leaves become wood, on the thinned cloud.
    cloud, gidx = cat.load([2], margin=0.0)
    leaves = np.flatnonzero((cloud.labels == 2) & (cloud.classes == LEAF))
    ops.set_class(cloud, leaves, WOOD, "wood")
    cat.apply(cloud, gidx)
    cat.save()

    after = np.asarray(laspy.read(path).classification)
    expected = cls.copy()
    expected[(tid == 2) & (cls == LEAF)] = WOOD
    # Every full-resolution leaf of tree 2, and nothing of tree 1's.
    assert after.tolist() == expected.tolist()


def test_whole_class_moves_can_count_class_zero():
    codes = np.array([7, 7, 8])
    before = np.array([0, 0, 1])
    after = np.array([2, 2, 1])
    assert density.whole_class_moves(codes, before, after) == {}
    assert density.whole_class_moves(
        codes, before, after, trees_only=False
    ) == {7: 2}
