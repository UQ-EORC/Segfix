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

"""Importing a folder of per-tree clouds as one project.

The IDs are the point of it: a per-tree file has no ID field, and segfix's
fallback numbers trees by their RGB colour, which has nothing to do with the
numbering the segmentation used. These pin that a merged set comes back out
of the file under the IDs its *filenames* carried, that the ground file
lands as unassigned, and that the points and their other fields survive the
trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from segfix import io, merge
from segfix.model import UNASSIGNED


# -- filenames -> IDs --------------------------------------------------------
@pytest.mark.parametrize("name, expected", [
    ("tree_7.ply", 7),
    ("tree_007.las", 7),
    ("plot12_tree_7.ply", 7),       # the last number, not the first
    ("cloud.ply", None),
    ("/data/plot/segmented_tree_42.laz", 42),
])
def test_the_default_pattern_reads_the_last_number(name, expected):
    assert merge.tree_id_from_name(name) == expected


def test_a_custom_pattern_can_pick_a_different_number():
    assert merge.tree_id_from_name("t12_s3.ply", r"t(\d+)") == 12


def test_a_broken_pattern_is_no_id_rather_than_a_crash():
    """The pattern is typed into a dialog, so it is invalid half the time
    it's being edited."""
    assert merge.tree_id_from_name("tree_7.ply", "tree_(") is None


def test_a_mapping_csv_beats_the_filename(tmp_path):
    csv_file = tmp_path / "map.csv"
    csv_file.write_text("path,tree_id\ntree_1.ply,500\n/elsewhere/tree_2.ply,501\n")
    mapping = merge.read_mapping(csv_file)
    sources = merge.plan_sources(
        [tmp_path / "tree_1.ply", tmp_path / "tree_2.ply", tmp_path / "tree_3.ply"],
        mapping=mapping,
    )
    # Matched by name, by full path as written, and falling back to the name.
    assert [s.tree_id for s in sources] == [500, 501, 3]


def test_a_mapping_csv_without_a_header_still_reads(tmp_path):
    csv_file = tmp_path / "map.csv"
    csv_file.write_text("# tiles\n\ntree_1.ply,9\n")
    assert merge.read_mapping(csv_file)["tree_1.ply"] == 9


def test_a_ground_file_comes_in_unassigned(tmp_path):
    ground = tmp_path / "ground.ply"
    sources = merge.plan_sources(
        [tmp_path / "tree_1.ply", ground], ground=[ground]
    )
    assert sources[1].tree_id == UNASSIGNED
    assert sources[1].is_ground


# -- what stops a merge ------------------------------------------------------
def test_files_with_no_id_are_refused_by_name(tmp_path):
    sources = merge.plan_sources([tmp_path / "trunk.ply", tmp_path / "tree_1.ply"])
    assert "trunk.ply" in merge.problems(sources)[0]


def test_two_files_cant_be_the_same_tree(tmp_path):
    sources = merge.plan_sources(
        [tmp_path / "a_tree_3.ply", tmp_path / "b_tree_3.ply"]
    )
    assert "same tree" in " ".join(merge.problems(sources))


def test_two_ground_files_are_fine(ply_set):
    """Unlike trees: a tile's ground often arrives in pieces."""
    paths, _ = ply_set
    sources = merge.plan_sources(paths[:2], ground=paths[:2])
    assert merge.problems(sources) == []


# -- PLY ---------------------------------------------------------------------
def _write_ply(path, coords, extra=None):
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if extra is not None:
        dtype.append(("intensity", "<u2"))
    rows = np.empty(len(coords), dtype=np.dtype(dtype))
    rows["x"], rows["y"], rows["z"] = coords.T
    if extra is not None:
        rows["intensity"] = extra
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(coords)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        + ("property ushort intensity\n" if extra is not None else "")
        + "end_header\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(rows.tobytes())


@pytest.fixture
def ply_set(tmp_path):
    """Two tree files and a ground file, with known coordinates."""
    rng = np.random.default_rng(0)
    paths, blocks = [], {}
    for name, tree, n in (("tree_4.ply", 4, 30), ("tree_9.ply", 9, 20),
                          ("ground.ply", UNASSIGNED, 50)):
        coords = rng.random((n, 3)).astype(np.float32) * 10 + tree
        path = tmp_path / name
        _write_ply(path, coords, extra=np.arange(n, dtype=np.uint16))
        paths.append(path)
        blocks[tree] = coords
    return paths, blocks


def test_a_merged_ply_carries_the_filename_ids(ply_set, tmp_path):
    paths, blocks = ply_set
    sources = merge.plan_sources(paths, ground=[paths[2]])
    merge.merge(sources, tmp_path / "merged.ply")

    cloud = io.load(str(tmp_path / "merged.ply"))
    assert cloud.label_field == "treeID"
    assert sorted(set(cloud.labels.tolist())) == [UNASSIGNED, 4, 9]
    for tree, coords in blocks.items():
        got = cloud.coords[cloud.labels == tree]
        assert len(got) == len(coords)
        assert np.allclose(np.sort(got, axis=0), np.sort(coords, axis=0))


def test_a_merged_ply_keeps_the_other_fields(ply_set, tmp_path):
    paths, _ = ply_set
    sources = merge.plan_sources(paths, ground=[paths[2]])
    merge.merge(sources, tmp_path / "merged.ply")
    cloud = io.load(str(tmp_path / "merged.ply"))
    assert "intensity" in cloud.attributes
    assert cloud.attributes["intensity"][cloud.labels == 4].max() == 29


def test_a_field_only_some_files_have_is_dropped(tmp_path):
    """Rather than invented for the rest."""
    rng = np.random.default_rng(1)
    with_extra = tmp_path / "tree_1.ply"
    without = tmp_path / "tree_2.ply"
    _write_ply(with_extra, rng.random((5, 3)).astype(np.float32),
               extra=np.arange(5, dtype=np.uint16))
    _write_ply(without, rng.random((5, 3)).astype(np.float32))
    merge.merge(merge.plan_sources([with_extra, without]), tmp_path / "m.ply")
    cloud = io.load(str(tmp_path / "m.ply"))
    assert "intensity" not in cloud.attributes
    assert cloud.n_points == 10


def test_an_existing_id_column_doesnt_fight_the_filename(tmp_path):
    """A per-tree file cut out of a segmented plot can still carry the old
    column; the filename is the authority, so the stale one is dropped."""
    rows = np.empty(4, dtype=np.dtype([("x", "<f4"), ("y", "<f4"),
                                       ("z", "<f4"), ("treeID", "<i4")]))
    rows["x"] = rows["y"] = rows["z"] = np.arange(4)
    rows["treeID"] = 999
    path = tmp_path / "tree_3.ply"
    with open(path, "wb") as fh:
        fh.write(b"ply\nformat binary_little_endian 1.0\nelement vertex 4\n"
                 b"property float x\nproperty float y\nproperty float z\n"
                 b"property int treeID\nend_header\n")
        fh.write(rows.tobytes())
    merge.merge(merge.plan_sources([path]), tmp_path / "m.ply")
    cloud = io.load(str(tmp_path / "m.ply"))
    assert set(cloud.labels.tolist()) == {3}


# -- LAS ---------------------------------------------------------------------
@pytest.fixture
def las_set(tmp_path):
    laspy = pytest.importorskip("laspy")
    rng = np.random.default_rng(2)
    paths, blocks = [], {}
    for name, tree, n, base in (("tree_4.las", 4, 30, 0.0),
                                ("tree_9.las", 9, 20, 600.0),
                                ("ground.las", UNASSIGNED, 40, 0.0)):
        header = laspy.LasHeader(version="1.4", point_format=6)
        header.scales = [0.001, 0.001, 0.001]
        header.offsets = [base, base, 0.0]
        las = laspy.LasData(header)
        coords = rng.random((n, 3)) * 10 + base
        las.x, las.y, las.z = coords.T
        las.intensity = np.arange(n, dtype=np.uint16)
        path = tmp_path / name
        las.write(str(path))
        paths.append(path)
        blocks[tree] = coords
    return paths, blocks


def test_a_merged_las_carries_the_filename_ids(las_set, tmp_path):
    laspy = pytest.importorskip("laspy")
    paths, blocks = las_set
    sources = merge.plan_sources(paths, ground=[paths[2]])
    merge.merge(sources, tmp_path / "merged.las")

    las = laspy.read(str(tmp_path / "merged.las"))
    assert "treeID" in las.point_format.dimension_names
    labels = np.asarray(las.treeID)
    assert sorted(set(labels.tolist())) == [UNASSIGNED, 4, 9]
    for tree, coords in blocks.items():
        got = np.asarray(las.xyz)[labels == tree]
        assert len(got) == len(coords)
        assert np.allclose(np.sort(got, axis=0), np.sort(coords, axis=0), atol=1e-3)


def test_a_merged_las_spanning_the_plot_keeps_its_precision(las_set, tmp_path):
    """The offsets have to cover every input, not just the first: tree 9 is
    600 m away from tree 4, which is where a first-file offset would start
    losing millimetres."""
    laspy = pytest.importorskip("laspy")
    paths, blocks = las_set
    merge.merge(merge.plan_sources(paths, ground=[paths[2]]),
                tmp_path / "merged.las")
    las = laspy.read(str(tmp_path / "merged.las"))
    got = np.asarray(las.xyz)[np.asarray(las.treeID) == 9]
    assert np.allclose(np.sort(got, axis=0), np.sort(blocks[9], axis=0), atol=1e-3)


def test_a_merged_las_keeps_the_other_dimensions(las_set, tmp_path):
    laspy = pytest.importorskip("laspy")
    paths, _ = las_set
    merge.merge(merge.plan_sources(paths, ground=[paths[2]]),
                tmp_path / "merged.las")
    las = laspy.read(str(tmp_path / "merged.las"))
    assert np.asarray(las.intensity)[np.asarray(las.treeID) == 4].max() == 29


def test_mixing_formats_is_refused(ply_set, las_set, tmp_path):
    sources = merge.plan_sources([ply_set[0][0], las_set[0][0]])
    assert "same format" in " ".join(merge.problems(sources))
    with pytest.raises(ValueError):
        merge.merge(sources, tmp_path / "merged.ply")


def test_the_suffix_follows_the_inputs(ply_set, las_set):
    assert merge.output_suffix(merge.plan_sources(ply_set[0])) == ".ply"
    assert merge.output_suffix(merge.plan_sources(las_set[0])) == ".las"


# -- progress ----------------------------------------------------------------
def test_progress_is_reported_by_points_not_by_file(ply_set, tmp_path):
    paths, _ = ply_set
    seen = []
    merge.merge(merge.plan_sources(paths, ground=[paths[2]]),
                tmp_path / "merged.ply", report=lambda m, f: seen.append(f))
    assert seen == sorted(seen) and seen[-1] == 1.0
    # 30 points of 100 in the first file: a per-file bar would say 0.33 after
    # the second, not the fraction of the data that is actually through.
    assert 0 <= seen[0] < 0.5
