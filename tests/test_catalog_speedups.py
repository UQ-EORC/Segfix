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

"""The catalogue's shortcuts give the answers the long way round gives.

Opening a tree used to cost three passes over every point in the file
(which trees are near, which unassigned points are near, and a re-sort of
every label afterwards), and opening the file two sorts it didn't need.
Each of those now goes through the index or a lookup table instead. None of
them is allowed to change an answer, so every test here runs the shortcut
against the plain version it replaced.
"""

import numpy as np
import pytest

from segfix import analysis, density, io
from segfix.model import NOISE, UNASSIGNED
from segfix.treecatalog import open_catalog
from tests.test_core import _write_arbor_las
from tests.test_rgb import _write_raycloud_ply


def _plot(seed: int = 0, trees: int = 12, per_tree: int = 400):
    """A plot of ``trees`` blobs on a grid, close enough that neighbours
    touch, over a sheet of unassigned ground. ``(xyz, tree id per point)``."""
    rng = np.random.default_rng(seed)
    xyz, tid = [], []
    for k in range(trees):
        centre = np.array([(k % 4) * 2.0, (k // 4) * 2.0, 3.0])
        xyz.append(centre + rng.normal(0, [0.6, 0.6, 1.5], (per_tree, 3)))
        tid.append(np.full(per_tree, k + 1))
    ground = np.column_stack([
        rng.uniform(-2, 8, 1500), rng.uniform(-2, 6, 1500),
        rng.normal(0, 0.05, 1500),
    ])
    xyz.append(ground)
    tid.append(np.zeros(len(ground), dtype=int))
    order = rng.permutation(sum(len(x) for x in xyz))  # not grouped by tree
    return np.vstack(xyz)[order], np.concatenate(tid)[order]


def _colour_of(tree_ids: np.ndarray) -> np.ndarray:
    """A distinct colour per tree id, black for unassigned."""
    rgb = np.zeros((len(tree_ids), 3), dtype=np.uint8)
    rgb[:, 0] = (tree_ids * 37) % 251
    rgb[:, 1] = (tree_ids * 91) % 241
    rgb[:, 2] = tree_ids
    rgb[tree_ids == 0] = 0
    return rgb


@pytest.fixture(params=["ply", "las"])
def plot_path(tmp_path, request):
    xyz, tid = _plot()
    if request.param == "ply":
        path = tmp_path / "plot.ply"
        _write_raycloud_ply(path, xyz, _colour_of(tid))
    else:
        path = tmp_path / "plot.las"
        _write_arbor_las(path, xyz, tid)
    return str(path)


def _rebuilt(catalog):
    """What a full re-sort of ``catalog``'s labels says the index should be:
    ``({label: indices}, {label: (count, bbox lo, bbox hi, centroid)})``."""
    labels = catalog.labels
    indices = {
        int(lab): np.flatnonzero(labels == lab) for lab in np.unique(labels)
    }
    records = {}
    for lab, idx in indices.items():
        if lab in (UNASSIGNED, NOISE):
            continue
        pts = catalog.coords[idx]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        records[lab] = (len(idx), lo, hi,
                        (float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)))
    return indices, records


def _assert_index_is_right(catalog):
    indices, records = _rebuilt(catalog)
    for lab, idx in indices.items():
        np.testing.assert_array_equal(catalog.indices_for(lab), idx)
    assert list(catalog.records) == sorted(records)  # same trees, same order
    for lab, (count, lo, hi, centroid) in records.items():
        rec = catalog.records[lab]
        assert (rec.label, rec.count) == (lab, count)
        np.testing.assert_array_equal(rec.bbox[0], lo)
        np.testing.assert_array_equal(rec.bbox[1], hi)
        assert rec.centroid == centroid
    # A tree that no longer exists has no points and no record.
    gone = set(catalog._starts) - set(indices)
    for lab in gone:
        assert catalog.indices_for(lab).size == 0
        assert lab not in catalog.records


# -- apply(): the index after an edit ---------------------------------------
def test_switching_away_from_an_unedited_tree_does_not_reindex(plot_path, monkeypatch):
    """Looking at a tree and moving on changes no label, so there is nothing
    to index: the 0.7s re-sort a 39M-point plot paid per switch is skipped."""
    catalog = open_catalog(plot_path)
    first = next(iter(catalog.records))
    cloud, idx = catalog.load([first, *sorted(catalog.neighbours(first, 1.0))])

    def boom():
        raise AssertionError("re-indexed the whole cloud for no edit")

    monkeypatch.setattr(catalog, "_build_index", boom)
    monkeypatch.setattr(catalog, "_update_index", lambda *a: boom())
    catalog.apply(cloud, idx)
    _assert_index_is_right(catalog)


def test_an_edit_updates_only_the_trees_it_touched(plot_path, monkeypatch):
    """Moving points between two trees must not re-sort the cloud either,
    and must leave the index as a full rebuild would."""
    catalog = open_catalog(plot_path)
    a = next(iter(catalog.records))
    b = sorted(catalog.neighbours(a, 1.0))[0]
    cloud, idx = catalog.load([a, b])
    monkeypatch.setattr(
        catalog, "_build_index",
        lambda: (_ for _ in ()).throw(AssertionError("full rebuild")),
    )
    untouched = {
        lab: catalog.indices_for(lab) for lab in catalog.records if lab not in (a, b)
    }

    half = np.flatnonzero(cloud.labels == a)[::2]
    cloud.set_labels(half, b, "move half of a into b")
    catalog.apply(cloud, idx)

    _assert_index_is_right(catalog)
    for lab, before in untouched.items():
        assert catalog.indices_for(lab) is not None
        np.testing.assert_array_equal(catalog.indices_for(lab), before)


def test_the_index_survives_every_kind_of_edit(plot_path):
    """New trees, whole merges, unassigning, noise, points adopted from the
    ground, undo across a flush: after each, the incremental index equals a
    rebuild from scratch."""
    catalog = open_catalog(plot_path)
    rng = np.random.default_rng(3)
    for step in range(30):
        labels = sorted(catalog.records)
        focus = labels[int(rng.integers(len(labels)))]
        scene = [focus, *sorted(catalog.neighbours(focus, 1.0))]
        cloud, idx = catalog.load(scene)
        mine = np.flatnonzero(cloud.labels == focus)
        ground = np.flatnonzero(cloud.labels == UNASSIGNED)
        kind = step % 6
        if kind == 0:    # split part off as a new tree
            cloud.set_labels(mine[: max(1, mine.size // 3)], cloud.next_free_id(), "split")
        elif kind == 1 and len(scene) > 1:  # merge the whole tree away
            cloud.set_labels(mine, scene[1], "merge")
        elif kind == 2:  # unassign some
            cloud.set_labels(mine[::3], UNASSIGNED, "unassign")
        elif kind == 3:  # dismiss some as noise
            cloud.set_labels(mine[::4], NOISE, "noise")
        elif kind == 4 and ground.size:  # adopt ground points
            cloud.set_labels(ground[::2], focus, "add")
        else:            # edit, flush, undo, flush: the tree comes back
            cloud.set_labels(mine, UNASSIGNED, "unassign whole tree")
            catalog.apply(cloud, idx)
            _assert_index_is_right(catalog)
            cloud.undo()
        catalog.apply(cloud, idx)
        _assert_index_is_right(catalog)
    assert catalog._moved, "nothing went through the incremental path"


def test_new_tree_ids_stay_unique_after_incremental_updates(plot_path):
    catalog = open_catalog(plot_path)
    seen = set(catalog.records)
    for _ in range(3):
        focus = sorted(catalog.records)[0]
        cloud, idx = catalog.load([focus])
        new_id = cloud.next_free_id()
        assert new_id not in seen
        seen.add(new_id)
        cloud.set_labels(np.flatnonzero(cloud.labels == focus)[:5], new_id, "split")
        catalog.apply(cloud, idx)
    assert catalog.next_free_id() > max(seen)


def test_applying_a_subset_that_holds_only_part_of_a_tree(plot_path):
    """load() always brings trees whole, but apply() takes any rows: a tree
    with points outside them is found by asking the labels."""
    catalog = open_catalog(plot_path)
    a, b = sorted(catalog.records)[:2]
    idx = np.sort(np.concatenate([
        catalog.indices_for(a)[:50], catalog.indices_for(b)[:50]
    ]))

    class Subset:
        labels = catalog.labels[idx].copy()
        classes = None

    Subset.labels[Subset.labels == a] = b
    catalog.apply(Subset, idx)
    _assert_index_is_right(catalog)


def test_the_index_is_compacted_once_enough_of_it_has_moved(plot_path, monkeypatch):
    """The per-tree copies are bounded: past a quarter of the cloud the index
    is rebuilt and they are dropped."""
    catalog = open_catalog(plot_path)
    rebuilds = []
    full = catalog._build_index
    monkeypatch.setattr(
        catalog, "_build_index", lambda: (rebuilds.append(1), full())[1]
    )
    monkeypatch.setattr(
        "segfix.treecatalog._COMPACT_FLOOR", 0, raising=True
    )
    for focus in sorted(catalog.records):
        if focus not in catalog.records:
            continue
        cloud, idx = catalog.load([focus])
        cloud.set_labels(np.flatnonzero(cloud.labels == focus)[:3], UNASSIGNED, "trim")
        catalog.apply(cloud, idx)
        _assert_index_is_right(catalog)
    assert rebuilds, "never compacted"
    assert catalog._moved_points <= catalog.labels.size // 4


# -- neighbours() and load() -------------------------------------------------
def test_neighbours_through_the_index_match_a_scan_of_the_whole_cloud(plot_path):
    """Same trees *and* same distances: the samples are drawn in the same
    order, so nothing about the review's neighbour buttons can shift."""
    catalog = open_catalog(plot_path)
    whole = io.load(plot_path)
    for label in catalog.records:
        for reach in (0.2, 1.0, 3.0):
            assert catalog.neighbour_distances(label, reach) == \
                analysis.neighbour_distances(whole, label, reach)
            assert catalog.neighbours(label, reach) == \
                analysis.neighbours_by_points(whole, label, reach)


def test_neighbours_stay_right_after_edits(plot_path):
    from segfix.model import PointCloud

    catalog = open_catalog(plot_path)
    a = sorted(catalog.records)[0]
    b = sorted(catalog.neighbours(a, 1.0))[0]
    cloud, idx = catalog.load([a, b])
    cloud.set_labels(np.flatnonzero(cloud.labels == a)[::2], cloud.next_free_id(), "split")
    cloud.set_labels(np.flatnonzero(cloud.labels == b), UNASSIGNED, "drop b")
    catalog.apply(cloud, idx)

    whole = PointCloud(coords=catalog.coords, labels=catalog.labels)
    for label in catalog.records:
        assert catalog.neighbour_distances(label, 1.0) == \
            analysis.neighbour_distances(whole, label, 1.0)


def test_neighbours_of_a_tree_that_is_not_there(plot_path):
    catalog = open_catalog(plot_path)
    assert catalog.neighbours(987654, 1.0) == set()


def test_load_picks_the_trees_and_the_unassigned_points_beside_them(plot_path):
    catalog = open_catalog(plot_path)
    for label in list(catalog.records)[:5]:
        scene = [label, *sorted(catalog.neighbours(label, 1.0))]
        cloud, idx = catalog.load(scene, margin=1.0)

        in_scene = np.isin(catalog.labels, scene)
        pts = catalog.coords[in_scene]
        lo, hi = pts.min(axis=0) - 1.0, pts.max(axis=0) + 1.0
        near = np.all((catalog.coords >= lo) & (catalog.coords <= hi), axis=1)
        expected = np.flatnonzero(in_scene | (near & (catalog.labels == UNASSIGNED)))
        np.testing.assert_array_equal(idx, expected)
        np.testing.assert_array_equal(cloud.labels, catalog.labels[expected])


# -- in_box on threads --------------------------------------------------------
def test_in_box_gives_the_same_mask_on_threads(monkeypatch):
    rng = np.random.default_rng(0)
    coords = rng.random((50_000, 3)).astype(np.float32)
    lo, hi = np.array([0.2, 0.3, 0.1]), np.array([0.7, 0.9, 0.5])
    expected = np.all((coords >= lo) & (coords <= hi), axis=1)

    np.testing.assert_array_equal(density.in_box(coords, lo, hi), expected)
    monkeypatch.setattr(density, "_PARALLEL_ABOVE", 1000)
    monkeypatch.setattr(density, "_PARALLEL_CHUNK", (7001, 7001))  # a ragged last chunk
    monkeypatch.setattr(density, "_workers", lambda: 4)
    np.testing.assert_array_equal(density.in_box(coords, lo, hi), expected)
    np.testing.assert_array_equal(density.in_box(coords[:0], lo, hi), expected[:0])


def test_in_box_raises_what_a_worker_raised(monkeypatch):
    monkeypatch.setattr(density, "_PARALLEL_ABOVE", 10)
    monkeypatch.setattr(density, "_workers", lambda: 2)
    with pytest.raises(IndexError):
        density.in_box(np.zeros((100, 2), np.float32), np.zeros(3), np.ones(3))


# -- colours to labels ---------------------------------------------------------
@pytest.mark.parametrize("with_black", [True, False])
def test_the_colour_table_numbers_trees_as_the_sort_did(monkeypatch, with_black):
    rng = np.random.default_rng(1)
    palette = rng.integers(0, 256, (300, 3)).astype(np.uint8)
    palette[0] = (0, 0, 0) if with_black else (0, 0, 1)
    rgb = palette[rng.integers(0, len(palette), 20_000)]
    red, green, blue = (np.ascontiguousarray(rgb[:, k]) for k in range(3))

    by_sort = io.labels_from_rgb(red, green, blue)
    monkeypatch.setattr(io, "_RGB_TABLE_ABOVE", 0)
    monkeypatch.setattr(io, "_RGB_CHUNK", 3000)  # several chunks
    by_table = io.labels_from_rgb(red, green, blue)

    np.testing.assert_array_equal(by_table[0], by_sort[0])
    assert by_table[0].dtype == by_sort[0].dtype == np.int32
    assert by_table[1] == by_sort[1]
    # ...and both are what np.unique says, spelled out.
    packed = (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2]
    uniq, inverse = np.unique(packed, return_inverse=True)
    expected = (inverse + 1).astype(np.int32)
    expected[packed == 0] = UNASSIGNED
    np.testing.assert_array_equal(by_table[0], expected)
    assert (UNASSIGNED in by_table[0]) == with_black
    assert all(colour != (0, 0, 0) for colour in by_table[1].values())


def test_an_rgb_catalog_opens_the_same_through_the_table(tmp_path, monkeypatch):
    xyz, tid = _plot()
    path = tmp_path / "plot.ply"
    _write_raycloud_ply(path, xyz, _colour_of(tid))
    plain = open_catalog(str(path))
    monkeypatch.setattr(io, "_RGB_TABLE_ABOVE", 0)
    table = open_catalog(str(path))

    np.testing.assert_array_equal(table.labels, plain.labels)
    assert table.label_colors == plain.label_colors
    np.testing.assert_array_equal(table.file_labels(), plain.file_labels())


# -- coordinates ---------------------------------------------------------------
@pytest.mark.parametrize("shift", [None, (-5.0, 12.5, 100.0)])
def test_coordinates_read_by_axis_match_the_stacked_decode(plot_path, monkeypatch, shift):
    monkeypatch.setattr("segfix.treecatalog._CHUNK", 1000)  # several chunks
    catalog = open_catalog(plot_path)
    raw = catalog._decode_coords_raw(catalog._mm)  # float64, whole file
    assert raw.dtype == np.float64

    shift = None if shift is None else np.asarray(shift)
    coords, (mins, maxs) = catalog._read_coords(shift)
    expected = raw if shift is None else raw + shift
    np.testing.assert_array_equal(coords, expected.astype(np.float32))
    assert coords.dtype == np.float32
    # Bounds are the file's own, measured before the shift and the cast.
    np.testing.assert_array_equal(mins, raw.min(axis=0))
    np.testing.assert_array_equal(maxs, raw.max(axis=0))


def test_reading_coordinates_leaves_the_file_alone(plot_path):
    """The shift is added to a decoded column in place; that column must be
    a copy, never a view of the memory-mapped records."""
    catalog = open_catalog(plot_path)
    before = catalog._decode_coords_raw(catalog._mm).copy()
    catalog._read_coords(np.array([1000.0, 2000.0, 3000.0]))
    np.testing.assert_array_equal(catalog._decode_coords_raw(catalog._mm), before)
    for axis in range(3):
        column = catalog._decode_axis_raw(catalog._mm[:10], axis)
        assert column.flags.c_contiguous and column.flags.owndata


def test_a_float32_ply_reads_its_coordinates_exactly(tmp_path):
    xyz, tid = _plot(per_tree=50)
    path = tmp_path / "f32.ply"
    rows = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("treeID", "<i4")])
    rows["x"], rows["y"], rows["z"] = xyz.T
    rows["treeID"] = tid
    with open(path, "wb") as fh:
        fh.write(
            b"ply\nformat binary_little_endian 1.0\n"
            + f"element vertex {len(xyz)}\n".encode()
            + b"property float x\nproperty float y\nproperty float z\n"
              b"property int treeID\nend_header\n"
        )
        fh.write(rows.tobytes())
    catalog = open_catalog(str(path))
    np.testing.assert_array_equal(catalog.coords, xyz.astype(np.float32))


# -- counting labels -------------------------------------------------------------
@pytest.mark.parametrize("labels", [
    np.random.default_rng(0).integers(-1, 40, 5000).astype(np.int32),
    np.array([], dtype=np.int32),
    np.zeros(7, dtype=np.int32),
    np.array([5, 2_000_000_000, -1], dtype=np.int32),   # too spread for a table
    np.array([3, 3, 9, 9, 9], dtype=np.int64),
])
def test_label_counts_is_np_unique(labels):
    values, counts = analysis.label_counts(labels)
    expected_values, expected_counts = np.unique(labels, return_counts=True)
    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(counts, expected_counts)
    assert values.dtype == labels.dtype
