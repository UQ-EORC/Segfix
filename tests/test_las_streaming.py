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

"""Saving and exporting a LAS without reading the whole plot into memory.

A project imported from a ``.laz`` mirrors every save back to a ``.laz``,
and a per-tree export copies trees out of the working ``.las``. Both used to
start with ``laspy.read`` of the entire file. The mirror is now streamed in
chunks and the export reads rows from the memory map the catalogue already
has; these tests make ``laspy.read`` an error while they run, and check the
files written are what the whole-file versions wrote.
"""

from contextlib import contextmanager

import numpy as np
import pytest

from segfix import export, workspace
from segfix.treecatalog import open_catalog
from tests.test_workspace import _laz_source


@contextmanager
def no_whole_file_reads():
    """``laspy.read`` is an error inside the block."""
    import laspy

    def refuse(*args, **kwargs):
        raise AssertionError("read the whole LAS into memory")

    real, laspy.read = laspy.read, refuse
    try:
        yield
    finally:
        laspy.read = real


def _assert_same_cloud(a, b, fields=("X", "Y", "Z", "treeID", "red")):
    assert a.header.point_count == b.header.point_count
    assert str(a.header.point_format) == str(b.header.point_format)
    assert list(a.header.scales) == list(b.header.scales)
    assert list(a.header.offsets) == list(b.header.offsets)
    assert list(a.header.mins) == list(b.header.mins)
    assert list(a.header.maxs) == list(b.header.maxs)
    for field in fields:
        assert np.array_equal(getattr(a, field), getattr(b, field)), field


# -- las -> laz in chunks -------------------------------------------------------
def test_compress_las_keeps_every_field_header_and_evlr(tmp_path, monkeypatch):
    import laspy

    monkeypatch.setattr(workspace, "_LAZ_CHUNK", 700)  # several chunks
    source = tmp_path / "plot.laz"
    _laz_source(source)
    las = workspace.create_workspace(source, tmp_path / "proj")
    original = laspy.read(str(las))
    out = tmp_path / "out.laz"

    with no_whole_file_reads():
        workspace.compress_las(las, out)

    with laspy.open(str(out)) as reader:
        assert reader.header.are_points_compressed
    copied = laspy.read(str(out))
    _assert_same_cloud(copied, original)
    assert len(copied.evlrs) == 1 and copied.evlrs[0].record_data == b"hi"
    assert out.stat().st_size < las.stat().st_size
    assert not (tmp_path / "out.laz.part").exists()


def test_compress_las_reports_progress_in_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "_LAZ_CHUNK", 1000)
    source = tmp_path / "plot.laz"
    _laz_source(source, n=5000)
    las = workspace.create_workspace(source, tmp_path / "proj")
    seen = []

    workspace.compress_las(las, tmp_path / "out.laz",
                           lambda m, f: seen.append((m, f)))

    assert [m for m, _ in seen] == ["Compressing"] * 5
    fractions = [f for _, f in seen]
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)


def test_an_interrupted_compress_leaves_the_old_laz(tmp_path, monkeypatch):
    """Written beside the target and moved into place: a save that dies
    part way must not leave half a .laz where a whole one was."""
    monkeypatch.setattr(workspace, "_LAZ_CHUNK", 1000)
    source = tmp_path / "plot.laz"
    _laz_source(source, n=5000)
    las = workspace.create_workspace(source, tmp_path / "proj")
    out = tmp_path / "out.laz"
    out.write_bytes(b"the previous export")

    def die_half_way(message, fraction):
        if fraction > 0.5:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        workspace.compress_las(las, out, die_half_way)

    assert out.read_bytes() == b"the previous export"
    assert not (tmp_path / "out.laz.part").exists()


# -- save in a project that came from a .laz ---------------------------------------
def test_saving_a_laz_project_streams_the_mirror(tmp_path, monkeypatch):
    import laspy

    monkeypatch.setattr(workspace, "_LAZ_CHUNK", 900)
    source = tmp_path / "plot.laz"
    _laz_source(source)
    seen = []

    with no_whole_file_reads():
        las = workspace.create_workspace(source, tmp_path / "proj")
        catalog = open_catalog(str(las))
        tree = next(iter(catalog.records))
        cloud, idx = catalog.load([tree], margin=0.0)
        cloud.set_labels(np.flatnonzero(cloud.labels == tree), 777, "renumber")
        catalog.apply(cloud, idx)
        catalog.save(progress=lambda m, f: seen.append((m, f)))

    mirror = tmp_path / "proj" / "plot.laz"
    saved, mirrored = laspy.read(str(las)), laspy.read(str(mirror))
    _assert_same_cloud(mirrored, saved)
    assert 777 in mirrored.treeID and tree not in mirrored.treeID
    # The bar keeps moving through the compression, and never goes back.
    fractions = [f for _, f in seen]
    assert fractions == sorted(fractions) and fractions[-1] == pytest.approx(1.0)
    assert sum("Compressing" in m for m, _ in seen) >= 2


# -- per-tree export from the memory map ---------------------------------------------
def test_las_export_reads_rows_not_the_whole_file(tmp_path):
    import laspy

    source = tmp_path / "plot.laz"
    _laz_source(source)
    las = workspace.create_workspace(source, tmp_path / "proj")
    original = laspy.read(str(las))
    catalog = open_catalog(str(las))
    out = tmp_path / "trees"

    with no_whole_file_reads():
        written = export.export_trees(catalog, str(out))

    tree_ids = np.asarray(original.treeID)
    expected = sorted(int(t) for t in np.unique(tree_ids) if t > 0)
    assert len(written) == len(expected)
    for label, path in zip(expected, written):
        tree = laspy.read(path)
        rows = np.flatnonzero(tree_ids == label)
        assert tree.header.point_count == rows.size
        for field in ("X", "Y", "Z", "treeID", "red", "intensity"):
            assert np.array_equal(
                getattr(tree, field), np.asarray(getattr(original, field))[rows]
            ), (label, field)
        # Extents are this tree's own, and the frame is the plot's.
        np.testing.assert_allclose(tree.header.mins, [
            np.asarray(original.x)[rows].min(), np.asarray(original.y)[rows].min(),
            np.asarray(original.z)[rows].min(),
        ])
        assert list(tree.header.scales) == list(original.header.scales)
        assert list(tree.header.offsets) == list(original.header.offsets)
        assert "treeID" in tree.point_format.extra_dimension_names
