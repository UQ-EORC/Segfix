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

"""A per-tree import, end to end: files in, a project that opens, trees out.

The round trip is what the feature promises — import ``tree_4.ply`` and
``tree_9.ply``, and Export Trees gives back files for trees 4 and 9, not
some renumbering invented from their colours. The dialog tests cover the
part that decides which file is which tree, since by the time the merge has
run, a wrong answer has already cost a copy of the data.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from segfix import merge, workspace  # noqa: E402
from segfix.model import UNASSIGNED  # noqa: E402

pytest.importorskip("qtpy")
from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtWidgets import QApplication, QDialogButtonBox  # noqa: E402

from segfix.multi_import_ui import (  # noqa: E402
    GROUND_COL,
    ID_COL,
    PerTreeImportDialog,
)


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def _write_ply(path, n, seed):
    rng = np.random.default_rng(seed)
    rows = np.empty(n, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]))
    coords = rng.random((n, 3)).astype(np.float32) * 5
    rows["x"], rows["y"], rows["z"] = coords.T
    with open(path, "wb") as fh:
        fh.write(
            b"ply\nformat binary_little_endian 1.0\n"
            + f"element vertex {n}\n".encode()
            + b"property float x\nproperty float y\nproperty float z\n"
            b"end_header\n"
        )
        fh.write(rows.tobytes())
    return coords


@pytest.fixture
def tree_files(tmp_path):
    """What raysplit leaves behind: a file per tree, plus the ground."""
    folder = tmp_path / "plot_a"
    folder.mkdir()
    counts = {}
    for name, n, seed in (("tree_4.ply", 40, 1), ("tree_9.ply", 25, 2),
                          ("ground.ply", 60, 3)):
        _write_ply(folder / name, n, seed)
        counts[name] = n
    return folder, counts


# -- the dialog decides which file is which tree -----------------------------
@pytest.fixture
def dialog(tree_files):
    folder, _ = tree_files
    dlg = PerTreeImportDialog()
    dlg._add(sorted(folder.iterdir()))
    yield dlg
    dlg.deleteLater()


def ids(dlg):
    return [s.tree_id for s in dlg.sources]


def test_the_table_shows_the_id_each_file_would_get(dialog):
    rows = [dialog.table.item(r, ID_COL).text()
            for r in range(dialog.table.rowCount())]
    assert rows == ["?", "4", "9"]  # ground.ply has no number in it


def test_import_is_refused_until_every_file_has_an_id(dialog):
    ok = dialog.buttons.button(QDialogButtonBox.Ok)
    assert not ok.isEnabled()
    assert "ground.ply" in dialog.problem_label.text()


def test_ticking_ground_settles_it(dialog):
    dialog.table.item(0, GROUND_COL).setCheckState(Qt.Checked)
    assert ids(dialog) == [UNASSIGNED, 4, 9]
    assert dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()
    assert dialog.problem_label.text() == ""


def test_an_id_can_be_typed_over_the_pattern(dialog):
    dialog.table.item(0, GROUND_COL).setCheckState(Qt.Checked)
    dialog.table.item(2, ID_COL).setText("77")
    assert ids(dialog) == [UNASSIGNED, 4, 77]


def test_editing_the_pattern_re_reads_every_name(dialog):
    dialog.pattern_edit.setText(r"tree_(\d+)")
    assert ids(dialog) == [None, 4, 9]
    dialog.pattern_edit.setText("(")  # mid-typing, and invalid
    assert ids(dialog) == [None, None, None]
    assert not dialog.buttons.button(QDialogButtonBox.Ok).isEnabled()


def test_a_mapping_csv_fills_in_what_the_names_cant(dialog, tmp_path):
    csv_file = tmp_path / "map.csv"
    csv_file.write_text("path,tree_id\nground.ply,0\ntree_4.ply,104\n")
    dialog._mapping = merge.read_mapping(csv_file)
    dialog._replan()
    assert ids(dialog) == [UNASSIGNED, 104, 9]


def test_the_same_file_twice_is_added_once(dialog, tree_files):
    folder, _ = tree_files
    before = dialog.table.rowCount()
    dialog._add([folder / "tree_4.ply"])
    assert dialog.table.rowCount() == before


def test_removing_a_row_forgets_what_was_set_for_it(dialog):
    dialog.table.item(0, GROUND_COL).setCheckState(Qt.Checked)
    dialog.table.selectRow(0)
    dialog._remove_selected()
    assert [s.path.name for s in dialog.sources] == ["tree_4.ply", "tree_9.ply"]
    assert dialog._ground == set()


# -- the import itself -------------------------------------------------------
def test_a_merged_project_opens_with_its_trees(tree_files, tmp_path):
    """The whole point: open the merged project and the trees are 4 and 9,
    with the ground unassigned beneath them."""
    from segfix.treecatalog import open_catalog

    folder, counts = tree_files
    sources = merge.plan_sources(
        sorted(folder.iterdir()), ground=[folder / "ground.ply"]
    )
    data = workspace.create_from_files(sources, tmp_path / "project")

    catalog = open_catalog(str(data))
    assert sorted(catalog.records) == [4, 9]
    assert catalog.records[4].count == counts["tree_4.ply"]
    assert catalog.records[9].count == counts["tree_9.ply"]
    assert int((catalog.file_labels() == UNASSIGNED).sum()) == counts["ground.ply"]


def test_the_manifest_records_where_every_point_came_from(tree_files, tmp_path):
    folder, _ = tree_files
    sources = merge.plan_sources(
        sorted(folder.iterdir()), ground=[folder / "ground.ply"]
    )
    workspace.create_from_files(sources, tmp_path / "project")
    manifest = json.loads(
        (tmp_path / "project" / workspace.MANIFEST_NAME).read_text()
    )
    assert [os.path.basename(s["path"]) for s in manifest["sources"]] == [
        "ground.ply", "tree_4.ply", "tree_9.ply"
    ]
    assert [s["tree_id"] for s in manifest["sources"]] == [UNASSIGNED, 4, 9]
    assert manifest["data_file"] == "project.ply"


def test_exporting_gives_the_per_tree_files_back(tree_files, tmp_path):
    from segfix.export import export_trees
    from segfix.treecatalog import open_catalog

    folder, counts = tree_files
    sources = merge.plan_sources(
        sorted(folder.iterdir()), ground=[folder / "ground.ply"]
    )
    data = workspace.create_from_files(sources, tmp_path / "project")
    written = export_trees(open_catalog(str(data)), str(tmp_path / "out"))
    assert [os.path.basename(p) for p in written] == [
        "project_tree_4.ply", "project_tree_9.ply"
    ]


def test_a_project_folder_in_use_is_refused(tree_files, tmp_path):
    folder, _ = tree_files
    taken = tmp_path / "project"
    taken.mkdir()
    (taken / "something").write_text("in the way")
    sources = merge.plan_sources(
        sorted(folder.iterdir()), ground=[folder / "ground.ply"]
    )
    with pytest.raises(FileExistsError):
        workspace.create_from_files(sources, taken)


def test_the_originals_are_left_alone(tree_files, tmp_path):
    folder, _ = tree_files
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    sources = merge.plan_sources(
        sorted(folder.iterdir()), ground=[folder / "ground.ply"]
    )
    workspace.create_from_files(sources, tmp_path / "project")
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
