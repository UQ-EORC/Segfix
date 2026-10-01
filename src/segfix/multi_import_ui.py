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

"""Pick the per-tree files for an import, and say which tree each one is.

The work is :mod:`segfix.merge`'s; this is the window onto it. Every control
re-plans the whole set and redraws the table, so the IDs on screen are
always the IDs that would be written — the thing that is easy to get wrong
by hand, and expensive to discover after a gigabyte of merging.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from . import merge

#: What the file pickers offer, matching the formats segfix reads.
FILE_FILTER = "Point clouds (*.ply *.las *.laz)"

#: Columns of the file table.
FILE_COL, ID_COL, GROUND_COL = 0, 1, 2


class PerTreeImportDialog(QDialog):
    """Choose per-tree files and how their tree IDs are decided.

    On accept, :attr:`sources` is the planned
    :class:`segfix.merge.Source` list, ready for
    :func:`segfix.workspace.create_from_files`.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import per-tree files")
        self.resize(700, 520)
        self._paths: list[Path] = []
        self._ground: set[str] = set()
        #: path -> ID typed into the table, which beats pattern and mapping.
        self._overrides: dict[str, int] = {}
        self._mapping: dict[str, int] = {}
        self._mapping_path: Path | None = None
        self.sources: list[merge.Source] = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "One file per tree, plus the ground/unassigned cloud if you have "
            "one. They are merged into a single project cloud, with each "
            "file's tree ID written per point - the originals aren't touched."
        ))

        add_row = QHBoxLayout()
        self.add_files_btn = QPushButton("Add files…")
        self.add_files_btn.clicked.connect(self._add_files)
        add_row.addWidget(self.add_files_btn)
        self.add_folder_btn = QPushButton("Add folder…")
        self.add_folder_btn.clicked.connect(self._add_folder)
        add_row.addWidget(self.add_folder_btn)
        self.remove_btn = QPushButton("Remove selected")
        self.remove_btn.clicked.connect(self._remove_selected)
        add_row.addWidget(self.remove_btn)
        add_row.addStretch(1)
        layout.addLayout(add_row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["File", "Tree ID", "Ground"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(
            FILE_COL, QHeaderView.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            GROUND_COL, QHeaderView.ResizeToContents
        )
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table, stretch=1)

        pattern_row = QHBoxLayout()
        pattern_row.addWidget(QLabel("Tree ID from filename:"))
        self.pattern_edit = QLineEdit(merge.DEFAULT_PATTERN)
        self.pattern_edit.setToolTip(
            "A regular expression; the last match in each filename is the "
            "tree ID. The default takes the last run of digits, so "
            "plot12_tree_7.ply is tree 7."
        )
        self.pattern_edit.textChanged.connect(self._replan)
        pattern_row.addWidget(self.pattern_edit, stretch=1)
        layout.addLayout(pattern_row)

        map_row = QHBoxLayout()
        self.mapping_btn = QPushButton("Mapping CSV…")
        self.mapping_btn.setToolTip(
            "A path,tree_id CSV. Where it has a row for a file, it wins over "
            "the filename pattern."
        )
        self.mapping_btn.clicked.connect(self._choose_mapping)
        map_row.addWidget(self.mapping_btn)
        self.mapping_label = QLabel("No mapping file")
        self.mapping_label.setStyleSheet("color: gray;")
        map_row.addWidget(self.mapping_label, stretch=1)
        self.clear_mapping_btn = QPushButton("Clear")
        self.clear_mapping_btn.clicked.connect(self._clear_mapping)
        self.clear_mapping_btn.setEnabled(False)
        map_row.addWidget(self.clear_mapping_btn)
        layout.addLayout(map_row)

        self.problem_label = QLabel()
        self.problem_label.setWordWrap(True)
        self.problem_label.setStyleSheet("color: #c04040;")
        layout.addWidget(self.problem_label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.buttons.button(QDialogButtonBox.Ok).setText("Import")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._replan()

    # -- picking files ---------------------------------------------------
    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add per-tree files", str(Path.home()), FILE_FILTER
        )
        self._add(paths)

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Add every cloud in a folder", str(Path.home())
        )
        if not folder:
            return
        found = sorted(
            path for path in Path(folder).iterdir()
            if path.suffix.lower() in (".ply", ".las", ".laz")
        )
        if not found:
            self.problem_label.setText(f"No .ply, .las or .laz files in {folder}.")
            return
        self._add(found)

    def _add(self, paths) -> None:
        known = {str(p) for p in self._paths}
        for path in paths:
            path = Path(path)
            if str(path) not in known:  # adding a folder twice isn't a mistake
                self._paths.append(path)
                known.add(str(path))
        self._replan()

    def _remove_selected(self) -> None:
        rows = {index.row() for index in self.table.selectedIndexes()}
        if not rows:
            return
        for row in sorted(rows, reverse=True):
            gone = self._paths.pop(row)
            self._ground.discard(str(gone))
            self._overrides.pop(str(gone), None)
        self._replan()

    # -- deciding the IDs ------------------------------------------------
    def _choose_mapping(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Mapping CSV", str(Path.home()), "CSV (*.csv *.txt)"
        )
        if not path:
            return
        try:
            self._mapping = merge.read_mapping(path)
        except OSError as exc:
            self.problem_label.setText(f"Couldn't read {path}: {exc}")
            return
        self._mapping_path = Path(path)
        self.mapping_label.setText(
            f"{self._mapping_path.name} - {len(self._mapping) // 2} row(s)"
        )
        self.clear_mapping_btn.setEnabled(True)
        self._replan()

    def _clear_mapping(self) -> None:
        self._mapping, self._mapping_path = {}, None
        self.mapping_label.setText("No mapping file")
        self.clear_mapping_btn.setEnabled(False)
        self._replan()

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """A typed ID, or the Ground box: both override what was worked out
        for that one file, and leave every other file alone."""
        if self._filling:
            return
        path = str(self._paths[item.row()])
        if item.column() == GROUND_COL:
            if item.checkState() == Qt.Checked:
                self._ground.add(path)
            else:
                self._ground.discard(path)
        elif item.column() == ID_COL:
            text = item.text().strip()
            try:
                self._overrides[path] = int(text)
            except ValueError:
                self._overrides.pop(path, None)  # back to the pattern's answer
        self._replan()

    def _replan(self) -> None:
        self.sources = merge.plan_sources(
            self._paths,
            pattern=self.pattern_edit.text() or merge.DEFAULT_PATTERN,
            # Typed IDs are keyed by full path, so they beat a CSV row for
            # the same file's bare name as well as one for its path.
            mapping={**self._mapping, **self._overrides},
            ground=self._ground,
        )
        found = merge.problems(self.sources) if self._paths else []
        self.problem_label.setText(" ".join(found))
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(
            bool(self._paths) and not found
        )
        self._fill_table()

    _filling = False

    def _fill_table(self) -> None:
        self._filling = True
        try:
            self.table.setRowCount(len(self.sources))
            for row, source in enumerate(self.sources):
                name = QTableWidgetItem(source.path.name)
                name.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                name.setToolTip(str(source.path))
                self.table.setItem(row, FILE_COL, name)

                if source.is_ground:
                    text = "ground"
                elif source.tree_id is None:
                    text = "?"
                else:
                    text = str(source.tree_id)
                id_item = QTableWidgetItem(text)
                if source.is_ground:
                    id_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                else:
                    id_item.setToolTip("Type an ID here to set it by hand")
                self.table.setItem(row, ID_COL, id_item)

                ground = QTableWidgetItem()
                ground.setFlags(
                    Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
                )
                ground.setCheckState(
                    Qt.Checked if source.is_ground else Qt.Unchecked
                )
                ground.setToolTip(
                    "Bring this file in as unassigned points rather than a tree"
                )
                self.table.setItem(row, GROUND_COL, ground)
        finally:
            self._filling = False

    # -- result ----------------------------------------------------------
    def summary(self) -> str:
        """One line for the status bar after the import."""
        trees = sum(1 for s in self.sources if not s.is_ground)
        ground = len(self.sources) - trees
        extra = f" + {ground} ground file(s)" if ground else ""
        return f"{trees} tree file(s){extra}"


def choose_per_tree_files(parent=None) -> list[merge.Source] | None:
    """Run the dialog. Returns the sources to merge, or None if cancelled."""
    dialog = PerTreeImportDialog(parent)
    if dialog.exec() == QDialog.Accepted and dialog.sources:
        return dialog.sources
    return None
