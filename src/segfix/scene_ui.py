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

"""The "All Trees" table for the default single-file mode: every tree in the
file (from a :mod:`~segfix.treecatalog` catalog); double-click loads that
tree plus its spatial neighbours into the shared segfix editing panel.

Docked at the top of the right-hand panel, directly above that editing panel
— see :func:`segfix.app._combined_panel`, which stacks the two into one dock.
"""

from __future__ import annotations

import json
import os

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QBrush, QColor, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .treecatalog import Catalog
from .viewer import busy, colors_for_labels

DEFAULT_REACH = 1.0  # metres; matches SegFixWidget's own "reach" spinner default


def update_done(cloud_path: str, tree_id: int, done: bool) -> str | None:
    """Mark one tree done (or not) in the sidecar, keeping whatever else it
    holds. Returns the sidecar path, or ``None`` if it couldn't be written.

    For the All Trees table's context menu, where the tree need not be the
    one loaded: the editing panel owns the list once a tree is loaded, and
    writes it the same way (see ``SegFixWidget._save_progress``).
    """
    path = f"{cloud_path}.segfix.json"
    saved = {}
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
        done_ids = {int(t) for t in saved.get("done", [])}
        (done_ids.add if done else done_ids.discard)(int(tree_id))
        saved["done"] = sorted(done_ids)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(saved, f)
    except (OSError, ValueError):
        return None
    return path


def read_done(cloud_path: str) -> set[int]:
    """The trees marked Done for the cloud at ``cloud_path``, from the
    ``<cloud>.segfix.json`` progress sidecar beside it; empty if there is no
    sidecar yet or it can't be read."""
    path = f"{cloud_path}.segfix.json"
    if not os.path.exists(path):
        return set()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {int(t) for t in data.get("done", [])}
    except (OSError, ValueError):
        return set()


class SceneController:
    """Owns the catalog and the shared segfix editing controller."""

    def __init__(self, view, catalog: Catalog, seg_controller,
                 point_size: float = 0.01, reach: float = DEFAULT_REACH):
        self.view = view
        self.catalog = catalog
        self.seg = seg_controller
        self.point_size = point_size
        self.reach = reach
        self.current_label: int | None = None
        self._global_idx: np.ndarray | None = None
        # Set by the widget to refresh its table's point counts after a save
        # — fires regardless of which Save button triggered it (this one, or
        # the shared segfix panel's), since both route through on_save_override.
        self.on_saved = None
        self.seg.on_save_override = self._save

    def load_tree(self, label: int) -> str:
        self._flush()  # persist edits on the outgoing scene first
        neighbours = self.catalog.neighbours(label, self.reach)
        cloud, global_idx = self.catalog.load(
            [label, *sorted(neighbours)], margin=self.reach
        )
        self.current_label = label
        self._global_idx = global_idx

        self.view.load_cloud(cloud, point_size=self.point_size)
        self.seg.set_cloud(cloud, focus=label)
        self.view.reset_view()
        return (
            f"Loaded tree {label} + {len(neighbours)} neighbour(s); "
            f"{cloud.n_points:,} points"
        )

    def _flush(self) -> None:
        if self._global_idx is not None:
            self.catalog.apply(self.seg.cloud, self._global_idx)

    def reload_current(self) -> str | None:
        """Load the tree under review afresh — after the class field changed,
        so the points on screen carry the new field. The caller flushes the
        live scene first; its arrays belong to the old field and must not be
        applied over the new one, so they are dropped here, not flushed."""
        if self.current_label is None:
            return None
        self._global_idx = None
        return self.load_tree(self.current_label)

    def has_unsaved_edits(self) -> bool:
        """Whether anything edited this session hasn't been saved — on the
        tree loaded now *or* any tree visited before it.

        The loaded cloud's undo stack can't answer that: it starts empty on
        every tree switch, while the edits made on the previous trees sit in
        the catalog. So the live scene is folded in first, as a switch or a
        save would, and the catalog is asked.
        """
        self._flush()
        return self.catalog.has_unsaved_edits()

    def _save(self) -> str:
        from .progress_ui import run_with_progress

        self._flush()  # capture the live scene too, not just prior ones
        # A decimated session interpolates its edits back onto every
        # full-resolution point here, and a Save As copies the whole file:
        # both scale with the cloud, so a save runs on the worker thread
        # behind the same progress window an open does, rather than blocking
        # the GUI thread into a "(Not Responding)" title.
        msg = run_with_progress(
            self.view.native.window(),
            "Saving",
            os.path.basename(self.catalog.path),
            lambda report, ask: self.catalog.save(progress=report),
        )
        if self.on_saved is not None:
            self.on_saved()
        return msg

class SceneWidget(QWidget):
    """Tree table for the whole file; stacked above the editing panel."""

    COLUMNS = ["Done", "Tree ID", "Points"]

    def __init__(self, controller: SceneController):
        super().__init__()
        self.c = controller
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # "All Trees": every tree in the source file. Double-click a row to
        # load it plus its spatial neighbours into the "Selected Tree +
        # Neighbours" panel underneath for editing.
        self.trees_box = QGroupBox("All Trees")
        blay = QVBoxLayout(self.trees_box)
        blay.setSpacing(4)

        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        blay.addWidget(self.path_label)
        # The done count as a bar, as every labelling tool shows a job's
        # progress: a plot is a few hundred trees, and "38%" reads slower
        # than a bar a third full.
        self.progress = QProgressBar()
        self.progress.setObjectName("doneProgress")
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        blay.addWidget(self.progress)

        # Find a tree by number, or show just what is left to do. A row per
        # tree is the right table for a plot of twenty; for one of four
        # hundred it is a scroll, and the one to fix next is the one the
        # field sheet names.
        filter_row = QHBoxLayout()
        filter_row.setSpacing(6)
        self.filter_edit = QLineEdit()
        self.filter_edit.setObjectName("treeFilter")
        self.filter_edit.setPlaceholderText("Find tree ID…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setToolTip(
            "Show only the trees whose ID starts with this"
        )
        self.filter_edit.textChanged.connect(lambda _t: self._apply_filter())
        filter_row.addWidget(self.filter_edit, stretch=1)
        self.pending_only = QCheckBox("To do only")
        self.pending_only.setToolTip("Hide the trees already marked done")
        self.pending_only.toggled.connect(lambda _c: self._apply_filter())
        filter_row.addWidget(self.pending_only)
        blay.addLayout(filter_row)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeaderItem(0).setToolTip(
            "Whether this tree has been marked reviewed in the panel below"
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        # Start in Tree ID order; the user can click any header to re-sort and
        # _populate then keeps whatever they picked (col 0 is now "Done").
        self.table.sortByColumn(1, Qt.SortOrder.AscendingOrder)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Done
        header.setSectionResizeMode(1, QHeaderView.Stretch)           # Tree ID
        self.table.setMinimumHeight(150)  # fill the dock's vertical space
        self.table.cellDoubleClicked.connect(lambda *_: self.on_load_tree())
        # Enter on a row loads it too, so a tree typed into the filter is
        # Enter, Enter away.
        self.table.itemActivated.connect(lambda *_: self.on_load_tree())
        self.filter_edit.returnPressed.connect(self._focus_first_row)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        blay.addWidget(self.table)
        #: fn(tree_id, done) -> path | None: set by the app to mark a tree
        #: done through the editing panel, which owns the done list once a
        #: tree is loaded. ``None`` back means it had nowhere to save, and
        #: the sidecar is written here instead.
        self.on_mark_done = None

        layout.addWidget(self.trees_box)

        controller.on_saved = self._populate
        # re-tint the "done" rows when the colour theme changes (also runs
        # _populate once now)
        theme.subscribe(lambda _mode: self._populate())

    def _read_done(self) -> set[int]:
        """Reuse SegFixWidget's own progress sidecar (keyed by source path,
        which stays constant across tree switches in this mode) so the two
        panels' "done" state stays in sync without duplicating a file."""
        return read_done(self.c.catalog.path)

    def _populate(self) -> None:
        done = self._read_done()
        records = sorted(self.c.catalog.records.values(), key=lambda r: r.label)
        total = len(records)
        n_done = sum(1 for rec in records if rec.label in done)
        pct = round(100 * n_done / total) if total else 0
        catalog = self.c.catalog
        decimated = (
            f" · downsampled to {catalog.voxel_size * 100:g} cm"
            if catalog.is_decimated else ""
        )
        self.path_label.setText(
            f"{n_done}/{total} trees ({pct}%) done in "
            f"{os.path.basename(catalog.path)}{decimated}"
        )
        if catalog.is_decimated:
            self.path_label.setToolTip(
                f"Editing {catalog.working_count:,} of {catalog.count:,} "
                f"points, one per {catalog.voxel_size:g} m voxel. Saving "
                "interpolates your edits back onto every original point."
            )
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(n_done)
        self.progress.setToolTip(f"{n_done} of {total} trees marked done")
        self.table.setSortingEnabled(False)
        self.table.setRowCount(total)
        done_brush = QBrush(theme.done_row_bg())
        # The same swatch the view and the queue below give each tree, so a
        # colour seen in the canopy can be found in the list.
        labels = np.array([rec.label for rec in records], dtype=np.int64)
        rgba = (colors_for_labels(labels, catalog.label_colors)
                if total else np.empty((0, 4)))
        for row, rec in enumerate(records):
            is_done = rec.label in done
            mark = "✓" if is_done else ""
            for col, value in enumerate([mark, rec.label, rec.count]):
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, value)
                item.setData(Qt.UserRole, rec.label)
                if is_done:
                    item.setBackground(done_brush)
                if col == 1:
                    r, g, b = (int(v * 255) for v in rgba[row][:3])
                    swatch = QPixmap(12, 12)
                    swatch.fill(QColor(r, g, b))
                    item.setData(Qt.DecorationRole, swatch)
                self.table.setItem(row, col, item)
        self.table.setSortingEnabled(True)  # re-sorts by the header indicator
        self._apply_filter()

    def _apply_filter(self) -> None:
        """Hide the rows the filter box and "To do only" rule out."""
        prefix = self.filter_edit.text().strip()
        pending = self.pending_only.isChecked()
        shown = 0
        for row in range(self.table.rowCount()):
            id_item = self.table.item(row, 1)
            done_item = self.table.item(row, 0)
            if id_item is None or done_item is None:
                continue
            keep = str(id_item.data(Qt.DisplayRole)).startswith(prefix)
            if pending and done_item.data(Qt.DisplayRole):
                keep = False
            self.table.setRowHidden(row, not keep)
            shown += keep
        self.trees_box.setTitle(
            "All Trees" if shown == self.table.rowCount()
            else f"All Trees ({shown} of {self.table.rowCount()} shown)"
        )

    def visible_labels(self) -> list[int]:
        """The tree IDs the table shows, after the filter, top to bottom."""
        return [
            self.table.item(row, 0).data(Qt.UserRole)
            for row in range(self.table.rowCount())
            if not self.table.isRowHidden(row)
        ]

    def _focus_first_row(self) -> None:
        """Enter in the filter box: select the first match, so a second
        Enter loads it."""
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row):
                self.table.selectRow(row)
                self.table.setFocus()
                return

    def _context_menu(self, pos) -> None:
        """Right-click a row: what double-click and the queue's Done box do,
        named, the way every tree view since the file manager has done it."""
        item = self.table.itemAt(pos)
        if item is None:
            return
        self.table.selectRow(item.row())
        label = item.data(Qt.UserRole)
        done = label in self._read_done()
        menu = QMenu(self.table)
        load = menu.addAction(f"Load tree {label} with neighbours")
        load.triggered.connect(self.on_load_tree)
        mark = menu.addAction(
            f"Mark tree {label} {'not done' if done else 'done'}"
        )
        mark.triggered.connect(lambda: self.set_done(label, not done))
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def set_done(self, label: int, done: bool) -> None:
        """Mark a tree done or not from this table."""
        saved = self.on_mark_done(label, done) if self.on_mark_done else None
        if saved is None:
            update_done(self.c.catalog.path, label, done)
            self.c.view.status = (
                f"Tree {label} marked {'done' if done else 'not done'}"
            )
        self._populate()

    def refresh(self) -> None:
        """Re-read done-state and point counts; call after external changes
        (e.g. the shared segfix panel marking a tree done)."""
        self._populate()

    def _selected_label(self) -> int | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        return self.table.item(row, 0).data(Qt.UserRole)

    def on_load_tree(self) -> None:
        label = self._selected_label()
        if label is None:
            self.c.view.status = "Select a tree row first"
            return
        busy(self.c.view, f"Loading tree {label}…")
        try:
            self.c.view.status = self.c.load_tree(label)
        except ValueError:
            # Edits flushed by load_tree() (e.g. reassigning/unassigning all
            # of this tree's points before switching away) can leave a
            # still-visible table row pointing at a label with no points
            # left. That's a stale selection, not a crash-worthy error.
            self.c.view.status = f"Tree {label} no longer has any points"
        self._populate()
