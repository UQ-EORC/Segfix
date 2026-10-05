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

"""Loading a stem map, and putting it where the trees are.

The alignment dialog is the whole of the coordinate problem: a stem map
arrives in the cloud's CRS, in a local plot frame, or in a global one from a
handheld GPS, and the operator has to be able to see which of those happened
and that the fit landed. So it fits automatically on open, says in metres
how well it did, and lets the shift be nudged by hand with the cylinders
redrawing as it moves.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from . import inventory


class AlignDialog(QDialog):
    """Fit, show and nudge the shift from stem-map to cloud coordinates.

    Live: every change is pushed to the panel as it happens, so the stems
    move over the cloud while the dialog is open. Cancel puts back the
    alignment that was in force when it opened.
    """

    def __init__(self, panel, trees, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Align stem map")
        self.panel = panel
        self.trees = list(trees)
        self._original = panel.alignment

        layout = QVBoxLayout(self)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        form = QFormLayout()
        self.dx = self._spin(panel.alignment.dx)
        self.dy = self._spin(panel.alignment.dy)
        form.addRow("Shift X (m)", self.dx)
        form.addRow("Shift Y (m)", self.dy)
        layout.addLayout(form)

        row = QHBoxLayout()
        self.fit_btn = QPushButton("Fit automatically")
        self.fit_btn.setToolTip(
            "Find the shift from the pattern of stems, ignoring trees that "
            "only one of the two has"
        )
        self.fit_btn.clicked.connect(self.fit)
        row.addWidget(self.fit_btn)
        self.zero_btn = QPushButton("No shift")
        self.zero_btn.setToolTip("For a stem map already in the cloud's CRS")
        self.zero_btn.clicked.connect(
            lambda: (self.dx.setValue(0.0), self.dy.setValue(0.0))
        )
        row.addWidget(self.zero_btn)
        row.addStretch(1)
        layout.addLayout(row)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._apply()

    def _spin(self, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        # Wide enough for a UTM easting: a local stem map's shift is the
        # whole coordinate.
        spin.setRange(-1e7, 1e7)
        spin.setDecimals(2)
        spin.setSingleStep(0.5)
        spin.setValue(value)
        spin.valueChanged.connect(self._apply)
        return spin

    def fit(self) -> None:
        fitted = inventory.fit_shift(self.panel.stems, self.trees)
        self.dx.blockSignals(True)
        self.dy.blockSignals(True)
        self.dx.setValue(fitted.dx)
        self.dy.setValue(fitted.dy)
        self.dx.blockSignals(False)
        self.dy.blockSignals(False)
        self._apply()

    def _apply(self) -> None:
        alignment = inventory.alignment_quality(
            self.panel.stems,
            self.trees,
            inventory.Alignment(dx=self.dx.value(), dy=self.dy.value()),
        )
        self.panel.set_alignment(alignment)
        self.summary.setText(
            f"{len(self.panel.stems)} stems · {len(self.trees)} trees in the "
            f"plot\n{alignment.describe()}"
        )

    def reject(self) -> None:
        self.panel.set_alignment(self._original)
        super().reject()


def _tree_stats(panel, catalog):
    """Stats for every tree in the plot, cheaply.

    From the catalog's per-tree boxes — the alignment needs the whole plot,
    and the loaded working set is one tree and its neighbours. The tree under
    review is measured properly from its points and substituted in, because
    that is the one the operator is looking at.
    """
    stats = inventory.stats_from_records(getattr(catalog, "records", {}))
    measured = panel.tree_stats()
    if measured is not None:
        stats = [measured if s.tree_id == measured.tree_id else s for s in stats]
    return stats


def load_stem_map(win, panel, catalog) -> None:
    """Inventory ▸ Load Stem Map…: read a CSV, align it, draw it."""
    path, _ = QFileDialog.getOpenFileName(
        win, "Load stem map", str(Path.home()), "CSV (*.csv *.txt);;All files (*)"
    )
    if not path:
        return
    try:
        stems, columns = inventory.load_stem_map(path)
    except (OSError, ValueError) as exc:
        QMessageBox.critical(win, "Couldn't read the stem map", str(exc))
        return
    if not stems:
        QMessageBox.warning(
            win, "Empty stem map", f"{path} has no rows with X/Y coordinates."
        )
        return

    _warn_if_unshifted(win, panel)
    panel.stem_map_path = path
    panel.set_stem_map(stems, inventory.Alignment())
    trees = _tree_stats(panel, catalog)
    dialog = AlignDialog(panel, trees, win)
    dialog.fit()  # an answer on screen before anything is touched by hand
    used = ", ".join(f"{k}={v}" for k, v in sorted(columns.items()))
    if dialog.exec() != QDialog.Accepted:
        panel.set_stem_map([], inventory.Alignment())
        panel.stem_map_path = None
        return
    panel.c.view.status = (
        f"Loaded {len(stems)} stems from {Path(path).name} ({used}). "
        f"{panel.alignment.describe()}"
    )


def _warn_if_unshifted(win, panel) -> None:
    """Say so when the cloud is still in raw georeferenced coordinates.

    Coordinates are held as float32, which carries about seven digits: a
    northing of 7,223,250 m is then kept to the nearest half metre, and both
    the stem positions and the circle fitted at breast height are built out
    of differences far smaller than that. Matching still runs, but it is
    matching mush — measured on a synthetic UTM plot, DBH came back 20 cm
    out against 1 cm with the shift applied. Reopening and accepting the
    global shift fixes it.
    """
    from .treecatalog import needs_global_shift

    coords = panel.c.view.coords
    if not len(coords):
        return
    if not needs_global_shift(coords.min(axis=0), coords.max(axis=0)):
        return
    QMessageBox.warning(
        win, "Large coordinates",
        "This cloud is still in its original georeferenced coordinates, "
        "which segfix holds to about half a metre.\n\nStem positions, "
        "heights and the DBH fitted at breast height will all be imprecise, "
        "and matching with them is unreliable. Reopen the project and "
        "accept the global shift it offers.",
    )


def align_stem_map(win, panel, catalog) -> None:
    """Inventory ▸ Align to Cloud…: re-open the alignment dialog."""
    if not panel.stems:
        QMessageBox.information(
            win, "No stem map", "Load a stem map first (Inventory ▸ Load Stem Map…)."
        )
        return
    AlignDialog(panel, _tree_stats(panel, catalog), win).exec()


def export_matches(win, panel, catalog) -> None:
    """Inventory ▸ Export Matches…: the whole plot's tree ↔ stem table.

    Links made by hand win; every other tree is filled in by the automatic
    one-to-one matching, with the stems already taken left out of it. The
    file therefore says what the operator decided *and* what the plot looks
    like where nobody has decided yet — and which is which, in the
    ``matched_by`` column.
    """
    if not panel.stems:
        QMessageBox.information(
            win, "No stem map", "Load a stem map first (Inventory ▸ Load Stem Map…)."
        )
        return
    trees = _tree_stats(panel, catalog)
    by_id = {stem.stem_id: stem for stem in panel.stems}
    matches = {}
    for tree_id, stem_id in panel.stem_links.items():
        stem = by_id.get(stem_id)
        tree = next((t for t in trees if t.tree_id == tree_id), None)
        if stem is None or tree is None:
            continue
        ranked = inventory.candidates(
            tree, [stem], panel.alignment,
            tolerances=inventory.Tolerances(max_distance=float("inf")),
        )
        if ranked:
            matches[tree_id] = ranked[0]

    rest = [t for t in trees if t.tree_id not in matches]
    free = [s for s in panel.stems if s.stem_id not in panel.stem_links.values()]
    matches.update(inventory.match_all(rest, free, panel.alignment))

    default = str(Path(panel.stem_map_path or "matches.csv").with_suffix(""))
    path, _ = QFileDialog.getSaveFileName(
        win, "Export inventory matches", f"{default}_matches.csv", "CSV (*.csv)"
    )
    if not path:
        return
    try:
        inventory.write_matches(
            path, matches, trees, panel.alignment, manual=panel.stem_links
        )
    except OSError as exc:
        QMessageBox.critical(win, "Export failed", str(exc))
        return
    panel.c.view.status = (
        f"Exported {len(matches)} match(es) to {path} "
        f"({len(panel.stem_links)} linked by hand)"
    )


def clear_stem_map(win, panel) -> None:
    """Inventory ▸ Clear Stem Map: stop drawing it. Links are kept — they
    live in the project sidecar, and reloading the same map brings them
    back."""
    panel.set_stem_map([], inventory.Alignment())
    panel.stem_map_path = None
    panel.c.view.status = "Stem map cleared"
