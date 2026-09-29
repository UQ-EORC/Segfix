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

"""The "Point classes" set-up dialog.

Point classes (leaf, wood, understorey, ground, ...) live in one numeric
per-point field of the cloud. This dialog picks that field and names the
values it holds — the file only stores the numbers, so the names are the
user's, remembered with the project — and adds any new class the user will
need. See :mod:`segfix.treecatalog` for how the field is read and saved.
"""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

#: Fields a class most likely lives in, most likely first.
CLASS_FIELD_CANDIDATES = (
    "classification", "semantic", "semantic_seg", "class", "classes",
    "label", "category", "scalar_classification",
)

#: The ASPRS standard classes, as a starting point for a LAS classification.
LAS_CLASS_NAMES = {
    0: "Never classified", 1: "Unclassified", 2: "Ground",
    3: "Low vegetation", 4: "Medium vegetation", 5: "High vegetation",
    6: "Building", 7: "Low point (noise)", 9: "Water",
}

NONE_ITEM = "(no class field)"


def default_class_field(fields: list[str]) -> str | None:
    lowered = {f.lower(): f for f in fields}
    for cand in CLASS_FIELD_CANDIDATES:
        if cand in lowered:
            return lowered[cand]
    return None


class ClassSetupDialog(QDialog):
    """Pick the class field and name its values.

    ``catalog`` is the open :class:`~segfix.treecatalog._BaseCatalog`;
    ``field`` and ``names`` are what the project already has, if anything.
    """

    def __init__(self, catalog, field: str | None = None,
                 names: dict[int, str] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Point classes")
        self.catalog = catalog
        self._saved_field = field
        self._saved_names = dict(names or {})

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Point classes (leaf, wood, ground…) are read from one per-point "
            "field of the cloud, and saved back to it. The file stores only "
            "numbers: name each value here. Add any class you'll need."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        row = QHBoxLayout()
        row.addWidget(QLabel("Class field"))
        self.field_combo = QComboBox()
        self.field_combo.addItem(NONE_ITEM)
        fields = catalog.class_fields()
        self.field_combo.addItems(fields)
        row.addWidget(self.field_combo, stretch=1)
        layout.addLayout(row)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Value", "Name"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.table)

        edit_row = QHBoxLayout()
        self.add_btn = QPushButton("Add class")
        self.add_btn.clicked.connect(self._add_row)
        edit_row.addWidget(self.add_btn)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setToolTip(
            "Drop the selected class from the list. Points that hold its "
            "value keep it, and show in grey."
        )
        self.remove_btn.clicked.connect(self._remove_rows)
        edit_row.addWidget(self.remove_btn)
        edit_row.addStretch(1)
        layout.addLayout(edit_row)

        self.error = QLabel()
        self.error.setStyleSheet("color: #d04040;")
        self.error.setWordWrap(True)
        layout.addWidget(self.error)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        start = field if field in fields else default_class_field(fields)
        self.field_combo.setCurrentText(start or NONE_ITEM)
        self.field_combo.currentTextChanged.connect(self._fill_table)
        self._fill_table(self.field_combo.currentText())
        self.resize(460, 420)

    # -- the value/name table -----------------------------------------
    def field(self) -> str | None:
        text = self.field_combo.currentText()
        return None if text == NONE_ITEM else text

    def _fill_table(self, _text: str = "") -> None:
        field = self.field()
        self.table.setRowCount(0)
        self.error.clear()
        on = field is not None
        for w in (self.table, self.add_btn, self.remove_btn):
            w.setEnabled(on)
        if not on:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            present = [int(v) for v in self.catalog.class_values_in(field)]
        finally:
            QApplication.restoreOverrideCursor()
        if field == self._saved_field:
            names = dict(self._saved_names)
        elif field.lower() == "classification":
            names = {v: LAS_CLASS_NAMES[v] for v in present if v in LAS_CLASS_NAMES}
        else:
            names = {}
        for value in sorted(set(present) | set(names)):
            self._append(value, names.get(value, f"Class {value}"))

    def _append(self, value: int, name: str) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(value)))
        self.table.setItem(row, 1, QTableWidgetItem(name))

    def _add_row(self) -> None:
        values = [v for v, _n in self._rows(strict=False)]
        lo, _hi = self.catalog.class_range(self.field())
        self._append(max(values) + 1 if values else max(lo, 0), "")
        self.table.setCurrentCell(self.table.rowCount() - 1, 1)
        self.table.editItem(self.table.item(self.table.rowCount() - 1, 1))

    def _remove_rows(self) -> None:
        for row in sorted({i.row() for i in self.table.selectedIndexes()},
                          reverse=True):
            self.table.removeRow(row)

    def _rows(self, strict: bool = True) -> list[tuple[int, str]]:
        out = []
        for row in range(self.table.rowCount()):
            value_item, name_item = self.table.item(row, 0), self.table.item(row, 1)
            text = value_item.text().strip() if value_item else ""
            name = name_item.text().strip() if name_item else ""
            try:
                value = int(text)
            except ValueError:
                if strict:
                    raise ValueError(f"Row {row + 1}: {text!r} isn't a whole number")
                continue
            out.append((value, name))
        return out

    def names(self) -> dict[int, str]:
        """``{value: name}`` as entered; raises ValueError on a problem."""
        field = self.field()
        if field is None:
            return {}
        rows = self._rows()
        lo, hi = self.catalog.class_range(field)
        seen_names: set[str] = set()
        out: dict[int, str] = {}
        for value, name in rows:
            if not name:
                raise ValueError(f"Class {value} needs a name")
            if value in out:
                raise ValueError(f"Value {value} is listed twice")
            if name.lower() in seen_names:
                raise ValueError(f"Two classes are called {name}")
            if not lo <= value <= hi:
                raise ValueError(
                    f"{field} holds values {lo} to {hi}; {value} won't fit"
                )
            seen_names.add(name.lower())
            out[value] = name
        return out

    def _accept(self) -> None:
        try:
            self.names()
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        self.accept()


def prompt_class_setup(parent, catalog, field=None, names=None):
    """Show :class:`ClassSetupDialog`; ``(field, names)`` or None if
    cancelled. ``field`` is None when the user turned classes off."""
    dlg = ClassSetupDialog(catalog, field, names, parent)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    return dlg.field(), dlg.names()
