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

"""The keyboard and mouse reference: Help ▸ Keyboard Shortcuts, or F1.

Every key of the review loop sits under the left hand, which is only a
help once the hand knows where they are. Until then the labels in the
panel are the reference, and they are scattered over four boxes and a
floating column. This is the one page that lists them all, grouped the way
the loop runs: select, edit, move through the queue, look.

:data:`SHORTCUT_HELP` is the single source for the page, and the tests hold
it against :func:`segfix.widgets.shortcut_bindings`, so a key can't be
bound without being written down here, or written down without being bound.
"""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTextBrowser,
    QVBoxLayout,
)

#: ``(group, key, what it does)``, in the order the page shows them. Keys
#: are spelled as :func:`segfix.widgets.shortcut_bindings` spells them.
SHORTCUT_HELP: tuple[tuple[str, str, str], ...] = (
    ("Select", "Q", "Lasso: drag a loop around points (Shift adds)"),
    ("Select", "W", "Lasso, but only points already in the current tree"),
    ("Select", "E", "Cluster: click a point to take the connected patch of "
                    "its tree; click the same spot again to grow it"),
    ("Select", "R", "Tighten the cluster gap one step"),
    ("Select", "T", "Loosen the cluster gap one step"),
    ("Select", "B", "Invert the selection within what is shown; with nothing "
                    "selected, select everything shown"),
    ("Select", "Esc", "Back to moving the camera"),
    ("Edit", "A", "Add the selection to the current tree"),
    ("Edit", "1", "Move the selection into the 1st neighbouring tree"),
    ("Edit", "2", "Move the selection into the 2nd neighbouring tree"),
    ("Edit", "3", "Move the selection into the 3rd neighbouring tree"),
    ("Edit", "4", "Move the selection into the 4th neighbouring tree"),
    ("Edit", "5", "Move the selection into the 5th neighbouring tree"),
    ("Edit", "S", "Split the selection off as a new tree"),
    ("Edit", "D", "Unassign the selection, or the whole current tree"),
    ("Edit", "X", "Mark the selection as noise, or the whole current tree"),
    ("Edit", "Ctrl+1", "Give the selection the 1st point class"),
    ("Edit", "Ctrl+2", "Give the selection the 2nd point class"),
    ("Edit", "Ctrl+3", "Give the selection the 3rd point class"),
    ("Edit", "Ctrl+4", "Give the selection the 4th point class"),
    ("Edit", "Ctrl+5", "Give the selection the 5th point class"),
    ("Edit", "Ctrl+Z", "Undo"),
    ("Edit", "Ctrl+Shift+Z", "Redo"),
    ("Queue", "Space", "Mark the current tree done and go to the next "
                       "unfinished one"),
    ("Queue", "Z", "Previous tree, without marking done"),
    ("Queue", "V", "Next tree, without marking done"),
    ("View", "F", "Show or hide the unassigned and noise points"),
    ("View", "G", "Hide every other loaded tree"),
    ("View", "Shift+G", "Fade every other loaded tree"),
    ("View", "Shift+F", "Colour by point class, or by tree"),
    ("View", "C", "Cross section on or off"),
    ("View", "Shift+Q", "Draw a lasso section outline"),
    ("View", "Shift+C", "Lasso section on or off"),
    ("View", "Home", "Fit the whole loaded cloud in the view"),
    ("Project", "Ctrl+S", "Save the project"),
    ("Project", "Ctrl+O", "Open another project"),
    ("Project", "F1", "This page"),
)

#: The keys the left-hand layout replaced, still bound for anyone who
#: learned them (see :func:`segfix.widgets.shortcut_bindings`).
LEGACY_NOTE = (
    "The keys these replaced still work: L, Ctrl+L, K, [, ], N, U, H, "
    "Shift+L and the arrow keys."
)

#: ``(action, how)`` for the mouse, which has no bindings table of its own.
MOUSE_HELP: tuple[tuple[str, str], ...] = (
    ("Left-drag", "Rotate the view"),
    ("Right-drag", "Pan"),
    ("Wheel", "Zoom"),
    ("Double-click a point", "Put the rotation centre there, or move "
                             "there while inside the cloud"),
    ("Ctrl+click a point", "Make that point's tree the current tree"),
    ("Shift while selecting", "Add to the selection instead of replacing it"),
    ("Ctrl+wheel", "Slide the cross-section slab along its axis"),
    ("Ctrl+Shift+wheel", "Thicken or thin the cross-section slab"),
    ("Drag the slab's arrows", "Move one face of the slab; the square "
                               "slides the whole slab"),
)


def groups() -> list[str]:
    """The groups of :data:`SHORTCUT_HELP`, in first-seen order."""
    seen: list[str] = []
    for group, _key, _what in SHORTCUT_HELP:
        if group not in seen:
            seen.append(group)
    return seen


def help_html() -> str:
    """The reference as one HTML page: one table, a heading row per group,
    so the key column lines up from top to bottom."""
    parts = ["<h2>Keyboard</h2>", "<table cellpadding='4' width='100%'>"]
    for group in groups():
        parts.append(f"<tr><th colspan='2' align='left'><br>{group}</th></tr>")
        for g, key, what in SHORTCUT_HELP:
            if g == group:
                parts.append(f"<tr><td width='110'><b><code>{key}</code></b>"
                             f"</td><td>{what}</td></tr>")
    parts.append("</table>")
    parts.append(f"<p><i>{LEGACY_NOTE}</i></p>")
    parts.append("<h2>Mouse</h2><table cellpadding='4' width='100%'>")
    for action, how in MOUSE_HELP:
        parts.append(f"<tr><td width='170'><b>{action}</b></td>"
                     f"<td>{how}</td></tr>")
    parts.append("</table>")
    return "\n".join(parts)


class ShortcutsDialog(QDialog):
    """A scrollable page of every key and mouse gesture. Modeless, so it can
    stay open beside the window while the hand learns the keys."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keyboard shortcuts")
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.resize(560, 640)
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        self.browser.setHtml(help_html())
        layout.addWidget(self.browser, stretch=1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)


def show_shortcuts(parent) -> ShortcutsDialog:
    """Open the reference (or raise it, if it is already open)."""
    dialog = getattr(parent, "_shortcuts_dialog", None)
    if dialog is None:
        dialog = ShortcutsDialog(parent)
        parent._shortcuts_dialog = dialog
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
