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

"""App-wide light / dark colour theme.

Qt widgets otherwise follow whatever palette the desktop hands us, while
segfix's canvas overlays and the 3-D background are styled for dark. This
module owns one explicit choice, applies it to the ``QApplication`` palette
(forcing the Fusion style so the two themes look the same on every
platform), remembers it in ``QSettings``, and lets the canvas / overlays
subscribe so they can restyle themselves live.
"""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import QSettings
from qtpy.QtGui import QColor, QPalette

_ORG, _APP = "segfix", "segfix"
#: What a first run looks like. Light: a point cloud is read against the
#: background all day, and most of the people using this are in a lit office
#: beside a window rather than a dark room. Either way it is one menu away,
#: and the choice is remembered from then on (see load/save below).
DEFAULT = "light"

#: vispy background-colour string per mode (see CloudView.set_background)
CANVAS_BG = {"dark": "#262626", "light": "#f2f2f2"}

_mode = DEFAULT
_listeners: list[Callable[[str], None]] = []


# -- palettes -------------------------------------------------------------
def _dark_palette() -> QPalette:
    p = QPalette()
    window = QColor("#2b2b2b")
    base = QColor("#232323")
    text = QColor("#e6e6e6")
    disabled = QColor("#787878")
    p.setColor(QPalette.ColorRole.Window, window)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, base)
    p.setColor(QPalette.ColorRole.AlternateBase, QColor("#2f2f2f"))
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, window)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.BrightText, QColor("#ff6060"))
    p.setColor(QPalette.ColorRole.ToolTipBase, window)
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.Highlight, QColor("#3d6ea5"))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.Link, QColor("#5aa9e6"))
    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.ButtonText,
    ):
        p.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    return p


def _light_palette() -> QPalette:
    p = QPalette()
    window = QColor("#f2f2f2")
    text = QColor("#1a1a1a")
    disabled = QColor("#9a9a9a")
    p.setColor(QPalette.ColorRole.Window, window)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor("#e9e9e9"))
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, window)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.BrightText, QColor("#c00000"))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor("#ffffdc"))
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.Highlight, QColor("#3d7eb8"))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorRole.Link, QColor("#1a6dbf"))
    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.ButtonText,
    ):
        p.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    return p


# -- overlay styling ----------------------------------------------------
def panel_colors(mode: str | None = None) -> dict[str, str]:
    """Backing / border / text colours for a floating canvas overlay panel
    (the point-size and current-tree boxes) at the given mode."""
    if (mode or _mode) == "light":
        return {
            "bg": "rgba(250, 250, 250, 230)",
            "border": "rgba(0, 0, 0, 55)",
            "text": "#1a1a1a",
            "subtext": "#555555",
        }
    return {
        "bg": "rgba(28, 28, 30, 210)",
        "border": "rgba(255, 255, 255, 45)",
        "text": "#e8e8e8",
        "subtext": "#d0d0d0",
    }


def canvas_ink(mode: str | None = None) -> QColor:
    """Pen/text colour for things drawn straight onto the canvas (the scale
    bar), i.e. readable against :data:`CANVAS_BG`."""
    return QColor("#202020" if (mode or _mode) == "light" else "#e8e8e8")


def done_row_bg(mode: str | None = None) -> QColor:
    """Table-row tint marking a reviewed tree — a dark green on the dark
    theme, a pale green on the light one, each readable behind that theme's
    text colour."""
    return QColor("#233e2a" if (mode or _mode) == "dark" else "#c8e8c8")


# -- state / application ----------------------------------------------
def current() -> str:
    return _mode


def load() -> str:
    value = str(QSettings(_ORG, _APP).value("theme", DEFAULT)).lower()
    return "light" if value == "light" else "dark"


def save(mode: str) -> None:
    QSettings(_ORG, _APP).setValue("theme", mode)


def apply(app, mode: str | None = None) -> None:
    """Apply ``mode`` (default: the saved choice) to ``app`` and notify every
    subscriber. Idempotent — safe to call more than once."""
    global _mode
    _mode = "light" if (mode or load()) == "light" else "dark"
    app.setStyle("Fusion")  # re-set each time: forces a full re-polish
    app.setPalette(_light_palette() if _mode == "light" else _dark_palette())
    _repolish_styled_widgets(app)
    for fn in list(_listeners):
        try:
            fn(_mode)
        except Exception:
            pass


def _repolish_styled_widgets(app) -> None:
    """Make widgets with their own stylesheet pick up the new palette.

    The re-polish that ``setStyle`` forces runs *before* ``setPalette``, and a
    widget with a stylesheet (the top bar's highlighted mode buttons: Lasso,
    Cluster, Draw ...) resolves every colour its rules don't set from the
    palette in force at that moment. So those widgets were polished against
    the *previous* theme and kept its colours -- dark buttons in the light
    theme, light ones after switching back. Polishing them again now, with
    the new palette in place, fixes both directions.
    """
    for w in app.allWidgets():
        if w.styleSheet():
            style = w.style()
            style.unpolish(w)
            style.polish(w)
            w.update()


def set_mode(app, mode: str) -> None:
    """Switch theme from a menu action: apply it and remember it."""
    apply(app, mode)
    save(_mode)


def subscribe(fn: Callable[[str], None]) -> None:
    """Register ``fn(mode)`` to run now and on every later theme change."""
    _listeners.append(fn)
    fn(_mode)
