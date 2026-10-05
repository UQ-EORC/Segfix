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

"""Theme switching reaches widgets that carry their own stylesheet.

Regression coverage for a real bug: the top bar's highlighted mode buttons
(Lasso, Cluster, Draw ...) each have a ``QPushButton:checked`` stylesheet,
and after a theme switch they kept the *previous* theme's colours -- dark
buttons in the light theme, light ones after switching back -- while plain
buttons beside them switched correctly.

Offscreen Qt, like the other widget tests; no GL needed.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication, QHBoxLayout, QPushButton, QWidget  # noqa: E402

from segfix import theme  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def _background(button) -> str:
    QApplication.processEvents()
    img = button.grab().toImage()
    return img.pixelColor(6, img.height() // 2).name()


def test_styled_buttons_follow_a_theme_switch_in_both_directions(monkeypatch):
    monkeypatch.setattr(theme, "save", lambda mode: None)  # leave QSettings alone
    app = QApplication.instance()
    host = QWidget()
    row = QHBoxLayout(host)
    plain = QPushButton("Slab…")
    styled = QPushButton("Lasso (L)")
    styled.setCheckable(True)
    # The same shape of stylesheet the real mode buttons carry: it only
    # styles the checked state, so an unchecked button draws in the palette.
    styled.setStyleSheet(
        "QPushButton:checked { background: #7a6a20; color: #ffe066; }"
    )
    row.addWidget(plain)
    row.addWidget(styled)
    host.show()

    theme.apply(app, "dark")
    assert _background(styled) == _background(plain)
    for mode in ("light", "dark", "light", "dark"):
        theme.set_mode(app, mode)
        assert _background(styled) == _background(plain), (
            f"stylesheet button kept the previous theme after switching to {mode}"
        )
    host.close()


def test_a_first_run_is_light():
    """The remembered choice wins once there is one; this is what a machine
    with no saved setting gets."""
    assert theme.DEFAULT == "light"
    assert theme.CANVAS_BG[theme.DEFAULT] == "#f2f2f2"


def test_qt_is_told_which_scheme_the_app_is_in():
    """Qt 6.5+ follows the desktop's light/dark preference and re-applies
    the system palette to widgets as they are created. On a prefer-dark
    desktop that overrode the light palette: the canvas went light and the
    window around it stayed dark."""
    from qtpy.QtCore import Qt

    app = QApplication.instance()
    hints = app.styleHints()
    if not hasattr(hints, "setColorScheme"):
        pytest.skip("Qt too old to have a colour scheme to set")
    theme.apply(app, "light")
    assert hints.colorScheme() == Qt.ColorScheme.Light
    theme.apply(app, "dark")
    assert hints.colorScheme() == Qt.ColorScheme.Dark
