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

"""Help ▸ Keyboard Shortcuts is the one page that lists every key.

It is only worth having if it is right, so it is held against the real
binding map both ways: a key can't be bound without being written down,
and nothing written down can be a key that does nothing.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication  # noqa: E402

from segfix import shortcuts_ui  # noqa: E402
from tests.test_shortcuts import EXPECTED, LEGACY, _panel  # noqa: E402

#: Keys bound as menu-action shortcuts in app._build_menus rather than in
#: shortcut_bindings: they are real keys all the same.
MENU_KEYS = {"Ctrl+Z", "Ctrl+Shift+Z", "Ctrl+S", "Ctrl+O", "Home", "F1"}


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def _help_keys() -> set[str]:
    return {key for _group, key, _what in shortcuts_ui.SHORTCUT_HELP}


def test_every_bound_key_is_on_the_page():
    from segfix.widgets import shortcut_bindings

    bound = set(shortcut_bindings(_panel())) - set(LEGACY)
    missing = bound - _help_keys()
    assert not missing, f"bound but not on the help page: {sorted(missing)}"


def test_nothing_on_the_page_is_a_key_that_does_nothing():
    unbound = _help_keys() - set(EXPECTED) - MENU_KEYS
    assert not unbound, f"on the help page but not bound: {sorted(unbound)}"


def test_the_legacy_keys_are_mentioned_once_not_listed():
    """One line, not thirty rows: they are there for a hand that already
    knows them, and that hand doesn't need to read them."""
    for key in LEGACY:
        assert key not in _help_keys()
        mention = "arrow keys" if key in ("Left", "Right") else key
        assert mention in shortcuts_ui.LEGACY_NOTE


def test_the_page_groups_the_loop_in_order():
    assert shortcuts_ui.groups() == ["Select", "Edit", "Queue", "View",
                                     "Project"]


def test_the_html_names_every_key_and_gesture():
    html = shortcuts_ui.help_html()
    for key in _help_keys():
        assert f"<code>{key}</code>" in html
    for action, _how in shortcuts_ui.MOUSE_HELP:
        assert action in html


def test_the_dialog_opens_once_and_is_raised_after(_qapp):
    from qtpy.QtWidgets import QMainWindow

    win = QMainWindow()
    first = shortcuts_ui.show_shortcuts(win)
    second = shortcuts_ui.show_shortcuts(win)
    assert first is second
    assert first.isVisible()
    assert "Keyboard" in first.browser.toPlainText()
    first.close()
