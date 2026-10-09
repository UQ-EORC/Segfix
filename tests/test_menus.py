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

"""The menu bar lists everything the panel can do, with its key beside it.

A menu is where a new user looks first, and the one place every action
can be read in one column. The keys stay bound as the panel's shortcuts;
the menus only show them. Binding a key twice would make it ambiguous,
which Qt answers by firing neither, so the menu's copy has a context no
key press from the window can satisfy, and the panel's copy still fires.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy")
from qtpy.QtCore import Qt  # noqa: E402
from qtpy.QtTest import QTest  # noqa: E402
from qtpy.QtWidgets import QApplication, QMainWindow, QMenu  # noqa: E402

from segfix import app  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def window():
    try:
        from segfix.cloudview import CloudView

        view = CloudView()
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no vispy canvas available: {exc}")
    from segfix.model import PointCloud
    from segfix.widgets import SegFixController, SegFixWidget

    rng = np.random.default_rng(0)
    coords = rng.random((60, 3)).astype(np.float32)
    labels = np.repeat([1, 2, 0], 20).astype(np.int32)
    empty = PointCloud(coords=np.empty((0, 3), np.float32),
                       labels=np.empty(0, np.int32))
    view.load_cloud(empty)
    seg = SegFixController(view, empty)
    panel = SegFixWidget(seg)
    cloud = PointCloud(coords=coords, labels=labels)
    view.load_cloud(cloud)
    seg.set_cloud(cloud)
    panel._set_current(1, fly=False)
    win = QMainWindow()
    app._build_menus(win, panel, open_path="/nowhere/plot.ply")
    yield win, panel
    panel.deleteLater()
    win.deleteLater()


def _menu(win, title) -> QMenu:
    for act in win.menuBar().actions():
        if act.text().replace("&", "") == title:
            return act.menu()
    raise AssertionError(f"no {title} menu")


def _actions(menu):
    return {a.text(): a for a in menu.actions() if a.text()}


def test_the_bar_reads_like_a_desktop_app(window):
    win, _ = window
    titles = [a.text().replace("&", "") for a in win.menuBar().actions()]
    assert titles == ["File", "Edit", "Tree", "View", "Inventory",
                      "Preferences", "Help"]


def _key(act) -> str:
    return act.shortcut().toString()


def test_the_loop_is_listed_with_its_keys(window):
    win, _ = window
    tree = _actions(_menu(win, "Tree"))
    assert _key(tree["Add selection to current tree"]) == "A"
    assert _key(tree["Mark done and go to next"]) == "Space"
    edit = _actions(_menu(win, "Edit"))
    assert _key(edit["Lasso"]) == "Q"
    assert _key(edit["Invert selection"]) == "B"
    view = _actions(_menu(win, "View"))
    assert _key(view["Fade other trees"]) == "Shift+G"


def test_the_panels_keys_are_shown_but_cannot_fire_from_the_window(window):
    """A QAction shortcut on Q beside the panel's QShortcut on Q would be an
    ambiguous key, which Qt answers by firing neither. The menu's copy has
    a widget-only context, so no key press from the window matches it."""
    win, _ = window
    from segfix.widgets import shortcut_bindings
    from tests.test_shortcuts import _panel

    panel_keys = set(shortcut_bindings(_panel()))
    for title in ("Edit", "Tree", "View"):
        for act in _menu(win, title).actions():
            if _key(act) in panel_keys:
                assert act.shortcutContext() == Qt.ShortcutContext.WidgetShortcut, (
                    act.text()
                )


def test_a_key_pressed_in_the_window_still_reaches_the_panel(window):
    """The proof that the menus' copies are not ambiguous: with the panel's
    shortcuts bound on the same window, Q arms the lasso and F hides the
    unassigned points, as they always did."""
    from segfix.widgets import bind_shortcuts

    win, panel = window
    bind_shortcuts(win, panel)
    win.show()
    QApplication.processEvents()
    assert not panel.lasso_btn.isChecked()
    QTest.keyClick(win, Qt.Key.Key_Q)
    assert panel.lasso_btn.isChecked()
    QTest.keyClick(win, Qt.Key.Key_Escape)
    assert panel.move_btn.isChecked() and not panel.lasso_btn.isChecked()
    assert panel.show_unassigned.isChecked()
    QTest.keyClick(win, Qt.Key.Key_F)
    assert not panel.show_unassigned.isChecked()


def test_the_view_menu_reports_the_toggles(window):
    win, panel = window
    view = _menu(win, "View")
    acts = _actions(view)
    panel.show_unassigned.setChecked(False)
    view.aboutToShow.emit()
    assert not acts["Show unassigned points"].isChecked()
    acts["Show unassigned points"].trigger()
    assert panel.show_unassigned.isChecked()
    view.aboutToShow.emit()
    assert acts["Show unassigned points"].isChecked()
    acts["Show unassigned points"].trigger()
    assert not panel.show_unassigned.isChecked()


def test_the_edit_menu_knows_the_mode(window):
    win, panel = window
    edit = _menu(win, "Edit")
    acts = _actions(edit)
    panel.lasso_btn.setChecked(True)
    edit.aboutToShow.emit()
    assert acts["Lasso"].isChecked()
    assert not acts["Move the camera"].isChecked()
    acts["Move the camera"].trigger()
    assert panel.move_btn.isChecked()
    assert not panel.lasso_btn.isChecked()


def test_the_mode_actions_are_one_group(window):
    win, _ = window
    acts = _actions(_menu(win, "Edit"))
    group = acts["Lasso"].actionGroup()
    assert group is not None and group.isExclusive()
    assert acts["Cluster"].actionGroup() is group


def test_fit_and_help_are_real_shortcuts(window):
    win, _ = window
    assert _actions(_menu(win, "View"))["Fit whole cloud"].shortcut().toString() == "Home"
    assert _actions(_menu(win, "Help"))["Keyboard Shortcuts…"].shortcut().toString() == "F1"


def test_open_recent_lists_the_registry_minus_this_project(window, tmp_path,
                                                            monkeypatch):
    from segfix import registry

    here = tmp_path / "here"
    here.mkdir()
    (here / "plot.ply").write_bytes(b"")
    other = tmp_path / "other"
    other.mkdir()
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"entries": [
        {"path": str(here), "type": "workspace",
         "last_opened": "2026-10-09T10:00:00"},
        {"path": str(other), "type": "workspace",
         "last_opened": "2026-10-08T10:00:00"},
    ]}))
    monkeypatch.setattr(registry, "registry_path", lambda: reg)
    win, panel = window
    recent = next(a.menu() for a in _menu(win, "File").actions()
                  if a.text() == "Open Recent")
    app._fill_recent_menu(recent, win, panel, None, str(here / "plot.ply"))
    shown = [a.text() for a in recent.actions()]
    assert len(shown) == 1 and shown[0].startswith(str(other))


def test_open_recent_says_so_when_there_is_nothing_else(window, tmp_path,
                                                         monkeypatch):
    from segfix import registry

    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"entries": []}))
    monkeypatch.setattr(registry, "registry_path", lambda: reg)
    win, panel = window
    recent = next(a.menu() for a in _menu(win, "File").actions()
                  if a.text() == "Open Recent")
    app._fill_recent_menu(recent, win, panel, None, None)
    acts = recent.actions()
    assert len(acts) == 1 and not acts[0].isEnabled()


def test_opening_a_recent_project_relaunches_on_its_data_file(window, tmp_path,
                                                               monkeypatch):
    from segfix import workspace

    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / "plot.ply").write_bytes(b"")
    (ws / workspace.MANIFEST_NAME).write_text(json.dumps({"data_file": "plot.ply"}))
    launched = []
    monkeypatch.setattr(app, "_relaunch", lambda: launched.append(dict(os.environ)))
    win, panel = window
    app._open_recent(win, panel, None,
                     {"path": str(ws), "type": "workspace"})
    assert len(launched) == 1
    env = launched[0]
    assert env["SEGFIX_OPEN"] == str(ws / "plot.ply")
    assert env["SEGFIX_OPEN_REGISTRY"] == str(ws)
    assert env["SEGFIX_OPEN_KIND"] == "workspace"
    for key in ("SEGFIX_OPEN", "SEGFIX_OPEN_REGISTRY", "SEGFIX_OPEN_KIND"):
        os.environ.pop(key, None)
