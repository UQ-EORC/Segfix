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

"""Check the storyboard against the app before spending a quarter of an hour.

A recording fails at the speed of the video: a name the panel no longer has
surfaces eighteen minutes in, with the machine's screen tied up the whole
time, and the only clue is a traceback in a log. Everything the storyboard
reaches for is written down as an attribute of a handful of objects, so it
can be checked statically in a second against the real ones.

Reads every ``panel.x``, ``scene.x``, ``view.x``, ``d.x`` and
``findChild(..., "name")`` out of the storyboard's syntax tree and looks for
each on the real widget, built offscreen. Dialog attributes are checked
against the union of the dialogs the walkthrough drives, since they all
arrive as ``d``.

Run by ``run_recording.sh`` before the recording; exits non-zero with a list
of what is missing.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

#: Names the storyboard binds to, and what they are.
TARGETS = ("panel", "scene", "view", "d", "win", "align")

#: Attributes reached through a stand-in rather than a real object, or
#: created by the storyboard itself, so there is nothing to check them on.
SKIP = {
    "d": set(),
    "win": {"menuBar", "findChild", "findChildren"},
    "view": set(),
    "panel": set(),
    "scene": set(),
    "align": set(),
}


def used_attributes(path: Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text())
    found: dict[str, set[str]] = {name: set() for name in TARGETS}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in found):
            found[node.value.id].add(node.attr)
    return found


def object_names(path: Path) -> set[str]:
    """Widget object names the storyboard looks up with findChild."""
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "findChild"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    names.add(arg.value)
                if isinstance(arg, ast.JoinedStr):  # f"view_{name}"
                    parts = [v.value for v in arg.values
                             if isinstance(v, ast.Constant)]
                    names.add("".join(parts) + "*")
    return names


def button_labels(path: Path):
    """Every ``button(..., "text", …)`` lookup the storyboard makes.

    The other way a recording dies late: a button whose label has been
    reworded since. Returns ``(text, prefix, contains)``.
    """
    tree = ast.parse(path.read_text())
    wanted = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "button"):
            continue
        texts = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not texts:
            continue
        flags = {k.arg: getattr(k.value, "value", False) for k in node.keywords}
        wanted.append((texts[0], bool(flags.get("prefix")),
                       bool(flags.get("contains"))))
    return wanted


def build():
    """The real widgets, offscreen, with a small cloud loaded."""
    from segfix.cloudview import CloudView
    from segfix.density_ui import DownsampleDialog
    from segfix.model import PointCloud
    from segfix.scene_ui import SceneWidget  # noqa: F401 - import check only
    from segfix.shift_ui import GlobalShiftDialog
    from segfix.startup_ui import StartupDialog  # noqa: F401 - import check
    from segfix.widgets import SegFixController, SegFixWidget

    view = CloudView()
    empty = PointCloud(coords=np.empty((0, 3), np.float32),
                       labels=np.empty(0, np.int32))
    view.load_cloud(empty)
    controller = SegFixController(view, empty)
    panel = SegFixWidget(controller)
    rng = np.random.default_rng(0)
    cloud = PointCloud(coords=(rng.random((200, 3)) * 8).astype(np.float32),
                       labels=np.ones(200, np.int32))
    view.load_cloud(cloud)
    controller.set_cloud(cloud)
    dialogs = (
        DownsampleDialog(0.012, 1000, 0.03, 0.42),
        GlobalShiftDialog(np.zeros(3), np.ones(3), np.zeros(3)),
    )
    return panel, view, dialogs, _scene(view, controller)


def _scene(view, controller):
    """A real SceneWidget: half of what the storyboard drives lives on the
    instance rather than the class, so checking the class proves nothing."""
    import tempfile

    from segfix.scene_ui import SceneController, SceneWidget
    from segfix.treecatalog import open_catalog

    rng = np.random.default_rng(1)
    rows = np.empty(400, dtype=np.dtype([("x", "<f4"), ("y", "<f4"),
                                         ("z", "<f4"), ("treeID", "<i4")]))
    coords = (rng.random((400, 3)) * 6).astype(np.float32)
    rows["x"], rows["y"], rows["z"] = coords.T
    rows["treeID"] = np.repeat([1, 2], 200)
    path = Path(tempfile.mkdtemp(prefix="segfix-preflight-")) / "plot.ply"
    with open(path, "wb") as fh:
        fh.write(b"ply\nformat binary_little_endian 1.0\nelement vertex 400\n"
                 b"property float x\nproperty float y\nproperty float z\n"
                 b"property int treeID\nend_header\n")
        fh.write(rows.tobytes())
    catalog = open_catalog(str(path))
    return SceneWidget(SceneController(view, catalog, controller))


def main() -> int:
    app = QApplication.instance() or QApplication([])  # noqa: F841
    story = HERE / "storyboard.py"
    used = used_attributes(story)
    panel, view, dialogs, scene = build()
    problems = []

    def check(label, obj, names, extra=()):
        for name in sorted(names - SKIP.get(label, set())):
            if hasattr(obj, name) or any(hasattr(e, name) for e in extra):
                continue
            problems.append(f"{label}.{name}")

    check("panel", panel, used["panel"])
    check("view", view, used["view"])
    check("scene", scene, used["scene"])
    if used["align"]:
        from segfix import inventory, inventory_ui

        panel.set_stem_map([inventory.Stem("S1", 0.0, 0.0, 0.3, 12.0)])
        check("align", inventory_ui.AlignDialog(panel, [], None), used["align"])
    for name in sorted(used["d"]):
        if not any(hasattr(d, name) for d in dialogs):
            problems.append(f"dialog.{name}")

    # Widgets looked up by object name have to exist on the panel.
    wanted = object_names(story)
    from qtpy.QtWidgets import QWidget

    present = {w.objectName()
               for holder in (panel, panel.top_bar, view.native)
               for w in holder.findChildren(QWidget)}
    for name in sorted(wanted):
        if name.endswith("*"):
            if not any(p.startswith(name[:-1]) for p in present):
                problems.append(f'findChild("{name}")')
        elif name not in present:
            problems.append(f'findChild("{name}")')

    from qtpy.QtWidgets import QAbstractButton

    # The top bar is docked separately by app.py, so it is not under the
    # panel; the overlays are parented to the canvas.
    buttons = [b.text().replace("&", "").strip()
               for holder in (panel, panel.top_bar, view.native)
               for b in holder.findChildren(QAbstractButton)]
    for text, prefix, contains in button_labels(story):
        if any(t == text or (prefix and t.startswith(text))
               or (contains and text in t) for t in buttons):
            continue
        if text in ("New Project",):  # on the startup dialog, not the panel
            continue
        problems.append(f'button("{text}")')

    if problems:
        print("storyboard asks for things the app hasn't got:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"preflight ok: {sum(len(v) for v in used.values())} attributes, "
          f"{len(wanted)} object names, {len(button_labels(story))} button labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
