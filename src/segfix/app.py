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

"""Command-line entry point.

``segfix`` always opens with a startup dialog to pick a recent project or
import a new file — there's no way to pass a cloud path directly on the
command line, so every session goes through the registry. See
:mod:`registry`/:mod:`startup_ui`.

Once a path is chosen it opens in the one editing mode: a table of every tree
in the file, where double-clicking a row loads that tree plus its spatial
neighbours rather than the whole cloud. See :mod:`treecatalog`/:mod:`scene_ui`.

That mode needs to seek to an arbitrary point without parsing everything
before it, so the accepted formats are the ones that store points as
fixed-size records at a known offset: binary PLY and uncompressed LAS.
arbor's ``.laz`` output is decompressed to ``.las`` on import (see
:mod:`segfix.workspace`). :func:`segfix.io.load` rejects the rest.
"""

from __future__ import annotations

import argparse
import sys


def _prefer_discrete_gpu() -> None:
    """Best-effort: on a hybrid-graphics machine, point OpenGL/Direct3D at
    the discrete GPU instead of whatever the platform defaults to (often
    the weaker integrated one — see the "GPU:" status-bar label the main
    window shows to confirm which one actually got used). Detects hardware
    rather than hardcoding one machine's GPU model name, never overrides a
    preference already set (env var on Linux, the registry key on Windows),
    and never raises — worst case this is a no-op and the platform default
    stands.

    Must run before any GL context is created (and before the Qt platform
    plugin is chosen): the Linux/WSL half relies on the underlying GL/EGL
    libraries not having picked an adapter yet, which happens at first
    context creation, not at process start — setting the env var any later
    risks losing the race. The Windows half doesn't have that race (it's a
    registry key, not an env var) but only takes effect from the *next*
    launch of this Python executable, not this one: Windows reads a
    process's GPU preference at process creation, before any of our code
    has had a chance to run.
    """
    try:
        if sys.platform == "win32":
            _prefer_discrete_gpu_windows()
        elif sys.platform == "linux":
            _prefer_discrete_gpu_linux()
        # macOS: no per-process lever like the two above. Apple Silicon has
        # a single GPU anyway; Intel-Mac dual-GPU switching is a system
        # Energy Saver setting, not something a process can request.
    except Exception:
        pass  # detection failing should never block startup


def _prefer_discrete_gpu_linux() -> None:
    import os
    import shutil
    import subprocess

    is_wsl = False
    try:
        with open("/proc/version", encoding="utf-8") as f:
            is_wsl = "microsoft" in f.read().lower()
    except OSError:
        pass
    has_nvidia = shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, timeout=2
    ).returncode == 0
    if not has_nvidia:
        return
    if is_wsl:
        # Mesa's D3D12 backend (the only GL path WSLg offers) matches this
        # against the adapter names Windows exposes, substring not exact —
        # so it still works if the discrete card is a different NVIDIA
        # model than whatever machine this was tested on.
        os.environ.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")
    else:
        # Native Linux NVIDIA Optimus/PRIME laptops: ask for the discrete
        # GPU's GLX vendor lib instead of the integrated one.
        os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
        os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")


def _prefer_discrete_gpu_windows() -> None:
    """Set this Python executable's GPU preference to "High performance" in
    the per-app registry key Settings > System > Display > Graphics also
    writes to — the documented, vendor-agnostic way one process can ask
    Windows for the discrete GPU without touching any other app's choice.
    HKEY_CURRENT_USER, so it needs no elevation; keyed by ``sys.executable``,
    so it only affects this specific Python (e.g. one conda env), not every
    Python on the machine, and never touches a value already set — by the
    user via Settings, or by this same call on a previous run.
    """
    import winreg

    exe = sys.executable
    key = winreg.CreateKeyEx(
        winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\DirectX\UserGpuPreferences",
        0,
        winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE,
    )
    try:
        try:
            winreg.QueryValueEx(key, exe)
            return  # already has a preference (ours or the user's) — leave it
        except FileNotFoundError:
            pass
        winreg.SetValueEx(key, exe, 0, winreg.REG_SZ, "GpuPreference=2;")
    finally:
        winreg.CloseKey(key)


def _disable_window_ghosting() -> None:
    """Stop Windows replacing a busy window with a grey "ghost".

    When a top-level window goes about five seconds without pumping its
    message queue, Windows hides it behind a stand-in of its own: the title
    gains "(Not Responding)", the contents wash out, and a click offers to
    close the program. The work is fine — it is the stand-in that reads as a
    crash.

    Most of that is answered by running the long operations on a worker
    thread (:func:`segfix.progress_ui.run_with_progress`), which keeps the
    queue pumped. This covers what is left: a stretch of GIL-bound Python, a
    driver call inside a repaint, or the moments before the worker exists.
    The window then simply holds its last painted frame — which, during a
    load, is the progress bar naming the phase it is on.

    Windows only, and best-effort: failing to disable it costs nothing.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.user32.DisableProcessWindowsGhosting()
    except Exception:
        pass


def _report_uncaught(exc_type, exc, tb) -> None:
    """``sys.excepthook`` for the GUI: say what went wrong, and keep going.

    Without a hook of its own, PyQt answers an exception raised inside a Qt
    slot — a menu action, a button — by aborting the process, so the window
    simply vanishes with nothing on screen to say why (issue #3 was one of
    those). With one, PyQt calls it instead: the traceback still goes to the
    terminal, and the user gets a message rather than a lost session.
    """
    import traceback

    text = "".join(traceback.format_exception(exc_type, exc, tb))
    sys.stderr.write(text)
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    try:
        from qtpy.QtWidgets import QApplication, QMessageBox

        if QApplication.instance() is None:
            return
        box = QMessageBox(
            QMessageBox.Icon.Critical,
            "Something went wrong",
            f"{exc_type.__name__}: {exc}\n\nsegfix is still running, but "
            "save your work before carrying on. Please report this at "
            "github.com/UQ-EORC/Segfix/issues with the details below.",
        )
        box.setDetailedText(text)
        box.exec()
    except Exception:
        pass  # never let the reporter itself be the next crash


def main(argv=None) -> int:
    sys.excepthook = _report_uncaught
    _prefer_discrete_gpu()
    _disable_window_ghosting()
    parser = argparse.ArgumentParser(
        prog="segfix",
        description="GUI tool to fix tree point-cloud instance segmentation.",
    )
    parser.add_argument(
        "--label-field",
        default=None,
        help="Name of the per-point tree/instance ID field (auto-detected if omitted)",
    )
    parser.add_argument(
        "--class-field",
        default=None,
        help="Name of a per-point class field (leaf, wood, ground... as "
             "numbers) to edit alongside the tree IDs; remembered per project",
    )
    parser.add_argument(
        "--point-size", type=float, default=3.0,
        help="Render size of points in screen pixels (default: 3)",
    )
    args = parser.parse_args(argv)

    # vispy + PyQt6 on a Wayland session with the NVIDIA driver hits GLX
    # context-creation failures; force the xcb (X11 / XWayland) platform
    # plugin before any QApplication exists. Respect an explicit override.
    import os

    if sys.platform.startswith("linux"):
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    from . import registry
    from .startup_ui import choose_project

    # "Open Project…" on the menu bar re-execs segfix with this set, so the
    # freshly picked project opens straight away instead of showing the
    # startup dialog a second time. Popped so a later re-exec starts clean.
    preselected = os.environ.pop("SEGFIX_OPEN", None)
    if preselected:
        open_path = preselected
        registry_path = os.environ.pop("SEGFIX_OPEN_REGISTRY", preselected)
        kind = os.environ.pop("SEGFIX_OPEN_KIND", "file")
    else:
        choice = choose_project()
        if choice is None:
            return 0
        open_path, registry_path, kind = choice
    args.cloud = open_path
    registry.add_entry(registry_path, kind=kind)

    return _run_scene(args)


def _read_license() -> str | None:
    """The GPL licence text — from the repo checkout segfix runs from, or
    the installed distribution's metadata. ``None`` if neither is found."""
    from pathlib import Path

    here = Path(__file__).resolve()
    for cand in (here.parents[2] / "LICENSE", here.parent / "LICENSE"):
        try:
            return cand.read_text(encoding="utf-8")
        except OSError:
            continue
    try:
        from importlib.metadata import files

        for f in files("segfix") or []:
            if Path(f.name).name in ("LICENSE", "LICENSE.txt", "COPYING"):
                return f.read_text()
    except Exception:
        pass
    return None


def _about(parent) -> None:
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QMessageBox

    from .icons import app_icon
    from .update import display_version

    box = QMessageBox(parent)
    box.setWindowTitle("About segfix")
    box.setTextFormat(Qt.TextFormat.RichText)
    box.setIconPixmap(app_icon().pixmap(64, 64))
    box.setText(
        f"<b>segfix {display_version()}</b>"
        "<p>GUI tool to fix instance segmentation of tree point clouds.</p>"
        "<p>GPL-3.0-or-later&nbsp;&nbsp;·&nbsp;&nbsp;© 2026 Tim Devereux, The University of Queensland<br>"
        "<a href='https://github.com/UQ-EORC/Segfix'>"
        "github.com/UQ-EORC/Segfix</a></p>"
    )
    licence = _read_license()
    if licence:
        box.setInformativeText("Full licence text under “Show Details”.")
        box.setDetailedText(licence)
    box.exec()


def _open_project(win, panel, scene=None) -> None:
    """Menu "Open Project…": offer to save, pick another project in the
    startup dialog, then re-exec segfix on it. Re-exec rather than an
    in-place swap so the catalog, docks and GL context all rebuild cleanly.

    ``scene`` (the :class:`~segfix.scene_ui.SceneController`) is what knows
    whether the session has unsaved edits: the loaded tree's undo stack only
    covers the tree on screen, not the ones visited before it.
    """
    import os

    from qtpy.QtWidgets import QMessageBox

    from .startup_ui import choose_project

    # can_undo is a property, not a method: calling it raised TypeError here,
    # which PyQt turns into an abort, and Open Project killed the window on
    # every use since it was added (issue #3).
    unsaved = (scene.has_unsaved_edits() if scene is not None
               else panel.c.cloud.can_undo)
    if unsaved:
        answer = QMessageBox.question(
            win,
            "Open another project",
            "Save changes to the current project first?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return
        if answer == QMessageBox.StandardButton.Save:
            panel.on_save()

    choice = choose_project()
    if choice is None:
        return
    open_path, registry_path, kind = choice
    # main() records it in the registry after the relaunch.
    os.environ["SEGFIX_OPEN"] = open_path
    os.environ["SEGFIX_OPEN_REGISTRY"] = registry_path
    os.environ["SEGFIX_OPEN_KIND"] = kind
    _relaunch()


#: What a relaunched interpreter runs: -c, so it doesn't matter how segfix
#: was started this time (console script, -m, an IDE).
_RELAUNCH_CODE = "import sys; from segfix.app import main; sys.exit(main())"


def relaunch_command() -> list[str]:
    """The command that starts segfix again the way this one was started,
    carrying the user's CLI flags across.

    A frozen build (the Windows installer or portable zip) *is* segfix:
    ``sys.executable`` is ``segfix.exe``, which takes segfix's own flags and
    would choke on ``-c``. Anything else is a Python interpreter, told to
    run segfix's main.
    """
    from .update import is_frozen

    if is_frozen():
        return [sys.executable, *sys.argv[1:]]
    return [sys.executable, "-c", _RELAUNCH_CODE, *sys.argv[1:]]


def _relaunch() -> None:
    """Replace this segfix with a fresh one, for Open Project.

    On Linux and macOS that is ``os.execv``: the same process, rebuilt from
    scratch. Not on Windows, where ``exec`` never replaced the process in
    the first place — it starts a new one and exits the old — and, worse,
    passes the arguments unquoted: the ``-c`` code arrived as ``-c import``
    and died with a SyntaxError, closing the window with nothing reopened
    (issue #3). There a ``Popen``, which quotes each argument properly,
    starts the new segfix and this one quits in the ordinary way.
    """
    import os

    cmd = relaunch_command()
    if sys.platform == "win32":
        import subprocess

        from qtpy.QtWidgets import QApplication

        subprocess.Popen(cmd)
        QApplication.instance().quit()
        return
    os.execv(cmd[0], cmd)


def _export_trees(win, panel, catalog) -> None:
    """Menu "Export Trees…": write every tree in the cloud as its own file,
    at full resolution and in the cloud's own format (see :mod:`export`)."""
    import os

    from qtpy.QtWidgets import QFileDialog, QMessageBox

    from . import export
    from .progress_ui import run_with_progress

    if catalog.has_unsaved_edits():
        answer = QMessageBox.question(
            win,
            "Export trees",
            "Trees are exported as the saved file has them, so edits you "
            "haven't saved won't be in them.\n\nSave them first?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return
        if answer == QMessageBox.StandardButton.Save:
            panel.on_save()

    # Plenty of people need a few trees per plot, not all of them (issue
    # #3): offer just the ones marked Done, when there are any to offer.
    from .scene_ui import read_done

    done = read_done(catalog.path) & set(catalog.records)
    only = None
    if done:
        box = QMessageBox(win)
        box.setWindowTitle("Export trees")
        box.setText(f"{len(done)} of {len(catalog.records)} trees are marked "
                    "Done. Which do you want to export?")
        all_btn = box.addButton(f"All {len(catalog.records)} trees",
                                QMessageBox.ButtonRole.AcceptRole)
        done_btn = box.addButton(f"Only the {len(done)} marked Done",
                                 QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(all_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is done_btn:
            only = done
        elif clicked is not all_btn:
            return
    count = len(only) if only is not None else len(catalog.records)

    out_dir = QFileDialog.getExistingDirectory(
        win, "Export the trees into this folder", os.path.dirname(catalog.path)
    )
    if not out_dir:
        return
    try:
        written = run_with_progress(
            win, "Exporting trees",
            f"{count} trees from {os.path.basename(catalog.path)}",
            lambda report, ask: export.export_trees(
                catalog, out_dir, progress=report, only=only
            ),
        )
    except Exception as exc:
        QMessageBox.critical(win, "Export failed", str(exc))
        return
    panel.c.view.status = f"Exported {len(written)} tree(s) to {out_dir}"
    QMessageBox.information(
        win, "Export finished",
        f"Wrote {len(written)} file(s) to\n{out_dir}",
    )


def _build_menus(win, panel, catalog=None, scene=None) -> None:
    """Window menu bar: File (open/save the project), Edit (undo/redo — the
    former "Session" panel box), Preferences (colour theme), Help (about)."""
    from qtpy.QtGui import QActionGroup
    from qtpy.QtWidgets import QApplication

    from . import inventory_ui, theme

    bar = win.menuBar()

    file_menu = bar.addMenu("&File")
    open_act = file_menu.addAction("Open Project…")
    open_act.setShortcut("Ctrl+O")
    open_act.triggered.connect(lambda: _open_project(win, panel, scene))
    save_act = file_menu.addAction("Save Project")
    save_act.setShortcut("Ctrl+S")
    save_act.triggered.connect(panel.on_save)
    if catalog is not None:
        export_act = file_menu.addAction("Export Trees…")
        export_act.triggered.connect(lambda: _export_trees(win, panel, catalog))

    edit_menu = bar.addMenu("&Edit")
    undo_act = edit_menu.addAction("Undo")
    undo_act.setShortcut("Ctrl+Z")
    undo_act.triggered.connect(panel.on_undo)
    redo_act = edit_menu.addAction("Redo")
    redo_act.setShortcut("Ctrl+Shift+Z")
    redo_act.triggered.connect(panel.on_redo)
    if catalog is not None:
        edit_menu.addSeparator()
        classes_act = edit_menu.addAction("Point Classes…")
        classes_act.triggered.connect(panel._on_setup_classes)

    # Inventory: a field stem map drawn over the cloud, and which measured
    # tree each segmented one is (see segfix.inventory).
    inv_menu = bar.addMenu("&Inventory")
    load_stems_act = inv_menu.addAction("Load Stem Map…")
    load_stems_act.triggered.connect(
        lambda: inventory_ui.load_stem_map(win, panel, catalog)
    )
    align_act = inv_menu.addAction("Align to Cloud…")
    align_act.triggered.connect(
        lambda: inventory_ui.align_stem_map(win, panel, catalog)
    )
    inv_menu.addSeparator()
    matches_act = inv_menu.addAction("Export Matches…")
    matches_act.triggered.connect(
        lambda: inventory_ui.export_matches(win, panel, catalog)
    )
    clear_stems_act = inv_menu.addAction("Clear Stem Map")
    clear_stems_act.triggered.connect(
        lambda: inventory_ui.clear_stem_map(win, panel)
    )

    pref_menu = bar.addMenu("&Preferences")
    class_place = pref_menu.addAction("Point class box in the top bar")
    class_place.setCheckable(True)
    class_place.setToolTip(
        "Move the Point class box out of the 3D view and into the top bar"
    )
    class_place.toggled.connect(panel._set_class_in_top_bar)
    theme_menu = pref_menu.addMenu("Theme")
    theme_group = QActionGroup(win)
    theme_group.setExclusive(True)
    for label, mode in (("Light", "light"), ("Dark", "dark")):
        act = theme_menu.addAction(label)
        act.setCheckable(True)
        act.setChecked(theme.current() == mode)
        theme_group.addAction(act)
        act.triggered.connect(
            lambda _checked, m=mode: theme.set_mode(QApplication.instance(), m)
        )

    help_menu = bar.addMenu("&Help")
    about_act = help_menu.addAction("About segfix")
    about_act.triggered.connect(lambda: _about(win))


def _class_names_from_settings(saved) -> dict[int, str]:
    """The ``{value: name}`` the manifest keeps as ``{"value": name}``."""
    out = {}
    for key, name in (saved or {}).items():
        try:
            out[int(key)] = str(name)
        except (TypeError, ValueError):
            continue
    return out


def _remember_classes(catalog, field, names) -> None:
    from . import workspace

    workspace.remember(
        catalog.path, class_field=field,
        class_names={str(k): v for k, v in sorted(names.items())},
    )


def _use_classes(panel, catalog, names) -> None:
    """Point the editing controller at the catalog's class field + names."""
    import numpy as np

    seg = panel.c
    seg.class_field = catalog.class_field
    seg.class_range = catalog.class_range() if catalog.class_field else None
    if catalog.class_field and catalog.classes is not None:
        # Every value in the file gets a name, even one the project never
        # named, so no class is left without a button.
        for value in np.unique(catalog.classes).tolist():
            names.setdefault(int(value), f"Class {value}")
    seg.class_names = names if catalog.class_field else {}
    if not catalog.class_field and panel.class_color_cb.isChecked():
        panel.class_color_cb.setChecked(False)


def _setup_classes(win, panel, catalog, scene) -> None:
    """Edit ▸ Point Classes… / the Point class box's Set up…: pick the class
    field and name its values, then reload the scene if the field changed."""
    from qtpy.QtWidgets import QMessageBox

    from .classes_ui import prompt_class_setup
    from .viewer import busy

    choice = prompt_class_setup(
        win, catalog, catalog.class_field, panel.c.class_names
    )
    if choice is None:
        return
    field, names = choice
    if field != catalog.class_field:
        scene._flush()  # so unsaved class edits on screen count too
        if catalog.has_unsaved_class_edits():
            answer = QMessageBox.question(
                win, "Change the class field",
                "Save the class edits to "
                f"{catalog.class_field} before switching field?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if answer != QMessageBox.StandardButton.Save:
                return
            panel.on_save()
            if catalog.has_unsaved_class_edits():
                return  # the save failed and said so
        busy(panel.c.view, f"Reading {field}…" if field else "Dropping classes…")
        try:
            catalog.set_class_field(field)
        except ValueError as exc:
            QMessageBox.warning(win, "Can't use that field", str(exc))
            return
        _use_classes(panel, catalog, names)
        _remember_classes(catalog, field, panel.c.class_names)
        msg = scene.reload_current()
        panel._rebuild_class_buttons()
        panel.c.view.status = msg or (
            f"Point classes from {field}" if field else "Point classes off"
        )
        return
    _use_classes(panel, catalog, names)
    _remember_classes(catalog, field, panel.c.class_names)
    panel._rebuild_class_buttons()
    panel._apply_transparency()
    panel.c.view.status = "Point classes updated"


def _combined_panel(*widgets):
    """Stack scene mode's "All Trees" table above the shared segfix editing
    panel in one right-hand dock (rather than a separate left dock).

    A vertical splitter, so the divider is draggable; it starts at roughly
    25 % top / 75 % bottom and keeps that ratio as the dock resizes.
    """
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QSplitter

    split = QSplitter(Qt.Orientation.Vertical)
    for w in widgets:
        split.addWidget(w)
    split.setChildrenCollapsible(False)
    for i in range(split.count()):
        split.setStretchFactor(i, 1 if i == 0 else 3)
    split.setSizes([1000] + [3000] * (split.count() - 1))
    # Floor so the tables get room to breathe; the dock edge stays draggable.
    split.setMinimumWidth(360)
    return split


def _bare_dock(widget, title: str):
    """A ``QDockWidget`` holding ``widget`` with no title bar and no
    float/close buttons — a plain fixed panel, like napari's stripped docks."""
    from qtpy.QtWidgets import QDockWidget, QWidget

    dock = QDockWidget(title)
    dock.setObjectName(title)
    dock.setTitleBarWidget(QWidget())  # empty widget → no visible title bar
    dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
    dock.setWidget(widget)
    return dock


#: How many draws gpu_renderer_info() gets to answer before the status bar
#: settles on "unknown" — see _report_gpu.
_GPU_READ_ATTEMPTS = 30


def _run_scene(args) -> int:
    """Default (only) mode: a tree table where picking a row loads that tree
    plus its spatial neighbours into the 3D view, instead of the whole cloud.
    """
    import os

    import numpy as np
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox

    from . import theme
    from .cloudview import CloudView
    from .density_ui import prompt_downsample
    from .icons import app_icon
    from .model import PointCloud
    from .overlays import ScaleBarOverlay
    from .progress_ui import run_with_progress
    from .scene_ui import SceneController, SceneWidget
    from .shift_ui import prompt_global_shift
    from .treecatalog import open_catalog
    from .update import display_version
    from .viewer import busy, gpu_renderer_info, gpu_status
    from .widgets import SegFixController, SegFixWidget, bind_shortcuts

    app = QApplication.instance() or QApplication(sys.argv)
    app.setWindowIcon(app_icon())
    theme.apply(app)  # saved light/dark palette (Preferences ▸ Theme)

    win = QMainWindow()

    def _refresh_title() -> None:
        win.setWindowTitle(f"segfix {display_version()} - {args.cloud}")

    _refresh_title()
    # Re-read the checkout version whenever the window is re-activated, so an
    # external `git pull` / reinstall (or the startup dialog's own updater)
    # shows up in the title without restarting.
    app.focusChanged.connect(
        lambda _old, now: _refresh_title()
        if now is not None and (now is win or win.isAncestorOf(now))
        else None
    )
    win.setWindowIcon(app_icon())

    view = CloudView()
    win.setCentralWidget(view.native)
    # Canvas clear colour follows the theme (points stay readable on either).
    theme.subscribe(lambda mode: view.set_background(theme.CANVAS_BG[mode]))
    # Scale bar + orientation axes over the canvas' bottom-left corner. Held
    # on the window so it outlives this function; it also parents to the
    # canvas widget, which keeps it alive regardless.
    win._scale_overlay = ScaleBarOverlay(view)

    status = win.statusBar()
    view.on_status = status.showMessage
    gpu_label = QLabel("GPU: detecting...")
    gpu_label.setStyleSheet("color: gray; padding: 0 6px;")
    status.addPermanentWidget(gpu_label)

    # Draws left to ask on before settling for "unknown". The first draw is
    # usually enough; this is headroom for a context that needs a moment.
    gpu_tries = [_GPU_READ_ATTEMPTS]

    def _report_gpu(event=None) -> None:
        # No GL context is current until the canvas has actually drawn once
        # (it's built with show=False, and .update() only queues a repaint —
        # see busy()'s docstring in viewer.py) -- reading it any earlier
        # always reports "unknown", regardless of platform or GPU.
        #
        # Drawing once is not quite the same as the context being ready to
        # answer, though: after the machine wakes from sleep, a driver reset
        # or a hybrid-graphics switch, the first draw can come back with
        # nothing. This used to unhook itself on that first draw whatever it
        # got, so a single unlucky moment left the label reading "unknown"
        # for the rest of the session. Keep asking on later draws instead,
        # and only give up once the budget is gone.
        gpu_tries[0] -= 1
        text, done = gpu_status(gpu_renderer_info(), gpu_tries[0])
        if text is not None:
            gpu_label.setText(text)
        if done:
            view.canvas.events.draw.disconnect(_report_gpu)

    view.canvas.events.draw.connect(_report_gpu)

    empty = PointCloud(
        coords=np.empty((0, 3), np.float32), labels=np.empty(0, np.int32)
    )
    view.load_cloud(empty, point_size=args.point_size)

    win.showMaximized()
    busy(view, f"Scanning trees in {args.cloud}…")

    # The load runs on a worker thread (see run_with_progress): on the GUI
    # thread each phase is one uninterrupted numpy call, and Windows calls a
    # window that hasn't pumped its message queue for five seconds "(Not
    # Responding)". Both prompts are raised from inside the load, so they go
    # back across to the GUI thread through `ask` — which also steps the bar
    # out of their way, since one frozen behind a question reads as a hang.
    def _open(report, ask):
        return open_catalog(
            args.cloud,
            label_field=args.label_field,
            shift_prompt=lambda mins, maxs, suggested:
                ask(prompt_global_shift, win, mins, maxs, suggested),
            density_prompt=lambda spacing, n_points, suggested, kept_fraction:
                ask(prompt_downsample, win, spacing, n_points, suggested,
                    kept_fraction),
            progress=report,
        )

    from . import workspace

    saved = workspace.settings(args.cloud)
    class_field = args.class_field or saved.get("class_field")
    class_problem: list[str] = []

    def _open_with_classes(report, ask):
        catalog = _open(report, ask)
        if class_field:
            # A field the file no longer has shouldn't stop it opening:
            # open without classes and say why.
            try:
                catalog.set_class_field(class_field)
            except ValueError as exc:
                class_problem.append(str(exc))
        return catalog

    try:
        catalog = run_with_progress(
            win, "Opening cloud", os.path.basename(args.cloud), _open_with_classes
        )
    except Exception as exc:
        # A wrong/corrupt file used to raise this far with the window already
        # shown maximized and app.exec() not yet reached — no event loop was
        # running to keep it responsive, so it looked exactly like a freeze
        # even though the process was about to exit with a traceback. Fail
        # visibly and exit cleanly instead.
        win.close()
        QMessageBox.critical(
            None, "Can't open this cloud",
            f"{args.cloud}\n\ncould not be opened:\n\n{exc}\n\n"
            "segfix reads binary PLY (RGB-segmented or with a label field) "
            "and uncompressed LAS with a treeID column.",
        )
        return 1

    seg = SegFixController(view, empty)
    panel = SegFixWidget(seg)
    scene_ctrl = SceneController(view, catalog, seg, point_size=args.point_size)
    scene_panel = SceneWidget(scene_ctrl)
    panel.on_done_changed = scene_panel.refresh

    top_dock = _bare_dock(panel.top_bar, "view")
    win.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, top_dock)
    right_dock = _bare_dock(_combined_panel(scene_panel, panel), "segfix")
    win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, right_dock)
    # Give the top-right corner to the right dock so it reaches y=0 instead
    # of being pushed down below the top strip.
    win.setCorner(
        Qt.Corner.TopRightCorner, Qt.DockWidgetArea.RightDockWidgetArea
    )
    win.resizeDocks([right_dock], [440], Qt.Orientation.Horizontal)
    panel.size_spin.setValue(args.point_size)
    _build_menus(win, panel, catalog, scene_ctrl)
    bind_shortcuts(win, panel)
    names = (_class_names_from_settings(saved.get("class_names"))
             if saved.get("class_field") == catalog.class_field else {})
    _use_classes(panel, catalog, names)
    if catalog.class_field and catalog.class_field != saved.get("class_field"):
        _remember_classes(catalog, catalog.class_field, panel.c.class_names)
    panel.on_setup_classes = lambda: _setup_classes(win, panel, catalog, scene_ctrl)
    panel.on_class_names_changed = (
        lambda names: _remember_classes(catalog, catalog.class_field, names)
    )
    panel._rebuild_class_buttons()

    decimated = (
        f" Downsampled to {catalog.voxel_size * 100:g} cm for editing "
        f"({catalog.working_count:,} of {catalog.count:,} points); edits are "
        "interpolated back to every point on save."
        if catalog.is_decimated else ""
    )
    view.status = (
        f"{len(catalog.records)} trees in {args.cloud}. "
        f"Double-click a tree to load it with neighbours.{decimated}"
    )
    if class_problem:
        view.status = f"No point classes: {class_problem[0]}"
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
