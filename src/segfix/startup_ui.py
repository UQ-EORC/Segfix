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

"""Startup picker: reopen a recent project, or import a point cloud file as
a new one.

``segfix`` takes no path argument, so this dialog is how every session picks
what to open — there is no way to skip it. Importing copies the chosen file
into a fresh workspace folder (:mod:`workspace`) and opens that copy, so the
source file is never the thing being edited.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from . import io, registry, update, workspace

_ICONS = {"workspace": "🗂", "file": "📄"}


class StartupDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("segfix")
        self.resize(560, 380)
        # What main() should open, what should be recorded in the registry,
        # and which kind it is — see create_workspace()'s docstring for why
        # the first two differ for a new workspace (open the data file
        # inside it, but register the workspace folder itself).
        self.open_path: str | None = None
        self.registry_path: str | None = None
        self.kind: str | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Recent projects:"))

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.itemDoubleClicked.connect(self._on_choose)
        # Return on the list opens it too, wherever the platform sends the
        # key — the list has focus at startup, so Enter is the whole
        # interaction for "reopen what I was working on".
        self.list.itemActivated.connect(self._on_choose)
        layout.addWidget(self.list, stretch=1)
        self._populate()

        hint = QLabel(
            "Double-click a recent project to reopen it, or start a new one "
            "from a point cloud file - or from a set of per-tree files, "
            "merged into one on the way in. Either way a private copy is "
            "made in a new project folder, and edits are saved to that copy, "
            "never the original."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        layout.addWidget(hint)

        # Update banner: hidden until the background check (below) finds a
        # newer PyPI release (installed copies) or new upstream commits (git
        # checkouts) — see segfix.update. Stays hidden when offline or when
        # there's nothing newer.
        update_row = QHBoxLayout()
        self.update_label = QLabel("")
        self.update_label.setStyleSheet("color: #b8860b;")
        self.update_label.hide()
        update_row.addWidget(self.update_label, stretch=1)
        self.update_btn = QPushButton("Update…")
        self.update_btn.hide()
        self.update_btn.clicked.connect(self._apply_update)
        update_row.addWidget(self.update_btn)
        layout.addLayout(update_row)
        self._update_status: update.UpdateStatus | None = None
        self._update_pool = ThreadPoolExecutor(max_workers=1)
        self._update_future = self._update_pool.submit(update.check_for_update)
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._poll_update_check)
        self._update_timer.start(300)

        # The two ways to start a project sit together on the left, named
        # as the same kind of thing: one cloud in, or a set of per-tree
        # clouds merged on the way in. "Per-tree Files…" beside "New
        # Project…" read as an action and a noun, which is a worse pair than
        # it looks — it is the only button here whose label doesn't say what
        # pressing it does.
        row = QHBoxLayout()
        self.new_btn = QPushButton("New Project…")
        self.new_btn.setToolTip(
            "Import one point cloud as a new project - a private copy is "
            "made, and the original is never edited"
        )
        self.new_btn.clicked.connect(self._new_project)
        row.addWidget(self.new_btn)
        self.per_tree_btn = QPushButton("New from Per-tree Files…")
        self.per_tree_btn.setToolTip(
            "Import a folder of one-cloud-per-tree files (raysplit output "
            "and the like) as a single project, keeping each file's tree ID"
        )
        self.per_tree_btn.clicked.connect(self._new_from_per_tree_files)
        row.addWidget(self.per_tree_btn)
        row.addStretch()
        self.open_btn = QPushButton("Open")
        self.open_btn.setToolTip(
            "Reopen the selected project (double-click it, or press Enter)"
        )
        self.open_btn.clicked.connect(self._on_choose)
        row.addWidget(self.open_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        row.addWidget(self.cancel_btn)
        layout.addLayout(row)

        # Enter opens the preselected project — the common case by far, and
        # the top row is already the most recent one. Only the button Enter
        # should press claims autoDefault, or whichever of them last had
        # focus would answer the key instead.
        for button in (self.new_btn, self.per_tree_btn, self.cancel_btn,
                       self.update_btn):
            button.setAutoDefault(False)
        has_recent = self.list.count() > 0
        opener = self.open_btn if has_recent else self.new_btn
        self.open_btn.setEnabled(has_recent)
        opener.setAutoDefault(True)
        opener.setDefault(True)
        # Focus goes to the list, so the arrow keys pick a different project
        # before Enter opens it; with nothing to reopen, to the one button
        # that can do anything.
        (self.list if has_recent else self.new_btn).setFocus()

    def _populate(self) -> None:
        # registry.load_registry() is already most-recently-opened first, so
        # the newest project lands at the top (and is preselected below).
        self.list.clear()
        for entry in registry.load_registry():
            mark = _ICONS.get(entry["type"], "📄")
            age = registry.describe_age(entry.get("last_opened", ""))
            label = f"{mark}  {entry['path']}"
            if age:
                label += f"   ({age})"
            item = QListWidgetItem(label)
            item.setToolTip(
                f"Last opened {entry.get('last_opened', 'unknown')}"
            )
            item.setData(Qt.UserRole, entry)
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)

    def _new_project(self) -> None:
        source, _ = QFileDialog.getOpenFileName(
            self, "Import point cloud", str(Path.home()),
            "Point clouds (*.ply *.las *.laz);;Binary PLY (*.ply);;"
            "LAS / LAZ (*.las *.laz)",
        )
        if not source:
            return

        # Reject a mismatched/bogus file immediately, before it's copied into
        # a workspace: a "wrong format" file — its content doesn't match the
        # extension it was picked under — used to only surface much later,
        # inside open_catalog(), sometimes after a slow scan of the file's
        # binary content that looked like a frozen app.
        ext = Path(source).suffix.lower()
        expected = "ply" if ext == ".ply" else "las" if ext in (".las", ".laz") else None
        if expected is None or io.sniff_format(source) != expected:
            QMessageBox.critical(
                self, "Wrong format",
                f"{source}\n\ndoesn't look like a point cloud segfix can "
                "read: its content doesn't match a binary PLY or LAS/LAZ "
                "header. Check you picked the right file.",
            )
            return

        parent, _ = str(Path.home()), None
        parent = QFileDialog.getExistingDirectory(
            self, "Choose where to create the project folder", parent,
        )
        if not parent:
            return

        stem = Path(source).stem
        candidate = Path(parent) / stem
        suffix = 1
        while candidate.exists() and any(candidate.iterdir()):
            suffix += 1
            candidate = Path(parent) / f"{stem}_{suffix}"

        # Importing is a multi-gigabyte copy, or a whole LAZ decompression,
        # before open_catalog() and its own bar are anywhere in sight. Behind
        # a bare call that was the longest unexplained pause in the app: the
        # dialog simply stopped repainting until a project appeared.
        from .progress_ui import run_with_progress

        try:
            data_path = run_with_progress(
                self, "Importing cloud", Path(source).name,
                lambda report, ask: workspace.create_workspace(
                    source, candidate, report=report
                ),
            )
        except Exception as exc:  # OSError, or laspy failing on a bad LAS/LAZ
            QMessageBox.critical(self, "Import failed", str(exc))
            return

        self.open_path = str(data_path)
        self.registry_path = str(candidate)
        self.kind = "workspace"
        self.accept()

    def _new_from_per_tree_files(self) -> None:
        """Import one-cloud-per-tree output as a single project.

        The same shape as :meth:`_new_project` — pick the input, pick where
        the project folder goes, copy under a bar — except the "copy" is a
        merge of many files into one, and which tree each file is has to be
        settled first (:mod:`multi_import_ui`). The format check lives in
        the dialog, which can't offer Import until the whole set is sound.
        """
        from .multi_import_ui import choose_per_tree_files

        sources = choose_per_tree_files(self)
        if not sources:
            return

        parent = QFileDialog.getExistingDirectory(
            self, "Choose where to create the project folder", str(Path.home()),
        )
        if not parent:
            return

        # Named after the folder the trees came out of, which is the only
        # name the set has — the files themselves are all called tree_<n>.
        stem = Path(sources[0].path).parent.name or "trees"
        candidate = Path(parent) / stem
        suffix = 1
        while candidate.exists() and any(candidate.iterdir()):
            suffix += 1
            candidate = Path(parent) / f"{stem}_{suffix}"

        from .progress_ui import run_with_progress

        try:
            data_path = run_with_progress(
                self, "Merging per-tree files", f"{len(sources)} files",
                lambda report, ask: workspace.create_from_files(
                    sources, candidate, report=report
                ),
            )
        except Exception as exc:  # OSError, laspy, or a set that won't merge
            QMessageBox.critical(self, "Import failed", str(exc))
            return

        self.open_path = str(data_path)
        self.registry_path = str(candidate)
        self.kind = "workspace"
        self.accept()

    def _on_choose(self, *_args) -> None:
        item = self.list.currentItem()
        if item is None:
            return
        entry = item.data(Qt.UserRole)
        kind = entry["type"]
        if kind == "workspace":
            try:
                self.open_path = str(workspace.data_file(entry["path"]))
            except (OSError, KeyError, ValueError) as exc:
                QMessageBox.critical(
                    self, "Can't open project",
                    f"{entry['path']} doesn't look like a valid project "
                    f"folder anymore: {exc}",
                )
                return
        else:
            self.open_path = entry["path"]
        self.registry_path = entry["path"]
        self.kind = kind
        self.accept()

    def _poll_update_check(self) -> None:
        if not self._update_future.done():
            return
        self._update_timer.stop()
        status = self._update_future.result()
        if status is None:
            return
        self._update_status = status
        self.update_label.setText(status.describe())
        self.update_label.show()
        self.update_btn.show()

    def _apply_update(self) -> None:
        if update.must_close_to_update():
            self._update_after_exit()
            return
        # Blocking is deliberate: this only runs after an explicit click, a
        # pip upgrade (or `git pull` + reinstall) is normally a few seconds,
        # and there's no meaningful "cancel an install halfway through" to
        # offer instead.
        self.update_btn.setEnabled(False)
        self.update_label.setText("Updating…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            update.apply_update(self._update_status)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            detail = exc.stderr if hasattr(exc, "stderr") and exc.stderr else str(exc)
            QMessageBox.critical(self, "Update failed", detail)
            self.update_btn.setEnabled(True)
            self.update_label.setText(self._update_status.describe())
            return
        QApplication.restoreOverrideCursor()
        self.update_label.setText("Updated. Restart segfix to use it.")
        self.update_btn.hide()
        QMessageBox.information(
            self, "Updated",
            "segfix has been updated. Restart it to use the new version.",
        )

    def _update_after_exit(self) -> None:
        """Windows: segfix can't be replaced while it runs, so hand the
        install to a helper window and close segfix (see
        :func:`update.must_close_to_update`)."""
        answer = QMessageBox.question(
            self, "Update segfix",
            "Windows can't replace segfix while it is running, so segfix "
            "will close and the update will install in a separate window.\n\n"
            "Start segfix again once that window says it's done.\n\n"
            "Close segfix and update now?",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Ok,
        )
        if answer != QMessageBox.StandardButton.Ok:
            return
        try:
            update.schedule_update_after_exit(self._update_status)
        except Exception as exc:
            QMessageBox.critical(self, "Update failed", str(exc))
            return
        app = QApplication.instance()
        self.reject()
        # Also ends the main window's event loop when this dialog was opened
        # from File > Open Project..., so segfix really exits.
        app.quit()

    def done(self, result) -> None:
        # Overridden rather than closeEvent(): QDialog.accept()/reject() call
        # done() directly without going through closeEvent, so that's the
        # one hook that covers every way this dialog can close.
        self._update_timer.stop()
        self._update_pool.shutdown(wait=False, cancel_futures=True)
        super().done(result)


_app = None  # kept alive for the process's lifetime once created here — a
# QApplication with no surviving Python reference gets garbage-collected
# immediately (PyQt/PySide tear down the underlying app with it), which
# crashes the very next QWidget construction.


def choose_project() -> tuple[str, str, str] | None:
    """Show the startup dialog. Returns ``(open_path, registry_path, kind)``
    — the path to actually open, the path to remember in the registry
    (differs for a new workspace: open its data file, register its folder),
    and ``kind`` (see :func:`registry.add_entry`) — or ``None`` if the user
    cancelled without choosing anything.

    Creates the process-wide ``QApplication`` (reused by the main window
    afterwards). ``app.main`` has already pinned ``QT_QPA_PLATFORM`` by the
    time this runs.
    """
    global _app
    import sys

    from qtpy.QtWidgets import QApplication

    from . import theme
    from .icons import app_icon

    _app = QApplication.instance() or QApplication(sys.argv)
    _app.setWindowIcon(app_icon())
    theme.apply(_app)  # so the picker matches the saved light/dark choice
    dialog = StartupDialog()
    if dialog.exec() == QDialog.Accepted and dialog.open_path:
        return dialog.open_path, dialog.registry_path, dialog.kind
    return None
