"""Record the segfix walkthrough.

    DISPLAY=:0 QT_QPA_PLATFORM=xcb XDG_CONFIG_HOME=<demo>/config \
        python make_video.py <story> <out.mp4>

<story> is a function name in storyboard.py (or "proto", defined here).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from qtpy.QtCore import QPoint, Qt  # noqa: E402
from qtpy.QtWidgets import QApplication, QMainWindow  # noqa: E402

from recorder import FPS, H, W, Recorder, Runner, find, outline_around  # noqa: E402


def patch_segfix(runner_ref):
    """Recording-only changes to segfix's runtime behaviour."""
    # Fixed 1920x1080 client area instead of maximizing onto a 3440 screen.
    def _show(self):
        self.setFixedSize(W, H)
        self.show()
    QMainWindow.showMaximized = _show

    # Each load/save phase blocks the event loop, so no ticks run during it:
    # hold each phase's picture for a moment so it can actually be read.
    from segfix import progress_ui
    orig = progress_ui.ProgressWindow.report

    def report(self, message, fraction):
        orig(self, message, fraction)
        r = runner_ref[0]
        if r is not None and r.recording:
            r.hold(0.8)
    progress_ui.ProgressWindow.report = report

    # Display-only: show the demo project under a tidy path instead of the
    # long scratch directory it really lives in. Files are untouched.
    real = os.path.join(os.path.dirname(os.environ["XDG_CONFIG_HOME"]), "projects")
    shown = "~/segfix-demo"

    from qtpy.QtWidgets import QStatusBar
    orig_msg = QStatusBar.showMessage

    def show_message(self, text, *args):
        orig_msg(self, str(text).replace(real, shown), *args)
    QStatusBar.showMessage = show_message

    from segfix import startup_ui
    orig_pop = startup_ui.StartupDialog._populate

    def populate(self):
        orig_pop(self)
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setText(item.text().replace(real, shown))
    startup_ui.StartupDialog._populate = populate


def main_ready():
    from segfix.scene_ui import SceneWidget
    win = find(QMainWindow)
    return win is not None and bool(win.findChildren(SceneWidget))


def handles():
    from segfix.scene_ui import SceneWidget
    from segfix.widgets import SegFixWidget
    win = find(QMainWindow)
    panel = win.findChildren(SegFixWidget)[0]
    scene = win.findChildren(SceneWidget)[0]
    return win, panel, scene, panel.c.view


def table_row_for(table, label, col=1):
    for r in range(table.rowCount()):
        item = table.item(r, col)
        if item is not None and item.data(Qt.ItemDataRole.DisplayRole) == label:
            return r
    return None


# -- a short storyboard to prove the pipeline end to end -----------------------
def proto(r: Runner):
    from segfix.density_ui import DownsampleDialog
    from segfix.shift_ui import GlobalShiftDialog
    from segfix.startup_ui import StartupDialog

    rec = r.rec
    yield r.until(lambda: find(StartupDialog), 30, "startup dialog")
    r.recording = True
    rec.set_caption("Run segfix and the startup dialog lists your recent projects.",
                    chapter="Opening a project")
    yield 2.5
    dlg = find(StartupDialog)
    item = dlg.list.item(0)
    yield from r.click(r.wpt(dlg.list.viewport(), dlg.list.visualItemRect(item).center()))
    yield 0.4
    dlg._on_choose()

    yield r.until(lambda: find(GlobalShiftDialog), 90, "shift prompt")
    rec.set_caption("This cloud is georeferenced (UTM), so segfix offers a global shift.",
                    chapter="Large coordinates")
    yield 3.0
    d = find(GlobalShiftDialog)
    yield from r.click(r.wpt(d.apply_btn))
    yield 0.3
    d.apply_btn.click()

    yield r.until(lambda: find(DownsampleDialog), 90, "downsample prompt")
    rec.set_caption("Points are under 2 cm apart, so it offers to downsample for editing.",
                    chapter="Dense clouds")
    yield 3.0
    d = find(DownsampleDialog)
    yield from r.click(r.wpt(d.downsample_btn))
    yield 0.3
    d.downsample_btn.click()

    yield r.until(main_ready, 120, "main window")
    win, panel, scene, view = handles()
    rec.set_caption("Double-click a tree to load it with its neighbours.",
                    chapter="The tree table")
    yield 1.5
    row = table_row_for(scene.table, 4)
    item = scene.table.item(row, 1)
    yield from r.click(r.wpt(scene.table.viewport(), scene.table.visualItemRect(item).center()))
    scene.table.selectRow(row)
    yield from r.smooth(view, scene.on_load_tree, 0.9)
    yield 1.5

    rec.set_caption("Press L and drag around the points to select them.",
                    chapter="Lasso")
    yield from r.press("L", lambda: panel.lasso_btn.setChecked(True))
    labels = panel.c.cloud.labels
    idx = np.flatnonzero(labels == 5)
    xy, valid = view.project_to_canvas(view.coords[idx])
    yield from r.lasso(panel.c.lasso, outline_around(xy[valid]), secs=2.0)
    yield 2.5
    print(f"frames={rec.frames}  grab ms: median={np.median(rec.grab_ms):.1f} "
          f"p95={np.percentile(rec.grab_ms, 95):.1f}", flush=True)


def main():
    story_name, out = sys.argv[1], sys.argv[2]
    if story_name == "proto":
        story = proto
    else:
        import storyboard
        story = getattr(storyboard, story_name)

    runner_ref = [None]
    patch_segfix(runner_ref)
    app = QApplication(sys.argv[:1])
    rec = Recorder(out, font_family=os.environ.get("DEMO_FONT", "Noto Sans"))
    runner = Runner(rec, story)
    runner_ref[0] = runner

    from segfix.app import main as segfix_main
    try:
        segfix_main([])
    finally:
        if not runner._done:
            runner.finish()
    if runner.error:
        print(runner.error, file=sys.stderr, flush=True)
        sys.exit(1)
    print(f"wrote {out}: {rec.frames} frames ({rec.frames / FPS:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
