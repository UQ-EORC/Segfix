"""The full walkthrough: every tool in the README, on the example cloud
(make_sample.py --spacing 0.01 --origin 204300 7223250 12), ending with every
tree in the table fixed and marked done.

Scene (IDs fixed by make_sample.py): 1-3 a touching group of 13 m, 7 m and
9.5 m trees, where part of tree 2's crown carries tree 1's ID; 4-6 stand
alone; 7/8 one tree split in two, with small 9 beside it; 10 two trees
sharing an ID; 11 a tree whose ID swallowed a patch of ground; 12 a bush
detected as a tree. Unassigned ground under all of it.

Pacing rules, applied everywhere:
* a caption that describes an action appears *before* it, and stays up for
  its full reading time before anything moves;
* every action is followed by a settle pause, so the result is seen before
  the next caption replaces this one;
* every section opens on a title card (a name, no number).
"""

from __future__ import annotations

import os
import types

import numpy as np
from qtpy.QtCore import QPoint, QPointF, Qt, QTimer
from qtpy.QtWidgets import QAbstractButton, QMessageBox, QToolButton

from segfix import inventory, inventory_ui

from make_video import handles, main_ready, table_row_for
from recorder import find, outline_around

#: The stem map the inventory chapter loads, written by run_recording.sh
#: next to the demo cloud (scripts/make_stem_map.py).
STEM_MAP = os.environ.get("SEGFIX_DEMO_STEMS", "/tmp/segfix-demo/stems.csv")

#: Class codes make_sample.py writes: leaf, wood (see classes_for there).
LEAF, WOOD = 5, 4

READ_WPS = 2.8   # words per second a viewer reads comfortably, beside a moving picture
SETTLE = 1.6     # seconds to let a result sink in before the next caption


# -- pacing ------------------------------------------------------------------
def read_time(text: str) -> float:
    return 1.2 + len(text.split()) / READ_WPS


def say(r, text, hold=None):
    """Show a caption and wait at least as long as it takes to read. A longer
    ``hold`` is allowed; a shorter one is not."""
    r.rec.set_caption(text)
    yield max(hold or 0.0, read_time(text))


def chapter(r, title):
    """A clear break: the old caption goes, a beat, then the title card."""
    r.rec.clear_caption()
    yield 0.5
    r.rec.chapter = ""  # no section label on captions: the title card announced it
    r.rec.title(title, 2.3)
    yield 2.3
    yield 0.2


def key_then(r, key, action, lead=0.5):
    """Show the key first, then act: the badge leads the change it causes."""
    r.rec.key(key, lead + 1.6)
    yield lead
    action()
    yield None


# -- scene helpers -----------------------------------------------------------
def button(parent, text, prefix=False, contains=False):
    """Any clickable thing with this label: the View group's toggles are
    checkboxes, the view buttons are tool buttons, the rest are push
    buttons, and the storyboard only ever wants to click them."""
    for b in parent.findChildren(QAbstractButton):
        t = b.text().replace("&", "").strip()
        if t == text or (prefix and t.startswith(text)) or (contains and text in t):
            return b
    raise LookupError(f"no button {text!r}")


def table_point(r, table, row, col):
    item = table.item(row, col)
    table.scrollToItem(item)
    return r.wpt(table.viewport(), table.visualItemRect(item).center())


def points_xy(view, mask):
    xy, valid = view.project_to_canvas(view.coords[mask])
    xy = xy[valid]
    if len(xy) < 3:
        raise RuntimeError("too few points on screen to outline")
    return xy


def screen_of(view, xyz):
    xy, _ = view.project_to_canvas(np.asarray([xyz], float))
    return xy[0]


def neighbour_button(panel, nid):
    """The button for tree ``nid``, and the number key it answers to (the
    picker is ordered nearest-first, and the first five get keys)."""
    ids = list(panel._neighbour_ids)
    if nid not in ids:
        raise LookupError(f"no neighbour button for {nid}")
    i = ids.index(nid)
    return panel._neighbour_btns[i], (i + 1 if i < 5 else None)


def menu_of(win, title):
    bar = win.menuBar()
    for act in bar.actions():
        if act.text().replace("&", "") == title:
            return bar, act, act.menu()
    raise LookupError(title)


def open_menu(r, win, title):
    bar, act, menu = menu_of(win, title)
    rect = bar.actionGeometry(act)
    yield from r.click(r.wpt(bar, rect.center()))
    menu.popup(bar.mapToGlobal(rect.bottomLeft()))
    yield 0.3
    return menu


def pick_theme(r, win, name):
    pref = yield from open_menu(r, win, "Preferences")
    theme_act = next(a for a in pref.actions() if a.menu() is not None)
    rect = pref.actionGeometry(theme_act)
    yield from r.move_to(r.wpt(pref, rect.center()), 0.5)
    theme_menu = theme_act.menu()
    theme_menu.popup(pref.mapToGlobal(rect.topRight()))
    yield 0.8
    act = next(a for a in theme_menu.actions() if a.text() == name)
    yield from r.click(r.wpt(theme_menu, theme_menu.actionGeometry(act).center()), 0.5)
    yield 0.3
    act.trigger()
    theme_menu.hide()
    pref.hide()


def select_current(r, panel, view, tid):
    """Click a tree's row in the lower table: it becomes current, the camera
    flies there (smoothed here) and a box marks it."""
    row = table_row_for(panel.tree_table, tid)
    yield from r.click(table_point(r, panel.tree_table, row, 1))
    yield from r.smooth(view, lambda: panel.tree_table.selectRow(row), 1.1)


def load_from_all_trees(r, scene, view, tid):
    row = table_row_for(scene.table, tid)
    yield from r.double_click(table_point(r, scene.table, row, 1))
    scene.table.selectRow(row)
    yield from r.smooth(view, scene.on_load_tree, 1.1)


def click_done(r, panel, view, settle=SETTLE):
    """Click "Done - next": the current tree is ticked and the camera flies
    to the next unfinished one in the queue."""
    done = button(panel, "Done - next", contains=True)
    yield from r.click(r.wpt(done))
    yield from r.smooth(view, done.click, 1.1)
    yield settle


def ground_z(panel, view):
    labels, xyz = panel.c.cloud.labels, view.coords
    return float(np.median(xyz[labels == 0, 2]))


def trunk_xy(panel, view, tid):
    """Where a tree's stem stands: its points just above the ground, below
    any crown (and above a swallowed patch of ground)."""
    labels, xyz = panel.c.cloud.labels, view.coords
    g = ground_z(panel, view)
    low = (labels == tid) & (xyz[:, 2] > g + 0.3) & (xyz[:, 2] < g + 1.2)
    return np.median(xyz[low, :2], axis=0)


def trunks_split_x(panel, view, tid):
    """The x between two stems sharing one ID, at the widest gap."""
    labels, xyz = panel.c.cloud.labels, view.coords
    g = ground_z(panel, view)
    low = (labels == tid) & (xyz[:, 2] > g + 0.3) & (xyz[:, 2] < g + 1.2)
    xs = np.sort(xyz[low, 0])
    cut = int(np.argmax(np.diff(xs)))
    return (xs[cut] + xs[cut + 1]) / 2


def mislabelled_band(panel, view, taker, owner_xy, gap=0.06):
    """The piece of ``taker``'s points that is physically separate from the
    rest of it and stands nearest ``owner_xy``: the band it took from that
    neighbour. By connectivity, not a radius -- a wide crown can reach into
    any circle drawn around the neighbour's stem."""
    from segfix import analysis

    labels, xyz = panel.c.cloud.labels, view.coords
    idx = np.flatnonzero(labels == taker)
    comp = analysis.connected_components_within(xyz[idx], gap)
    best, best_d = None, np.inf
    for k in np.unique(comp):
        members = idx[comp == k]
        d = float(np.hypot(*(np.median(xyz[members, :2], axis=0) - owner_xy)))
        if len(members) >= 20 and d < best_d:
            best, best_d = members, d
    mask = np.zeros(len(labels), dtype=bool)
    mask[best] = True
    return mask


def selected_trees(panel):
    return sorted({int(t) for t in panel.c.cloud.labels[panel.c.selected_indices()] if t > 0})


def and_list(ids):
    ids = [str(i) for i in ids]
    return ids[0] if len(ids) == 1 else ", ".join(ids[:-1]) + " and " + ids[-1]


def class_key(panel, code: int) -> str:
    """The Ctrl+number that presses a class button, or its name if the class
    is past the keyed five."""
    codes = list(panel._class_codes)
    if code in codes and codes.index(code) < 5:
        return f"Ctrl+{codes.index(code) + 1}"
    return panel.c.class_names.get(code, str(code))


def finish_the_stand(r, scene, panel, view, per_tree=0.45):
    """Walk whatever is left of the plot: load the lowest unreviewed tree,
    then Space through everything that came with it, and repeat.

    Quicker than the narrated passes above because there is nothing to say
    about a clean tree, and the point being shown is the shape of the loop
    on a full plot rather than any one tree: the counter climbing, the
    table filling with ticks.
    """
    done = set(panel.done_ids)
    remaining = [t for t in sorted(scene.c.catalog.records) if t not in done]
    guard = 0
    while remaining and guard < 60:
        guard += 1
        yield from load_from_all_trees(r, scene, view, remaining[0])
        loaded = {int(t) for t in panel.c.cloud.tree_ids}
        for _ in range(len(loaded)):
            if panel.current is None:
                break
            done_btn = button(panel, "Done - next", contains=True)
            yield from r.click(r.wpt(done_btn), 0.3)
            yield from r.smooth(view, done_btn.click, 0.6)
            yield per_tree
        done = set(panel.done_ids)
        remaining = [t for t in sorted(scene.c.catalog.records) if t not in done]


# -- the walkthrough ---------------------------------------------------------
def full(r):
    from segfix.density_ui import DownsampleDialog
    from segfix.shift_ui import GlobalShiftDialog
    from segfix.startup_ui import StartupDialog

    rec = r.rec

    # intro · opening a project -------------------------------------------------
    yield r.until(lambda: find(StartupDialog), 30, "startup dialog")
    r.recording = True
    dlg = find(StartupDialog)
    rec.title("segfix · A walkthrough", 2.8)
    yield 3.0
    yield from say(r, "A tour of every tool, fixing every tree in the example cloud from "
                      "make_sample.py.")

    yield from chapter(r, "Opening a project")
    yield from say(r, "segfix starts with this dialog. New Project… imports a point cloud by "
                      "copying it into a project folder, so the original is never edited.")
    yield from r.move_to(r.wpt(button(dlg, "New Project", prefix=True)), 0.8)
    yield 1.4
    yield from say(r, "Imported projects are listed here. Double-click one to open it.")
    item = dlg.list.item(0)
    yield from r.double_click(r.wpt(dlg.list.viewport(), dlg.list.visualItemRect(item).center()))
    yield 0.6
    dlg._on_choose()

    # large coordinates -------------------------------------------------------
    yield r.until(lambda: find(GlobalShiftDialog), 120, "shift prompt")
    d = find(GlobalShiftDialog)
    yield from chapter(r, "Large coordinates")
    yield from say(r, "This cloud is georeferenced in UTM, with coordinates in the millions. "
                      "Stored as 32-bit floats, they would lose sub-metre detail.")
    yield from r.move_to(r.wpt(d.spins[1]), 0.8)
    yield 0.8
    yield from say(r, "Apply Shift moves the cloud near the origin for this session. Nothing is "
                      "written back, so saved files keep their real coordinates.")
    yield from r.click(r.wpt(d.apply_btn))
    yield 0.6
    d.apply_btn.click()

    # dense clouds --------------------------------------------------------------
    yield r.until(lambda: find(DownsampleDialog), 120, "downsample prompt")
    d = find(DownsampleDialog)
    yield from chapter(r, "Dense clouds")
    yield from say(r, f"segfix measures point spacing on load. These points are about "
                      f"{d.spacing * 100:.1f} cm apart, far finer than fixing a segmentation "
                      f"needs.")
    yield from r.move_to(r.wpt(d.spin), 0.8)
    yield 0.8
    yield from say(r, "Downsample keeps one point per 3 cm voxel, and says as you change it how "
                      "much of the cloud that keeps. Your edits are copied back onto every "
                      "original point when you save.")
    yield from r.click(r.wpt(d.downsample_btn))
    yield 0.6
    rec.set_caption("Large files show a progress window while they load.")
    yield 0.4
    d.downsample_btn.click()

    # the tree table ------------------------------------------------------------
    yield r.until(main_ready, 300, "main window")
    win, panel, scene, view = handles()
    panel.c.cluster.CHAIN_SECONDS = 60  # recording only: reading time between repeat clicks
    cam = view.view.camera
    yield SETTLE
    yield from chapter(r, "The tree table")
    yield from r.move_to(r.wpt(scene.path_label), 0.8)
    yield from say(r, "All Trees lists every tree in the file, with how many are done. Its "
                      "header also shows this session is downsampled to 3 cm.")
    yield from say(r, "Double-click a tree to load it, together with the trees it touches.")
    yield from load_from_all_trees(r, scene, view, 1)
    yield SETTLE
    yield from say(r, "Tree 1 came with trees 2 and 3, which touch it: 13, 7 and 9.5 metres "
                      "tall.")

    # moving around -------------------------------------------------------------
    yield from chapter(r, "Moving around")
    cv = rec.canvas_rect()
    c = QPointF(cv.center().x(), cv.center().y() - 60)
    yield from say(r, "Navigation works like CloudCompare: left-drag rotates, right-drag pans, "
                      "and the mouse wheel zooms.")
    rec.key("left-drag", 2.6)
    yield 0.5
    yield from r.drag_camera(view, c + QPointF(-180, 0), c + QPointF(200, 0), 2.2,
                             azimuth=cam.azimuth + 70)
    yield 1.0
    rec.key("right-drag", 2.4)
    yield 0.5
    c0 = np.asarray(cam.center, float)
    yield from r.drag_camera(view, c, c + QPointF(140, 70), 1.6,
                             center=tuple(c0 + [1.0, -1.0, 0.6]))
    yield 1.0
    rec.key("wheel", 3.0)
    yield 0.5
    sf = cam.scale_factor
    yield from r.camera(view, 1.2, scale_factor=sf * 0.65)
    yield 0.6
    yield from r.camera(view, 1.1, scale_factor=sf)
    yield from r.camera(view, 0.9, center=tuple(c0))
    yield SETTLE
    yield from say(r, "The buttons under Point size snap the camera to a fixed view, as "
                      "CloudCompare does: Top, Front, and 3D to tilt back again.")
    for name in ("top", "front", "3d"):
        vb = win.findChild(QToolButton, f"view_{name}")
        yield from r.click(r.wpt(vb))
        yield from r.smooth(view, vb.click, 1.0)
        yield 1.2

    yield from say(r, "Inside stands the camera at the pivot and switches to perspective, "
                      "so a crowded plot stops reading as a flat wall.")
    inside = win.findChild(QToolButton, "view_inside")
    yield from r.click(r.wpt(inside))
    yield from r.smooth(view, inside.click, 1.0)
    yield 1.0
    yield from r.drag_camera(view, c + QPointF(-120, 0), c + QPointF(120, 0), 2.2,
                             azimuth=cam.azimuth + 55)
    yield from say(r, "Drag to look around, double-click a point to stand there, scroll to "
                      "pull back out.")
    yield from r.click(r.wpt(inside))
    yield from r.smooth(view, inside.click, 1.0)
    yield SETTLE

    labels, xyz = panel.c.cloud.labels, view.coords
    g = ground_z(panel, view)
    trunk = np.flatnonzero((labels == 1) & (xyz[:, 2] > g + 1.0) & (xyz[:, 2] < g + 1.4))
    txy = screen_of(view, xyz[trunk[len(trunk) // 2]])
    yield from say(r, "Double-click a point to make it the centre of rotation.")
    yield from r.double_click(r.canvas_to_frame(view, *txy))
    before = cam.center
    view._on_double_click(types.SimpleNamespace(button=1, pos=(float(txy[0]), float(txy[1]))))
    after = cam.center
    cam.center = before
    yield from r.camera(view, 0.9, center=after)
    yield 0.6
    yield from r.drag_camera(view, c + QPointF(-140, 0), c + QPointF(160, 0), 1.8,
                             azimuth=cam.azimuth - 60)
    yield SETTLE
    yield from r.smooth(view, view.reset_view, 1.0)
    yield SETTLE

    # the review queue ------------------------------------------------------------
    yield from chapter(r, "The review queue")
    yield from say(r, "Selected Tree + Neighbours is the queue for what's loaded. Click a row to "
                      "make that tree the current one.")
    yield from select_current(r, panel, view, 1)
    yield SETTLE
    yield from say(r, "The camera flies to it and a box marks it. Its fix actions float on the "
                      "right of the view.")
    yield from say(r, "The Fade column ghosts a tree but keeps it selectable.")
    row2 = table_row_for(panel.tree_table, 2)
    yield from r.click(table_point(r, panel.tree_table, row2, panel.FADE_COL))
    panel.tree_table.item(row2, panel.FADE_COL).setCheckState(Qt.CheckState.Checked)
    yield SETTLE + 0.5
    yield from r.click(table_point(r, panel.tree_table, row2, panel.FADE_COL), 0.3)
    panel.tree_table.item(row2, panel.FADE_COL).setCheckState(Qt.CheckState.Unchecked)
    yield 1.2
    yield from say(r, "The eye column hides it completely.")
    row2 = table_row_for(panel.tree_table, 2)
    yield from r.click(table_point(r, panel.tree_table, row2, panel.HIDE_COL))
    panel.tree_table.item(row2, panel.HIDE_COL).setCheckState(Qt.CheckState.Checked)
    yield SETTLE + 0.5
    yield from r.click(table_point(r, panel.tree_table, row2, panel.HIDE_COL), 0.3)
    panel.tree_table.item(row2, panel.HIDE_COL).setCheckState(Qt.CheckState.Unchecked)
    yield 1.2
    yield from say(r, "Fade others and Hide others do this to every tree except the current "
                      "one, on Shift+G and G.")
    fade_others = button(panel.top_bar, "Fade others", prefix=True)
    yield from r.click(r.wpt(fade_others))
    fade_others.click()
    yield SETTLE + 0.5
    yield from r.click(r.wpt(fade_others), 0.3)
    fade_others.click()
    yield 1.0
    hide_others = button(panel.top_bar, "Hide others", prefix=True)
    yield from r.click(r.wpt(hide_others))
    hide_others.click()
    yield SETTLE + 0.5
    yield from r.click(r.wpt(hide_others), 0.3)
    hide_others.click()
    yield 1.2
    yield from say(r, "F hides the unassigned ground points, and shows them again.")
    yield from key_then(r, "F", panel.show_unassigned.toggle)
    yield SETTLE + 0.5
    yield from key_then(r, "F", panel.show_unassigned.toggle)
    yield SETTLE

    # lasso and lasso tree: fixing the band --------------------------------------
    yield from chapter(r, "Lasso and lasso tree")
    labels, xyz = panel.c.cloud.labels, view.coords
    g = ground_z(panel, view)
    c1, c2 = trunk_xy(panel, view, 1), trunk_xy(panel, view, 2)
    # Look straight across the gap between trees 1 and 2.
    mid = np.array([(c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2, g + 4.5])
    yield from r.smooth(view, lambda: (view.fly_to(mid, 7.5),
                                       setattr(cam, "azimuth", 0.0),
                                       setattr(cam, "elevation", 10.0)), 1.6)
    yield 1.0
    patch = mislabelled_band(panel, view, taker=1, owner_xy=c2)
    patch_xy = points_xy(view, patch)
    yield from r.move_to(r.canvas_to_frame(view, *np.median(patch_xy, axis=0)), 0.9)
    yield from say(r, "Where trees 1 and 2 touch, part of tree 2's crown was given tree 1's ID: "
                      "the blue band on the green crown.")
    outline = outline_around(patch_xy, margin=16, seed=3)

    yield from say(r, "Q arms the lasso: drag an outline, and every point inside it is "
                      "selected. Hold Shift to add to a selection.")
    yield from key_then(r, "Q", panel.lasso_btn.toggle)
    yield from r.lasso(panel.c.lasso, outline, 2.4)
    yield SETTLE
    yield from r.move_to(r.wpt(panel.sel_info), 0.8)
    yield from say(r, f"But the outline also covers tree 2's own points, so the lasso caught "
                      f"points of trees {and_list(selected_trees(panel))}.")
    yield from say(r, "W is Lasso tree: the same outline, but it only takes points already "
                      "in the current tree, tree 1.")
    yield from key_then(r, "W", panel.tree_lasso_btn.toggle)
    yield from r.lasso(panel.c.lasso, outline, 2.4)
    yield SETTLE
    yield from r.move_to(r.wpt(panel.sel_info), 0.8)
    yield from say(r, f"Now only tree {and_list(selected_trees(panel))}'s points are selected: "
                      "exactly the mislabelled band.")
    nb, nkey = neighbour_button(panel, 2)
    yield from r.move_to(r.wpt(nb), 0.8)
    yield from say(r, "The buttons in the Current tree panel send a selection straight to a "
                      "neighbour, nearest tree first. This band belongs to tree 2.")
    yield from say(r, f"The first five carry a number key, so the edit never leaves the left "
                      f"hand: tree 2 is key {nkey}.")
    yield from key_then(r, str(nkey), lambda: panel.send_to_nth_neighbour(nkey))
    yield SETTLE + 0.5
    yield from say(r, "Fixed. Every edit can be undone with Ctrl+Z, and redone with "
                      "Ctrl+Shift+Z.")
    yield from key_then(r, "Ctrl+Z", panel.on_undo)
    yield SETTLE
    yield from key_then(r, "Ctrl+Shift+Z", panel.on_redo)
    yield SETTLE
    yield from key_then(r, "Esc", panel.on_move_mode)
    yield 1.0

    # marking trees done ----------------------------------------------------------
    yield from chapter(r, "Marking trees done")
    yield from r.smooth(view, view.reset_view, 1.0)
    done = button(panel, "Done - next", contains=True)
    yield from r.move_to(r.wpt(done), 0.8)
    yield from say(r, "When a tree is right, click Done, or press Space. It gets a tick, and "
                      "the next tree in the queue comes up.")
    yield from click_done(r, panel, view)
    yield from say(r, "Trees 2 and 3 have no errors, so they're done too.")
    yield from click_done(r, panel, view)
    yield from click_done(r, panel, view)
    yield from r.move_to(r.wpt(scene.path_label), 0.8)
    yield from say(r, "All three are ticked, here and in All Trees, and progress is saved beside "
                      "the project, so you can pick up where you left off.")

    # merging a split tree --------------------------------------------------------
    yield from chapter(r, "Merging a split tree")
    yield from say(r, "Tree 7 was split in two by mistake: its top half became tree 8.")
    yield from load_from_all_trees(r, scene, view, 7)
    yield from select_current(r, panel, view, 7)
    yield SETTLE
    yield from say(r, "Cluster (E) selects by connectivity: click a point, and it takes the "
                      "connected patch of that point's tree.")
    yield from key_then(r, "E", panel.cluster_btn.toggle)
    yield 0.8
    # The gap control stays open for the whole demo, so every step is seen:
    # the slider moves and its label reads the new gap as each click lands.
    yield from r.move_to(r.wpt(panel.cluster_gap_btn), 0.8)
    yield from say(r, "How wide a gap still counts as connected is set by the arrow beside "
                      "Cluster. It starts tight, at 1× the point spacing.")
    yield from r.click(None)
    panel._toggle_popover(panel._cluster_gap_popover, panel.cluster_gap_btn)
    yield SETTLE
    labels, xyz = panel.c.cloud.labels, view.coords
    g = ground_z(panel, view)
    # Low on tree 8 (just above the split), well clear of the open slider.
    crown = np.flatnonzero((labels == 8) & (xyz[:, 2] > g + 6.3) & (xyz[:, 2] < g + 6.9))
    cxy, _ = view.project_to_canvas(xyz[crown])
    seed_xy = cxy[np.argmin(np.abs(cxy[:, 0] - np.median(cxy[:, 0])))]
    seed_frame = r.canvas_to_frame(view, *seed_xy)

    def gap_click():
        panel.c.cluster._select_at((float(seed_xy[0]), float(seed_xy[1])), False)
        rec.key(f"gap {panel.c.cluster_gap_factor:g}× spacing", 2.4)

    yield from r.move_to(seed_frame, 0.8)
    yield from say(r, "At 1×, a first click takes only a small seed.")
    yield from r.click(None)
    gap_click()
    yield SETTLE + 0.5
    yield from say(r, "Click the same spot again: each click loosens the gap one step, and the "
                      "patch grows.")
    # Three repeats (1x -> 1.5x -> 2x -> 3x): measured offline on this scene,
    # 2x still leaves 4 stray points of tree 8 behind; 3x takes all of it.
    for _ in range(3):
        yield from r.click(None)
        gap_click()
        yield SETTLE + 0.8
    yield from say(r, "R and T step the gap too. It goes back to 1× when you switch Cluster off "
                      "or clear the selection.")
    panel._cluster_gap_popover.hide()
    yield 0.8
    yield from say(r, "All of tree 8 is selected now. Press A to add it to the current tree, "
                      "tree 7.")
    yield from key_then(r, "A", panel.on_add)
    yield SETTLE + 0.5
    yield from say(r, "The two halves are one tree again, and tree 8 has left the table.")
    yield from key_then(r, "Esc", panel.on_move_mode)
    yield 1.0
    yield from say(r, "Tree 7 is fixed, and its neighbour, tree 9, has no errors. Done, and "
                      "done.")
    yield from click_done(r, panel, view)
    yield from click_done(r, panel, view)

    # splitting two trees ---------------------------------------------------------
    yield from chapter(r, "Splitting two trees")
    yield from say(r, "Tree 10 is really two trees that share one ID.")
    yield from load_from_all_trees(r, scene, view, 10)
    yield from select_current(r, panel, view, 10)
    yield SETTLE
    yield from say(r, "Seen from above, the two crowns are easy to tell apart.")
    yield from r.camera(view, 1.8, elevation=89.0, azimuth=0.0)
    yield SETTLE
    labels, xyz = panel.c.cloud.labels, view.coords
    split = trunks_split_x(panel, view, 10)
    right = (labels == 10) & (xyz[:, 0] > split)
    yield from say(r, "Lasso tree (W) again, so the ground underneath isn't picked up. Outline the "
                      "right-hand tree…")
    yield from key_then(r, "W", panel.tree_lasso_btn.toggle)
    yield from r.lasso(panel.c.lasso, outline_around(points_xy(view, right), margin=18), 2.6)
    yield SETTLE
    yield from say(r, "…and press S to split it off as a new tree. It joins the queue, not yet "
                      "reviewed.")
    yield from key_then(r, "S", panel.on_create_new)
    yield SETTLE + 0.5
    yield from key_then(r, "Esc", panel.on_move_mode)
    yield from r.camera(view, 1.8, elevation=22.0, azimuth=cam.azimuth + 35)
    yield SETTLE
    new_id = int(panel.c.cloud.labels.max())

    # cutting the view down -------------------------------------------------------
    yield from chapter(r, "Cutting the view down")
    yield from say(r, "Cross section (C) shows only a slab of the cloud. Its axis and extent "
                      "are set under Slab….")
    yield from key_then(r, "C", panel.cross_enable.toggle)
    yield 0.8
    slab = button(panel.cross_box, "Slab…")
    yield from r.click(r.wpt(slab))
    panel._toggle_popover(panel._cross_popover, slab)
    yield 1.0
    s = panel.cross_max_slider
    steps = panel.CROSS_SECTION_STEPS
    y_mid = s.height() // 2
    yield from r.move_to(r.wpt(s, QPoint(s.width() - 6, y_mid)), 0.7)
    for v in np.linspace(steps, steps * 0.32, 48):
        s.setValue(int(v))
        rec.cursor = r.wpt(s, QPoint(int(6 + (s.width() - 12) * v / steps), y_mid))
        yield None
    yield SETTLE
    yield from say(r, "Only points inside the slab are shown or selectable, so a lasso can't "
                      "reach through to anything behind.")
    panel._cross_popover.hide()
    yield from key_then(r, "C", panel.cross_enable.toggle)
    yield SETTLE
    yield from say(r, "Lasso section keeps any shape you draw: press Shift+Q, then drag an "
                      "outline.")
    yield from key_then(r, "Shift+Q", panel.section_draw_btn.toggle)
    yield from r.lasso(panel.c.lasso,
                       outline_around(points_xy(view, panel.c.cloud.labels == new_id), margin=30),
                       2.4)
    yield SETTLE
    yield from say(r, "The outline is frozen into the view, so you can orbit freely.")
    yield from key_then(r, "Esc", panel.on_move_mode)
    yield from r.drag_camera(view, c + QPointF(-150, 0), c + QPointF(150, 0), 2.0,
                             azimuth=cam.azimuth + 60)
    yield 1.0
    reset = button(panel.lasso_section_box, "Reset")
    yield from r.move_to(r.wpt(reset), 0.8)
    yield from say(r, "Reset clears it.")
    yield from r.click(None)
    reset.click()
    yield SETTLE
    yield from say(r, f"Tree 10 and the new tree {new_id} are both right now. Done, and done.")
    yield from select_current(r, panel, view, 10)
    yield 0.6
    yield from click_done(r, panel, view)
    yield from click_done(r, panel, view)

    # unassigning -----------------------------------------------------------------
    yield from chapter(r, "Unassigning points")
    yield from say(r, "Tree 11's ID also swallowed a patch of the ground beside it.")
    yield from load_from_all_trees(r, scene, view, 11)
    yield from select_current(r, panel, view, 11)
    yield from r.camera(view, 1.8, azimuth=0.0, elevation=14.0)
    yield SETTLE
    yield from say(r, "Hide the real ground with F, and the patch that belongs to tree 11 "
                      "stands out.")
    yield from key_then(r, "F", panel.show_unassigned.toggle)
    yield SETTLE + 1.0
    yield from key_then(r, "F", panel.show_unassigned.toggle)
    yield 1.2
    labels, xyz = panel.c.cloud.labels, view.coords
    g = float(np.median(xyz[labels == 0, 2]))
    stem = trunk_xy(panel, view, 11)
    patch = (labels == 11) & (xyz[:, 2] < g + 0.25) & (xyz[:, 0] > stem[0] + 0.4)
    yield from say(r, "Lasso tree (W) around it takes only tree 11's points: not the ground, and "
                      "not the bush beside it.")
    yield from key_then(r, "W", panel.tree_lasso_btn.toggle)
    yield from r.lasso(panel.c.lasso, outline_around(points_xy(view, patch), margin=14), 2.2)
    yield SETTLE
    yield from say(r, "Press D to unassign it.")
    yield from key_then(r, "D", panel.on_unassign)
    yield SETTLE + 0.5
    yield from say(r, "The patch turns grey: it's just ground again. With nothing selected, D "
                      "unassigns the whole current tree.")
    yield from key_then(r, "Esc", panel.on_move_mode)
    yield 1.0

    # marking noise ---------------------------------------------------------------
    yield from chapter(r, "Marking noise")
    yield from say(r, "Tree 12 isn't a tree at all: it's a bush.")
    yield from select_current(r, panel, view, 12)
    yield from r.camera(view, 1.2, scale_factor=view.view.camera.scale_factor * 1.6)
    yield SETTLE
    yield from say(r, "With nothing selected, X marks the whole current tree as noise. Delete "
                      "does the same, and with a selection only that is marked.")
    yield from key_then(r, "X", panel.on_noise)
    yield SETTLE + 0.5
    yield from say(r, "The bush stays in the file, flagged as noise, and drops out of the tree "
                      "table. F hides noise along with the ground.")
    yield from key_then(r, "F", panel.show_unassigned.toggle)
    yield SETTLE
    yield from key_then(r, "F", panel.show_unassigned.toggle)
    yield SETTLE
    yield from say(r, "Tree 11 is fixed, and the bush is gone. Done.")
    yield from select_current(r, panel, view, 11)
    yield 0.6
    yield from click_done(r, panel, view)

    # leaf and wood ---------------------------------------------------------------
    yield from chapter(r, "Leaf and wood")
    yield from load_from_all_trees(r, scene, view, 4)
    yield from select_current(r, panel, view, 4)
    yield SETTLE
    yield from say(r, "A cloud can also carry what each point is, not just which tree it "
                      "belongs to: leaf, wood, ground.")
    yield from r.move_to(r.wpt(panel.class_color_cb), 0.8)
    yield from say(r, "Shift+F colours by class instead of by tree. The names come from "
                      "Edit ▸ Point Classes…, once per project.")
    yield from key_then(r, "Shift+F", panel.class_color_cb.toggle)
    yield SETTLE + 0.8

    labels, xyz = panel.c.cloud.labels, view.coords
    classes = panel.c.cloud.classes
    g = ground_z(panel, view)
    stem = trunk_xy(panel, view, 4)
    # A patch of trunk the classifier called leaf: the band just under the
    # crown, where a real leaf/wood filter gets it wrong too.
    mistaken = (
        (labels == 4) & (classes == LEAF) & (xyz[:, 2] > g + 2.0)
        & (xyz[:, 2] < g + 3.2)
        & (np.hypot(xyz[:, 0] - stem[0], xyz[:, 1] - stem[1]) < 0.5)
    )
    if mistaken.sum() > 50:
        yield from say(r, "Where the filter got it wrong, fix it the same way you fix a tree: "
                          "lasso the points…")
        yield from r.camera(view, 1.4, center=(float(stem[0]), float(stem[1]), g + 2.6),
                            scale_factor=6.0)
        yield from key_then(r, "W", panel.tree_lasso_btn.toggle)
        yield from r.lasso(panel.c.lasso, outline_around(points_xy(view, mistaken), margin=12),
                           2.2)
        yield SETTLE
        wood_key = class_key(panel, WOOD)
        yield from say(r, f"…and press {wood_key} for wood. The class buttons work on a "
                          f"selection exactly as the tree buttons do.")
        yield from key_then(r, wood_key, lambda: panel.on_set_class(WOOD))
        yield SETTLE + 0.5
        yield from key_then(r, "Esc", panel.on_move_mode)
    yield from say(r, "Class edits undo, redo and save with the tree edits, and only the "
                      "values that changed are written.")
    yield from key_then(r, "Shift+F", panel.class_color_cb.toggle)
    yield from r.smooth(view, view.reset_view, 1.0)
    yield SETTLE

    # field inventory ---------------------------------------------------------------
    yield from chapter(r, "Field inventory")
    yield from say(r, "A plot usually has a stem map long before it has a point cloud: "
                      "measured positions, DBH and height.")
    stems, _columns = inventory.load_stem_map(STEM_MAP)
    trees = inventory.stats_from_records(scene.c.catalog.records)
    panel.set_stem_map(stems, inventory.Alignment())
    yield from say(r, f"Inventory ▸ Load Stem Map… reads the CSV. This one has "
                      f"{len(stems)} stems, in the plot's own local coordinates.")
    align = inventory_ui.AlignDialog(panel, trees, win)
    align.show()
    yield 0.8
    yield from r.move_to(r.wpt(align.fit_btn), 0.8)
    yield from say(r, "Segfix finds the shift from the pattern of the stems, so a local map "
                      "lands on a georeferenced cloud.")
    yield from r.click(None)
    align.fit()
    yield SETTLE + 1.0
    yield from say(r, align.summary.text().replace("\n", " · "))
    yield from r.click(r.wpt(align.buttons), 0.6)
    align.accept()
    yield SETTLE

    yield from say(r, "Every measured tree is drawn as a cylinder at its DBH and height, so "
                      "you can see it against the trunk it belongs to.")
    yield from r.smooth(view, view.reset_view, 1.0)
    yield from r.drag_camera(view, c + QPointF(-140, 0), c + QPointF(140, 0), 2.2,
                             azimuth=cam.azimuth + 50)
    yield SETTLE
    yield from select_current(r, panel, view, 4)
    yield 0.8
    yield from say(r, "For the tree under review, the table ranks the stems it could be, on "
                      "position, height and DBH together.")
    yield from r.move_to(r.wpt(panel.inventory_table), 0.8)
    if panel.inventory_table.rowCount():
        yield from r.click(None)
        panel.inventory_table.selectRow(0)
        yield SETTLE + 0.5
        yield from say(r, "Picking a row lights that stem up. Link records the match, and the "
                          "cylinder turns green.")
        yield from r.move_to(r.wpt(panel.link_btn), 0.8)
        yield from r.click(None)
        panel.on_link_stem()
        yield SETTLE + 0.8
    yield from say(r, "Inventory ▸ Export Matches… writes the whole plot out as a CSV: which "
                      "tree is which stem, and how far apart they were.")
    yield SETTLE

    # the rest of the plot ----------------------------------------------------------
    yield from chapter(r, "The rest of the plot")
    yield from say(r, "Trees 4, 5 and 6 have no errors: load each one, look it over, and click "
                      "Done.")
    for tid in (4, 5, 6):
        yield from load_from_all_trees(r, scene, view, tid)
        yield from select_current(r, panel, view, tid)
        yield 0.8
        yield from click_done(r, panel, view, settle=1.0)
    yield from say(r, "The rest of the stand is clean too. Loading a tree brings the ones "
                     "touching it, so Space walks a whole group before the next load.")
    yield from finish_the_stand(r, scene, panel, view)
    yield from r.move_to(r.wpt(scene.path_label), 0.8)
    yield 0.6
    yield from say(r, "Every tree in the table is done.")

    # saving ----------------------------------------------------------------------
    yield from chapter(r, "Saving")
    total = scene.c.catalog.count
    yield from say(r, "Ctrl+S saves into the project copy. Only points whose label changed are "
                      "written.")
    yield from say(r, "This session was downsampled, so the edits are interpolated back onto "
                      f"all {total:,} original points.")
    rec.key("Ctrl+S", 2.0)
    yield 0.5
    panel.on_save()
    yield SETTLE
    yield from say(r, "The saved file keeps every point, and every other field, byte for byte.")
    yield from say(r, "File ▸ Export Trees… writes each tree out as a file of its own, into a "
                      "folder you pick.")
    file_menu = yield from open_menu(r, win, "File")
    exp = next(a for a in file_menu.actions() if "Export" in a.text())
    yield from r.move_to(r.wpt(file_menu, file_menu.actionGeometry(exp).center()), 0.8)
    yield 1.6
    file_menu.hide()
    yield 0.6

    # help and preferences --------------------------------------------------------
    yield from chapter(r, "Help and preferences")
    yield from say(r, "Help ▸ About shows the version, links and licence.")
    help_menu = yield from open_menu(r, win, "Help")
    about = help_menu.actions()[0]
    yield from r.click(r.wpt(help_menu, help_menu.actionGeometry(about).center()), 0.5)
    help_menu.hide()
    # A modal exec() inside a script step would block the recorder; start it
    # from the event loop instead, then close it from a later step.
    QTimer.singleShot(0, about.trigger)
    yield r.until(lambda: find(QMessageBox), 10, "about box")
    yield 3.4
    find(QMessageBox).accept()
    yield 0.8
    yield from say(r, "Preferences ▸ Theme switches between light and dark, and remembers your "
                      "choice.")
    yield from pick_theme(r, win, "Dark")
    yield 3.0
    yield from pick_theme(r, win, "Light")
    yield SETTLE

    # the end -------------------------------------------------------------------
    rec.clear_caption()
    yield 0.6
    rec.chapter = ""
    rec.title("That's segfix", 2.6)
    yield 2.8
    yield from say(r, "Install it with: pip install segfix\n"
                      "Code, issues and docs: github.com/UQ-EORC/Segfix", hold=6.0)
    rec.cursor = None
    yield 1.0
