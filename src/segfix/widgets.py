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

"""Qt dock widget wiring the lasso selection to the fix operations.

The workflow is a review queue.  The table lists the trees currently loaded —
the one picked in scene mode's "All Trees" plus its neighbours, or the whole
file when it was loaded in one go.  Space marks the current tree done, flies
the camera to the next unfinished one and jumps to it.  The current tree is
always the implicit target: lasso points (Q) and press A to add them to it,
N to split a new tree off, U/X to unassign or trash.
"""

from __future__ import annotations

import json
import os
import threading
import time

import numpy as np
from qtpy.QtCore import QRectF, QSize, Qt
from qtpy.QtGui import QBrush, QColor, QIcon, QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import analysis
from . import inventory
from . import operations as ops
from . import theme
from .cloudview import INSIDE_FOV
from .icons import icon
from .lasso import ClusterTool, LassoTool
from .model import NOISE, UNASSIGNED, PointCloud
from .viewer import (
    busy,
    class_colors,
    colors_for_labels,
    refresh_view,
    visibility_mask,
)


#: Inventory stems, drawn as wireframe cylinders over the cloud: amber for
#: the stem map as loaded, brighter for the candidate under the cursor, green
#: for the one this tree is matched to. RGBA, slightly translucent so a
#: cylinder never hides the trunk it is being compared with.
STEM_COLOR = (0.88, 0.70, 0.25, 0.55)
ACTIVE_STEM_COLOR = (1.0, 0.95, 0.5, 0.95)
LINKED_STEM_COLOR = (0.45, 0.85, 0.5, 0.9)

#: Settings for the cluster tool's gap: how many point spacings of empty space
#: one click will bridge. Discrete steps rather than a free value because the
#: useful range is multiplicative -- 1x to 2x is as big a change in what gets
#: grabbed as 8x to 16x.
#: How many neighbouring trees the number keys reach: 1-5 is as far as a
#: left hand goes without moving. Further neighbours keep their button.
NEIGHBOUR_KEYS = 5

#: How many point classes Ctrl+number reaches, for the same reason.
CLASS_KEYS = 5

def key_badge(digit: str, colour: QColor, size: int = 16) -> QPixmap:
    """A little keycap with ``digit`` on it, for the button that key presses.

    Drawn rather than typed: the keycap emoji (1\ufe0f\u20e3) renders as
    nothing at all in Qt with Noto Color Emoji and as an empty box without
    it, and a bare digit beside the tree id reads as a second number rather
    than as a key.
    """
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(colour, 1.2))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(QRectF(0.9, 0.9, size - 1.8, size - 1.8), 3.5, 3.5)
    font = p.font()
    font.setPixelSize(int(size * 0.66))
    font.setBold(True)
    p.setFont(font)
    p.setPen(colour)
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, digit)
    p.end()
    return pm

CLUSTER_GAP_FACTORS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)
#: Start as tight as it goes: a first click is a small seed, and each repeat
#: click on the same spot loosens it a step (see ClusterTool). Switching the
#: Cluster tool off, or the selection being cleared, comes back here -- see
#: SegFixWidget.reset_cluster_gap.
DEFAULT_CLUSTER_GAP_FACTOR = 1.0

#: How many (tree, gap) labellings SegFixController keeps. One tree's full
#: ladder is len(CLUSTER_GAP_FACTORS) of them, so this is the last seven-odd
#: trees -- far more than a review revisits, and a ceiling on what the
#: prefetch can accumulate: a labelling is 4 bytes a point, so a plot walked
#: tree by tree without a reload would otherwise climb without limit.
CLUSTER_CACHE_ENTRIES = 64

#: Threads the gap-ladder prefetch runs on. One is not enough on a big tree:
#: a labelling there takes about as long as a user takes to click again, so a
#: single worker walks the ladder at exactly the speed it is being chased and
#: never gets ahead. Three pull from the same queue in priority order, which
#: puts it comfortably in front on a 300k-point tree, and SciPy drops the GIL
#: for all of this so they cost cores rather than responsiveness.
CLUSTER_PREFETCH_WORKERS = 3


class SegFixController:
    """Holds the editable cloud + the 3D view, and applies operations."""

    def __init__(self, view, cloud: PointCloud):
        self.view = view
        self.cloud = cloud
        self.save_path = cloud.source_path
        # Tree IDs the panel has "faded": rendered at FADED_ALPHA opacity but
        # still shown and still selectable (unlike a hidden tree). Lives here,
        # not on the widget like hidden_ids, because _after_edit re-applies the
        # face colours and must keep these ghosted through the refresh.
        self.faded_ids: set[int] = set()
        # Optional override: when set (project/scene mode), Save delegates
        # here instead of writing a single file. Signature: () -> str.
        self.on_save_override = None
        # Set by the panel so it can re-hook view state after a reload.
        self.on_cloud_changed = None
        # Optional fn(indices)->indices narrowing a lasso's result before it
        # becomes the selection — set by the panel for the "current tree
        # only" lasso mode; persists across set_cloud (it's a mode choice,
        # not per-cloud state).
        self.lasso_filter = None
        # Optional fn(indices)->None: when set (drawing a lasso section),
        # a completed lasso is redirected here instead of becoming the
        # selection — see _on_lasso. Also a mode choice, not per-cloud state.
        self.on_lasso_section = None
        # Cluster-tool caches, all keyed to the current point array and
        # dropped in set_cloud: the measured point spacing, and the
        # connected-component labellings, one per tree and gap, each built
        # the first time that combination is asked for and kept thereafter.
        self._cluster_spacing: float | None = None
        #: (tree label, gap) -> (point indices, component id per index).
        #: Written from the prefetch worker as well as from here, so every
        #: mutation goes through _cache_cluster/_cluster_cc_get under
        #: _cluster_lock -- a plain read of one key needs no lock, but the
        #: eviction below walks the dict and must not run beside an insert.
        self._cluster_cc: dict[tuple[int, float], tuple] = {}
        self._cluster_lock = threading.Lock()
        #: The gap-ladder prefetch (see _start_cluster_prefetch): the worker
        #: warming the rest of the ladder for the tree last clicked, the tree
        #: it is doing, and the flag that tells it to give up.
        self._prefetch: tuple[threading.Thread, ...] = ()
        self._prefetch_label: int | None = None
        self._prefetch_stop = threading.Event()
        #: Off in tests that count labellings, on in the app.
        self.cluster_prefetch = True
        # How many point spacings of empty space a cluster click bridges; the
        # panel's "Cluster gap" popover sets it. A mode choice like
        # lasso_filter, so it survives set_cloud.
        self.cluster_gap_factor = DEFAULT_CLUSTER_GAP_FACTOR
        self.lasso = LassoTool(view, self._on_lasso)
        self.cluster = ClusterTool(view, self._grow_cluster, self._on_cluster)
        #: The tree this cloud was loaded for, if it was loaded for one; see
        #: set_cloud. None when the whole file is the scene.
        self.focus_label: int | None = None
        #: Point classes: ``{value: name}`` for the cloud's class field
        #: (empty when there is none), and whether the view is coloured by
        #: class rather than by tree. Both outlive set_cloud.
        self.class_names: dict[int, str] = {}
        self.class_field: str | None = None
        self.color_by_class = False
        #: ``(lowest, highest)`` value the class field can store, or None;
        #: a new class must fit. Set by the app from the catalog.
        self.class_range: tuple[int, int] | None = None

    def class_colours(self) -> dict | None:
        """``{value: rgb}`` while colouring by class, else None."""
        if not self.color_by_class or self.cloud.classes is None:
            return None
        return class_colors(self.class_names)

    def set_cloud(self, cloud: PointCloud, focus: int | None = None) -> None:
        """Re-point the controller at a freshly loaded cloud (the view has
        already been handed the new points by the caller).

        ``focus`` is the tree the cloud was loaded *for* — the one picked in
        the All Trees table, as opposed to the neighbours that came with it.
        The panel starts its review on that tree rather than on nothing.
        """
        was_lasso, was_cluster = self.lasso.armed, self.cluster.armed
        self.lasso.set_armed(False)
        self.cluster.set_armed(False)
        self._stop_cluster_prefetch()
        self._cluster_spacing, self._cluster_cc = None, {}  # for the old points
        self.cloud = cloud
        self.focus_label = focus
        self.save_path = cloud.source_path
        self.faded_ids = set()  # a fresh cloud starts with nothing faded
        if was_lasso:
            self.lasso.set_armed(True)
        if was_cluster:
            self.cluster.set_armed(True)
        if self.on_cloud_changed is not None:
            self.on_cloud_changed()

    def _on_lasso(self, indices: np.ndarray, additive: bool) -> None:
        if self.on_lasso_section is not None:
            self.on_lasso_section(indices)
            return
        if self.lasso_filter is not None:
            indices = self.lasso_filter(indices)
        total = self.view.select(indices, additive=additive)
        self.view.status = f"Lasso selected {total} points"

    @property
    def cluster_gap(self) -> float:
        """The cluster tool's bridging distance, in metres: the loaded
        points' typical spacing times :attr:`cluster_gap_factor`. Measuring
        the spacing builds a KD-tree over what's loaded, so it's done once
        per cloud, on first need."""
        from . import analysis

        if self._cluster_spacing is None:
            self._cluster_spacing = analysis.point_spacing(self.view.coords)
        return self._cluster_spacing * self.cluster_gap_factor

    def set_cluster_gap_factor(self, factor: float) -> bool:
        """Change how much empty space a cluster click bridges, and re-run
        the last click at the new setting so its patch updates on screen.
        Returns whether there was a click to re-run."""
        factor = float(factor)
        if factor == self.cluster_gap_factor:
            return False
        self.cluster_gap_factor = factor
        return self.cluster.reapply()

    def reset_cluster_gap(self) -> None:
        """Back to the default gap, leaving the selection exactly as it is.

        The click sequence ends *first*: otherwise changing the gap would
        re-run the last click at the default and shrink the selection --
        right as the user switches to Move to press A on it."""
        self.cluster.end_chain()
        self.set_cluster_gap_factor(DEFAULT_CLUSTER_GAP_FACTOR)

    def _cluster_component(self, seed_label: int, gap: float):
        """(sorted point indices, component-id per index) for one tree's
        ``gap``-connected blobs — computed once per tree and gap, then cached.

        Keyed on the gap as well as the tree, because the workflow this
        serves walks up and down the gap steps: clicking the same spot again
        loosens it, [ and ] step either way, and the slider can land back
        where it was. Wiping the cache whenever the gap moved meant every one
        of those recomputed the whole tree's blobs from scratch — seconds
        each, and the same answer as two clicks ago.

        Which points *are* the tree is re-read every time and the entry is
        only used if it still agrees, because the other half of the workflow
        edits: select a patch, press A or N, click again. An entry cached
        before the edit describes a tree that no longer exists — clicking
        tree 1 after splitting half of it off as tree 3 kept handing back
        all of the original points, tree 3's included, so the next A pulled
        them straight back in. The check is two O(n) passes against a
        labelling the KD-tree work dwarfs, and it keeps the entries for the
        trees the edit did not touch — and the pre-edit ones across an undo,
        which restores the very array they were built for.
        """
        same = np.flatnonzero(self.cloud.labels == seed_label)
        key = (seed_label, float(gap))
        cached = self._cluster_cc.get(key)
        if cached is None or not np.array_equal(cached[0], same):
            cached = (same, self._label_blobs(self.view.coords[same], gap))
            self._cache_cluster(key, cached)
        return cached

    @staticmethod
    def _label_blobs(points: np.ndarray, gap: float) -> np.ndarray:
        """One component id per point -- the expensive bit, and the only bit
        the prefetch worker runs, so it touches nothing but its arguments."""
        from . import analysis

        if not points.size:
            return np.empty(0, np.int64)
        return analysis.connected_components_within(points, gap)

    def _cache_cluster(self, key, value) -> None:
        """Store a labelling, evicting the oldest once the cache is full.

        The GUI thread and every prefetch worker insert here. Python would
        make each assignment safe on its own, but the eviction reads the
        insertion order back out, and that cannot run beside another
        thread's insert.
        """
        with self._cluster_lock:
            self._cluster_cc.pop(key, None)  # re-insert: youngest again
            self._cluster_cc[key] = value
            while len(self._cluster_cc) > CLUSTER_CACHE_ENTRIES:
                self._cluster_cc.pop(next(iter(self._cluster_cc)))

    def _start_cluster_prefetch(self, seed_label: int, same: np.ndarray) -> None:
        """Warm the rest of the gap ladder for ``seed_label`` off the GUI thread.

        A click is the start of a sequence, not the end of one: the tool's
        whole idiom is to click again to loosen, and ] and the slider walk
        the same ladder. Each of those steps is a gap the cache has never
        been asked for, so every one of them used to stop the GUI thread for
        as long as the labelling took -- a fifth of a second on a 120k-point
        tree, more on a big one, once per step.

        So the click hands the ladder to a set of workers and the steps
        become cache hits. They get a snapshot -- the tree's indices, and
        its points copied out -- and compute on that alone, never reading
        the cloud or the view, because an edit may land while they run. A
        labelling that an edit has overtaken is not a hazard either way:
        _cluster_component re-reads the membership and ignores any entry
        that no longer matches.

        SciPy does this work with the GIL released (a Python thread beside
        query_pairs, the KD-tree build and connected_components measured
        88-103% of its idle rate), so the warming really is off the GUI
        thread rather than merely off the call stack.

        Clicking and then stepping to the top of the ladder every 400ms, the
        time the GUI thread spends blocked across the whole sequence:

        ==============  ==========  =========
        tree            before      after
        ==============  ==========  =========
        120k points     1.95s, 9/9  0.28s, 1/9
        300k points     5.23s, 9/9  1.22s, 2/9
        ==============  ==========  =========

        (steps over 100ms out of the nine). What is left is the first click,
        which has nothing warmed yet and never will, and on a big tree the
        step right behind it, which arrives before the first labelling the
        workers started has finished.
        """
        spacing = self._cluster_spacing
        if not self.cluster_prefetch or spacing is None or same.size <= 1:
            return
        if self._prefetch_label == seed_label and self._prefetch:
            if any(worker.is_alive() for worker in self._prefetch):
                return  # already walking this tree's ladder
        factors = list(CLUSTER_GAP_FACTORS)
        here = (
            factors.index(self.cluster_gap_factor)
            if self.cluster_gap_factor in factors
            else 0
        )
        # Loosening is the move the tool is built around, so the steps above
        # the click come first, nearest first; the tighter ones follow for []
        # and the slider going back down.
        order = factors[here + 1:] + factors[:here][::-1]

        def missing(factor: float) -> bool:
            cached = self._cluster_cc.get((seed_label, float(spacing * factor)))
            return cached is None or not np.array_equal(cached[0], same)

        # Every gap step calls back in here, and by the end of a ladder walk
        # there is nothing left to do; without this a step would keep
        # starting workers whose only job was to find the cache already warm.
        order = [factor for factor in order if missing(factor)]
        if not order:
            return
        points = self.view.coords[same]  # a copy, so the workers share it
        stop = threading.Event()
        self._prefetch_stop.set()  # whatever was running, stand down
        self._prefetch_stop = stop

        # One queue, taken in order, so the workers between them always hold
        # the nearest gaps the user has not reached rather than a fixed slice
        # each -- the tail of the ladder is the cheapest part and would
        # otherwise finish first while the next step along is still missing.
        pending, turn = iter(order), threading.Lock()

        def warm() -> None:
            while not stop.is_set():
                with turn:
                    factor = next(pending, None)
                if factor is None:
                    return
                if not missing(factor):
                    continue  # the GUI thread got to it first
                gap = spacing * factor
                try:
                    comp = self._label_blobs(points, gap)
                except Exception:  # noqa: BLE001
                    return  # a warm cache is a luxury; the click recomputes
                if not stop.is_set():
                    self._cache_cluster((seed_label, float(gap)), (same, comp))

        self._prefetch_label = seed_label
        self._prefetch = tuple(
            threading.Thread(
                target=warm, name=f"segfix-cluster-prefetch-{i}", daemon=True
            )
            for i in range(CLUSTER_PREFETCH_WORKERS)
        )
        for worker in self._prefetch:
            worker.start()

    def join_cluster_prefetch(self, timeout: float = 30.0) -> bool:
        """Block until the warming finishes. For tests and benchmarks -- the
        app never calls it, because waiting is the thing being removed."""
        deadline = time.monotonic() + timeout
        for worker in self._prefetch:
            worker.join(max(0.0, deadline - time.monotonic()))
        return not any(worker.is_alive() for worker in self._prefetch)

    def _stop_cluster_prefetch(self) -> None:
        """Tell the running prefetch to give up, without waiting for it.

        Not joined: the workers only check between gaps, so a join would
        hand the GUI thread exactly the stall the prefetch exists to remove.
        They are left to finish the gap they are on and fall out; anything
        they write after this lands in a cache that set_cloud has already
        replaced, or is ignored by _cluster_component's membership check.

        The handles stay so :meth:`join_cluster_prefetch` can still wait for
        them to wind down; it is clearing the label that makes the next
        click start a fresh set rather than assume these are still on it.
        """
        self._prefetch_stop.set()
        self._prefetch_label = None

    def _grow_cluster(self, seed: int) -> np.ndarray:
        """The cluster tool's payload (see :class:`~segfix.lasso.ClusterTool`):
        the blob of the seed's own tree that is connected to it across gaps
        no wider than :attr:`cluster_gap`, intersected with what's shown.

        Growing it is the gap's job — the slider, [ and ], or clicking the
        same spot again all loosen or tighten this one setting.
        """
        labels = self.cloud.labels
        seed_label = int(labels[seed])
        same, comp = self._cluster_component(seed_label, self.cluster_gap)
        self._start_cluster_prefetch(seed_label, same)
        if same.size <= 1:
            idx = same
        else:
            local = int(np.searchsorted(same, seed))
            idx = same[comp == comp[local]]

        shown = np.asarray(self.view.shown, dtype=bool)
        if shown.shape[0] == len(labels):
            idx = idx[shown[idx]]
        return idx

    def _on_cluster(self, indices: np.ndarray, additive: bool) -> None:
        total = self.view.select(indices, additive=additive)
        self.view.status = (
            f"Cluster selected {total} points at "
            f"{self.cluster_gap_factor:g}× spacing - click again to loosen"
        )

    def selected_indices(self) -> np.ndarray:
        return np.flatnonzero(self.view.selected_mask)

    def invert_selection(self) -> int:
        """Select every shown point that isn't selected, and vice versa.

        Bounded by ``view.shown``, not by the whole cloud: only shown points
        can be selected in the first place (the lasso and the cluster tool
        both intersect their result with it), so an inversion inside a cross
        section, a lasso section, or a tree isolated out of its neighbours
        has to stay inside that same subset — otherwise one keypress would
        hand you the entire cloud hiding behind the slab.

        With nothing selected this selects everything shown, which is the
        "select all visible" the section tools otherwise had no way to ask
        for. Returns how many points ended up selected.
        """
        size = len(self.view.coords)
        shown = np.asarray(self.view.shown, dtype=bool)
        if shown.shape[0] != size:
            shown = np.ones(size, dtype=bool)
        selected = self.view.selected_mask
        if selected.shape[0] != size:
            selected = np.zeros(size, dtype=bool)
        return self.view.select(shown & ~selected)

    def _after_edit(self, message: str) -> None:
        # Recolour only the points whose label just moved (the op records them
        # on the cloud); a whole-cloud recompute per keystroke is the slow path.
        refresh_view(
            self.view, self.cloud, self.faded_ids,
            changed=self.cloud.last_changed,
            class_colours=self.class_colours(),
        )
        self.view.selected = set()
        self.view.status = message


class SegFixWidget(QWidget):
    """The dock panel: the tree queue plus the few ops the loop needs."""

    HIDE_COL = 3
    FADE_COL = 4
    OVERLAY_W = 280  # fixed width of the "Current tree" box floating on the view
    # Rows of *unkeyed* neighbours visible before the list scrolls. The
    # keyed ones sit above it and never scroll, so this is three rather than
    # four and the box is the height it always was.
    NEIGHBOUR_ROWS = 3

    def __init__(self, controller: SegFixController):
        super().__init__()
        self.c = controller
        self.current: int | None = None  # the tree under review
        self.done_ids: set[int] = set()  # trees marked done in the table
        self.hidden_ids: set[int] = set()  # trees manually hidden via 👁
        # Whether the last _update_selection saw any points selected, so it
        # can spot the selection being cleared. Starts False: nothing to
        # reset before the gap slider even exists.
        self._had_selection = False
        self._table_updating = False
        # Optional fn()->None: set by scene mode so its own tree table (which
        # mirrors done-state from the same sidecar file) refreshes the moment
        # a tree is marked done here, instead of waiting for the next save.
        self.on_done_changed = None
        # Optional fn()->None: set by the app to open the class set-up
        # dialog (which needs the file, not just the loaded cloud).
        self.on_setup_classes = None
        # Optional fn(names)->None: set by the app to remember the class
        # names with the project when a class is added here.
        self.on_class_names_changed = None
        self._bbox_ids: set[int] = set()
        self._bbox_busy = False
        controller.on_cloud_changed = self._on_cloud_changed
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.info = QLabel()

        # -- the queue: one row per tree, click = review that tree ------
        # "Selected Tree + Neighbours": just the trees currently loaded into
        # the 3D view (the row picked in "All Trees" above, plus whatever's
        # spatially near it) — not the whole file, which that other table
        # covers. Title is immediately overwritten with the live done count
        # by _update_done_title() below.
        self.trees_box = QGroupBox("Selected Tree + Neighbours")
        tlay = QVBoxLayout(self.trees_box)
        tlay.setSpacing(4)
        tlay.addWidget(self.info)  # "N points · M trees" for the loaded cloud
        self.tree_table = QTableWidget(0, 5)
        self.tree_table.setHorizontalHeaderLabels(
            ["Done", "Tree ID", "Points", "Hide", "Fade"]
        )
        self.tree_table.horizontalHeaderItem(0).setToolTip(
            "Whether this tree has been marked reviewed"
        )
        self.tree_table.horizontalHeaderItem(self.HIDE_COL).setIcon(icon("hide"))
        self.tree_table.horizontalHeaderItem(self.HIDE_COL).setToolTip(
            "Hide this tree from the 3D view"
        )
        self.tree_table.horizontalHeaderItem(self.FADE_COL).setIcon(icon("fade"))
        self.tree_table.horizontalHeaderItem(self.FADE_COL).setToolTip(
            "Fade this tree in the 3D view - ghosted for context, but still "
            "shown and still selectable"
        )
        self.tree_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tree_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tree_table.setSortingEnabled(True)
        self.tree_table.verticalHeader().setVisible(False)
        header = self.tree_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.HIDE_COL, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.FADE_COL, QHeaderView.ResizeToContents)
        self.tree_table.setMinimumHeight(150)  # fills its splitter pane
        self.tree_table.itemSelectionChanged.connect(self._on_table_selection)
        self.tree_table.itemChanged.connect(self._on_tree_item_changed)
        tlay.addWidget(self.tree_table)

        nav_row = QHBoxLayout()
        prev_btn = QPushButton("Prev")
        prev_btn.setIcon(icon("prev"))
        prev_btn.setIconSize(QSize(18, 18))
        prev_btn.clicked.connect(lambda: self._step(-1))
        nav_row.addWidget(prev_btn)
        self.done_btn = QPushButton("✓ Done - next (Space)")
        self.done_btn.setIcon(icon("next"))
        self.done_btn.setIconSize(QSize(18, 18))
        self.done_btn.clicked.connect(self.on_done_next)
        nav_row.addWidget(self.done_btn, stretch=1)
        tlay.addLayout(nav_row)

        layout.addWidget(self.trees_box)
        self._build_inventory_box(layout)

        # Interaction / View / the two section tools live in a separate
        # horizontal bar (docked at the top of the window by app.py, not
        # stacked in this side panel) — they're about the canvas/viewport,
        # not the tree-review workflow the rest of this panel is organised
        # around. Each section box shows its controls at a constant size with
        # its on/off toggle inside, so switching one never resizes the bar.
        self.top_bar = QWidget()
        top_bar_row = QHBoxLayout(self.top_bar)
        # Left/right buffer so the first group's border/title isn't jammed
        # against the window edge.
        top_bar_row.setContentsMargins(10, 0, 6, 0)
        top_bar_row.setSpacing(6)

        # The top bar is a single thin strip: every group is one content row,
        # and the two section tools collapse to just their title/checkbox until
        # switched on. Keep it that way when adding controls here.

        # -- interaction mode: move (camera) vs lasso -----------------
        # The active mode shows as a coloured checked button — no separate
        # indicator. Exactly one is ever checked (see _uncheck_other_modes /
        # on_move_mode); section_draw_btn, built in the Lasso section box
        # below, is the fourth mutually-exclusive member.
        interaction_box = QGroupBox("Interaction")
        interaction = QHBoxLayout(interaction_box)
        interaction.setSpacing(3)
        self.move_btn = QPushButton("Move (Esc)")
        self.move_btn.setIcon(icon("move"))
        self.move_btn.setIconSize(QSize(18, 18))
        self.move_btn.setCheckable(True)
        self.move_btn.setChecked(True)
        self.move_btn.setToolTip("Drag rotates the view")
        self.move_btn.clicked.connect(self.on_move_mode)
        interaction.addWidget(self.move_btn)
        self.lasso_btn = QPushButton("Lasso (Q)")
        self.lasso_btn.setIcon(icon("lasso"))
        self.lasso_btn.setIconSize(QSize(18, 18))
        self.lasso_btn.setCheckable(True)
        self.lasso_btn.setToolTip("Drag selects points")
        self.lasso_btn.toggled.connect(self.on_toggle_lasso)
        interaction.addWidget(self.lasso_btn)
        self.tree_lasso_btn = QPushButton("Lasso tree (W)")
        self.tree_lasso_btn.setIcon(icon("lasso"))
        self.tree_lasso_btn.setIconSize(QSize(18, 18))
        self.tree_lasso_btn.setCheckable(True)
        self.tree_lasso_btn.setToolTip(
            "Freehand-select, but only points already belonging to the tree "
            "under review - grabs a clean patch out of a crowded/overlapping "
            "area without also picking up neighbouring trees."
        )
        self.tree_lasso_btn.toggled.connect(self.on_toggle_tree_lasso)
        interaction.addWidget(self.tree_lasso_btn)
        self.cluster_btn = QPushButton("Cluster (E)")
        self.cluster_btn.setIcon(icon("grow"))
        self.cluster_btn.setIconSize(QSize(18, 18))
        self.cluster_btn.setCheckable(True)
        self.cluster_btn.setToolTip(
            "Click a point to select the connected patch of the SAME tree's "
            "points around it (a spatial region grow) - e.g. to grab an "
            "over-segmented fragment or a wrongly-attached limb. Shift-click "
            "adds to the selection. Click the same spot again to loosen the "
            "gap a step - the same as ] or the gap slider."
        )
        self.cluster_btn.toggled.connect(self.on_toggle_cluster)
        # The gap setting hangs off a narrow arrow glued to the Cluster
        # button, opening a popover: a slider inline would roughly double
        # the Interaction box's width for a control touched now and then.
        cluster_pair = QHBoxLayout()
        cluster_pair.setSpacing(0)
        cluster_pair.addWidget(self.cluster_btn)
        self.cluster_gap_btn = QToolButton()
        self.cluster_gap_btn.setArrowType(Qt.ArrowType.DownArrow)
        self.cluster_gap_btn.setFixedWidth(16)
        self.cluster_gap_btn.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )
        self.cluster_gap_btn.setToolTip(
            "Cluster gap - how much empty space one click bridges ([ and ])"
        )
        cluster_pair.addWidget(self.cluster_gap_btn)
        interaction.addLayout(cluster_pair)
        self._build_cluster_gap_popover()
        # A repeat click on the same spot is one notch looser -- routed
        # through the slider so the two can never disagree.
        self.c.cluster.on_repeat = lambda: self.step_cluster_gap(1)
        self.cluster_gap_btn.clicked.connect(
            lambda: self._toggle_popover(
                self._cluster_gap_popover, self.cluster_gap_btn
            )
        )
        interaction.addStretch(1)
        # Each mode button lights up in its own colour while active.
        for btn, bg, fg in (
            (self.move_btn, "#2c4a33", "#9fd8a8"),
            (self.lasso_btn, "#7a6a20", "#ffe066"),
            (self.tree_lasso_btn, "#6a3a7a", "#e0b3ff"),
            (self.cluster_btn, "#20506a", "#a8d8ff"),
        ):
            btn.setStyleSheet(
                f"QPushButton:checked {{ background: {bg}; color: {fg}; "
                "font-weight: bold; }"
            )
        top_bar_row.addWidget(interaction_box)

        # -- view: what's visible/selectable, and how big it renders ---
        # One row: unassigned toggle · declutter buttons · point size. The
        # "others" buttons act on the whole loaded set, not the selection, so
        # they belong here, not with the "Current tree" fix actions.
        view_box = QGroupBox("View")
        view = QVBoxLayout(view_box)
        view.setSpacing(3)
        view_row = QHBoxLayout()
        view_row.setSpacing(6)
        # A plain checkbox, like the Cross section / Lasso section "On"
        # toggles it sits next to — not the odd-one-out checkable button it
        # used to be.
        self.show_unassigned = QCheckBox("Show unassigned (F)")
        self.show_unassigned.setChecked(True)
        self.show_unassigned.setToolTip(
            "Show or hide the unassigned + noise points"
        )
        self.show_unassigned.toggled.connect(self._on_show_unassigned)
        view_row.addWidget(self.show_unassigned)
        # Checkboxes, like the toggle beside them: all three turn something
        # in the view on and off and stay on until turned off, and as push
        # buttons the two "others" ones showed nothing of that state — with
        # every neighbour hidden, "Hide others" looked exactly as it did
        # with none hidden, and only the status line said which.
        self.hide_others_cb = QCheckBox("Hide others (G)")
        self.hide_others_cb.setToolTip(
            "Hide every other loaded tree, leaving the tree under review"
        )
        self.hide_others_cb.toggled.connect(self._set_hide_others)
        view_row.addWidget(self.hide_others_cb)
        self.fade_others_cb = QCheckBox("Fade others (Shift+G)")
        self.fade_others_cb.setToolTip(
            "Ghost every other loaded tree - still shown, still selectable"
        )
        self.fade_others_cb.toggled.connect(self._set_fade_others)
        view_row.addWidget(self.fade_others_cb)
        view_row.addStretch(1)
        view.addLayout(view_row)

        top_bar_row.addWidget(view_box)

        # Point size lives as a small floating control in the canvas' top-left
        # corner (not in this bar) so it's next to the cloud it sizes.
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(0.5, 30.0)
        self.size_spin.setDecimals(1)
        self.size_spin.setSingleStep(0.5)
        self.size_spin.setSuffix(" px")
        self.size_spin.setValue(3.0)
        self.size_spin.valueChanged.connect(self._on_point_size)
        self._build_point_size_overlay()

        # -- cross section: an interactive slab along one axis; while on,
        # only points inside it are shown and selectable (folded into the
        # same layer.shown mechanism as the per-tree hide checkboxes).
        # The box in the bar stays tiny and fixed — just the On toggle and a
        # "Slab…" button; the axis/extent controls live in a floating popover
        # so switching or adjusting the tool never resizes or crowds the bar.
        self.cross_box = QGroupBox("Cross section (C)")
        cross = QHBoxLayout(self.cross_box)
        cross.setContentsMargins(6, 2, 6, 4)
        self.cross_enable = QCheckBox("On")
        self.cross_enable.setToolTip(
            "Slice the cloud to a slab along one axis. While on, only points "
            "inside the slab are shown or selectable."
        )
        self.cross_enable.toggled.connect(self._on_cross_section_toggled)
        cross.addWidget(self.cross_enable)
        slab_btn = QPushButton("Slab…")
        slab_btn.setToolTip("Axis and slab extent")
        cross.addWidget(slab_btn)

        self._cross_popover = QWidget(self, Qt.Popup)
        cp = QVBoxLayout(self._cross_popover)
        cross_top = QHBoxLayout()
        cross_top.addWidget(QLabel("Axis"))
        self.cross_axis_combo = QComboBox()
        self.cross_axis_combo.addItems(["X", "Y", "Z"])
        self.cross_axis_combo.setCurrentIndex(2)  # Z: height, usually most useful
        self.cross_axis_combo.currentIndexChanged.connect(self._on_cross_axis_changed)
        cross_top.addWidget(self.cross_axis_combo)
        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Widen the slab back out to the full extent")
        reset_btn.clicked.connect(self._on_cross_reset)
        cross_top.addWidget(reset_btn)
        self.cross_range_label = QLabel()
        cross_top.addWidget(self.cross_range_label, stretch=1)
        cp.addLayout(cross_top)
        slab_row = QHBoxLayout()
        slab_row.addWidget(QLabel("Min"))
        self.cross_min_slider = QSlider(Qt.Horizontal)
        self.cross_min_slider.setMinimumWidth(160)
        self.cross_min_slider.valueChanged.connect(self._on_cross_range_changed)
        slab_row.addWidget(self.cross_min_slider)
        slab_row.addWidget(QLabel("Max"))
        self.cross_max_slider = QSlider(Qt.Horizontal)
        self.cross_max_slider.setMinimumWidth(160)
        self.cross_max_slider.valueChanged.connect(self._on_cross_range_changed)
        slab_row.addWidget(self.cross_max_slider)
        cp.addLayout(slab_row)
        slab_btn.clicked.connect(
            lambda: self._toggle_popover(self._cross_popover, slab_btn)
        )
        top_bar_row.addWidget(self.cross_box)
        self._cross_lo, self._cross_hi = 0.0, 1.0
        self._reset_cross_section_range()

        # -- lasso section: same idea as cross section, but the kept
        # region is a hand-drawn outline instead of an axis-aligned slab.
        # Draw arms the shared lasso tool in "section" mode (see
        # SegFixController.on_lasso_section); the outline it produces is
        # a one-shot boolean mask, not a live screen region, so it stays
        # put as the camera moves — same persistence model as the slab.
        # No sliders here, so it fits inline; the kept-count goes to the box
        # tooltip and the status bar rather than a width-hungry label.
        self.lasso_section_box = QGroupBox("Lasso section (Shift+C)")
        lsec = QHBoxLayout(self.lasso_section_box)
        lsec.setContentsMargins(6, 2, 6, 4)
        self.lasso_section_enable = QCheckBox("On")
        self.lasso_section_enable.setToolTip(
            "Keep only points inside a hand-drawn outline. Like cross section, "
            "but the kept region is whatever shape you draw."
        )
        self.lasso_section_enable.toggled.connect(self._on_lasso_section_toggled)
        lsec.addWidget(self.lasso_section_enable)
        self.section_draw_btn = QPushButton("Draw (Shift+Q)")
        self.section_draw_btn.setIcon(icon("lasso"))
        self.section_draw_btn.setIconSize(QSize(18, 18))
        self.section_draw_btn.setCheckable(True)
        self.section_draw_btn.setToolTip(
            "Drag on the canvas to outline the region to keep"
        )
        self.section_draw_btn.setStyleSheet(
            "QPushButton:checked { background: #3a5a7a; color: #a8d4ff; "
            "font-weight: bold; }"
        )
        self.section_draw_btn.toggled.connect(self.on_toggle_lasso_section)
        # Keep the canvas' double-click behaviour in step with the mode: pivot
        # recentre in plain move mode, tool input while a selection tool is on.
        for _mode_btn in (
            self.lasso_btn, self.tree_lasso_btn,
            self.cluster_btn, self.section_draw_btn,
        ):
            _mode_btn.toggled.connect(
                lambda _checked: self._refresh_double_click_mode()
            )
        lsec.addWidget(self.section_draw_btn)
        section_reset_btn = QPushButton("Reset")
        section_reset_btn.setToolTip("Clear the outline - show every point again")
        section_reset_btn.clicked.connect(self._on_lasso_section_reset)
        lsec.addWidget(section_reset_btn)
        self.lasso_section_label = QLabel(self)  # not shown; feeds the tooltip
        self.lasso_section_label.hide()
        top_bar_row.addWidget(self.lasso_section_box)
        self._lasso_section_mask: np.ndarray | None = None
        self._update_lasso_section_label()

        # Kept so the "Point class" box can be docked in here on request —
        # it floats over the canvas by default (see _set_class_in_top_bar).
        self._top_bar_row = top_bar_row
        self._class_bar_slot = top_bar_row.count()
        top_bar_row.addStretch()

        # -- fixing the current tree --------------------------------------
        # This box floats over the 3D view as a column pinned to its right
        # edge (parented to the canvas, not stacked in the side panel), so
        # the fix actions sit next to the points they act on. Positioned by
        # _position_current_tree_overlay, kept pinned on canvas resize.
        sel_box = QGroupBox("Current tree", self.c.view.native)
        sel_box.setObjectName("currentTreeOverlay")
        sel_box.setFixedWidth(self.OVERLAY_W)  # styled by _apply_overlay_theme
        self._current_tree_overlay = sel_box
        sel = QVBoxLayout(sel_box)
        sel.setSpacing(3)
        crow = QHBoxLayout()
        self.current_swatch = QLabel()
        self.current_swatch.setFixedSize(18, 18)
        crow.addWidget(self.current_swatch)
        self.current_label = QLabel()
        crow.addWidget(self.current_label, stretch=1)
        sel.addLayout(crow)
        self.sel_info = QLabel()
        sel.addWidget(self.sel_info)
        # Next to what it acts on: it reads the selection line above it and
        # rewrites it. Stays enabled with nothing selected, where it means
        # "select everything shown".
        self.invert_btn = self._button(
            sel, "Invert selection (B)", self.on_invert_selection, "invert"
        )
        self.invert_btn.setToolTip(
            "Select every shown point that isn't selected - and only shown "
            "points, so a cross section or lasso section still bounds it"
        )

        sel.addWidget(
            self._subheading("Move selection into a tree")
        )
        self.add_btn = QPushButton()
        self.add_btn.setIcon(icon("reassign"))
        self.add_btn.setIconSize(QSize(18, 18))
        self.add_btn.clicked.connect(self.on_add)
        sel.addWidget(self.add_btn)
        # The keyed neighbours sit above the scroll area, always visible:
        # they are the ones a key can reach, so scrolling them out of sight
        # hides the fast path. The rest scroll below.
        self.keyed_grid = QGridLayout()
        self.keyed_grid.setContentsMargins(0, 0, 0, 0)
        self.keyed_grid.setSpacing(4)
        sel.addLayout(self.keyed_grid)
        # The neighbour buttons go in a 2-column grid inside a fixed-height
        # scroll area, so the whole "Current tree" box stays one size no
        # matter how many neighbours the current tree has — extras scroll
        # rather than stretching the box (and, before, drawing over the
        # buttons below it).
        self.neighbour_grid = QGridLayout()
        self.neighbour_grid.setContentsMargins(0, 0, 0, 0)
        self.neighbour_grid.setSpacing(4)
        neighbour_host = QWidget()
        neighbour_host.setLayout(self.neighbour_grid)
        neighbour_host.setAutoFillBackground(False)
        self.neighbour_scroll = QScrollArea()
        self.neighbour_scroll.setObjectName("neighbourScroll")
        self.neighbour_scroll.setWidget(neighbour_host)
        self.neighbour_scroll.setWidgetResizable(True)
        self.neighbour_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.neighbour_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.neighbour_scroll.viewport().setAutoFillBackground(False)
        row_h = self._neighbour_row_height()
        self.neighbour_scroll.setFixedHeight(
            self.NEIGHBOUR_ROWS * row_h + (self.NEIGHBOUR_ROWS - 1) * 4
        )
        sel.addWidget(self.neighbour_scroll)
        # Under the buttons it decides the contents of, not above them: this
        # is what says which trees are listed at all.
        reach_row = QHBoxLayout()
        reach_label = QLabel("Trees within")
        reach_label.setToolTip(
            "Which trees are listed above: those whose points come within "
            "this distance of the current tree's points."
        )
        reach_row.addWidget(reach_label)
        self.focus_margin = QDoubleSpinBox()
        self.focus_margin.setRange(0.1, 50.0)
        self.focus_margin.setDecimals(1)
        self.focus_margin.setSingleStep(0.5)
        self.focus_margin.setValue(1.0)
        self.focus_margin.setSuffix(" m")
        self.focus_margin.setToolTip(
            "Neighbour reach: trees whose points come within this distance "
            "of the current tree's points are listed above."
        )
        self.focus_margin.valueChanged.connect(self._update_neighbour_picker)
        reach_row.addWidget(self.focus_margin)
        reach_row.addWidget(QLabel("of this tree"))
        reach_row.addStretch(1)
        sel.addLayout(reach_row)

        sel.addWidget(self._subheading("Remove selection from its tree"))
        self.split_btn = self._button(
            sel, "Split off as new tree (S)", self.on_create_new, "new"
        )
        self._button(sel, "Unassign (D)", self.on_unassign, "unassign")
        self._button(sel, "Noise (X)", self.on_noise, "noise")
        sel_box.show()
        sel_box.raise_()
        self._build_class_overlay()
        self._position_current_tree_overlay()
        self.c.view.canvas.events.resize.connect(
            self._position_current_tree_overlay
        )

        # Undo / Redo / Save Project used to live in a "Session" group box
        # here; they're on the window menu bar now (see app._build_menus),
        # driven by the same on_undo / on_redo / on_save methods.

        # No trailing stretch: this panel is a splitter pane now, so let the
        # "Selected Tree + Neighbours" box (and its table) fill the height.
        layout.setStretch(0, 1)
        # Style the two canvas overlays for the current theme and restyle
        # them whenever it changes (their backing/text can't ride the Qt
        # palette — they're translucent panels over the 3-D view).
        theme.subscribe(self._apply_overlay_theme)
        # Freeze the "Current tree" box at its natural size now (empty
        # grids, theme applied) so it never resizes later — the neighbour
        # scroll area absorbs every change in neighbour count. The keyed
        # buttons are outside that scroll area, so their rows have to be
        # reserved here or they push the box over its own edge.
        keyed_rows = -(-NEIGHBOUR_KEYS // 2)  # two buttons to a row
        row_h = self._neighbour_row_height()
        self._current_tree_overlay.setFixedSize(
            self.OVERLAY_W,
            self._current_tree_overlay.sizeHint().height() + 8
            + keyed_rows * (row_h + 4),
        )
        self._on_cloud_changed()

    # -- point classes ------------------------------------------------
    CLASS_ROWS = 3  # rows of class buttons shown before the box scrolls

    def _build_class_overlay(self) -> None:
        """The "Point class" box, floating under "Current tree": a button
        per class to give the selection that class, the colour-by-class
        toggle, and the way to add a class or set the field up."""
        box = QGroupBox("Point class", self.c.view.native)
        box.setObjectName("classOverlay")
        box.setFixedWidth(self.OVERLAY_W)
        self._class_overlay = box
        lay = QVBoxLayout(box)
        lay.setSpacing(3)
        self.class_color_cb = QCheckBox("Colour by class (Shift+F)")
        self.class_color_cb.setToolTip(
            "Colour points by their class instead of by tree"
        )
        self.class_color_cb.toggled.connect(self._on_color_by_class)
        lay.addWidget(self.class_color_cb)
        self.class_info = QLabel()
        self.class_info.setWordWrap(True)
        lay.addWidget(self.class_info)
        self.class_grid = QGridLayout()
        self.class_grid.setContentsMargins(0, 0, 0, 0)
        self.class_grid.setSpacing(4)
        host = QWidget()
        host.setLayout(self.class_grid)
        host.setAutoFillBackground(False)
        self.class_scroll = QScrollArea()
        self.class_scroll.setObjectName("classScroll")
        self.class_scroll.setWidget(host)
        self.class_scroll.setWidgetResizable(True)
        self.class_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.class_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.class_scroll.viewport().setAutoFillBackground(False)
        row_h = self._neighbour_row_height()
        self.class_scroll.setFixedHeight(
            self.CLASS_ROWS * row_h + (self.CLASS_ROWS - 1) * 4
        )
        lay.addWidget(self.class_scroll)
        buttons = QHBoxLayout()
        self.new_class_btn = QPushButton("New class…")
        self.new_class_btn.setToolTip(
            "Add a class - and give it to the selection, if there is one"
        )
        self.new_class_btn.clicked.connect(self.on_new_class)
        buttons.addWidget(self.new_class_btn)
        self.setup_class_btn = QPushButton("Set up…")
        self.setup_class_btn.setToolTip(
            "Choose the file's class field and name its classes"
        )
        self.setup_class_btn.clicked.connect(self._on_setup_classes)
        buttons.addWidget(self.setup_class_btn)
        lay.addLayout(buttons)
        box.show()
        box.raise_()
        self._class_btns: list[QPushButton] = []
        self._class_codes: list[int] = []

    #: Rows of class buttons before the box scrolls, floating and docked.
    #: Fewer in the bar: the whole strip grows to its tallest box, and a
    #: half-height scroll there costs less than pushing the canvas down.
    CLASS_ROWS_DOCKED = 2
    CLASS_BAR_W = 360

    @property
    def class_in_top_bar(self) -> bool:
        return self._class_docked

    def _set_class_in_top_bar(self, docked: bool) -> None:
        """Move the Point class box between the canvas and the top bar.

        Driven by Preferences ▸ Point Class Box ▸ In the top bar, not by a
        tick in the View group: it is a choice about where the window puts
        things rather than about what the view shows, and the bar is wide
        enough as it is. On a 1920-wide screen one more checkbox there ran
        the section boxes under the side panel.

        Reparented, not duplicated: one set of buttons, one colour-by-class
        state, so there is no second copy to keep in step. Floating it is
        the default — the box is next to the points it labels there — but it
        covers the view, which on a laptop screen is most of it.
        """
        box = getattr(self, "_class_overlay", None)
        if box is None or docked == self._class_docked:
            return
        self._class_docked = docked
        row_h = self._neighbour_row_height()
        if docked:
            rows = self.CLASS_ROWS_DOCKED
            box.setParent(None)
            box.setStyleSheet("")  # a plain group box, like its neighbours
            # A constant width, like the section boxes beside it: left to
            # the layout the box is squeezed to whatever is spare and the
            # class names elide. Wider than it floats, because two buttons
            # sit side by side and the bar has room the canvas doesn't.
            box.setFixedWidth(self.CLASS_BAR_W)
            self._top_bar_row.insertWidget(self._class_bar_slot, box)
        else:
            rows = self.CLASS_ROWS
            self._top_bar_row.removeWidget(box)
            box.setParent(self.c.view.native)
            box.setFixedWidth(self.OVERLAY_W)
            self._apply_overlay_theme(theme.current())
        self.class_scroll.setFixedHeight(rows * row_h + (rows - 1) * 4)
        box.show()
        if not docked:
            box.raise_()
            self._position_current_tree_overlay()

    #: True while the box is docked in the top bar rather than floating.
    _class_docked = False

    def _on_setup_classes(self) -> None:
        if self.on_setup_classes is not None:
            self.on_setup_classes()

    def _rebuild_class_buttons(self) -> None:
        """One button per named class, in value order; the first
        :data:`CLASS_KEYS` wear the Ctrl+number that presses them."""
        while self.class_grid.count():
            w = self.class_grid.takeAt(0).widget()
            if w is not None:
                # Hidden now, not just when deleteLater gets round to it: a
                # rename would otherwise show the old buttons until then.
                w.hide()
                w.deleteLater()
        has_field = self.c.class_field is not None
        self._class_codes = sorted(self.c.class_names) if has_field else []
        self._class_btns = []
        colours = class_colors(self.c.class_names)
        text_colour = self.palette().buttonText().color()
        for i, code in enumerate(self._class_codes):
            name = self.c.class_names[code]
            r, g, b = (int(v * 255) for v in colours[code])
            btn = QPushButton(f" {name}")
            keyed = i < CLASS_KEYS
            if keyed:
                btn.setIcon(QIcon(key_badge(str(i + 1), text_colour)))
                btn.setIconSize(QSize(16, 16))
            btn.setStyleSheet(
                f"QPushButton {{ border: 2px solid rgb({r},{g},{b}); "
                "border-radius: 3px; padding: 2px 4px; text-align: left; }"
            )
            key = f" (Ctrl+{i + 1})" if keyed else ""
            btn.setToolTip(
                f"Give the selection class {name} (value {code}){key}. "
                "With nothing selected, the whole current tree."
            )
            btn.clicked.connect(
                lambda _checked=False, c=code: self.on_set_class(c)
            )
            self.class_grid.addWidget(btn, i // 2, i % 2)
            self._class_btns.append(btn)
        self.class_color_cb.setEnabled(has_field)
        self.new_class_btn.setEnabled(has_field)
        self.class_scroll.setVisible(has_field)
        self.setup_class_btn.setVisible(self.on_setup_classes is not None)
        if has_field:
            self.class_info.setText(f"Field: {self.c.class_field}")
        else:
            self.class_info.setText(
                "No class field - Set up… to edit leaf, wood, ground…"
            )
        if not self._class_docked:
            self._class_overlay.adjustSize()
            self._position_current_tree_overlay()

    def _on_color_by_class(self, checked: bool) -> None:
        self.c.color_by_class = checked
        self._apply_transparency()
        self.c.view.status = (
            "Coloured by point class" if checked else "Coloured by tree"
        )

    def toggle_color_by_class(self) -> None:
        """Shift+F: flip "Colour by class" (when there are classes)."""
        if self.class_color_cb.isEnabled():
            self.class_color_cb.toggle()
        else:
            self.c.view.status = "No class field - Set up… in Point class"

    def on_set_class(self, code: int) -> None:
        """Selection → class ``code``; with nothing selected, the whole
        current tree (the way D and X act on it)."""
        if self.c.cloud.classes is None:
            self.c.view.status = "No class field - Set up… in Point class"
            return
        idx = self._selection_or_current_tree("classify")
        if idx is None:
            return
        name = self.c.class_names.get(code, str(code))
        self._apply(ops.set_class(self.c.cloud, idx, code, name))

    def set_nth_class(self, n: int) -> None:
        """Ctrl+``n``: the ``n``-th class button, counting from 1."""
        if self.c.cloud.classes is None:
            self.c.view.status = "No class field - Set up… in Point class"
            return
        if n > len(self._class_codes):
            self.c.view.status = (
                f"Only {len(self._class_codes)} class"
                f"{'es' if len(self._class_codes) != 1 else ''}"
            )
            return
        self.on_set_class(self._class_codes[n - 1])

    def on_new_class(self) -> None:
        """Name a new class, numbered one above the highest there is, and
        give it to the selection if there is one."""
        from qtpy.QtWidgets import QInputDialog

        if self.c.cloud.classes is None:
            return
        name, ok = QInputDialog.getText(self, "New class", "Class name:")
        name = name.strip()
        if not ok or not name:
            return
        if name.lower() in {n.lower() for n in self.c.class_names.values()}:
            self.c.view.status = f"There is already a class called {name}"
            return
        code = self.new_class_code()
        if code is None:
            QMessageBox.warning(
                self, "No room for another class",
                f"The field {self.c.class_field} can't hold another "
                f"class value (it tops out at {self.c.class_range[1]}).",
            )
            return
        self.c.class_names[code] = name
        if self.on_class_names_changed is not None:
            self.on_class_names_changed(dict(self.c.class_names))
        self._rebuild_class_buttons()
        if self.c.selected_indices().size:
            self.on_set_class(code)
        else:
            self.c.view.status = f"Added class {name} (value {code})"

    def new_class_code(self) -> int | None:
        """The value a new class gets: one above every value named or
        present in the loaded points, or None if the field can't hold it."""
        used = set(self.c.class_names)
        if self.c.cloud.classes is not None and self.c.cloud.classes.size:
            used.add(int(self.c.cloud.classes.max()))
        code = max(used) + 1 if used else 0
        lo, hi = self.c.class_range or (0, np.iinfo(np.int32).max)
        code = max(code, lo)
        return code if code <= hi else None

    # -- helpers -----------------------------------------------------
    def _apply_overlay_theme(self, mode: str) -> None:
        col = theme.panel_colors(mode)
        ps = getattr(self, "_point_size_overlay", None)
        if ps is not None:
            ps.setStyleSheet(
                "QWidget#pointSizeOverlay { background: " + col["bg"]
                + "; border-radius: 5px; } "
                "QWidget#pointSizeOverlay QLabel { color: " + col["text"] + "; }"
            )
        for box in (getattr(self, "_current_tree_overlay", None),
                    getattr(self, "_class_overlay", None)):
            if box is None:
                continue
            if box is getattr(self, "_class_overlay", None) and self._class_docked:
                continue  # docked in the bar: styled like the boxes beside it
            name = box.objectName()
            box.setStyleSheet(
                f"QGroupBox#{name} {{"
                " background: " + col["bg"] + ";"
                " border: 1px solid " + col["border"] + ";"
                " border-radius: 6px; margin-top: 8px; }"
                f"QGroupBox#{name}::title {{"
                " subcontrol-origin: margin; left: 8px; padding: 0 4px;"
                " color: " + col["subtext"] + "; }"
                f"QGroupBox#{name} QLabel, QGroupBox#{name} QCheckBox "
                "{ color: " + col["text"] + "; }"
                # the button scrollers must not paint their own rectangle
                # over the translucent panel
                "QScrollArea#neighbourScroll, "
                "QScrollArea#neighbourScroll > QWidget > QWidget, "
                "QScrollArea#classScroll, "
                "QScrollArea#classScroll > QWidget > QWidget "
                "{ background: transparent; }"
            )
        # Re-tint the "done" rows for the new theme (dark vs pale green).
        tt = getattr(self, "tree_table", None)
        if tt is not None:
            tt.blockSignals(True)
            for row in range(tt.rowCount()):
                self._style_done_row(row, self._row_id(row) in self.done_ids)
            tt.blockSignals(False)

    def _button(self, parent_layout, text, slot, icon_name=None):
        btn = QPushButton(text)
        if icon_name:
            btn.setIcon(icon(icon_name))
            btn.setIconSize(QSize(18, 18))
        btn.clicked.connect(slot)
        parent_layout.addWidget(btn)
        return btn

    def _toggle_popover(self, popover: QWidget, anchor) -> None:
        """Show/hide a ``Qt.Popup`` widget just under ``anchor`` — a floating
        detail panel that costs the toolbar no layout space."""
        if popover.isVisible():
            popover.hide()
            return
        popover.adjustSize()
        popover.move(anchor.mapToGlobal(anchor.rect().bottomLeft()))
        popover.show()

    # -- inventory (field stem map) --------------------------------------
    #: Columns of the candidates table.
    STEM_COL, DIST_COL, DH_COL, DDBH_COL, SCORE_COL = range(5)

    def _build_inventory_box(self, layout) -> None:
        """The candidates table: which measured stem this tree might be.

        Hidden until a stem map is loaded, so a project that never touches
        field data looks exactly as it did.
        """
        self.inventory_box = QGroupBox("Inventory match")
        box = QVBoxLayout(self.inventory_box)
        box.setSpacing(3)
        self.inventory_label = QLabel("No stem map loaded")
        self.inventory_label.setWordWrap(True)
        box.addWidget(self.inventory_label)

        self.inventory_table = QTableWidget(0, 5)
        self.inventory_table.setHorizontalHeaderLabels(
            ["Stem", "Dist", "ΔH", "ΔDBH", "Score"]
        )
        self.inventory_table.verticalHeader().setVisible(False)
        self.inventory_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.inventory_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.inventory_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.inventory_table.setMaximumHeight(150)
        header = self.inventory_table.horizontalHeader()
        header.setSectionResizeMode(self.STEM_COL, QHeaderView.Stretch)
        # The numbers are what the ranking is read off; at a default column
        # width the Score scrolled off the right-hand edge of the panel.
        for column in (self.DIST_COL, self.DH_COL, self.DDBH_COL,
                       self.SCORE_COL):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        # Selecting a row lights that stem up in the 3D view, so the ranking
        # can be checked against the trunk rather than taken on trust.
        self.inventory_table.itemSelectionChanged.connect(self._on_candidate_row)
        self.inventory_table.cellDoubleClicked.connect(
            lambda *_: self.on_link_stem()
        )
        box.addWidget(self.inventory_table)

        row = QHBoxLayout()
        self.link_btn = QPushButton("Link")
        self.link_btn.setToolTip(
            "Record the selected stem as this tree's match (double-click a "
            "row does the same)"
        )
        self.link_btn.clicked.connect(self.on_link_stem)
        row.addWidget(self.link_btn)
        self.unlink_btn = QPushButton("Unlink")
        self.unlink_btn.clicked.connect(self.on_unlink_stem)
        row.addWidget(self.unlink_btn)
        row.addStretch(1)
        box.addLayout(row)

        self.inventory_box.hide()
        layout.addWidget(self.inventory_box)

        #: Loaded stem map and how it sits on the cloud.
        self.stems: list = []
        self.alignment = inventory.Alignment()
        #: tree id -> stem id, saved in the project sidecar.
        self.stem_links: dict[int, str] = {}
        #: Where the loaded stem map came from, for the export's default name.
        self.stem_map_path: str | None = None
        self._candidates: list = []
        self._stem_geometry = None

    def set_stem_map(self, stems, alignment=None) -> None:
        """Show a loaded stem map: draw it, and rank it for the current tree."""
        self.stems = list(stems)
        if alignment is not None:
            self.alignment = alignment
        self.inventory_box.setVisible(bool(self.stems))
        self._stem_geometry = None
        self._draw_stems()
        self._refresh_candidates()

    def set_alignment(self, alignment) -> None:
        self.alignment = alignment
        self._stem_geometry = None
        self._draw_stems()
        self._refresh_candidates()

    def _ground_z(self) -> float:
        """Where to stand the drawn stems: the loaded cloud's floor.

        A stem map carries no z at all, and a cylinder drawn from z=0 in a
        cloud whose ground is at 120 m is somewhere underground. The 1st
        percentile rather than the minimum, for the same reason the DBH fit
        uses it: one stray point below the plot shouldn't drop every stem.
        """
        coords = self.c.view.coords
        if not len(coords):
            return 0.0
        return float(np.percentile(coords[:, 2], 1.0))

    def _draw_stems(self) -> None:
        if not self.stems:
            self.c.view.clear_stems()
            return
        if self._stem_geometry is None:
            self._stem_geometry = inventory.stem_geometry(
                self.stems, self.alignment, base_z=self._ground_z()
            )
        segments, owner = self._stem_geometry
        colors = np.tile(np.array(STEM_COLOR, np.float32), (len(segments), 1))
        linked = self.stem_links.get(self.current)
        highlight = self._highlight_stem
        for stem_id, color in ((linked, LINKED_STEM_COLOR),
                               (highlight, ACTIVE_STEM_COLOR)):
            if stem_id is None:
                continue
            index = next(
                (i for i, s in enumerate(self.stems) if s.stem_id == stem_id),
                None,
            )
            if index is not None:
                colors[owner == index] = color
        self._stem_colors = colors  # what was drawn, for tests and redraws
        self.c.view.set_stems(segments, colors)

    _highlight_stem = None

    def tree_stats(self, tree_id: int | None = None):
        """Position, height and fitted DBH for a loaded tree."""
        tree_id = self.current if tree_id is None else tree_id
        if tree_id is None or not len(self.c.view.coords):
            return None
        return inventory.stats_from_points(
            self.c.view.coords, self.c.cloud.labels, tree_id
        )

    def _refresh_candidates(self) -> None:
        """Re-rank the stem map against the tree under review."""
        if not self.stems:
            return
        stats = self.tree_stats()
        self._candidates = []
        if stats is not None:
            self._candidates = inventory.candidates(
                stats, self.stems, self.alignment,
                exclude={
                    stem_id for tree, stem_id in self.stem_links.items()
                    if tree != self.current
                },
            )
        self._fill_candidate_table(stats)

    def _fill_candidate_table(self, stats) -> None:
        table = self.inventory_table
        table.setRowCount(len(self._candidates))
        linked = self.stem_links.get(self.current)
        for row, candidate in enumerate(self._candidates):
            cells = [
                candidate.stem.label,
                f"{candidate.distance:.1f} m",
                "-" if candidate.height_diff is None else f"{candidate.height_diff:+.1f}",
                "-" if candidate.dbh_diff is None else f"{candidate.dbh_diff * 100:+.0f} cm",
                f"{candidate.score:.2f}",
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if candidate.stem.stem_id == linked:
                    item.setBackground(QColor(60, 110, 70))
                table.setItem(row, column, item)
            table.item(row, self.STEM_COL).setToolTip(
                "Scored on " + ", ".join(candidate.used)
            )
        self.inventory_label.setText(self._inventory_summary(stats, linked))
        self.unlink_btn.setEnabled(linked is not None)
        self.link_btn.setEnabled(bool(self._candidates))

    def _inventory_summary(self, stats, linked) -> str:
        if self.current is None:
            return f"{len(self.stems)} stems loaded - pick a tree to match"
        if stats is None:
            return "Load the tree's points to match it"
        measured = f"{stats.height:.1f} m"
        if stats.dbh is not None:
            flag = "" if (stats.dbh_quality or 0) <= analysis.GOOD_FIT else " (poor fit)"
            measured += f", DBH {stats.dbh * 100:.0f} cm{flag}"
        matched = f" - matched to {linked}" if linked else ""
        return f"Tree {self.current}: {measured}{matched}"

    def _on_candidate_row(self) -> None:
        rows = {i.row() for i in self.inventory_table.selectedIndexes()}
        if not rows:
            return
        candidate = self._candidates[min(rows)]
        self._highlight_stem = candidate.stem.stem_id
        self._draw_stems()

    def on_link_stem(self) -> None:
        """Record the selected candidate as this tree's inventory match."""
        if self.current is None or not self._candidates:
            return
        rows = {i.row() for i in self.inventory_table.selectedIndexes()}
        candidate = self._candidates[min(rows) if rows else 0]
        # One stem belongs to one tree: linking it here takes it off whatever
        # it was linked to, rather than leaving the plot double-counted.
        for tree_id, stem_id in list(self.stem_links.items()):
            if stem_id == candidate.stem.stem_id:
                del self.stem_links[tree_id]
        self.stem_links[self.current] = candidate.stem.stem_id
        self._save_progress()
        self._refresh_candidates()
        # The link is the state now, so the stem goes green rather than
        # staying lit as "the row I am looking at".
        self._highlight_stem = None
        self._draw_stems()
        self.c.view.status = (
            f"Tree {self.current} matched to stem {candidate.stem.stem_id} "
            f"({candidate.distance:.1f} m away, score {candidate.score:.2f})"
        )

    def on_unlink_stem(self) -> None:
        if self.stem_links.pop(self.current, None) is None:
            return
        self._save_progress()
        self._refresh_candidates()
        self._draw_stems()
        self.c.view.status = f"Tree {self.current} unmatched"

    def _build_cluster_gap_popover(self) -> None:
        """The Cluster gap popover: a Tight-to-Loose slider over
        :data:`CLUSTER_GAP_FACTORS`, and a line saying what the setting
        means for the cloud that's loaded."""
        pop = QWidget(self, Qt.Popup)
        self._cluster_gap_popover = pop
        lay = QVBoxLayout(pop)
        title = QLabel("Cluster gap")
        title.setStyleSheet("font-weight: bold;")
        lay.addWidget(title)
        row = QHBoxLayout()
        row.addWidget(QLabel("Tight"))
        self.cluster_gap_slider = QSlider(Qt.Horizontal)
        self.cluster_gap_slider.setRange(0, len(CLUSTER_GAP_FACTORS) - 1)
        self.cluster_gap_slider.setPageStep(1)
        self.cluster_gap_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.cluster_gap_slider.setMinimumWidth(180)
        self.cluster_gap_slider.setValue(
            CLUSTER_GAP_FACTORS.index(DEFAULT_CLUSTER_GAP_FACTOR)
        )
        self.cluster_gap_slider.valueChanged.connect(self._on_cluster_gap_changed)
        row.addWidget(self.cluster_gap_slider)
        row.addWidget(QLabel("Loose"))
        lay.addLayout(row)
        self.cluster_gap_label = QLabel()
        lay.addWidget(self.cluster_gap_label)
        hint = QLabel(
            "Looser bridges wider holes, so one click grabs more of a tree.\n"
            "Click the same spot again, or press T, to loosen a step;\n"
            "R tightens. The last cluster click updates live.\n"
            f"Back to {DEFAULT_CLUSTER_GAP_FACTOR:g}× when Cluster is switched off\n"
            "or the selection is cleared."
        )
        hint.setStyleSheet("color: gray;")
        lay.addWidget(hint)
        self._update_cluster_gap_label()

    def _update_cluster_gap_label(self) -> None:
        factor = self.c.cluster_gap_factor
        text = f"{factor:g}× the point spacing"
        # Only quote metres once there's a cloud to measure: the spacing of
        # an empty view is a placeholder, not a number worth showing.
        if len(self.c.view.coords) >= 2:
            text += f"  (≈ {self.c.cluster_gap * 100:.1f} cm here)"
        self.cluster_gap_label.setText(text)
        self.cluster_btn.setToolTip(
            self.cluster_btn.toolTip().split("\n\nGap:")[0]
            + f"\n\nGap: {factor:g}× point spacing"
        )

    def _on_cluster_gap_changed(self, index: int) -> None:
        reapplied = self.c.set_cluster_gap_factor(CLUSTER_GAP_FACTORS[index])
        self._update_cluster_gap_label()
        if not reapplied:
            self.c.view.status = (
                f"Cluster gap {self.c.cluster_gap_factor:g}× point spacing"
            )

    def reset_cluster_gap(self) -> None:
        """Back to the default gap when the Cluster tool is switched off —
        the loosening its clicks built up belongs to that use of the tool.

        The slider moves with its signals blocked: its handler would post a
        "Cluster gap …" status over the tool's own "Cluster off", and there's
        nothing for it to re-run anyway (see SegFixController.reset_cluster_gap).
        """
        self.c.reset_cluster_gap()
        s = self.cluster_gap_slider
        s.blockSignals(True)
        s.setValue(CLUSTER_GAP_FACTORS.index(DEFAULT_CLUSTER_GAP_FACTOR))
        s.blockSignals(False)
        self._update_cluster_gap_label()

    def step_cluster_gap(self, step: int) -> None:
        """Move the gap one notch tighter (-1) or looser (+1): the [ and ]
        keys, and a repeat cluster click, so it can be tuned without leaving
        the canvas."""
        s = self.cluster_gap_slider
        target = s.value() + step
        if not s.minimum() <= target <= s.maximum():
            # Already at the end: say so, or a click that does nothing reads
            # as the tool having stopped working.
            end = "loosest" if step > 0 else "tightest"
            self.c.view.status = (
                f"Cluster gap already at its {end} "
                f"({self.c.cluster_gap_factor:g}× spacing)"
            )
            return
        s.setValue(target)

    def _subheading(self, text: str) -> QLabel:
        """A small bold label dividing a group box into sub-sections,
        lighter-weight than nesting another QGroupBox."""
        label = QLabel(text)
        label.setStyleSheet(
            "font-weight: bold; color: gray; margin-top: 2px;"
        )
        return label

    def _on_point_size(self, value: float) -> None:
        if len(self.c.view.coords):
            self.c.view.size = value

    # (label, tooltip) for the view buttons under the point size; the keys
    # are cloudview.VIEWS.
    _VIEW_BUTTONS = {
        "top": ("Top", "Top view: look straight down"),
        "front": ("Front", "Front view: look along +Y"),
        "back": ("Back", "Back view: look along -Y"),
        "left": ("Left", "Left view: look along +X"),
        "right": ("Right", "Right view: look along -X"),
        "bottom": ("Bottom", "Bottom view: look straight up"),
        "3d": ("3D", "Back to the tilted 3D view"),
    }

    def _build_point_size_overlay(self) -> None:
        """Float the point-size spinner over the canvas' top-left corner,
        with CloudCompare-style view buttons below it. A view button only
        turns the camera: pivot and zoom stay."""
        native = self.c.view.native
        box = QWidget(native)
        box.setObjectName("pointSizeOverlay")  # styled by _apply_overlay_theme
        col = QVBoxLayout(box)
        col.setContentsMargins(6, 4, 6, 4)
        col.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(QLabel("Point size"))
        row.addWidget(self.size_spin)
        row.addStretch(1)
        col.addLayout(row)
        views = QHBoxLayout()
        views.setSpacing(2)
        for name, (label, tip) in self._VIEW_BUTTONS.items():
            btn = QToolButton()
            btn.setObjectName(f"view_{name}")
            btn.setText(label)
            btn.setToolTip(tip)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # keep keys on the canvas
            btn.clicked.connect(lambda _c=False, n=name: self.c.view.set_view(n))
            views.addWidget(btn)
        col.addLayout(views)

        # Inside view: stand at a point in the cloud and look around, where
        # the orthographic view above draws a dense plot as a flat wall.
        inside = QHBoxLayout()
        inside.setSpacing(4)
        self.inside_btn = QToolButton()
        self.inside_btn.setObjectName("view_inside")
        self.inside_btn.setText("Inside")
        self.inside_btn.setCheckable(True)
        self.inside_btn.setToolTip(
            "Stand inside the cloud and look around from there, in "
            "perspective - double-click a point first to choose where, or "
            "double-click again while inside to move. Scroll to pull back out."
        )
        self.inside_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Lit while it is on, like the Interaction mode buttons: this one
        # changes how the whole canvas is projected, and a flat toolbutton
        # looks identical whether you are standing in the cloud or outside
        # it — the only other clue is the picture itself.
        self.inside_btn.setStyleSheet(
            "QToolButton:checked { background: #1f4a52; color: #8fe8e0; "
            "font-weight: bold; border: 1px solid #8fe8e0; "
            "border-radius: 3px; }"
        )
        self.inside_btn.toggled.connect(self._on_inside_view)
        inside.addWidget(self.inside_btn)
        inside.addWidget(QLabel("FOV"))
        self.fov_spin = QSpinBox()
        self.fov_spin.setRange(20, 120)
        self.fov_spin.setSuffix("°")
        self.fov_spin.setValue(int(INSIDE_FOV))
        self.fov_spin.setToolTip(
            "How wide the lens is inside the cloud: narrow picks a gap apart, "
            "wide shows what is around you"
        )
        self.fov_spin.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.fov_spin.setEnabled(False)  # nothing to widen from outside
        self.fov_spin.valueChanged.connect(self.c.view.set_inside_fov)
        inside.addWidget(self.fov_spin)
        inside.addStretch(1)
        col.addLayout(inside)
        box.adjustSize()
        box.move(10, 10)
        box.show()
        box.raise_()
        self._point_size_overlay = box

    def _on_inside_view(self, inside: bool) -> None:
        """The Inside toggle: stand at the pivot, or step back out.

        The pivot is wherever the last double-click put it, which is how the
        camera already works in move mode — so "look through this thicket"
        is double-click the spot, then Inside.
        """
        view = self.c.view
        self.fov_spin.setEnabled(inside)
        if not inside:
            view.leave_inside_view()
            view.status = "Back to the orthographic view"
            return
        if not len(view.coords):
            self.inside_btn.setChecked(False)
            return
        view.enter_inside_view(fov=self.fov_spin.value())
        x, y, z = view.view.camera.center
        view.status = (
            f"Inside the cloud at {x:,.1f}, {y:,.1f}, {z:,.1f} - drag to look "
            "around, double-click a point to move there, scroll to pull back"
        )

    def _sync_inside_button(self) -> None:
        """Keep the toggle honest when something else left the bubble — a
        queue step or Reset view flies the camera back out."""
        inside = self.c.view.inside_view
        if self.inside_btn.isChecked() != inside:
            self.inside_btn.blockSignals(True)
            self.inside_btn.setChecked(inside)
            self.inside_btn.blockSignals(False)
            self.fov_spin.setEnabled(inside)

    def _position_current_tree_overlay(self, *_event) -> None:
        """Pin the fixed-size "Current tree" box to the canvas' right edge,
        top-aligned. Also the vispy resize-event handler."""
        box = getattr(self, "_current_tree_overlay", None)
        if box is None:
            return
        native = self.c.view.native
        margin = 10
        x = max(margin, native.width() - box.width() - margin)
        box.move(x, margin)
        classes = getattr(self, "_class_overlay", None)
        if classes is not None and not self._class_docked:
            classes.move(x, margin + box.height() + 6)

    def on_toggle_lasso(self, checked: bool) -> None:
        if checked:
            # Stand the other tool down *first*: disarming hands the mouse
            # back to the camera, so doing it second undoes the arm below.
            # See _uncheck_other_modes.
            self.c.cluster.set_armed(False)
        self.c.lasso.set_armed(checked)
        self.c.lasso_filter = None
        self.c.on_lasso_section = None
        if checked:
            self._uncheck_other_modes(self.lasso_btn)
        else:
            self.move_btn.setChecked(True)

    def on_toggle_tree_lasso(self, checked: bool) -> None:
        if checked:
            self.c.cluster.set_armed(False)  # before arming — see on_toggle_lasso
        self.c.lasso.set_armed(checked)
        self.c.lasso_filter = self._filter_to_current_tree if checked else None
        self.c.on_lasso_section = None
        if checked:
            self._uncheck_other_modes(self.tree_lasso_btn)
            if self.current is None:
                self.c.view.status = (
                    "Lasso tree: pick a tree to review first "
                    "(press Space or click a table row)"
                )
        else:
            self.move_btn.setChecked(True)

    def on_toggle_cluster(self, checked: bool) -> None:
        """Arm/disarm the click-to-select-a-connected-patch tool."""
        if checked:
            self.c.lasso.set_armed(False)  # before arming — see on_toggle_lasso
            self.c.lasso_filter = None
            self.c.on_lasso_section = None
        self.c.cluster.set_armed(checked)
        if checked:
            self._uncheck_other_modes(self.cluster_btn)
        else:
            self.reset_cluster_gap()
            self.move_btn.setChecked(True)

    def on_toggle_lasso_section(self, checked: bool) -> None:
        """Arm/disarm the shared lasso tool in "section" mode: a completed
        drag becomes the kept-region outline (via _on_lasso_section_drawn)
        instead of a selection. See SegFixController._on_lasso."""
        if checked:
            self.c.cluster.set_armed(False)  # before arming — see on_toggle_lasso
        self.c.lasso.set_armed(checked)
        self.c.lasso_filter = None
        self.c.on_lasso_section = self._on_lasso_section_drawn if checked else None
        if checked:
            self._uncheck_other_modes(self.section_draw_btn)
            self.c.view.status = "Lasso section: drag to outline the kept region"
        else:
            self.move_btn.setChecked(True)

    def _filter_to_current_tree(self, indices: np.ndarray) -> np.ndarray:
        """Keep only the points among ``indices`` already in the current
        tree. With no tree under review there is nothing to narrow to, so
        select nothing — falling through to the whole lasso would just make
        this behave like the plain Lasso tool."""
        if self.current is None:
            return indices[:0]
        return indices[self.c.cloud.labels[indices] == self.current]

    def _refresh_double_click_mode(self) -> None:
        """A canvas double-click recentres the turntable pivot only in plain
        move mode; while any selection tool is armed the double-click is the
        tool's (e.g. cluster's click-to-grow), so suppress the recentre."""
        tool_armed = any(b.isChecked() for b in (
            self.lasso_btn, self.tree_lasso_btn,
            self.cluster_btn, self.section_draw_btn,
        ))
        self.c.view.recenter_on_double_click = not tool_armed

    def _uncheck_other_modes(self, active_btn) -> None:
        for btn in (
            self.move_btn, self.lasso_btn, self.tree_lasso_btn,
            self.cluster_btn, self.section_draw_btn,
        ):
            if btn is active_btn:
                continue
            was_checked = btn.isChecked()
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
            if btn is self.cluster_btn and was_checked:
                # Unchecked with signals blocked, so on_toggle_cluster never
                # hears about it: going straight from Cluster to another mode
                # has to reset the gap here, or it would only reset via K/Esc.
                self.reset_cluster_gap()

    def on_move_mode(self) -> None:
        """Revert to camera/movement controls (Escape)."""
        self.lasso_btn.setChecked(False)  # disarms lasso via on_toggle_lasso
        self.tree_lasso_btn.setChecked(False)  # ditto, tree-only lasso
        self.cluster_btn.setChecked(False)  # ditto, cluster picker
        self.section_draw_btn.setChecked(False)  # ditto, lasso-section drawing
        self.move_btn.setChecked(True)  # stay checked even if already in move

    def _on_cloud_changed(self) -> None:
        """A new cloud was loaded: reset per-cloud state, re-hook."""
        # Start the review on the tree this scene was loaded for. Picking a
        # tree in the All Trees table used to load it with its neighbours and
        # then review none of them, so the first thing to do with a freshly
        # opened tree was to find it in the panel's own table and click it
        # again. _refresh_tree_table -> _sync_current selects its row from
        # here, once the table it has to select in exists.
        self.current = self.c.focus_label
        self.hidden_ids = set()
        # A slab computed for the previous cloud's coordinate space doesn't
        # carry over; turn the tool off and recompute its range for this one.
        self.cross_enable.setChecked(False)
        self._reset_cross_section_range()
        # Same for a lasso section: its mask is indices into the *previous*
        # cloud's points array, meaningless for a new one.
        self._lasso_section_mask = None
        self.lasso_section_enable.setChecked(False)
        self._update_lasso_section_label()
        self._load_progress()  # done-tree set lives beside the source file
        self._on_point_size(self.size_spin.value())  # size persists across loads
        # The view drew the new cloud in tree colours; a class colouring
        # carries across loads, like the point size.
        self._rebuild_class_buttons()
        if self.c.class_colours() is not None:
            self._apply_transparency()
        self._update_info()
        self._update_selection()
        # Keep the selection readout live as the lasso changes it.
        self.c.view.on_selection_changed = self._update_selection

    def _update_info(self) -> None:
        cloud = self.c.cloud
        self.info.setText(f"{cloud.n_points:,} points · {len(cloud.tree_ids)} trees")
        self._refresh_tree_table()

    # -- tree table --------------------------------------------------
    def _refresh_tree_table(self) -> None:
        """Rebuild the table from the cloud: done, swatch+ID, point count."""
        cloud = self.c.cloud
        vals, counts = np.unique(cloud.labels, return_counts=True)
        real = (vals != UNASSIGNED) & (vals != NOISE)
        vals, counts = vals[real], counts[real]
        rgba = colors_for_labels(vals, cloud.label_colors)

        id_set = {int(t) for t in vals}
        self.hidden_ids &= id_set  # drop ids for trees that no longer exist
        self.c.faded_ids &= id_set
        self._table_updating = True
        self.tree_table.blockSignals(True)
        self.tree_table.setSortingEnabled(False)
        self.tree_table.clearSelection()
        self.tree_table.setRowCount(len(vals))
        for row, (tid, count) in enumerate(zip(vals, counts)):
            done = int(tid) in self.done_ids
            done_item = QTableWidgetItem()
            done_item.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
            )
            done_item.setCheckState(Qt.Checked if done else Qt.Unchecked)
            self.tree_table.setItem(row, 0, done_item)
            id_item = QTableWidgetItem()
            id_item.setData(Qt.DisplayRole, int(tid))  # int → numeric sort
            id_item.setData(Qt.UserRole, int(tid))
            r, g, b = (int(v * 255) for v in rgba[row][:3])
            swatch = QPixmap(12, 12)
            swatch.fill(QColor(r, g, b))
            id_item.setData(Qt.DecorationRole, swatch)
            self.tree_table.setItem(row, 1, id_item)
            count_item = QTableWidgetItem()
            count_item.setData(Qt.DisplayRole, int(count))
            self.tree_table.setItem(row, 2, count_item)
            hidden = int(tid) in self.hidden_ids
            hide_item = QTableWidgetItem()
            hide_item.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
            )
            hide_item.setCheckState(Qt.Checked if hidden else Qt.Unchecked)
            hide_item.setToolTip("Hide this tree's points in the 3D view")
            self.tree_table.setItem(row, self.HIDE_COL, hide_item)
            faded = int(tid) in self.c.faded_ids
            fade_item = QTableWidgetItem()
            fade_item.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
            )
            fade_item.setCheckState(Qt.Checked if faded else Qt.Unchecked)
            fade_item.setToolTip(
                "Fade this tree - ghosted for context, still selectable"
            )
            self.tree_table.setItem(row, self.FADE_COL, fade_item)
            self._style_done_row(row, done)
        self.tree_table.setSortingEnabled(True)
        self.tree_table.blockSignals(False)
        self._table_updating = False
        self._update_done_title()
        self._sync_current({int(t) for t in vals})
        self._sync_view_toggles()

    def _sync_current(self, existing: set[int]) -> None:
        """After a rebuild: keep reviewing the same tree, or advance if it
        vanished (absorbed into another tree, trashed whole, …)."""
        if self.current is not None and self.current not in existing:
            self._set_current(self._next_pending(-1))
        else:
            self._set_current(self.current, fly=False)

    def _row_of(self, tid: int) -> int | None:
        for row in range(self.tree_table.rowCount()):
            if self.tree_table.item(row, 1).data(Qt.UserRole) == tid:
                return row
        return None

    def _row_id(self, row: int) -> int:
        return int(self.tree_table.item(row, 1).data(Qt.UserRole))

    def _on_table_selection(self) -> None:
        if self._table_updating:
            return
        rows = {item.row() for item in self.tree_table.selectedItems()}
        self._set_current(self._row_id(rows.pop()) if rows else None)

    # -- the review queue ---------------------------------------------
    def _set_current(self, tid: int | None, fly: bool = True) -> None:
        """Make ``tid`` the tree under review: select its row, focus the
        view on it, box it, and (on a real change) fly the camera there."""
        changed = tid != self.current
        self.current = tid
        self._table_updating = True
        row = self._row_of(tid) if tid is not None else None
        if row is None:
            self.tree_table.clearSelection()
        else:
            self.tree_table.selectRow(row)
        self._table_updating = False
        if changed:
            self.c.view.selected = set()
        self._update_tree_bbox([] if tid is None else [tid])
        self._update_current_info()
        # "others" means others than this one, so the two toggles answer to
        # a different set now.
        self._sync_view_toggles()
        if changed and self.stems:
            self._highlight_stem = None
            self._refresh_candidates()
            self._draw_stems()
        if changed and fly and tid is not None:
            self._fly_to(tid)
            self.c.view.status = (
                f"Tree {tid} - lasso (Q) then A/S/D/X to fix, "
                "Space to mark done and continue"
            )
        # Last: framing a tree takes the camera out of the inside view, and
        # the toggle has to say so.
        self._sync_inside_button()

    def _step(self, delta: int) -> None:
        """Prev/Next: move through the table in its current order."""
        rows = self.tree_table.rowCount()
        if not rows:
            return
        cur = self._row_of(self.current) if self.current is not None else None
        row = 0 if cur is None else (cur + delta) % rows
        self._set_current(self._row_id(row))

    def _next_pending(self, after_row: int) -> int | None:
        """First not-done tree after ``after_row`` in table order, wrapping."""
        rows = self.tree_table.rowCount()
        for step in range(1, rows + 1):
            tid = self._row_id((after_row + step) % rows)
            if tid not in self.done_ids:
                return tid
        return None

    def on_done_next(self) -> None:
        """Space: mark the current tree done, save, jump to the next."""
        if not self.tree_table.rowCount():
            self.c.view.status = "No trees to review"
            return
        start = -1
        if self.current is not None:
            self._mark_done(self.current, True)
            start = self._row_of(self.current)
        nxt = self._next_pending(start if start is not None else -1)
        if nxt is None:
            self._set_current(None)
            self.c.view.status = "All trees done - save when ready"
            return
        self._set_current(nxt)

    def _fly_to(self, tid: int) -> None:
        """Centre the camera on the tree and zoom to roughly fit it."""
        cloud = self.c.cloud
        pts = cloud.coords[cloud.labels == tid]
        if not len(pts):
            return
        center = pts.mean(axis=0)
        span = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
        self.c.view.fly_to(center, span)

    def _other_loaded_trees(self) -> set[int]:
        """The loaded trees that aren't the one under review.

        Membership of the loaded set, not a fresh distance test: the cloud
        is fixed at load time to one tree plus its one-hop neighbours, but
        "current" moves to another member of that group as the queue
        advances, and two loaded trees needn't be within reach of *each
        other*. Recomputing distances from current therefore under-hid in
        scene mode.
        """
        if self.current is None:
            return set()
        return {int(t) for t in self.c.cloud.tree_ids} - {self.current}

    def _needs_current(self) -> bool:
        if self.current is not None:
            return True
        self.c.view.status = (
            "No tree under review - press Space or click a table row"
        )
        return False

    def _set_hide_others(self, hide: bool) -> None:
        """The View group's "Hide others" toggle."""
        if not self._needs_current():
            self._sync_view_toggles()  # nothing happened; say so
            return
        others = self._other_loaded_trees()
        self.hidden_ids = (
            self.hidden_ids | others if hide else self.hidden_ids - others
        )
        self._refresh_tree_table()  # updates the per-tree eye checkboxes
        self._apply_visibility()
        self.c.view.status = (
            f"{'Hid' if hide else 'Shown'} {len(others)} other tree(s)"
        )

    def _set_fade_others(self, fade: bool) -> None:
        """The View group's "Fade others" toggle: same set as
        :meth:`_set_hide_others`, but the others stay visible and
        selectable."""
        if not self._needs_current():
            self._sync_view_toggles()
            return
        others = self._other_loaded_trees()
        self.c.faded_ids = (
            self.c.faded_ids | others if fade else self.c.faded_ids - others
        )
        self._refresh_tree_table()  # updates the per-tree Fade checkboxes
        self._apply_transparency()
        self.c.view.status = (
            f"{'Faded' if fade else 'Restored'} {len(others)} other tree(s)"
        )

    def _sync_view_toggles(self) -> None:
        """Point the two "others" toggles at the truth.

        The same state is reachable one tree at a time through the table's
        Hide and Fade columns, and the loaded set itself changes as the
        queue advances, so neither toggle can just remember its own clicks.
        """
        others = self._other_loaded_trees()
        for box, ids in ((getattr(self, "hide_others_cb", None), self.hidden_ids),
                         (getattr(self, "fade_others_cb", None), self.c.faded_ids)):
            if box is None:
                continue
            on = bool(others) and others <= ids
            if box.isChecked() != on:
                box.blockSignals(True)
                box.setChecked(on)
                box.blockSignals(False)

    def on_hide_neighbours(self) -> None:
        """G: flip "Hide others" — the key and the toggle are one control."""
        if self._needs_current():
            self.hide_others_cb.toggle()

    def on_fade_neighbours(self) -> None:
        """Shift+G: flip "Fade others"."""
        if self._needs_current():
            self.fade_others_cb.toggle()

    def _apply_visibility(self) -> None:
        if not len(self.c.view.coords):
            return
        hidden = (
            np.isin(self.c.cloud.labels, list(self.hidden_ids))
            if self.hidden_ids else None
        )
        self.c.view.shown = visibility_mask(
            self.c.cloud.labels,
            hide_unassigned=not self.show_unassigned.isChecked(),
            hidden=hidden,
            cross_section=self._combined_section_mask(),
        )

    def _combined_section_mask(self) -> np.ndarray | None:
        """AND of the cross section and lasso section, whichever are
        currently enabled — both fold into the same "shown" mechanism as
        one combined region, so this is the only thing _apply_visibility
        needs to pass along."""
        cross = self._cross_section_mask()
        lasso = self._lasso_section_mask_active()
        if cross is None:
            return lasso
        if lasso is None:
            return cross
        return cross & lasso

    def _on_show_unassigned(self, checked: bool) -> None:
        self._apply_visibility()
        self.c.view.status = (
            "Unassigned/noise points shown" if checked
            else "Unassigned/noise points hidden"
        )

    # -- cross section --------------------------------------------------
    CROSS_SECTION_STEPS = 1000

    def _cross_axis_bounds(self) -> tuple[float, float]:
        """(lo, hi) of the current cloud's extent along the selected axis."""
        coords = self.c.cloud.coords
        if not len(coords):
            return 0.0, 1.0
        axis = self.cross_axis_combo.currentIndex()
        lo, hi = float(coords[:, axis].min()), float(coords[:, axis].max())
        return (lo, hi) if hi > lo else (lo, lo + 1.0)

    def _reset_cross_section_range(self) -> None:
        """(Re)initialise the slab to the current cloud's full extent on the
        selected axis — called on axis change, Reset, and on a new cloud."""
        self._cross_lo, self._cross_hi = self._cross_axis_bounds()
        for slider, value in (
            (self.cross_min_slider, 0),
            (self.cross_max_slider, self.CROSS_SECTION_STEPS),
        ):
            slider.blockSignals(True)
            slider.setRange(0, self.CROSS_SECTION_STEPS)
            slider.setValue(value)
            slider.blockSignals(False)
        self._update_cross_range_label()

    def _slider_to_value(self, slider_value: int) -> float:
        lo, hi = self._cross_lo, self._cross_hi
        return lo + (hi - lo) * (slider_value / self.CROSS_SECTION_STEPS)

    def _on_cross_section_toggled(self, checked: bool) -> None:
        self._apply_visibility()
        self.c.view.status = (
            "Cross section on - only the slab is shown/selectable"
            if checked else "Cross section off"
        )

    def _on_cross_axis_changed(self, _index: int) -> None:
        self._reset_cross_section_range()
        self._apply_visibility()

    def _on_cross_reset(self) -> None:
        self._reset_cross_section_range()
        self._apply_visibility()

    def _on_cross_range_changed(self, _value: int) -> None:
        # Keep min <= max by nudging the other slider if a drag crosses it;
        # setValue re-enters this handler, which then just updates in place.
        if self.cross_min_slider.value() > self.cross_max_slider.value():
            if self.sender() is self.cross_min_slider:
                self.cross_max_slider.setValue(self.cross_min_slider.value())
            else:
                self.cross_min_slider.setValue(self.cross_max_slider.value())
            return
        self._update_cross_range_label()
        self._apply_visibility()

    def _update_cross_range_label(self) -> None:
        axis_name = "XYZ"[self.cross_axis_combo.currentIndex()]
        lo = self._slider_to_value(self.cross_min_slider.value())
        hi = self._slider_to_value(self.cross_max_slider.value())
        self.cross_range_label.setText(f"{axis_name}: {lo:.2f} – {hi:.2f} m")

    def _cross_section_mask(self) -> np.ndarray | None:
        """Per-point mask, True inside the current slab; None when off."""
        if not self.cross_enable.isChecked() or not len(self.c.cloud.coords):
            return None
        axis = self.cross_axis_combo.currentIndex()
        lo = self._slider_to_value(self.cross_min_slider.value())
        hi = self._slider_to_value(self.cross_max_slider.value())
        coords_axis = self.c.cloud.coords[:, axis]
        return (coords_axis >= lo) & (coords_axis <= hi)

    # -- lasso section ----------------------------------------------------
    def _on_lasso_section_drawn(self, indices: np.ndarray) -> None:
        """Completed drag while section_draw_btn is armed: freeze it as a
        per-point mask (not a live screen region — the camera can move
        freely afterwards) and switch the section on to show it."""
        n = len(self.c.cloud.coords)
        mask = np.zeros(n, dtype=bool)
        mask[indices] = True
        self._lasso_section_mask = mask
        self.lasso_section_enable.blockSignals(True)
        self.lasso_section_enable.setChecked(True)
        self.lasso_section_enable.blockSignals(False)
        self._update_lasso_section_label()
        self._apply_visibility()
        self.c.view.status = (
            f"Lasso section drawn - kept {mask.sum():,} of {n:,} points"
        )

    def _on_lasso_section_toggled(self, checked: bool) -> None:
        self._apply_visibility()
        self.c.view.status = (
            "Lasso section on - only the outline is shown/selectable"
            if checked else "Lasso section off"
        )

    def _on_lasso_section_reset(self) -> None:
        self._lasso_section_mask = None
        self.lasso_section_enable.setChecked(False)
        self._update_lasso_section_label()
        self._apply_visibility()
        self.c.view.status = "Lasso section cleared"

    def _update_lasso_section_label(self) -> None:
        mask = self._lasso_section_mask
        text = (
            "No outline drawn" if mask is None
            else f"{int(mask.sum()):,} / {len(mask):,} points kept"
        )
        self.lasso_section_label.setText(text)
        self.lasso_section_box.setToolTip(
            "Keep only points inside a hand-drawn outline.\n" + text
        )

    def _lasso_section_mask_active(self) -> np.ndarray | None:
        """The drawn mask, if the section is on and it still matches the
        currently loaded cloud (a new cloud invalidates it — see
        _on_cloud_changed); None otherwise, meaning no restriction."""
        mask = self._lasso_section_mask
        if not self.lasso_section_enable.isChecked() or mask is None:
            return None
        if len(mask) != len(self.c.cloud.coords):
            return None
        return mask

    # -- done tracking ------------------------------------------------
    def _on_tree_item_changed(self, item) -> None:
        if self._table_updating:
            return
        tid = self._row_id(item.row())
        if item.column() == 0:
            self._mark_done(tid, item.checkState() == Qt.Checked)
        elif item.column() == self.HIDE_COL:
            self._set_hidden(tid, item.checkState() == Qt.Checked)
        elif item.column() == self.FADE_COL:
            self._set_faded(tid, item.checkState() == Qt.Checked)

    def _set_hidden(self, tid: int, hidden: bool) -> None:
        if hidden:
            self.hidden_ids.add(tid)
        else:
            self.hidden_ids.discard(tid)
        self._apply_visibility()
        self.c.view.status = f"Tree {tid} {'hidden' if hidden else 'shown'}"

    def _set_faded(self, tid: int, faded: bool) -> None:
        if faded:
            self.c.faded_ids.add(tid)
        else:
            self.c.faded_ids.discard(tid)
        self._apply_transparency()
        self.c.view.status = (
            f"Tree {tid} {'faded' if faded else 'back to full opacity'}"
        )

    def _apply_transparency(self) -> None:
        """Re-colour the cloud so trees in ``faded_ids`` render at
        FADED_ALPHA. Parallel to _apply_visibility, but for opacity — faded
        points stay shown and selectable."""
        if not len(self.c.view.coords):
            return
        refresh_view(self.c.view, self.c.cloud, self.c.faded_ids,
                     class_colours=self.c.class_colours())

    def _mark_done(self, tid: int, done: bool) -> None:
        if done:
            self.done_ids.add(tid)
        else:
            self.done_ids.discard(tid)
        row = self._row_of(tid)
        if row is not None:
            self._table_updating = True  # styling sets data; don't re-enter
            state = Qt.Checked if done else Qt.Unchecked
            self.tree_table.item(row, 0).setCheckState(state)
            self._style_done_row(row, done)
            self._table_updating = False
        self._update_done_title()
        path = self._save_progress()
        if self.on_done_changed is not None:
            self.on_done_changed()
        self.c.view.status = (
            f"Tree {tid} marked {'done' if done else 'not done'}"
            + (f" - saved to {os.path.basename(path)}" if path else "")
        )

    def _style_done_row(self, row: int, done: bool) -> None:
        brush = QBrush(theme.done_row_bg()) if done else QBrush()
        for col in range(self.tree_table.columnCount()):
            it = self.tree_table.item(row, col)
            if it is not None:
                it.setBackground(brush)

    def _update_done_title(self) -> None:
        n = self.tree_table.rowCount()
        done = sum(
            1 for row in range(n) if self._row_id(row) in self.done_ids
        )
        title = "Selected Tree + Neighbours"
        if n:
            pct = round(100 * done / n)
            self.trees_box.setTitle(f"{title} - {done}/{n} ({pct}%) done")
        else:
            self.trees_box.setTitle(title)

    def _progress_path(self) -> str | None:
        src = self.c.cloud.source_path or self.c.save_path
        return f"{src}.segfix.json" if src else None

    def _load_progress(self) -> None:
        """Restore the done-tree set from the sidecar file, if present."""
        self.done_ids = set()
        path = self._progress_path()
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.done_ids = {int(t) for t in data.get("done", [])}
            self.stem_links = {
                int(tree): str(stem)
                for tree, stem in dict(data.get("inventory", {})).items()
            }
        except (OSError, ValueError) as exc:
            self.c.view.status = f"Could not read progress file: {exc}"

    def _save_progress(self) -> str | None:
        """Write the done-tree set next to the source file; returns the path."""
        path = self._progress_path()
        if not path:
            return None
        try:
            # Merged, not overwritten: the sidecar also carries the inventory
            # links, and a later key written by another part of segfix should
            # survive a plain Done click.
            saved = {}
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    saved = json.load(f)
            saved["done"] = sorted(self.done_ids)
            saved["inventory"] = {
                str(tree): stem for tree, stem in sorted(self.stem_links.items())
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(saved, f)
        except (OSError, ValueError) as exc:  # unreadable, or corrupt JSON
            self.c.view.status = f"Could not save progress: {exc}"
            return None
        return path

    # -- bounding box around the current tree -------------------------
    def _update_tree_bbox(self, ids) -> None:
        """Draw a wireframe bounding box around each given tree (or clear).

        Called on every current-tree change, so it no-ops when the set of
        boxed trees is unchanged rather than rebuilding.
        """
        ids = {int(t) for t in ids}
        if ids == self._bbox_ids or self._bbox_busy:
            return
        self._bbox_busy = True
        try:
            self._rebuild_tree_bbox(ids)
        finally:
            self._bbox_busy = False

    def _rebuild_tree_bbox(self, ids: set[int]) -> None:
        self._bbox_ids = ids
        segments, colors = [], []
        for tid in ids:
            pts = self.c.cloud.coords[self.c.cloud.labels == tid]
            if not len(pts):
                continue
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            # Corner i picks hi for each set bit of its index, lo otherwise;
            # edges join corners whose indices differ in exactly one bit.
            corners = np.array(
                [[(hi if i & (1 << b) else lo)[b] for b in range(3)]
                 for i in range(8)]
            )
            rgba = colors_for_labels(
                np.array([tid]), self.c.cloud.label_colors
            )[0]
            for i in range(8):
                for b in range(3):
                    if not i & (1 << b):
                        j = i | (1 << b)
                        segments.append(corners[i])
                        segments.append(corners[j])
                        colors.append(rgba)
                        colors.append(rgba)
        if segments:
            self.c.view.set_bbox(
                np.asarray(segments, dtype=np.float32),
                np.asarray(colors, dtype=np.float32),
            )
        else:
            self.c.view.clear_bbox()

    # -- selection info -----------------------------------------------
    def _update_selection(self) -> None:
        idx = self.c.selected_indices()
        had_selection, self._had_selection = self._had_selection, idx.size > 0
        if idx.size == 0:
            if had_selection:
                # The selection just went away (A/S/D/X, an empty-space
                # click, another tree, a reload): whatever the cluster
                # clicks loosened the gap to was for that selection.
                # Watching for the *transition* keeps a gap set on the
                # slider before the first click, with nothing selected yet.
                self.reset_cluster_gap()
            self.sel_info.setText("No selection")
            self._refresh_selection_actions()
            return
        trees = self._selected_trees(idx)
        self.sel_info.setText(
            f"Selected: {idx.size:,} points across {len(trees)} tree(s)"
        )
        self._refresh_selection_actions()

    def _neighbour_row_height(self) -> int:
        """How tall one row of neighbour buttons really is.

        Measured from a button styled like the real ones, not from a plain
        one: the 2px coloured border and padding make them several pixels
        taller, and reserving the plain height sliced the bottom row of the
        scroll box in half.
        """
        cached = getattr(self, "_neighbour_row_h", None)
        if cached is None:
            probe = QPushButton(" 000")
            probe.setIcon(QIcon(key_badge("1", QColor("white"))))
            probe.setIconSize(QSize(16, 16))
            probe.setStyleSheet(
                "QPushButton { border: 2px solid gray; border-radius: 3px; "
                "padding: 2px 4px; }"
            )
            cached = self._neighbour_row_h = probe.sizeHint().height()
            probe.deleteLater()
        return cached

    def _refresh_selection_actions(self) -> None:
        """Enable only what the current state can actually do.

        With nothing selected, "Add", "Split off" and the neighbour buttons
        have nothing to act on — they used to look live and answer a click
        with a status line. Unassign and Noise stay enabled: with no
        selection those act on the whole current tree, which is how a bush
        gets dismissed in one key.
        """
        has_selection = self.c.selected_indices().size > 0
        for btn in (getattr(self, "add_btn", None),
                    getattr(self, "split_btn", None),
                    *getattr(self, "_neighbour_btns", ())):
            if btn is not None:
                btn.setEnabled(has_selection)

    def _selected_trees(self, idx: np.ndarray) -> np.ndarray:
        """Distinct real tree IDs (no unassigned/noise) under the selection."""
        labels = self.c.cloud.labels[idx]
        return np.unique(labels[(labels != UNASSIGNED) & (labels != NOISE)])

    def _update_current_info(self) -> None:
        self._update_neighbour_picker()
        if self.current is None:
            self.current_swatch.setStyleSheet(
                "background: none; border: 1px dashed gray;"
            )
            self.current_label.setText("none - press Space or click a row")
            self.add_btn.setText("Add selection (A)")
            return
        rgba = colors_for_labels(
            np.array([self.current]), self.c.cloud.label_colors
        )[0]
        r, g, b = (int(v * 255) for v in rgba[:3])
        self.current_swatch.setStyleSheet(f"background: rgb({r},{g},{b});")
        n = int(np.count_nonzero(self.c.cloud.labels == self.current))
        self.current_label.setText(f"tree {self.current} · {n:,} points")
        self.add_btn.setText(f"Add selection to tree {self.current} (A)")

    def _update_neighbour_picker(self) -> None:
        """Rebuild the "send selection to neighbour" button row for the
        current tree — a quicker path than switching current tree and
        pressing Add when moving a patch to an adjacent tree."""
        for grid in (self.keyed_grid, self.neighbour_grid):
            while grid.count():
                item = grid.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()

        from . import analysis

        near = analysis.neighbour_distances(
            self.c.cloud, self.current, self.focus_margin.value()
        )
        # Nearest first, so key 1 is the tree whose crown the selection most
        # likely belongs to. Sorted by id, the keys landed on whichever five
        # trees happened to have the lowest numbers — in a closed canopy
        # with a dozen neighbours, that is nobody's idea of the right five.
        self._neighbour_ids = sorted(near, key=lambda n: (near[n], n))
        self._neighbour_btns = []
        text_colour = self.palette().buttonText().color()
        for i, nid in enumerate(self._neighbour_ids):
            rgba = colors_for_labels(
                np.array([nid]), self.c.cloud.label_colors
            )[0]
            r, g, b = (int(v * 255) for v in rgba[:3])
            keyed = i < NEIGHBOUR_KEYS
            btn = QPushButton(f" {nid}")
            btn.setProperty("segfix_key", i + 1 if keyed else None)
            # The tree's colour is the button's border either way. A keyed
            # button spends its icon on the key instead of repeating that
            # colour as a swatch, so the two numbers the button used to
            # carry — key and tree id — no longer read as a pair.
            if keyed:
                btn.setIcon(QIcon(key_badge(str(i + 1), text_colour)))
                btn.setIconSize(QSize(16, 16))
            else:
                swatch = QPixmap(12, 12)
                swatch.fill(QColor(r, g, b))
                btn.setIcon(QIcon(swatch))
                btn.setIconSize(QSize(12, 12))
            btn.setStyleSheet(
                f"QPushButton {{ border: 2px solid rgb({r},{g},{b}); "
                "border-radius: 3px; padding: 2px 4px; }"
            )
            key = f", key {i + 1}" if keyed else ""
            btn.setToolTip(
                f"Move the current selection to tree {nid} "
                f"({near[nid]:.2f} m away{key})"
            )
            self._neighbour_btns.append(btn)
            btn.clicked.connect(
                lambda _checked=False, n=nid: self.on_send_to_neighbour(n)
            )
            grid = self.keyed_grid if keyed else self.neighbour_grid
            row = i // 2 if keyed else (i - NEIGHBOUR_KEYS) // 2
            grid.addWidget(btn, row, i % 2 if keyed else (i - NEIGHBOUR_KEYS) % 2)
        # Nothing to scroll to when every neighbour has a key of its own.
        overflowing = len(self._neighbour_ids) > NEIGHBOUR_KEYS
        self.neighbour_scroll.setVisible(overflowing)
        # The scroll area's bar eats into its buttons' width, so the keyed
        # rows above give up the same strip — but only when a bar is
        # actually there, which is when the extras outgrow the rows on show.
        # Reserving it either way leaves one block or the other short.
        extras = len(self._neighbour_ids) - NEIGHBOUR_KEYS
        scrolls = extras > self.NEIGHBOUR_ROWS * 2
        bar = self.neighbour_scroll.verticalScrollBar().sizeHint().width()
        self.keyed_grid.setContentsMargins(0, 0, bar if scrolls else 0, 0)
        self._refresh_selection_actions()

    def send_to_nth_neighbour(self, n: int) -> None:
        """Move the selection into the ``n``-th neighbouring tree, counting
        from 1 in the order the buttons are shown.

        The buttons themselves are a mouse away from a hand that is already
        holding the mouse for the lasso; the number keys are not.
        """
        ids = getattr(self, "_neighbour_ids", [])
        if not ids:
            self.c.view.status = (
                "No neighbouring trees to move the selection to"
            )
            return
        if n > len(ids):
            self.c.view.status = (
                f"Only {len(ids)} neighbouring tree"
                f"{'s' if len(ids) != 1 else ''}: keys 1-{len(ids)}"
            )
            return
        self.on_send_to_neighbour(ids[n - 1])

    def _require_selection(self) -> np.ndarray | None:
        idx = self.c.selected_indices()
        if idx.size == 0:
            self.c.view.status = "Select points first (Q for lasso)"
            return None
        return idx

    def _require_current(self) -> int | None:
        if self.current is None:
            self.c.view.status = (
                "No tree under review - press Space or click a table row"
            )
        return self.current

    # -- ops on the current tree --------------------------------------
    def on_add(self) -> None:
        """Selection → the current tree (missing branches, unassigned…)."""
        tid = self._require_current()
        if tid is None:
            return
        idx = self._require_selection()
        if idx is None:
            return
        self._apply(ops.reassign(self.c.cloud, idx, tid))

    def on_invert_selection(self) -> None:
        """B / the Invert selection button: swap selected for unselected,
        within what's on screen (see
        :meth:`SegFixController.invert_selection`)."""
        if not len(self.c.view.coords):
            return
        total = self.c.invert_selection()
        bounded = not np.asarray(self.c.view.shown, dtype=bool).all()
        where = " of the shown points" if bounded else ""
        self.c.view.status = f"Inverted selection: {total:,} points{where}"

    def on_send_to_neighbour(self, target_id: int) -> None:
        """Selection → a neighbouring tree, without switching current."""
        idx = self._require_selection()
        if idx is None:
            return
        self._apply(ops.reassign(self.c.cloud, idx, target_id))

    def on_create_new(self) -> None:
        idx = self._require_selection()
        if idx is None:
            return
        self._apply(ops.create_new(self.c.cloud, idx))

    def on_unassign(self) -> None:
        idx = self._selection_or_current_tree("unassign")
        if idx is None:
            return
        self._apply(ops.unassign(self.c.cloud, idx))

    def on_noise(self) -> None:
        idx = self._selection_or_current_tree("trash")
        if idx is None:
            return
        self._apply(ops.mark_noise(self.c.cloud, idx))

    def _selection_or_current_tree(self, verb: str) -> np.ndarray | None:
        """U/X act on the selection, or the whole current tree if nothing is
        selected — that's how a non-tree blob is dismissed in one key."""
        idx = self.c.selected_indices()
        if idx.size:
            return idx
        if self.current is None:
            self.c.view.status = (
                f"Select points to {verb} (Q), or pick a tree first"
            )
            return None
        return np.flatnonzero(self.c.cloud.labels == self.current)

    def _apply(self, msg: str) -> None:
        self.c._after_edit(msg)
        self._update_info()  # rebuilds the table; re-syncs bbox/current tree
        # What's shown depends on labels (the H toggle hides unassigned and
        # noise; the eye column hides whole trees), so an edit can change it:
        # points unassigned or marked noise while H is off, or moved into a
        # hidden tree, have to disappear now -- not at the next unrelated
        # redraw. After _update_info, so a tree that no longer exists has
        # already dropped out of hidden_ids.
        self._apply_visibility()
        self._update_selection()

    # -- history -----------------------------------------------------
    def on_undo(self) -> None:
        desc = self.c.cloud.undo()
        self._apply(f"Undid: {desc}" if desc else "Nothing to undo")

    def on_redo(self) -> None:
        desc = self.c.cloud.redo()
        self._apply(f"Redid: {desc}" if desc else "Nothing to redo")

    # -- save --------------------------------------------------------
    def on_save(self) -> None:
        if self.c.on_save_override is not None:
            busy(self.c.view, "Saving…")
            try:
                msg = self.c.on_save_override()
            except Exception as exc:
                QMessageBox.critical(self, "Save failed", str(exc))
                return
            self.c.view.status = msg
            self._save_progress()
            self._update_info()
            return
        if not self.c.save_path:
            self.c.view.status = "Nothing loaded to save"
            return
        self._do_save(self.c.save_path)

    def _do_save(self, path: str) -> None:
        from . import io

        busy(self.c.view, f"Saving to {path}…")
        try:
            io.save(self.c.cloud, path)
        except Exception as exc:  # surface IO errors instead of crashing
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._save_progress()
        self.c.view.status = f"Saved → {path}"


#: Keys a left hand reaches without moving off its home position, with the
#: right hand on the mouse. Everything in :func:`shortcut_bindings` is one of
#: these (bar the legacy keys it also keeps).
LEFT_HAND_KEYS = frozenset(
    "QWERT" "ASDFG" "ZXCVB" "12345"
) | {"Esc", "Space", "Shift+Q", "Shift+W", "Shift+E", "Shift+C", "Shift+G",
     "Shift+F"} | {f"Ctrl+{n}" for n in range(1, CLASS_KEYS + 1)}


def shortcut_bindings(panel) -> dict:
    """``{key: what it does}`` for the whole review loop.

    Separate from :func:`bind_shortcuts` so the map itself can be read and
    tested without a window to hang ``QShortcut``s on.
    """
    # Everything sits under the left hand, because the right one is on the
    # mouse for the whole review loop: the tools on QWE with the cluster gap
    # beside them, the edits on the home row, and the rest on ZXCV. Reaching
    # for L, K, N, U, H or the bracket keys meant letting go of the mouse or
    # looking down, dozens of times a tree.
    bindings = {
        # tools
        "Q": panel.lasso_btn.toggle,
        "W": panel.tree_lasso_btn.toggle,
        "E": panel.cluster_btn.toggle,
        "R": lambda: panel.step_cluster_gap(-1),   # tighter, as "[" is
        "T": lambda: panel.step_cluster_gap(1),    # looser, as "]" is
        # edits
        "A": panel.on_add,
        # 1-5: move the selection into a neighbouring tree, in the order
        # its buttons are shown. Moving a patch to the tree next door is
        # the commonest edit there is, and it was the one thing in the loop
        # that could only be done by clicking.
        **{
            str(n): (lambda n=n: panel.send_to_nth_neighbour(n))
            for n in range(1, NEIGHBOUR_KEYS + 1)
        },
        # Ctrl+1-5: give the selection the 1st-5th point class, the class
        # counterpart of 1-5 above.
        **{
            f"Ctrl+{n}": (lambda n=n: panel.set_nth_class(n))
            for n in range(1, CLASS_KEYS + 1)
        },
        "Shift+F": panel.toggle_color_by_class,
        "S": panel.on_create_new,
        # Invert sits on B, beside the other selection-wide keys and still
        # under the same hand: it is a selection op, not an edit.
        "B": panel.on_invert_selection,
        "D": panel.on_unassign,
        "F": panel.show_unassigned.toggle,
        # Hiding the neighbours before a lasso is routine in a dense canopy,
        # and was the last step of the loop that needed the mouse (issue #3).
        "G": panel.on_hide_neighbours,
        "Shift+G": panel.on_fade_neighbours,
        "X": panel.on_noise,
        "C": panel.cross_enable.toggle,
        # moving through the queue
        "Esc": panel.on_move_mode,
        "Space": panel.on_done_next,
        "Z": lambda: panel._step(-1),
        "V": lambda: panel._step(1),
        "Shift+Q": panel.section_draw_btn.toggle,
        "Shift+C": panel.lasso_section_enable.toggle,
        # The keys these replaced, still bound: a hand that already knows
        # them shouldn't have to unlearn anything.
        "L": panel.lasso_btn.toggle,
        "Ctrl+L": panel.tree_lasso_btn.toggle,
        "K": panel.cluster_btn.toggle,
        "[": lambda: panel.step_cluster_gap(-1),
        "]": lambda: panel.step_cluster_gap(1),
        "Left": lambda: panel._step(-1),
        "Right": lambda: panel._step(1),
        "N": panel.on_create_new,
        "U": panel.on_unassign,
        "H": panel.show_unassigned.toggle,
        "Shift+L": panel.section_draw_btn.toggle,
        # Ctrl+Z / Ctrl+Shift+Z / Ctrl+S are the Edit/File menu actions'
        # shortcuts now (app._build_menus) — binding them here too would make
        # Qt see an ambiguous overload and fire neither.
    }
    return bindings


def bind_shortcuts(window, panel: SegFixWidget) -> None:
    """Bind :func:`shortcut_bindings` on the main window as ``QShortcut``s;
    ``WindowShortcut`` context, so they fire wherever focus sits in the
    window (canvas or a dock)."""
    from qtpy.QtGui import QKeySequence, QShortcut

    panel._shortcuts = []  # keep refs alive
    for key, fn in shortcut_bindings(panel).items():
        sc = QShortcut(QKeySequence(key), window)
        sc.activated.connect(fn)
        panel._shortcuts.append(sc)
