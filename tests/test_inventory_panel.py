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

"""The inventory panel: candidates for the tree under review, and the links.

A link is a claim about field data that outlives the session, so the things
worth pinning are that it survives a reopen, that one stem can't be the
match for two trees at once, and that the table follows the tree under
review rather than whatever it was showing when the map was loaded.

Builds the real panel on a real (offscreen) vispy canvas; skipped where no
canvas can be created.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication  # noqa: E402

from segfix.inventory import Alignment, Stem  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    yield QApplication.instance() or QApplication([])


def tree_points(centre, height, diameter, n=900, seed=0):
    """A trunk plus a crown, so the DBH fit has something to find."""
    rng = np.random.default_rng(seed)
    angle = rng.random(n) * 2 * np.pi
    z = rng.random(n) * min(3.0, height)
    stem = np.column_stack([
        centre[0] + diameter / 2 * np.cos(angle),
        centre[1] + diameter / 2 * np.sin(angle),
        z,
    ])
    crown = np.column_stack([
        rng.normal(centre[0], 1.0, n),
        rng.normal(centre[1], 1.0, n),
        rng.random(n) * (height - 3.0) + 3.0,
    ])
    return np.vstack([stem, crown]).astype(np.float32)


#: Two trees, deliberately confusable by position alone: 1 is tall and fat,
#: 2 is short and thin, and they stand 3 m apart.
TREES = {1: ((10.0, 10.0), 24.0, 0.50), 2: ((13.0, 10.0), 11.0, 0.18)}


@pytest.fixture
def panel(tmp_path):
    try:
        from segfix.cloudview import CloudView

        view = CloudView()
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no vispy canvas available: {exc}")
    from segfix.model import PointCloud
    from segfix.widgets import SegFixController, SegFixWidget

    blocks, labels = [], []
    for tree_id, (centre, height, diameter) in TREES.items():
        points = tree_points(centre, height, diameter, seed=tree_id)
        blocks.append(points)
        labels.append(np.full(len(points), tree_id, dtype=np.int32))
    coords = np.vstack(blocks)
    empty = PointCloud(coords=np.empty((0, 3), np.float32),
                       labels=np.empty(0, np.int32))
    view.load_cloud(empty)
    seg = SegFixController(view, empty)
    p = SegFixWidget(seg)
    cloud = PointCloud(
        coords=coords, labels=np.concatenate(labels),
        source_path=str(tmp_path / "plot.ply"),
    )
    view.load_cloud(cloud)
    seg.set_cloud(cloud)
    p._set_current(1, fly=False)
    yield p, view
    p.deleteLater()


#: A stem map of the same two trees, 5 m off in X — as a GPS stem map is.
STEMS = [
    Stem("A1", 5.0, 10.0, dbh=0.49, height=23.5),
    Stem("A2", 8.0, 10.0, dbh=0.19, height=11.5),
]
SHIFT = Alignment(dx=5.0, dy=0.0)


def test_the_box_stays_hidden_until_a_stem_map_is_loaded(panel):
    p, _ = panel
    assert p.inventory_box.isHidden()
    p.set_stem_map(STEMS, SHIFT)
    assert not p.inventory_box.isHidden()


def test_the_tree_is_measured_from_its_own_points(panel):
    p, _ = panel
    stats = p.tree_stats()
    assert stats.height == pytest.approx(24.0, abs=0.5)
    assert stats.dbh == pytest.approx(0.50, abs=0.03)
    assert (stats.x, stats.y) == pytest.approx((10.0, 10.0), abs=0.2)


def test_size_decides_the_ranking_where_position_cant(panel):
    """Tree 1 is 3 m from both stems once shifted; only height and DBH say
    which is which."""
    p, _ = panel
    p.set_stem_map(STEMS, SHIFT)
    assert [c.stem.stem_id for c in p._candidates][0] == "A1"
    p._set_current(2, fly=False)
    assert [c.stem.stem_id for c in p._candidates][0] == "A2"


def test_the_table_shows_the_differences(panel):
    p, _ = panel
    p.set_stem_map(STEMS, SHIFT)
    row = [p.inventory_table.item(0, c).text() for c in range(5)]
    assert row[0] == "A1"
    assert row[1].endswith("m")
    assert row[2].startswith("+")      # the cloud tree is 0.5 m taller
    assert row[3].endswith("cm")
    assert float(row[4]) < 0.5         # a good match scores low


def test_linking_records_the_match_and_shows_it(panel):
    p, view = panel
    said = []
    view.on_status = said.append
    p.set_stem_map(STEMS, SHIFT)
    p.inventory_table.selectRow(0)
    p.on_link_stem()
    assert p.stem_links == {1: "A1"}
    assert "A1" in said[-1]
    assert "matched to A1" in p.inventory_label.text()


def test_one_stem_cant_match_two_trees(panel):
    """Linking a stem that is already another tree's match moves it, rather
    than leaving the plot double-counted. The candidate list hides taken
    stems, so this is reachable from a project whose links were restored
    from the sidecar after the ranking was built."""
    p, _ = panel
    p.set_stem_map(STEMS, SHIFT)
    p._set_current(2, fly=False)
    p.stem_links = {1: "A1"}  # as a reopened project would have it
    ids = [c.stem.stem_id for c in p._candidates]
    p.inventory_table.selectRow(ids.index("A1"))
    p.on_link_stem()
    assert p.stem_links == {2: "A1"}


def test_a_stem_linked_elsewhere_drops_out_of_the_candidates(panel):
    p, _ = panel
    p.set_stem_map(STEMS, SHIFT)
    p.inventory_table.selectRow(0)
    p.on_link_stem()          # tree 1 -> A1
    p._set_current(2, fly=False)
    assert "A1" not in [c.stem.stem_id for c in p._candidates]


def test_unlinking_puts_the_stem_back(panel):
    p, _ = panel
    p.set_stem_map(STEMS, SHIFT)
    p.inventory_table.selectRow(0)
    p.on_link_stem()
    p.on_unlink_stem()
    assert p.stem_links == {}
    assert p.unlink_btn.isEnabled() is False


def test_links_survive_a_reopen(panel, tmp_path):
    """They are a claim about field data, so they live in the project
    sidecar beside the Done list rather than in the session."""
    p, _ = panel
    p.set_stem_map(STEMS, SHIFT)
    p.inventory_table.selectRow(0)
    p.on_link_stem()
    p.done_ids = {1}
    p._save_progress()

    saved = json.loads((tmp_path / "plot.ply.segfix.json").read_text())
    assert saved["inventory"] == {"1": "A1"}
    assert saved["done"] == [1]

    p.stem_links = {}
    p._load_progress()
    assert p.stem_links == {1: "A1"}
    assert p.done_ids == {1}


def test_marking_a_tree_done_doesnt_drop_the_links(panel, tmp_path):
    p, _ = panel
    p.set_stem_map(STEMS, SHIFT)
    p.inventory_table.selectRow(0)
    p.on_link_stem()
    p.done_ids = {1, 2}
    p._save_progress()
    assert json.loads(
        (tmp_path / "plot.ply.segfix.json").read_text()
    )["inventory"] == {"1": "A1"}


# -- what gets drawn ---------------------------------------------------------
def test_the_stems_are_drawn_where_the_shift_puts_them(panel):
    p, view = panel
    p.set_stem_map(STEMS, SHIFT)
    assert view.stems.visible
    segments, owner = p._stem_geometry
    first = segments[owner == 0]
    assert first[:, 0].mean() == pytest.approx(10.0, abs=0.3)  # 5.0 + the shift
    # and standing on the cloud's floor, not at z=0 in some other datum
    assert first[:, 2].min() == pytest.approx(view.coords[:, 2].min(), abs=0.5)


def test_moving_the_shift_moves_the_cylinders(panel):
    p, _ = panel
    p.set_stem_map(STEMS, SHIFT)
    before = p._stem_geometry[0][:, 0].mean()
    p.set_alignment(Alignment(dx=25.0))
    assert p._stem_geometry[0][:, 0].mean() == pytest.approx(before + 20.0, abs=0.1)


def test_the_matched_stem_is_drawn_in_its_own_colour(panel):
    from segfix.widgets import LINKED_STEM_COLOR, STEM_COLOR

    p, view = panel
    p.set_stem_map(STEMS, SHIFT)
    p.inventory_table.selectRow(0)
    p.on_link_stem()
    assert view.stems.visible
    _, owner = p._stem_geometry
    linked = p._stem_colors[owner == 0]
    assert np.allclose(linked[0], LINKED_STEM_COLOR, atol=0.01)
    other = p._stem_colors[owner == 1]
    assert np.allclose(other[0], STEM_COLOR, atol=0.01)


def test_clearing_the_map_takes_the_cylinders_with_it(panel):
    p, view = panel
    p.set_stem_map(STEMS, SHIFT)
    p.set_stem_map([], Alignment())
    assert not view.stems.visible
    assert p.inventory_box.isHidden()
