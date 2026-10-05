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

"""Matching a field stem map to the segmented trees.

Two things decide whether this is useful at all: finding the shift between
a stem map's coordinates and the cloud's without being thrown by the trees
only one side has, and ranking the stems for a tree by more than position —
in a closed stand the nearest stem is routinely the wrong one, and height
and DBH are what tell the tall thin tree from the short fat one beside it.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from segfix import inventory
from segfix.inventory import Alignment, Stem, Tolerances, TreeStats


# -- reading a stem map ------------------------------------------------------
def write_csv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_the_usual_column_names_are_found(tmp_path):
    path = write_csv(
        tmp_path / "stems.csv",
        [["A1", "10.5", "20.5", "31.2", "18.4", "Euc"]],
        ["Tree ID", "Easting", "Northing", "DBH (cm)", "Height", "Species"],
    )
    stems, used = inventory.load_stem_map(path)
    assert used["x"] == "Easting" and used["y"] == "Northing"
    stem = stems[0]
    assert (stem.stem_id, stem.x, stem.y, stem.species) == ("A1", 10.5, 20.5, "Euc")
    assert stem.dbh == pytest.approx(0.312)  # centimetres, detected
    assert stem.height == 18.4


def test_dbh_already_in_metres_is_left_alone(tmp_path):
    path = write_csv(tmp_path / "s.csv", [["1", "0", "0", "0.31"]],
                     ["id", "x", "y", "dbh"])
    assert inventory.load_stem_map(path)[0][0].dbh == pytest.approx(0.31)


def test_the_units_can_be_forced(tmp_path):
    """A plot of saplings is all under 5 cm, where the guess would read
    centimetres as metres."""
    path = write_csv(tmp_path / "s.csv", [["1", "0", "0", "3.5"]],
                     ["id", "x", "y", "dbh"])
    stems, _ = inventory.load_stem_map(path, dbh_units="cm")
    assert stems[0].dbh == pytest.approx(0.035)


def test_rows_without_coordinates_are_skipped_not_fatal(tmp_path):
    path = write_csv(
        tmp_path / "s.csv",
        [["1", "10", "20"], ["2", "", ""], ["total", "n/a", "n/a"], ["3", "11", "21"]],
        ["id", "x", "y"],
    )
    stems, _ = inventory.load_stem_map(path)
    assert [s.stem_id for s in stems] == ["1", "3"]


def test_a_file_with_no_coordinates_says_so(tmp_path):
    path = write_csv(tmp_path / "s.csv", [["1", "30"]], ["id", "dbh"])
    with pytest.raises(ValueError, match="X/Y"):
        inventory.load_stem_map(path)


def test_columns_can_be_named_by_hand(tmp_path):
    path = write_csv(tmp_path / "s.csv", [["1", "10", "20", "5", "6"]],
                     ["id", "a", "b", "c", "d"])
    stems, _ = inventory.load_stem_map(path, columns={"x": "a", "y": "b", "dbh": "c"})
    assert (stems[0].x, stems[0].y) == (10.0, 20.0)


# -- lining up the coordinates -----------------------------------------------
def plot_trees(n=20, seed=0):
    """A plot of trees at known positions, with heights and DBHs."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 60, size=(n, 2))
    return [
        TreeStats(
            tree_id=i + 1, x=float(x), y=float(y),
            height=float(10 + 15 * rng.random()),
            dbh=float(0.15 + 0.5 * rng.random()),
        )
        for i, (x, y) in enumerate(xy)
    ]


def stems_from(trees, dx=0.0, dy=0.0, jitter=0.0, seed=1):
    rng = np.random.default_rng(seed)
    return [
        Stem(
            stem_id=f"S{t.tree_id}",
            x=t.x - dx + rng.normal(0, jitter),
            y=t.y - dy + rng.normal(0, jitter),
            dbh=t.dbh,
            height=t.height,
        )
        for t in trees
    ]


def test_a_local_stem_map_finds_its_place_in_a_utm_cloud():
    """The case this exists for: a plot-local map against a georeferenced
    cloud, so the shift is hundreds of kilometres."""
    trees = plot_trees()
    shifted = [
        TreeStats(t.tree_id, t.x + 204300.0, t.y + 7223250.0, t.height, t.dbh)
        for t in trees
    ]
    stems = [Stem(f"S{t.tree_id}", t.x, t.y, t.dbh, t.height) for t in trees]
    fit = inventory.fit_shift(stems, shifted)
    assert fit.dx == pytest.approx(204300.0, abs=0.05)
    assert fit.dy == pytest.approx(7223250.0, abs=0.05)
    assert fit.matched == len(trees)


def test_the_fit_survives_trees_only_one_side_has():
    """A stem map always has trees the scan missed, and the scan always has
    trees the crew never measured. A centroid fit moves with every one of
    them; this one shouldn't."""
    trees = plot_trees(n=24)
    stems = stems_from(trees[:16], dx=12.0, dy=-7.0, jitter=0.25)
    stems += [Stem("ghost1", 300.0, 300.0), Stem("ghost2", -50.0, 80.0)]
    fit = inventory.fit_shift(stems, trees)
    assert fit.dx == pytest.approx(12.0, abs=0.4)
    assert fit.dy == pytest.approx(-7.0, abs=0.4)
    assert fit.matched >= 14


def test_a_small_plot_with_real_position_error_still_lines_up():
    """Found by testing against a generated stem map: a dozen trees give the
    true shift only a handful of votes, and with a metre of position error a
    bin boundary splits them — leaving an accidental alignment of three
    wrong pairs with more votes than the right answer. Several bins are
    refined now, and only the true one pulls the whole plot in."""
    trees = plot_trees(n=12, seed=4)
    stems = stems_from(trees[:10], dx=204295.9, dy=7223245.0, jitter=1.0, seed=5)
    fit = inventory.fit_shift(stems, trees)
    assert fit.dx == pytest.approx(204295.9, abs=1.0)
    assert fit.dy == pytest.approx(7223245.0, abs=1.0)
    assert fit.matched >= 8


def test_the_residual_reports_how_well_it_landed():
    trees = plot_trees()
    loose = inventory.fit_shift(stems_from(trees, dx=5, dy=5, jitter=0.8), trees)
    tight = inventory.fit_shift(stems_from(trees, dx=5, dy=5, jitter=0.05), trees)
    assert tight.residual < loose.residual
    assert "Shift" in tight.describe()


def test_no_stems_or_no_trees_is_a_zero_shift():
    assert inventory.fit_shift([], plot_trees()) == Alignment()
    assert inventory.fit_shift(stems_from(plot_trees()), []) == Alignment()


def test_a_hand_nudged_shift_can_be_rescored():
    trees = plot_trees()
    stems = stems_from(trees, dx=5.0, dy=0.0, jitter=0.1)
    nudged = inventory.alignment_quality(stems, trees, Alignment(dx=5.0))
    assert nudged.matched == len(trees)
    far = inventory.alignment_quality(stems, trees, Alignment(dx=500.0))
    assert far.matched == 0  # the whole map is off the plot


# -- ranking candidates ------------------------------------------------------
def test_height_and_dbh_break_a_tie_position_cant():
    """Two stems a metre either side of the tree: the one that is also the
    right size is the match."""
    tree = TreeStats(1, 0.0, 0.0, height=22.0, dbh=0.45)
    wrong = Stem("near-but-small", 0.9, 0.0, dbh=0.15, height=9.0)
    right = Stem("further-but-right", -1.4, 0.0, dbh=0.44, height=21.5)
    ranked = inventory.candidates(tree, [wrong, right])
    assert ranked[0].stem is right
    assert ranked[0].distance > ranked[1].distance  # and still wins


def test_a_missing_measurement_is_skipped_not_guessed():
    tree = TreeStats(1, 0.0, 0.0, height=20.0, dbh=None)
    stem = Stem("S1", 0.5, 0.0, dbh=0.4, height=20.0)
    best = inventory.candidates(tree, [stem])[0]
    assert best.dbh_diff is None
    assert best.used == ("position", "height")


def test_a_doubtful_dbh_is_not_scored_on():
    """A third of a trunk fits a third of its diameter. Ranking on that is
    worse than not ranking on DBH at all, so a poor circle fit is left out
    of the score (it is still shown in the table)."""
    doubtful = TreeStats(1, 0.0, 0.0, height=20.0, dbh=0.11, dbh_quality=0.30)
    right = Stem("right", 0.6, 0.0, dbh=0.42, height=20.0)
    wrong = Stem("wrong", 0.8, 0.0, dbh=0.11, height=9.0)
    ranked = inventory.candidates(doubtful, [right, wrong])
    assert ranked[0].stem is right
    assert "dbh" not in ranked[0].used


def test_a_perfect_match_scores_zero():
    tree = TreeStats(1, 10.0, 10.0, height=20.0, dbh=0.4)
    stem = Stem("S1", 10.0, 10.0, dbh=0.4, height=20.0)
    assert inventory.candidates(tree, [stem])[0].score == pytest.approx(0.0)


def test_stems_beyond_reach_are_not_candidates():
    tree = TreeStats(1, 0.0, 0.0, height=20.0, dbh=0.4)
    far = Stem("far", 40.0, 0.0, dbh=0.4, height=20.0)
    assert inventory.candidates(tree, [far]) == []
    assert inventory.candidates(
        tree, [far], tolerances=Tolerances(max_distance=50.0)
    )[0].stem is far


def test_the_alignment_is_applied_to_the_stems():
    tree = TreeStats(1, 100.0, 50.0, height=20.0, dbh=0.4)
    stem = Stem("S1", 0.0, 0.0, dbh=0.4, height=20.0)
    assert inventory.candidates(tree, [stem]) == []
    with_shift = inventory.candidates(
        tree, [stem], Alignment(dx=100.0, dy=50.0)
    )
    assert with_shift[0].score == pytest.approx(0.0)


def test_stems_already_matched_can_be_excluded():
    tree = TreeStats(1, 0.0, 0.0, height=20.0, dbh=0.4)
    stems = [Stem("taken", 0.1, 0.0), Stem("free", 1.0, 0.0)]
    ranked = inventory.candidates(tree, stems, exclude={"taken"})
    assert [c.stem.stem_id for c in ranked] == ["free"]


# -- whole-plot matching and the CSV -----------------------------------------
def test_matching_the_plot_is_one_to_one():
    trees = plot_trees(n=12)
    stems = stems_from(trees, dx=3.0, dy=2.0, jitter=0.2)
    matched = inventory.match_all(trees, stems, Alignment(dx=3.0, dy=2.0))
    assert len(matched) == len(trees)
    assert len({c.stem.stem_id for c in matched.values()}) == len(trees)
    assert all(matched[t.tree_id].stem.stem_id == f"S{t.tree_id}" for t in trees)


def test_a_plot_with_fewer_stems_leaves_trees_unmatched():
    trees = plot_trees(n=10)
    stems = stems_from(trees[:4], jitter=0.1)
    matched = inventory.match_all(trees, stems)
    assert len(matched) == 4


def test_a_hopeless_pair_is_left_unmatched():
    """What is left at the end of a greedy pass is the trees nobody measured
    and the stems nobody scanned. Pairing those off would export confident
    nonsense, so a pair scoring worse than the cut is simply not made."""
    trees = [TreeStats(1, 0.0, 0.0, height=20.0, dbh=0.4)]
    stranger = Stem("measured-elsewhere", 11.0, 0.0, dbh=0.1, height=5.0)
    assert inventory.match_all(trees, [stranger]) == {}
    # It is still offered in the panel's ranking, where a person decides.
    assert inventory.candidates(trees[0], [stranger])[0].stem is stranger


def test_the_match_csv_has_a_row_per_tree(tmp_path):
    trees = plot_trees(n=5)
    stems = stems_from(trees, dx=3.0, jitter=0.1)
    alignment = Alignment(dx=3.0)
    matched = inventory.match_all(trees, stems, alignment)
    out = inventory.write_matches(tmp_path / "m.csv", matched, trees, alignment)

    text = out.read_text().splitlines()
    assert text[0].startswith("# segfix inventory matches")
    rows = list(csv.DictReader(text[1:]))
    assert len(rows) == 5
    assert set(rows[0]) == set(inventory.MATCH_HEADER)
    first = rows[0]
    assert first["tree_id"] == "1" and first["stem_id"] == "S1"
    # The stem goes out where the user saw it: shifted onto the cloud.
    assert float(first["stem_x"]) == pytest.approx(trees[0].x, abs=0.3)
