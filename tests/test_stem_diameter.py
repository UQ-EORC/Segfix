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

"""Estimating DBH from a tree's points, to match it against a stem map.

The number matters less than knowing when not to trust it: a circle fitted
through a fork, a leaning stem or one side of a trunk will happily return a
diameter, and if that fed the matching silently it would rank the wrong stem
first. Hence the residual beside every answer, and None rather than a guess
where there is nothing stem-shaped to fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from segfix.analysis import BREAST_HEIGHT, stem_diameter


def stem_points(diameter=0.30, base=0.0, n=1200, arc=2 * np.pi, noise=0.0,
                centre=(5.0, 7.0), seed=0, lean=0.0):
    """A hollow cylinder of points: a scanned trunk, with options for the
    ways a real one is harder than that."""
    rng = np.random.default_rng(seed)
    angle = rng.random(n) * arc
    z = base + rng.random(n) * 3.0
    radius = diameter / 2 + rng.normal(0, noise, n)
    return np.column_stack([
        centre[0] + radius * np.cos(angle) + lean * (z - base),
        centre[1] + radius * np.sin(angle),
        z,
    ])


def crown_points(n=2000, base=0.0, centre=(5.0, 7.0), seed=1):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.normal(centre[0], 1.5, n),
        rng.normal(centre[1], 1.5, n),
        base + 3.0 + rng.random(n) * 8.0,
    ])


def test_a_clean_stem_measures_its_diameter():
    got = stem_diameter(np.vstack([stem_points(0.42), crown_points()]))
    assert got is not None
    diameter, residual = got
    assert diameter == pytest.approx(0.42, abs=0.01)
    assert residual < 0.05


def test_scanner_noise_widens_the_residual_not_the_diameter():
    clean = stem_diameter(stem_points(0.30, noise=0.0))
    noisy = stem_diameter(stem_points(0.30, noise=0.02))
    assert noisy[0] == pytest.approx(0.30, abs=0.02)
    assert noisy[1] > clean[1]


def test_a_tree_standing_on_a_slope_is_measured_from_its_own_foot():
    """base_z is the tree's own lowest points, not z=0, so a plot with 40 m
    of relief doesn't measure 40 m up the neighbouring stem."""
    high = np.vstack([stem_points(0.33, base=120.0), crown_points(base=120.0)])
    diameter, _ = stem_diameter(high)
    assert diameter == pytest.approx(0.33, abs=0.01)


def test_the_base_can_be_given_when_the_ground_is_known():
    points = np.vstack([stem_points(0.33, base=0.0), crown_points()])
    # Told the ground is 1.3 m up, it measures at 2.6 m — still the stem
    # here, so it still fits, which is what makes the argument useful for a
    # cloud with a ground model.
    assert stem_diameter(points, base_z=BREAST_HEIGHT) is not None


def test_a_crown_with_no_stem_is_not_measured():
    assert stem_diameter(crown_points()) is None


def test_too_few_points_is_not_measured():
    assert stem_diameter(stem_points(0.3, n=4)) is None


def test_a_trunk_seen_from_one_side_says_it_is_unsure():
    """A third of an arc still fits a circle; the residual is the only thing
    that knows it was a guess."""
    sliver = np.vstack([
        stem_points(0.30, arc=np.pi / 3, noise=0.01, n=300), crown_points()
    ])
    whole = np.vstack([stem_points(0.30, noise=0.01), crown_points()])
    assert stem_diameter(sliver)[1] > stem_diameter(whole)[1]


def test_points_that_arent_a_circle_are_refused():
    """A wall, or a slab of ground: a circle can be fitted to anything, and
    the only thing that knows better is how far the points sit from it."""
    rng = np.random.default_rng(3)
    wall = np.column_stack([
        rng.random(600) * 3.0, rng.normal(0, 0.02, 600),
        1.0 + rng.random(600) * 0.6,
    ])
    assert stem_diameter(wall, base_z=0.0) is None


def test_a_believable_fit_is_flagged_as_one():
    """GOOD_FIT is where scoring stops trusting the number: a clean trunk,
    a noisy one and half a trunk are under it; a third of an arc, which
    comes back a third of the true size, is not."""
    from segfix.analysis import GOOD_FIT

    def quality(**kwargs):
        return stem_diameter(np.vstack([stem_points(**kwargs), crown_points()]))[1]

    assert quality(noise=0.02) <= GOOD_FIT
    assert quality(noise=0.01, lean=0.25) <= GOOD_FIT
    assert quality(arc=np.pi, noise=0.01) <= GOOD_FIT
    assert quality(arc=np.pi / 3, noise=0.01, n=300) > GOOD_FIT
