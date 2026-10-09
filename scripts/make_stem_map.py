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

"""Make a synthetic field stem map for a segmented cloud, to test matching.

Reads the trees out of a cloud segfix can open and writes the CSV a field
crew would have produced for the same plot — with the errors that make
matching a real problem rather than a lookup:

* **positions are metres out**, as a handheld GPS or a tape-and-compass
  traverse leaves them, and optionally in a **local plot frame**, so the two
  coordinate systems don't line up at all;
* **the crew missed trees** the scanner found, and **measured trees** the
  scan doesn't have (outside the clipped plot, or under a canopy gap);
* **heights are a few per cent out**, as hypsometer readings are;
* **DBH** is measured properly on the ground, in centimetres — where the
  cloud's own stems are fittable it is taken from them and then given a
  measurement error, and otherwise it comes from the tree's height.

So a run of this against the plot the cloud came from is a fair test of
Inventory ▸ Load Stem Map: the shift should be found, most stems should land
on the right tree, and the ones that can't be matched should be the ones
that were never in the cloud.

Usage:
    python scripts/make_stem_map.py sample/sample.ply stems.csv
    python scripts/make_stem_map.py plot.las stems.csv --local --jitter 1.5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from segfix import analysis, inventory  # noqa: E402
from segfix.treecatalog import open_catalog  # noqa: E402

#: Species to spread over the plot, so the species column is worth having.
SPECIES = ("Euca obliqua", "Euca regnans", "Acacia dealbata", "Nothofagus cunn.")

#: Height-to-DBH when the cloud has no fittable stem: a 20 m tree at about
#: 35 cm, which is the right order for a wet-forest eucalypt. Only a stand-in
#: for a measurement — the point is plausible numbers, not an allometry.
_DBH_PER_METRE = 0.018


def tree_table(path: str):
    """Every tree in the cloud, with its position, height and (where the
    points allow it) a fitted DBH. Positions are in the file's own frame.

    The cloud is read with the global shift segfix would offer, because the
    catalog holds coordinates as float32: a raw UTM northing is kept only to
    about half a metre, and a circle fitted through points rounded that
    coarsely is no circle. Read unshifted, a georeferenced plot got a
    fitted DBH for half its trees and a height-based guess for the rest,
    and the guesses were what the matching was then scored against.
    """
    from dataclasses import replace

    catalog = open_catalog(
        path, shift_prompt=lambda mins, maxs, suggested: suggested
    )
    stats = inventory.stats_from_records(catalog.records)
    labels = catalog.file_labels()
    coords = np.asarray(catalog.coords)
    fitted = {}
    for tree in stats:
        points = coords[labels == tree.tree_id]
        measured = analysis.stem_diameter(points) if len(points) else None
        # Only a fit good enough to believe: a cone of points fits a 2 m
        # "stem" quite happily, and writing that into a stem map would make
        # the test meaningless.
        if measured and measured[1] <= analysis.GOOD_FIT:
            fitted[tree.tree_id] = measured[0]
    shift = catalog.global_shift
    if shift is not None:
        stats = [replace(t, x=t.x - float(shift[0]), y=t.y - float(shift[1]))
                 for t in stats]
    return stats, fitted


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cloud", help="a segmented cloud segfix can open")
    ap.add_argument("out", help="stem map CSV to write")
    ap.add_argument("--local", action="store_true",
                    help="write plot-local coordinates (origin at the plot's "
                         "SW corner) instead of the cloud's own")
    ap.add_argument("--jitter", type=float, default=1.0,
                    help="position error in metres, 1-sigma (default 1.0)")
    ap.add_argument("--height-error", type=float, default=0.05,
                    help="height error as a fraction, 1-sigma (default 0.05)")
    ap.add_argument("--dbh-error", type=float, default=0.02,
                    help="DBH error in metres, 1-sigma (default 0.02)")
    ap.add_argument("--missing", type=int, default=2,
                    help="trees the crew never measured (default 2)")
    ap.add_argument("--extra", type=int, default=1,
                    help="stems measured that the scan doesn't have (default 1)")
    ap.add_argument("--rotate", type=float, default=0.0,
                    help="rotate the map by this many degrees about the plot "
                         "centre — segfix fits a shift, not a rotation, so "
                         "this is how to see it refuse to line up")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    stats, fitted = tree_table(args.cloud)
    if not stats:
        print(f"{args.cloud} has no trees to make a stem map from", file=sys.stderr)
        return 1

    xy = np.array([[t.x, t.y] for t in stats])
    origin = xy.min(axis=0) - 5.0 if args.local else np.zeros(2)
    centre = xy.mean(axis=0)

    keep = list(stats)
    rng.shuffle(keep)
    missed = {t.tree_id for t in keep[:max(0, args.missing)]}

    rows = []
    for tree in stats:
        if tree.tree_id in missed:
            continue
        dbh = fitted.get(tree.tree_id, (tree.height or 10.0) * _DBH_PER_METRE)
        dbh = max(0.05, dbh + rng.normal(0, args.dbh_error))
        point = np.array([tree.x, tree.y]) + rng.normal(0, args.jitter, 2)
        rows.append([
            f"P{tree.tree_id:03d}",
            *_place(point, origin, centre, args.rotate),
            round(dbh * 100, 1),
            round((tree.height or 0.0) * rng.normal(1.0, args.height_error), 1),
            SPECIES[tree.tree_id % len(SPECIES)],
        ])

    # Trees the crew measured that the scan hasn't got: just outside the
    # cloud's footprint, which is where they really come from.
    span = xy.max(axis=0) - xy.min(axis=0)
    for n in range(max(0, args.extra)):
        point = xy.min(axis=0) - (span * 0.15) * rng.random(2) - 2.0
        rows.append([
            f"X{n + 1:03d}",
            *_place(point, origin, centre, args.rotate),
            round(float(rng.uniform(15, 55)), 1),
            round(float(rng.uniform(8, 25)), 1),
            SPECIES[n % len(SPECIES)],
        ])

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Tree ID", "Easting", "Northing", "DBH (cm)",
                         "Height (m)", "Species"])
        writer.writerows(rows)

    frame = "plot-local" if args.local else "the cloud's own"
    print(
        f"Wrote {len(rows)} stems to {args.out}: {len(stats) - len(missed)} of "
        f"{len(stats)} trees in {args.cloud}, {args.extra} never scanned, "
        f"{len(missed)} never measured."
    )
    print(
        f"Coordinates are {frame}"
        + (f", offset by {origin[0]:,.1f}, {origin[1]:,.1f} m" if args.local else "")
        + f". Positions ±{args.jitter} m, heights ±{args.height_error:.0%}."
    )
    print(
        f"DBH came from the cloud for {len(fitted)} tree(s); the rest is "
        "from height (the stems weren't round enough to fit)."
    )
    if args.rotate:
        print(f"Rotated by {args.rotate}°: the fit should *not* line this up.")
    return 0


def _place(point, origin, centre, rotate_deg: float):
    """A measured position in the output frame: rotated about the plot
    centre if asked, then moved into local coordinates."""
    if rotate_deg:
        angle = np.radians(rotate_deg)
        offset = point - centre
        point = centre + np.array([
            offset[0] * np.cos(angle) - offset[1] * np.sin(angle),
            offset[0] * np.sin(angle) + offset[1] * np.cos(angle),
        ])
    placed = point - origin
    return round(float(placed[0]), 2), round(float(placed[1]), 2)


if __name__ == "__main__":
    sys.exit(main())
