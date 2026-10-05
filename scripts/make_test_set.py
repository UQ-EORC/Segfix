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

"""Generate a set of synthetic plots with matching field inventory data.

``make_sample.py`` builds one plot of deliberate *segmentation* errors, for
practising the editing loop. This builds a handful of plots for the other
half of the job — measuring and matching — each one a different thing that
makes it hard:

===================  =========================================================
``open_plot``        15 well-spaced trees. The easy case, and the one to
                     check a change against first.
``closed_canopy``    40 trees in 30 m, crowns overlapping and stems a metre
                     apart, where the nearest stem to a tree is routinely
                     the wrong one.
``sloped``           15 trees on ground falling 15°, so breast height has to
                     be measured from each stem's own foot, not from a plot
                     elevation.
``leaning``          stems leaning up to 20°, which is what a circle fitted
                     through a horizontal slab is worst at — the fit should
                     say so rather than report a wide stem.
``utm_plot``         the open plot georeferenced into UTM and scanned finer,
                     so it also triggers the large-coordinate and dense-cloud
                     prompts on load.
===================  =========================================================

Each plot is written with the truth it was built from and two stem maps:

* ``<plot>.las`` (``open_plot`` also as ``.ply``) — the cloud, with a
  ``treeID`` per point and unassigned ground beneath;
* ``<plot>_truth.csv`` — exactly where each tree is, and its real height and
  DBH, so a matching run can be scored rather than eyeballed;
* ``<plot>_stems.csv`` — the stem map a field crew would have produced:
  positions metres out, heights a few per cent out, DBH to the nearest
  centimetre, some trees never measured and some stems never scanned;
* ``<plot>_stems_local.csv`` — the same map in plot-local coordinates, which
  is the case the alignment exists for.

Unlike ``make_stem_map.py``, which derives a map from a cloud you already
have, the maps here come from the truth: the error in them is known, so
"matched 38 of 40, and the two it missed are the two the crew never
measured" is a statement about the matching rather than about the data.

Usage:
    python scripts/make_test_set.py ~/segfix_test_data
    python scripts/make_test_set.py ~/segfix_test_data --only closed_canopy
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from segfix.model import NOISE, UNASSIGNED  # noqa: E402

#: Points per square metre of surface. A terrestrial scan is far denser than
#: this; it is chosen so a whole set generates in seconds and still leaves
#: ~50 points around a 30 cm stem at breast height, which is what a circle
#: fit needs.
DENSITY = 900.0

SPECIES = ("Euca obliqua", "Euca regnans", "Acacia dealbata", "Nothofagus cunn.")


@dataclass
class Tree:
    """The truth about one generated tree."""

    tree_id: int
    x: float
    y: float
    base_z: float
    height: float
    dbh: float
    crown_radius: float
    lean: float = 0.0
    species: str = ""


# -- shapes ------------------------------------------------------------------
def trunk_points(tree: Tree, rng) -> np.ndarray:
    """A hollow stem: points on the bark, not inside it.

    Hollow matters. A solid column of points fits a circle through its
    middle and reports a diameter of whatever the column is wide; only a
    shell behaves like a scanned trunk, where the fit has to find the
    surface and the residual means something.
    """
    stem_height = tree.height * 0.75
    area = np.pi * tree.dbh * stem_height
    n = max(400, int(area * DENSITY))
    z = rng.random(n) * stem_height
    # A gentle taper, as a stem has: full diameter at the foot, four fifths
    # of it at the top of the bole.
    radius = tree.dbh / 2 * (1.0 - 0.2 * z / max(stem_height, 1e-6))
    theta = rng.random(n) * 2 * np.pi
    lean_x = np.tan(np.radians(tree.lean)) * z
    return np.column_stack([
        tree.x + lean_x + radius * np.cos(theta),
        tree.y + radius * np.sin(theta),
        tree.base_z + z,
    ])


def crown_points(tree: Tree, rng) -> np.ndarray:
    """A shell of foliage over the top two thirds of the tree."""
    base = tree.height * 0.35
    span = tree.height - base
    area = np.pi * tree.crown_radius * (tree.crown_radius + span)
    n = max(600, int(area * DENSITY * 0.45))
    t = rng.random(n)
    z = base + t * span
    # Widest a third of the way up the crown, tapering to the leader.
    profile = np.sin(np.clip(t, 0, 1) * np.pi * 0.85 + 0.15)
    radius = tree.crown_radius * profile * (0.75 + 0.25 * rng.random(n))
    theta = rng.random(n) * 2 * np.pi
    lean_x = np.tan(np.radians(tree.lean)) * z
    return np.column_stack([
        tree.x + lean_x + radius * np.cos(theta),
        tree.y + radius * np.sin(theta),
        tree.base_z + z + rng.normal(0, 0.15, n),
    ])


def ground_points(bounds, slope_deg: float, rng) -> np.ndarray:
    (xlo, xhi), (ylo, yhi) = bounds
    n = int((xhi - xlo) * (yhi - ylo) * 60)
    x = rng.uniform(xlo, xhi, n)
    y = rng.uniform(ylo, yhi, n)
    return np.column_stack([x, y, terrain(x, y, xlo, slope_deg) + rng.normal(0, 0.03, n)])


def terrain(x, y, x0: float, slope_deg: float):
    """Ground elevation: a plane falling across the plot, plus a low roll so
    it is not perfectly flat."""
    return (
        np.tan(np.radians(slope_deg)) * (np.asarray(x) - x0)
        + 0.3 * np.sin(np.asarray(x) / 7.0)
        + 0.2 * np.cos(np.asarray(y) / 9.0)
    )


def noise_points(bounds, top: float, n, rng) -> np.ndarray:
    (xlo, xhi), (ylo, yhi) = bounds
    return np.column_stack([
        rng.uniform(xlo, xhi, n), rng.uniform(ylo, yhi, n),
        rng.uniform(top * 0.4, top * 1.3, n),
    ])


# -- plots -------------------------------------------------------------------
def place_trees(count, bounds, rng, spacing: float, lean: float = 0.0,
                slope_deg: float = 0.0) -> list[Tree]:
    """Scatter trees, no two closer than ``spacing``, each sized plausibly
    for its height."""
    (xlo, xhi), (ylo, yhi) = bounds
    placed: list[Tree] = []
    attempts = 0
    while len(placed) < count and attempts < count * 400:
        attempts += 1
        x, y = rng.uniform(xlo, xhi), rng.uniform(ylo, yhi)
        if any(np.hypot(t.x - x, t.y - y) < spacing for t in placed):
            continue
        height = float(rng.uniform(9.0, 28.0))
        placed.append(Tree(
            tree_id=len(placed) + 1,
            x=float(x), y=float(y),
            base_z=float(terrain(x, y, xlo, slope_deg)),
            height=height,
            # Tall trees are thick trees, with real spread around it.
            dbh=float(np.clip(height * rng.normal(0.019, 0.004), 0.08, 1.2)),
            crown_radius=float(height * rng.uniform(0.12, 0.22)),
            lean=float(rng.uniform(-lean, lean)),
            species=SPECIES[len(placed) % len(SPECIES)],
        ))
    return placed


def build_cloud(trees, bounds, slope_deg, rng, origin=(0.0, 0.0, 0.0)):
    """Points and labels for a whole plot: trees, ground, a little noise."""
    blocks, labels = [], []
    for tree in trees:
        points = np.vstack([trunk_points(tree, rng), crown_points(tree, rng)])
        blocks.append(points)
        labels.append(np.full(len(points), tree.tree_id, dtype=np.int32))

    floor = ground_points(bounds, slope_deg, rng)
    blocks.append(floor)
    labels.append(np.full(len(floor), UNASSIGNED, dtype=np.int32))

    top = max((t.base_z + t.height for t in trees), default=10.0)
    flyers = noise_points(bounds, top, 20, rng)
    blocks.append(flyers)
    labels.append(np.full(len(flyers), NOISE, dtype=np.int32))

    coords = np.vstack(blocks) + np.asarray(origin, dtype=float)
    return coords, np.concatenate(labels)


def thin_to_spacing(coords, labels, spacing: float | None, rng):
    """Drop points to roughly one per ``spacing`` cube, so a plot can be
    made as fine or coarse as the test needs."""
    if spacing is None:
        return coords, labels
    keys = np.floor(coords / spacing).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep.sort()
    return coords[keep], labels[keep]


# -- writing -----------------------------------------------------------------
def write_cloud(coords, labels, path: Path) -> None:
    if path.suffix.lower() == ".ply":
        _write_ply(coords, labels, path)
    else:
        _write_las(coords, labels, path)


def _write_las(coords, labels, path: Path) -> None:
    import laspy

    header = laspy.LasHeader(version="1.4", point_format=6)
    header.offsets = np.floor(coords.min(axis=0))
    header.scales = [0.001, 0.001, 0.001]
    header.add_extra_dim(laspy.ExtraBytesParams(
        name="treeID", type=np.int32, description="Unique ID per tree"
    ))
    las = laspy.LasData(header)
    las.x, las.y, las.z = coords[:, 0], coords[:, 1], coords[:, 2]
    las.treeID = np.where(labels == NOISE, UNASSIGNED, labels).astype(np.int32)
    las.write(str(path))


def _write_ply(coords, labels, path: Path) -> None:
    rows = np.empty(len(coords), dtype=np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("treeID", "<i4"),
    ]))
    rows["x"], rows["y"], rows["z"] = coords.T
    rows["treeID"] = labels
    with open(path, "wb") as fh:
        fh.write(
            b"ply\nformat binary_little_endian 1.0\n"
            + f"element vertex {len(coords)}\n".encode()
            + b"property float x\nproperty float y\nproperty float z\n"
              b"property int treeID\nend_header\n"
        )
        fh.write(rows.tobytes())


def write_truth(trees, path: Path, origin=(0.0, 0.0, 0.0)) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tree_id", "x", "y", "base_z", "height_m", "dbh_m",
                         "lean_deg", "species"])
        for tree in trees:
            writer.writerow([
                tree.tree_id, round(tree.x + origin[0], 3),
                round(tree.y + origin[1], 3), round(tree.base_z + origin[2], 3),
                round(tree.height, 2), round(tree.dbh, 3),
                round(tree.lean, 1), tree.species,
            ])


def write_stem_map(trees, path: Path, rng, origin=(0.0, 0.0, 0.0),
                   jitter=1.0, height_error=0.05, dbh_error=0.01,
                   missing=2, extra=1, local=False) -> dict:
    """The CSV a field crew would hand over, and what was done to it."""
    shift = (0.0, 0.0) if not local else (origin[0], origin[1])
    chosen = list(trees)
    rng.shuffle(chosen)
    missed = {t.tree_id for t in chosen[:max(0, missing)]}

    rows = []
    for tree in trees:
        if tree.tree_id in missed:
            continue
        rows.append([
            f"P{tree.tree_id:03d}",
            round(tree.x + origin[0] - shift[0] + rng.normal(0, jitter), 2),
            round(tree.y + origin[1] - shift[1] + rng.normal(0, jitter), 2),
            round(tree.dbh * 100 + rng.normal(0, dbh_error * 100), 1),
            round(tree.height * rng.normal(1.0, height_error), 1),
            tree.species,
        ])
    xs = [t.x for t in trees]
    ys = [t.y for t in trees]
    for n in range(max(0, extra)):
        rows.append([
            f"X{n + 1:03d}",
            round(min(xs) + origin[0] - shift[0] - rng.uniform(4, 9), 2),
            round(min(ys) + origin[1] - shift[1] - rng.uniform(4, 9), 2),
            round(float(rng.uniform(15, 60)), 1),
            round(float(rng.uniform(9, 26)), 1),
            SPECIES[n % len(SPECIES)],
        ])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Tree ID", "Easting", "Northing", "DBH (cm)",
                         "Height (m)", "Species"])
        writer.writerows(rows)
    return {"measured": len(rows) - max(0, extra), "missed": sorted(missed),
            "never_scanned": max(0, extra)}


# -- the set -----------------------------------------------------------------
#: name -> (trees, plot size m, min stem spacing m, lean°, slope°, origin,
#:          point spacing m or None, extra formats)
PLOTS = {
    "open_plot": dict(count=15, size=30.0, spacing=5.0, lean=0.0, slope=0.0,
                      origin=(0.0, 0.0, 0.0), thin=0.02, also_ply=True),
    "closed_canopy": dict(count=40, size=30.0, spacing=2.2, lean=4.0, slope=0.0,
                          origin=(0.0, 0.0, 0.0), thin=0.03, also_ply=False),
    "sloped": dict(count=15, size=30.0, spacing=5.0, lean=0.0, slope=15.0,
                   origin=(0.0, 0.0, 0.0), thin=0.02, also_ply=False),
    "leaning": dict(count=12, size=30.0, spacing=5.0, lean=20.0, slope=0.0,
                    origin=(0.0, 0.0, 0.0), thin=0.02, also_ply=False),
    "utm_plot": dict(count=15, size=30.0, spacing=5.0, lean=0.0, slope=3.0,
                     origin=(204300.0, 7223250.0, 112.0), thin=0.012,
                     also_ply=False),
}


def build(name: str, out_dir: Path, seed: int) -> list[str]:
    spec = PLOTS[name]
    rng = np.random.default_rng(seed)
    size = spec["size"]
    bounds = ((0.0, size), (0.0, size))
    trees = place_trees(spec["count"], bounds, rng, spec["spacing"],
                        lean=spec["lean"], slope_deg=spec["slope"])
    coords, labels = build_cloud(trees, bounds, spec["slope"], rng,
                                 origin=spec["origin"])
    coords, labels = thin_to_spacing(coords, labels, spec["thin"], rng)

    written = []
    cloud = out_dir / f"{name}.las"
    write_cloud(coords, labels, cloud)
    written.append(cloud.name)
    if spec["also_ply"]:
        ply = out_dir / f"{name}.ply"
        write_cloud(coords.astype(np.float32), labels, ply)
        written.append(ply.name)

    write_truth(trees, out_dir / f"{name}_truth.csv", spec["origin"])
    written.append(f"{name}_truth.csv")
    for suffix, local in (("_stems.csv", False), ("_stems_local.csv", True)):
        write_stem_map(trees, out_dir / f"{name}{suffix}",
                       np.random.default_rng(seed + 11), origin=spec["origin"],
                       local=local)
        written.append(f"{name}{suffix}")
    print(
        f"  {name}: {len(trees)} trees, {len(coords):,} points "
        f"({cloud.stat().st_size / 1e6:.1f} MB)"
    )
    return written


def check(out_dir: Path, names) -> int:
    """Score the generated set against its own truth.

    The set is only worth having if the right answer is reachable from it,
    so this reports what the matching actually does: how many stems land on
    the tree they were measured from, and how close the fitted DBH is to the
    one the tree was built with. A plot that scores badly here is a bug in
    segfix or in this generator, not a hard plot.
    """
    from segfix import analysis, inventory
    from segfix.treecatalog import open_catalog

    print(f"\n{'plot':15s} {'DBH fitted':>11} {'DBH error':>10} "
          f"{'residual':>9} {'matched':>8} {'correct':>8}")
    worst = 0
    for name in names:
        cloud = out_dir / f"{name}.las"
        if not cloud.exists():
            continue
        # Accepting the global shift, as a user does on load: raw UTM in
        # float32 is held to about half a metre, which is coarser than
        # everything being measured here.
        catalog = open_catalog(
            str(cloud), shift_prompt=lambda mins, maxs, suggested: suggested
        )
        labels = catalog.file_labels()
        coords = np.asarray(catalog.coords)
        truth = {
            int(row["tree_id"]): row
            for row in csv.DictReader(open(out_dir / f"{name}_truth.csv"))
        }
        stats, errors, fitted = [], [], 0
        for tree_id in sorted(catalog.records):
            measured = inventory.stats_from_points(coords, labels, tree_id)
            stats.append(measured)
            if (measured.dbh is not None and measured.dbh_quality is not None
                    and measured.dbh_quality <= analysis.GOOD_FIT):
                fitted += 1
                errors.append(abs(measured.dbh - float(truth[tree_id]["dbh_m"])))

        stems, _ = inventory.load_stem_map(out_dir / f"{name}_stems_local.csv")
        alignment = inventory.fit_shift(stems, stats)
        matched = inventory.match_all(stats, stems, alignment)
        right = sum(1 for tree_id, candidate in matched.items()
                    if candidate.stem.stem_id == f"P{tree_id:03d}")
        worst = max(worst, len(matched) - right)
        error = f"{np.mean(errors) * 100:7.1f}cm" if errors else "        -"
        print(f"{name:15s} {fitted:6d}/{len(stats):<4d} {error} "
              f"{alignment.residual:8.2f}m {len(matched):8d} {right:8d}")
    print(
        "\nEvery matched tree should be the one its stem was measured from; "
        "the trees left unmatched should be the ones the crew never measured."
    )
    return 1 if worst else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", help="directory to write the set into")
    ap.add_argument("--only", choices=sorted(PLOTS), action="append",
                    help="build just this plot (repeatable)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--check", action="store_true",
                    help="after writing, score the set against its own truth")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = args.only or list(PLOTS)
    print(f"Writing {len(names)} plot(s) to {out_dir}")
    for index, name in enumerate(names):
        build(name, out_dir, args.seed + index * 100)

    (out_dir / "README.md").write_text(_readme(names))
    print(f"\nWrote {out_dir / 'README.md'} describing the set.")
    return check(out_dir, names) if args.check else 0


def _readme(names) -> str:
    lines = [
        "# segfix synthetic test set",
        "",
        "Generated by `scripts/make_test_set.py`. Each plot has a cloud, the",
        "truth it was built from, and two field stem maps, one in the",
        "cloud's own coordinates, one plot-local.",
        "",
        "| file | what it is |",
        "|------|------------|",
        "| `<plot>.las` | the cloud: `treeID` per point, `0` = ground |",
        "| `<plot>_truth.csv` | real position, height, DBH and lean per tree |",
        "| `<plot>_stems.csv` | field stem map: positions ±1 m, heights ±5%, "
        "DBH ±1 cm, 2 trees never measured, 1 stem never scanned |",
        "| `<plot>_stems_local.csv` | the same map in plot-local coordinates |",
        "",
        "## The plots",
        "",
    ]
    blurbs = {
        "open_plot": "15 well-spaced trees. The easy case, also written as "
                     "`.ply` to cover the other format.",
        "closed_canopy": "40 trees in 30 m, crowns overlapping, stems about "
                         "2 m apart: the nearest stem to a tree is often the "
                         "wrong one.",
        "sloped": "Ground falling 15°, so breast height has to come from each "
                  "stem's own foot.",
        "leaning": "Stems leaning up to 20°, the case a circle fitted through "
                   "a horizontal slab is worst at.",
        "utm_plot": "Georeferenced into UTM and scanned finer, which also "
                    "trips the large-coordinate and dense-cloud prompts on "
                    "load.",
    }
    for name in names:
        lines.append(f"* **`{name}`**: {blurbs.get(name, '')}")
    lines += [
        "",
        "## Trying the inventory matching",
        "",
        "Open a plot, then **Inventory ▸ Load Stem Map…** and pick its",
        "`_stems_local.csv`: the shift should be found automatically, and",
        "the stems should land on the trees. `_truth.csv` says what the",
        "right answer was.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
