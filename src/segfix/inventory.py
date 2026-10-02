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

"""Field inventory (stem map) matching: which measured tree is this one?

A plot usually has a stem map long before it has a point cloud — a CSV of
stem positions with DBH and height, measured on the ground. Lining the two
up is how a segmented cloud becomes comparable with field data, and the two
hard parts are both here:

* **the coordinates.** A stem map may be in the cloud's CRS, in a local plot
  frame, or in a global one measured with a handheld GPS. :func:`fit_shift`
  finds the translation between the two sets from the stem *pattern* rather
  than from any one tree, so a local map and a 10 km offset cost the same.
* **which stem is which tree.** Position alone is ambiguous in a closed
  stand, where neighbours are a metre apart and the map's own positions are
  metres out. :func:`candidates` ranks by position, height and DBH together,
  so a tall thin tree is not matched to the short fat one beside it.

Nothing here touches Qt or the cloud: it takes stems, tree stats and numbers,
and returns rankings. :mod:`segfix.inventory_ui` is the window onto it.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

#: Column names seen in the wild, by what they mean. Matched case- and
#: separator-insensitively (``Tree ID``, ``tree_id`` and ``TreeID`` are one).
COLUMN_ALIASES = {
    "stem_id": ("stemid", "treeid", "id", "tree", "stem", "number", "no",
                "plottreeid", "tag"),
    "x": ("x", "easting", "xcoord", "xm", "east", "gpsx", "longitude", "lon"),
    "y": ("y", "northing", "ycoord", "ym", "north", "gpsy", "latitude", "lat"),
    "dbh": ("dbh", "diameter", "dbhcm", "dbhm", "d", "diametercm", "dbh13"),
    "height": ("height", "h", "ht", "treeheight", "heightm", "totalheight"),
    "species": ("species", "spp", "sp", "taxon"),
}

#: A DBH column this large is centimetres, not metres — the median is taken,
#: so one bogus row doesn't flip the whole file. 5 m DBH is beyond any tree;
#: 5 cm is a sapling, so there is no overlap to get wrong.
_CM_THRESHOLD = 5.0


@dataclass(frozen=True)
class Stem:
    """One measured tree from the stem map. Distances in metres."""

    stem_id: str
    x: float
    y: float
    dbh: float | None = None
    height: float | None = None
    species: str = ""

    @property
    def label(self) -> str:
        return f"{self.stem_id} ({self.species})" if self.species else self.stem_id


@dataclass(frozen=True)
class TreeStats:
    """What the cloud knows about one tree, to match a :class:`Stem` against.

    ``dbh_quality`` is the circle fit's relative residual (see
    :func:`segfix.analysis.stem_diameter`) — small is round and well sampled.
    ``None`` anywhere means "not measurable", and scoring skips that term
    rather than guessing a value for it.
    """

    tree_id: int
    x: float
    y: float
    height: float | None = None
    dbh: float | None = None
    dbh_quality: float | None = None


@dataclass(frozen=True)
class Tolerances:
    """How far out each measurement may be before it counts as a mismatch.

    Not thresholds: each difference is divided by its tolerance, so they set
    how the three trade off. The defaults say a metre of position error is
    worth about 1.5 m of height error or 2 cm of DBH — roughly how far each
    one is trusted between a handheld-GPS stem map and a segmented cloud.
    """

    position: float = 2.0
    height: float = 3.0
    dbh: float = 0.04
    #: Beyond this, a stem is not a candidate for the tree at all.
    max_distance: float = 15.0
    #: Circle-fit residual above which a tree's DBH is shown but not scored
    #: on — see :data:`segfix.analysis.GOOD_FIT`. A third of a trunk fits a
    #: third of its diameter, and ranking on that is worse than not ranking
    #: on DBH at all.
    dbh_quality: float = 0.2


@dataclass(frozen=True)
class Candidate:
    """One stem ranked against one tree. ``score`` is a distance: 0 is a
    perfect match, and anything under about 1 is within tolerance on every
    measurement that both sides have."""

    stem: Stem
    distance: float
    height_diff: float | None
    dbh_diff: float | None
    score: float
    used: tuple[str, ...] = ()


@dataclass(frozen=True)
class Alignment:
    """The shift from stem-map coordinates onto the cloud's."""

    dx: float = 0.0
    dy: float = 0.0
    #: How many stems landed within :attr:`Tolerances.position` of a tree.
    matched: int = 0
    #: Median nearest-tree distance over those, in metres.
    residual: float = float("nan")

    def apply(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, dtype=float) + (self.dx, self.dy)

    def shift(self, stem: Stem) -> tuple[float, float]:
        return stem.x + self.dx, stem.y + self.dy

    def describe(self) -> str:
        if not self.matched:
            return "Not aligned yet"
        return (
            f"Shift {self.dx:+.2f}, {self.dy:+.2f} m · {self.matched} stem(s) "
            f"within reach · median {self.residual:.2f} m"
        )


# -- reading a stem map ------------------------------------------------------
def _norm(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def detect_columns(header: Sequence[str]) -> dict[str, str]:
    """Guess which column is which, as ``{field: column name}``.

    Only x and y are required downstream; everything else is matched if it
    is there. A column already claimed by an earlier field is not reused —
    ``x`` would otherwise also answer to a bare ``dbh`` alias of ``d``.
    """
    taken: set[str] = set()
    found: dict[str, str] = {}
    for meaning, aliases in COLUMN_ALIASES.items():
        for column in header:
            if column in taken:
                continue
            if _norm(column) in aliases:
                found[meaning] = column
                taken.add(column)
                break
    return found


def load_stem_map(
    path: str | Path,
    columns: dict[str, str] | None = None,
    dbh_units: str = "auto",
) -> tuple[list[Stem], dict[str, str]]:
    """Read a stem-map CSV into :class:`Stem` objects.

    ``columns`` overrides the guesses from :func:`detect_columns`.
    ``dbh_units`` is ``"auto"`` (centimetres if the median is over 5 — see
    :data:`_CM_THRESHOLD`), ``"cm"`` or ``"m"``. Rows without usable
    coordinates are skipped rather than failing the file: a stem map
    routinely carries a blank row, a total, or a tree that was never found.

    Returns the stems and the columns actually used, so a dialog can show
    what it decided.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        header = [name for name in (reader.fieldnames or []) if name]
        used = dict(detect_columns(header))
        if columns:
            used.update({k: v for k, v in columns.items() if v})
        if "x" not in used or "y" not in used:
            raise ValueError(
                "No X/Y columns found in the stem map — name them x and y, "
                "or pick the columns by hand."
            )
        rows = [row for row in reader]

    def number(row, meaning):
        name = used.get(meaning)
        if not name:
            return None
        try:
            return float(str(row.get(name, "")).strip())
        except (TypeError, ValueError):
            return None

    raw: list[tuple[dict, float, float]] = []
    for row in rows:
        x, y = number(row, "x"), number(row, "y")
        if x is None or y is None:
            continue
        raw.append((row, x, y))

    dbhs = [d for d in (number(row, "dbh") for row, _, _ in raw) if d]
    if dbh_units == "auto":
        scale = 0.01 if dbhs and float(np.median(dbhs)) > _CM_THRESHOLD else 1.0
    else:
        scale = 0.01 if dbh_units == "cm" else 1.0

    stems = []
    for index, (row, x, y) in enumerate(raw, start=1):
        dbh = number(row, "dbh")
        stem_id = str(row.get(used.get("stem_id", ""), "")).strip()
        stems.append(Stem(
            stem_id=stem_id or str(index),
            x=x,
            y=y,
            dbh=dbh * scale if dbh else None,
            height=number(row, "height"),
            species=str(row.get(used.get("species", ""), "")).strip(),
        ))
    return stems, used


# -- lining the two coordinate systems up ------------------------------------
def fit_shift(
    stems: Sequence[Stem],
    trees: Sequence[TreeStats],
    tolerance: float = 2.0,
    bin_size: float = 1.0,
) -> Alignment:
    """Find the translation that puts ``stems`` onto ``trees``.

    By vote, then refinement. Every stem→tree difference vector is a
    candidate shift; the true one is proposed once per correctly paired
    tree, while wrong pairings scatter, so the fullest bin of a 2D histogram
    of those vectors is the answer to within ``bin_size``. The winning bin's
    pairs are then averaged for a continuous shift, twice, each time
    re-pairing every stem with its nearest tree.

    This is robust where a least-squares fit of the two centroids is not:
    the sets rarely cover the same trees (a stem map has trees the scan
    missed and vice versa), and a centroid moves with every one of those.
    Returns a zero :class:`Alignment` when either side is empty.
    """
    stem_xy = np.array([[s.x, s.y] for s in stems], dtype=float)
    tree_xy = np.array([[t.x, t.y] for t in trees], dtype=float)
    if not len(stem_xy) or not len(tree_xy):
        return Alignment()

    # Every pairwise difference, binned. Capped: a 2000-stem map against a
    # 2000-tree plot is 4M vectors, which is still a blink, but the cap
    # keeps the memory flat on anything larger.
    offsets = (tree_xy[None, :, :] - stem_xy[:, None, :]).reshape(-1, 2)
    keys = np.round(offsets / bin_size).astype(np.int64)
    _, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    best = int(np.argmax(counts))
    shift = offsets[inverse == best].mean(axis=0)

    for _ in range(2):
        shift = _refine_shift(stem_xy + shift, tree_xy, tolerance) + shift
    return _alignment_for(stem_xy, tree_xy, shift, tolerance)


def _nearest(points: np.ndarray, targets: np.ndarray):
    """Nearest target to each point: ``(index, distance)``."""
    deltas = targets[None, :, :] - points[:, None, :]
    distances = np.hypot(deltas[:, :, 0], deltas[:, :, 1])
    index = np.argmin(distances, axis=1)
    return index, distances[np.arange(len(points)), index]


def _refine_shift(
    shifted: np.ndarray, tree_xy: np.ndarray, tolerance: float
) -> np.ndarray:
    index, distance = _nearest(shifted, tree_xy)
    close = distance <= tolerance
    if not close.any():
        return np.zeros(2)
    return (tree_xy[index[close]] - shifted[close]).mean(axis=0)


def _alignment_for(
    stem_xy: np.ndarray, tree_xy: np.ndarray, shift, tolerance: float
) -> Alignment:
    _, distance = _nearest(stem_xy + shift, tree_xy)
    close = distance <= tolerance
    return Alignment(
        dx=float(shift[0]),
        dy=float(shift[1]),
        matched=int(close.sum()),
        residual=float(np.median(distance[close])) if close.any() else float("nan"),
    )


def alignment_quality(
    stems: Sequence[Stem],
    trees: Sequence[TreeStats],
    alignment: Alignment,
    tolerance: float = 2.0,
) -> Alignment:
    """``alignment`` with its match count and residual recomputed — what a
    hand-nudged shift needs before it can be shown."""
    stem_xy = np.array([[s.x, s.y] for s in stems], dtype=float)
    tree_xy = np.array([[t.x, t.y] for t in trees], dtype=float)
    if not len(stem_xy) or not len(tree_xy):
        return replace(alignment, matched=0, residual=float("nan"))
    quality = _alignment_for(
        stem_xy, tree_xy, np.array([alignment.dx, alignment.dy]), tolerance
    )
    return replace(alignment, matched=quality.matched, residual=quality.residual)


# -- ranking stems against one tree ------------------------------------------
def candidates(
    tree: TreeStats,
    stems: Iterable[Stem],
    alignment: Alignment = Alignment(),
    tolerances: Tolerances = Tolerances(),
    limit: int = 8,
    exclude: Iterable[str] = (),
) -> list[Candidate]:
    """The stems that could be ``tree``, best first.

    The score is the root-mean-square of each difference over its tolerance,
    across the measurements *both* sides have — so a stem map with no
    heights still ranks by position and DBH, and a tree whose DBH could not
    be fitted is not penalised for it. Averaging rather than summing keeps
    a two-measurement score comparable with a three-measurement one.

    ``exclude`` drops stems already matched to another tree.
    """
    excluded = set(exclude)
    out: list[Candidate] = []
    for stem in stems:
        if stem.stem_id in excluded:
            continue
        sx, sy = alignment.shift(stem)
        distance = math.hypot(sx - tree.x, sy - tree.y)
        if distance > tolerances.max_distance:
            continue
        terms = [(distance / tolerances.position) ** 2]
        used = ["position"]
        height_diff = dbh_diff = None
        if stem.height is not None and tree.height is not None:
            height_diff = tree.height - stem.height
            terms.append((height_diff / tolerances.height) ** 2)
            used.append("height")
        believable = (
            tree.dbh_quality is None or tree.dbh_quality <= tolerances.dbh_quality
        )
        if stem.dbh is not None and tree.dbh is not None and believable:
            dbh_diff = tree.dbh - stem.dbh
            terms.append((dbh_diff / tolerances.dbh) ** 2)
            used.append("dbh")
        out.append(Candidate(
            stem=stem,
            distance=distance,
            height_diff=height_diff,
            dbh_diff=dbh_diff,
            score=math.sqrt(sum(terms) / len(terms)),
            used=tuple(used),
        ))
    out.sort(key=lambda c: c.score)
    return out[:limit]


def match_all(
    trees: Sequence[TreeStats],
    stems: Sequence[Stem],
    alignment: Alignment = Alignment(),
    tolerances: Tolerances = Tolerances(),
) -> dict[int, Candidate]:
    """A one-to-one matching over the whole plot, best pairs first.

    Greedy, not optimal: the best pair in the plot is taken, both sides
    struck out, and so on. An optimal assignment would trade a good pair
    for two mediocre ones somewhere else, which is the wrong answer to give
    someone checking a plot tree by tree — here, the pair the operator can
    see is best stays best.
    """
    ranked: list[tuple[float, int, Candidate]] = []
    for tree in trees:
        for candidate in candidates(
            tree, stems, alignment, tolerances, limit=len(stems)
        ):
            ranked.append((candidate.score, tree.tree_id, candidate))
    ranked.sort(key=lambda row: (row[0], row[1]))

    taken_trees: set[int] = set()
    taken_stems: set[str] = set()
    matched: dict[int, Candidate] = {}
    for _, tree_id, candidate in ranked:
        if tree_id in taken_trees or candidate.stem.stem_id in taken_stems:
            continue
        matched[tree_id] = candidate
        taken_trees.add(tree_id)
        taken_stems.add(candidate.stem.stem_id)
    return matched


# -- writing the result out --------------------------------------------------
#: Columns of the match CSV, in order.
MATCH_HEADER = (
    "tree_id", "stem_id", "species", "matched_by", "distance_m",
    "height_diff_m", "dbh_diff_m", "score", "tree_height_m", "tree_dbh_m",
    "stem_height_m", "stem_dbh_m", "stem_x", "stem_y",
)


def write_matches(
    path: str | Path,
    matches: dict[int, Candidate],
    trees: Sequence[TreeStats] = (),
    alignment: Alignment = Alignment(),
    manual: Iterable[int] = (),
) -> Path:
    """Write ``{tree_id: Candidate}`` as the match CSV.

    Stem coordinates go out *shifted onto the cloud*, which is where the
    user saw them; the shift itself is in a comment line above the header so
    the original map can still be recovered from the file.

    ``manual`` lists the trees an operator linked by hand; everything else
    is marked ``auto`` in the ``matched_by`` column. Anyone reading the file
    later needs to know which rows a person stood behind.
    """
    stats = {tree.tree_id: tree for tree in trees}
    by_hand = set(manual)
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(
            f"# segfix inventory matches · stem map shifted by "
            f"{alignment.dx:+.3f}, {alignment.dy:+.3f} m\n"
        )
        writer = csv.writer(fh)
        writer.writerow(MATCH_HEADER)
        for tree_id in sorted(matches):
            candidate = matches[tree_id]
            stem = candidate.stem
            tree = stats.get(tree_id)
            sx, sy = alignment.shift(stem)
            writer.writerow([
                tree_id,
                stem.stem_id,
                stem.species,
                "manual" if tree_id in by_hand else "auto",
                f"{candidate.distance:.3f}",
                "" if candidate.height_diff is None else f"{candidate.height_diff:.2f}",
                "" if candidate.dbh_diff is None else f"{candidate.dbh_diff:.3f}",
                f"{candidate.score:.3f}",
                "" if tree is None or tree.height is None else f"{tree.height:.2f}",
                "" if tree is None or tree.dbh is None else f"{tree.dbh:.3f}",
                "" if stem.height is None else f"{stem.height:.2f}",
                "" if stem.dbh is None else f"{stem.dbh:.3f}",
                f"{sx:.3f}",
                f"{sy:.3f}",
            ])
    return path


# -- tree statistics from the cloud ------------------------------------------
def stats_from_records(records: dict, heights: dict[int, float] | None = None) -> list[TreeStats]:
    """Cheap stats for every tree in a catalog — position and height from
    the per-tree bounding boxes, no points read.

    This is what the alignment fits against: it needs the whole plot, and
    loading every tree's points to measure it would cost minutes where the
    boxes are already in memory. DBH is left unmeasured here; it takes
    points, so only the loaded trees get one (see :func:`stats_from_points`).
    """
    out = []
    for label, record in records.items():
        lo, hi = record.bbox
        out.append(TreeStats(
            tree_id=int(label),
            x=float(record.centroid[0]),
            y=float(record.centroid[1]),
            height=(heights or {}).get(int(label), float(hi[2] - lo[2])),
        ))
    out.sort(key=lambda t: t.tree_id)
    return out


def stats_from_points(coords: np.ndarray, labels: np.ndarray, tree_id: int) -> TreeStats | None:
    """Full stats for one loaded tree: stem position, height and fitted DBH.

    The position is the stem at breast height, not the centroid of the
    points — a leaning tree or a one-sided crown drags a centroid metres
    away from the trunk, which is exactly the error the matching cannot
    afford. Falls back to the centroid when no stem could be fitted.
    """
    from .analysis import BREAST_HEIGHT, stem_diameter

    points = np.asarray(coords)[np.asarray(labels) == tree_id]
    if not len(points):
        return None
    base = float(np.percentile(points[:, 2], 1.0))
    height = float(points[:, 2].max() - base)
    x, y = (float(v) for v in points[:, :2].mean(axis=0))
    dbh = quality = None

    fitted = stem_diameter(points, base_z=base)
    if fitted is not None:
        dbh, quality = fitted
        slab = np.abs(points[:, 2] - (base + BREAST_HEIGHT)) <= 0.15
        if slab.any():
            x, y = (float(v) for v in points[slab][:, :2].mean(axis=0))
    return TreeStats(
        tree_id=int(tree_id), x=x, y=y, height=height,
        dbh=dbh, dbh_quality=quality,
    )


# -- drawing the stems -------------------------------------------------------
#: A stem with no measured height or DBH is still worth drawing — it is a
#: tree someone recorded — so it gets drawn at these, visibly nominal.
DEFAULT_HEIGHT = 8.0
DEFAULT_DBH = 0.25

#: Points around each drawn circle. Sixteen is round enough at the zoom a
#: stem is inspected at, and keeps a 500-stem plot to a few thousand lines.
_SIDES = 16


def stem_geometry(
    stems: Sequence[Stem],
    alignment: Alignment = Alignment(),
    base_z: float | Sequence[float] = 0.0,
    sides: int = _SIDES,
) -> tuple[np.ndarray, np.ndarray]:
    """Line segments drawing each stem as a cylinder of its DBH and height.

    Returns ``(segments, owner)``: ``(2S, 3)`` endpoint pairs for a
    segment-mode line visual, and a ``(2S,)`` index saying which stem each
    endpoint belongs to, which is what lets one visual colour the matched
    stem differently from the rest.

    Three rings — the foot, breast height where the DBH was measured, and
    the top — plus four verticals. A full mesh would hide the points inside
    it; this reads as a cylinder from any angle and the cloud stays visible
    through it, which is the whole job: comparing the drawn stem against the
    trunk it is supposed to sit on.
    """
    if len(stems) == 0:
        return np.empty((0, 3), np.float32), np.empty(0, np.int32)
    bases = (
        np.full(len(stems), float(base_z))
        if np.isscalar(base_z) else np.asarray(base_z, dtype=float)
    )
    angle = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring_a, ring_b = angle, np.roll(angle, -1)

    segments, owner = [], []
    for index, stem in enumerate(stems):
        radius = (stem.dbh or DEFAULT_DBH) / 2.0
        height = stem.height or DEFAULT_HEIGHT
        x, y = alignment.shift(stem)
        foot = bases[index]
        for level in (foot, foot + min(1.3, height * 0.5), foot + height):
            segments.append(np.column_stack([
                x + radius * np.cos(ring_a), y + radius * np.sin(ring_a),
                np.full(sides, level),
            ]))
            segments.append(np.column_stack([
                x + radius * np.cos(ring_b), y + radius * np.sin(ring_b),
                np.full(sides, level),
            ]))
            owner.extend([index] * (2 * sides))
        for corner in range(4):
            theta = corner * np.pi / 2
            px, py = x + radius * np.cos(theta), y + radius * np.sin(theta)
            segments.append(np.array([[px, py, foot]]))
            segments.append(np.array([[px, py, foot + height]]))
            owner.extend([index, index])

    # Interleave: a segment visual wants start, end, start, end…, while the
    # rings above are built a whole circle at a time.
    pairs = np.concatenate([
        np.stack([a, b], axis=1).reshape(-1, 3)
        for a, b in zip(segments[0::2], segments[1::2])
    ])
    return pairs.astype(np.float32), np.asarray(owner, dtype=np.int32)
