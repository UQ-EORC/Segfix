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

"""Segmentation-quality analysis: spatial neighbour queries.

UI-agnostic, like :mod:`~segfix.operations`.  Point-set distances are
approximated by sampling each tree down to a few thousand points and querying
a KD-tree — exact enough for "does this touch?" questions at forest scale,
and fast enough to run per keypress.
"""

from __future__ import annotations

import numpy as np

from .model import NOISE, UNASSIGNED, PointCloud

_SAMPLE_CAP = 3000


def _sample(rng, idx: np.ndarray, cap: int = _SAMPLE_CAP) -> np.ndarray:
    if idx.size <= cap:
        return idx
    return rng.choice(idx, cap, replace=False)


def _kdtree(points):
    """A cKDTree built for speed rather than for a tidy tree.

    ``balanced_tree``/``compact_nodes`` make cKDTree find the median at every
    split and shrink each node to its points — worth it for a tree queried
    many times, but everything here builds a tree, asks it one thing and
    throws it away. Turning both off measured 2.5x faster to build (59ms
    against 24ms on 120k points) for queries that are no slower in practice.
    """
    from scipy.spatial import cKDTree

    return cKDTree(points, balanced_tree=False, compact_nodes=False)


def point_spacing(coords: np.ndarray, rng=None) -> float:
    """Median nearest-neighbour distance over a sample — the cloud's typical
    point pitch.  Used to pick the gap a region grow is allowed to bridge.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = len(coords)
    if n < 2:
        return 0.01
    rng = rng or np.random.default_rng(0)
    sample = coords if n <= 20000 else coords[rng.choice(n, 20000, replace=False)]
    d, _ = _kdtree(coords).query(sample, k=2, workers=-1)
    return float(np.median(d[:, 1])) or 0.01


#: Above this many points, connectivity is computed on one point per
#: (eps / :data:`_VOXEL_DIVISOR`) voxel rather than on every point. Below it
#: the exact pass already answers in well under a tenth of a second, and
#: nothing is gained by approximating.
_VOXEL_ABOVE = 20_000

#: How much finer than ``eps`` the voxels are. At 3 a voxel's diagonal is
#: ``eps * sqrt(3) / 3``, about 0.58 eps -- still comfortably inside ``eps``,
#: which is what makes collapsing one sound (see
#: :func:`connected_components_within`).
_VOXEL_DIVISOR = 3

#: Voxels are only used when they shrink the problem to at most this much of
#: it. Below ``eps``-sized structure -- a gap of about one point spacing --
#: almost every voxel holds a single point, so the grouping pass costs its
#: ~25ms and hands the KD-tree the same work it would have had. Past that the
#: exact answer is both faster and exact, so it is the one to take.
_VOXEL_MIN_COLLAPSE = 0.9


def _components_exact(coords: np.ndarray, eps: float) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(coords)
    pairs = _kdtree(coords).query_pairs(eps, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n, dtype=np.int64)
    data = np.ones(len(pairs), dtype=np.int8)
    graph = coo_matrix(
        (data, (pairs[:, 0], pairs[:, 1])), shape=(n, n)
    ).tocsr()
    return connected_components(graph, directed=False)[1]


def _voxel_groups(coords: np.ndarray, size: float):
    """``(voxel id per point, one representative point per voxel)``."""
    origin = coords.min(axis=0)
    cell = np.floor((coords - origin) / size).astype(np.int64)
    dims = cell.max(axis=0) + 1
    if float(dims[0]) * float(dims[1]) * float(dims[2]) >= 2.0**62:
        return None  # pragma: no cover - a vast extent against a tiny gap
    flat = (cell[:, 0] * dims[1] + cell[:, 1]) * dims[2] + cell[:, 2]
    _, first, inverse = np.unique(flat, return_index=True, return_inverse=True)
    return inverse, first


def connected_components_within(coords: np.ndarray, eps: float) -> np.ndarray:
    """Label every point 0..k-1 by which ``eps``-connected blob it belongs to.

    Runs on the given points only (call it per tree, not per plot). Points
    within ``eps`` of each other are in the same blob.

    The work is the number of point pairs within ``eps``, which grows with
    the cube of the gap: on a 120k-point tree that is 4M pairs at one point
    spacing but 113M at sixteen — 1.8GB for the pair array alone, and the
    cluster tool's gap slider goes that far. So past :data:`_VOXEL_ABOVE`
    points the graph is built over one representative per voxel of side
    ``eps / _VOXEL_DIVISOR`` instead, and each point takes its voxel's
    answer. Collapsing a voxel is sound in the direction that matters: its
    diagonal is 0.58 eps, so its points are within ``eps`` of one another and
    genuinely are one blob. What the reduction can miss is a link between two
    points in different voxels whose representatives are further apart than
    ``eps``. In exchange the cost stops following the gap, and the
    representative count falls as fast as the pair count would have climbed.

    The voxels are only taken when they actually shrink the problem — to
    :data:`_VOXEL_MIN_COLLAPSE` of it or less. That gate is what sets the
    divisor: with the voxels always on, a fine grid was needed to keep the
    tight end honest, and the tight end is where the grid does nothing
    anyway. Gated, the tight gaps fall through to the exact pass (which is
    *faster* there as well as exact — 150ms against 180ms at 1x spacing on a
    120k-point tree, because a grid that collapses 120,000 points to 118,674
    hands the KD-tree the same work after paying 25ms to build), and the
    grid is free to be coarser where it is the only thing keeping the gap
    affordable. Measured on that tree against the exact pass, per gap
    — time, and points landing in a different blob:

    ====  =========  ==========  ===========
    gap   exact      divisor 4   divisor 3
    ====  =========  ==========  ===========
    1x    0.150s     0.180s/255  0.179s/517
    2x    0.204s     0.220s/40   0.210s/154
    4x    0.449s     0.310s/10   0.237s/43
    8x    1.375s     0.366s/0    0.226s/0
    16x   5.334s     0.278s/0    0.120s/0
    ====  =========  ==========  ===========

    so the gate takes exact through 2x and the coarser grid roughly halves
    the loose end (0.278s to 0.120s at 16x). Over the whole ladder that is
    1.4x, on both that tree and a 300k-point one. The drift does not
    disappear, it moves: the tight gaps, where the default first click
    lives, become exact, and what the coarser grid costs lands in the middle
    of the ladder instead — worst measured 499 points of 300,000, under two
    hundredths of a percent. Never the other way, and not by luck: every
    edge in the reduced graph joins two real points within ``eps``, so a
    blob can come back split but never merged with its neighbour, and a
    split is what the next click along already undoes.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    n = len(coords)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if n > _VOXEL_ABOVE and eps > 0:
        grouped = _voxel_groups(coords, eps / _VOXEL_DIVISOR)
        if grouped is not None:
            inverse, first = grouped
            if len(first) <= _VOXEL_MIN_COLLAPSE * n:  # it earned its keep
                return _components_exact(coords[first], eps)[inverse]
    return _components_exact(coords, eps)


def connected_patch(
    coords: np.ndarray, labels: np.ndarray, seed: int, eps: float
) -> np.ndarray:
    """Indices of the ``eps``-connected blob of the seed's own tree that the
    seed sits in — i.e. one physically continuous lump of a single tree ID.
    """
    seed = int(seed)
    same = np.flatnonzero(labels == labels[seed])
    if same.size <= 1:
        return same
    comp = connected_components_within(coords[same], eps)
    local = int(np.searchsorted(same, seed))
    return same[comp == comp[local]]


def neighbours_by_points(
    cloud: PointCloud, tid: int, reach: float, rng=None
) -> set[int]:
    """IDs of trees whose points come within ``reach`` metres of tree
    ``tid``'s points. :func:`neighbour_distances` also says how close."""
    return set(neighbour_distances(cloud, tid, reach, rng))


def neighbour_distances(
    cloud: PointCloud, tid: int, reach: float, rng=None
) -> dict[int, float]:
    """``{tree id: how close it comes to tree ``tid``, in metres}`` for every
    tree within ``reach``.

    Bounding-box tests massively over-count neighbours in a closed canopy
    (a tall tree's box spans its whole crown), so boxes are only used as a
    prefilter; candidates are confirmed by sampled point-to-point distance.
    That distance is what the panel orders its buttons by, so the nearest
    tree — the one whose crown the selection probably belongs to — is the
    first button, and the first number key.
    """
    from scipy.spatial import cKDTree

    rng = rng or np.random.default_rng(0)
    labels, coords = cloud.labels, cloud.coords
    mine = np.flatnonzero(labels == tid)
    if mine.size == 0:
        return {}
    lo = coords[mine].min(axis=0) - reach
    hi = coords[mine].max(axis=0) + reach
    # density.in_box rather than the chained expression this used to spell
    # out: same per-axis test, but into two reused buffers instead of eleven
    # (N,) temporaries, which matters because this runs over the whole
    # catalogue every time a tree is opened.
    from .density import in_box as _in_box

    in_box = _in_box(coords, lo, hi)
    in_box[mine] = False
    # Everything below works off the in-box points only, so the per-candidate
    # loop no longer rescans all N labels once per candidate.
    box_idx = np.flatnonzero(in_box)
    box_labels = labels[box_idx]
    cand = np.unique(box_labels)
    cand = cand[(cand != UNASSIGNED) & (cand != NOISE) & (cand != tid)]
    if not cand.size:
        return {}
    kd = cKDTree(coords[_sample(rng, mine)])
    out: dict[int, float] = {}
    for t in cand:
        theirs = box_idx[box_labels == t]
        d, _ = kd.query(
            coords[_sample(rng, theirs)], k=1, distance_upper_bound=reach
        )
        if np.isfinite(d).any():
            out[int(t)] = float(np.min(d[np.isfinite(d)]))
    return out


# -- stem measurements -------------------------------------------------------
#: Where a diameter is measured, and how thick a slab is taken around it:
#: breast height, 1.3 m, with ±15 cm of stem either side of it. Thicker
#: averages a taper and a lean into the circle; thinner is too few points to
#: fit on a cloud sampled every couple of centimetres.
BREAST_HEIGHT = 1.3
_SLAB = 0.15

#: A fit outside these is not a tree stem — a branch, a patch of ground, or
#: a circle through noise. Diameters in metres.
_MIN_DBH, _MAX_DBH = 0.02, 3.0

#: Relative residual at or below which the diameter is worth believing.
#: Measured on synthetic stems: a clean trunk fits at 0.03, one with 2 cm of
#: scanner noise at 0.12, a leaning one at 0.11, and half a trunk at 0.06 —
#: while a third of an arc, which returns a third of the true diameter, sits
#: at 0.27. Callers that *score* on DBH should ignore anything above this
#: (see segfix.inventory); callers that only display it needn't.
GOOD_FIT = 0.2

#: Past this the points were not a circle at all, and no number comes back.
_MAX_RESIDUAL = 0.45


def stem_diameter(
    coords: np.ndarray,
    base_z: float | None = None,
    height: float = BREAST_HEIGHT,
    slab: float = _SLAB,
) -> tuple[float, float] | None:
    """Estimate a stem's diameter at breast height from its points.

    Returns ``(diameter, residual)`` in metres, where ``residual`` is the
    RMS distance of the slab's points from the fitted circle divided by its
    radius — a unitless "how round was it really", which is what tells a
    clean stem (a few per cent) from a circle fitted through a fork, a lean,
    or one side of a trunk the scanner only saw from one angle. None when
    there is nothing fittable: too few points, or a diameter outside
    :data:`_MIN_DBH`..:data:`_MAX_DBH`.

    The fit is Kåsa's: the algebraic least-squares circle, which is one
    linear solve rather than an iteration, and is biased only when the
    points cover a short arc — exactly the case the residual then reports.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] < 8:
        return None
    if base_z is None:
        # The stem's own foot: the 1st percentile rather than the minimum, so
        # one stray point under the tree doesn't drop breast height into the
        # ground.
        base_z = float(np.percentile(coords[:, 2], 1.0))
    z = coords[:, 2]
    in_slab = (z >= base_z + height - slab) & (z <= base_z + height + slab)
    points = coords[in_slab][:, :2]
    if points.shape[0] < 8:
        return None

    centre, radius = _fit_circle(points)
    if radius is None:
        return None
    diameter = 2.0 * radius
    if not (_MIN_DBH <= diameter <= _MAX_DBH):
        return None
    residual = np.hypot(*(points - centre).T) - radius
    quality = float(np.sqrt(np.mean(residual ** 2)) / radius)
    return (diameter, quality) if quality <= _MAX_RESIDUAL else None


def _fit_circle(points: np.ndarray):
    """Kåsa circle fit: ``(centre, radius)``, or ``(None, None)``."""
    centred = points - points.mean(axis=0)  # conditioning, not a shortcut
    x, y = centred[:, 0], centred[:, 1]
    design = np.column_stack([x, y, np.ones(len(x))])
    try:
        solution, *_ = np.linalg.lstsq(design, x ** 2 + y ** 2, rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate input
        return None, None
    cx, cy = solution[0] / 2.0, solution[1] / 2.0
    squared = solution[2] + cx ** 2 + cy ** 2
    if not np.isfinite(squared) or squared <= 0:
        return None, None
    return np.array([cx, cy]) + points.mean(axis=0), float(np.sqrt(squared))
