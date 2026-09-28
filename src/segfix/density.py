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

"""Point-density estimation and voxel decimation for very dense clouds.

A terrestrial or drone LiDAR plot can carry a point every few millimetres.
Nothing in segfix *needs* that detail to fix a segmentation — a tree's
extent is decided at the decimetre scale — but every millimetre of it costs
memory, GPU upload and lasso time, so a sub-2cm cloud makes the review
workflow crawl for no reviewing benefit.

This module supplies the three pieces of the answer:

* :func:`estimate_spacing` — the cloud's typical point pitch, measured
  without ever building a KD-tree over the whole thing (see its docstring;
  that's the part that has to work on a 200-million-point file).
* :func:`voxel_indices` — a "keep one real point per voxel" decimation,
  returning *indices* rather than new points, so labels, attributes and the
  file rows they came from all stay addressable.
* :func:`class_trees` / :func:`expand_labels` — the way back: which
  full-resolution points an edit on the decimated cloud stands for.

Editing then happens on the decimated cloud, and
:meth:`segfix.treecatalog._BaseCatalog.save` interpolates the edited labels
back onto every full-resolution point before writing, so the file keeps all
of its points. The prompt itself is :mod:`segfix.density_ui`.
"""

from __future__ import annotations

import numpy as np

#: Below this typical point spacing (metres) a cloud is dense enough that
#: reviewing it at full resolution is mostly wasted work — 2cm still
#: resolves a small branch, which is well past what re-labelling a stem
#: needs.
DENSE_SPACING = 0.02

#: Voxel size offered for a cloud that is denser than that. Deliberately
#: coarser than the threshold: a 1.7cm cloud voxelised at 2cm keeps 92% of
#: its points, which is all of the cost of the pass and none of the relief,
#: while 3cm is still finer than any detail re-labelling a stem needs. The
#: prompt's spin box takes anything else.
DEFAULT_VOXEL = 0.03

# How many points a spacing measurement aims to look at, and how many query
# points it takes nearest-neighbour distances from. Both are "enough for a
# median", not tuning knobs.
_BLOCK_TARGET = 40_000
_QUERY_CAP = 20_000
_BLOCK_SEEDS = 3


def _median_nn(points: np.ndarray, rng, cap: int = _QUERY_CAP) -> float:
    """Median distance from a sample of ``points`` to their nearest other
    point in the *whole* set.

    Sampling the query points (not the tree's points) is what keeps this
    unbiased: thinning the set itself would push every nearest neighbour
    further away and overstate the spacing.
    """
    from scipy.spatial import cKDTree

    pts = np.ascontiguousarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    queries = pts if len(pts) <= cap else pts[rng.choice(len(pts), cap, replace=False)]
    d, _ = cKDTree(pts).query(queries, k=2, workers=-1)
    return float(np.median(d[:, 1]))


def in_box(coords: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Boolean mask of points inside an axis-aligned box, tested axis by axis
    so a huge cloud never materialises two full ``(N, 3)`` temporaries.

    The six comparisons write into two reused buffers rather than leaving
    eleven ``(N,)`` temporaries behind: :func:`estimate_spacing` calls this up
    to twenty-four times while it hunts for a block size, and a large
    allocate/free round trip costs roughly fifty times more on Windows (fresh
    pages committed through VirtualAlloc, kernel-zeroed) than on Linux, where
    glibc hands the same warm arena back.
    """
    n = len(coords)
    mask = np.ones(n, dtype=bool)
    tmp = np.empty(n, dtype=bool)
    for axis in range(3):
        column = coords[:, axis]
        np.greater_equal(column, lo[axis], out=tmp)
        mask &= tmp
        np.less_equal(column, hi[axis], out=tmp)
        mask &= tmp
    return mask


def estimate_spacing(coords: np.ndarray, rng=None) -> float:
    """The cloud's typical nearest-neighbour distance, in metres.

    :func:`segfix.analysis.point_spacing` answers the same question but
    builds a KD-tree over every point, which is exactly what can't be done
    on the clouds this check exists for (a 200M-point tree is tens of
    gigabytes). Instead this measures inside a few small boxes: local
    neighbourhoods are complete within a box, so the nearest-neighbour
    distances measured there are the real ones, and the median over several
    randomly placed boxes doesn't care that most of the cloud was never
    looked at.

    Returns ``0.0`` for a cloud too small or too sparse to measure.
    """
    coords = np.asarray(coords)
    n = len(coords)
    if n < 2:
        return 0.0
    rng = rng or np.random.default_rng(0)
    if n <= _BLOCK_TARGET:
        return _median_nn(coords, rng)

    return spacing_from_blocks(sample_blocks(coords, rng), rng)


def spacing_from_blocks(blocks, rng=None) -> float:
    """:func:`estimate_spacing` for blocks already sampled — so an open that
    also wants :func:`estimate_kept_fraction` pays for the sampling once.
    Scanning for the blocks is a pass over every point per attempt, which on
    a 480M-point cloud is twenty seconds, not a rounding error."""
    rng = rng or np.random.default_rng(0)
    measured = [_median_nn(block, rng) for block in blocks if len(block) >= 50]
    if not measured:
        return 0.0
    return float(np.median(measured))


def sample_blocks(coords: np.ndarray, rng=None) -> list[np.ndarray]:
    """A few small, *complete* boxes of the cloud, each holding roughly
    :data:`_BLOCK_TARGET` points.

    Complete is the point: everything inside a box is there, so a
    neighbourhood measured in one is the real neighbourhood, and a voxel
    grid laid over one thins exactly as it would over the whole cloud.
    That is what lets both the spacing measurement and
    :func:`estimate_kept_fraction` answer a question about a 480M-point
    cloud by looking at 40k points.
    """
    coords = np.asarray(coords)
    n = len(coords)
    rng = rng or np.random.default_rng(0)
    if n < 2:
        return []
    lo_all, hi_all = coords.min(axis=0), coords.max(axis=0)
    extent = float(np.max(hi_all - lo_all))
    if extent <= 0:
        return []
    # First guess at a box side holding ~_BLOCK_TARGET points, assuming the
    # points were spread evenly through the bounding box. They never are, so
    # the loop below corrects it.
    half = 0.5 * extent * max((_BLOCK_TARGET / n) ** (1 / 3), 1e-4)

    blocks: list[np.ndarray] = []
    for _ in range(_BLOCK_SEEDS):
        centre = coords[int(rng.integers(n))].astype(np.float64)
        block = np.empty((0, 3))
        for _ in range(8):
            mask = in_box(coords, centre - half, centre + half)
            count = int(mask.sum())
            if count > 8 * _BLOCK_TARGET:
                half *= 0.5
                continue
            if count < _BLOCK_TARGET // 8:
                half *= 2.0
                continue
            block = coords[mask]
            break
        else:
            block = coords[mask]
        blocks.append(block)
    return blocks


def estimate_kept_fraction(blocks, voxel: float) -> float | None:
    """What fraction of the cloud a ``voxel`` decimation would keep, judged
    from :func:`sample_blocks`, or ``None`` if the blocks can't say.

    Worth measuring rather than predicting: the answer depends on how the
    points are actually distributed, not on the median spacing. A cloud
    measuring 1.7cm keeps 92% of its points at a 2cm voxel and 85% at 3cm —
    most voxels already hold a single point — where an even spread would
    suggest a two- or threefold thinning.

    Only the block's interior counts, on each axis that has an interior. A
    voxel straddling a face of the box sees just the sliver inside it, so it
    keeps a representative for a handful of points where the real voxel
    holds many, which would overstate what the decimation keeps. An axis the
    cloud is barely thicker than a voxel along — a slab, a single-storey
    canopy — has no interior to retreat to, and is left alone: there every
    voxel is cut the same way, so the fraction is honest as it stands.
    """
    fractions = []
    for block in blocks:
        if len(block) < 50:
            continue
        lo, hi = block.min(axis=0), block.max(axis=0)
        roomy = (hi - lo) >= 3 * voxel
        lo = np.where(roomy, lo + voxel, lo)
        hi = np.where(roomy, hi - voxel, hi)
        inner = in_box(block, lo, hi)
        total = int(inner.sum())
        if total < 20:
            continue
        kept = int(inner[voxel_indices(block, voxel)].sum())
        fractions.append(kept / total)
    if not fractions:
        return None
    return float(np.median(fractions))


def suggest_voxel(spacing: float) -> float:
    """Voxel size to offer for a cloud measured at ``spacing``.

    :data:`DEFAULT_VOXEL` unless the cloud is coarser than that already, in
    which case the offer is its own spacing — never a voxel that would be
    finer than the points themselves, which would thin nothing.
    """
    return DEFAULT_VOXEL if spacing < DEFAULT_VOXEL else spacing


def _run_starts(ordered: np.ndarray) -> np.ndarray:
    """Where each run of equal values starts in a sorted array — the rows of
    one voxel are the block between two rows that differ.
    (``np.unique(..., return_index=True)`` would answer this by sorting all
    N keys yet again.)"""
    first = np.empty(len(ordered), dtype=bool)
    first[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=first[1:])
    return np.flatnonzero(first)


def voxel_indices(coords: np.ndarray, voxel: float) -> np.ndarray:
    """Indices of one representative point per occupied ``voxel``-sized cube.

    Returns *indices into* ``coords`` — real points, not voxel centroids —
    so the caller keeps every per-point label and attribute, and can still
    map each kept point back to the file row it came from.

    The point kept from each voxel is the one nearest its centre, not
    whichever came first in the file. That matters for what happens on save:
    a full-resolution point is given the label of the nearest kept point
    (:func:`nearest_label_expansion`), and centred representatives make
    "nearest kept point" line up with "own voxel". Keeping an arbitrary point
    instead leaves representatives bunched against voxel corners, where a
    neighbouring voxel's representative can be the closer one — which shows
    up as a stripe of points along an edited tree's edge quietly keeping
    their old label.
    """
    coords = np.asarray(coords)
    if voxel <= 0:
        raise ValueError(f"voxel size must be positive, got {voxel}")
    n = len(coords)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    origin = coords.min(axis=0)
    span = coords.max(axis=0) - origin
    dims = np.floor(span / voxel).astype(np.int64) + 1
    # Voxel key and centre-offset, one axis at a time into buffers allocated
    # once, and only ever four of them: this runs on the biggest array the
    # app ever touches (a 480M-point plot), where each (N,) float64 buffer is
    # 3.8GB, so a temporary that whole-array expressions would leave behind
    # is measured in gigabytes rather than being free. Large allocate/free
    # round trips also cost roughly fifty times more on Windows (fresh pages
    # committed through VirtualAlloc, kernel-zeroed) than on Linux, where
    # glibc hands the same warm arena back.
    #
    # The three axis keys are folded into one flat key as they are computed,
    # rather than kept as a (3, N) array and combined afterwards — same
    # arithmetic, a third of the memory.
    flat = np.zeros(n, dtype=np.int64)
    # Squared distance to the voxel centre, in voxel units — the tie-break
    # that decides which point of each voxel is kept.
    offset = np.zeros(n, dtype=np.float32)
    buf = np.empty(n, dtype=np.float64)
    cell = np.empty(n, dtype=np.float64)
    cell_i = np.empty(n, dtype=np.int64)
    flattenable = float(dims[0]) * float(dims[1]) * float(dims[2]) < 2.0**62
    keys = np.empty((3, n), dtype=np.int64) if not flattenable else None
    for axis in range(3):
        np.subtract(coords[:, axis], origin[axis], out=buf, dtype=np.float64)
        buf /= voxel
        np.floor(buf, out=cell)
        np.copyto(cell_i, cell, casting="unsafe")  # into the buffer, no temporary
        if keys is not None:  # pragma: no cover - needs a >10^6 m extent
            keys[axis] = cell_i
        else:
            if axis:
                flat *= dims[axis]
            flat += cell_i
        buf -= cell  # fractional position within the voxel, in [0, 1)
        buf -= 0.5
        buf *= buf
        offset += buf  # accumulated in float32: the tie-break needs no more
    del buf, cell, cell_i
    if keys is not None:  # pragma: no cover - needs a >10^6 m extent
        # Flattening three axes into one int64 is much faster than a
        # structured unique, but only while the product fits.
        flat = np.unique(keys.T, axis=0, return_inverse=True)[1].astype(np.int64)
        del keys

    # Group the points by voxel, then take the centre-most of each group.
    #
    # ``np.lexsort((offset, flat))`` says that in one line, but it is two
    # full sorts of N keys where one will do, and sorting is ~80% of this
    # function. So: one stable sort by voxel, and the tie-break folded into
    # an integer min-reduction over each run.
    order = np.argsort(flat, kind="stable")
    if n >= 2**32:  # pragma: no cover - 4 billion points is ~50GB of coords
        # No room to pack an index alongside the offset; fall back.
        best = np.lexsort((offset, flat))[_run_starts(flat[order])]
        return np.sort(best.astype(np.int64))
    ordered = flat[order]
    del flat  # the sorted copy is all the rest of this needs
    starts = _run_starts(ordered)
    del ordered

    # IEEE-754 bit patterns of non-negative floats compare as integers in the
    # same order as the floats, and ``offset`` is a sum of squares, so packing
    # it above the point's own index makes one int64 that orders by distance
    # to the voxel centre and then, for a tie, by position — which is exactly
    # the row lexsort's stable second key would have left first in the run.
    packed = offset[order].view(np.uint32).astype(np.int64)
    del offset
    packed <<= 32
    packed |= order
    best = np.minimum.reduceat(packed, starts)
    del packed
    best &= 0xFFFFFFFF  # unpack the index the winning offset carried
    return np.sort(best.astype(np.int64))


def class_trees(coords: np.ndarray, codes: np.ndarray, candidates: np.ndarray):
    """One KD-tree per original label, over the working points in
    ``candidates``.

    Grouping by label is what makes the expansion faithful. A plain nearest
    working point is not enough: a trunk point at ground level has ground
    points within a centimetre, so re-labelling the trunk would leave a ring
    of stragglers whose nearest working point was a piece of ground nobody
    touched. Asking instead for the nearest working point *that was the same
    tree* answers the question actually being asked — which edited point does
    this point belong with — and keeps the seam between two trees exactly
    where the decimated cloud put it.

    Returns ``{code: (tree, positions)}`` where ``positions`` indexes
    ``coords``.
    """
    from scipy.spatial import cKDTree

    out = {}
    if not len(candidates):
        return out
    sub_codes = codes[candidates]
    for code in np.unique(sub_codes):
        positions = candidates[sub_codes == code]
        pts = np.ascontiguousarray(coords[positions], dtype=np.float64)
        out[int(code)] = (cKDTree(pts), positions)
    return out


def whole_class_moves(codes: np.ndarray, before: np.ndarray,
                      after: np.ndarray, trees_only: bool = True) -> dict[int, int]:
    """Original label codes whose *every* working point changed to one and
    the same new label, as ``{code: new_label}``.

    That is an edit of a whole tree -- X or U on the current tree, a complete
    merge -- and it has to reach every one of that tree's full-resolution
    points. :func:`expand_labels` can't promise that on its own: a point in
    a voxel where a *different* class's point was the one kept (a bush's
    rim, sharing voxels with the ground) has no working point of its own
    tree nearby to be matched to, and would keep its old label.

    Only real trees qualify (original label above zero): moving every kept
    unassigned point is not a statement about unassigned points never seen.
    ``trees_only=False`` drops that rule, for point classes, where every
    value -- class 0 included -- is a real class.
    ``codes``, ``before`` and ``after`` are aligned per working point.
    """
    changed = before != after
    out: dict[int, int] = {}
    if not changed.any():
        return out
    for code in np.unique(codes[changed]):
        members = codes == code
        if not changed[members].all() or (trees_only and before[members].min() <= 0):
            continue
        new = np.unique(after[members])
        if new.size == 1:
            out[int(code)] = int(new[0])
    return out


def expand_labels(
    trees: dict,
    query_coords: np.ndarray,
    query_codes: np.ndarray,
    changed: np.ndarray,
    n_working: int,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Which of ``query_coords`` inherit an edited label, and from where.

    Each query point is matched to the nearest working point that carried the
    same original label (``trees``, from :func:`class_trees`), within
    ``max_distance``. Points whose match is not one of the ``changed``
    working positions are left out entirely, so a save only ever touches the
    neighbourhoods of points the user actually re-labelled.

    Returns ``(local_indices, working_positions)`` — where the pairs are in
    ``query_coords``, and which working point each takes its label from.
    """
    empty = np.empty(0, dtype=np.int64)
    if not len(query_coords) or not trees:
        return empty, empty

    is_changed = np.zeros(n_working, dtype=bool)
    is_changed[changed] = True

    local_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    for code, (tree, positions) in trees.items():
        rows = np.flatnonzero(query_codes == code)
        if not rows.size:
            continue
        dist, hit = tree.query(
            np.ascontiguousarray(query_coords[rows], dtype=np.float64),
            k=1, workers=-1, distance_upper_bound=max_distance,
        )
        # Out-of-range queries come back as len(positions) with inf distance.
        found = np.isfinite(dist)
        if not found.any():
            continue
        rows, hit = rows[found], hit[found]
        source = positions[hit]
        keep = is_changed[source]
        if not keep.any():
            continue
        local_parts.append(rows[keep])
        source_parts.append(source[keep])

    if not local_parts:
        return empty, empty
    return np.concatenate(local_parts), np.concatenate(source_parts)
