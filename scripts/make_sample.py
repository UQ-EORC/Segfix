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

"""Generate a synthetic forest plot with deliberate segmentation errors.

A dozen cone-ish "trees" of clearly different sizes -- from a 5.5 m sapling
to a 14 m giant -- standing alone and in touching groups, with the five
mistakes a segmentation pipeline really makes, one of each to practise on:

* **wrong side of a boundary** -- where trees 1 and 2 touch, the part of
  tree 2's crown facing tree 1 was given tree 1's ID;
* **over-segmented** -- one tree split into two IDs (7 = trunk and lower
  crown, 8 = the top), next to a small neighbour (9);
* **under-segmented** -- two trees sharing one ID (10);
* **leaked into the ground** -- a tree (11) whose ID also swallowed a patch
  of ground beside it, for Unassign (U);
* **false positive** -- a bush detected as a tree (12), for Noise (X);

plus the two non-tree labels real scans always carry: unassigned ground,
and a handful of noise points floating with nothing near them. Trees 1-3
stand close enough that loading any one brings the other two, which is
where lasso and "lasso tree" (current tree only) visibly differ -- and
where the boundary error above gets fixed.

Usage:  python scripts/make_sample.py sample.ply
        python scripts/make_sample.py --format las sample.las
        python scripts/make_sample.py --spacing 0.01 dense.las
        python scripts/make_sample.py --spacing 0.01 --origin 204300 7223250 12 \
            dense_utm.las

The ``.las`` form mimics arbor's output: XYZ plus a ``treeID`` Extra-Bytes
column, so the LAS editing path (:class:`segfix.treecatalog.LasCatalog`) has
something to open without running the arbor pipeline.

``--spacing`` switches to a scan-like cloud: instead of a few thousand points
filling each tree's volume, points are laid on the surfaces a scanner
actually sees (trunk, crown shell, ground) at the requested pitch. That is
what makes a cloud *dense* — a metre of trunk is one ring of points, but at
5mm there are two hundred of them — and it is how to get a file that trips
segfix's "points closer than 2cm" check and its downsample offer (see
:mod:`segfix.density`). The same deliberate segmentation errors are in it,
so the whole fix-and-save round trip can be exercised at full density.

``--origin`` puts the plot somewhere georeferenced (a UTM easting/northing,
say) instead of at the origin, which is what a real survey looks like and
what trips segfix's *other* load-time question, the global shift (see
:mod:`segfix.shift_ui`). Combined with ``--spacing`` it produces a cloud that
asks both questions in one open, which is the only way to exercise them
together.
"""

import argparse
import os

import numpy as np

from segfix import PointCloud
from segfix import io
from segfix.model import NOISE, UNASSIGNED

# -- the plot -------------------------------------------------------------------
# (tree ID, trunk position (x, y), height m, crown radius m). IDs are fixed so
# a walkthrough can refer to them.
_CLEAN = [
    # A touching group of three very different sizes.
    (1, (0.0, 0.0), 13.0, 3.0),
    (2, (4.6, 0.8), 7.0, 1.8),
    (3, (1.8, 4.6), 9.5, 2.2),
    # Stand-alone trees, large to small.
    (4, (18.0, 13.0), 14.0, 2.8),
    (5, (26.0, 12.0), 5.5, 1.3),
    (6, (32.0, 8.5), 9.0, 2.0),
    # The small neighbour of the over-segmented tree below.
    (9, (17.6, 3.6), 6.0, 1.5),
]
# Wrong side of a boundary: (tree that took the points, tree they belong to).
_BOUNDARY = (1, 2)
_OVER_SEGMENTED = ((7, 8), (14.0, 2.0), 11.0, 2.5, 0.55)  # ids, xy, h, r, split
_UNDER_SEGMENTED = (10, [((26.5, 1.0), 10.0, 2.0), ((29.3, 1.6), 8.0, 1.7)])
_LEAKY = (11, (6.0, 14.0), 8.5, 2.2)
# The ground patch tree 11 swallowed: beside its trunk, clear of it.
_LEAK_PATCH = ((6.8, 8.6), (13.0, 15.4))  # (x range, y range)
_BUSH = (12, (9.2, 12.4), 1.1, 0.9)       # id, xy, height, radius
_BOUNDS = ((-4.0, 35.0), (-4.0, 26.0))

# The rest of the stand: trees with nothing wrong with them, filling the plot
# so it reads as a piece of forest rather than a dozen specimens on a lawn.
# They get IDs from 13 up, after the ones the errors above are pinned to.
_STAND_COUNT = 22
#: How much clear air a new tree's crown must leave around every tree already
#: placed. Enough that it never joins one of their neighbour sets: the
#: walkthrough loads tree 1 and talks about the two trees that touch it, and a
#: third arriving unannounced would make a liar of it. New trees may still
#: crowd each other, which is the point.
_STAND_CLEARANCE = 1.6
_FIRST_STAND_ID = 13


def stand(existing, bounds, count=_STAND_COUNT, rng=None):
    """Extra clean trees on a jittered grid, skipping anywhere too close to
    a tree already placed.

    ``existing`` is ``(xy, crown radius)`` per tree already in the plot.
    Returns ``(id, xy, height, radius)`` rows like :data:`_CLEAN`.
    """
    rng = rng or np.random.default_rng()
    (xlo, xhi), (ylo, yhi) = bounds
    pinned = list(existing)
    placed = []
    step = 2.5
    spots = [
        (x + float(rng.normal(0, 0.7)), y + float(rng.normal(0, 0.7)))
        for x in np.arange(xlo + 2.0, xhi - 1.0, step)
        for y in np.arange(ylo + 2.0, yhi - 1.0, step)
    ]
    rng.shuffle(spots)
    for x, y in spots:
        if len(placed) == count:
            break
        height = float(rng.uniform(6.0, 15.0))
        radius = float(height * rng.uniform(0.16, 0.26))
        # Clear of the trees the walkthrough talks about, but free to crowd
        # the rest of the stand: neighbouring crowns overlapping is what a
        # plot looks like, and what the neighbour buttons are for.
        if any(np.hypot(x - ox, y - oy) < radius + orad + _STAND_CLEARANCE
               for (ox, oy), orad in pinned):
            continue
        if any(np.hypot(x - ox, y - oy) < max(radius, orad)
               for (ox, oy), orad in ((xy, r) for _i, xy, _h, r in placed)):
            continue  # stems too close to tell apart
        placed.append((_FIRST_STAND_ID + len(placed), (x, y), height, radius))
    return placed


# -- volumetric (default) shapes ---------------------------------------------------
def tree(center, height=8.0, radius=2.0, n=None, rng=None):
    rng = rng or np.random.default_rng()
    if n is None:  # scale with crown volume, so big trees look it
        n = int(np.clip(2000 * height * radius * radius / 32.0, 600, 7000))
    z = rng.random(n) ** 0.5 * height          # denser near the base
    r = (1 - z / height) * radius * rng.random(n) ** 0.5
    theta = rng.random(n) * 2 * np.pi
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack([x, y, center[2] + z]).astype(np.float32)


def bush(center, height=1.1, radius=0.9, n=500, rng=None):
    """A low rounded shrub -- exactly the kind of blob a segmentation
    pipeline happily calls a tree."""
    rng = rng or np.random.default_rng()
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    s = rng.random(n) ** (1 / 3)
    return np.column_stack([
        center[0] + d[:, 0] * radius * s,
        center[1] + d[:, 1] * radius * s,
        center[2] + np.abs(d[:, 2]) * height * s,
    ]).astype(np.float32)


def ground(bounds, n=5000, z=0.0, jitter=0.15, rng=None):
    """A scatter of unassigned points across the plot's footprint, at
    roughly ground level — the terrain/understory real scans are full of,
    which the review workflow lassoes and assigns into a tree with A."""
    rng = rng or np.random.default_rng()
    (xlo, xhi), (ylo, yhi) = bounds
    x = rng.uniform(xlo, xhi, n)
    y = rng.uniform(ylo, yhi, n)
    z = z + rng.normal(0, jitter, n)
    return np.column_stack([x, y, z]).astype(np.float32)


def floaters(bounds, n=15, z_range=(3.0, 12.0), rng=None):
    """A few stray points scattered mid-air with no tree nearby — sensor
    noise, to practise marking Noise (X) instead of assigning to a tree."""
    rng = rng or np.random.default_rng()
    (xlo, xhi), (ylo, yhi) = bounds
    x = rng.uniform(xlo, xhi, n)
    y = rng.uniform(ylo, yhi, n)
    z = rng.uniform(*z_range, n)
    return np.column_stack([x, y, z]).astype(np.float32)


# -- scan-like (--spacing) shapes -------------------------------------------------
def _jittered_grid(u_size, v_size, spacing, rng):
    """``(u, v)`` samples covering a ``u_size`` x ``v_size`` patch at roughly
    ``spacing`` pitch.

    A jittered grid rather than uniform-random points: random points on a
    surface clump, and their median nearest-neighbour distance comes out
    around half the pitch you asked for, which would make "generate a 1cm
    cloud" quietly produce a 5mm one. Jittering a grid by a fraction of a
    cell keeps the measured spacing close to the requested one while still
    looking scanned rather than plotted.
    """
    nu = max(1, int(round(u_size / spacing)))
    nv = max(1, int(round(v_size / spacing)))
    u, v = np.meshgrid(
        (np.arange(nu) + 0.5) * (u_size / nu),
        (np.arange(nv) + 0.5) * (v_size / nv),
        indexing="ij",
    )
    u = u.ravel() + rng.uniform(-0.2, 0.2, u.size) * spacing
    v = v.ravel() + rng.uniform(-0.2, 0.2, v.size) * spacing
    return u, v


def dense_tree(center, height=8.0, radius=2.0, spacing=0.01, rng=None):
    """A tree as a scanner sees it: a trunk cylinder and a conical crown
    shell, both sampled at ``spacing``.

    Surfaces, not a filled volume — that is the difference between a cloud
    that is merely large and one that is genuinely dense. Filling the same
    cone's 34 cubic metres at 5mm would take 270 million points; its
    surfaces take two million.
    """
    rng = rng or np.random.default_rng()
    trunk_r = max(0.04, radius * 0.07)
    crown_base = 0.3 * height

    # Trunk: unroll the cylinder to a (circumference x height) patch.
    u, z = _jittered_grid(2 * np.pi * trunk_r, height, spacing, rng)
    theta = u / trunk_r
    r = trunk_r + rng.normal(0, spacing * 0.3, theta.size)  # bark roughness
    trunk = np.column_stack([
        center[0] + r * np.cos(theta),
        center[1] + r * np.sin(theta),
        center[2] + z,
    ])

    # Crown: the cone's lateral surface, unrolled the same way. Its radius
    # tapers to nothing at the top, so the grid is laid out in slant length
    # against the *widest* circumference and then thinned per ring - which
    # is also what keeps the crown from turning into a solid cap up top.
    crown_h = height - crown_base
    slant = float(np.hypot(radius, crown_h))
    u, s = _jittered_grid(2 * np.pi * radius, slant, spacing, rng)
    frac = np.clip(s / slant, 0, 1)          # 0 at the base, 1 at the tip
    ring_r = radius * (1 - frac)
    # Keep each ring's points at the requested pitch: a ring of radius
    # ring_r only has room for a fraction of the widest ring's points.
    keep = rng.random(u.size) < np.maximum(ring_r / radius, 1e-6)
    theta = u[keep] / radius
    ring_r, frac = ring_r[keep], frac[keep]
    jitter = rng.normal(0, spacing * 0.5, theta.size)
    crown = np.column_stack([
        center[0] + (ring_r + jitter) * np.cos(theta),
        center[1] + (ring_r + jitter) * np.sin(theta),
        center[2] + crown_base + frac * crown_h,
    ])
    return np.vstack([trunk, crown]).astype(np.float32)


def dense_bush(center, height=1.1, radius=0.9, spacing=0.01, rng=None):
    """A shrub's outer surface at ``spacing``: a dome, as a scanner sees it.

    A Fibonacci lattice on the hemisphere rather than random points, for the
    same reason as :func:`_jittered_grid` -- even spacing, so the measured
    pitch matches the requested one.
    """
    rng = rng or np.random.default_rng()
    area = 2 * np.pi * radius * max(radius, height)  # roughly, for a dome
    n = max(200, int(area / spacing ** 2))
    i = np.arange(n) + 0.5
    zu = 1.0 - i / n                       # 1 at the crown, 0 at the rim
    ru = np.sqrt(1.0 - zu * zu)
    theta = i * np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    rough = 1.0 + rng.normal(0, 0.03, n)      # leafy, not a smooth dome
    return np.column_stack([
        center[0] + ru * radius * rough * np.cos(theta),
        center[1] + ru * radius * rough * np.sin(theta),
        center[2] + zu * height * rough,
    ]).astype(np.float32)


def dense_ground(centers, spacing=0.01, reach=3.0, rng=None):
    """Ground under the trees, at ``spacing``: one grid over the whole plot,
    keeping only what falls within ``reach`` of a trunk.

    Near the trees rather than across the whole rectangle, because that is
    all the review workflow ever lassoes — and at these pitches the empty
    gaps between trees would otherwise be most of the file. One shared grid
    rather than a disc each, so overlapping discs can't quietly double the
    density where two trees stand close together.
    """
    rng = rng or np.random.default_rng()
    centers = np.asarray(centers, dtype=np.float64)
    lo = centers[:, :2].min(axis=0) - reach
    hi = centers[:, :2].max(axis=0) + reach
    u, v = _jittered_grid(hi[0] - lo[0], hi[1] - lo[1], spacing, rng)
    x, y = lo[0] + u, lo[1] + v

    near = np.zeros(x.size, dtype=bool)
    for cx, cy, _cz in centers:
        near |= (x - cx) ** 2 + (y - cy) ** 2 <= reach * reach
    x, y = x[near], y[near]
    z = centers[:, 2].mean() + rng.normal(0, spacing, x.size)  # slight relief
    return np.column_stack([x, y, z]).astype(np.float32)


def facing_band(pts, xy, height, toward, half_angle=50.0, band=(0.5, 0.8)):
    """A tree's crown points on the side facing ``toward``, in a height band
    -- where a touching neighbour's segmentation typically bleeds in.

    The band sits high enough in the crown that it stays clear of the
    neighbour's own points: a real mislabelling to fix, not a tangle.
    """
    v = pts[:, :2] - np.asarray(xy, dtype=np.float64)
    d = np.asarray(toward, dtype=np.float64) - np.asarray(xy, dtype=np.float64)
    d /= np.linalg.norm(d)
    r = np.linalg.norm(v, axis=1)
    facing = (v @ d) > np.cos(np.radians(half_angle)) * np.maximum(r, 1e-9)
    z = pts[:, 2]
    in_band = (z > band[0] * height) & (z < band[1] * height)
    return facing & in_band & (r > 0.3)  # crown only, not the stem


def _write_las(coords: np.ndarray, labels: np.ndarray, out: str) -> None:
    """Write an arbor-shaped LAS/LAZ: XYZ + an int ``treeID`` Extra-Bytes
    column (``0`` = unassigned, as arbor writes it)."""
    import laspy

    header = laspy.LasHeader(version="1.4", point_format=6)
    header.offsets = np.floor(coords.min(axis=0))
    header.scales = [0.001, 0.001, 0.001]
    header.add_extra_dim(
        laspy.ExtraBytesParams(name="treeID", type=np.int32,
                               description="Unique ID per tree")
    )
    las = laspy.LasData(header)
    las.x, las.y, las.z = coords[:, 0], coords[:, 1], coords[:, 2]
    tid = np.where(labels == NOISE, UNASSIGNED, labels)  # LAS has no noise id
    las.treeID = tid.astype(np.int32)
    las.write(out)


def main(out, spacing=None, origin=None):
    rng = np.random.default_rng(42)
    parts, labels = [], []

    def add(pts, lab):
        parts.append(pts)
        labels.append(np.full(len(pts), lab, dtype=np.int32))

    def make(xy, height, radius):
        center = (xy[0], xy[1], 0.0)
        if spacing is None:
            return tree(center, height=height, radius=radius, rng=rng)
        return dense_tree(center, height=height, radius=radius, spacing=spacing, rng=rng)

    trunks = []  # every stem, for where the ground goes

    # The error cases are placed first and the rest of the stand fills in
    # around them, so their neighbours stay exactly as the walkthrough
    # describes however many extra trees there are.
    pinned = [(xy, r) for _t, xy, _h, r in _CLEAN]
    pinned += [(_OVER_SEGMENTED[1], _OVER_SEGMENTED[3])]
    pinned += [(xy, r) for xy, _h, r in _UNDER_SEGMENTED[1]]
    pinned += [(_LEAKY[1], _LEAKY[3]), (_BUSH[1], _BUSH[3])]
    extra = stand(pinned, _BOUNDS, rng=rng)

    taker, owner = _BOUNDARY
    taker_xy = next(xy for tid, xy, _h, _r in _CLEAN if tid == taker)
    for tid, xy, h, r in _CLEAN:
        pts = make(xy, h, r)
        if tid == owner:
            # Wrong side of a boundary: the neighbour took this band.
            taken = facing_band(pts, xy, h, toward=taker_xy)
            add(pts[~taken], tid)
            add(pts[taken], taker)
        else:
            add(pts, tid)
        trunks.append(xy)

    for tid, xy, h, r in extra:
        add(make(xy, h, r), tid)
        trunks.append(xy)

    # Over-segmented: one physical tree split by height into two IDs.
    (low_id, high_id), xy, h, r, split = _OVER_SEGMENTED
    t = make(xy, h, r)
    add(t[t[:, 2] < split * h], low_id)
    add(t[t[:, 2] >= split * h], high_id)
    trunks.append(xy)

    # Under-segmented: two physical trees, one ID.
    tid, members = _UNDER_SEGMENTED
    for xy, h, r in members:
        add(make(xy, h, r), tid)
        trunks.append(xy)

    # Leaked into the ground: a tree, plus (below) a patch of ground with its ID.
    leak_id, xy, h, r = _LEAKY
    add(make(xy, h, r), leak_id)
    trunks.append(xy)

    # False positive: a bush that got its own tree ID.
    bush_id, xy, bh, br = _BUSH
    center = (xy[0], xy[1], 0.0)
    add(bush(center, bh, br, rng=rng) if spacing is None
        else dense_bush(center, bh, br, spacing=spacing, rng=rng), bush_id)
    trunks.append(xy)

    # Ground, with the leaked patch carrying tree 11's ID, and stray noise.
    if spacing is None:
        g = ground(_BOUNDS, rng=rng)
    else:
        g = dense_ground([(x, y, 0.0) for x, y in trunks], spacing=spacing, rng=rng)
    (px0, px1), (py0, py1) = _LEAK_PATCH
    leaked = (g[:, 0] >= px0) & (g[:, 0] <= px1) & (g[:, 1] >= py0) & (g[:, 1] <= py1)
    add(g[~leaked], UNASSIGNED)
    add(g[leaked], leak_id)
    add(floaters(_BOUNDS, rng=rng), NOISE)

    # float64 from here on: adding a UTM-sized origin to float32 coordinates
    # would round away the millimetres this just went to the trouble of
    # generating, before laspy ever sees them. (LAS itself is safe -- it
    # stores xyz as scaled offsets from a header origin, not as floats.)
    coords = np.vstack(parts).astype(np.float64)
    if origin is not None:
        coords += np.asarray(origin, dtype=np.float64)
    labels = np.concatenate(labels)

    if os.path.splitext(out)[1].lower() in (".las", ".laz"):
        _write_las(coords, labels, out)
        n_points = len(coords)
        tree_ids = sorted(set(labels.tolist()) - {UNASSIGNED, NOISE})
    else:
        cloud = PointCloud(coords=coords, labels=labels)
        io.save(cloud, out)
        n_points = cloud.n_points
        tree_ids = sorted(cloud.tree_ids.tolist())

    n_unassigned = int(np.sum(labels == UNASSIGNED))
    n_noise = int(np.sum(labels == NOISE))
    print(
        f"Wrote {n_points:,} points, trees {tree_ids} "
        f"+ {n_unassigned:,} unassigned + {n_noise:,} noise → {out} "
        f"({os.path.getsize(out) / 1e6:,.0f} MB)"
    )
    if origin is not None:
        from segfix.treecatalog import needs_global_shift

        mins, maxs = coords.min(axis=0), coords.max(axis=0)
        verdict = (
            "opening this also offers a global shift"
            if needs_global_shift(mins, maxs) else
            "not large enough for segfix to offer a global shift"
        )
        print(
            f"Plot placed at ({mins[0]:,.0f}, {mins[1]:,.0f}, {mins[2]:,.0f}) "
            f"- {verdict}"
        )
    if spacing is not None:
        # Measured with the same estimator segfix runs on load, so the
        # number printed here is the one the app will report back.
        from segfix.density import DENSE_SPACING, estimate_spacing

        measured = estimate_spacing(coords)
        verdict = (
            "below segfix's 2cm threshold: opening this offers to downsample"
            if measured < DENSE_SPACING else
            "above segfix's 2cm threshold: no downsample will be offered"
        )
        print(f"Measured point spacing {measured * 100:.2f} cm - {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", help="output path (extension picks format)")
    ap.add_argument("--format", choices=["ply", "las"], default="ply",
                    help="format when OUT is omitted (default: ply)")
    ap.add_argument(
        "--spacing", type=float, default=None, metavar="METRES",
        help="generate a scan-like cloud with points this far apart on the "
             "trunk, crown and ground surfaces, instead of a few thousand "
             "points per tree. Anything under 0.02 trips segfix's "
             "dense-cloud check; the script prints the point count and file "
             "size it wrote.",
    )
    ap.add_argument(
        "--origin", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="place the plot here instead of at (0, 0, 0), e.g. --origin "
             "204300 7223250 12 for a UTM-referenced cloud that also trips "
             "segfix's large-coordinate (global shift) prompt. LAS only past "
             "10 km from the origin: a PLY sample stores xyz as float32, "
             "which at those magnitudes quantises the cloud to metres.",
    )
    args = ap.parse_args()
    if args.spacing is not None and args.spacing <= 0:
        ap.error("--spacing must be positive")
    default_name = "dense" if args.spacing is not None else "sample"
    out = args.out or f"{default_name}.{args.format}"
    if args.origin is not None and os.path.splitext(out)[1].lower() == ".ply":
        from segfix.treecatalog import GLOBAL_SHIFT_THRESHOLD

        if max(abs(v) for v in args.origin) > GLOBAL_SHIFT_THRESHOLD:
            ap.error(
                f"--origin {args.origin} needs a .las output: this script's "
                "PLY writer stores xyz as float32, which past "
                f"{GLOBAL_SHIFT_THRESHOLD:,.0f} m keeps under a metre of "
                "precision - the cloud would arrive already ruined, which is "
                "not a useful test of the prompt that exists to prevent that."
            )
    main(out, spacing=args.spacing, origin=args.origin)
