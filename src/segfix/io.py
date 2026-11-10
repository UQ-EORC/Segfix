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

"""Load and save segmented point clouds as binary PLY or uncompressed LAS.

Both formats store their points as fixed-size records at a known offset, which
is what lets :mod:`treecatalog` memory-map a plot and pull back one tree at a
time — the whole review workflow is built on that. ASCII PLY has no such
stride and is refused; compressed ``.laz`` (arbor's output) is decompressed to
``.las`` on import by :mod:`workspace`.

The instance label lives either in a per-point field whose name varies between
pipelines (``treeID``, ``PredInstance``, ``label`` ...), auto-detected from a
list of common candidates, or in the point's RGB colour (raycloudtools).

This module's :func:`load` / :func:`save` operate on a whole cloud at once;
the tree-at-a-time path the GUI actually uses is :mod:`treecatalog`.
"""

from __future__ import annotations

import os

import numpy as np

from .model import NOISE, UNASSIGNED, PointCloud

# PLY scalar type name -> numpy dtype string.
PLY_TYPE_MAP = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
    "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
    "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
    "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8",
}

# Field names that commonly hold a per-point tree/instance ID, best first.
LABEL_CANDIDATES = (
    "treeID",
    "tree_id",
    "TreeID",
    "PredInstance",
    "instance",
    "InstanceID",
    "segment_id",
    "label",
    "scalar_treeID",
    "scalar_label",
    "ID",
)


def sniff_format(path: str) -> str | None:
    """Peek the first bytes of ``path`` to tell a real PLY or LAS/LAZ from
    anything else — fast, a handful of bytes, never a scan.

    This guards the real header parsers below (:func:`_parse_ply_header`,
    ``laspy.open``): handed an arbitrary binary file with a ``.ply``
    extension, ``_parse_ply_header`` reads line by line looking for
    ``end_header`` — on non-text binary data that can mean minutes spent
    reading a multi-gigabyte file mostly-one-fake-"line"-at-a-time before it
    ever fails (readline() has nothing bounding how far it looks for a
    newline). A file picked with the wrong format should fail in
    microseconds, not freeze the app. Returns ``"ply"``, ``"las"`` (also
    matches ``.laz`` — LASzip keeps the plain LAS magic), or ``None``.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
    except OSError:
        return None
    if head[:3] == b"ply":
        return "ply"
    if head[:4] == b"LASF":
        return "las"
    return None


def _parse_ply_header(fh):
    """Parse a binary/ascii PLY header from an open binary file handle.

    Returns ``(fmt, vertex_props, vertex_count, data_offset)`` where
    ``vertex_props`` is a list of ``(name, ply_type)`` and ``data_offset`` is
    the byte offset where vertex data starts.  Handles the zero-padded vertex
    counts that raycloudtools writes.
    """
    fmt = "ascii"
    props: list[tuple[str, str]] = []
    count = 0
    current = None
    while True:
        line = fh.readline()
        if not line:
            break
        text = line.decode("ascii", "replace").strip()
        parts = text.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            current = parts[1]
            if current == "vertex":
                count = int(parts[2])  # int() copes with leading zeros
        elif parts[0] == "property" and current == "vertex":
            # skip list properties (faces); vertices here are all scalars
            if parts[1] != "list":
                props.append((parts[2], parts[1]))
        elif parts[0] == "end_header":
            break
    return fmt, props, count, fh.tell()


def _is_rgb_segmented(props, label_field) -> bool:
    """True when there is no label field but RGB colour is present."""
    names = [p[0] for p in props]
    if _pick_label_field(names, label_field) is not None:
        return False
    lowered = {n.lower() for n in names}
    return {"red", "green", "blue"}.issubset(lowered)


def load(path: str, label_field: str | None = None) -> PointCloud:
    """Read a binary PLY or LAS/LAZ. Raises ``ValueError`` for anything else."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".ply", ".las", ".laz"):
        raise ValueError(
            f"segfix reads binary PLY and LAS/LAZ, not {ext or path!r}"
        )
    # Magic-byte check before any header parsing: a file whose actual
    # content doesn't match its extension must fail fast here, not stall
    # inside _parse_ply_header's line-by-line scan of what turns out to be
    # arbitrary binary data (see sniff_format's docstring).
    kind = sniff_format(path)
    expected = "ply" if ext == ".ply" else "las"
    if kind != expected:
        raise ValueError(
            f"{path} doesn't look like a {expected.upper()} file (extension "
            f"says {ext}, but its header doesn't match) - wrong format?"
        )
    if ext in (".las", ".laz"):
        return _load_las(path, label_field)
    # Peek at the header: if the segmentation is encoded as RGB (no label
    # field but colour present), take the fast RGB-segmented path.
    with open(path, "rb") as fh:
        fmt, props, _count, _off = _parse_ply_header(fh)
    if "binary" not in fmt:
        raise ValueError(
            f"{path} is an ASCII PLY; segfix reads binary PLY only "
            "(its fixed-size records are what make tree-at-a-time loading work)"
        )
    if _is_rgb_segmented(props, label_field):
        return load_rgb_segmented(path)
    return _load_ply(path, label_field)


#: Points decoded at a time by :func:`labels_from_rgb`, and the size from
#: which its lookup table beats a sort. The table is 80MB whatever the
#: cloud, so below a million points the sort is both quicker and smaller.
_RGB_CHUNK = 4_000_000
_RGB_TABLE_ABOVE = 1_000_000


def _pack_rgb(red, green, blue) -> np.ndarray:
    """``red << 16 | green << 8 | blue`` as uint32, built in place so the
    only array the size of the input is the answer."""
    packed = np.asarray(red).astype(np.uint32)
    packed <<= 8
    packed |= np.asarray(green)
    packed <<= 8
    packed |= np.asarray(blue)
    return packed


def labels_from_rgb(red, green, blue):
    """``(labels int32, {label: (r, g, b)})`` for per-point colour columns.

    Each distinct colour is a tree, numbered from 1 in order of its packed
    ``r << 16 | g << 8 | b`` value; pure black is :data:`UNASSIGNED` and
    takes a number (if present) without getting a colour.

    ``np.unique(packed, return_inverse=True)`` says the same thing by
    sorting every point, which was 2.1s of a 5s open on a 39M-point plot.
    A colour is 24 bits, so there are only 16.7M possible values: marking
    which occur in a table that size and reading each point's rank back out
    of it is two linear passes (0.5s on that plot) and never holds more
    than a chunk's worth of packed colours, where the sort needed an int64
    permutation of the whole cloud. The columns are read a chunk at a time,
    so they can be memory-mapped fields.
    """
    n = len(red)
    if n < _RGB_TABLE_ABOVE:
        packed = _pack_rgb(red, green, blue)
        uniq, inverse = np.unique(packed, return_inverse=True)
        labels = (inverse + 1).astype(np.int32)
        labels[packed == 0] = UNASSIGNED  # black = unsegmented
    else:
        present = np.zeros(1 << 24, dtype=bool)
        for start in range(0, n, _RGB_CHUNK):
            stop = start + _RGB_CHUNK
            present[_pack_rgb(red[start:stop], green[start:stop],
                              blue[start:stop])] = True
        uniq = np.flatnonzero(present)
        # rank[colour] is the colour's position among those present, plus
        # one: exactly `inverse + 1` above.
        rank = np.cumsum(present, dtype=np.int32)
        del present
        rank[0] = UNASSIGNED
        labels = np.empty(n, dtype=np.int32)
        for start in range(0, n, _RGB_CHUNK):
            stop = start + _RGB_CHUNK
            labels[start:stop] = rank[_pack_rgb(
                red[start:stop], green[start:stop], blue[start:stop]
            )]
    label_colors = {
        k + 1: ((u >> 16) & 255, (u >> 8) & 255, u & 255)
        for k, u in enumerate(uniq.tolist())
        if u != 0
    }
    return labels, label_colors


def load_rgb_segmented(path: str) -> PointCloud:
    """Load a binary PLY whose tree segmentation is encoded as per-point RGB.

    Each distinct colour becomes a tree label; pure black ``(0, 0, 0)`` is
    treated as unsegmented.  Coordinates are read as float32; ``time``, the
    normals and ``alpha`` are kept as attributes, and the per-label colours are
    remembered so the file can be written back out (recoloured) on save.
    """
    with open(path, "rb") as fh:
        fmt, props, count, offset = _parse_ply_header(fh)

    byteorder = ">" if fmt == "binary_big_endian" else "<"
    dtype = np.dtype([
        (name, byteorder + PLY_TYPE_MAP.get(ptype.lower(), "f4"))
        for name, ptype in props
    ])
    data = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))

    names = {n.lower(): n for n in data.dtype.names}
    coords = np.column_stack([
        data[names["x"]], data[names["y"]], data[names["z"]]
    ]).astype(np.float32)

    labels, label_colors = labels_from_rgb(
        data[names["red"]], data[names["green"]], data[names["blue"]]
    )

    # Keep remaining columns so we can round-trip the raycloud format on save.
    keep = {names.get(n) for n in ("x", "y", "z", "red", "green", "blue")}
    attributes = {
        name: np.ascontiguousarray(data[name])
        for name in data.dtype.names
        if name not in keep
    }

    return PointCloud(
        coords=coords,
        labels=labels,
        attributes=attributes,
        source_path=path,
        label_field="treeID",
        source_format="raycloud_rgb",
        label_colors=label_colors,
    )


def save(cloud: PointCloud, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".las", ".laz"):
        _save_las(cloud, path)
        return
    if ext != ".ply":
        raise ValueError(
            f"segfix writes binary PLY and LAS/LAZ, not {ext or path!r}"
        )
    if cloud.source_format == "raycloud_rgb":
        save_rgb_segmented(cloud, path)
    else:
        _save_ply(cloud, path)


def color_for_label(label: int, label_colors=None) -> tuple[int, int, int]:
    """RGB (0-255) for a tree label: its original colour, or a stable hash.

    Labels carried over from the source keep their colour; labels created
    during editing (splits, new trees) get a deterministic colour derived from
    the id so they are distinct and reproducible.
    """
    from .model import NOISE, UNASSIGNED as _UN

    if label in (_UN, NOISE):
        return (0, 0, 0)
    if label_colors and label in label_colors:
        return tuple(label_colors[label])
    h = (label * 0.61803398875) % 1.0
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    v, p, q, t = 0.95, 0.33, 0.95 * (1 - f * 0.65), 0.95 * (1 - (1 - f) * 0.65)
    rgb = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return tuple(int(round(c * 255)) for c in rgb)


def save_rgb_segmented(cloud: PointCloud, path: str) -> None:
    """Write a raycloud-style binary PLY, recolouring points by tree label.

    Reconstructs the original column layout (double xyz, plus whatever
    attributes were carried — ``time``, normals, ``alpha``) and writes each
    point the colour of its current tree label, so a fixed segmentation is
    saved in exactly the input format.
    """
    n = cloud.n_points
    labels = cloud.labels

    # Resolve a colour per point from the (possibly edited) labels.
    colour = np.zeros((n, 3), dtype=np.uint8)
    for label in np.unique(labels):
        r, g, b = color_for_label(int(label), cloud.label_colors)
        mask = labels == label
        colour[mask] = (r, g, b)

    # Column order matches raycloudtools: x y z time nx ny nz r g b alpha,
    # but we only emit attributes actually present on the cloud.
    fields = [("x", "<f8"), ("y", "<f8"), ("z", "<f8")]
    extra_order = ["time", "nx", "ny", "nz"]
    present_extra = [k for k in extra_order if k in cloud.attributes]
    # any other carried attributes, appended after the known ones
    present_extra += [
        k for k in cloud.attributes if k not in present_extra
    ]

    columns = {
        "x": cloud.coords[:, 0].astype("<f8"),
        "y": cloud.coords[:, 1].astype("<f8"),
        "z": cloud.coords[:, 2].astype("<f8"),
    }
    for k in present_extra:
        arr = np.asarray(cloud.attributes[k])
        fields.append((k, arr.dtype.newbyteorder("<").str))
        columns[k] = arr
    for c, name in zip("rgb", ("red", "green", "blue")):
        fields.append((name, "u1"))
        columns[name] = colour[:, "rgb".index(c)]
    if "alpha" not in columns:
        fields.append(("alpha", "u1"))
        columns["alpha"] = np.full(n, 255, dtype=np.uint8)

    struct = np.empty(n, dtype=np.dtype(fields))
    for name, _ in fields:
        struct[name] = columns[name]

    ply_type = {
        "f8": "double", "f4": "float", "u1": "uchar", "u4": "uint",
        "i4": "int", "u2": "ushort", "i2": "short", "i1": "char",
    }
    header_lines = ["ply", "format binary_little_endian 1.0",
                    "comment written by segfix", f"element vertex {n}"]
    for name, dt in fields:
        key = np.dtype(dt).str.lstrip("<>|")
        header_lines.append(f"property {ply_type.get(key, 'float')} {name}")
    header_lines.append("end_header\n")
    header = ("\n".join(header_lines)).encode("ascii")

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(struct.tobytes())


def _normalize_labels(raw: np.ndarray) -> np.ndarray:
    """Coerce a raw tree-ID column to ``int32``, folding anything that can't
    be a real instance ID into :data:`~segfix.model.UNASSIGNED`.

    A real ID is non-negative and fits in signed 32-bit. ``NOISE`` (-1) is a
    first-class segfix label and is kept. Everything else is an upstream
    "no tree" sentinel — other negatives (e.g. ``INT32_MIN``, which the
    CHERLET pipeline writes), or, when the source column is unsigned, values
    past ``2**31 - 1`` (e.g. a ``uint32`` all-ones) — and becomes
    ``UNASSIGNED`` so it doesn't surface as a spurious tree.
    """
    wide = np.asarray(raw).astype(np.int64)  # widen first: no lossy wrap
    int32_max = np.iinfo(np.int32).max
    bad = (wide > int32_max) | ((wide < 0) & (wide != NOISE))
    return np.where(bad, UNASSIGNED, wide).astype(np.int32)


def _pick_label_field(names, requested: str | None) -> str | None:
    if requested is not None:
        return requested if requested in names else None
    lowered = {n.lower(): n for n in names}
    for cand in LABEL_CANDIDATES:
        if cand in names:
            return cand
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


# -- PLY -----------------------------------------------------------------
def _load_ply(path: str, label_field: str | None) -> PointCloud:
    from plyfile import PlyData

    ply = PlyData.read(path)
    vertex = ply["vertex"]
    coords = np.column_stack(
        [vertex["x"], vertex["y"], vertex["z"]]
    ).astype(np.float32)

    prop_names = [p.name for p in vertex.properties]
    field = _pick_label_field(prop_names, label_field)
    if field is not None:
        labels = _normalize_labels(vertex[field])
    else:
        labels = np.zeros(len(coords), dtype=np.int32)
        field = "treeID"

    skip = {"x", "y", "z", field}
    attributes = {
        name: np.asarray(vertex[name]) for name in prop_names if name not in skip
    }
    return PointCloud(
        coords=coords,
        labels=labels,
        attributes=attributes,
        source_path=path,
        label_field=field,
    )


def _save_ply(cloud: PointCloud, path: str) -> None:
    from plyfile import PlyData, PlyElement

    field = cloud.label_field
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), (field, "i4")]
    columns = [
        cloud.coords[:, 0],
        cloud.coords[:, 1],
        cloud.coords[:, 2],
        cloud.labels,
    ]
    for name, values in cloud.attributes.items():
        values = np.asarray(values)
        dtype.append((name, values.dtype.str))
        columns.append(values)

    vertex = np.empty(cloud.n_points, dtype=dtype)
    for (name, _), col in zip(dtype, columns):
        vertex[name] = col

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=False).write(path)


# -- LAS / LAZ ---------------------------------------------------------------
def _load_las(path: str, label_field: str | None) -> PointCloud:
    """Read a whole LAS/LAZ into a :class:`PointCloud`.

    The GUI never takes this path — it uses :class:`treecatalog.LasCatalog`,
    which memory-maps the uncompressed records — but it keeps :func:`load`
    format-complete for scripts and tests.
    """
    import laspy

    las = laspy.read(path)
    coords = np.asarray(las.xyz, dtype=np.float32)

    names = list(las.point_format.dimension_names)
    field = _pick_label_field(names, label_field)
    if field is not None:
        labels = _normalize_labels(las[field])
    else:
        labels = np.zeros(len(coords), dtype=np.int32)
        field = "treeID"

    skip = {"X", "Y", "Z", "x", "y", "z", field}
    attributes = {
        name: np.asarray(las[name]) for name in names if name not in skip
    }
    return PointCloud(
        coords=coords,
        labels=labels,
        attributes=attributes,
        source_path=path,
        label_field=field,
        source_format="las",
    )


def _save_las(cloud: PointCloud, path: str) -> None:
    """Write ``cloud`` as LAS/LAZ by cloning its source file and overwriting
    only the label column, so every other dimension round-trips byte for byte.

    Requires the cloud to have been loaded from a LAS/LAZ (``source_path``)
    with the same point count — the label edits are mapped back position for
    position. Raises ``ValueError`` otherwise.
    """
    import laspy

    src = cloud.source_path
    if not src or os.path.splitext(src)[1].lower() not in (".las", ".laz"):
        raise ValueError(
            "saving LAS/LAZ needs a cloud loaded from one (to preserve its "
            "header and every other per-point column)"
        )
    las = laspy.read(src)
    if len(las.points) != cloud.n_points:
        raise ValueError(
            f"{src} has {len(las.points)} points but the cloud has "
            f"{cloud.n_points}; can't map labels back"
        )

    field = cloud.label_field or "treeID"
    if field not in las.point_format.dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name=field, type=np.int32,
                                                 description="tree instance ID"))
    values = np.asarray(cloud.labels)
    if las[field].dtype.kind == "u":
        from .model import NOISE, UNASSIGNED

        values = np.where(values == NOISE, UNASSIGNED, values)
    las[field] = values.astype(las[field].dtype)
    las.write(path)
