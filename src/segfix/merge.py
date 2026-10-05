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

"""Import a folder of per-tree clouds as one project.

Plenty of pipelines emit one file per tree — raycloudtools' ``raysplit``,
and most per-tree extraction tools — plus a ground/unassigned cloud beside
them. segfix edits one cloud with a tree-ID field per point, so this merges
such a set into exactly that on import: each file's points carry the tree ID
its *filename* (or a CSV mapping) gives, written into a ``treeID`` field.

The ID comes from the name, never from the file's own contents, which is the
whole point: a per-tree file usually has no ID field at all, and segfix's
fallback — numbering trees by their RGB colour — invents numbering that has
nothing to do with the one the segmentation used. Merging by filename keeps
the original IDs, so a tree is the same tree in segfix as in whatever wrote
the files, and :mod:`segfix.export` writes the set back out under those IDs.

Everything downstream (the catalog, saving, exporting) works on the merged
copy in the project folder, exactly as for a single imported cloud — the
originals are never written to.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from . import io
from .model import UNASSIGNED

#: ``(message, fraction) -> None`` — as :data:`segfix.workspace.ProgressFn`.
ProgressFn = Callable[[str, float], None]

#: Where the merged tree IDs are written. ``treeID`` is what arbor and
#: raycloudtools write and the first name :data:`segfix.io.LABEL_CANDIDATES`
#: looks for, so the merged file opens as a segmented cloud anywhere.
LABEL_FIELD = "treeID"

#: Default filename → tree ID rule: the last run of digits in the stem, so
#: ``plot12_tree_7.ply`` is tree 7, not tree 12.
DEFAULT_PATTERN = r"(\d+)"

#: Points per chunk while merging — big enough that per-chunk overhead
#: vanishes, small enough to keep peak memory flat however large the inputs.
_CHUNK = 4_000_000


@dataclass(frozen=True)
class Source:
    """One input file and the tree ID its points will carry.

    ``tree_id`` is ``None`` when nothing could be worked out for it (the
    import dialog shows those, and :func:`problems` refuses them), and
    :data:`~segfix.model.UNASSIGNED` for a ground/unassigned file, whose
    points come in as unassigned rather than as a tree.
    """

    path: Path
    tree_id: int | None

    @property
    def is_ground(self) -> bool:
        return self.tree_id == UNASSIGNED


# -- working out which file is which tree ------------------------------------
def tree_id_from_name(name: str | Path, pattern: str = DEFAULT_PATTERN) -> int | None:
    """The tree ID in a filename, or None if the pattern doesn't match.

    The *last* match in the stem wins, and the first capture group of it (or
    the whole match, for a pattern with no group). Last, because the tree
    number is what the name ends with far more often than it starts with:
    ``plot12_tree_7.ply`` is tree 7, and a leading plot or tile number would
    otherwise claim every file in the folder for the same tree.
    """
    try:
        rx = re.compile(pattern)
    except re.error:
        return None
    last = None
    for match in rx.finditer(Path(name).stem):
        last = match.group(1) if match.groups() else match.group(0)
    if last is None:
        return None
    try:
        return int(last)
    except ValueError:
        return None


def read_mapping(path: str | Path) -> dict[str, int]:
    """Read a ``path,tree_id`` CSV into ``{key: id}``.

    Both the path as written and its bare filename become keys, so a mapping
    file works whether it lists absolute paths, paths relative to somewhere
    else, or just names. A header row is skipped if its second column isn't
    a number; blank and ``#`` lines are ignored.
    """
    mapping: dict[str, int] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            if len(row) < 2 or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            key, value = row[0].strip(), row[1].strip()
            try:
                tree_id = int(float(value))
            except ValueError:
                continue  # a header, or a line that isn't a mapping
            mapping[key] = tree_id
            mapping[Path(key).name] = tree_id
    return mapping


def plan_sources(
    paths: Iterable[str | Path],
    *,
    pattern: str = DEFAULT_PATTERN,
    mapping: dict[str, int] | None = None,
    ground: Iterable[str | Path] = (),
) -> list[Source]:
    """Pair each path with the tree ID it will carry.

    A path in ``ground`` comes in as unassigned. Otherwise ``mapping`` (a
    CSV, see :func:`read_mapping`) wins where it has an entry — it is the
    explicit answer — and ``pattern`` is the fallback.
    """
    ground_keys = {str(Path(p)) for p in ground}
    sources = []
    for path in paths:
        path = Path(path)
        if str(path) in ground_keys:
            tree_id: int | None = UNASSIGNED
        elif mapping and (str(path) in mapping or path.name in mapping):
            tree_id = mapping.get(str(path), mapping.get(path.name))
        else:
            tree_id = tree_id_from_name(path, pattern)
        sources.append(Source(path=path, tree_id=tree_id))
    return sources


def problems(sources: Sequence[Source]) -> list[str]:
    """Everything that would stop :func:`merge`, in the order worth fixing.

    Written for a dialog to show as it stands: no exceptions, no first-error
    exit, so a folder with three unmatched names says so once rather than
    three times over three attempts.
    """
    found: list[str] = []
    if not sources:
        return ["Add some per-tree files to import."]

    missing = [s.path.name for s in sources if s.tree_id is None]
    if missing:
        shown = ", ".join(missing[:4]) + ("…" if len(missing) > 4 else "")
        found.append(
            f"{len(missing)} file(s) have no tree ID: {shown}. Change the "
            "pattern, load a mapping CSV, or mark them as ground."
        )

    seen: dict[int, str] = {}
    clashes: list[str] = []
    for source in sources:
        if source.tree_id is None or source.is_ground:
            continue
        if source.tree_id in seen:
            clashes.append(
                f"{seen[source.tree_id]} and {source.path.name} are both "
                f"tree {source.tree_id}"
            )
        seen[source.tree_id] = source.path.name
    if clashes:
        found.append(
            "Two files can't be the same tree: " + "; ".join(clashes[:3])
        )

    formats = {}
    for source in sources:
        fmt = io.sniff_format(str(source.path))
        formats.setdefault(fmt, source.path.name)
    if None in formats:
        found.append(
            f"{formats[None]} isn't a binary PLY or LAS/LAZ that segfix can read."
        )
    elif len(formats) > 1:
        found.append(
            "All the files have to be the same format; this set mixes "
            + " and ".join(sorted(f.upper() for f in formats))
            + "."
        )
    return found


def output_suffix(sources: Sequence[Source]) -> str:
    """The extension the merged copy gets: ``.las`` for LAS/LAZ inputs (the
    working copy is always uncompressed — see :mod:`segfix.workspace`),
    ``.ply`` for PLY."""
    fmt = io.sniff_format(str(sources[0].path))
    return ".las" if fmt == "las" else ".ply"


# -- the merge itself --------------------------------------------------------
def merge(
    sources: Sequence[Source], dest: str | Path, report: ProgressFn | None = None
) -> Path:
    """Write ``sources`` into one cloud at ``dest``, tagged by tree ID.

    Streams: peak memory is one chunk, not one input file and never the
    whole set, so merging a plot of several hundred trees costs what merging
    one costs. Raises ``ValueError`` if :func:`problems` has anything to say.
    """
    found = problems(sources)
    if found:
        raise ValueError(" ".join(found))
    dest = Path(dest)
    if io.sniff_format(str(sources[0].path)) == "las":
        _merge_las(sources, dest, report)
    else:
        _merge_ply(sources, dest, report)
    return dest


def _progress_steps(sources: Sequence[Source]) -> list[int]:
    """Cumulative point counts, for a bar that moves with the data rather
    than with the file count — one ground file routinely outweighs every
    tree file in the set put together."""
    counts = [_point_count(source.path) for source in sources]
    total = sum(counts) or 1
    running, out = 0, []
    for count in counts:
        running += count
        out.append(running / total)
    return out


def _point_count(path: Path) -> int:
    if io.sniff_format(str(path)) == "las":
        import laspy

        with laspy.open(str(path)) as reader:
            return int(reader.header.point_count)
    with open(path, "rb") as fh:
        return io._parse_ply_header(fh)[2]


# -- LAS ---------------------------------------------------------------------
def _merge_las(
    sources: Sequence[Source], dest: Path, report: ProgressFn | None
) -> None:
    """Merge LAS/LAZ inputs, keeping the first file's point format.

    Offsets come from the whole set's bounds rather than the first file's,
    so a plot spread over hundreds of metres still quantises cleanly; scales
    are the first file's, which is what its own coordinates round-tripped
    through. Dimensions a later file doesn't have are left at zero rather
    than refused: a ground cloud is routinely plainer than the tree files.
    """
    import laspy

    header = _las_header(sources)
    shared_cache: dict[str, list[str]] = {}
    marks = _progress_steps(sources)
    with laspy.open(str(dest), mode="w", header=header) as writer:
        for source, done in zip(sources, marks):
            if report is not None:
                report(f"Merging {source.path.name}", done)
            with laspy.open(str(source.path)) as reader:
                names = shared_cache.setdefault(
                    str(source.path),
                    [
                        name
                        for name in reader.header.point_format.dimension_names
                        if name in header.point_format.dimension_names
                        and name not in ("X", "Y", "Z", LABEL_FIELD)
                    ],
                )
                for chunk in reader.chunk_iterator(_CHUNK):
                    points = laspy.ScaleAwarePointRecord.zeros(
                        len(chunk), header=header
                    )
                    points.x, points.y, points.z = chunk.x, chunk.y, chunk.z
                    for name in names:
                        points[name] = chunk[name]
                    points[LABEL_FIELD] = np.full(
                        len(chunk), source.tree_id, dtype=np.int32
                    )
                    writer.write_points(points)
    if report is not None:
        report("Finishing", 1.0)


def _las_header(sources: Sequence[Source]):
    """A header for the merged file: the first input's version, point format
    and scales, offsets covering every input, and a ``treeID`` field."""
    import laspy

    with laspy.open(str(sources[0].path)) as reader:
        first = reader.header
        header = laspy.LasHeader(
            version=first.version, point_format=first.point_format
        )
        header.scales = first.scales
    mins = np.array(first.mins, dtype=np.float64)
    for source in sources[1:]:
        with laspy.open(str(source.path)) as reader:
            mins = np.minimum(mins, np.asarray(reader.header.mins, dtype=np.float64))
    header.offsets = mins
    if LABEL_FIELD not in header.point_format.dimension_names:
        header.add_extra_dim(
            laspy.ExtraBytesParams(
                name=LABEL_FIELD, type=np.int32, description="tree instance ID"
            )
        )
    return header


# -- PLY ---------------------------------------------------------------------
def _merge_ply(
    sources: Sequence[Source], dest: Path, report: ProgressFn | None
) -> None:
    """Merge binary PLY inputs.

    The merged file carries the properties *every* input has — anything one
    file is missing would be made up for the rest — plus ``treeID``. Each
    input is read a chunk at a time through a memmap, so a big ground cloud
    costs a chunk of memory like everything else.
    """
    layouts = {str(s.path): _ply_layout(s.path) for s in sources}
    shared = _shared_properties([layouts[str(s.path)] for s in sources])
    first = layouts[str(sources[0].path)]
    out_dtype = np.dtype(
        [(name, first.dtype[first.names[name]]) for name in shared]
        + [(LABEL_FIELD, "<i4")]
    )
    total = sum(layouts[str(s.path)].count for s in sources)

    with open(dest, "wb") as out:
        out.write(_ply_header(out_dtype, total))
        done = 0
        for source in sources:
            layout = layouts[str(source.path)]
            if report is not None:
                report(f"Merging {source.path.name}", done / (total or 1))
            rows = np.memmap(
                source.path, dtype=layout.dtype, mode="r",
                offset=layout.offset, shape=(layout.count,),
            )
            for start in range(0, layout.count, _CHUNK):
                block = rows[start:start + _CHUNK]
                buffer = np.empty(len(block), dtype=out_dtype)
                for name in shared:
                    buffer[name] = block[layout.names[name]]
                buffer[LABEL_FIELD] = source.tree_id
                out.write(buffer.tobytes())
                done += len(block)
            del rows
    if report is not None:
        report("Finishing", 1.0)


@dataclass(frozen=True)
class _PlyLayout:
    dtype: np.dtype
    #: lowercased property name -> the name as the file spells it
    names: dict[str, str]
    count: int
    offset: int


def _ply_layout(path: Path) -> _PlyLayout:
    with open(path, "rb") as fh:
        fmt, props, count, offset = io._parse_ply_header(fh)
    if not fmt.startswith("binary"):
        raise ValueError(
            f"{path.name} is an ASCII PLY; segfix reads binary PLY only"
        )
    byteorder = ">" if "big" in fmt else "<"
    dtype = np.dtype([
        (name, byteorder + io.PLY_TYPE_MAP.get(ptype.lower(), "f4"))
        for name, ptype in props
    ])
    return _PlyLayout(
        dtype=dtype,
        names={name.lower(): name for name in dtype.names},
        count=count,
        offset=offset,
    )


def _shared_properties(layouts: Sequence[_PlyLayout]) -> list[str]:
    """The properties every input has, in the first one's order, with any
    existing label field dropped — the filename is the authority on which
    tree a point belongs to, so a stale ID column would only contradict it.
    """
    label_names = {name.lower() for name in io.LABEL_CANDIDATES}
    label_names.add(LABEL_FIELD.lower())
    common = set(layouts[0].names)
    for layout in layouts[1:]:
        common &= set(layout.names)
    return [
        name for name in layouts[0].names
        if name in common and name not in label_names
    ]


#: numpy kind + itemsize -> the PLY type name for it.
_PLY_NAMES = {
    ("f", 4): "float", ("f", 8): "double",
    ("i", 1): "char", ("i", 2): "short", ("i", 4): "int",
    ("u", 1): "uchar", ("u", 2): "ushort", ("u", 4): "uint",
}


def _ply_header(dtype: np.dtype, count: int) -> bytes:
    endian = "big" if dtype[0].byteorder == ">" else "little"
    lines = [
        "ply",
        f"format binary_{endian}_endian 1.0",
        "comment merged from per-tree files by segfix",
        f"element vertex {count}",
    ]
    for name in dtype.names:
        field = dtype[name]
        ply_type = _PLY_NAMES.get((field.kind, field.itemsize))
        if ply_type is None:
            raise ValueError(f"no PLY type for {name} ({field})")
        lines.append(f"property {ply_type} {name}")
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")
