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

"""A workspace is a folder holding a private working copy of an imported
point cloud, plus a small manifest recording where it came from. All edits
happen on the copy — the original source file is never opened for writing.

One workspace wraps exactly one imported file: the startup dialog
(:mod:`startup_ui`) creates it on import, and the folder — not the copy
inside it — is what gets remembered in :mod:`registry`.

A ``.laz`` source (what arbor's pipeline emits) is decompressed to a ``.las``
working copy on import — segfix edits by memory-mapping fixed-size point
records, which compressed LAZ has no room for. The manifest still records the
original ``.laz`` path, so :meth:`treecatalog.LasCatalog.save` knows to
re-compress the corrected cloud back to ``.laz`` beside the ``.las``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

MANIFEST_NAME = "segfix_project.json"

#: ``(message, fraction) -> None``, fraction 0..1 — the same shape as
#: :data:`segfix.treecatalog.ProgressFn`, so one progress window can carry an
#: import straight through into the open that follows it. Omitting it reports
#: nothing, which is what headless callers and the tests do.
ProgressFn = Callable[[str, float], None]

#: Bytes per read while copying, and points per chunk while decompressing.
#: Both are "big enough that the per-chunk overhead vanishes, small enough
#: that the bar still moves" — a 16MiB block is a few dozen updates on a
#: gigabyte file, and 4M points is a second or two of LAZ.
_COPY_CHUNK = 16 << 20
_LAZ_CHUNK = 4_000_000


def data_file(workspace_dir: str | Path) -> Path:
    """The working copy's path inside a workspace folder."""
    manifest = json.loads((Path(workspace_dir) / MANIFEST_NAME).read_text())
    return Path(workspace_dir) / manifest["data_file"]


def _copy_with_progress(source: Path, dest: Path, report: ProgressFn | None):
    """``shutil.copyfile`` in blocks, reporting how far through it is.

    A plot file is routinely several gigabytes, and copying it is the whole
    of an import for a ``.ply`` or ``.las``. ``shutil.copyfile`` has no way
    to say how far it has got, so the copy is spelled out here instead —
    the block size is large enough that reading in Python costs nothing
    measurable against the disk.
    """
    total = source.stat().st_size
    done = 0
    with open(source, "rb") as src_file, open(dest, "wb") as dest_file:
        while True:
            block = src_file.read(_COPY_CHUNK)
            if not block:
                break
            dest_file.write(block)
            done += len(block)
            if report is not None:
                report("Copying", done / total if total else 1.0)


def _decompress_with_progress(
    source: Path, dest: Path, report: ProgressFn | None
) -> None:
    """Stream a ``.laz`` out to an uncompressed ``.las``, chunk by chunk.

    ``laspy.read(src).write(dest)`` decompresses the entire cloud into
    memory first, which on the files this tool exists for is tens of
    gigabytes and no way to tell how far along it is. Reading in chunks and
    writing each one straight out holds a few million points at a time
    instead, and every chunk is a step on the bar.

    The header goes across whole (point format, scales, offsets, VLRs, the
    Extra-Bytes layout arbor's ``treeID`` lives in), and the EVLRs follow the
    points, which is where the format wants them.
    """
    import laspy

    with laspy.open(str(source)) as reader:
        total = reader.header.point_count
        with laspy.open(
            str(dest), mode="w", header=reader.header, do_compress=False
        ) as writer:
            done = 0
            for chunk in reader.chunk_iterator(_LAZ_CHUNK):
                writer.write_points(chunk)
                done += len(chunk)
                if report is not None:
                    report("Decompressing", done / total if total else 1.0)
            if reader.evlrs:
                writer.write_evlrs(reader.evlrs)


def create_workspace(
    source: str | Path,
    workspace_dir: str | Path,
    report: ProgressFn | None = None,
) -> Path:
    """Copy ``source`` into a new ``workspace_dir``, write a manifest, and
    return the path to the working copy.

    Raises ``FileExistsError`` if ``workspace_dir`` already exists and is
    non-empty, so callers needing an available name (e.g. the startup
    dialog, auto-naming from the source's filename) can retry with a
    different one rather than silently mixing two projects together.

    ``report`` (:data:`ProgressFn`) is called as the bytes or points go by.
    This is the first half of opening a project for the first time — a
    multi-gigabyte copy, or a LAZ decompression, before
    :func:`segfix.treecatalog.open_catalog` has even been reached — so it is
    worth a bar of its own rather than a window that paints nothing.
    """
    source = Path(source)
    workspace_dir = Path(workspace_dir)
    if workspace_dir.exists() and any(workspace_dir.iterdir()):
        raise FileExistsError(f"{workspace_dir} already exists and is not empty")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".laz":
        # Decompress to a .las working copy: segfix patches points by
        # memory-mapping fixed-size records, which LAZ doesn't allow.
        dest = workspace_dir / (source.stem + ".las")
        _decompress_with_progress(source, dest, report)
    else:
        dest = workspace_dir / source.name
        _copy_with_progress(source, dest, report)
    manifest = {
        "source": str(source.resolve()),
        "data_file": dest.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (workspace_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return dest


def create_from_files(
    sources,
    workspace_dir: str | Path,
    report: ProgressFn | None = None,
) -> Path:
    """Merge a set of per-tree files (:class:`segfix.merge.Source`) into one
    working copy in a new ``workspace_dir``, and return its path.

    The per-tree counterpart of :func:`create_workspace`: same folder, same
    manifest, same guarantee that nothing is written back to the inputs —
    only the copy is one cloud built out of many, with each file's tree ID
    (from its name or a mapping CSV) written per point. The manifest records
    every source and the ID it was given, so a project can say what it was
    built from long after the import.
    """
    from . import merge

    workspace_dir = Path(workspace_dir)
    if workspace_dir.exists() and any(workspace_dir.iterdir()):
        raise FileExistsError(f"{workspace_dir} already exists and is not empty")
    sources = list(sources)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    dest = workspace_dir / (workspace_dir.name + merge.output_suffix(sources))
    merge.merge(sources, dest, report)
    manifest = {
        "source": str(Path(sources[0].path).resolve().parent),
        "sources": [
            {"path": str(Path(s.path).resolve()), "tree_id": s.tree_id}
            for s in sources
        ],
        "data_file": dest.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (workspace_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return dest


# -- remembered project settings and derived caches --------------------------
#: Where derived arrays live inside a workspace. Nothing in here is source
#: data — deleting the folder costs a slower open and nothing else.
CACHE_DIR = ".segfix_cache"


def _manifest_path(data_file: str | Path) -> Path | None:
    """The manifest next to ``data_file``, or None if it isn't in a project.

    A cloud opened straight off disk has no workspace, so everything below
    turns into a no-op for it rather than an error.
    """
    manifest = Path(data_file).parent / MANIFEST_NAME
    return manifest if manifest.exists() else None


def settings(data_file: str | Path) -> dict:
    """What the user already decided about this project.

    Keys are only present once decided, so a caller can tell "declined the
    downsample" (``{"voxel_size": None}``) from "never asked" (no key) —
    which is the difference between reopening quietly and asking again.
    """
    manifest = _manifest_path(data_file)
    if manifest is None:
        return {}
    try:
        return dict(json.loads(manifest.read_text()).get("settings", {}))
    except (OSError, ValueError, TypeError):
        return {}


def remember(data_file: str | Path, **decided) -> None:
    """Record decisions in the manifest, leaving the rest of it alone.

    Best-effort: a project on read-only media still opens, it just asks
    again next time.
    """
    manifest = _manifest_path(data_file)
    if manifest is None:
        return
    try:
        content = json.loads(manifest.read_text())
        content.setdefault("settings", {}).update(decided)
        manifest.write_text(json.dumps(content, indent=2))
    except (OSError, ValueError, TypeError):
        pass


def _stamp(data_file: Path) -> str:
    """Identifies the exact bytes a cache was derived from."""
    stat = data_file.stat()
    return f"{stat.st_size}-{int(stat.st_mtime_ns)}"


def cached_array(data_file: str | Path, kind: str) -> "object | None":
    """A previously cached array for ``data_file``, or None.

    None whenever anything is off — no project, no file, a stamp that does
    not match the data file as it stands now, or an unreadable cache. The
    caller recomputes, which is what it would have done anyway.
    """
    import numpy as np

    manifest = _manifest_path(data_file)
    if manifest is None:
        return None
    path = manifest.parent / CACHE_DIR / f"{Path(data_file).name}.{kind}.npz"
    try:
        with np.load(path) as loaded:
            if str(loaded["stamp"]) != _stamp(Path(data_file)):
                return None
            return loaded["value"]
    except (OSError, ValueError, KeyError):
        return None


def cache_array(data_file: str | Path, kind: str, value) -> None:
    """Cache ``value`` against ``data_file``'s current bytes. Best-effort."""
    import numpy as np

    manifest = _manifest_path(data_file)
    if manifest is None:
        return
    folder = manifest.parent / CACHE_DIR
    try:
        folder.mkdir(exist_ok=True)
        np.savez(
            folder / f"{Path(data_file).name}.{kind}.npz",
            stamp=_stamp(Path(data_file)),
            value=value,
        )
    except (OSError, ValueError):
        pass
