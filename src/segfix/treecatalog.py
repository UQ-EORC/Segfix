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

"""Label-grouped index over one big point cloud, for on-demand tree+neighbour
loading instead of loading (and rendering) the whole cloud at once.

Memory-maps the file once, keeps a few cheap resident arrays for fast
repeated queries, and loads/saves only the points a given operation actually
needs — grouped by tree label, since the review workflow already works one
tree (+ neighbours) at a time.

Two on-disk formats get this treatment, because both store their points as
fixed-size records at a known offset that ``numpy.memmap`` can index and patch
in place:

* **binary PLY** (:class:`TreeCatalog`) — the raycloudtools / label-field clouds
  segfix has always handled.
* **uncompressed LAS** (:class:`LasCatalog`) — what arbor's pipeline produces
  (a ``treeID`` Extra-Bytes column). arbor writes ``.laz``; :mod:`workspace`
  decompresses it to a ``.las`` working copy on import, and :meth:`LasCatalog.save`
  re-compresses back to ``.laz`` so arbor can re-read the corrected cloud.

:func:`open_catalog` picks the backend from the path extension.

A cloud too dense to review comfortably (points closer than
:data:`segfix.density.DENSE_SPACING`) can be voxel-decimated for the session
— the catalog then indexes, loads and edits one point per voxel, and
:meth:`_BaseCatalog.save` interpolates those edits back onto every
full-resolution point before writing, so the file keeps all of its points.
"""

from __future__ import annotations

import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import io
from .model import NOISE, UNASSIGNED, PointCloud

# Beyond this magnitude, float32 mantissa precision (~7 significant digits)
# starts eating into sub-metre point detail — e.g. a UTM northing around
# 7,000,000 only keeps ~1m precision once cast to float32. CloudCompare uses
# the same "global shift" idea for the same reason.
GLOBAL_SHIFT_THRESHOLD = 1.0e4

#: Points handled per chunk by the passes that walk the whole cloud —
#: reading coordinates and labels, ranking labels for the sort. Big enough
#: that the per-chunk overhead is nothing against the work itself, small
#: enough that a chunk's wide (float64 / int64) working copies stay a few
#: hundred MB however many points the cloud has.
_CHUNK = 4_000_000

#: ``(mins, maxs, suggested_shift) -> (dx, dy, dz) | None`` — ``None`` means
#: "load unshifted". Wired to a Qt dialog by the app; ``None`` (the default)
#: skips the prompt entirely and never shifts, which is what every existing
#: (small-coordinate) catalog and test expects.
ShiftPrompt = Callable[[np.ndarray, np.ndarray, np.ndarray], "tuple[float, float, float] | None"]

#: ``(spacing, n_points, suggested_voxel) -> voxel_size | None`` — ``None``
#: means "review at full resolution". Wired to a Qt dialog by the app; as
#: with :data:`ShiftPrompt`, omitting it (the default) skips the check
#: entirely, so every existing caller and test loads every point.
DensityPrompt = Callable[
    [float, int, float, "Callable[[float], float | None]"], "float | None"
]

#: ``(message, fraction) -> None``, fraction 0..1 — where a long open or save
#: has got to. Wired to a progress window by the app; omitting it (the
#: default) reports nothing, so headless callers and tests are unaffected.
ProgressFn = Callable[[str, float], None]

# The phases of an open and roughly what share of the time each takes, so
# the bar moves at a believable rate. Measured on a 39M-point PLY: decoding
# coordinates 0.5s, labels 2.4s, measuring density 1.9s, decimating 12.7s,
# indexing 4.0s. Decimating is now ~2.4x faster (density.voxel_indices sorts
# once rather than twice), so its 12.7s is ~5.3s and the shares come out at
# 4 / 17 / 13 / 38 / 28 percent. Only the ratios matter; leaving the old
# 55% in place would race the bar through the longest phase and then sit on
# "Indexing trees" for a third of the wait.
#
# Applying a shift is a second decode of the same coordinates, so it is
# worth what the first one is: 0.77s against 0.82s on a 6.5M-point LAS.
# It went in at the coordinate read's own 5%, and the five phases that were
# already here keep their ratios to one another, each scaled by 1/1.05 so
# the six still sum to 1.
_LOAD_PHASES = (
    ("Reading coordinates", 0.05),
    ("Applying global shift", 0.05),
    ("Reading tree labels", 0.16),
    ("Measuring point density", 0.12),
    ("Downsampling", 0.35),
    ("Indexing trees", 0.27),
)

# Where a save's own phases sit on the bar: everything before is the
# interpolation pass, which is the only part that scales with the file.
_SAVE_WRITE_AT = 0.85
_SAVE_FINALIZE_AT = 0.95

# Full-resolution points decoded at a time when a decimated session's edits
# are interpolated back on save (see _BaseCatalog._expand_to_full). Big
# enough that the per-chunk overhead vanishes, small enough that the decoded
# float64 coordinates stay a few hundred MB.
_EXPAND_CHUNK = 4_000_000


class _PhaseReporter:
    """Turns named load phases into the 0..1 fractions a progress callback
    wants, so the phases themselves stay readable.

    A phase that gets skipped (there is no downsampling on a cloud that
    doesn't need it) simply never reports, and the next one starts from its
    own place on the bar — a jump forward, which is honest: that work isn't
    happening.
    """

    def __init__(self, fn: ProgressFn | None):
        self._fn = fn
        self._start = {}
        acc = 0.0
        for name, weight in _LOAD_PHASES:
            self._start[name] = acc
            acc += weight

    def __call__(self, phase: str, within: float = 0.0) -> None:
        if self._fn is None:
            return
        start = self._start[phase]
        weight = dict(_LOAD_PHASES)[phase]
        self._fn(phase + "…", start + weight * max(0.0, min(1.0, within)))

    def done(self) -> None:
        if self._fn is not None:
            self._fn("Ready", 1.0)


def needs_global_shift(mins: np.ndarray, maxs: np.ndarray) -> bool:
    """True when any coordinate is far enough from the origin that float32
    storage would start losing meaningful precision."""
    if mins.size == 0:
        return False
    return bool(max(np.abs(mins).max(), np.abs(maxs).max()) > GLOBAL_SHIFT_THRESHOLD)


def suggest_global_shift(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    """A round shift bringing the bounding box's minimum corner to the origin
    — the same convention CloudCompare's "Global Shift" dialog suggests."""
    return -np.floor(mins)


@dataclass
class TreeRecord:
    """Cheap, always-available stats for one tree — no points loaded."""

    label: int
    count: int
    centroid: tuple[float, float]
    bbox: tuple[np.ndarray, np.ndarray]


def _label_order(labels: np.ndarray) -> np.ndarray:
    """``np.argsort(labels, kind="stable")``, via a key numpy radix-sorts.

    numpy's stable sort only takes the radix path for keys of two bytes or
    fewer; an int32 label array gets a comparison sort instead. Tree ids are
    a few thousand distinct values at most, so ranking them densely first
    (one bincount over the label range, then a lookup) gives a uint16 key
    that sorts in a linear pass — measured 872ms against 219ms on 6M points,
    for the identical permutation. This runs on every open *and* after every
    apply(), so it is on the path of a tree switch and a save as well.

    Falls back to sorting the labels themselves when the dense key would not
    fit: more than 65536 distinct trees, or ids spread so far apart that the
    bincount would be bigger than the cloud.
    """
    if not labels.size:
        return np.empty(0, dtype=np.intp)
    lo = int(labels.min())
    span = int(labels.max()) - lo + 1
    if span > max(labels.size, 1_000_000):
        return np.argsort(labels, kind="stable")
    # Chunked, so the int64 widening of the labels is a chunk's worth at a
    # time rather than 8 bytes a point over the whole cloud.
    counts = np.zeros(span, dtype=np.int64)
    for start in range(0, labels.size, _CHUNK):
        block = labels[start:start + _CHUNK].astype(np.int64)
        block -= lo
        counts += np.bincount(block, minlength=span)
    present = counts > 0
    del counts
    if int(present.sum()) > np.iinfo(np.uint16).max + 1:
        return np.argsort(labels, kind="stable")
    rank_of = (np.cumsum(present) - 1).astype(np.uint16)
    key = np.empty(labels.size, dtype=np.uint16)
    for start in range(0, labels.size, _CHUNK):
        block = labels[start:start + _CHUNK].astype(np.int64)
        block -= lo
        key[start:start + _CHUNK] = rank_of[block]
    return np.argsort(key, kind="stable")


class _BaseCatalog:
    """Format-agnostic label index over a memory-mapped point cloud.

    Subclasses implement the file-format specifics (:meth:`_open`,
    :meth:`_decode_coords_raw`, :meth:`_decode_labels`, :meth:`_write_labels`,
    :meth:`_finalize_save`); everything else here — the grouping index,
    neighbour search, subset load/apply and the diff-based save — is shared.

    ``labels`` and ``coords`` are decoded/copied once and kept resident
    (tens of MB even for multi-million-point clouds) so neighbour search and
    the tree table work instantly; the bulk per-point data (attributes, and
    coords/labels for anything not currently loaded) stays memory-mapped and
    is only read for the points actually requested via :meth:`load`.
    """

    def __init__(
        self,
        path: str,
        label_field: str | None = None,
        shift_prompt: ShiftPrompt | None = None,
        density_prompt: DensityPrompt | None = None,
        progress: ProgressFn | None = None,
        class_field: str | None = None,
    ):
        self.path = path
        self._label_field_req = label_field
        report = _PhaseReporter(progress)
        # Subclass fills in: offset, count, dtype, _names, is_rgb, label_field,
        # and _mm (the memmap of fixed-size point records).
        self._open()

        # Decoded at full (float64) precision — shifting after the fact
        # would already have lost whatever a premature float32 cast rounded
        # away — but a chunk at a time, straight into the float32 array the
        # session keeps. Decoding the whole cloud at once instead costs
        # 24 bytes a point for the float64 copy and as much again for the
        # temporaries around it, which on a 480M-point plot is tens of
        # gigabytes of peak, for an array that ends up 12 bytes a point.
        report("Reading coordinates")
        self.coords, bounds = self._read_coords(None)
        self.global_shift = self._resolve_global_shift(bounds, shift_prompt)
        if self.global_shift is not None:
            # Only a cloud that is actually being shifted pays for a second
            # pass, and it is the one case where the shift has to be applied
            # in float64, before the cast: subtracting it from coordinates
            # already rounded to float32 is exactly the precision loss the
            # shift exists to avoid.
            report("Applying global shift")
            self.coords, _ = self._read_coords(self.global_shift, out=self.coords)

        report("Reading tree labels")
        self.labels, self.label_colors = self._read_labels()

        # New tree IDs must clear every label in the *file*, including trees
        # that decimation may be about to drop from the working set below.
        self._next_id = int(self.labels.max()) + 1 if self.labels.size else 1
        self._resolve_decimation(density_prompt, report)

        # Snapshot to diff against on save — see save().
        self._original_labels = self.labels.copy()
        self.class_field: str | None = None
        self.classes: np.ndarray | None = None
        self._original_classes: np.ndarray | None = None
        if class_field:
            self.set_class_field(class_field)
        report("Indexing trees")
        self._build_index()
        report.done()

    # -- decimation ("this cloud is very dense") ---------------------------
    def _resolve_decimation(
        self, density_prompt: DensityPrompt | None, report=None
    ) -> None:
        """Offer to review a very dense cloud voxel-decimated, and take the
        offer up if the prompt accepts it.

        When it does, ``coords``/``labels`` (and therefore the tree index,
        neighbour search and every :meth:`load`) cover only the kept points,
        while ``_sub_idx`` remembers which file row each of them came from —
        that mapping is what lets :meth:`save` put the edits back onto all
        the points, not just the kept ones.

        As with the global shift, nothing happens without a prompt: a
        headless open always works at full resolution.
        """
        from . import density

        self.spacing: float | None = None
        self.voxel_size: float | None = None
        # File row of each working point; None while working at full
        # resolution, where the two are the same thing.
        self._sub_idx: np.ndarray | None = None
        if density_prompt is None or self.labels.size < 2:
            return

        # Large coordinates that weren't shifted have already lost sub-metre
        # detail to the float32 cast -- that is what the shift prompt warns
        # about. Both the spacing measurement and the voxel grid would then
        # be reading quantisation rather than the cloud: a 5mm scan at a UTM
        # northing measures as "1mm spacing" and voxelises four times harder
        # than asked, because points that really are distinct now share a
        # coordinate. Don't offer a choice made on those numbers.
        if self.global_shift is None and needs_global_shift(
            self.coords.min(axis=0), self.coords.max(axis=0)
        ):
            return

        from . import workspace

        # What the user already said about this project. "voxel_size" present
        # and None means they declined; absent means they were never asked.
        decided = workspace.settings(self.path)

        # Sample boxes of the cloud at most once, and only if something
        # asks: the spacing measurement wants them, so does the prompt's
        # "keeps about N% of the points", and a reopen with a remembered
        # spacing may want neither. Finding them is a pass over every point
        # per attempt — twenty seconds on a 480M-point cloud.
        blocks: list = []

        def sampled():
            if not blocks:
                blocks.extend(density.sample_blocks(self.coords))
            return blocks

        if "spacing" in decided:
            self.spacing = decided["spacing"]
        else:
            if report is not None:
                report("Measuring point density")
            self.spacing = density.spacing_from_blocks(sampled())
            workspace.remember(self.path, spacing=self.spacing)
        if not (0.0 < (self.spacing or 0.0) < density.DENSE_SPACING):
            return

        if "voxel_size" in decided:
            chosen = decided["voxel_size"]
        else:
            chosen = density_prompt(
                self.spacing, int(self.count), density.suggest_voxel(self.spacing),
                lambda voxel: density.estimate_kept_fraction(sampled(), voxel),
            )
            workspace.remember(
                self.path,
                voxel_size=None if chosen is None else float(chosen),
            )
        if chosen is None:
            return

        if report is not None:
            report("Downsampling")
        # The kept rows depend only on the coordinates and the voxel size,
        # and coordinates are never rewritten — a save patches labels. So
        # this survives every edit, and the stamp only fails it if the file
        # is replaced underneath.
        keep = workspace.cached_array(self.path, "voxel")
        if keep is None:
            keep = density.voxel_indices(self.coords, float(chosen))
            workspace.cache_array(self.path, "voxel", keep)
        keep = np.asarray(keep, dtype=np.int64)
        if keep.size >= self.count or (keep.size and keep.max() >= self.count):
            return  # nothing to gain, or a cache that doesn't fit this file
        self._sub_idx = keep
        self.coords = self.coords[keep]
        self.labels = self.labels[keep]
        self.voxel_size = float(chosen)

    @property
    def is_decimated(self) -> bool:
        """True when the working set is a voxel-decimated subset of the file."""
        return self._sub_idx is not None

    @property
    def working_count(self) -> int:
        """Points actually loaded/edited — ``count`` unless decimated."""
        return int(self.labels.size)

    def _rows(self, idx: np.ndarray) -> np.ndarray:
        """File rows for working-set positions ``idx``."""
        return idx if self._sub_idx is None else self._sub_idx[idx]

    # -- point classes ----------------------------------------------------
    #: LAS bit-packed flag bytes: never offered as a class field.
    _FLAG_FIELDS = ("bit_fields", "classification_flags")

    def class_fields(self) -> list[str]:
        """Per-point columns that could hold a point class: integer or float
        scalars other than the coordinates and the tree label (and, for an
        RGB-segmented PLY, the colour that *is* the tree label).

        A LAS in point formats 0-5 stores its classification in the low five
        bits of a byte laspy calls ``raw_classification``; it is offered by
        its usual name, ``classification``.
        """
        skip = {self._names.get(n) for n in ("x", "y", "z")}
        skip.add(self.label_field)
        if self.is_rgb:
            skip |= {self._names.get(n) for n in ("red", "green", "blue")}
        out = []
        for name in self.dtype.names:
            if name in skip or name in self._FLAG_FIELDS:
                continue
            dt = self.dtype[name]
            if dt.shape or dt.kind not in "iuf":
                continue
            out.append("classification" if name == "raw_classification" else name)
        return out

    def _class_column(self, field: str) -> tuple[str, int | None]:
        """``(record field, bit mask or None)`` holding class field ``field``."""
        if field in self.dtype.names and field not in self._FLAG_FIELDS:
            column, mask = field, None
        elif (field.lower() == "classification"
              and "raw_classification" in self.dtype.names):
            column, mask = "raw_classification", 0x1F
        else:
            raise ValueError(f"{os.path.basename(self.path)} has no field {field!r}")
        if column == self.label_field:
            raise ValueError(f"{field!r} is the tree ID field, not a class field")
        dt = self.dtype[column]
        if dt.shape or dt.kind not in "iuf":
            raise ValueError(f"{field!r} isn't a numeric per-point field")
        return column, mask

    def _class_bounds(self, column: str, mask: int | None) -> tuple[int, int]:
        if mask is not None:
            return 0, mask
        dt = self.dtype[column]
        if dt.kind in "iu":
            info = np.iinfo(dt)
            int32 = np.iinfo(np.int32)
            return max(int(info.min), int32.min), min(int(info.max), int32.max)
        # A float holds every integer exactly up to its mantissa: 2**24 for
        # float32, the narrowest a PLY property can be.
        return -(2 ** 24), 2 ** 24

    def class_range(self, field: str | None = None) -> tuple[int, int]:
        """The smallest and largest class value ``field`` (by default the
        class field in use) can store — a new class has to fit, or saving
        would silently wrap it."""
        if field is not None:
            return self._class_bounds(*self._class_column(field))
        if self.class_field is None:
            raise ValueError("no class field set")
        return self._class_bounds(*self._class_col)

    def _decode_class_column(self, sub, column: str, mask: int | None) -> np.ndarray:
        raw = np.asarray(sub[column])
        if mask is not None:
            raw = raw & mask
        if raw.dtype.kind == "f":
            raw = np.rint(np.nan_to_num(raw))
        lo, hi = self._class_bounds(column, mask)
        return np.clip(raw.astype(np.int64), lo, hi)

    def _read_class_column(self, column: str, mask: int | None) -> np.ndarray:
        out = np.empty(self.count, dtype=np.int32)
        for start in range(0, self.count, _CHUNK):
            stop = min(start + _CHUNK, self.count)
            out[start:stop] = self._decode_class_column(
                self._mm[start:stop], column, mask
            )
        return out

    def class_values_in(self, field: str) -> np.ndarray:
        """The distinct values field ``field`` holds across the whole file —
        what the class set-up dialog lists before a field is chosen."""
        column, mask = self._class_column(field)
        found = np.empty(0, dtype=np.int64)
        for start in range(0, self.count, _CHUNK):
            stop = min(start + _CHUNK, self.count)
            block = self._decode_class_column(self._mm[start:stop], column, mask)
            found = np.union1d(found, np.unique(block))
        return found

    def set_class_field(self, field: str | None) -> None:
        """Read (or, with ``None``, drop) the per-point class field.

        Refused while class edits are unsaved: they belong to the field
        that is set now, and switching would throw them away unasked.
        """
        if self.has_unsaved_class_edits():
            raise ValueError("Save the class edits before changing the class field")
        if not field:
            self.class_field = None
            self.classes = self._original_classes = None
            return
        column, mask = self._class_column(field)
        classes = self._read_class_column(column, mask)
        if self._sub_idx is not None:
            classes = classes[self._sub_idx]
        self._class_col = (column, mask)
        self.class_field = field
        self.classes = classes
        self._original_classes = classes.copy()

    def has_unsaved_class_edits(self) -> bool:
        return self.classes is not None and bool(
            np.any(self.classes != self._original_classes)
        )

    def _raw_class_codes(self, sub) -> np.ndarray:
        """Each record's class as the file stores it (see _raw_label_codes)."""
        return self._decode_class_column(sub, *self._class_col)

    def _tree_class_codes(self, sub) -> np.ndarray:
        """One code per (file tree, file class) pair, so a class edit on one
        tree is interpolated only onto points of that same tree and class —
        a leaf edit on one crown never bleeds into the leaves next door.

        The pair is hashed into one int64 (wrapping multiply by an odd
        constant); the codes are only ever compared for equality.
        """
        tree = self._raw_label_codes(sub).astype(np.uint64)
        cls = self._raw_class_codes(sub).astype(np.uint64)
        with np.errstate(over="ignore"):
            return (tree * np.uint64(0x9E3779B97F4A7C15) + cls).view(np.int64)

    def _write_classes(self, out, rows: np.ndarray, values: np.ndarray) -> None:
        column, mask = self._class_col
        dt = self.dtype[column]
        if mask is not None:
            # Only the class bits: the rest of the byte is the synthetic /
            # key-point / withheld flags, which stay as they were.
            kept = out[column][rows] & np.array(~mask & 0xFF, dtype=dt)
            out[column][rows] = kept | (values & mask).astype(dt)
        else:
            out[column][rows] = values.astype(dt)

    # -- global shift ------------------------------------------------------
    def _read_coords(self, shift, out: np.ndarray | None = None):
        """``(coords, (mins, maxs))``: every point's coordinates as float32,
        with ``shift`` added first when given, plus the cloud's true bounds
        measured in float64 before the cast.

        Read in chunks into one preallocated array (reusing ``out`` when the
        caller has one to refill), so the peak is the array itself rather
        than the array plus a float64 copy of the whole cloud.
        """
        if out is None:
            out = np.empty((self.count, 3), dtype=np.float32)
        mins = np.full(3, np.inf)
        maxs = np.full(3, -np.inf)
        for start in range(0, self.count, _CHUNK):
            stop = min(start + _CHUNK, self.count)
            block = self._decode_coords_raw(self._mm[start:stop])
            np.minimum(mins, block.min(axis=0), out=mins)
            np.maximum(maxs, block.max(axis=0), out=maxs)
            if shift is not None:
                block += shift
            out[start:stop] = block
        return out, (mins, maxs)

    def _read_labels(self):
        """``(labels int32, label_colors)`` for every point in the file.

        Chunked, like the coordinates: :func:`io._normalize_labels` widens
        to int64 and builds a mask and a ``np.where`` result on the way to
        int32, so doing it in one bite costs about 20 bytes a point in
        temporaries — 10GB on a 480M-point plot — for a 4-byte-a-point
        answer. An RGB-segmented cloud is decoded whole instead: there the
        label *is* the colour, numbered across the file's distinct colours,
        which a chunk can't know.
        """
        if self.is_rgb:
            labels, colors = self._decode_labels(self._mm)
            return np.asarray(labels).astype(np.int32), colors
        out = np.empty(self.count, dtype=np.int32)
        for start in range(0, self.count, _CHUNK):
            stop = min(start + _CHUNK, self.count)
            block, _colors = self._decode_labels(self._mm[start:stop])
            out[start:stop] = block
        return out, None

    def _resolve_global_shift(
        self, bounds: tuple[np.ndarray, np.ndarray],
        shift_prompt: ShiftPrompt | None,
    ) -> np.ndarray | None:
        """``None`` (load unshifted) unless the cloud's coordinates are large
        enough to need it *and* a prompt is wired up and says yes — see
        :data:`ShiftPrompt`. Never shifts on its own initiative: an
        unattended/headless open (no ``shift_prompt``) always gets the file's
        original coordinates, unchanged."""
        if shift_prompt is None or not self.count:
            return None
        mins, maxs = bounds
        if not needs_global_shift(mins, maxs):
            return None

        # Asked once per project, not once per open: the shift is a decision
        # about how to review this cloud, and re-asking it every time invited
        # a different answer and a scene that no longer matched the notes.
        from . import workspace

        decided = workspace.settings(self.path)
        if "global_shift" in decided:
            chosen = decided["global_shift"]
        else:
            chosen = shift_prompt(mins, maxs, suggest_global_shift(mins, maxs))
            workspace.remember(
                self.path,
                global_shift=None if chosen is None else [float(c) for c in chosen],
            )
        return None if chosen is None else np.asarray(chosen, dtype=np.float64)

    def _shift_and_cast(self, raw: np.ndarray) -> np.ndarray:
        if self.global_shift is not None:
            raw = raw + self.global_shift
        return raw.astype(np.float32)

    # -- format hooks (subclass) -----------------------------------------
    def _open(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _decode_coords_raw(self, sub) -> np.ndarray:  # pragma: no cover - abstract
        """Full-precision (float64) coordinates, before any global shift."""
        raise NotImplementedError

    def _decode_coords(self, sub) -> np.ndarray:
        """Shifted, float32 coordinates for a subset — used by :meth:`load`,
        so a loaded tree's points line up with the resident ``self.coords``."""
        return self._shift_and_cast(self._decode_coords_raw(sub))

    def _decode_labels(self, sub):  # pragma: no cover - abstract
        raise NotImplementedError

    def _write_labels(self, out, changed: np.ndarray, values: np.ndarray) -> None:
        raise NotImplementedError  # pragma: no cover - abstract

    def _raw_label_codes(self, sub) -> np.ndarray:
        """The label each record carries *as the file stores it*, as integers.

        Used only to group points by which tree they belonged to before this
        session's edits (see :meth:`_expand_to_full`), so the raw value is
        what's wanted — no folding of sentinels, no renumbering, and for an
        RGB-segmented PLY the colour itself.
        """
        return np.asarray(sub[self.label_field]).astype(np.int64)

    def _finalize_save(self, target: str, is_new_target: bool) -> None:
        """Hook after the in-place patch is flushed (e.g. re-export LAZ)."""

    # -- export ------------------------------------------------------------
    def file_labels(self) -> np.ndarray:
        """Every point's label as the *file* currently holds it: full
        resolution, and without edits that haven't been saved yet.

        :attr:`labels` is the working set instead — the decimated subset on a
        downsampled session, carrying unsaved edits. Export writes whole
        trees out of the file, so it asks the file.
        """
        labels, _colors = self._decode_labels(self._mm)
        return np.asarray(labels).astype(np.int32)

    def has_unsaved_edits(self) -> bool:
        """Whether any label or class has changed since the file was last
        written."""
        return (bool(np.any(self.labels != self._original_labels))
                or self.has_unsaved_class_edits())

    def subset_writer(self):
        """A context manager yielding ``write(rows, dest)``, which writes the
        file rows ``rows`` to ``dest`` as a file of the same format, keeping
        every field and the file's own (unshifted) coordinates.

        Whatever is expensive to set up — the PLY header, laspy's read of a
        LAS — is done once for the whole ``with`` block, so exporting a
        thousand trees reads the cloud once, not a thousand times.
        """
        raise NotImplementedError  # pragma: no cover - abstract

    # -- grouping index --------------------------------------------------
    def _build_index(self) -> None:
        """Sort-by-label once so any tree's point indices are an O(1) slice.

        Rebuilt (O(N log N)) after every :meth:`apply` — that's only on a
        tree switch or Save, never per-render, so it's not on the hot path
        that made loading the whole cloud slow.
        """
        self.order = _label_order(self.labels)
        sorted_labels = self.labels[self.order]
        # Run starts, not np.unique: `sorted_labels` is sorted by
        # construction, and np.unique would sort all N labels a second time
        # to work that out.
        if sorted_labels.size:
            first = np.empty(sorted_labels.size, dtype=bool)
            first[0] = True
            np.not_equal(sorted_labels[1:], sorted_labels[:-1], out=first[1:])
            starts = np.flatnonzero(first)
            uniq = sorted_labels[starts]
            counts = np.diff(np.append(starts, sorted_labels.size))
        else:
            starts = uniq = counts = np.empty(0, dtype=np.int64)
        self._starts = dict(zip(uniq.tolist(), starts.tolist()))
        self._counts = dict(zip(uniq.tolist(), counts.tolist()))

        # Per-tree bbox/centroid in two vectorised reductions over the
        # label-sorted coords, instead of a fancy-index + min/max per tree —
        # this runs on every apply() (tree switch, Save), so the Python loop
        # it replaces was O(trees) numpy calls on multi-thousand-tree plots.
        self.records: dict[int, TreeRecord] = {}
        if not uniq.size:
            return
        # One axis at a time: the label-sorted gather is the widest
        # temporary here, and three (N,) columns in turn peak at a third of
        # the (N, 3) copy — 2GB rather than 6GB on a 480M-point plot.
        mins = np.empty((starts.size, 3), dtype=self.coords.dtype)
        maxs = np.empty_like(mins)
        for axis in range(3):
            column = self.coords[:, axis][self.order]
            np.minimum.reduceat(column, starts, out=mins[:, axis])
            np.maximum.reduceat(column, starts, out=maxs[:, axis])
            del column
        for k, lab in enumerate(uniq.tolist()):
            if lab in (UNASSIGNED, NOISE):
                continue
            lo, hi = mins[k], maxs[k]
            self.records[lab] = TreeRecord(
                label=lab,
                count=int(counts[k]),
                centroid=(float((lo[0] + hi[0]) / 2),
                          float((lo[1] + hi[1]) / 2)),
                bbox=(lo, hi),
            )

    def indices_for(self, label: int) -> np.ndarray:
        start = self._starts.get(label)
        if start is None:
            return np.empty(0, dtype=np.int64)
        return self.order[start:start + self._counts[label]]

    # -- neighbour search --------------------------------------------------
    def neighbours(self, label: int, reach: float) -> set[int]:
        """Trees whose points come within ``reach`` of tree ``label``.

        Delegates to :func:`analysis.neighbours_by_points` against the
        resident (not memory-mapped) coords/labels — same algorithm the
        review panel's own neighbour picker uses, just pointed at the whole
        file's data instead of whatever's currently loaded.
        """
        from . import analysis

        tmp = PointCloud(coords=self.coords, labels=self.labels)
        return analysis.neighbours_by_points(tmp, label, reach)

    # -- loading -----------------------------------------------------------
    def load(self, labels: list[int], margin: float = 1.0) -> tuple[PointCloud, np.ndarray]:
        """Load ``labels`` plus nearby unassigned points into a real cloud.

        Unassigned points near the loaded trees are included (not just the
        labelled ones) because the review workflow relies on lassoing them
        and pressing A to add them to the current tree.
        """
        from . import density

        labels = [int(t) for t in labels]
        parts = [self.indices_for(t) for t in labels]
        tree_idx = (
            np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        )
        if tree_idx.size == 0:
            raise ValueError("No points for the requested trees")

        pts = self.coords[tree_idx]
        lo, hi = pts.min(axis=0) - margin, pts.max(axis=0) + margin
        # density.in_box, not `np.all((coords >= lo) & (coords <= hi), 1)`:
        # self.coords is the whole catalogue, so the latter builds three
        # (N, 3) boolean temporaries — 1.8GB of churn per tree load on a
        # 200M-point file, for a mask that fits in 200MB.
        near = density.in_box(self.coords, lo, hi)
        near &= self.labels == UNASSIGNED
        # Union through a mask rather than np.union1d, which concatenates and
        # sorts: the trees' own rows plus the unassigned ones run to millions
        # on a plot-sized file, and sorting them was two thirds of the time to
        # open a tree. Marking a mask and reading it back is one pass, and
        # comes out ascending and duplicate-free just the same.
        picked = np.zeros(len(self.labels), dtype=bool)
        picked[tree_idx] = True
        picked |= near
        global_idx = np.flatnonzero(picked)

        sub = self._mm[self._rows(global_idx)]
        coords = self._decode_coords(sub)
        cloud_labels = self.labels[global_idx].copy()

        skip = {self._names.get(n) for n in ("x", "y", "z")}
        if self.is_rgb:
            skip |= {self._names.get(n) for n in ("red", "green", "blue")}
        else:
            skip.add(self.label_field)
        if self.class_field is not None:
            skip.add(self._class_col[0])
        attributes = {
            name: np.ascontiguousarray(sub[name])
            for name in self.dtype.names
            if name not in skip
        }

        cloud = PointCloud(
            coords=coords,
            labels=cloud_labels,
            attributes=attributes,
            source_path=self.path,
            label_field=self.label_field or "treeID",
            source_format="raycloud_rgb" if self.is_rgb else "auto",
            label_colors=self.label_colors,
            global_shift=(
                tuple(self.global_shift.tolist())
                if self.global_shift is not None else None
            ),
            classes=(
                self.classes[global_idx].copy()
                if self.classes is not None else None
            ),
            class_field=self.class_field,
        )
        # New tree IDs (splits, grow) must be unique across the *whole* file,
        # not just this subset — see next_free_id(). This mirrors the
        # instance-level override slot SegFixController.on_save_override
        # already uses elsewhere in this codebase.
        cloud.next_free_id = self.next_free_id
        return cloud, global_idx

    def next_free_id(self) -> int:
        """A session-wide-unique tree ID, unlike PointCloud's own (subset-
        scoped) implementation. Stateful/incrementing by design: repeated
        calls before an intervening apply() must still not collide."""
        nid = self._next_id
        self._next_id += 1
        return nid

    # -- persisting edits ----------------------------------------------
    def apply(self, cloud: PointCloud, global_idx: np.ndarray) -> None:
        """Scatter a loaded (and possibly edited) subset's labels back into
        the master array, so the catalog and any later `load()` reflect it."""
        if global_idx is None or global_idx.size == 0:
            return
        self.labels[global_idx] = cloud.labels
        if self.classes is not None and cloud.classes is not None:
            self.classes[global_idx] = cloud.classes
        self._build_index()
        if self.labels.size:
            self._next_id = max(self._next_id, int(self.labels.max()) + 1)

    # -- saving --------------------------------------------------------
    def save(
        self, output: str | None = None, progress: ProgressFn | None = None
    ) -> str:
        """Write only the points whose label changed since the last save.

        Diffs against the snapshot taken at open time / after the previous
        save, so this reflects edits made across *any* tree visited this
        session, not just whatever is currently loaded.

        On a decimated session the diff covers the kept points only, so it is
        first interpolated back onto every full-resolution point it stands
        for — see :meth:`_expand_to_full`. ``progress`` (:data:`ProgressFn`)
        is called through that pass and the write, which on a big file are
        the parts worth watching.
        """
        changed = np.flatnonzero(self.labels != self._original_labels)
        class_changed = (
            np.flatnonzero(self.classes != self._original_classes)
            if self.classes is not None else np.empty(0, dtype=np.int64)
        )
        target = output or self.path
        is_new_target = os.path.abspath(target) != os.path.abspath(self.path)

        if not is_new_target and changed.size == 0 and class_changed.size == 0:
            return "Nothing changed since last save"

        if is_new_target and not os.path.exists(target):
            # A full copy of the whole file, not just whatever's loaded —
            # needed even with zero edits so "Save As" still exports a
            # complete copy at the new path, not nothing.
            if progress is not None:
                progress(f"Copying to {os.path.basename(target)}…", 0.0)
            shutil.copyfile(self.path, target)

        if changed.size == 0 and class_changed.size == 0:
            self._finalize_save(target, is_new_target)
            return f"Saved (no changes) → {target}"

        # The interpolation passes share the bar up to the write, the label
        # pass first; a pass with nothing to do takes no share of it.
        split = (_SAVE_WRITE_AT * changed.size / (changed.size + class_changed.size)
                 if class_changed.size and changed.size else
                 (_SAVE_WRITE_AT if changed.size else 0.0))
        empty = np.empty(0, dtype=np.int64)
        rows, values = (
            self._changed_rows(changed, progress, (0.0, split))
            if changed.size else (empty, self.labels[empty])
        )
        class_rows, class_values = (
            self._changed_class_rows(class_changed, progress, (split, _SAVE_WRITE_AT))
            if class_changed.size else (empty, empty)
        )
        if progress is not None:
            progress(f"Writing {rows.size + class_rows.size:,} points…",
                     _SAVE_WRITE_AT)
        out = np.memmap(target, dtype=self.dtype, mode="r+",
                         offset=self.offset, shape=(self.count,))
        if rows.size:
            self._write_labels(out, rows, values)
        if class_rows.size:
            self._write_classes(out, class_rows, class_values)
        out.flush()
        del out
        if progress is not None:
            progress("Finishing…", _SAVE_FINALIZE_AT)
        self._finalize_save(target, is_new_target)

        if not is_new_target:
            # Only a write to self.path advances the diff baseline — a
            # Save As to a different file doesn't touch self.path, so an
            # in-place Save afterwards must still see these as pending.
            self._original_labels = self.labels.copy()
            if self.classes is not None:
                self._original_classes = self.classes.copy()
        if not class_rows.size:
            return f"Saved {rows.size:,} changed point(s) → {target}"
        if not rows.size:
            return f"Saved {class_rows.size:,} reclassified point(s) → {target}"
        return (f"Saved {rows.size:,} changed point(s) and "
                f"{class_rows.size:,} reclassified → {target}")

    def _changed_rows(
        self, changed: np.ndarray, progress: ProgressFn | None = None,
        span: tuple[float, float] = (0.0, _SAVE_WRITE_AT),
    ):
        """``(file_rows, new_labels)`` to patch for the changed working-set
        positions ``changed`` — the positions themselves at full resolution,
        their full-resolution neighbourhoods when decimated."""
        if self._sub_idx is None:
            return changed, self.labels[changed]
        return self._expand_to_full(changed, progress, span)

    def _changed_class_rows(
        self, changed: np.ndarray, progress: ProgressFn | None = None,
        span: tuple[float, float] = (0.0, _SAVE_WRITE_AT),
    ):
        """:meth:`_changed_rows` for the point classes. A decimated session
        interpolates them the same way, matching each point to the nearest
        working point of the same tree *and* class."""
        if self._sub_idx is None:
            return changed, self.classes[changed]
        return self._expand_to_full(
            changed, progress, span,
            values=self.classes, original=self._original_classes,
            codes_of=self._tree_class_codes, trees_only=False,
        )

    def _expand_to_full(
        self, changed: np.ndarray, progress: ProgressFn | None = None,
        span: tuple[float, float] = (0.0, _SAVE_WRITE_AT),
        values: np.ndarray | None = None,
        original: np.ndarray | None = None,
        codes_of=None,
        trees_only: bool = True,
    ):
        """Interpolate a decimated session's edits back onto every point.

        Each full-resolution point follows the nearest working point that was
        the *same tree as it* (:func:`~segfix.density.class_trees` explains
        why same-tree and not simply nearest), and only where that working
        point is one the user re-labelled — so parts of the cloud nobody
        touched keep the file's original labels exactly, down to the last
        point, and a save stays a diff rather than a rewrite.

        The scan is chunked over the memory-mapped records and the matching
        skipped entirely outside the bounding box of the edits, so its cost
        tracks the size of what was edited, not the size of the file.

        One exception to nearest-point matching: a tree moved *whole* (X or
        U on the current tree, a complete merge -- see
        :func:`~segfix.density.whole_class_moves`) takes every one of its
        points with it, wherever they are in the file, including any the
        working set never showed because another class's point was kept for
        their voxel.

        ``values``, ``original`` and ``codes_of`` default to the tree labels;
        the point classes pass their own (see :meth:`_changed_class_rows`).
        """
        from .density import class_trees, expand_labels, in_box, whole_class_moves

        if values is None:
            values, original = self.labels, self._original_labels
        if codes_of is None:
            codes_of = self._raw_label_codes
        lo_f, hi_f = span

        # Which full-resolution points are worth looking at: every occupied
        # voxel keeps a point, so no point sits further than one voxel
        # diagonal (~1.73 x voxel) from the nearest working point, and
        # anything more than two voxels outside the edited points' own box
        # is therefore nearest to a working point that didn't change.
        voxel = self.voxel_size or 0.0
        reach = 2.0 * voxel
        box_lo = self.coords[changed].min(axis=0)
        box_hi = self.coords[changed].max(axis=0)
        lo, hi = box_lo - reach, box_hi + reach
        # Unchanged working points are candidates too: they are what stops an
        # edit bleeding across a boundary into a tree the user left alone.
        # Their box is wider again by the same diagonal, so a query point at
        # the very edge can still be matched to the point that really is
        # nearest to it.
        cand = np.flatnonzero(
            in_box(self.coords, box_lo - 2 * reach, box_hi + 2 * reach)
        )
        codes = codes_of(self._mm[self._sub_idx])
        trees = class_trees(self.coords, codes, cand)
        whole = whole_class_moves(codes, original, values, trees_only=trees_only)
        whole_codes = np.array(sorted(whole), dtype=np.int64)
        whole_labels = np.array([whole[k] for k in whole_codes],
                                dtype=values.dtype)

        rows: list[np.ndarray] = []
        new_values: list[np.ndarray] = []
        for start in range(0, self.count, _EXPAND_CHUNK):
            stop = min(start + _EXPAND_CHUNK, self.count)
            if progress is not None:
                progress(
                    "Interpolating edits back to full resolution…",
                    lo_f + (hi_f - lo_f) * start / max(self.count, 1),
                )
            block = self._mm[start:stop]
            block_codes = codes_of(block)

            # Whole-tree moves: by label alone, anywhere in the chunk -- a
            # stranded point can sit outside the box the edits span.
            is_whole = (np.isin(block_codes, whole_codes) if whole_codes.size
                        else np.zeros(block_codes.size, dtype=bool))
            if is_whole.any():
                w = np.flatnonzero(is_whole)
                rows.append(start + w)
                new_values.append(whole_labels[np.searchsorted(whole_codes, block_codes[w])])

            coords = self._decode_coords(block)
            inside = np.flatnonzero(in_box(coords, lo, hi) & ~is_whole)
            if not inside.size:
                continue
            local, source = expand_labels(
                trees,
                coords[inside],
                block_codes[inside],
                changed,
                values.size,
                max_distance=reach,
            )
            if not local.size:
                continue
            rows.append(start + inside[local])
            new_values.append(values[source])

        if not rows:
            return (np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=values.dtype))
        return np.concatenate(rows), np.concatenate(new_values)


class TreeCatalog(_BaseCatalog):
    """A memory-mapped binary PLY, grouped by tree label.

    The instance label lives either in a per-point field (``treeID``,
    ``PredInstance``, ... — auto-detected) or, for raycloudtools clouds, in the
    point's RGB colour.
    """

    def _open(self) -> None:
        # Fails in microseconds on a wrong/mismatched file, instead of
        # _parse_ply_header stalling — potentially for minutes on a huge
        # file — scanning arbitrary binary content line by line looking for
        # a header that isn't there. See io.sniff_format.
        if io.sniff_format(self.path) != "ply":
            raise ValueError(
                f"{self.path} doesn't look like a binary PLY (bad header) - "
                "wrong file, or an ASCII PLY?"
            )
        with open(self.path, "rb") as fh:
            fmt, props, count, offset = io._parse_ply_header(fh)
        if "binary" not in fmt:
            raise ValueError("Tree catalog requires a binary PLY")

        self.offset = offset
        self.count = count
        byteorder = ">" if fmt == "binary_big_endian" else "<"
        self.dtype = np.dtype([
            (name, byteorder + io.PLY_TYPE_MAP.get(ptype.lower(), "f4"))
            for name, ptype in props
        ])
        self._names = {n.lower(): n for n in self.dtype.names}
        self.is_rgb = io._is_rgb_segmented(props, self._label_field_req)
        self.label_field = (
            None if self.is_rgb
            else io._pick_label_field(list(self.dtype.names), self._label_field_req)
        )
        self._mm = np.memmap(self.path, dtype=self.dtype, mode="r",
                              offset=self.offset, shape=(self.count,))

    def _decode_coords_raw(self, sub) -> np.ndarray:
        return np.column_stack([
            sub[self._names["x"]],
            sub[self._names["y"]],
            sub[self._names["z"]],
        ]).astype(np.float64)

    def _decode_labels(self, sub):
        if self.is_rgb:
            return self._labels_from_rgb(sub)
        return io._normalize_labels(sub[self.label_field]), None

    def _labels_from_rgb(self, sub):
        r = sub[self._names["red"]].astype(np.uint32)
        g = sub[self._names["green"]].astype(np.uint32)
        b = sub[self._names["blue"]].astype(np.uint32)
        packed = (r << 16) | (g << 8) | b
        uniq, inverse = np.unique(packed, return_inverse=True)
        labels = (inverse + 1).astype(np.int32)
        labels[packed == 0] = UNASSIGNED
        label_colors = {
            int(k + 1): (int((u >> 16) & 255), int((u >> 8) & 255), int(u & 255))
            for k, u in enumerate(uniq)
            if u != 0
        }
        return labels, label_colors

    def _write_labels(self, out, changed: np.ndarray, values: np.ndarray) -> None:
        if self.is_rgb:
            colour = self._colours_for(values)
            out[self._names["red"]][changed] = colour[:, 0]
            out[self._names["green"]][changed] = colour[:, 1]
            out[self._names["blue"]][changed] = colour[:, 2]
        else:
            out[self.label_field][changed] = values

    def _raw_label_codes(self, sub) -> np.ndarray:
        if not self.is_rgb:
            return super()._raw_label_codes(sub)
        return (
            (sub[self._names["red"]].astype(np.int64) << 16)
            | (sub[self._names["green"]].astype(np.int64) << 8)
            | sub[self._names["blue"]].astype(np.int64)
        )

    @contextmanager
    def subset_writer(self):
        with open(self.path, "rb") as fh:
            header = fh.read(self.offset)
        _reject_extra_ply_elements(header)

        def write(rows: np.ndarray, dest: str) -> None:
            count = str(len(rows)).encode()
            out_header = re.sub(
                rb"(element[ \t]+vertex[ \t]+)\d+",
                lambda m: m.group(1) + count, header, count=1,
            )
            with open(dest, "wb") as out:
                out.write(out_header)
                out.write(np.ascontiguousarray(self._mm[rows]).tobytes())

        yield write

    def _colours_for(self, labels: np.ndarray) -> np.ndarray:
        colour = np.zeros((labels.size, 3), dtype=np.uint8)
        for lab in np.unique(labels):
            if lab in (UNASSIGNED, NOISE):
                continue
            colour[labels == lab] = io.color_for_label(int(lab), self.label_colors)
        return colour


class LasCatalog(_BaseCatalog):
    """A memory-mapped uncompressed LAS, grouped by its ``treeID`` column.

    Reads the header and Extra-Bytes layout with laspy but never lets laspy
    decode the points: the fixed-size records are indexed and patched with
    ``numpy.memmap``, exactly like the PLY path. On save the in-place ``.las``
    patch is mirrored to a ``.laz`` beside it when the project came from one
    (see :meth:`_finalize_save`), so arbor can re-read the corrected cloud.
    """

    is_rgb = False  # LAS always carries an explicit label column

    def _open(self) -> None:
        import laspy

        ext = os.path.splitext(self.path)[1].lower()
        if ext == ".laz":
            raise ValueError(
                "LasCatalog needs an uncompressed .las; compressed .laz cannot "
                "be memory-mapped (workspace.create_workspace decompresses on "
                "import)"
            )
        if io.sniff_format(self.path) != "las":
            raise ValueError(
                f"{self.path} doesn't look like a LAS file (bad header) - "
                "wrong file, or actually compressed .laz?"
            )

        with laspy.open(self.path) as fh:
            header = fh.header
        self.dtype = header.point_format.dtype()
        self.offset = int(header.offset_to_point_data)
        self.count = int(header.point_count)
        self._scales = np.asarray(header.scales, dtype=np.float64)
        self._offsets = np.asarray(header.offsets, dtype=np.float64)
        self._names = {n.lower(): n for n in self.dtype.names}

        expected = self.offset + self.count * self.dtype.itemsize
        actual = os.path.getsize(self.path)
        if self.dtype.itemsize != header.point_format.size or actual < expected:
            raise ValueError(
                f"{self.path}: LAS point records don't line up "
                f"(record {self.dtype.itemsize}B x {self.count} + {self.offset}B "
                f"header = {expected}B, file is {actual}B) - not a plain "
                "uncompressed LAS?"
            )

        self.label_field = io._pick_label_field(
            list(self.dtype.names), self._label_field_req
        )
        if self.label_field is None:
            raise ValueError(
                f"{self.path}: no treeID / instance-ID column "
                f"(looked for {', '.join(io.LABEL_CANDIDATES)}). "
                "Is this an arbor '_segmented' cloud? Pass --label-field to "
                "name it explicitly."
            )
        self._label_is_unsigned = self.dtype[self.label_field].kind == "u"

        self._mm = np.memmap(self.path, dtype=self.dtype, mode="r",
                              offset=self.offset, shape=(self.count,))

    def _decode_coords_raw(self, sub) -> np.ndarray:
        return np.column_stack([
            sub[self._names["x"]].astype(np.float64) * self._scales[0] + self._offsets[0],
            sub[self._names["y"]].astype(np.float64) * self._scales[1] + self._offsets[1],
            sub[self._names["z"]].astype(np.float64) * self._scales[2] + self._offsets[2],
        ])

    def _decode_labels(self, sub):
        return io._normalize_labels(sub[self.label_field]), None

    def _write_labels(self, out, changed: np.ndarray, values: np.ndarray) -> None:
        if self._label_is_unsigned:
            # An unsigned Extra-Bytes column can't hold NOISE (-1); write it
            # back as UNASSIGNED (0). segfix's own <cloud>.segfix.json sidecar
            # still remembers which points were dismissed as noise, but a
            # reader of the LAS alone (arbor, lidR) sees them as unassigned.
            values = np.where(values == NOISE, UNASSIGNED, values)
        out[self.label_field][changed] = values.astype(self.dtype[self.label_field])

    @contextmanager
    def subset_writer(self):
        import laspy

        las = laspy.read(self.path)

        def write(rows: np.ndarray, dest: str) -> None:
            # laspy recomputes the point count and the extents from the
            # points it is given, and carries the header's scales, offsets,
            # CRS and Extra-Bytes layout over unchanged.
            laspy.LasData(header=las.header, points=las.points[rows]).write(dest)

        try:
            yield write
        finally:
            del las

    def _finalize_save(self, target: str, is_new_target: bool) -> None:
        laz = _laz_export_path(self.path, target)
        if laz is None:
            return
        import laspy

        laspy.read(target).write(laz)


def _reject_extra_ply_elements(header: bytes) -> None:
    """Refuse a PLY whose header declares more than the vertex element (a
    mesh's faces, say): only vertex rows are copied, so the exported file's
    header would promise data that isn't there."""
    for line in header.splitlines():
        parts = line.decode("ascii", "replace").split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] != "vertex":
            if int(parts[2]) > 0:
                raise ValueError(
                    f"this PLY also contains '{parts[1]}' data, which a "
                    "per-tree export can't carry"
                )


def _laz_export_path(source_las: str, target_las: str) -> str | None:
    """Where to re-export a corrected LAS as LAZ, or ``None`` if not needed.

    Returns a sibling ``.laz`` only when the project's working copy was
    decompressed from one on import — recorded as a ``.laz`` ``source`` in the
    :mod:`workspace` manifest next to the ``.las``.
    """
    from . import workspace

    manifest = os.path.join(os.path.dirname(source_las), workspace.MANIFEST_NAME)
    if not os.path.exists(manifest):
        return None
    try:
        import json

        with open(manifest, encoding="utf-8") as fh:
            src = json.load(fh).get("source", "")
    except (OSError, ValueError):
        return None
    if os.path.splitext(src)[1].lower() != ".laz":
        return None
    return os.path.splitext(target_las)[0] + ".laz"


def open_catalog(
    path: str,
    label_field: str | None = None,
    shift_prompt: ShiftPrompt | None = None,
    density_prompt: DensityPrompt | None = None,
    progress: ProgressFn | None = None,
    class_field: str | None = None,
) -> _BaseCatalog:
    """Open ``path`` with the backend its extension calls for.

    ``.ply`` → :class:`TreeCatalog`; ``.las`` → :class:`LasCatalog`. ``.laz`` is
    rejected here — :mod:`workspace` decompresses it to ``.las`` on import.

    ``shift_prompt``, if given, is offered a chance to apply a "global shift"
    (see :func:`needs_global_shift`) when the cloud's coordinates are large
    enough that float32 storage would start losing precision; omit it (the
    default) to always load coordinates unshifted, exactly as before.

    ``density_prompt``, likewise, is offered a chance to voxel-decimate a
    cloud finer than :data:`segfix.density.DENSE_SPACING` for the session
    (edits are interpolated back onto every point on save); omit it and the
    whole cloud is reviewed at full resolution, exactly as before.

    ``progress`` is called as the load moves between phases — see
    :data:`ProgressFn` and :mod:`segfix.progress_ui`.

    ``class_field`` names a per-point point-class column (leaf, wood, ground
    ... as numbers) to load for editing alongside the tree labels; see
    :meth:`_BaseCatalog.set_class_field`.
    """
    ext = os.path.splitext(path)[1].lower()
    kwargs = dict(
        label_field=label_field,
        shift_prompt=shift_prompt,
        density_prompt=density_prompt,
        progress=progress,
        class_field=class_field,
    )
    if ext == ".ply":
        return TreeCatalog(path, **kwargs)
    if ext in (".las", ".laz"):
        return LasCatalog(path, **kwargs)
    raise ValueError(f"segfix opens .ply and .las clouds, not {ext or path!r}")


# Backwards-compatible alias for type hints that just want "some catalog".
Catalog = _BaseCatalog
