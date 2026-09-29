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

"""Label→colour mapping and visibility maths for tree point clouds.

Each tree instance gets a stable, distinct colour derived from its ID (so the
same tree keeps its colour across edits), while the special UNASSIGNED and
NOISE labels get muted greys that read as "not a tree".

The 3D canvas itself lives in :mod:`segfix.cloudview`; this module is just the
UI-agnostic colour/mask helpers it and the panel share.
"""

from __future__ import annotations

import numpy as np

from .model import NOISE, UNASSIGNED, PointCloud


def busy(view, message: str) -> None:
    """Show a status message and force it to paint immediately.

    Setting ``view.status`` only queues a repaint — it won't actually appear
    until control returns to the Qt event loop, which is too late if the very
    next line is a blocking load/save. Call this right before such a call so
    there's visible feedback while it runs.
    """
    view.status = message
    from qtpy.QtWidgets import QApplication

    QApplication.processEvents()


def gpu_renderer_info() -> str | None:
    """Best-effort OpenGL renderer string for the active GPU context (e.g.
    "NVIDIA RTX A1000 Laptop GPU" vs. a software/Mesa renderer) — lets a user
    confirm whether a GPU-offload env var actually took effect, without
    shelling out to ``glxinfo``. ``None`` if it can't be determined (context
    not ready yet, PyOpenGL missing, ...).

    Deliberately does *not* call ``makeCurrent()`` — vispy already has its own
    context current by the time the canvas has drawn once, and forcing it
    again (bypassing vispy's context tracking) makes the next draw call crash
    with "Attempt to retrieve context when no valid context". Just read
    whatever context vispy has already made current.

    Asks vispy before PyOpenGL. Both read the same GL_RENDERER off the same
    current context, but vispy is a hard dependency and PyOpenGL is not one
    at all — it was only ever present because something else happened to
    pull it in. A packaged build has no reason to carry it, so the PyOpenGL
    route reported "unknown" on every machine it shipped to.
    """
    try:
        from vispy.gloo import gl

        renderer = gl.glGetParameter(gl.GL_RENDERER)
        if renderer:
            return renderer if isinstance(renderer, str) else renderer.decode()
    except Exception:
        pass
    try:
        from OpenGL import GL

        renderer = GL.glGetString(GL.GL_RENDERER)
        return renderer.decode() if renderer else None
    except Exception:
        return None
    return None


def gpu_status(renderer: str | None, tries_left: int) -> tuple[str | None, bool]:
    """What the GPU status label should say, and whether to stop asking.

    Returns ``(text, done)``. ``text`` of ``None`` means no answer yet —
    leave the label on "detecting…" and ask again on the next draw. The
    label used to be written from the first draw whatever came back, so a
    context that was not ready at that instant (the machine waking from
    sleep, a driver reset, a hybrid-graphics switch) pinned it to "unknown"
    for the rest of the session.
    """
    if renderer:
        return f"GPU: {renderer}", True
    if tries_left > 0:
        return None, False
    return "GPU: unknown", True


# Muted, recessive greys: unassigned/noise points (whole ground + understory
# on a big plot) are context and shouldn't compete with the tree colours.
UNASSIGNED_COLOR = np.array([0.4, 0.41, 0.43, 1.0], dtype=np.float32)
NOISE_COLOR = np.array([0.25, 0.25, 0.27, 1.0], dtype=np.float32)

# Opacity for trees the panel has "faded": ghosted for context but, unlike a
# hidden tree, still rendered and still selectable by the lasso.
FADED_ALPHA = 0.32


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised HSV→RGB; ``h``, ``s`` and ``v`` are arrays (or scalars)
    in [0, 1) that broadcast together."""
    h, s, v = np.broadcast_arrays(
        np.asarray(h, float), np.asarray(s, float), np.asarray(v, float)
    )
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


# Irrational, mutually unrelated strides. Multiplying an integer tree ID by
# each and taking the fractional part spreads a set of consecutive *or*
# arbitrary IDs evenly along that axis.
_HUE_STRIDE = 0.61803398875   # 1/φ
_SAT_STRIDE = 0.41421356237   # √2 − 1
_VAL_STRIDE = 0.73205080757   # √3 − 1


def _hashed_sv(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-ID saturation and value, hashed across a narrow always-vivid band
    (never muddy) so two trees that land on a near-identical hue still pull
    apart in saturation or lightness."""
    sat = 0.70 + 0.28 * np.mod(ids * _SAT_STRIDE, 1.0)   # 0.70 – 0.98
    val = 0.80 + 0.20 * np.mod(ids * _VAL_STRIDE, 1.0)   # 0.80 – 1.00
    return sat, val


def colors_for_labels(labels: np.ndarray, label_colors=None, faded=None) -> np.ndarray:
    """Return an ``(N, 4)`` RGBA array colouring points by their tree ID.

    When ``label_colors`` is given (e.g. an RGB-segmented source), each label
    keeps its original hue (brightened for the dark canvas); labels without
    an entry fall back to a hashed hue.  Without ``label_colors`` everything
    is hashed.  UNASSIGNED and NOISE always get fixed muted greys.

    ``faded`` is an optional iterable of tree IDs whose points get
    :data:`FADED_ALPHA` in place of full opacity — the panel's per-tree "fade"
    control (see :meth:`SegFixWidget._set_faded`).
    """
    labels = np.asarray(labels)
    colors = np.empty((len(labels), 4), dtype=np.float32)

    is_unassigned = labels == UNASSIGNED
    is_noise = labels == NOISE
    is_tree = ~(is_unassigned | is_noise)

    colors[is_unassigned] = UNASSIGNED_COLOR
    colors[is_noise] = NOISE_COLOR

    if is_tree.any():
        ids = labels[is_tree].astype(np.int64)
        # Compute one colour per distinct label, then scatter to points via
        # searchsorted — keeps the work O(N log K), not O(N*K), on big clouds.
        uniq = np.unique(ids)
        uniq_f = uniq.astype(np.float64)
        # Hue from the golden-ratio hash; saturation and value from their own
        # hashes (see _hashed_sv) so the palette varies on all three axes.
        hue = np.mod(uniq_f * _HUE_STRIDE, 1.0)
        sat, val = _hashed_sv(uniq_f)
        if label_colors:
            # Keep a source-coloured tree in its own colour family, but nudge
            # its hue a little per-ID (so two trees the source painted alike
            # still diverge) and take the hashed, always-vivid S/V.
            import colorsys

            nudge = (np.mod(uniq_f * _VAL_STRIDE * 2.0, 1.0) - 0.5) * 0.10
            for k, lab in enumerate(uniq):
                src = label_colors.get(int(lab))
                if src is not None:
                    src_h, _s, _v = colorsys.rgb_to_hsv(
                        src[0] / 255, src[1] / 255, src[2] / 255
                    )
                    hue[k] = (src_h + nudge[k]) % 1.0
        per_label = _hsv_to_rgb(hue, sat, val).astype(np.float32)
        pos = np.searchsorted(uniq, ids)
        colors[is_tree, :3] = per_label[pos]
        colors[is_tree, 3] = 1.0

    if faded is not None:
        faded = list(faded)
        if faded:
            colors[np.isin(labels, faded), 3] = FADED_ALPHA
    return colors


# -- point classes -------------------------------------------------------
# Colours for the names people actually give their classes, so a cloud
# coloured by class reads at a glance: wood brown, leaves green, ground tan.
# Matched on the lower-cased name's start, first hit wins.
_NAMED_CLASS_COLOURS = (
    (("wood", "stem", "trunk", "branch", "bark"), (0.55, 0.33, 0.16)),
    (("leaf", "leaves", "foliage", "canopy", "crown"), (0.20, 0.70, 0.25)),
    (("under", "shrub", "bush", "low veg"), (0.62, 0.82, 0.25)),
    (("ground", "terrain", "soil"), (0.78, 0.66, 0.45)),
    (("noise", "outlier"), (0.35, 0.35, 0.38)),
    (("unclass", "never", "unknown"), (0.60, 0.61, 0.63)),
    (("build", "structure", "wall", "man"), (0.80, 0.30, 0.30)),
    (("water",), (0.25, 0.50, 0.85)),
)

# Distinct, fairly saturated colours for everything else, in order — the
# Okabe-Ito set minus black, then Tableau's extras.
_CLASS_PALETTE = (
    (0.90, 0.62, 0.00), (0.34, 0.71, 0.91), (0.00, 0.62, 0.45),
    (0.94, 0.89, 0.26), (0.00, 0.45, 0.70), (0.84, 0.37, 0.00),
    (0.80, 0.47, 0.65), (0.58, 0.40, 0.74), (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76), (0.74, 0.74, 0.13), (0.09, 0.75, 0.81),
)


def class_colors(names: dict[int, str]) -> dict[int, tuple[float, float, float]]:
    """``{code: (r, g, b)}`` for a class mapping ``{code: name}``.

    A recognised name gets its natural colour; the rest take the palette in
    code order, skipping colours a named class already has. Codes only ever
    get added at the top (a new class is ``max + 1``), so adding one never
    recolours the classes already on screen.
    """
    out: dict[int, tuple[float, float, float]] = {}
    for code, name in names.items():
        low = name.strip().lower()
        for prefixes, rgb in _NAMED_CLASS_COLOURS:
            if low.startswith(prefixes):
                out[code] = rgb
                break
    free = [c for c in _CLASS_PALETTE if c not in out.values()]
    k = 0
    for code in sorted(names):
        if code in out:
            continue
        if k < len(free):
            out[code] = free[k]
        else:
            h = np.mod(code * _HUE_STRIDE, 1.0)
            out[code] = tuple(float(v) for v in _hsv_to_rgb(h, 0.75, 0.9))
        k += 1
    return out


def colors_for_classes(classes: np.ndarray, colours: dict, labels=None,
                       faded=None) -> np.ndarray:
    """``(N, 4)`` RGBA colouring points by their point class.

    A value with no entry in ``colours`` (not in the mapping) is drawn grey.
    ``labels`` + ``faded`` keep the per-tree fade working in class colours.
    """
    classes = np.asarray(classes)
    rgba = np.empty((len(classes), 4), dtype=np.float32)
    rgba[:] = UNASSIGNED_COLOR
    for code, rgb in colours.items():
        rgba[classes == code, :3] = rgb
    rgba[:, 3] = 1.0
    if faded is not None and labels is not None:
        faded = list(faded)
        if faded:
            rgba[np.isin(labels, faded), 3] = FADED_ALPHA
    return rgba


def point_colors(cloud: PointCloud, faded=None, class_colours=None,
                 rows=None) -> np.ndarray:
    """Colours for ``cloud`` (or just its ``rows``): by point class when
    ``class_colours`` is given and the cloud has classes, else by tree."""
    labels = np.asarray(cloud.labels)
    if rows is not None:
        labels = labels[rows]
    if class_colours is not None and cloud.classes is not None:
        classes = cloud.classes if rows is None else cloud.classes[rows]
        return colors_for_classes(classes, class_colours, labels, faded)
    return colors_for_labels(labels, cloud.label_colors, faded)


def visibility_mask(labels: np.ndarray, hide_unassigned=False,
                    hidden=None, cross_section=None):
    """Per-point ``shown`` mask, ANDing together every visibility filter.

    ``hide_unassigned`` drops the unassigned and noise points; ``hidden`` is
    a per-point boolean mask of individually hidden points (the table's hide
    checkboxes); ``cross_section`` is a per-point boolean mask of the region
    the section tools kept. Points left unshown are also unselectable — the
    lasso intersects its result with this same mask.
    """
    shown = np.ones(len(labels), dtype=bool)
    if hide_unassigned:
        shown &= (labels != UNASSIGNED) & (labels != NOISE)
    if hidden is not None and len(hidden) == len(labels):
        shown &= ~hidden
    if cross_section is not None and len(cross_section) == len(labels):
        shown &= cross_section
    return shown


def refresh_view(view, cloud: PointCloud, faded=None, changed=None,
                 class_colours=None) -> None:
    """Re-apply point colours to ``view`` after the labels have changed.

    ``faded`` (an iterable of tree IDs) keeps those trees ghosted through the
    refresh, so an edit doesn't silently un-fade them.

    ``changed`` is the indices whose label just moved (``cloud.last_changed``).
    When given non-empty, only those rows of the existing colour array are
    recomputed instead of the whole cloud — the common case on a per-edit
    keystroke. An empty ``changed`` means the op was a no-op, so nothing needs
    redrawing at all.

    ``class_colours`` (see :func:`class_colors`) colours by point class
    instead of by tree.
    """
    if changed is not None and len(changed) == 0:
        return
    fc = view.face_color
    if (
        changed is not None
        and isinstance(fc, np.ndarray)
        and fc.shape == (cloud.n_points, 4)
    ):
        fc = fc.copy()
        fc[changed] = point_colors(cloud, faded, class_colours, rows=changed)
    else:
        fc = point_colors(cloud, faded, class_colours)
    view.face_color = fc
