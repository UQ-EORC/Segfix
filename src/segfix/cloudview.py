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

"""The 3D point-cloud canvas, backed directly by vispy.

This is the single object the rest of the app talks to for anything on the
canvas: the point cloud, the camera, the freehand-lasso projection, the
current selection and the bounding box around the tree under review.  It
replaces what used to be a napari ``Viewer`` plus a ``Points`` layer, and
with it go all the napari workarounds that surrounded them (the
``_PointSliceRequest`` monkeypatch for ``shown`` in full 3D, the
``viewbox_mouse_event`` monkeypatch for CloudCompare-style mouse, ``strip_ui``,
and the ``_qt_viewer``/``_qt_window`` reach-throughs).

Coordinates are plain right-handed XYZ with Z up — the camera is told
``up='+z'`` and nothing is reordered, so ``coords[:, 2]`` is height
everywhere, no ``dims.order`` remap.
"""

from __future__ import annotations

import numpy as np
from vispy import scene
from vispy.scene.cameras import TurntableCamera
from vispy.util import keys


# Camera (azimuth, elevation) in degrees for the standard views, Z up. Top
# looks down with X right and Y up the screen; Front looks along +Y, Back
# along -Y, Left along +X, Right along -X. "3d" is the starting angle.
VIEWS = {
    "top": (0.0, 90.0),
    "front": (0.0, 0.0),
    "back": (180.0, 0.0),
    "left": (-90.0, 0.0),
    "right": (90.0, 0.0),
    "bottom": (0.0, -90.0),
    "3d": (30.0, 30.0),
}


#: The inside ("bubble") view: a wide lens, and the pivot a few centimetres
#: in front of the eye so the turntable turns your head instead of circling
#: the plot. 70° is about what a person sees as "normal" rather than
#: fisheyed; the distance is small enough to stand in a trunk and large
#: enough that the near clipping plane doesn't eat the scene.
INSIDE_FOV = 70.0
INSIDE_DISTANCE = 0.05


def selection_mask(indices, size: int, base: np.ndarray | None = None):
    """A boolean mask of ``indices`` over ``size`` points.

    ``indices`` may be point indices (an array, set, or any sized iterable)
    or a boolean mask of the right length already. ``base`` is an existing
    mask to add to rather than replace, and is never modified.

    A mask, not a set: a lasso can select millions of points, and every set
    operation on that is a Python-level pass — building one, unioning it and
    turning it back into an array to draw the halo came to about a quarter
    of the time to complete a lasso, for tens of megabytes where a 3MB mask
    says the same thing.
    """
    mask = (
        base.copy()
        if base is not None and base.shape[0] == size
        else np.zeros(size, dtype=bool)
    )
    if isinstance(indices, np.ndarray) and indices.dtype == bool:
        if indices.shape[0] == size:
            mask |= indices
        return mask
    if not isinstance(indices, np.ndarray):
        # A set or list — np.asarray would make a 0-d object array.
        indices = np.fromiter(indices, dtype=np.int64, count=len(indices))
    if indices.size:
        mask[indices.astype(np.int64, copy=False)] = True
    return mask


class _CloudCompareCamera(TurntableCamera):
    """Turntable camera with CloudCompare's mouse map: left-drag orbits,
    right-drag pans, wheel zooms.

    vispy's turntable puts pan on Shift+left-drag and (a rougher) zoom on
    right-drag.  Rather than reimplement the orbit/pan maths, a plain
    right-drag move is presented to the base handler as the Shift+left-drag
    it already pans with, then the underlying event is put back.  Wheel zoom
    and left-drag orbit are untouched.
    """

    def viewbox_mouse_event(self, event):
        me = getattr(event, "mouse_event", None)
        if (
            me is not None
            and event.type == "mouse_move"
            and me.press_event is not None
            and 2 in event.buttons
            and 1 not in event.buttons
            and not event.modifiers
        ):
            # ``MouseEvent.buttons`` is a plain list; ``modifiers`` is a
            # read-only property backed by ``_modifiers``. Swap both, run the
            # base pan, restore.
            saved_buttons = list(me._buttons)
            saved_mods = me._modifiers
            me._buttons[:] = [1]
            me._modifiers = (keys.SHIFT,)
            try:
                super().viewbox_mouse_event(event)
            finally:
                me._buttons[:] = saved_buttons
                me._modifiers = saved_mods
            return
        super().viewbox_mouse_event(event)


class CloudView:
    """A vispy canvas showing one editable point cloud.

    The rest of the app holds one of these (built in ``app._run_scene``) and
    passes it where it used to pass a napari ``Viewer``.
    """

    def __init__(self, bgcolor: str = "#262626"):
        self.canvas = scene.SceneCanvas(
            keys=None, bgcolor=bgcolor, show=False, size=(800, 800)
        )
        grid = self.canvas.central_widget
        self.view = grid.add_view()
        # fov=0 → orthographic, like CloudCompare; Z is up.
        self.view.camera = _CloudCompareCamera(fov=0.0, up="+z")
        # Position the camera explicitly so nothing ever asks the (possibly
        # empty) scene for its bounds on the first draw.
        self.view.camera.center = (0.0, 0.0, 0.0)
        self.view.camera.scale_factor = 10.0

        # Fixed pixel-size flat discs. Not shaded spheres (vispy's spherical
        # shading darkens the back hemisphere until a zoomed-out plot is a
        # near-black silhouette) and not scene-unit scaling (vispy 0.16's
        # canvas_size_limits clamp doesn't hold, so metre-sized points go
        # sub-pixel and vanish). Pixels keep every point its true colour and
        # visible at any zoom — the size is the panel's "Point size" spinner.
        self.markers = scene.visuals.Markers(
            parent=self.view.scene, scaling="fixed", spherical=False
        )
        self.markers.antialias = 0.3
        # Depth-tested alpha blending: opaque points (alpha 1) composite as a
        # plain overwrite, but the panel's per-tree "fade" sets alpha to
        # FADED_ALPHA and needs real blending to show. depth_test keeps the
        # nearest point per pixel, so opaque points don't accumulate toward
        # white the way a plain translucent-without-depth pass would.
        self.markers.set_gl_state(
            "translucent", depth_test=True, cull_face=False
        )
        # A second marker set drawn on top: a translucent white halo on the
        # currently selected points (napari's Points layer did this itself).
        self.highlight = scene.visuals.Markers(
            parent=self.view.scene, scaling="fixed", spherical=False
        )
        self.highlight.set_gl_state("translucent", depth_test=False)
        # Wireframe box(es) around the tree under review.
        self.bbox = scene.visuals.Line(
            parent=self.view.scene, connect="segments", width=2, antialias=True
        )
        self.bbox.set_gl_state(depth_test=False, blend=True)
        # Inventory stems: wireframe cylinders at the field-measured DBH and
        # height (see segfix.inventory.stem_geometry). Depth-tested, unlike
        # the box above: the whole point is to see whether a drawn stem sits
        # on the trunk or in front of it, which an always-on-top line can't
        # show.
        self.stems = scene.visuals.Line(
            parent=self.view.scene, connect="segments", width=1.5,
            antialias=True,
        )
        self.stems.set_gl_state("translucent", depth_test=True, blend=True)
        self.stems.visible = False

        self._coords = np.empty((0, 3), np.float32)
        self._face_color = np.empty((0, 4), np.float32)
        self._shown = np.empty(0, dtype=bool)
        self._selected = np.zeros(0, dtype=bool)  # mask over _coords
        #: (fov, scale_factor, center) to go back to, while inside the cloud.
        self._outside = None
        self._size = 3.0  # marker diameter in screen pixels

        #: fn(str) -> None, set by the shell to write the status bar
        self.on_status = None
        #: fn() -> None, set by the panel to refresh its selection readout
        self.on_selection_changed = None

        #: while True, a double-click in the canvas recentres the turntable
        #: pivot on the clicked point. The panel clears this whenever a
        #: selection tool (lasso/cluster/…) is armed so a double-click there
        #: only feeds the tool — see SegFixWidget._refresh_double_click_mode.
        self.recenter_on_double_click = True
        # Move mode: double-click a point to make it the camera's pivot, so
        # orbiting turns around whatever you're looking at rather than the
        # cloud's centroid.
        self.canvas.events.mouse_double_click.connect(self._on_double_click)

        self._redraw()

    def _on_double_click(self, event) -> None:
        """Recentre the turntable on the double-clicked point (move mode only).

        Suppressed while a selection tool is armed: those double-clicks belong
        to the tool (e.g. cluster's click-to-grow), not the camera. The panel
        toggles ``recenter_on_double_click`` to track the mode; the camera's
        own ``interactive`` flag is a second guard.
        """
        if getattr(event, "button", None) != 1:
            return
        if not self.recenter_on_double_click or not self.view.camera.interactive:
            return
        idx = self.pick_point((float(event.pos[0]), float(event.pos[1])))
        if idx is None:
            return
        self.view.camera.center = tuple(float(c) for c in self._coords[idx])
        self.canvas.update()
        self.status = "Rotation centre moved to the clicked point"

    # -- points ---------------------------------------------------------
    def load_cloud(self, cloud, point_size: float | None = None) -> None:
        """Show ``cloud`` (a :class:`~segfix.model.PointCloud`): fresh colours,
        everything shown, selection cleared.  Does not move the camera."""
        from .viewer import colors_for_labels

        if point_size is not None:
            self._size = float(point_size)
        self._coords = np.ascontiguousarray(cloud.coords, dtype=np.float32)
        self._face_color = (
            colors_for_labels(cloud.labels, cloud.label_colors)
            if cloud.n_points
            else np.empty((0, 4), np.float32)
        )
        self._shown = np.ones(len(self._coords), dtype=bool)
        self._selected = np.zeros(len(self._coords), dtype=bool)
        self._redraw()
        self._redraw_highlight()

    @property
    def coords(self) -> np.ndarray:
        return self._coords

    @property
    def face_color(self) -> np.ndarray:
        return self._face_color

    @face_color.setter
    def face_color(self, value: np.ndarray) -> None:
        self._face_color = np.asarray(value, dtype=np.float32).reshape(-1, 4)
        self._redraw()
        self._redraw_highlight()

    @property
    def size(self) -> float:
        return self._size

    @size.setter
    def size(self, value: float) -> None:
        self._size = float(value)
        self._redraw()
        self._redraw_highlight()

    @property
    def shown(self) -> np.ndarray:
        return self._shown

    @shown.setter
    def shown(self, mask) -> None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if len(mask) != len(self._coords):
            return
        self._shown = mask
        self._redraw()
        self._redraw_highlight()

    def _redraw(self) -> None:
        vis = self._shown if len(self._shown) == len(self._coords) else None
        pos = self._coords if vis is None else self._coords[vis]
        if len(pos) == 0:
            self.markers.visible = False
            self.canvas.update()
            return
        fc = self._face_color
        if len(fc) != len(self._coords):
            fc = np.ones((len(self._coords), 4), np.float32)
        fc = fc if vis is None else fc[vis]
        self.markers.visible = True
        self.markers.set_data(
            pos=pos, face_color=fc, size=self._size, edge_width=0
        )
        self.canvas.update()

    # -- selection ----------------------------------------------------
    # Held as a boolean mask (see selection_mask). `select()` and
    # `selected_mask` are the numpy path the tools use; `selected` stays for
    # set algebra on small selections, and materialises a set each read.
    @property
    def selected(self) -> set[int]:
        """The selected indices as a set — O(selection) to build, so prefer
        :attr:`selected_mask` or :meth:`select` anywhere it could be big."""
        return set(np.flatnonzero(self._selected).tolist())

    @selected.setter
    def selected(self, indices) -> None:
        self.select(indices)

    @property
    def selected_mask(self) -> np.ndarray:
        """Boolean mask over :attr:`coords` of what is selected.

        The live array, not a copy: take a ``.copy()`` before holding on to
        it across a selection change.
        """
        return self._selected

    def select(self, indices, additive: bool = False) -> int:
        """Select ``indices``, replacing the selection or adding to it.

        ``indices`` may be point indices (any array or iterable) or a
        boolean mask the length of the cloud. Returns how many points are
        selected afterwards, which every caller wants for the status line
        and which is free here.
        """
        self._selected = selection_mask(
            indices, len(self._coords), self._selected if additive else None
        )
        self._redraw_highlight()
        if self.on_selection_changed is not None:
            self.on_selection_changed()
        return int(np.count_nonzero(self._selected))

    def _redraw_highlight(self) -> None:
        mask = self._selected
        if mask.shape[0] != len(self._coords):
            mask = np.zeros(len(self._coords), dtype=bool)
        if len(self._shown) == len(self._coords):
            mask = mask & self._shown
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            self.highlight.visible = False
            self.canvas.update()
            return
        self.highlight.visible = True
        self.highlight.set_data(
            pos=self._coords[idx],
            face_color=(1.0, 1.0, 1.0, 0.9),
            size=self._size + 1.5,
            edge_width=0,
        )
        self.canvas.update()

    # -- bounding box ----------------------------------------------------
    def set_bbox(self, segments: np.ndarray, colors: np.ndarray) -> None:
        """Draw line segments (``(2E, 3)`` endpoints, ``(2E, 4)`` colours)."""
        if segments is None or len(segments) == 0:
            self.clear_bbox()
            return
        self.bbox.set_data(
            pos=np.asarray(segments, np.float32),
            color=np.asarray(colors, np.float32),
        )
        self.bbox.visible = True
        self.canvas.update()

    def clear_bbox(self) -> None:
        self.bbox.visible = False
        self.canvas.update()

    # -- inventory stems -------------------------------------------------
    def set_stems(self, segments: np.ndarray, colors: np.ndarray) -> None:
        """Draw the stem map's cylinders (``(2E, 3)`` endpoints, ``(2E, 4)``
        colours), or clear them when given nothing."""
        if segments is None or len(segments) == 0:
            self.clear_stems()
            return
        self.stems.set_data(
            pos=np.asarray(segments, np.float32),
            color=np.asarray(colors, np.float32),
        )
        self.stems.visible = True
        self.canvas.update()

    def clear_stems(self) -> None:
        self.stems.visible = False
        self.canvas.update()

    # -- appearance ---------------------------------------------------
    def set_background(self, color: str) -> None:
        """Swap the canvas clear colour (the light/dark theme drives this)."""
        self.canvas.bgcolor = color
        self.canvas.update()

    # -- camera --------------------------------------------------------
    # -- inside (bubble) view ---------------------------------------------
    def enter_inside_view(self, origin=None, fov: float = INSIDE_FOV) -> None:
        """Stand the camera *at* a point in the cloud and look around.

        An orthographic view of a dense plot is a wall: every stem at every
        distance is drawn the same size, so what is in front of what cannot
        be read, and a crown ten metres behind the one being edited looks
        like part of it. From inside, a perspective projection sorts the
        scene out the way standing in the forest does — near points spread
        apart, far ones converge, and a gap you can see through is a gap.

        The turntable then orbits the eye about a pivot a few centimetres
        away (:data:`INSIDE_DISTANCE`), which is turning your head rather
        than circling the plot. Scrolling pulls back out of the bubble, and
        :meth:`leave_inside_view` puts the camera back exactly where it was.
        """
        cam = self.view.camera
        if self._outside is None:
            self._outside = (cam.fov, cam.scale_factor, tuple(cam.center))
        if origin is not None:
            cam.center = tuple(float(v) for v in origin)
        cam.fov = float(fov)
        cam.scale_factor = INSIDE_DISTANCE
        self.canvas.update()

    def leave_inside_view(self) -> None:
        """Back to the orthographic view the camera had before."""
        if self._outside is None:
            return
        fov, scale_factor, center = self._outside
        self._outside = None
        cam = self.view.camera
        cam.fov = fov
        cam.scale_factor = scale_factor
        cam.center = center
        self.canvas.update()

    @property
    def inside_view(self) -> bool:
        return self._outside is not None

    def set_inside_fov(self, fov: float) -> None:
        """Widen or narrow the lens without leaving the bubble."""
        if self._outside is not None:
            self.view.camera.fov = float(fov)
            self.canvas.update()

    def reset_view(self) -> None:
        self.leave_inside_view()  # framing the whole cloud means standing back
        if not len(self._coords):
            return
        lo = self._coords.min(axis=0)
        hi = self._coords.max(axis=0)
        # Explicit ranges, computed from the cloud itself: vispy's own
        # ``set_range()`` would walk every sibling visual's ``.bounds`` and
        # trips over the halo/box visuals when they have no data yet.
        self.view.camera.set_range(
            x=(float(lo[0]), float(hi[0])),
            y=(float(lo[1]), float(hi[1])),
            z=(float(lo[2]), float(hi[2])),
            margin=0.05,
        )

    def set_view(self, name: str) -> None:
        """Look at the cloud from one of :data:`VIEWS`, like CloudCompare's
        view buttons: only the viewing direction changes, so the pivot and
        zoom stay where they were."""
        cam = self.view.camera
        cam.azimuth, cam.elevation = VIEWS[name]

    def fly_to(self, center_xyz, span: float) -> None:
        # Framing a whole tree is an outside view of it, so a queue step
        # (Space, a table click) takes the camera back out of the bubble
        # rather than leaving it buried in the new tree's trunk.
        self.leave_inside_view()
        cam = self.view.camera
        cam.center = tuple(float(c) for c in center_xyz)
        cam.scale_factor = max(float(span), 0.5) * 1.6

    #: The ViewBox events a camera listens to for navigation. Disconnecting
    #: these is how a selection tool takes the mouse away from it.
    _CAMERA_MOUSE_EVENTS = (
        "mouse_press", "mouse_release", "mouse_move",
        "mouse_wheel", "gesture_zoom", "gesture_rotate",
    )

    def set_camera_interactive(self, on: bool) -> None:
        """Hand the mouse to the camera, or take it away for a tool.

        ``camera.interactive = False`` looks like it should do this, and is
        what this used to do on its own, but in vispy 0.16 that property is
        write-only — nothing in the library ever reads ``_interactive``. So
        the camera kept navigating throughout a lasso. It was not obvious
        because of how the event reaches it: the canvas emitter runs every
        callback regardless of ``event.handled`` (only ``blocked`` stops it),
        so after the lasso recorded a drag, SceneCanvas._process_mouse_event
        ran anyway and passed the same drag down the scene graph. That starts
        with ``visual_at(event.pos)``, so a drag begun over empty background
        picked nothing and behaved, while one begun on top of the cloud
        walked up to the ViewBox and orbited the view mid-lasso.

        Disconnecting the camera's own handler from the ViewBox is what
        actually stops it, and it covers the wheel and the right-drag pan
        too, which no selection tool ever looked at. ``interactive`` is still
        set, because :meth:`_on_double_click` reads it as our own flag.
        """
        camera = self.view.camera
        camera.interactive = bool(on)
        for name in self._CAMERA_MOUSE_EVENTS:
            emitter = getattr(self.view.events, name, None)
            if emitter is None:  # pragma: no cover - vispy gained/lost one
                continue
            try:
                emitter.disconnect(camera.viewbox_mouse_event)
            except (ValueError, TypeError):
                pass
            if on:
                emitter.connect(camera.viewbox_mouse_event)

    # -- lasso support ------------------------------------------------
    @property
    def native(self):
        """The canvas' Qt widget — parent for the lasso overlay."""
        return self.canvas.native

    @property
    def canvas_size(self) -> tuple[int, int]:
        return tuple(self.canvas.size)

    def project_to_canvas(self, coords: np.ndarray):
        """Project ``coords`` (``(N, 3)`` world XYZ) to canvas pixels.

        Returns ``(xy, valid)`` — ``xy`` is ``(N, 2)`` and ``valid`` masks out
        points behind the camera (non-positive homogeneous w).
        """
        xy, _, valid = self._project(coords, with_depth=False)
        return xy, valid

    def _canvas_matrix(self):
        """The visual→canvas transform as ``(origin, linear, base)``, where
        ``(p - origin) @ linear + base`` is the homogeneous canvas position
        of a world point ``p``.

        ``markers.get_transform()`` hands back three nested ChainTransforms,
        and ``.map()`` walks them one at a time — every level allocating and
        touching its own full ``(N, 4)`` array, so a lasso paid six or seven
        passes over the cloud for what is one linear map. Every transform in
        the chain is linear in homogeneous coordinates, so mapping a point
        and its three unit offsets recovers the composite, and one matmul
        then replaces the walk.

        The probe sits at the camera's pivot rather than at the world origin,
        and the result stays relative to it. A recovered column is a
        difference of two mapped points, and on a cloud left in UTM
        coordinates the world origin maps millions of pixels off screen —
        differencing two such numbers cancels away most of their significant
        digits, which measured as a 0.66 pixel error in the projection.
        Probing at the pivot keeps both mapped points on or near the canvas,
        where the same subtraction costs nothing. Subtracting the origin from
        the cloud is free: it replaces the float32→float64 promotion the
        matmul needed anyway.

        Rebuilt per call rather than cached: it costs four mapped points, and
        anything cached would have to be invalidated on every camera move.
        """
        tr = self.markers.get_transform(map_from="visual", map_to="canvas")
        origin = np.asarray(self.view.camera.center, dtype=np.float64)
        probe = origin + np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        mapped = np.asarray(tr.map(probe), dtype=np.float64)
        base = mapped[0]
        return origin, mapped[1:] - base, base

    def _project(self, coords: np.ndarray, with_depth: bool):
        """``coords`` to canvas pixels, as ``(xy, depth, valid)``.

        ``depth`` is None unless asked for. Only the columns wanted are
        computed — a quarter to a half less work than mapping all four — and
        the projection is the single largest cost of completing a lasso.
        """
        origin, linear, base = self._canvas_matrix()
        # Orthographic — which is what the app runs (the camera is built
        # fov=0). w is then the constant 1, so there is no perspective divide
        # to do and no point can be behind the camera: leave the w column out
        # and every division with it. Decided from the matrix rather than
        # from the camera, so a perspective fov still takes the slow path.
        affine = not linear[:, 3].any() and base[3] == 1.0
        columns = [0, 1] + ([2] if with_depth else []) + ([] if affine else [3])

        # Relative to the same origin the matrix was probed at, and promoted
        # to float64 in the same pass the matmul needed anyway.
        local = np.subtract(coords, origin, dtype=np.float64)
        out = local @ np.ascontiguousarray(linear[:, columns])
        out += base[columns]

        if affine:
            valid = np.ones(len(out), dtype=bool)
            return out[:, :2], (out[:, 2] if with_depth else None), valid
        w = out[:, -1]
        valid = w > 0
        w_safe = np.where(valid, w, 1.0)
        depth = out[:, 2] / w_safe if with_depth else None
        return out[:, :2] / w_safe[:, None], depth, valid

    def pick_point(self, click_xy, radius: float = 9.0) -> int | None:
        """Index of the point under ``click_xy`` (canvas pixels), or ``None``.

        Among shown points whose projection lands within ``radius`` pixels of
        the click, the frontmost (nearest the camera) wins, so clicking a
        near tree doesn't grab a far one behind it. Falls back to the plain
        nearest shown point within a looser radius.
        """
        if len(self._coords) == 0:
            return None
        # Same collapsed projection as the lasso: a click used to walk the
        # whole cloud through the transform chain, so every cluster-tool
        # click cost what a completed lasso did.
        xy, depth, valid = self._project(self._coords, with_depth=True)
        shown = (
            self._shown if len(self._shown) == len(self._coords)
            else np.ones(len(self._coords), bool)
        )
        d2 = ((xy - np.asarray(click_xy, dtype=float)) ** 2).sum(1)
        near = valid & shown & (d2 <= radius * radius)
        if near.any():
            cand = np.flatnonzero(near)
            return int(cand[np.argmin(depth[cand])])
        loose = valid & shown & (d2 <= (radius * 3) ** 2)
        if loose.any():
            cand = np.flatnonzero(loose)
            return int(cand[np.argmin(d2[cand])])
        return None

    # -- status ------------------------------------------------------
    @property
    def status(self):  # write-only in practice; getter kept for symmetry
        return None

    @status.setter
    def status(self, message: str) -> None:
        if self.on_status is not None:
            self.on_status(str(message))
