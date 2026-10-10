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

"""Lightweight Qt widgets painted over the vispy canvas.

These sit on top of :attr:`CloudView.native` as plain child ``QWidget``s (the
same trick :mod:`segfix.lasso` uses for its drag outline) rather than as vispy
visuals — Qt text and crisp 1px rules are far easier this way, and nothing
here needs to be part of the 3-D scene.
"""

from __future__ import annotations

import math

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from qtpy.QtWidgets import QWidget

from . import theme

_AXIS_COLORS = (QColor("#e08080"), QColor("#86c98a"), QColor("#7fb3e0"))
_AXIS_LABELS = ("X", "Y", "Z")


class ScaleBarOverlay(QWidget):
    """A metric scale bar and an orientation axis-tripod over the canvas'
    bottom-left corner.

    Read-only: transparent to mouse events and repainted whenever the canvas
    redraws (which is whenever the camera moves), so both the bar length and
    the tripod stay in step with the live view.
    """

    _MARGIN = 10
    _PANEL_W = 188
    _PANEL_H = 92
    _TARGET_PX = 96  # rough on-screen bar length aimed for before rounding
    _ARM_PX = 20  # axis-tripod arm length

    def __init__(self, view):
        super().__init__(view.native)
        self._view = view
        self._ink = theme.canvas_ink()
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._sync_geometry()
        view.canvas.events.resize.connect(lambda _e=None: self._sync_geometry())
        view.canvas.events.draw.connect(lambda _e=None: self.update())
        theme.subscribe(self._on_theme)
        self.show()
        self.raise_()

    def _on_theme(self, mode: str) -> None:
        self._ink = theme.canvas_ink(mode)
        self.update()

    def _sync_geometry(self) -> None:
        n = self._view.native
        self.setGeometry(0, 0, n.width(), n.height())

    # -- scale maths ---------------------------------------------------
    def _projected_basis(self):
        """Screen-pixel images of the camera centre and the three world unit
        axes, or ``None`` if any land behind the camera."""
        center = np.asarray(self._view.view.camera.center, dtype=float)
        pts = np.vstack([center, center + np.eye(3)])
        xy, valid = self._view.project_to_canvas(pts)
        if not bool(np.all(valid)):
            return None
        return xy[0], xy[1:] - xy[0]

    @staticmethod
    def _metres_per_pixel(deltas: np.ndarray) -> float | None:
        """World metres spanned by one screen pixel.

        Orthographic projection of a rotated orthonormal 3-frame: the squared
        lengths of the three projected axis vectors sum to ``2 / mpp**2``
        (projecting to 2-D drops exactly one unit of squared length),
        whatever the camera orientation — no dependence on ``scale_factor``.
        """
        sumsq = float((deltas ** 2).sum())
        if sumsq <= 1e-9:
            return None
        return math.sqrt(2.0 / sumsq)

    @staticmethod
    def _nice_length(raw: float) -> float:
        """Round ``raw`` metres to the nearest 1/2/5 × 10ⁿ."""
        if raw <= 0:
            return 0.0
        exp = math.floor(math.log10(raw))
        base = raw / (10 ** exp)
        nice = 1 if base < 1.5 else 2 if base < 3.5 else 5 if base < 7.5 else 10
        return nice * (10 ** exp)

    @staticmethod
    def _format_length(metres: float) -> str:
        if metres >= 1000:
            return f"{metres / 1000:g} km"
        if metres >= 1:
            return f"{metres:g} m"
        if metres >= 0.01:
            return f"{metres * 100:g} cm"
        return f"{metres * 1000:g} mm"

    # -- painting ----------------------------------------------------
    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt signature)
        basis = self._projected_basis()
        if basis is None:
            return
        _, deltas = basis
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        h = self.height()
        # Bottom-left anchor region for the readout. Nothing is drawn behind
        # it — the bar, tripod and labels sit straight on the canvas.
        panel = QRectF(
            self._MARGIN,
            h - self._MARGIN - self._PANEL_H,
            self._PANEL_W,
            self._PANEL_H,
        )

        font = QFont(painter.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - 0.5))
        painter.setFont(font)

        self._paint_axes(
            painter, panel.left() + 30, panel.top() + 22, deltas
        )
        mpp = self._metres_per_pixel(deltas)
        if mpp:
            self._paint_scale_bar(
                painter, panel.left() + 16, panel.top() + 74, mpp
            )

    def _paint_axes(self, painter, ox, oy, deltas: np.ndarray) -> None:
        for i in range(3):
            d = deltas[i]
            n = float(np.hypot(d[0], d[1]))
            if n < 1e-6:
                continue
            dx, dy = d[0] / n * self._ARM_PX, d[1] / n * self._ARM_PX
            pen = QPen(_AXIS_COLORS[i])
            pen.setWidthF(2.0)
            painter.setPen(pen)
            painter.drawLine(QPointF(ox, oy), QPointF(ox + dx, oy + dy))
            painter.drawText(
                QPointF(ox + dx * 1.35 - 4, oy + dy * 1.35 + 4), _AXIS_LABELS[i]
            )

    def _paint_scale_bar(self, painter, x, y, mpp: float) -> None:
        length_m = self._nice_length(self._TARGET_PX * mpp)
        if length_m <= 0:
            return
        px = length_m / mpp
        pen = QPen(self._ink)
        pen.setWidthF(2.0)
        painter.setPen(pen)
        painter.drawLine(QPointF(x, y), QPointF(x + px, y))
        painter.drawLine(QPointF(x, y - 4), QPointF(x, y + 4))
        painter.drawLine(QPointF(x + px, y - 4), QPointF(x + px, y + 4))
        painter.drawText(QPointF(x, y - 8), self._format_length(length_m))


# -- cross-section slab handles ---------------------------------------------
#: The slab's own colour, shared with the box the view draws for it.
SLAB_COLOR = QColor(255, 158, 26)


class _SlabHandle(QWidget):
    """One grip on the slab: an arrow on a face, or a square at the middle.

    A real child widget rather than a painted region, so Qt hands it the
    press and the canvas (and any armed lasso) never sees the drag.
    """

    SIZE = 26

    def __init__(self, parent, kind: str, on_press, on_drag, on_release):
        super().__init__(parent)
        self.kind = kind
        self._on_press = on_press
        self._on_drag = on_drag
        self._on_release = on_release
        self._angle = 0.0  # screen direction the arrow points, radians
        self._origin = None
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.SizeAllCursor if kind == "mid"
                       else Qt.CursorShape.OpenHandCursor)
        self.setToolTip({
            "lo": "Drag to move this face of the slab",
            "hi": "Drag to move this face of the slab",
            "mid": "Drag to slide the whole slab",
        }[kind])

    def set_angle(self, angle: float) -> None:
        if angle != self._angle:
            self._angle = angle
            self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._origin = event.globalPosition()
        self._on_press(self.kind)
        if self.kind != "mid":
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._origin is None:
            return
        delta = event.globalPosition() - self._origin
        self._on_drag(self.kind, float(delta.x()), float(delta.y()))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._origin is None:
            return
        self._origin = None
        if self.kind != "mid":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._on_release(self.kind)
        event.accept()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = self.SIZE / 2
        p.setPen(QPen(QColor(0, 0, 0, 160), 1.5))
        p.setBrush(SLAB_COLOR)
        if self.kind == "mid":
            p.drawRect(QRectF(c - 6, c - 6, 12, 12))
            return
        p.translate(c, c)
        p.rotate(math.degrees(self._angle))
        # An arrowhead pointing along +x, then rotated to the axis.
        p.drawPolygon(QPolygonF([QPointF(10, 0), QPointF(-6, -8),
                                 QPointF(-3, 0), QPointF(-6, 8)]))


class SlabHandles:
    """CloudCompare-style grips on the cross-section slab.

    An arrow at the centre of each face along the slab's axis moves that
    face; a square at the slab's centre slides the whole thing. They follow
    the box as the camera moves (re-placed on every canvas draw, like the
    scale bar) and turn pixels dragged into metres along the axis using
    where a metre of that axis lands on screen right now.

    ``on_press()`` takes a snapshot of the slab, ``on_drag(kind, metres)``
    reports the cumulative movement since, and ``on_release()`` ends it.
    """

    #: Pixels of screen a metre of axis must span before a drag counts:
    #: an axis pointing straight at the camera has no direction to drag in.
    MIN_AXIS_PX = 2.0

    def __init__(self, view, on_press, on_drag, on_release):
        self._view = view
        self._on_press = on_press
        self._on_drag_metres = on_drag
        self._on_release = on_release
        self._axis = 2
        self._points = None  # (3, 3) world: lo face, hi face, middle
        self._axis_px = None
        self._handles = {
            kind: _SlabHandle(view.native, kind, self._press, self._drag,
                              self._release)
            for kind in ("lo", "hi", "mid")
        }
        self.hide()
        view.canvas.events.draw.connect(lambda _e=None: self.place())
        view.canvas.events.resize.connect(lambda _e=None: self.place())

    # -- what to show -----------------------------------------------------
    def show(self, lo_face, hi_face, mid, axis: int) -> None:
        self._points = np.array([lo_face, hi_face, mid], dtype=np.float64)
        self._axis = int(axis)
        self.place()

    def hide(self) -> None:
        self._points = None
        for h in self._handles.values():
            h.hide()

    @property
    def visible(self) -> bool:
        return self._points is not None

    def place(self) -> None:
        """Put each handle on its face centre's screen position."""
        if self._points is None:
            return
        xy, valid = self._view.project_to_canvas(self._points)
        w, h = self._view.native.width(), self._view.native.height()
        axis_px = self._view.axis_pixels_per_metre(self._points[2], self._axis)
        self._axis_px = axis_px
        angle = (math.atan2(axis_px[1], axis_px[0])
                 if axis_px is not None and np.hypot(*axis_px) > 1e-6 else 0.0)
        for i, kind in enumerate(("lo", "hi", "mid")):
            handle = self._handles[kind]
            x, y = float(xy[i][0]), float(xy[i][1])
            on_screen = bool(valid[i]) and -20 <= x <= w + 20 and -20 <= y <= h + 20
            if not on_screen:
                handle.hide()
                continue
            handle.set_angle(angle + (math.pi if kind == "lo" else 0.0))
            handle.move(int(round(x - handle.SIZE / 2)),
                        int(round(y - handle.SIZE / 2)))
            handle.show()
            handle.raise_()

    # -- drag maths -------------------------------------------------------
    def metres_for(self, dx: float, dy: float) -> float | None:
        """How far along the axis a drag of ``(dx, dy)`` pixels is.

        The projection of the drag onto the axis' screen direction, scaled
        by how many pixels a metre of it spans: a drag across the axis is
        nothing, a drag along it is the full distance.
        """
        d = self._axis_px
        if d is None:
            return None
        norm_sq = float(d[0] * d[0] + d[1] * d[1])
        if norm_sq < self.MIN_AXIS_PX ** 2:
            return None
        return float(dx * d[0] + dy * d[1]) / norm_sq

    def _press(self, kind: str) -> None:
        self.place()  # fresh axis vector for this drag
        self._on_press()

    def _drag(self, kind: str, dx: float, dy: float) -> None:
        metres = self.metres_for(dx, dy)
        if metres is not None:
            self._on_drag_metres(kind, metres)

    def _release(self, kind: str) -> None:
        self._on_release()
