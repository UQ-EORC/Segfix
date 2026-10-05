"""Scripted, captioned recording of segfix, driven from inside its own Qt
event loop.

A QTimer ticks at FPS. Each tick composites one frame -- the main window
(which includes the vispy GL canvas via QWidget.grab), plus any dialogs,
popovers and menus, which are separate top-level windows and so have to be
pasted on at their own positions -- then paints captions, chapter title
cards, a cursor, click ripples and key badges over it, and pipes the raw RGB
into ffmpeg.

A storyboard is a generator taking the Runner. What it yields decides what
happens before its next step:
    yield None            -> one frame
    yield 2.5             -> hold for 2.5 s of video
    yield r.until(pred)   -> keep capturing until pred() is true
and `yield from` the helpers (move_to, click, lasso, camera, ...) animate
one frame at a time, so motion is smooth however long a frame takes to grab.
"""

from __future__ import annotations

import math
import subprocess
import time
import traceback
import types

import numpy as np
from qtpy.QtCore import QPoint, QPointF, QRect, QRectF, Qt, QTimer
from qtpy.QtGui import (
    QColor, QFont, QFontMetrics, QImage, QPainter, QPen, QPolygonF,
)
from qtpy.QtWidgets import QApplication, QMainWindow

W, H, FPS = 1920, 1080, 24
BACKDROP = QColor("#17191d")
TITLE_H = 30        # drawn title strip above composited dialogs (no WM frame in a grab)
FADE_FRAMES = 8     # caption fade-in, and title-card fade in/out


def ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def find(cls):
    """The visible top-level widget of type ``cls``, if any."""
    for w in QApplication.topLevelWidgets():
        if isinstance(w, cls) and w.isVisible():
            return w
    return None


class Recorder:
    def __init__(self, out_path: str, font_family: str = "Noto Sans"):
        self.out_path = out_path
        self.font_family = font_family
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "medium", "-crf", "18",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
            stdin=subprocess.PIPE,
        )
        self.win: QMainWindow | None = None
        self.caption = ""
        self.caption_since = 0
        self.chapter = ""
        self.title_text: str | None = None
        self.title_start = 0
        self.title_len = 0
        self.key_text: str | None = None
        self.key_frames = 0
        self.cursor: QPointF | None = None
        self.ripples: list[list] = []
        self.frames = 0
        self.captions_log: list[tuple[int, str]] = []  # (frame, text); "" ends a cue
        self.grab_ms: list[float] = []

    # -- state the storyboard sets --------------------------------------
    def set_caption(self, text: str, chapter: str | None = None) -> None:
        if chapter is not None:
            self.chapter = chapter
        if text != self.caption:
            self.caption_since = self.frames
        self.caption = text
        self.captions_log.append((self.frames, text))

    def clear_caption(self) -> None:
        self.caption = ""
        self.captions_log.append((self.frames, ""))

    def title(self, text: str, secs: float) -> None:
        """A chapter title card: the frame dims and ``text`` fades in, large,
        in the middle, for ``secs``. Captions are held off while it shows."""
        self.title_text, self.title_start, self.title_len = text, self.frames, round(secs * FPS)
        self.captions_log.append((self.frames, text))

    def key(self, text: str, secs: float = 1.4) -> None:
        self.key_text, self.key_frames = text, int(secs * FPS)

    # -- compositing ------------------------------------------------------
    def _main_window(self):
        if self.win is None:
            for w in QApplication.topLevelWidgets():
                if isinstance(w, QMainWindow) and w.isVisible():
                    self.win = w
        return self.win if (self.win is not None and self.win.isVisible()) else None

    def origin(self) -> QPoint | None:
        win = self._main_window()
        return win.mapToGlobal(QPoint(0, 0)) if win is not None else None

    def top_left_of(self, top) -> QPointF:
        """Where top-level window ``top`` is drawn in the frame."""
        o = self.origin()
        if o is not None:
            return QPointF(top.mapToGlobal(QPoint(0, 0)) - o)
        return QPointF((W - top.width()) / 2, (H - top.height()) / 2 + TITLE_H / 2)

    def compose(self) -> QImage:
        t0 = time.perf_counter()
        img = QImage(W, H, QImage.Format.Format_RGB888)
        img.fill(BACKDROP)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        win = self._main_window()
        if win is not None:
            p.drawPixmap(0, 0, win.grab())
        tops = [
            w for w in QApplication.topLevelWidgets()
            if w.isVisible() and w is not win and not isinstance(w, QMainWindow)
            and w.windowType() not in (Qt.WindowType.ToolTip,)
            and w.width() > 4 and w.height() > 4
        ]
        # dialogs first, popups/menus on top of them
        tops.sort(key=lambda w: w.windowType() == Qt.WindowType.Popup)
        for w in tops:
            pos = self.top_left_of(w)
            popup = w.windowType() == Qt.WindowType.Popup
            self._window_chrome(p, w, pos, popup)
            p.drawPixmap(pos, w.grab())
        self._overlays(p)
        p.end()
        self.grab_ms.append((time.perf_counter() - t0) * 1000)
        return img

    def _window_chrome(self, p, w, pos: QPointF, popup: bool) -> None:
        """A shadow, and for real windows a title strip -- the WM draws the
        real one, but outside what a grab can see."""
        top = pos.y() - (0 if popup else TITLE_H)
        rect = QRectF(pos.x(), top, w.width(), w.height() + (0 if popup else TITLE_H))
        for i, a in ((18, 18), (10, 30), (4, 50)):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, a))
            p.drawRoundedRect(rect.adjusted(-i, -i + 6, i, i + 6), 10, 10)
        if popup:
            return
        pal = w.palette()
        strip = QRectF(pos.x(), top, w.width(), TITLE_H)
        p.setBrush(pal.window().color().darker(135))
        p.drawRoundedRect(strip, 6, 6)
        p.drawRect(strip.adjusted(0, TITLE_H / 2, 0, 0))
        f = QFont(self.font_family)
        f.setPixelSize(14)
        f.setBold(True)
        p.setFont(f)
        p.setPen(pal.windowText().color())
        p.drawText(strip, Qt.AlignmentFlag.AlignCenter, w.windowTitle())

    def canvas_rect(self) -> QRectF:
        win = self._main_window()
        if win is None or win.centralWidget() is None:
            return QRectF(0, 0, W, H)
        c = win.centralWidget()
        tl = c.mapTo(win, QPoint(0, 0))
        return QRectF(tl.x(), tl.y(), c.width(), c.height())

    def _title_alpha(self) -> float:
        if self.title_text is None:
            return 0.0
        t = self.frames - self.title_start
        if t < 0 or t >= self.title_len:
            return 0.0
        return max(0.0, min(1.0, t / FADE_FRAMES, (self.title_len - t) / FADE_FRAMES))

    def _overlays(self, p: QPainter) -> None:
        for pt, age in self.ripples:
            t = min(age / 14, 1.0)
            r = 8 + 34 * t
            p.setPen(QPen(QColor(255, 214, 0, int(230 * (1 - t))), 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(pt, r, r)
        if self.cursor is not None:
            self._draw_cursor(p, self.cursor)
        area = self.canvas_rect()
        title_a = self._title_alpha()
        if title_a > 0:
            self._draw_title(p, title_a)
            return  # nothing else competes with a title card
        caption_top = self._draw_caption(p, area)
        if self.key_text and self.key_frames > 0:
            self._draw_key(p, area, caption_top)

    def _draw_title(self, p, a: float) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(8, 9, 12, int(175 * a)))
        p.drawRect(QRectF(0, 0, W, H))
        number, _, name = self.title_text.partition(" · ")
        if not name:
            number, name = "", self.title_text
        big = QFont(self.font_family)
        big.setPixelSize(58)
        big.setBold(True)
        small = QFont(self.font_family)
        small.setPixelSize(24)
        small.setBold(True)
        small.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3)
        cy = H / 2
        if number:
            p.setFont(small)
            p.setPen(QColor(255, 214, 77, int(255 * a)))
            p.drawText(QRectF(0, cy - 90, W, 40), Qt.AlignmentFlag.AlignCenter,
                       f"CHAPTER {number}" if number.isdigit() else number.upper())
        p.setFont(big)
        p.setPen(QColor(250, 250, 250, int(255 * a)))
        p.drawText(QRectF(0, cy - 45, W, 90), Qt.AlignmentFlag.AlignCenter, name)
        p.setPen(QPen(QColor(255, 214, 77, int(200 * a)), 3))
        p.drawLine(QPointF(W / 2 - 60, cy + 58), QPointF(W / 2 + 60, cy + 58))

    def _draw_cursor(self, p, at: QPointF) -> None:
        pts = [(0, 0), (0, 23), (6, 18), (10.5, 27), (14, 25.5), (9.8, 16.5), (17, 16.5)]
        poly = QPolygonF([QPointF(at.x() + x, at.y() + y) for x, y in pts])
        p.setPen(QPen(QColor(0, 0, 0), 1.6))
        p.setBrush(QColor(255, 255, 255))
        p.drawPolygon(poly)

    def _draw_caption(self, p, area: QRectF) -> float:
        self._caption_box = None
        if not self.caption:
            return area.bottom()
        a = min(1.0, (self.frames - self.caption_since) / FADE_FRAMES)
        body = QFont(self.font_family)
        body.setPixelSize(25)
        head = QFont(self.font_family)
        head.setPixelSize(15)
        head.setBold(True)
        head.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
        maxw = int(min(1180, area.width() - 120))
        fm = QFontMetrics(body)
        flags = Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignHCenter
        br = fm.boundingRect(QRect(0, 0, maxw, 2000), int(flags), self.caption)
        hh = QFontMetrics(head).height() if self.chapter else 0
        pad_x, pad_y, gap = 30, 13, (6 if self.chapter else 0)
        bw = max(br.width(), QFontMetrics(head).horizontalAdvance(self.chapter.upper())
                 if self.chapter else 0) + 2 * pad_x
        bh = br.height() + hh + gap + 2 * pad_y
        cx = area.center().x()
        # Low in the canvas, over the ground rather than the trees; centred,
        # so it stays clear of the scale bar and axes in the far-left corner.
        bottom = area.bottom() - 26
        box = QRectF(cx - bw / 2, bottom - bh, bw, bh)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(12, 13, 16, int(215 * a)))
        p.drawRoundedRect(box, 14, 14)
        self._caption_box = box
        y = box.top() + pad_y
        if self.chapter:
            p.setFont(head)
            p.setPen(QColor(255, 214, 77, int(255 * a)))
            p.drawText(QRectF(box.left(), y, bw, hh), Qt.AlignmentFlag.AlignCenter,
                       self.chapter.upper())
            y += hh + gap
        p.setFont(body)
        p.setPen(QColor(245, 245, 245, int(255 * a)))
        p.drawText(QRectF(box.left() + pad_x, y, bw - 2 * pad_x, br.height()),
                   int(flags), self.caption)
        return box.top()

    def _draw_key(self, p, area: QRectF, caption_top: float) -> None:
        f = QFont(self.font_family)
        f.setPixelSize(30)
        f.setBold(True)
        fm = QFontMetrics(f)
        w = fm.horizontalAdvance(self.key_text) + 44
        h = fm.height() + 22
        # Bottom-right of the canvas: empty in every scene (the Current tree
        # panel is top-right), so it never sits on the points being edited.
        box = QRectF(area.right() - w - 40, area.bottom() - h - 34, w, h)
        if self._caption_box is not None and box.intersects(self._caption_box):
            # A long caption reaches the corner: sit just above it instead.
            box.moveBottom(caption_top - 12)
        fade = min(1.0, self.key_frames / 6)
        p.setPen(QPen(QColor(255, 255, 255, int(120 * fade)), 2))
        p.setBrush(QColor(40, 44, 52, int(235 * fade)))
        p.drawRoundedRect(box, 10, 10)
        p.setPen(QColor(255, 255, 255, int(255 * fade)))
        p.setFont(f)
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, self.key_text)

    # -- output -----------------------------------------------------------
    def frame(self, repeat: int = 1) -> None:
        img = self.compose()
        ptr = img.constBits()
        ptr.setsize(img.sizeInBytes())
        data = bytes(ptr)
        for _ in range(max(1, repeat)):
            self.proc.stdin.write(data)
            self.frames += 1
        for r in self.ripples:
            r[1] += repeat
        self.ripples = [r for r in self.ripples if r[1] <= 14]
        if self.key_frames > 0:
            self.key_frames -= repeat

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()
        self._write_srt()

    def _write_srt(self) -> None:
        def ts(frame):
            s = frame / FPS
            return f"{int(s // 3600):02}:{int(s % 3600 // 60):02}:{int(s % 60):02},{int(s * 1000 % 1000):03}"
        log = self.captions_log
        cues = []
        for i, (f, text) in enumerate(log):
            if not text:
                continue
            end = log[i + 1][0] if i + 1 < len(log) else self.frames
            if end > f:
                cues.append((f, end, text))
        with open(self.out_path.rsplit(".", 1)[0] + ".srt", "w", encoding="utf-8") as fh:
            for n, (f, end, text) in enumerate(cues, 1):
                fh.write(f"{n}\n{ts(f)} --> {ts(end)}\n{text}\n\n")


class Runner:
    def __init__(self, rec: Recorder, story):
        self.rec = rec
        self.recording = False
        self._wait_frames = 0
        self._pred = None
        self._busy = False
        self._done = False
        self.error: str | None = None
        self._gen = story(self)
        self.timer = QTimer()
        self.timer.setInterval(1000 // FPS)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # -- scheduling -------------------------------------------------------
    def _tick(self) -> None:
        if self._busy or self._done:
            return  # re-entered from a nested event loop mid-step
        self._busy = True
        try:
            if self.recording:
                self.rec.frame()
            if self._wait_frames > 0:
                self._wait_frames -= 1
                return
            if self._pred is not None:
                pred, deadline, desc = self._pred
                if not pred():
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"timed out waiting for: {desc}")
                    return
                self._pred = None
            cmd = next(self._gen)
            if isinstance(cmd, (int, float)):
                self._wait_frames = max(0, round(cmd * FPS) - 1)
            elif isinstance(cmd, tuple) and cmd and cmd[0] == "until":
                self._pred = (cmd[1], time.monotonic() + cmd[2], cmd[3])
        except StopIteration:
            self.finish()
        except Exception:
            self.error = traceback.format_exc()
            self.finish()
        finally:
            self._busy = False

    def finish(self) -> None:
        if self._done:
            return
        self._done = True
        self.timer.stop()
        self.rec.close()
        QApplication.quit()

    def until(self, pred, timeout: float = 60.0, desc: str = ""):
        return ("until", pred, timeout, desc or getattr(pred, "__name__", "?"))

    def hold(self, secs: float) -> None:
        """Hold the current picture (for blocking phases: no ticks run)."""
        if self.recording:
            self.rec.frame(repeat=round(secs * FPS))

    # -- coordinates ------------------------------------------------------
    def wpt(self, widget, local: QPoint | None = None) -> QPointF:
        """Frame coordinates of a point on ``widget`` (its centre by default)."""
        local = widget.rect().center() if local is None else local
        top = widget.window()
        if top is self.rec._main_window():
            return QPointF(widget.mapTo(top, local))
        return self.rec.top_left_of(top) + QPointF(widget.mapTo(top, local))

    def canvas_to_frame(self, view, x: float, y: float) -> QPointF:
        return QPointF(view.native.mapTo(self.rec._main_window(), QPoint(int(x), int(y))))

    # -- animated actions (use with `yield from`) ---------------------------
    def move_to(self, pt: QPointF, secs: float = 0.6):
        start = self.rec.cursor or QPointF(W * 0.55, H * 0.55)
        n = max(1, round(secs * FPS))
        for i in range(1, n + 1):
            t = ease(i / n)
            self.rec.cursor = QPointF(start.x() + (pt.x() - start.x()) * t,
                                      start.y() + (pt.y() - start.y()) * t)
            yield None

    def click(self, pt: QPointF | None = None, secs: float = 0.6):
        if pt is not None:
            yield from self.move_to(pt, secs)
        self.rec.ripples.append([QPointF(self.rec.cursor), 0])
        yield None

    def double_click(self, pt: QPointF | None = None, secs: float = 0.6):
        if pt is not None:
            yield from self.move_to(pt, secs)
        self.rec.ripples.append([QPointF(self.rec.cursor), 0])
        yield None
        yield None
        self.rec.ripples.append([QPointF(self.rec.cursor), 0])
        yield None

    def drag_camera(self, view, cursor_from: QPointF, cursor_to: QPointF,
                    secs: float = 1.6, **target):
        """A mouse drag that moves the camera: the cursor sweeps from one
        point to the other while the camera eases to ``target``."""
        yield from self.move_to(cursor_from, 0.4)
        self.rec.ripples.append([QPointF(cursor_from), 0])
        cam = view.view.camera
        start = {k: getattr(cam, k) for k in target}
        n = max(1, round(secs * FPS))
        for i in range(1, n + 1):
            t = ease(i / n)
            self.rec.cursor = QPointF(
                cursor_from.x() + (cursor_to.x() - cursor_from.x()) * t,
                cursor_from.y() + (cursor_to.y() - cursor_from.y()) * t)
            _set_camera(cam, start, target, t)
            view.canvas.update()
            yield None

    def lasso(self, tool, canvas_path, secs: float = 1.8, additive: bool = False):
        view = tool.view
        path = resample(np.asarray(canvas_path, float), max(8, round(secs * FPS)))
        def ev(xy):
            return types.SimpleNamespace(
                pos=(float(xy[0]), float(xy[1])), button=1,
                modifiers=("Shift",) if additive else (), handled=False)
        yield from self.move_to(self.canvas_to_frame(view, *path[0]), 0.5)
        tool._on_press(ev(path[0]))
        for xy in path[1:]:
            tool._on_move(ev(xy))
            self.rec.cursor = self.canvas_to_frame(view, *xy)
            yield None
        tool._on_move(ev(path[0]))
        tool._on_release(ev(path[0]))
        yield None

    def camera(self, view, secs: float = 1.0, **target):
        cam = view.view.camera
        start = {k: getattr(cam, k) for k in target}
        n = max(1, round(secs * FPS))
        for i in range(1, n + 1):
            _set_camera(cam, start, target, ease(i / n))
            view.canvas.update()
            yield None

    def smooth(self, view, action, secs: float = 0.9):
        """Run an action that jumps the camera, then glide there instead."""
        cam = view.view.camera
        keys = ("center", "scale_factor", "azimuth", "elevation")
        before = {k: getattr(cam, k) for k in keys}
        action()
        after = {k: getattr(cam, k) for k in keys}
        for k, v in before.items():
            setattr(cam, k, v)
        yield from self.camera(view, secs, **after)


def _set_camera(cam, start, target, t):
    for k, v in target.items():
        a = start[k]
        if k == "center":
            a, b = np.asarray(a, float), np.asarray(v, float)
            cam.center = tuple(a + (b - a) * t)
        elif k == "azimuth":
            d = (v - a + 180) % 360 - 180
            cam.azimuth = a + d * t
        else:
            setattr(cam, k, a + (v - a) * t)


# -- geometry helpers ---------------------------------------------------------
def resample(path: np.ndarray, n: int) -> np.ndarray:
    """``n`` points evenly spaced along a closed polyline."""
    closed = np.vstack([path, path[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], n, endpoint=False)
    return np.column_stack([np.interp(s, cum, closed[:, 0]), np.interp(s, cum, closed[:, 1])])


def outline_around(xy: np.ndarray, margin: float = 22.0, wobble: float = 4.0,
                   seed: int = 0) -> np.ndarray:
    """A hand-drawn-looking loop around 2D points: their hull, pushed out."""
    from scipy.spatial import ConvexHull

    xy = np.asarray(xy, float)
    hull = xy[ConvexHull(xy).vertices]
    ring = resample(hull, 60)
    c = ring.mean(axis=0)
    d = ring - c
    norm = np.hypot(d[:, 0], d[:, 1])[:, None]
    rng = np.random.default_rng(seed)
    push = margin + wobble * np.sin(np.linspace(0, 2 * math.pi * 3, len(ring)) + rng.uniform(0, 6))
    return ring + d / np.maximum(norm, 1e-6) * push[:, None]
