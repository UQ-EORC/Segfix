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

"""Build the README poster: a walkthrough frame, a play button, a banner.

    python scripts/demo/make_poster.py frame.png docs/walkthrough.jpg \\
        --minutes 17

The banner covers the frame's own caption, so pick a frame with one: it is
dead space in a still either way.
"""

from __future__ import annotations

import argparse
import sys

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter, QPolygonF


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("frame", help="a still from the walkthrough")
    ap.add_argument("out", help="the poster to write (.jpg)")
    ap.add_argument("--minutes", type=int, default=17)
    ap.add_argument("--subtitle", default=None)
    args = ap.parse_args(argv)

    app = QGuiApplication(sys.argv[:1])  # noqa: F841 - QImage needs one
    img = QImage(args.frame).convertToFormat(QImage.Format.Format_RGB32)
    if img.isNull():
        print(f"can't read {args.frame}", file=sys.stderr)
        return 1
    width, height = img.width(), img.height()
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)

    band = QRectF(0, height - 190, width, 190)
    p.setBrush(QColor(10, 11, 14, 235))
    p.drawRect(band)
    big = QFont("Noto Sans")
    big.setPixelSize(52)
    big.setBold(True)
    small = QFont("Noto Sans")
    small.setPixelSize(28)
    p.setPen(QColor(250, 250, 250))
    p.setFont(big)
    p.drawText(QRectF(0, height - 175, width, 80), Qt.AlignmentFlag.AlignCenter,
               "Watch the segfix walkthrough")
    p.setPen(QColor(255, 214, 77))
    p.setFont(small)
    subtitle = args.subtitle or (
        f"{args.minutes} minutes, captioned: every tool, on the example plot"
    )
    p.drawText(QRectF(0, height - 95, width, 50), Qt.AlignmentFlag.AlignCenter,
               subtitle)

    # Play button, over the point view rather than the panels.
    cx, cy, r = width * 0.38, (height - 190) / 2 + 40, 95
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(0, 0, 0, 170))
    p.drawEllipse(QPointF(cx, cy), r, r)
    p.setBrush(QColor(255, 255, 255, 240))
    p.drawPolygon(QPolygonF([QPointF(cx - 30, cy - 48), QPointF(cx - 30, cy + 48),
                             QPointF(cx + 52, cy)]))
    p.end()

    img.scaled(1280, 720, Qt.AspectRatioMode.KeepAspectRatio,
               Qt.TransformationMode.SmoothTransformation).save(args.out, quality=88)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
