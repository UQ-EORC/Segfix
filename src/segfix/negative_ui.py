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

"""Ask what the cloud's ``-1`` tree IDs mean.

segfix writes ``-1`` for a point it has dismissed as noise. Sylva's
``trees --segment``, and others, write it for every point no stem claimed,
which is segfix's *unassigned*. Nothing in the file tells the two apart, and
reading it the wrong way round matters: a plot whose ground arrives as noise
is hidden by ``F`` and missing from the grey points the workflow lassoes
back into trees.

So it is asked, once per project, the first time such a cloud is opened.
:mod:`segfix.treecatalog` applies the answer and remembers it.
"""

from __future__ import annotations

from qtpy.QtWidgets import QMessageBox

#: What the two answers mean, as :data:`segfix.treecatalog.NegativePrompt`
#: spells them.
UNASSIGNED, NOISE = "unassigned", "noise"


def prompt_negative_labels(parent, n_points: int) -> str:
    """Ask, and return ``"unassigned"`` or ``"noise"``.

    Defaults to unassigned, which is what every pipeline that writes ``-1``
    means by it; a cloud segfix saved itself is the exception, and the
    project it belongs to has the answer recorded already, so this is not
    asked for it twice.
    """
    box = QMessageBox(parent)
    box.setWindowTitle("Points with no tree")
    box.setIcon(QMessageBox.Icon.Question)
    box.setText(
        f"{n_points:,} points in this cloud have a tree ID of -1.\n\n"
        "What does that mean here?"
    )
    box.setInformativeText(
        "Sylva and most other segmentation tools write -1 for every point "
        "no tree claimed, which segfix calls unassigned: the grey points you "
        "lasso back into a tree.\n\n"
        "segfix itself writes -1 for points dismissed as noise. Pick that if "
        "this cloud was saved by segfix without its project folder.\n\n"
        "Either way the file keeps -1 for both when you save, and the answer "
        "is remembered with the project."
    )
    unassigned = box.addButton("Unassigned", QMessageBox.ButtonRole.AcceptRole)
    noise = box.addButton("Noise", QMessageBox.ButtonRole.DestructiveRole)
    box.setDefaultButton(unassigned)
    box.exec()
    return NOISE if box.clickedButton() is noise else UNASSIGNED
