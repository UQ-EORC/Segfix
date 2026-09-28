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

"""Segmentation-fixing operations on a :class:`~segfix.model.PointCloud`.

Each function mutates the cloud's label array through ``set_labels`` (so every
op is undoable) and returns a short human-readable summary for the status bar.
The functions are deliberately UI-agnostic: the GUI passes in point indices
(from the napari selection) or tree IDs, and these decide what the new labels
should be.
"""

from __future__ import annotations

from .model import NOISE, UNASSIGNED, PointCloud


def reassign(cloud: PointCloud, indices, target_id: int) -> str:
    """Move the selected points to an existing tree ``target_id``."""
    n = cloud.set_labels(indices, int(target_id), f"reassign → {target_id}")
    return f"Reassigned {n} points to tree {target_id}"


def create_new(cloud: PointCloud, indices) -> str:
    """Split the selected points off into a brand-new tree instance.

    This covers both "create a new tree from unassigned points" and "split an
    under-segmented blob": select the subset that should become its own tree
    and the rest keeps its original ID.
    """
    new_id = cloud.next_free_id()
    n = cloud.set_labels(indices, new_id, f"new tree {new_id}")
    return f"Created tree {new_id} from {n} points"


def mark_noise(cloud: PointCloud, indices) -> str:
    """Flag the selected points as noise / non-tree (kept, not deleted)."""
    n = cloud.set_labels(indices, NOISE, "mark noise")
    return f"Marked {n} points as noise"


def unassign(cloud: PointCloud, indices) -> str:
    """Return the selected points to the unassigned pool."""
    n = cloud.set_labels(indices, UNASSIGNED, "unassign")
    return f"Unassigned {n} points"


def set_class(cloud: PointCloud, indices, code: int, name: str) -> str:
    """Give the selected points the point class ``code`` (named ``name``)."""
    n = cloud.set_classes(indices, int(code), f"class → {name}")
    return f"Set {n} points to class {name}"
