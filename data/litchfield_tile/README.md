# Litchfield test tile

A small real dataset for trying Segfix: a 15 × 15 m tile cut from the TERN
Litchfield savanna plot (Northern Territory, terrestrial laser scan, 2021).

| file | what it is |
|------|------------|
| `litchfield_tile.laz` | 705,777 points, 52 trees, segmented automatically by raycloudtools (`rayextract trees`). `treeID` per point; `0` is unassigned (ground and understorey). `leafwood` per point: `0` leaf, `1` wood, `2` ground. `classification` is `2` for ground, `1` otherwise. |
| `litchfield_tile_stems.csv` | the 28 field-measured stems inside the tile: `tree_id`, `x`, `y`, `dbh_m`, `species`. |

The tile spans x 52–67 m and y −83 to −68 m in the plot's local coordinates,
and the stem map uses the same frame. Trees whose stem base lies inside the
tile are kept whole, so some crowns reach past its edge. The segmentation
has not been corrected, so it still has the usual errors.

The leaf–wood labels come from arbor (`segment_ground`, `wood_likelihood`,
`segment_semantic`), run on this tile as it is. They are automatic too,
and have not been corrected.

## Trying it

1. Open `litchfield_tile.laz` in Segfix. It is decompressed into a new
   project on first open.
2. Review and fix trees one at a time.
3. **Inventory ▸ Load Stem Map…** and pick `litchfield_tile_stems.csv`.
   27 of the 28 stems should match, with a residual of about 0.26 m.
4. For leaf and wood, pick `leafwood` in the **Field** combo of the
   **Point class** box, name its values in **Set up…** (`0` Leaf, `1` Wood,
   `2` Ground), and turn on **Colour by class** (`Shift+F`).

## Source and licence

TERN Litchfield SuperSite TLS data and stem map, TERN (Terrestrial
Ecosystem Research Network), licensed CC BY 4.0.
