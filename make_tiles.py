from pathlib import Path
import json
import rasterio
import math

# here I will tile and buffer the calculation, then crop an dosaic the results. faster and faster.

# pixel size ~135 m (x) and 109 m (y)
# if you’re using outer_window=11, max(level)=32, then radius is ~40–50 km

# pixel buffer ≈ radius / pixel_size
# A safe buffer is:
# using the smaller pixel size (more conservative): 109 m
# buffer_px ≈ 47,600 / 109 ≈ 437 px

# Core tile size
# You want core tiles big enough that buffer overhead isn’t ridiculous. If buffer is ~450 px:
# core 3000 px → buffered is 3900 px (OK)
# core 2000 px → buffered is 2900 px (more overhead)
# I’d start with:
# CORE = 3000 px, BUFFER = 450 px.

# so, first make the tiles:

# after making the tiles, count them:
# python -c "import json; print(len(json.load(open('tiles_core3000_buf450.json'))['tiles']))"


IN_TIF = "/datasets/work/lw-nkmp/work/Koala_Habitat_Condition_Mapping_Project/Data/outputs/condition_mosaics/condition_2010.tif"
OUT_JSON = "tiles_core3000_buf450.json"

CORE = 3000
BUF  = 450

tiles = []

with rasterio.open(IN_TIF) as ds:
    W, H = ds.width, ds.height
    nx = math.ceil(W / CORE)
    ny = math.ceil(H / CORE)

    tid = 0
    for ty in range(ny):
        for tx in range(nx):
            x0 = tx * CORE
            y0 = ty * CORE
            x1 = min(x0 + CORE, W)
            y1 = min(y0 + CORE, H)

            bx0 = max(0, x0 - BUF)
            by0 = max(0, y0 - BUF)
            bx1 = min(W, x1 + BUF)
            by1 = min(H, y1 + BUF)

            tiles.append({
                "tile_id": tid,
                "tx": tx, "ty": ty,
                "core": [x0, y0, x1, y1],
                "buf":  [bx0, by0, bx1, by1],
            })
            tid += 1

Path(OUT_JSON).write_text(json.dumps({
    "in_tif": IN_TIF,
    "core_px": CORE,
    "buffer_px": BUF,
    "tiles": tiles
}, indent=2))
print(f"Wrote {OUT_JSON} with {len(tiles)} tiles")
