from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import rasterio
from rasterio.windows import Window

from connectivity import create_overviews, connectedness

def have_file(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0

def main():
    if len(sys.argv) != 4:
        print("Usage: run_tile_connectedness.py <tiles.json> <tile_id> <out_dir>", file=sys.stderr)
        return 2

    tiles_json = Path(sys.argv[1])
    tile_id = int(sys.argv[2])
    out_dir = Path(sys.argv[3])

    cfg = json.loads(tiles_json.read_text())
    in_tif = cfg["in_tif"]
    tiles = cfg["tiles"]

    t = tiles[tile_id]
    x0,y0,x1,y1 = t["core"]
    bx0,by0,bx1,by1 = t["buf"]

    out_dir.mkdir(parents=True, exist_ok=True)

    # deterministic filenames
    tile_tag = f"tx{t['tx']:03d}_ty{t['ty']:03d}"
    scratch = Path(os.environ.get("TMPDIR", "/tmp"))

    buf_tif  = scratch / f"cond_buf_{tile_tag}.tif"
    buf_out  = scratch / f"conn_buf_{tile_tag}.tif"
    core_out = out_dir / f"connectedness_{tile_tag}.tif"

    if have_file(core_out):
        print(f"[SKIP] exists: {core_out}", flush=True)
        return 0

    # 1) write buffered condition tile
    with rasterio.open(in_tif) as src:
        w = Window(bx0, by0, bx1-bx0, by1-by0)
        profile = src.profile.copy()
        profile.update(
            width=int(w.width),
            height=int(w.height),
            transform=rasterio.windows.transform(w, src.transform),
            tiled=True,
            compress="LZW",
            BIGTIFF="IF_SAFER",
        )
        data = src.read(1, window=w)

    with rasterio.open(buf_tif, "w", **profile) as dst:
        dst.write(data, 1)

    # 2) create overviews on buffered tile (in-place)
    create_overviews(
        input_raster=str(buf_tif),
        output_raster=None,
        overview_levels=[2,4,8,16,32],   # must match connectedness levels you use
    )

    # 3) run connectedness on buffered tile
    connectedness(
        condition_file=str(buf_tif),
        lambdas=[2,20,200],
        max_cost=2.0,
        window_size=5,
        outer_window=11,
        levels=[2,4,8,16,32],
        sigma=1,
        option=3,
        filename=str(buf_out),
    )

    # 4) crop buffered output back to core window
    # core window relative to buffered window
    cx0 = x0 - bx0
    cy0 = y0 - by0
    cx1 = x1 - bx0
    cy1 = y1 - by0

    with rasterio.open(buf_out) as src:
        wcore = Window(cx0, cy0, cx1-cx0, cy1-cy0)
        profile = src.profile.copy()
        profile.update(
            width=int(wcore.width),
            height=int(wcore.height),
            transform=rasterio.windows.transform(wcore, src.transform),
            tiled=True,
            compress="LZW",
            BIGTIFF="IF_SAFER",
        )
        core = src.read(1, window=wcore)

    with rasterio.open(core_out, "w", **profile) as dst:
        dst.write(core, 1)

    # cleanup scratch
    for p in [buf_tif, buf_out]:
        try: p.unlink()
        except: pass

    print(f"[DONE] {tile_id} -> {core_out}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
