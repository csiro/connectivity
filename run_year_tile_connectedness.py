from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import rasterio
from rasterio.windows import Window

from connectivity import create_overviews, connectedness


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def have_file(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def main():
    if len(sys.argv) != 6:
        print(
            "Usage: run_year_tile_connectedness.py "
            "<tiles.json> <year> <tile_id> <cond_dir> <out_base_dir>",
            file=sys.stderr,
        )
        return 2

    t0_all = time.time()

    tiles_json = Path(sys.argv[1])
    year = int(sys.argv[2])
    tile_id = int(sys.argv[3])
    cond_dir = Path(sys.argv[4])
    out_base = Path(sys.argv[5])

    cfg = json.loads(tiles_json.read_text())
    tiles = cfg["tiles"]
    t = tiles[tile_id]

    x0, y0, x1, y1 = t["core"]
    bx0, by0, bx1, by1 = t["buf"]

    tile_tag = f"tx{t['tx']:03d}_ty{t['ty']:03d}"
    log(f"[START] year={year} tile={tile_tag} tile_id={tile_id}")

    cond_tif = cond_dir / f"condition_{year}.tif"
    if not cond_tif.exists():
        log(f"[ERROR] Missing input raster: {cond_tif}")
        return 1

    out_dir = out_base / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    core_out = out_dir / f"connectedness_{tile_tag}.tif"

    if have_file(core_out):
        log(f"[SKIP] Output exists: {core_out}")
        return 0

    scratch = Path(os.environ.get("TMPDIR", "/tmp"))
    buf_tif = scratch / f"cond_{year}_{tile_tag}_buf.tif"
    buf_out = scratch / f"conn_{year}_{tile_tag}_buf.tif"

    # -----------------------------
    # Stage 1: extract buffered tile
    # -----------------------------
    t0 = time.time()
    log("[STAGE 1/4] Writing buffered condition tile")

    with rasterio.open(cond_tif) as src:
        w = Window(bx0, by0, bx1 - bx0, by1 - by0)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            width=int(w.width),
            height=int(w.height),
            transform=rasterio.windows.transform(w, src.transform),
            count=1,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="LZW",
            BIGTIFF="IF_SAFER",
        )
        data = src.read(1, window=w)

    log(
        f"[INFO] Buffered window size: "
        f"{int(w.width)} x {int(w.height)} "
        f"({int(w.width*w.height):,} pixels)"
    )

    with rasterio.open(buf_tif, "w", **profile) as dst:
        dst.write(data, 1)

    log(f"[TIME] Stage 1 completed in {time.time() - t0:.1f}s")

    # -----------------------------
    # Stage 2: build overviews
    # -----------------------------
    t0 = time.time()
    log("[STAGE 2/4] Creating overviews on buffered tile")

    create_overviews(
        input_raster=str(buf_tif),
        output_raster=None,
        overview_levels=[2, 4, 8, 16, 32],
    )

    log(f"[TIME] Stage 2 completed in {time.time() - t0:.1f}s")

    # -----------------------------
    # Stage 3: connectedness
    # -----------------------------
    t0 = time.time()
    log(
        "[STAGE 3/4] Running connectedness "
        "(this is the long step; no intermediate output expected)"
    )

    connectedness(
        condition_file=str(buf_tif),
        lambdas=[2, 20, 200],
        max_cost=2.0,
        window_size=5,
        outer_window=11,
        levels=[2, 4, 8, 16, 32],
        sigma=1,
        option=3,
        filename=str(buf_out),
    )

    log(f"[TIME] Stage 3 completed in {time.time() - t0:.1f}s")

    # -----------------------------
    # Stage 4: crop to core
    # -----------------------------
    t0 = time.time()
    log("[STAGE 4/4] Cropping buffered output to core tile")

    cx0 = x0 - bx0
    cy0 = y0 - by0
    cx1 = x1 - bx0
    cy1 = y1 - by0

    with rasterio.open(buf_out) as src:
        wcore = Window(cx0, cy0, cx1 - cx0, cy1 - cy0)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            width=int(wcore.width),
            height=int(wcore.height),
            transform=rasterio.windows.transform(wcore, src.transform),
            count=1,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="LZW",
            BIGTIFF="IF_SAFER",
        )
        core = src.read(1, window=wcore)

    with rasterio.open(core_out, "w", **profile) as dst:
        dst.write(core, 1)

    log(f"[TIME] Stage 4 completed in {time.time() - t0:.1f}s")

    # -----------------------------
    # Cleanup
    # -----------------------------
    for p in (buf_tif, buf_out):
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    log(
        f"[DONE] year={year} tile={tile_tag} "
        f"total_time={time.time() - t0_all:.1f}s "
        f"output={core_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




# from __future__ import annotations
# import json, os, sys
# from pathlib import Path

# import rasterio
# from rasterio.windows import Window

# from connectivity import create_overviews, connectedness

# def have_file(p: Path) -> bool:
#     return p.exists() and p.stat().st_size > 0

# def main():
#     if len(sys.argv) != 6:
#         print("Usage: run_year_tile_connectedness.py <tiles.json> <year> <tile_id> <cond_dir> <out_base_dir>", file=sys.stderr)
#         return 2

#     tiles_json = Path(sys.argv[1])
#     year = int(sys.argv[2])
#     tile_id = int(sys.argv[3])
#     cond_dir = Path(sys.argv[4])
#     out_base = Path(sys.argv[5])

#     cfg = json.loads(tiles_json.read_text())
#     tiles = cfg["tiles"]
#     t = tiles[tile_id]

#     x0,y0,x1,y1 = t["core"]
#     bx0,by0,bx1,by1 = t["buf"]

#     cond_tif = cond_dir / f"condition_{year}.tif"
#     if not cond_tif.exists():
#         print(f"[ERROR] missing {cond_tif}", file=sys.stderr)
#         return 1

#     # output dir per year
#     out_dir = out_base / str(year)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     tile_tag = f"tx{t['tx']:03d}_ty{t['ty']:03d}"
#     core_out = out_dir / f"connectedness_{tile_tag}.tif"

#     if have_file(core_out):
#         print(f"[SKIP] {year} {tile_tag} exists", flush=True)
#         return 0

#     scratch = Path(os.environ.get("TMPDIR", "/tmp"))
#     buf_tif = scratch / f"cond_{year}_{tile_tag}_buf.tif"
#     buf_out = scratch / f"conn_{year}_{tile_tag}_buf.tif"

#     # 1) write buffered condition tile
#     with rasterio.open(cond_tif) as src:
#         w = Window(bx0, by0, bx1-bx0, by1-by0)
#         profile = src.profile.copy()
#         profile.update(
#             width=int(w.width),
#             height=int(w.height),
#             transform=rasterio.windows.transform(w, src.transform),
#             tiled=True,
#             compress="LZW",
#             BIGTIFF="IF_SAFER",
#         )
#         data = src.read(1, window=w)

#     with rasterio.open(buf_tif, "w", **profile) as dst:
#         dst.write(data, 1)

#     # 2) create overviews IN-PLACE on the buffered tile (package method)
#     create_overviews(
#         input_raster=str(buf_tif),
#         output_raster=None,
#         overview_levels=[2, 4, 8, 16, 32],
#     )

#     # 3) run connectedness on buffered tile
#     connectedness(
#         condition_file=str(buf_tif),
#         lambdas=[2, 20, 200],
#         max_cost=2.0,
#         window_size=5,
#         outer_window=11,
#         levels=[2, 4, 8, 16, 32],
#         sigma=1,
#         option=3,
#         filename=str(buf_out),
#     )

#     # 4) crop buffered output to core
#     cx0 = x0 - bx0
#     cy0 = y0 - by0
#     cx1 = x1 - bx0
#     cy1 = y1 - by0

#     with rasterio.open(buf_out) as src:
#         wcore = Window(cx0, cy0, cx1-cx0, cy1-cy0)
#         profile = src.profile.copy()
#         profile.update(
#             width=int(wcore.width),
#             height=int(wcore.height),
#             transform=rasterio.windows.transform(wcore, src.transform),
#             tiled=True,
#             compress="LZW",
#             BIGTIFF="IF_SAFER",
#         )
#         core = src.read(1, window=wcore)

#     with rasterio.open(core_out, "w", **profile) as dst:
#         dst.write(core, 1)

#     # cleanup scratch
#     for p in (buf_tif, buf_out):
#         try:
#             p.unlink()
#         except FileNotFoundError:
#             pass

#     print(f"[DONE] year={year} tile={tile_tag} -> {core_out}", flush=True)
#     return 0

# if __name__ == "__main__":
#     raise SystemExit(main())
