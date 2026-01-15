#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import threading, resource
from pathlib import Path

from connectivity import overview_info, create_overviews, connectedness


def have_file(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0


def has_any_overviews(tif: Path) -> bool:
    """
    Lightweight check using GDAL (preferred). Falls back to False if GDAL not available.
    """
    try:
        from osgeo import gdal
        ds = gdal.Open(str(tif))
        if ds is None:
            return False
        b = ds.GetRasterBand(1)
        return b.GetOverviewCount() > 0
    except Exception:
        return False

def start_heartbeat(label: str, every_sec: int = 300):
    stop = threading.Event()

    def _beat():
        t0 = time.time()
        while not stop.wait(every_sec):
            elapsed_min = (time.time() - t0) / 60
            # ru_maxrss is KB on Linux
            maxrss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"[HEARTBEAT] {label} elapsed={elapsed_min:.1f} min maxRSS={maxrss_mb:.0f} MB time={time.strftime('%F %T')}", flush=True)

    th = threading.Thread(target=_beat, daemon=True)
    th.start()
    return stop, th


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: connectivity_year.py <YEAR>", file=sys.stderr)
        return 2

    year = int(sys.argv[1])

    base = Path("/datasets/work/lw-nkmp/work/Koala_Habitat_Condition_Mapping_Project/Data/outputs")
    in_dir = base / "condition_mosaics"
    ovr_dir = base / "condition_mosaics_overview"
    out_dir = base / "connectedness"

    # in_tif = in_dir / f"condition_{year}.tif"
    # ovr_tif = ovr_dir / f"condition_{year}.tif"
    # out_tif = out_dir / f"connectedness_{year}.tif"

    in_tif = in_dir / f"seq_{year}.tif"
    ovr_tif = ovr_dir / f"seq_{year}.tif"
    out_tif = out_dir / f"connectedness_seq2_{year}.tif"

    if not in_tif.exists():
        print(f"[ERROR] Missing input: {in_tif}", file=sys.stderr)
        return 1

    ovr_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Step 1: create overviews (authoritative method) -----
    if ovr_tif.exists() and has_any_overviews(ovr_tif):
        print(f"[SKIP] Overview raster exists and has overviews: {ovr_tif}", flush=True)
    else:
        print(f"[INFO] Creating overview raster for {year}", flush=True)
        print("[INFO] Input overview status:", flush=True)
        overview_info(str(in_tif))

        create_overviews(
            input_raster=str(in_tif),
            output_raster=str(ovr_tif),
            overview_levels=[2, 4, 8, 16, 32],
        )

        print("[INFO] Output overview status:", flush=True)
        overview_info(str(ovr_tif))

    # ----- Step 2: connectedness -----
    if have_file(out_tif):
        print(f"[SKIP] Connectedness exists: {out_tif}", flush=True)
        return 0

    print(f"[INFO] Running connectedness for {year} → {out_tif}", flush=True)
    stop, th = start_heartbeat(f"year={year}", every_sec=600)


    # shit taking too long?
    # reduce outer_window
    # 11 → 7: radius becomes ~25–30 km
    # 11 → 5: radius becomes ~18–22 km
    # 11 → 3: radius becomes ~11–13 km
    # and reduce max level:
    # If you change levels to [2,4,8,16] (max=16), you half the radius:
    # x: 11 × 16 × 135 m ≈ 23.8 km
    # default outer_window: 11 --> @ level 32, radius ~ 45km
    # default levels=[2, 4, 8, 16, 32],

    # Pixel size ≈ 135.19 m (x), 109.41 m (y)
    # → call it ~120 m effective

    # Max overview level = 32
    # outer_window = 11

    # The connectivity code’s definition (as in the README) is:
    # radius = outer_window × max(level) × pixel_size

    # Plug in the numbers
    # Using x-direction (worst case)
    # radius = 11 × 32 × 135.19 m
    #        ≈ 47,589 m
    #        ≈ 47.6 km

    # Using y-direction
    # radius = 11 × 32 × 109.41 m
    #        ≈ 38,537 m
    #        ≈ 38.5 km

    # Practical interpretation
    # Effective radius: ~40–50 km

    try:
        connectedness(
            condition_file=str(ovr_tif),
            lambdas=[2, 20, 200],
            max_cost=2.0,
            window_size=5,
            outer_window=11,
            levels=[2, 4, 8, 16, 32],
            sigma=1,
            option=3,
            filename=str(out_tif),
        )
    finally:
        stop.set()
        th.join(timeout=1)

    print(f"[DONE] {year}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# #!/usr/bin/env python3
# from __future__ import annotations

# import os
# import sys
# from pathlib import Path

# from connectivity import overview_info, create_overviews, connectedness


# def have_file(p: Path) -> bool:
#     return p.exists() and p.stat().st_size > 0


# def main() -> int:
#     if len(sys.argv) != 2:
#         print("Usage: connectivity_year.py <YEAR>", file=sys.stderr)
#         return 2

#     year = int(sys.argv[1])

#     base = Path("/datasets/work/lw-nkmp/work/Koala_Habitat_Condition_Mapping_Project/Data/outputs")
#     in_dir = base / "condition_mosaics"
#     ovr_dir = base / "condition_mosaics_overview"
#     out_dir = base / "connectedness"

#     in_tif = in_dir / f"condition_{year}.tif"
#     ovr_tif = ovr_dir / f"condition_{year}.tif"
#     out_tif = out_dir / f"connectedness_{year}.tif"

#     if not in_tif.exists():
#         print(f"[ERROR] Missing input: {in_tif}", file=sys.stderr)
#         return 1

#     ovr_dir.mkdir(parents=True, exist_ok=True)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # ----- Step 1: create overviews (authoritative method) -----
#     if have_file(ovr_tif):
#         print(f"[SKIP] Overview raster exists: {ovr_tif}")
#     else:
#         print(f"[INFO] Creating overview raster for {year}")
#         print("[INFO] Input overview status:")
#         overview_info(str(in_tif))

#         # Write to a NEW file (recommended)
#         create_overviews(
#             input_raster=str(in_tif),
#             output_raster=str(ovr_tif),
#             overview_levels=[2, 4, 8, 16, 32],
#         )

#         print("[INFO] Output overview status:")
#         overview_info(str(ovr_tif))

#     # ----- Step 2: connectedness -----
#     if have_file(out_tif):
#         print(f"[SKIP] Connectedness exists: {out_tif}")
#         return 0

#     print(f"[INFO] Running connectedness for {year} → {out_tif}")

#     connectedness(
#         condition_file=str(ovr_tif),
#         lambdas=[2, 20, 200],
#         max_cost=2.0,
#         window_size=5,
#         outer_window=11,
#         levels=[2, 4, 8, 16, 32],
#         sigma=1,
#         option=3,
#         filename=str(out_tif),
#     )

#     print(f"[DONE] {year}")
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())
