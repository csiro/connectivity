#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import threading, resource
from pathlib import Path
import rasterio

from connectivity import overview_info, create_overviews, connectedness


# ---- CONFIG (single source of truth) ----------------------------------------
OVERVIEW_LEVELS = [2, 4, 8, 16, 32, 64]          # add 64 only if you also create it
CONNECTEDNESS_LEVELS = OVERVIEW_LEVELS       # usually keep these identical
# CONNECTEDNESS_LEVELS = [2, 4, 8, 16, 32, 64]  # only if OVERVIEW_LEVELS also includes 64

LAMBDAS = [2, 10, 20, 40]
OUTER_WINDOW = 11
WINDOW_SIZE = 5
MAX_COST = 2.0
SIGMA = 1
OPTION = 3
# -----------------------------------------------------------------------------


n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
print(f"[INFO] Using n_threads={n_threads}", flush=True)

def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def have_file(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0



# def _get_overview_factors_gdal(tif: Path) -> list[int]:
#     """
#     Return overview decimation factors for band 1 using GDAL.
#     e.g. [2,4,8,16,32]
#     """
#     from osgeo import gdal
#     ds = gdal.Open(str(tif))
#     if ds is None:
#         return []
#     b = ds.GetRasterBand(1)
#     n = b.GetOverviewCount()
#     if n <= 0:
#         return []
#     x0, y0 = b.XSize, b.YSize
#     facs = []
#     for i in range(n):
#         ov = b.GetOverview(i)
#         # factor is approx base_size / ov_size
#         fx = int(round(x0 / ov.XSize))
#         fy = int(round(y0 / ov.YSize))
#         # take max to be conservative; should match for square-ish pixels
#         facs.append(max(fx, fy))
#     # unique & sorted
#     return sorted(set(facs))

# def has_required_overviews(tif: Path, required: list[int]) -> bool:
#     """
#     True if tif has all required overview factors on band 1.
#     """
#     try:
#         have = _get_overview_factors_gdal(tif)
#         missing = sorted(set(required) - set(have))
#         if missing:
#             log(f"[INFO] Missing overview levels on {tif.name}: {missing} (have {have})")
#             return False
#         return True
#     except Exception as e:
#         log(f"[WARN] Could not check overviews via GDAL for {tif}: {e}")
#         return False

def overview_factors_rasterio(tif: Path) -> list[int]:
    with rasterio.open(tif) as ds:
        base_w, base_h = ds.width, ds.height
        ovs = ds.overviews(1)  # list of decimation ints if available; sometimes empty
        if ovs:
            # If rasterio returns decimation ints directly, use them
            return sorted(set(int(x) for x in ovs))
        # Fallback: compute factors from overview shapes (rarely needed)
        return []

def has_required_overviews(tif: Path, required: list[int], tol: int = 1) -> bool:
    try:
        have = overview_factors_rasterio(tif)
        if not have:
            return False
        missing = []
        for req in required:
            if not any(abs(h - req) <= tol for h in have):
                missing.append(req)
        if missing:
            log(f"[INFO] Missing overview levels (tol=±{tol}) on {tif.name}: {missing} (have {have})")
            return False
        return True
    except Exception as e:
        log(f"[WARN] Could not check overviews via rasterio for {tif}: {e}")
        return False
    
def start_heartbeat(label: str, every_sec: int = 600):
    """
    Note: if connectedness() holds the GIL, this may not print during the call.
    Use sstat for guaranteed progress monitoring.
    """
    stop = threading.Event()

    def _beat():
        t0 = time.time()
        while not stop.wait(every_sec):
            elapsed_min = (time.time() - t0) / 60
            maxrss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            log(f"[HEARTBEAT] {label} elapsed={elapsed_min:.1f} min maxRSS={maxrss_mb:.0f} MB")

    th = threading.Thread(target=_beat, daemon=True)
    th.start()
    return stop, th


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: connectivity_year.py <YEAR>", file=sys.stderr)
        return 2

    year = int(sys.argv[1])
    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    log(f"[INFO] YEAR={year} n_threads={n_threads}")
    log(f"[INFO] OVERVIEW_LEVELS={OVERVIEW_LEVELS}")
    log(f"[INFO] CONNECTEDNESS_LEVELS={CONNECTEDNESS_LEVELS}")

    base = Path("/datasets/work/lw-nkmp/work/Koala_Habitat_Condition_Mapping_Project/Data/outputs")
    in_dir = base / "condition_mosaics"
    ovr_dir = base / "condition_mosaics_overview"
    out_dir = base / "connectedness"

    # swap back to condition_{year}.tif when ready
    in_tif = in_dir / f"condition_{year}.tif"
    ovr_tif = ovr_dir / f"condition_{year}.tif"
    out_tif = out_dir / f"connectedness_64_{year}.tif"

    if not in_tif.exists():
        log(f"[ERROR] Missing input: {in_tif}")
        return 1

    ovr_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Step 1: ensure overviews -----
    if ovr_tif.exists() and has_required_overviews(ovr_tif, OVERVIEW_LEVELS):
        log(f"[SKIP] Overviews OK: {ovr_tif}")
        # human-readable confirmation (optional but nice)
        overview_info(str(ovr_tif))
    else:
        log(f"[INFO] Creating/refreshing overviews for: {ovr_tif.name}")
        log("[INFO] Input overview status:")
        overview_info(str(in_tif))

        create_overviews(
            input_raster=str(in_tif),
            output_raster=str(ovr_tif),
            overview_levels=OVERVIEW_LEVELS,
        )

        log("[INFO] Output overview status:")
        overview_info(str(ovr_tif))

        # hard check
        if not has_required_overviews(ovr_tif, OVERVIEW_LEVELS):
            log(f"[ERROR] Overviews still missing after create_overviews(): {ovr_tif}")
            return 3

    # ----- Step 2: connectedness -----
    if have_file(out_tif):
        log(f"[SKIP] Connectedness exists: {out_tif}")
        return 0

    log(f"[INFO] Running connectedness → {out_tif}")
    stop, th = start_heartbeat(f"year={year}", every_sec=600)

    try:
        connectedness(
            condition_file=str(ovr_tif),
            lambdas=LAMBDAS,
            max_cost=MAX_COST,
            window_size=WINDOW_SIZE,
            outer_window=OUTER_WINDOW,
            levels=CONNECTEDNESS_LEVELS,
            sigma=SIGMA,
            option=OPTION,
            n_threads=n_threads,
            filename=str(out_tif),
        )
    finally:
        stop.set()
        th.join(timeout=1)

    log(f"[DONE] {year}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





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





# def start_heartbeat(label: str, every_sec: int = 300):
#     stop = threading.Event()

#     def _beat():
#         t0 = time.time()
#         while not stop.wait(every_sec):
#             elapsed_min = (time.time() - t0) / 60
#             # ru_maxrss is KB on Linux
#             maxrss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
#             print(f"[HEARTBEAT] {label} elapsed={elapsed_min:.1f} min maxRSS={maxrss_mb:.0f} MB time={time.strftime('%F %T')}", flush=True)

#     th = threading.Thread(target=_beat, daemon=True)
#     th.start()
#     return stop, th


# def main() -> int:
#     if len(sys.argv) != 2:
#         print("Usage: connectivity_year.py <YEAR>", file=sys.stderr)
#         return 2

#     year = int(sys.argv[1])

#     base = Path("/datasets/work/lw-nkmp/work/Koala_Habitat_Condition_Mapping_Project/Data/outputs")
#     in_dir = base / "condition_mosaics"
#     ovr_dir = base / "condition_mosaics_overview"
#     out_dir = base / "connectedness"

#     # in_tif = in_dir / f"condition_{year}.tif"
#     # ovr_tif = ovr_dir / f"condition_{year}.tif"
#     # out_tif = out_dir / f"connectedness_{year}.tif"

#     in_tif = in_dir / f"seq_{year}.tif"
#     ovr_tif = ovr_dir / f"seq_{year}.tif"
#     out_tif = out_dir / f"connectedness_64_{year}.tif"

#     if not in_tif.exists():
#         print(f"[ERROR] Missing input: {in_tif}", file=sys.stderr)
#         return 1

#     ovr_dir.mkdir(parents=True, exist_ok=True)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # ----- Step 1: create overviews (authoritative method) -----
#     if ovr_tif.exists() and has_any_overviews(ovr_tif):
#         print(f"[SKIP] Overview raster exists and has overviews: {ovr_tif}", flush=True)
#     else:
#         print(f"[INFO] Creating overview raster for {year}", flush=True)
#         print("[INFO] Input overview status:", flush=True)
#         overview_info(str(in_tif))

#         create_overviews(
#             input_raster=str(in_tif),
#             output_raster=str(ovr_tif),
#             overview_levels=[2, 4, 8, 16, 32],
#         )

#         print("[INFO] Output overview status:", flush=True)
#         overview_info(str(ovr_tif))

#     # ----- Step 2: connectedness -----
#     if have_file(out_tif):
#         print(f"[SKIP] Connectedness exists: {out_tif}", flush=True)
#         return 0

#     print(f"[INFO] Running connectedness for {year} → {out_tif}", flush=True)
#     stop, th = start_heartbeat(f"year={year}", every_sec=600)


#     try:
#         connectedness(
#             condition_file=str(ovr_tif),
#             lambdas=[2, 10, 20, 40],
#             max_cost=2.0,
#             window_size=5,
#             outer_window=11,
#             levels=[2, 4, 8, 16, 32, 64],
#             sigma=1,
#             option=3,
#             n_threads=n_threads, 
#             filename=str(out_tif),
#         )
#     finally:
#         stop.set()
#         th.join(timeout=1)

#     print(f"[DONE] {year}", flush=True)
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())


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
