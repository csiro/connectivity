# Tiled connectedness (year × tile) with buffer halo

## What this script does

`run_year_tile_connectedness.py` computes **connectedness / connected habitat** for a **single year** and a **single tile** of a large condition raster, using a **buffer (halo)** to avoid edge artifacts.

It is designed to be run as a **Slurm array job**, where each task is one `(year, tile_id)` pair.

### Why tiling is needed

Running `connectedness()` over the full Australia raster can take **days** at large radii (e.g., `outer_window=11` and `max(level)=32`). Tiling allows:

* smaller, more schedulable jobs
* parallel execution
* easier retries (rerun only failed tiles)
* predictable memory usage

### Why the buffer halo is required

Connectedness uses a search radius that extends beyond each cell. If you run the algorithm on a tile without extra context, cells near the tile edge will be wrong because the algorithm can’t “see” habitat just outside the tile.

So for each tile, we compute connectedness on a **buffered window**:

```
buffered tile = core tile + halo (buffer pixels on all sides)
```

Then we crop the result back to the **core tile** only. When all core tiles are mosaicked, seams are minimized (assuming buffer is at least the max search radius).

---

## Inputs and outputs

### Inputs

* `tiles.json`
  A JSON file containing a list of tiles with pixel windows:

  * `core`: `[x0, y0, x1, y1]` (core tile window in pixels)
  * `buf`:  `[bx0, by0, bx1, by1]` (buffered window in pixels)
  * plus `tx`, `ty` indices for naming

* `condition_YYYY.tif`
  The condition raster for a given year, typically:

  ```
  .../outputs/condition_mosaics/condition_2010.tif
  ...
  .../outputs/condition_mosaics/condition_2023.tif
  ```

### Outputs

For each `(year, tile)` the script writes:

```
<out_base_dir>/<year>/connectedness_tx###_ty###.tif
```

Example:

```
.../outputs/connectedness_tiles/2019/connectedness_tx004_ty007.tif
```

These outputs are **core tiles only** (buffer removed), ready for mosaicking.

### Temporary files (per task)

The script writes buffered intermediates into `$TMPDIR` (Slurm job temp), e.g.:

* `cond_<year>_<tile_tag>_buf.tif` (buffered condition tile)
* `conn_<year>_<tile_tag>_buf.tif` (buffered connectedness output)

These are deleted at the end of a successful run.

---

## The algorithm in 4 stages

### Stage 1 — Extract the buffered condition tile

* Reads `condition_YYYY.tif`
* Extracts the buffered pixel window `buf=[bx0,by0,bx1,by1]`
* Writes a small GeoTIFF to `$TMPDIR` (`buf_tif`)

This reduces I/O and ensures the connectivity algorithm is working on a compact raster.

### Stage 2 — Build overviews on the buffered tile (required)

Connectivity relies on overview pyramids to perform multi-resolution computation efficiently.

We run:

```python
create_overviews(input_raster=buf_tif, output_raster=None, overview_levels=[2,4,8,16,32])
```

Key points:

* `output_raster=None` means **in-place** overview creation.
* Overviews are stored **inside the buffered tile GeoTIFF** (like `gdaladdo`).
* Overviews are **not constructed on the fly**; they are real on-disk pyramid layers.
* These overviews exist only for the buffered tile and are deleted with it.

### Stage 3 — Run connectedness on the buffered tile

We run:

```python
connectedness(condition_file=buf_tif, ..., outer_window=11, levels=[2,4,8,16,32], filename=buf_out)
```

This is the expensive step. It may take minutes to hours depending on:

* radius (`outer_window × max(level) × pixel_size`)
* tile size
* disk throughput

There is typically **no incremental output** during this phase.

### Stage 4 — Crop buffered output back to the core tile

We compute the core window’s coordinates relative to the buffered window:

```
cx0 = x0 - bx0
cy0 = y0 - by0
cx1 = x1 - bx0
cy1 = y1 - by0
```

Then we read that window from `buf_out` and write it to the final core tile output:

```
connectedness_tx###_ty###.tif
```

This ensures the final tiles align perfectly to the original grid without halos.

---

## How to choose the buffer size

The buffer must be at least the **maximum effective radius** (in pixels) implied by your parameters:

```
radius_m ≈ outer_window × max(levels) × pixel_size_m
buffer_px ≈ radius_m / pixel_size_m
```

For the koala condition rasters:

* pixel size is ~110–135 m
* with `outer_window=11` and `max(level)=32`, the radius is ~40–50 km
* buffer of ~350–450 pixels is typically sufficient

If the buffer is too small, you will see seam artifacts at tile edges.

---

## Idempotency and reruns

* If the final core tile output already exists and is non-empty, the script prints `[SKIP]` and exits.
* This allows safe reruns and partial completion recovery (rerun array; completed tiles skip).

---

## Logging and “is it alive?”

The script is intentionally stage-verbose (recommended), because the connectedness call can run for a long time without producing output.

Expected log flow:

* `[STAGE 1/4]` buffered extraction
* `[STAGE 2/4]` overview building
* `[STAGE 3/4]` connectedness (long)
* `[STAGE 4/4]` cropping + write core output
* `[DONE]`

For deeper monitoring during Stage 3, use Slurm tools:

* `sstat` for CPU/mem/I/O
* `sacct` after completion

---

## Typical Slurm usage

This script is meant to be called by an array launcher that maps `SLURM_ARRAY_TASK_ID` → `(year, tile_id)`.

Pseudo-mapping:

```
N_TILES = len(tiles)
YEAR = 2010 + task_id // N_TILES
TILE = task_id % N_TILES
```

---

## Mosaicking (after all tiles complete)

Once all tiles for a given year are finished:

1. Build a VRT:

```bash
gdalbuildvrt connectedness_2019.vrt connectedness_tiles/2019/connectedness_tx*_ty*.tif
```

2. Translate to a compressed tiled GeoTIFF:

```bash
gdal_translate -of GTiff -co TILED=YES -co COMPRESS=LZW -co BIGTIFF=IF_SAFER connectedness_2019.vrt connectedness_2019.tif
```

(Optional) add overviews to the final mosaic:

```bash
gdaladdo -r average connectedness_2019.tif 2 4 8 16 32
```

---

## Safety: running alongside other pipelines

This tiling pipeline:

* reads `condition_YYYY.tif` (read-only)
* writes outputs to its own directory structure (`connectedness_tiles/<year>/...`)
* uses per-job `$TMPDIR` intermediates

It can run side-by-side with full-domain connectedness jobs without file conflicts (unless both write to the same output paths).

---

## Connectedness parameterisation and effective radius

--- 

## Summary (TL;DR)

* **Effective connectedness radius**: ~40–50 km
* **Controlled by**: `outer_window × max(levels) × pixel_size`
* **Buffer required**: ≥ ~350 pixels (~45 km)
* **Chosen as a pragmatic compromise** between ecological scale and HPC feasibility
* **Tiling is mandatory** at this radius for national-scale rasters

---

This pipeline uses the following **connectedness parameters**, chosen to balance ecological realism with computational feasibility on HPC.

### Parameter values in use

```python
connectedness(
    condition_file=...,
    lambdas=[2, 20, 200],
    max_cost=2.0,
    window_size=5,
    outer_window=11,
    levels=[2, 4, 8, 16, 32],
    sigma=1,
    option=3,
)
```

Only a subset of these parameters control the **spatial radius** of interaction. The critical ones are:

| Parameter      |               Value | Role                                                    |
| -------------- | ------------------: | ------------------------------------------------------- |
| `levels`       | `[2, 4, 8, 16, 32]` | Multi-resolution overview pyramid factors               |
| `outer_window` |                `11` | Half-window (in overview pixels) searched at each level |
| `pixel_size`   |          ~110–135 m | Native grid resolution (EPSG:9473 Albers)               |

---

## How the effective search radius is computed

The connectivity algorithm operates hierarchically across overview levels.
The **maximum interaction distance** is approximately:

```
max_distance ≈ outer_window × max(levels) × pixel_size
```

### For this project

* **Maximum overview level**:

  ```
  max(levels) = 32
  ```

* **Outer window**:

  ```
  outer_window = 11
  ```

* **Native pixel size (EPSG:9473)**:
  From `gdalinfo` on condition rasters:

  ```
  Pixel Size ≈ 110–135 m
  ```

### Resulting radius

Using a conservative mid-range pixel size (~125 m):

```
max_distance ≈ 11 × 32 × 125 m
             ≈ 44,000 m
             ≈ 44 km
```

So the **connectedness radius is ~40–50 km**, depending on local pixel dimensions.

This is the **radius that must be fully contained within each tile’s buffer** to avoid edge effects.

---

## Implications for tiling and buffering

### Required buffer size (rule of thumb)

To ensure correctness at tile edges:

```
buffer_pixels ≥ outer_window × max(levels)
```

For this configuration:

```
buffer_pixels ≥ 11 × 32 = 352 pixels
```

Converted to distance:

```
352 px × ~125 m ≈ 44 km
```

This is why the tiling JSON defines **core** and **buffered** windows, and why connectedness is run on the buffered tile and cropped back to the core.

---

## Why we are *not* increasing the radius (for now)

Increasing the radius can be done by increasing either:

* `outer_window`, or
* the maximum value in `levels`

However, runtime and memory cost scale **super-linearly** with radius. Empirically:

* Southeast QLD (small extent): minutes
* Full Australia (single raster): days
* Full Australia (tiled, buffered): hours–days depending on tile size

Given current runtime (~4–9 hours per Australia-scale year at this configuration), we **intentionally fix**:

```
outer_window = 11
levels = [2, 4, 8, 16, 32]
```

and focus on:

* tiling
* buffering
* robust HPC execution

Larger radii can be explored later once baseline products are complete.

---


