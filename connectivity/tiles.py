import math
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box


def _check_positive_int(value, name):
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _aligned_breaks(size, parts, align_to):
    _check_positive_int(size, "size")
    _check_positive_int(parts, "parts")
    _check_positive_int(align_to, "align_to")

    if parts == 1:
        return [0, size]

    internal_multiples = (size - 1) // align_to
    if internal_multiples < parts - 1:
        raise ValueError(
            f"Cannot split {size} pixels into {parts} non-empty aligned parts "
            f"with align_to={align_to}."
        )

    breaks = [0]
    for k in range(1, parts):
        remaining_internal = parts - k - 1
        min_block = breaks[-1] // align_to + 1
        max_block = internal_multiples - remaining_internal

        target = math.floor((k * size) / (parts * align_to) + 0.5)
        block = min(max(target, min_block), max_block)
        breaks.append(block * align_to)

    breaks.append(size)
    return breaks


def _check_non_negative_number(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite number")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _check_rectilinear(transform):
    if not math.isclose(transform.b, 0.0) or not math.isclose(transform.d, 0.0):
        raise ValueError("make_tiles currently supports only north-up rectilinear rasters.")


def _balanced_breaks(costs, parts, align_to, size):
    _check_positive_int(parts, "parts")

    costs = np.asarray(costs, dtype=np.float64)
    if costs.ndim != 1:
        raise ValueError("costs must be one-dimensional")

    nblocks = costs.size
    if nblocks < parts:
        raise ValueError(
            f"Cannot split {nblocks} aligned blocks into {parts} non-empty parts."
        )

    total_cost = float(np.nansum(costs))
    if total_cost <= 0:
        return _aligned_breaks(size, parts, align_to)

    cumsum = np.cumsum(np.nan_to_num(costs, nan=0.0, posinf=0.0, neginf=0.0))
    breaks = [0]
    previous_block = 0

    for k in range(1, parts):
        remaining_parts = parts - k
        min_block = previous_block + 1
        max_block = nblocks - remaining_parts
        target = total_cost * k / parts

        candidate = int(np.searchsorted(cumsum, target, side="left")) + 1
        candidates = [candidate]
        if candidate > 1:
            candidates.append(candidate - 1)

        block = min(max(candidate, min_block), max_block)
        best_error = abs(cumsum[block - 1] - target)
        for item in candidates:
            item = min(max(item, min_block), max_block)
            error = abs(cumsum[item - 1] - target)
            if error < best_error:
                block = item
                best_error = error

        breaks.append(min(block * align_to, size))
        previous_block = block

    breaks.append(size)
    return breaks


def _block_costs(src, align_to, io_weight):
    _check_non_negative_number(io_weight, "io_weight")

    block_rows = math.ceil(src.height / align_to)
    block_cols = math.ceil(src.width / align_to)

    mask = src.read_masks(
        1,
        out_shape=(block_rows, block_cols),
        resampling=Resampling.average,
    ).astype(np.float32)
    valid_fraction = mask / 255.0

    row_sizes = np.full(block_rows, align_to, dtype=np.float32)
    col_sizes = np.full(block_cols, align_to, dtype=np.float32)
    row_sizes[-1] = src.height - align_to * (block_rows - 1)
    col_sizes[-1] = src.width - align_to * (block_cols - 1)
    block_area = row_sizes[:, None] * col_sizes[None, :]

    return block_area * (valid_fraction + float(io_weight))


def _make_records(src, nrows, ncols, align_to, row_breaks, col_breaks_by_row, tile_costs=None):
    records = []
    tile_id = 0
    for row in range(nrows):
        row_start = row_breaks[row]
        row_stop = row_breaks[row + 1]
        col_breaks = col_breaks_by_row[row]
        for col in range(ncols):
            col_start = col_breaks[col]
            col_stop = col_breaks[col + 1]

            window = Window(
                col_off=col_start,
                row_off=row_start,
                width=col_stop - col_start,
                height=row_stop - row_start,
            )
            left, bottom, right, top = window_bounds(window, src.transform)

            record = {
                "id": tile_id,
                "row": row,
                "col": col,
                "row_start": row_start,
                "row_stop": row_stop,
                "col_start": col_start,
                "col_stop": col_stop,
                "align_to": align_to,
                "geometry": box(left, bottom, right, top),
            }
            if tile_costs is not None:
                record["cost"] = float(tile_costs[row][col])
            records.append(record)
            tile_id += 1

    return records


def make_tiles(
    raster_file: str,
    nrows: int,
    ncols: int,
    align_to: int = 1,
    balanced: bool = False,
    io_weight: float = 0.1,
) -> gpd.GeoDataFrame:
    """Create non-overlapping, aggregation-aligned tile core polygons.

    Internal row and column breaks are snapped to multiples of ``align_to`` pixels
    from the source raster origin. The outer raster edge is always preserved, even
    when the raster dimension is not divisible by ``align_to``.

    If ``balanced=True``, row and per-row column breaks are chosen from a
    deterministic raster mask cost estimate. ``io_weight`` adds a per-pixel read
    cost to every aligned block so mostly empty regions still carry some cost.
    """
    _check_positive_int(nrows, "nrows")
    _check_positive_int(ncols, "ncols")
    _check_positive_int(align_to, "align_to")
    if not isinstance(balanced, bool):
        raise TypeError("balanced must be a boolean")

    with rasterio.open(raster_file) as src:
        _check_rectilinear(src.transform)

        if balanced:
            costs = _block_costs(src, align_to, io_weight)
            row_breaks = _balanced_breaks(costs.sum(axis=1), nrows, align_to, src.height)
            col_breaks_by_row = []
            tile_costs = []
            for row in range(nrows):
                row_block_start = row_breaks[row] // align_to
                row_block_stop = math.ceil(row_breaks[row + 1] / align_to)
                col_costs = costs[row_block_start:row_block_stop, :].sum(axis=0)
                col_breaks = _balanced_breaks(col_costs, ncols, align_to, src.width)
                col_breaks_by_row.append(col_breaks)

                row_costs = []
                for col in range(ncols):
                    col_block_start = col_breaks[col] // align_to
                    col_block_stop = math.ceil(col_breaks[col + 1] / align_to)
                    row_costs.append(
                        costs[
                            row_block_start:row_block_stop,
                            col_block_start:col_block_stop,
                        ].sum()
                    )
                tile_costs.append(row_costs)
        else:
            row_breaks = _aligned_breaks(src.height, nrows, align_to)
            col_breaks = _aligned_breaks(src.width, ncols, align_to)
            col_breaks_by_row = [col_breaks for _ in range(nrows)]
            tile_costs = None

        records = _make_records(
            src,
            nrows,
            ncols,
            align_to,
            row_breaks,
            col_breaks_by_row,
            tile_costs=tile_costs,
        )

        return gpd.GeoDataFrame(records, geometry="geometry", crs=src.crs)


def make_tile(
    raster_file: str,
    nrows: int,
    ncols: int,
    tile_id: int,
    align_to: int = 1,
    balanced: bool = False,
    io_weight: float = 0.1,
) -> gpd.GeoDataFrame:
    """Create one aggregation-aligned tile core polygon by zero-based tile id."""
    _check_positive_int(nrows, "nrows")
    _check_positive_int(ncols, "ncols")
    if not isinstance(tile_id, int):
        raise TypeError("tile_id must be an integer")

    tile_count = nrows * ncols
    if tile_id < 0 or tile_id >= tile_count:
        raise ValueError(f"tile_id must be in 0..{tile_count - 1}, got {tile_id}")

    tiles = make_tiles(
        raster_file,
        nrows=nrows,
        ncols=ncols,
        align_to=align_to,
        balanced=balanced,
        io_weight=io_weight,
    )
    return tiles.iloc[[tile_id]].copy()
