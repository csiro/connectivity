import math

import geopandas as gpd
import rasterio
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


def _check_rectilinear(transform):
    if not math.isclose(transform.b, 0.0) or not math.isclose(transform.d, 0.0):
        raise ValueError("make_tiles currently supports only north-up rectilinear rasters.")


def make_tiles(raster_file: str, nrows: int, ncols: int, align_to: int = 1) -> gpd.GeoDataFrame:
    """Create non-overlapping, aggregation-aligned tile core polygons.

    Internal row and column breaks are snapped to multiples of ``align_to`` pixels
    from the source raster origin. The outer raster edge is always preserved, even
    when the raster dimension is not divisible by ``align_to``.
    """
    _check_positive_int(nrows, "nrows")
    _check_positive_int(ncols, "ncols")
    _check_positive_int(align_to, "align_to")

    with rasterio.open(raster_file) as src:
        _check_rectilinear(src.transform)

        row_breaks = _aligned_breaks(src.height, nrows, align_to)
        col_breaks = _aligned_breaks(src.width, ncols, align_to)

        records = []
        tile_id = 0
        for row in range(nrows):
            row_start = row_breaks[row]
            row_stop = row_breaks[row + 1]
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

                records.append(
                    {
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
                )
                tile_id += 1

        return gpd.GeoDataFrame(records, geometry="geometry", crs=src.crs)


def make_tile(
    raster_file: str,
    nrows: int,
    ncols: int,
    tile_id: int,
    align_to: int = 1,
) -> gpd.GeoDataFrame:
    """Create one aggregation-aligned tile core polygon by zero-based tile id."""
    _check_positive_int(nrows, "nrows")
    _check_positive_int(ncols, "ncols")
    if not isinstance(tile_id, int):
        raise TypeError("tile_id must be an integer")

    tile_count = nrows * ncols
    if tile_id < 0 or tile_id >= tile_count:
        raise ValueError(f"tile_id must be in 0..{tile_count - 1}, got {tile_id}")

    tiles = make_tiles(raster_file, nrows=nrows, ncols=ncols, align_to=align_to)
    return tiles.iloc[[tile_id]].copy()
