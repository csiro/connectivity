from rasterio.features import geometry_window
from rasterio.mask import mask
from rasterio.features import geometry_mask
from rasterio.windows import Window
from affine import Affine
from shapely.geometry import box
from shapely.geometry import mapping
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
import math
import numpy as np
import rasterio
import geopandas as gpd
from rust_conn import pixel_coverage as _rust_pixel_coverage


def _guess_geographic(src):
    """Guess CRS type from transform and bounds when CRS is missing."""
    transform = src.transform
    xres = abs(transform.a)
    yres = abs(transform.e)
    bounds = src.bounds

    # Heuristic 1: resolution
    if xres < 1 and yres < 1:
        return True  # likely geographic (degrees)
    if xres > 1 and yres > 1:
        return False  # likely projected (metres)

    # Heuristic 2: extent ranges
    if (-180 <= bounds.left <= 180 and
        -180 <= bounds.right <= 180 and
        -90 <= bounds.bottom <= 90 and
        -90 <= bounds.top <= 90):
        return True

    return False  # default to projected if unsure


# reading full or masked raster
def read_raster(
    file_path: str,
    polygon: gpd.GeoDataFrame | None = None,
    levels: list[int] | None = None,
    scale: float | None = None,
    expand_px: int = 0,
    valid_margin_px: int = 32,
):
    """Reads data from a multi-band GeoTIFF (base resolution) and returns NaN-filled array + transforms + masks + offsets.

    Parameters:
        - file_path (str): Path to the GeoTIFF or raster file.
        - polygon (GeoPandas): Optional polygon to mask the raster.
        - levels (list of int): List of overview reduction factors (e.g., [1, 2, 4, 8]).
        - scale (float or None): Scaling factor. If None, 0, or 1, data returned unchanged; otherwise divided by scale.
        - expand_px (int): Number of pixels to buffer the polygon for read-window context.
          0 means closed boundary (tight crop).
        - valid_margin_px (int): Extra pixels to buffer the polygon for valid analysis area
          in non-closed mode. This reduces NaN halo before filtering while preserving final
          crop to the original polygon.
    
    Modes:
      - closed (expand_px == 0): crop to polygon extent AND mask outside polygon
      - non-closed (expand_px > 0): crop to bounding box of buffered polygon, DO NOT mask outside polygon

    Returns:
      - data_array: (rows, cols) for single-band or (rows, cols, bands) for multi-band, float32 with NaNs
      - mask_array: 2D bool, True where cells are invalid for analysis (outside valid mask or nodata/NaN)
      - tran_dict: dict[level] -> GDAL-style 6-tuple transform
      - is_geo: bool
      - tile_row0, tile_col0 -> mask/polygon offsets from raster origin for generating consistent overviews in Rust.
    """
        
    if levels is None or len(levels) == 0:
        levels = [1]

    if any(l <= 0 for l in levels):
        raise ValueError(f"levels must be positive ints, got: {levels}")
    if valid_margin_px < 0:
        raise ValueError(f"valid_margin_px must be >= 0, got: {valid_margin_px}")

    # ensure levels contain 1
    levels = sorted({1, *levels})

    # In non-closed mode, valid analysis margin must stay within the expanded read window.
    # Read expansion is computed with (expand_px + 3) * max(levels) base pixels.
    if expand_px > 0:
        max_margin_px = int((expand_px + 3) * max(levels))
        if valid_margin_px > max_margin_px:
            raise ValueError(
                "valid_margin_px cannot exceed the non-closed read-buffer limit: "
                f"{valid_margin_px} > {max_margin_px} "
                f"(=(expand_px + 3) * max(levels), expand_px={expand_px}, levels={levels})"
            )

    with rasterio.open(file_path) as src:
        try:
            is_geo = _guess_geographic(src) if src.crs is None else bool(src.crs.is_geographic)
        except Exception as e:
            raise RuntimeError(f"Error reading CRS info: {e}")

        base_transform = src.transform
        base_res_x, base_res_y = src.res

        if polygon is None:
            data_ma = src.read(masked=True).astype(np.float32)  # (bands, rows, cols)
            out_transform = base_transform
            out_image_data = np.ma.getdata(data_ma)
            data_mask = np.ma.getmaskarray(data_ma)
            core_geom_mask = None
            valid_geom_mask = None
            tile_row0, tile_col0 = 0, 0

        else:
            if polygon.crs != src.crs:
                polygon = polygon.to_crs(src.crs)

            closed = (expand_px == 0)

            if closed:
                extent_geoms = [geom for geom in polygon.geometry]
            else:
                # pad in map units: expand_px (in base pixels) * max_level scaling * max(res)
                max_level = max(levels)
                eps = 0.5 * float(max(base_res_x, base_res_y)) # add half-pixel to avoid missing rows/cols
                # add an offset to expand to ensure all required cells are taken.
                pad_size = float(expand_px + 3) * float(max_level) * float(max(base_res_x, base_res_y)) + eps
                extent_geoms = [box(*geom.buffer(pad_size).bounds) for geom in polygon.geometry]

            # Compute exact dataset-pixel offsets for this crop
            win0 = geometry_window(
                src,
                [mapping(g) for g in extent_geoms],
                pad_x=0,
                pad_y=0,
            ).round_offsets().round_lengths()

            if closed:
                read_win = win0
            else:
                # Snap the non-closed read window to the coarsest overview grid so
                # per-level valid-cell counts are computed from full global blocks.
                # This keeps habitat-area weights tile-invariant while still reflecting
                # partial valid area (e.g., coastlines/nodata).
                snap = int(max(levels))
                r0 = int(win0.row_off)
                c0 = int(win0.col_off)
                r1 = int(win0.row_off + win0.height)
                c1 = int(win0.col_off + win0.width)

                r0 = (r0 // snap) * snap
                c0 = (c0 // snap) * snap
                r1 = ((r1 + snap - 1) // snap) * snap
                c1 = ((c1 + snap - 1) // snap) * snap

                r0 = max(0, r0)
                c0 = max(0, c0)
                r1 = min(src.height, r1)
                c1 = min(src.width, c1)

                read_win = Window(
                    col_off=c0,
                    row_off=r0,
                    width=max(0, c1 - c0),
                    height=max(0, r1 - r0),
                )

            tile_row0 = int(read_win.row_off)
            tile_col0 = int(read_win.col_off)

            if closed:
                # Closed mode: crop to polygon and mask outside geometry
                out_ma, out_transform = mask(
                    src,
                    [mapping(g) for g in extent_geoms],
                    crop=True,
                    filled=False,
                    all_touched=True,
                )
            else:
                # Non-closed mode: read snapped window directly; keep all pixels for context
                out_ma = src.read(window=read_win, masked=True).astype(np.float32)
                out_transform = src.window_transform(read_win)

            out_image_data = np.ma.getdata(out_ma).astype(np.float32)
            data_mask = np.ma.getmaskarray(out_ma)

            core_geom_mask = geometry_mask(
                geometries=[mapping(g) for g in polygon.geometry],
                out_shape=out_image_data.shape[1:],
                transform=out_transform,
                invert=True,
                all_touched=True,
            )

            if closed or valid_margin_px == 0:
                valid_geom_mask = core_geom_mask
            else:
                valid_pad = float(valid_margin_px) * float(max(base_res_x, base_res_y))
                valid_geoms = [geom.buffer(valid_pad) for geom in polygon.geometry]
                valid_geom_mask = geometry_mask(
                    geometries=[mapping(g) for g in valid_geoms],
                    out_shape=out_image_data.shape[1:],
                    transform=out_transform,
                    invert=True,
                    all_touched=True,
                )

            # Add NaNs to data_mask as invalid too (both modes)
            data_mask |= np.isnan(out_image_data)
            # Closed mode: also mask outside original polygon
            if closed:
                data_mask |= ~core_geom_mask[np.newaxis, :, :]

            # Final masked array
            data_ma = np.ma.masked_array(out_image_data, mask=data_mask)

        # masked -> ndarray with NaNs
        data_array = data_ma.filled(np.nan).astype(np.float32, copy=False)
        data_array = np.squeeze(data_array)

        # Build 2D mask_array
        if polygon is not None:
            if data_mask.ndim == 3:
                invalid_2d = np.any(data_mask, axis=0)
            else:
                invalid_2d = data_mask.astype(bool)

            valid_inside = valid_geom_mask & ~invalid_2d
            mask_array = ~valid_inside
        else:
            if data_mask.ndim == 3:
                mask_array = np.any(data_mask, axis=0)
            else:
                mask_array = data_mask.astype(bool)

        # multiband reshape: (bands, rows, cols) -> (rows, cols, bands)
        if data_array.ndim == 3:
            data_array = np.moveaxis(data_array, 0, -1)

        if scale not in (None, 0, 1):
            data_array = data_array / float(scale)

        # Transforms anchored to DATASET origin, shifted to this tile’s overview cell start
        tran_dict: dict[int, tuple[float, float, float, float, float, float]] = {}
        for level in levels:
            t_level = base_transform * Affine.scale(level)
            ov_row_start = tile_row0 // level
            ov_col_start = tile_col0 // level
            overview_transform = t_level * Affine.translation(ov_col_start, ov_row_start)
            tran_dict[int(level)] = tuple(overview_transform)[:6]

        return data_array, mask_array, tran_dict, is_geo, tile_row0, tile_col0


# fractional pixel coverage by polygons
def pixel_coverage(
    polygon: gpd.GeoDataFrame | str,
    raster_file: str,
    n_threads: int | None = None,
    filename: str = "",
):
    """Compute the proportion of each raster pixel covered by polygon(s).

    For every pixel of `raster_file`, returns the fraction (in [0, 1]) of that
    pixel's area covered by the union of all polygons in `polygon`. Implemented
    in Rust with Sutherland-Hodgman per-pixel clipping and Rayon parallelism.

    Parameters
    ----------
    polygon : GeoDataFrame or str
        A GeoPandas GeoDataFrame containing polygon geometries, or a path to a
        polygon file (Shapefile, GeoJSON, etc.) readable by `geopandas.read_file`.
        All geometries are unioned before coverage is computed.
    raster_file : str
        Path to the reference raster file. The output array matches the raster's
        shape and transform; the polygon is reprojected to the raster's CRS if needed.
    n_threads : int, optional
        Number of CPU threads for the Rust computation. Default is None (all cores).
    filename : str, optional
        If a path is given (length > 3), the coverage array is also written to disk
        as a GeoTIFF using `raster_file` as the metadata/CRS template, so the output
        shares `raster_file`'s grid exactly. This makes it directly usable as the
        `pa_file` argument to `connectedness()`. Default is "" (no file written).

    Returns
    -------
    np.ndarray
        2D float32 array of shape `(raster.height, raster.width)` with values in
        [0, 1] giving the fraction of each pixel covered by the polygon(s). When
        `filename` is set, the same array is also written to that path.
    """
    if isinstance(polygon, str):
        polygon = gpd.read_file(polygon)

    with rasterio.open(raster_file) as src:
        raster_transform = src.transform
        raster_shape = (src.height, src.width)
        raster_crs = src.crs

    if raster_crs is not None and polygon.crs is not None and polygon.crs != raster_crs:
        polygon = polygon.to_crs(raster_crs)

    geom = unary_union(list(polygon.geometry.values))
    rings = _polygon_to_rings(geom)
    if not rings:
        coverage = np.zeros(raster_shape, dtype=np.float32)
    else:
        coverage = _rust_pixel_coverage(
            rings,
            tuple(raster_transform)[:6],
            raster_shape,
            n_threads,
        )

    # Optionally persist to a GeoTIFF that inherits raster_file's grid/CRS, so the
    # result is directly usable as `pa_file` in connectedness(). Mirrors the
    # `filename` writing convention used by connectedness().
    if len(filename) > 3:
        write_raster(coverage, outfile=filename, template=raster_file)

    return coverage


def _polygon_to_rings(geom):
    """Flatten a (Multi)Polygon into a list of (exterior, [holes]) coord arrays."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    else:
        raise TypeError(
            f"Geometry must be Polygon or MultiPolygon, got {type(geom).__name__}"
        )

    rings = []
    for p in polys:
        if p.is_empty:
            continue
        ext = np.asarray(p.exterior.coords, dtype=np.float64)
        holes = [np.asarray(h.coords, dtype=np.float64) for h in p.interiors]
        rings.append((ext, holes))
    return rings


def write_raster(
        in_array: str, 
        outfile: str = "output.tif", 
        template: str = "somefile.tif", 
        transform: tuple | Affine = None
    ):
    """Write a numpy array to a GeoTIFF file using the geographic transformation
    from the transform argument and projection/other metadata from a template raster file.
    Args:
        - np.array: input array to write to disk
        - str: output path to the file
        - str: a file to be used as tempalte for getting the meta data for writing
        - tuple: a tuple of transform information for croping the file when writing
    """
    array = np.asanyarray(in_array)
    if np.ma.isMaskedArray(in_array):
        if np.issubdtype(array.dtype, np.floating):
            array = np.ma.filled(in_array, np.nan).astype(array.dtype, copy=False)
        else:
            array = np.ma.filled(in_array)

    # Open the template raster to get metadata
    with rasterio.open(template) as src:
        # Get the metadata from the template
        meta = src.meta.copy()
        
        # Update metadata with new array dimensions and datatype
        meta.update(
            dtype=array.dtype,
            count=1 if array.ndim == 2 else array.shape[0],
            width=array.shape[-1],
            height=array.shape[-2]
        )

        if np.issubdtype(array.dtype, np.floating):
            meta["nodata"] = np.nan
        
        # Update transform if provided
        if transform is not None:
            if isinstance(transform, (tuple, list)):
                meta['transform'] = Affine(*transform[:6])
            else:
                meta['transform'] = transform
        
        # Write the new raster
        with rasterio.open(outfile, 'w', **meta) as dst:
            if array.ndim == 2:
                dst.write(array, 1)
            else:
                for i in range(array.shape[0]):
                    dst.write(array[i], i+1)


def _format_compact(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4g}"


def _format_resolution(x_res: float, y_res: float, unit: str) -> str:
    if np.isclose(x_res, y_res):
        return f"{_format_compact(x_res)} {unit}"
    return f"{_format_compact(x_res)} x {_format_compact(y_res)} {unit}"


def _raster_unit(crs) -> str:
    if crs is None:
        return "map units"
    if crs.is_geographic:
        return "degrees"
    return getattr(crs, "linear_units", None) or "map units"


def _linear_unit_km_factor(crs):
    if crs is None or crs.is_geographic:
        return None

    factor = getattr(crs, "linear_units_factor", None)
    if factor is not None:
        try:
            if isinstance(factor, tuple):
                return float(factor[-1]) / 1000.0
            return float(factor) / 1000.0
        except (TypeError, ValueError):
            pass

    unit = (_raster_unit(crs) or "").lower()
    if unit in {"metre", "meter", "metres", "meters", "m"}:
        return 0.001
    if unit in {"kilometre", "kilometer", "kilometres", "kilometers", "km"}:
        return 1.0
    if unit in {"foot", "feet", "ft", "us survey foot", "us survey feet"}:
        return 0.0003048
    return None


def _resolution_km(x_res: float, y_res: float, crs, bounds):
    if crs is None:
        return None, None, None

    if crs.is_geographic:
        center_lat = (float(bounds.bottom) + float(bounds.top)) / 2.0
        lat_km_per_degree = 111.32
        lon_km_per_degree = lat_km_per_degree * math.cos(math.radians(center_lat))
        return (
            abs(x_res) * abs(lon_km_per_degree),
            abs(y_res) * lat_km_per_degree,
            f"approximate km values use centre latitude {center_lat:.3f}",
        )

    factor = _linear_unit_km_factor(crs)
    if factor is None:
        return None, None, None
    return abs(x_res) * factor, abs(y_res) * factor, None


def resolution_info(
    file_path: str,
    levels: list[int] | None = None,
    min_dimension: int = 16,
    outer_window: int = 5,
):
    """Display expected dimensions and search reach for candidate raster levels.

    This is a planning helper for choosing levels that are not too coarse for a
    given input raster. Levels are generated internally by the Rust library; no
    external GeoTIFF overview layers are required.
    """
    if levels is None:
        levels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    levels = sorted({int(level) for level in levels})
    if not levels or any(level <= 0 for level in levels):
        raise ValueError(f"levels must be positive integers, got: {levels}")
    min_dimension = max(1, int(min_dimension))
    outer_window = max(1, int(outer_window))

    with rasterio.open(file_path) as ds:
        base_width = ds.width
        base_height = ds.height
        base_cells = base_width * base_height
        x_res, y_res = (abs(v) for v in ds.res)
        unit = _raster_unit(ds.crs)
        x_res_km, y_res_km, km_note = _resolution_km(x_res, y_res, ds.crs, ds.bounds)

        print(f"\nFile: {file_path}")
        print(
            "Base raster: "
            f"{base_width} x {base_height} cells "
            f"({_format_compact(base_cells)} total)"
        )
        print(f"Base resolution: {_format_resolution(x_res, y_res, unit)}")
        print(f"CRS: {ds.crs if ds.crs is not None else 'unknown'}")
        print(f"Bands: {ds.count}")
        print(f"Data type(s): {', '.join(ds.dtypes)}")
        print(f"Nodata: {ds.nodata}")
        print(f"Outer window: {outer_window}")
        print(f"Small-dimension warning threshold: < {min_dimension} cells")
        if km_note is not None:
            print(f"Distance note: {km_note}")

        print("\nCandidate internally generated levels:")
        print(
            f"{'Level':>7} {'Width':>8} {'Height':>8} {'Cells':>10} "
            f"{'% base':>8} {'Resolution':>20} {'Reach':>16} {'Reach km':>12}  Note"
        )
        for level in levels:
            width = (base_width + level - 1) // level
            height = (base_height + level - 1) // level
            cells = width * height
            pct_base = 100.0 * cells / base_cells if base_cells else 0.0
            level_res = _format_resolution(x_res * level, y_res * level, unit)
            reach_native = _format_compact(outer_window * level * max(x_res, y_res))
            if x_res_km is None or y_res_km is None:
                reach_km = "-"
            else:
                reach_km = f"{_format_compact(outer_window * level * max(x_res_km, y_res_km))} km"
            note = "small grid; review" if min(width, height) < min_dimension else ""
            print(
                f"{level:>7} {width:>8} {height:>8} "
                f"{_format_compact(cells):>10} {pct_base:>7.3f}% "
                f"{level_res:>20} {reach_native + ' ' + unit:>16} {reach_km:>12}  {note}"
            )


def overview_info(
    file_path: str,
    levels: list[int] | None = None,
    min_dimension: int = 16,
    outer_window: int = 5,
):
    """Backward-compatible wrapper for resolution_info()."""
    return resolution_info(
        file_path=file_path,
        levels=levels,
        min_dimension=min_dimension,
        outer_window=outer_window,
    )
