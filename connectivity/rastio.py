from rasterio.features import geometry_window
from rasterio.mask import mask
from rasterio.features import geometry_mask
from rasterio.windows import Window
from affine import Affine
from shapely.geometry import box
from shapely.geometry import mapping
import numpy as np
import rasterio
import geopandas as gpd
from .utils import guess_geographic


# reading full or masked raster
def read_raster(
    file_path: str,
    polygon: gpd.GeoDataFrame | None = None,
    levels: list[int] | None = None,
    scale: float | None = None,
    expand_px: int = 0,
):
    """Reads data from a multi-band GeoTIFF (base resolution) and returns NaN-filled array + transforms + masks + offsets.

    Parameters:
        - file_path (str): Path to the GeoTIFF or raster file.
        - polygon (GeoPandas): Optional polygon to mask the raster.
        - levels (list of int): List of overview reduction factors (e.g., [1, 2, 4, 8]).
        - scale (float or None): Scaling factor. If None, 0, or 1, data returned unchanged; otherwise divided by scale.
        - expand_px (int): Number of pixels to buffer the polygon. 0 means closed boundary (tight crop).
    
    Modes:
      - closed (expand_px == 0): crop to polygon extent AND mask outside polygon
      - non-closed (expand_px > 0): crop to bounding box of buffered polygon, DO NOT mask outside polygon

    Returns:
      - data_array: (rows, cols) for single-band or (rows, cols, bands) for multi-band, float32 with NaNs
      - mask_array: 2D bool, True where inside original polygon AND valid (not nodata/NaN) in the returned extent
      - tran_dict: dict[level] -> GDAL-style 6-tuple transform
      - is_geo: bool
      - tile_row0, tile_col0 -> mask/polygon offsets from raster origin for generating consistent overviews in Rust.
    """
        
    if levels is None or len(levels) == 0:
        levels = [1]

    if any(l <= 0 for l in levels):
        raise ValueError(f"levels must be positive ints, got: {levels}")

    # ensure levels contain 1
    levels = sorted({1, *levels})

    with rasterio.open(file_path) as src:
        try:
            is_geo = guess_geographic(src) if src.crs is None else bool(src.crs.is_geographic)
        except Exception as e:
            raise RuntimeError(f"Error reading CRS info: {e}")

        base_transform = src.transform
        base_res_x, base_res_y = src.res

        if polygon is None:
            data_ma = src.read(masked=True).astype(np.float32)  # (bands, rows, cols)
            out_transform = base_transform
            out_image_data = np.ma.getdata(data_ma)
            data_mask = np.ma.getmaskarray(data_ma)
            geom_mask = None
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
                # add 1 to expand_px so it skip any partial edge
                pad_size = float(expand_px * 2) * float(max_level) * float(max(base_res_x, base_res_y)) + eps
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

            if closed:
                 valid_geoms = [geom for geom in polygon.geometry]
            else:
                 pad_size = float(max(base_res_x, base_res_y)) * 10
                 valid_geoms = [box(*geom.buffer(pad_size).bounds) for geom in polygon.geometry]

            geom_mask = geometry_mask(
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
                data_mask |= ~geom_mask[np.newaxis, :, :]

            # Final masked array
            data_ma = np.ma.masked_array(out_image_data, mask=data_mask)

        # masked -> ndarray with NaNs
        data_array = np.where(data_ma.mask, np.nan, data_ma.data).astype(np.float32)
        data_array = np.squeeze(data_array)

        # Build 2D mask_array
        if polygon is not None:
            if data_mask.ndim == 3:
                invalid_2d = np.any(data_mask, axis=0)
            else:
                invalid_2d = data_mask.astype(bool)

            valid_inside = geom_mask & ~invalid_2d
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
    # Open the template raster to get metadata
    with rasterio.open(template) as src:
        # Get the metadata from the template
        meta = src.meta.copy()
        
        # Update metadata with new array dimensions and datatype
        meta.update(
            dtype=in_array.dtype,
            count=1 if in_array.ndim == 2 else in_array.shape[0],
            width=in_array.shape[-1],
            height=in_array.shape[-2]
        )
        
        # Update transform if provided
        if transform is not None:
            if isinstance(transform, (tuple, list)):
                meta['transform'] = Affine(*transform[:6])
            else:
                meta['transform'] = transform
        
        # Write the new raster
        with rasterio.open(outfile, 'w', **meta) as dst:
            if in_array.ndim == 2:
                dst.write(in_array, 1)
            else:
                for i in range(in_array.shape[0]):
                    dst.write(in_array[i], i+1)


def overview_info(file_path: str, levels: list = None):
    """Display information about possible overview dimensions that could be generated.
     
    Parameters:
    - file_path (str): Path to the TIF/raster file.
    - levels (list): List of overview levels (powers of 2). Default: [2, 4, 8, 16, 32, 64, 128]
    """
    if levels is None:
        levels = [2, 4, 8, 16, 32, 64, 128]
    
    print(f"\nFile: {file_path}")
    
    with rasterio.open(file_path) as ds:
        base_width = ds.width
        base_height = ds.height
        
        print(f"Base Resolution: {base_width} x {base_height}")
        print(f"Bands: {ds.count}")
        print(f"CRS: {ds.crs}")
        
        print(f"\nPossible overview levels:")
        print("Estimated overview resolutions:")
        for level in levels:
            # Calculate dimensions using the same logic as Rust make_overview
            estimated_width = (base_width + level - 1) // level
            estimated_height = (base_height + level - 1) // level            
            # Stop if either dimension is lower than 5
            if estimated_width < 5 or estimated_height < 5:
                break
            
            print(f"  Level {level}: {estimated_width} x {estimated_height}")
