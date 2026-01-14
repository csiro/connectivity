import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.features import geometry_mask
from rasterio.shutil import copy as rio_copy
from shapely.geometry import box, mapping
import geopandas as gpd
from .utils import guess_geographic, round_to_pow2


def create_overviews(
        input_raster: str, 
        output_raster: str | None = None, 
        overview_levels: list[int] = [2, 4, 8, 16, 32]
    ):
    """Reads a raster file and saves it with overviews at specified levels,
    using rasterio.
    
    Args:
        input_raster (str): Path to the input raster file
        output_raster (str, optional): Path to the output raster file. If None,
            overviews are added to the input file in-place.
        overview_levels (list, optional): List of overview decimation factors to
            generate (e.g., [2, 4, 8, 16, 32]). Default is [2, 4, 8, 16, 32],
            but levels <= 1 are ignored for overview creation.
        
    Returns:
        No returns
    """

    # rasterio expects factors > 1
    overview_levels = [round_to_pow2(lvl) for lvl in overview_levels if lvl > 1]

    if not overview_levels:
        raise ValueError("overview_levels must contain values > 1.")

    try:
        # If output file is specified, make a copy of the input file first
        if output_raster and (output_raster != input_raster):
            rio_copy(
                input_raster,
                output_raster,
                driver="GTiff",
                compress="LZW",
                tiled=True,
                BIGTIFF="IF_SAFER",
            )
            target_raster = output_raster
        else:
            # Otherwise, add overviews to the original file
            target_raster = input_raster

        # Open the target raster in read/write mode
        with rasterio.open(target_raster, "r+") as dst:
            print(f"Building overviews with levels: {overview_levels}")
            print(f"Using resampling method: average")

            # Build the overviews
            dst.build_overviews(overview_levels, resampling=Resampling.average)

            # Optional: store resampling method in tags (rasterio convention)
            dst.update_tags(ns="rio_overview", resampling="average")

        print(f"Successfully created overviews for {target_raster}")

    except Exception as e:
        print(f"Error creating overviews: {e}")


def overview_info(file_path: str):
    """Display information about all available overviews and their actual dimensions.
     
    Parameters:
    - file_path (str): Path to the TIF/raster file.
    """
    
    print(f"\nFile: {file_path}")
    
    with rasterio.open(file_path) as ds:
        print(f"Resolution: {ds.width} x {ds.height}")
        print(f"Bands: {ds.count}")
        print(f"CRS: {ds.crs}")

        for band in range(1, ds.count + 1):
            overviews = ds.overviews(band)

            if not overviews:
                print(f"No overview for band {band}.")
                continue

            print(f"\nBand {band} overviews: {overviews}")
            print("Overview resolutions:")
            for i, level in enumerate(overviews):
                # Open the actual overview level
                with rasterio.open(file_path, overview_level=i) as src:
                    h, w = src.shape  # (rows, cols)
                print(f"  Level {level}: {w} x {h}")


def has_overview(file_path: str, levels: list = None) -> bool:
    """Check if specified overview levels exist for all bands.
     
    Parameters:
    - file_path (str): Path to the TIF/raster file.
    - levels (list): Overview levels to check for (e.g., [2, 4, 8, 16])
    
    Returns:
    - bool: True if all specified levels exist for all bands
    """
    if levels is None:
        raise ValueError("levels parameter is required")
    
    with rasterio.open(file_path) as ds:
        for band in range(1, ds.count + 1):
            overviews = ds.overviews(band)
            if not overviews:
                print(f"No overview for band {band}.")
                return False
            if not set(levels).issubset(set(overviews)):
                print(f"Band {band} does not contain the required overviews.")
                print(f"Band {band} has overviews: {overviews}, needs: {levels}")
                return False
            
        return True


def read_raster(
        file: str,
        polygon: gpd.GeoDataFrame | None = None,
        levels: list[int] | None = None, 
        scale: float | None = None, 
        expand_px: int = 3
    ):
    """Reads specified overview levels from a multi-band GeoTIFF file
    and stores them in a dictionary.

    Parameters:
    - file (str): Path to the GoeTIFF or raster file with overviews.
    - polygon (GeoPandas):
    - levels (list of int): List of overview reduction factors to read (e.g., [2, 4, 8]). None returns all available.
    - scale (float or None): Scaling factor. If None, 0, or 1, the data is returned unchanged; otherwise it is divided by scale.
    - expand_px (bool): The number of pixels to pad the polygon; this depends on the window size selected

    Returns:
    - dict: A two dictionaries where keys are overview levels and values are 3D numpy arrays
            with shape (bands, height, width) for the corresponding overview, and their respective transform info.
    """
    data_dict = {}
    tran_dict = {}

    if expand_px < 1:
        raise ValueError("expand_px must be larger than 1.")
    
    # Get overview levels, and the bbox geometry
    with rasterio.open(file) as src:
        # Is the crs geographic or projected?
        try:
            if src.crs is None:
                is_geo = guess_geographic(src)
            else:
                is_geo = src.crs.is_geographic
        except Exception as e:
            raise RuntimeError(f"Error reading CRS info: {e}")
        # Read the overviews
        overviews = src.overviews(1)
        if not overviews:
            raise ValueError("The dataset does not contain any overviews.")
        # Get the original overviews
        if polygon is not None:
            res_x, res_y = src.res
            pad_x = expand_px * res_x
            pad_y = expand_px * res_y
            orig_buffer_geoms = [geom.buffer(max(pad_x, pad_y)) for geom in polygon.geometry]

    # Round to the nearest pow 2; fixes an issue with rasterio/gdal overview level naming
    # Also, overviews levels are always higher than one, e.g. 2, 4, 8...
    overviews = [round_to_pow2(x) for x in overviews if x > 1]
    # If levels are not provided get them; make sure 1 is there and keep unique records
    levels = sorted(set(overviews if levels is None else levels) | {1})

    # If polygon is not provided read the entire dataset
    if polygon is None:
        # Read all overview levels; here levels must conatin 1 as well!
        for i, level in enumerate(levels):
            # check user-supplied levels with corrected overviews
            if level != 1 and level not in overviews:
                raise ValueError(f"Overview level {level} not found in the dataset.")
            
            index = i - 1 # index -1 is the original resolution
            with rasterio.open(file, overview_level=index, resampling=Resampling.average) as dataset:
                level_transform = dataset.transform
                # Read the masked data
                data = dataset.read(masked=True)
                
            # Convert any no-data to nan to be skiped in Rust model
            data_array = np.where(data.mask, np.nan, data).squeeze().astype(np.float32)
            # Change the array memory arrangement for faster access in Rust
            if data_array.ndim > 2:
                data_array = np.stack(data_array, axis=-1)
            data_dict[level] = data_array / scale if scale not in (None, 0, 1) else data_array
            tran_dict[level] = tuple(level_transform)[:6]
    
    # else read only padded polygon area 
    else:
        # Max level to get the correct buffered area that includes all required pixels for the neighbours
        max_level = len(overviews) - 1 # needs the index of the last one
        # Use coarsest overview to calculate buffer size to avoid edge effect; for all levels
        with rasterio.open(file, overview_level=max_level) as src:
            if polygon.crs != src.crs:
                print("Transforming polyong CRS to match the raster file.")
                polygon = polygon.to_crs(src.crs)
            # Get the buffered geom bbox
            res_x, res_y = src.res
            pad_x = expand_px * res_x + (res_x / max(levels) / 4) # Just add 1/4 pixel more to make sure gets all borders
            pad_y = expand_px * res_y + (res_y / max(levels) / 4)
            # Get the bbox of the buffered version to read all overviews based on it
            buffer_geoms = [box(*geom.buffer(max(pad_x, pad_y)).bounds) for geom in polygon.geometry]

        # Read all overview layers
        for i, level in enumerate(levels):
            if level != 1 and level not in overviews:
                raise ValueError(f"Overview decimation {level} not found in dataset.")
            
            # Choose masking geometry based on level; for base level get base_level geom so the output map
            # be the same size and shape of the original layer; for the rest get the biggest buffe
            masking_geom = buffer_geoms if level > 1 else orig_buffer_geoms # polygon.geometry.tolist()
            
            index = i - 1            
            # Read data using desired level
            with rasterio.open(file, overview_level=index, resampling=Resampling.average) as dataset:
                # Step 1: Mask with buffered geometry (for extent) of the coarsest level i.e. 32
                out_image, out_transform = mask(
                    dataset,
                    [mapping(geom) for geom in buffer_geoms],
                    crop=True,
                    filled=True,
                    all_touched=True
                )
                # Step 2: Create rasterized mask of original geometries (same shape as out_image)
                # Geometry mask returns True outside, False inside → invert it
                geom_mask = geometry_mask(
                    geometries=[mapping(g) for g in masking_geom],
                    out_shape=out_image.shape[1:],  # height x width  ### NOTE: check this line..
                    transform=out_transform,
                    invert=True,
                    all_touched=True
                )

                # Step 3: Combine with nodata mask
                nodata = dataset.nodata
                nodata_mask = (out_image == nodata)

                # Mask everything not in original geometry or nodata
                final_mask = nodata_mask | ~geom_mask[np.newaxis, :, :]  # add band dim
                masked = np.ma.masked_array(out_image, mask=final_mask)
                # Get only the masked areas and convert to f32
                nan_array = masked.squeeze().astype(np.float32).filled(np.nan)
                # Dvivid by the scale to make it in the 0-1 range
                # Change the array memory arrangement for faster access in Rust
                if nan_array.ndim > 2:
                    nan_array = np.stack(nan_array, axis=-1)
                data_dict[level] = nan_array / scale if scale not in (None, 0, 1) else nan_array
                tran_dict[level] = tuple(out_transform)[:6]

    return data_dict, tran_dict, is_geo


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

