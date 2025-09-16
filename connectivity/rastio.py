import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from shapely.geometry import box, mapping
from osgeo import gdal
from .utils import guess_geographic


def read_raster(file, gdf=None, levels=None, expand_px=3):
    """
    Reads specified overview levels from a multi-band Cloud-Optimized GeoTIFF (COG) file
    and stores them in a dictionary.

    Parameters:
    - file (str): Path to the COG file.
    - gdf (GeoPandas):
    - levels (list of int): List of overview reduction factors to read (e.g., [2, 4, 8]). None returns all available.
    - expand_px (bool): The number of pixels to pad the polygon; this depends on the 

    Returns:
    - dict: A two dictionaries where keys are overview levels and values are 3D numpy arrays
            with shape (bands, height, width) for the corresponding overview, and their respective transform info.
    """
    data_dict = {}
    tran_dict = {}

    # Get overview levels
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
        if gdf is not None:
            res_x, res_y = src.res
            pad_x = expand_px * res_x
            pad_y = expand_px * res_y
            orig_buffer_geoms = [geom.buffer(max(pad_x, pad_y)) for geom in gdf.geometry]

    # If levels are not provided get them
    levels = overviews if levels is None else levels

    # If gdf is not provided read the entire dataset
    if gdf is None:
        with rasterio.open(file) as dataset:

            original_transform = dataset.transform

            for level in levels:
                if level not in overviews:
                    raise(ValueError(f"Overview level {level} not found in the dataset."))

                # Calculate the shape for output based on the overview level
                scale = level
                out_height = dataset.height // scale
                out_width = dataset.width // scale

                # Update the transform
                scale_x = dataset.width / out_width
                scale_y = dataset.height / out_height
                level_transform = original_transform * rasterio.Affine.scale(scale_x, scale_y)

                # Read all bands at this resolution
                data = dataset.read(
                    out_shape=(
                        dataset.count,
                        out_height,
                        out_width
                    ),
                    resampling=Resampling.average,
                    masked=True
                )
                # Convert any no-data to nan to be skiped in Rust model
                data = np.where(data.mask, np.nan, data)
                    
                data_dict[level] = data.squeeze().astype(np.float32)
                tran_dict[level] = tuple(level_transform)[:6]
    else:
        # Max level to get the correct buffered area that includes all required pixels for the neighbours
        max_level = max(levels)
        max_level_idx = overviews.index(max_level)

        # Use coarsest overview to calculate buffer size to avoid edge effect
        with rasterio.open(file, overview_level=max_level_idx) as src:
            if gdf.crs != src.crs:
                gdf = gdf.to_crs(src.crs)

            res_x, res_y = src.res
            pad_x = expand_px * res_x + (res_x / max(levels) / 4) # Just add 1/4 pixel more to make sure gets all borders
            pad_y = expand_px * res_y + (res_y / max(levels) / 4)
            # Get the bbox of the buffered version to read all overviews based on it
            buffer_geoms = [box(*geom.buffer(max(pad_x, pad_y)).bounds) for geom in gdf.geometry]

        # Get all overview layers
        for level in levels:
            if level not in overviews:
                raise ValueError(f"Overview decimation {level} not found in dataset.")
            
            # NOTE: check this works correctly for all overview levels
            ov_idx = overviews.index(level) + 1  # +1 because 0 is base resolution
            
            # Choose masking geometry
            masking_geom = buffer_geoms if level > 1 else orig_buffer_geoms # gdf.geometry.tolist()
            
            # Read data using desired level
            with rasterio.open(file, overview_level=ov_idx, resampling=Resampling.average) as src:
                # Step 1: Mask with buffered geometry (for extent) of the coarsest level i.e. 32
                out_image, out_transform = mask(
                    src,
                    [mapping(geom) for geom in buffer_geoms],
                    crop=True,
                    filled=True,
                    all_touched=True
                )
                # Step 2: Create rasterized mask of original geometries (same shape as out_image)
                # Geometry mask returns True outside, False inside → invert it
                geom_mask = geometry_mask(
                    geometries=[mapping(g) for g in masking_geom],
                    out_shape=out_image.shape[1:],  # height x width
                    transform=out_transform,
                    invert=True,
                    all_touched=True
                )

                # Step 3: Combine with nodata mask
                nodata = src.nodata
                nodata_mask = (out_image == nodata)

                # Mask everything not in original geometry or nodata
                final_mask = nodata_mask | ~geom_mask[np.newaxis, :, :]  # add band dim
                masked = np.ma.masked_array(out_image, mask=final_mask)

                if level > 1:
                    data_dict[level] = masked[0].squeeze().data.astype(np.float32)
                else:
                    nan_array = masked[0].squeeze().filled(np.nan)
                    data_dict[level] = nan_array.astype(np.float32)
                
                tran_dict[level] = tuple(out_transform)[:6]

    return data_dict, tran_dict, is_geo



def write_raster(in_array, outfile="output.tif", template="somefile.tif"):
    """
    Write a numpy array to a GeoTIFF file using the geographic transformation
    and projection information from a template raster file.
    
    Parameters:
    -----------
    in_array : numpy.ndarray
        Array containing the data to be written to the raster file
    outfile : str, default="output.tif"
        Path to the output GeoTIFF file
    template : str, default="somefile.tif"
        Path to the template GeoTIFF file containing the desired 
        transform and projection information
    
    Returns:
    --------
    None
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
        
        # Write the new raster
        with rasterio.open(outfile, 'w', **meta) as dst:
            if in_array.ndim == 2:
                dst.write(in_array, 1)
            else:
                for i in range(in_array.shape[0]):
                    dst.write(in_array[i], i+1)
                    
    print(f"Successfully wrote raster to {outfile}")


def create_overviews(input_raster, output_raster=None, overview_levels=[1, 2, 4, 8, 16, 32]):
    """
    Reads a raster file and saves it with overviews at specified levels.
    
    Args:
        input_raster (str): Path to the input raster file
        output_raster (str, optional): Path to the output raster file. If None, overviews are added to the input file.
        overview_levels (list, optional): List of overview levels to generate. Default is [1, 2, 4, 8, 16, 32].
        
    Returns:
        bool: True if operation was successful, False otherwise
    """

    resampling_method = "AVERAGE"

    try:
        # If output file is specified, make a copy of the input file first
        if output_raster:
            print(f"Creating a copy of the input raster at {output_raster}")
            gdal.Translate(output_raster, input_raster, format='GTiff', 
                           creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"])
            ds = gdal.Open(output_raster, gdal.GA_Update)
        else:
            # Otherwise, add overviews to the original file
            ds = gdal.Open(input_raster, gdal.GA_Update)
            output_raster = input_raster
        
        if ds is None:
            print(f"Error: Could not open raster file {input_raster if output_raster is None else output_raster}")
            return False
        
        print(f"Building overviews with levels: {overview_levels}")
        print(f"Using resampling method: {resampling_method}")
        
        ds.BuildOverviews(resampling_method, overview_levels)
        
        # Close the dataset to flush changes to disk
        ds = None
        
        print(f"Successfully created overviews for {output_raster}")
        return True
        
    except Exception as e:
        print(f"Error creating overviews: {e}")
        return False


def overview_info(file_path):
    """
    Display information about all available overviews in a COG file.
    
    Parameters:
    - file_path (str): Path to the COG file.
    """
    with rasterio.open(file_path) as dataset:
        print(f"File: {file_path}")
        print(f"Resolution: {dataset.width} x {dataset.height}")
        print(f"Bands: {dataset.count}")
        print(f"CRS: {dataset.crs}")
        
        # Get overview information for each band
        print("\nOverview Information:")
        for band in range(1, dataset.count + 1):
            overviews = dataset.overviews(band)
            print(f"  Band {band} overviews: {overviews}")
            
            # Show resolution for each overview
            if overviews:
                print("  Overview resolutions:")
                for level in overviews:
                    width = dataset.width // level
                    height = dataset.height // level
                    print(f"    Level {level}: {width} x {height}")

