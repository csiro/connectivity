import rasterio
import numpy as np
from osgeo import gdal
from rasterio.enums import Resampling


# Check grids are equal
def check_grids(x, y):
    x_shape = x if isinstance(x, tuple) else x.shape
    y_shape = y if isinstance(y, tuple) else y.shape
    return x_shape[-2:] == y_shape[-2:]

# Mask final resutls based on input data
def mask_gird(array, mask_path):
    """Mask an array based on the original layer"""
    with rasterio.open(mask_path) as dataset:
        nodata_value = dataset.nodata

        # Read all bands at this resolution
        data = dataset.read(1)
        # Fill the no-data values with 0
        if np.isnan(nodata_value):
            masked_array = np.where(np.isnan(data), np.nan, array)
        else:
            masked_array = np.where(data == nodata_value, np.nan, array)
       
    return masked_array


def read_overviews(file_path, levels=None, fill_nodata=True):
    """
    Reads specified overview levels from a multi-band Cloud-Optimized GeoTIFF (COG) file
    and stores them in a dictionary.

    Parameters:
    - file_path (str): Path to the COG file.
    - levels (list of int): List of overview reduction factors to read (e.g., [2, 4, 8]). None returns all available.
    - fill_nodata (bool): to fill no-data value with 0s. If False, the no-data is returned as 'nodata' dictionary key

    Returns:
    - dict: A dictionary where keys are overview levels and values are 3D numpy arrays
            with shape (bands, height, width) for the corresponding overview.
    """
    data_dict = {}

    with rasterio.open(file_path) as dataset:
        overviews = dataset.overviews(1)
        nodata_value = dataset.nodata

        if not overviews:
            raise ValueError("The dataset does not contain any overviews.")
        
        if levels is None:
            levels = overviews

        for level in levels:
            if level not in overviews:
                print(f"Overview level {level} not found in the dataset.")
                continue

            # Calculate the shape for output based on the overview level
            scale = level
            out_height = dataset.height // scale
            out_width = dataset.width // scale

            # Read all bands at this resolution
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    out_height,
                    out_width
                ),
                resampling=Resampling.average
            )
            # Fill the no-data values with 0
            if fill_nodata:
                if np.isnan(nodata_value):
                    data = np.where(np.isnan(data), 0, data)
                else:
                    data = np.where(data == nodata_value, 0, data)
            # Insert to dictionary 
            data_dict[level] = data.squeeze().astype(np.float32)

        if not fill_nodata:
            data_dict['nodata'] = nodata_value

    return data_dict


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
    
    Examples:
    ---------
    >>> import numpy as np
    >>> # Create a sample array
    >>> data = np.random.rand(100, 100)
    >>> # Write array to a GeoTIFF using template
    >>> write_raster(data, "new_raster.tif", "existing_raster.tif")
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
        resampling_method (str, optional): The resampling method to use. Default is "AVERAGE".
                                          Other options include "NEAREST", "GAUSS", "CUBIC", etc.
    
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

                    
