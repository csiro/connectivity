import rasterio
import numpy as np

def read_overviews(file_path, levels):
    """
    Reads specified overview levels from a Cloud-Optimized GeoTIFF (COG) file
    and stores them in a dictionary.

    Parameters:
    - file_path (str): Path to the COG file.
    - levels (list of int): List of overview levels to read.

    Returns:
    - dict: A dictionary where keys are overview levels and values are the corresponding data arrays.
    """
    data_dict = {}

    with rasterio.open(file_path) as dataset:
        # Ensure the dataset has overviews
        if not dataset.overviews(1):
            raise ValueError("The dataset does not contain any overviews.")

        for level in levels:
            # Check if the requested level exists in the dataset's overviews
            if level in dataset.overviews(1):
                # Read the data for the specified overview level
                data = dataset.read(1, out_shape=(
                    dataset.height // level,
                    dataset.width // level
                ))
                data_dict[level] = data.astype(np.float32)
            else:
                print(f"Overview level {level} not found in the dataset.")

    return data_dict


import numpy as np
import rasterio


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

