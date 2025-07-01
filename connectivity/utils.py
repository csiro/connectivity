import rasterio
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter
import numpy as np
import geopandas as gpd


# Check grids are equal
def check_grids(x, y):
    x_shape = x if isinstance(x, tuple) else x.shape
    y_shape = y if isinstance(y, tuple) else y.shape
    return x_shape[-2:] == y_shape[-2:]


# Mask an array based on the original layer
def mask_gird(array, mask_path):
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


def polygon_mask(gdf, raster, overview_level=1, use_bbox=True):
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise ValueError("First argument must be a geopandas GeoDataFrame")
    
    # Get geometries from GeoDataFrame
    if use_bbox:
        # Use bounding boxes instead of actual geometries
        from shapely.geometry import box
        geometries = [box(*geom.bounds) for geom in gdf.geometry]
    else:
        geometries = gdf.geometry.tolist()
    
    # Get raster info from specific overview level
    with rasterio.open(raster) if isinstance(raster, str) else raster as src:
        # Get overview at specified level
        if overview_level < len(src.overviews(1)):
            overview = src.overviews(1)[overview_level]
            # Calculate overview dimensions and transform
            height = src.height // overview
            width = src.width // overview
            transform = src.transform * src.transform.scale(
                (src.width / width),
                (src.height / height)
            )
        else:
            # Fallback to full resolution if overview doesn't exist
            height = src.height
            width = src.width
            transform = src.transform
    
    mask = rasterize(
        geometries,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
        all_touched=True
    )
    
    return mask


def smoothing_filter(data, sigma=3, **kwargs):
    """
    Apply a Gaussian filter on the an array to smooth the values. Only filters valid data points, 
    NaN areas remain NaN.

    Args:
        sigma: the standard deviation for Gaussian kernel.
    """
    # Store original NaN mask
    original_nan_mask = np.isnan(data)
    
    # If no valid data, return original
    if np.all(original_nan_mask):
        return data
    
    # Start with original data
    result = data.copy()
    
    # Create a working copy where NaN -> 0
    work_data = np.where(original_nan_mask, 0, data)
    
    # Create weights
    weights = (~original_nan_mask).astype(float)
    
    # Apply Gaussian filter on both data and weights
    filtered_data = gaussian_filter(work_data * weights, sigma, **kwargs)
    filtered_weights = gaussian_filter(weights, sigma, **kwargs)
    
    # Only update non-NaN locations
    valid_filter_mask = filtered_weights > 1e-10
    update_mask = (~original_nan_mask) & valid_filter_mask    
    result[update_mask] = filtered_data[update_mask] / filtered_weights[update_mask]
    
    # Ensure NaN areas stay NaN
    result[original_nan_mask] = np.nan
    
    return result

