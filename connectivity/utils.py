from scipy.ndimage import gaussian_filter
import numpy as np
import geopandas as gpd
from rasterio.windows import from_bounds
from rasterio.transform import Affine


# Calculate connected condition
def fn(connectivity, habitat, option=3):
    match option:
        case 1:
            return connectivity
        case 2:
            return habitat * connectivity
        case 3:
            return np.sqrt(habitat * connectivity)
        case _:
            raise ValueError("option must be one of 1, 2, or 3.")


def round_to_pow2(n: int) -> int:
    # closest lower power of 2
    lower = 1 << (n.bit_length() - 1)
    # closest upper power of 2
    upper = lower << 1
    # choose whichever is nearer
    return lower if n - lower <= upper - n else upper


# Check grids are equal
def check_grids(x, y):
    x_shape = x if isinstance(x, tuple) else x.shape
    y_shape = y if isinstance(y, tuple) else y.shape
    return x_shape[-2:] == y_shape[-2:]


def guess_geographic(src):
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


def crop_array(array, transform, polygon_gdf):
    """
    Crop a numpy array to the bounding box of a geopandas polygon
    """
    # Convert transform to Affine object if it's a tuple/list
    if isinstance(transform, (tuple, list)):
        transform = Affine(*transform[:6])

    # Get the total bounds of the polygon(s)
    minx, miny, maxx, maxy = polygon_gdf.total_bounds
    
    # Create a window from the bounds
    window = from_bounds(minx, miny, maxx, maxy, transform)
    
    # Round window to integer pixel coordinates
    window = window.round_offsets().round_lengths()
    
    # Convert window to slices
    row_start = int(window.row_off)
    row_stop = int(window.row_off + window.height)
    col_start = int(window.col_off)
    col_stop = int(window.col_off + window.width)
    
    # Clip to array bounds
    row_start = max(0, row_start)
    col_start = max(0, col_start)
    row_stop = min(array.shape[-2], row_stop)
    col_stop = min(array.shape[-1], col_stop)
    
    # Crop the array
    if array.ndim == 3:
        cropped_array = array[:, row_start:row_stop, col_start:col_stop]
    else:
        cropped_array = array[row_start:row_stop, col_start:col_stop]
    
    # Update the transform for the cropped region
    cropped_transform = transform * Affine.translation(col_start, row_start)
    
    return cropped_array, cropped_transform


# Gaussian smoothing with no edge effect
def smoothing_filter(data, sigma=3, **kwargs):
    """Apply a Gaussian filter on the an array to smooth the values. 
    Only filters valid data points, NaN areas remain NaN.

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

