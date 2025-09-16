import rasterio
from scipy.ndimage import gaussian_filter
import numpy as np

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
    

# Gaussian smoothing with no edge effect
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

