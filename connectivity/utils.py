import numpy as np
from rasterio.windows import from_bounds
from rasterio.transform import Affine
from rasterio.features import geometry_mask


def _normalise_window_mode(window_mode: str) -> str:
    if not isinstance(window_mode, str):
        raise TypeError("window_mode must be one of 'square' or 'circular'.")
    mode = window_mode.strip().lower()
    if mode not in {"square", "circular"}:
        raise ValueError("window_mode must be one of 'square' or 'circular'.")
    return mode


# Calculate connected condition
def fn(connectivity, habitat, option=3):
    match option:
        case 1:
            return connectivity
        case 2:
            return habitat * connectivity
        case 3:
            out = np.multiply(habitat, connectivity, dtype=np.float32)
            np.sqrt(out, out=out)
            return out
        case _:
            raise ValueError("option must be one of 1, 2, or 3.")


def round_to_pow2(n: int) -> int:
    # closest lower power of 2
    lower = 1 << (n.bit_length() - 1)
    # closest upper power of 2
    upper = lower << 1
    # choose whichever is nearer
    return lower if n - lower <= upper - n else upper


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


def crop_array(array, transform, polygon):
    """
    Crop a numpy array to the polygon bounding box AND mask to polygon geometry.
    """
    # Normalise transform
    if isinstance(transform, (tuple, list)):
        transform = Affine(*transform[:6])
    elif not isinstance(transform, Affine):
        raise TypeError("transform must be an affine.Affine or a 6-element tuple/list")

    # Get bounds and window
    minx, miny, maxx, maxy = polygon.total_bounds
    window = from_bounds(minx, miny, maxx, maxy, transform)
    window = window.round_offsets().round_lengths()

    # Window -> integer slices
    row_start = int(window.row_off)
    row_stop  = int(window.row_off + window.height)
    col_start = int(window.col_off)
    col_stop  = int(window.col_off + window.width)

    # Clip to array bounds
    nrows = array.shape[-2]
    ncols = array.shape[-1]
    row_start = max(0, row_start)
    col_start = max(0, col_start)
    row_stop  = min(nrows, row_stop)
    col_stop  = min(ncols, col_stop)

    if row_stop <= row_start or col_stop <= col_start:
        raise ValueError("Crop window is empty (polygon may be outside the raster extent).")

    # Crop
    if array.ndim == 3:
        cropped = array[:, row_start:row_stop, col_start:col_stop]
    elif array.ndim == 2:
        cropped = array[row_start:row_stop, col_start:col_stop]
    else:
        raise ValueError("array must be 2D (rows, cols) or 3D (bands, rows, cols).")

    # Update transform for the cropped window
    cropped_transform = transform * Affine.translation(col_start, row_start)

    # Build a mask for the cropped grid:
    # geometry_mask returns True for "masked" pixels by default.
    geoms = polygon.geometry.values
    mask_outside = geometry_mask(
        geoms,
        out_shape=(row_stop - row_start, col_stop - col_start),
        transform=cropped_transform,
        all_touched=True,
        invert=False,  # False means pixels outside polygon are True (masked)
    )

    # Apply mask (broadcast across bands if needed)
    if cropped.ndim == 3:
        masked = np.ma.array(cropped, mask=np.broadcast_to(mask_outside, cropped.shape))
    else:
        masked = np.ma.array(cropped, mask=mask_outside)

    return masked, cropped_transform
