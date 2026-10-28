import numpy as np
import warnings
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
def _connected_habitat(connectivity, habitat, option=3):
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


def _round_to_pow2(n: int) -> int:
    # closest lower power of 2
    lower = 1 << (n.bit_length() - 1)
    # closest upper power of 2
    upper = lower << 1
    # choose whichever is nearer
    return lower if n - lower <= upper - n else upper


def _check_continuous_levels(levels: list[int]) -> None:
    for current, next_level in zip(levels, levels[1:]):
        expected = current * 2
        if next_level != expected:
            raise ValueError(
                "levels must form a continuous power-of-two sequence; "
                f"missing level {expected} between {current} and {next_level}"
            )


def _crop_array(array, transform, polygon):
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


def _make_traversal(cond_array: np.ndarray, res_array: np.ndarray | None) -> np.ndarray | None:
    """Build the per-cell traversal value fed to the Rust path weight `w`.

    The resistance raster is decoupled from condition: it drives only path traversal, while
    condition still drives the indicator values (habitat area and the value multiplier).

    Returns
    -------
    np.ndarray or None
        ``None`` when no resistance raster is supplied, so Rust falls back to condition for
        path weighting (bit-identical to a resistance-free run). Otherwise a float32 array of
        ``1 - resistance`` (high resistance -> high traversal cost) confined to the condition
        domain (NaN where condition is NaN, since only condition-valid cells are graph nodes).

    Notes
    -----
    Cells that are valid in condition but have no resistance value fall back to condition for
    traversal and raise a ``UserWarning`` reporting the count; their result is unchanged from a
    resistance-free run (no cell is dropped, so the valid domain never changes silently).
    """
    if res_array is None:
        return None

    res_nan = np.isnan(res_array)
    # Inverted resistance in [0, 1]: high resistance -> low traversal value -> high weight.
    tval = np.clip(1.0 - res_array, 0.0, 1.0)

    # Only the condition-valid ∩ resistance-NaN direction changes anything; warn on its count.
    gap = res_nan & ~np.isnan(cond_array)
    n_gap = int(np.count_nonzero(gap))
    if n_gap:
        warnings.warn(
            f"{n_gap} cell(s) have valid condition but missing resistance; using condition "
            "for path traversal there (unchanged from a resistance-free run).",
            UserWarning,
            stacklevel=3,
        )
        tval = np.where(res_nan, cond_array, tval)

    # Confine traversal to the condition domain: non-condition cells are not graph nodes.
    tval = np.where(np.isnan(cond_array), np.nan, tval)
    return tval.astype(np.float32, copy=False)

