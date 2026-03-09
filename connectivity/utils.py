import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from rasterio.windows import from_bounds
from rasterio.transform import Affine
from rasterio.features import geometry_mask
from rust_conn import inpaint_nans_diffusion


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


# Notch mask for removing grid effect function
def make_notch_mask(rows, cols, notch_width=3, center_radius=25, soft=False, sigma=2.0):
    """Build a cross-notch mask for suppressing row/column grid artifacts."""
    crow, ccol = rows // 2, cols // 2
    rr = (np.arange(rows, dtype=np.float32) - crow)[:, None]
    cc = (np.arange(cols, dtype=np.float32) - ccol)[None, :]

    if not soft:
        mask = np.ones((rows, cols), dtype=np.float32)
        mask[crow - notch_width : crow + notch_width + 1, :] = 0.0
        mask[:, ccol - notch_width : ccol + notch_width + 1] = 0.0
    else:
        g_row = np.exp(-(rr ** 2) / (2.0 * sigma ** 2))
        g_col = np.exp(-(cc ** 2) / (2.0 * sigma ** 2))
        cross = np.maximum(g_row, g_col)
        mask = (1.0 - cross).astype(np.float32)

    r2 = rr ** 2 + cc ** 2
    preserve = (r2 <= float(center_radius) ** 2).astype(np.float32)
    mask = mask * (1.0 - preserve) + preserve

    return mask.astype(np.float32)


def remove_grid_effect(
    data,
    notch_width=3,
    center_radius=25,
    inpaint_size=11,
    inpaint_init="nearest",
    inpaint_max_iter=200,
    inpaint_tol=1e-3,
    soft_notch=False,
    soft_sigma=2.0,
    adaptive_quantile=0.75,
    correction_strength=1.0,
    fft_pad_px=32,
    n_threads=None,
):
    """
    Remove regular grid artifacts from a 2D array using FFT notch filtering.

    The workflow is:
    1. Inpaint NaN cells (nodata) using the Rust diffusion solver.
    2. Apply an FFT notch mask to suppress row/column grid artifacts.
    3. Restore NaN cells to their original locations.

    Parameters
    ----------
    data : np.ndarray
        Input 2D array.
    notch_width : int, optional
        Half-width of the suppressed cross band in frequency space.
    center_radius : int, optional
        Radius of low-frequency core preserved in the FFT mask.
    inpaint_size : int, optional
        Uniform-filter kernel size used by diffusion inpainting.
    inpaint_init : {"nearest", "mean"}, optional
        Initial fill strategy for NaN cells before diffusion.
    inpaint_max_iter : int, optional
        Maximum diffusion iterations.
    inpaint_tol : float, optional
        Convergence threshold for diffusion updates.
    soft_notch : bool, optional
        Use Gaussian-tapered notch instead of hard binary notch.
    soft_sigma : float, optional
        Gaussian width for the soft-notch mode.
    adaptive_quantile : float | None, optional
        If set (e.g. 0.75), apply most of the correction only where the estimated
        grid component is strong (top quantile of correction magnitude).
        Set to `None` to apply full global correction.
    correction_strength : float, optional
        Multiplier in [0, 1] for correction amplitude (1 = full, 0 = none).
    fft_pad_px : int, optional
        Reflect-padding width applied before FFT to reduce boundary wrap-around
        artifacts and extent-dependent periodic differences.
    n_threads : int | None, optional
        Number of Rust worker threads for inpainting. `None` uses all cores.

    Returns
    -------
    np.ndarray
        Filtered 2D array with original NaN mask restored.
    """
    img = np.asarray(data)
    if img.ndim != 2:
        raise ValueError("remove_grid_effect expects a 2D array.")

    img = img.astype(np.float32, copy=True)
    img[~np.isfinite(img)] = np.nan
    nan_mask = np.isnan(img)
    valid_global = ~nan_mask
    if not np.any(valid_global):
        return img

    # Process only the finite-data bounding box to avoid extent-sized NaN halos
    # driving FFT behavior.
    rows_v, cols_v = np.where(valid_global)
    r0 = int(rows_v.min())
    r1 = int(rows_v.max()) + 1
    c0 = int(cols_v.min())
    c1 = int(cols_v.max()) + 1

    work = img[r0:r1, c0:c1].copy()
    work_nan = np.isnan(work)

    if np.any(work_nan):
        work_filled = inpaint_nans_diffusion(
            work,
            size=inpaint_size,
            max_iter=inpaint_max_iter,
            tol=inpaint_tol,
            init=inpaint_init,
            n_threads=n_threads,
        )
    else:
        work_filled = work

    pad = max(0, int(fft_pad_px))
    if pad > 0:
        pad_r = min(pad, max(work_filled.shape[0] - 1, 0))
        pad_c = min(pad, max(work_filled.shape[1] - 1, 0))
    else:
        pad_r = pad_c = 0

    if pad_r > 0 or pad_c > 0:
        fft_input = np.pad(work_filled, ((pad_r, pad_r), (pad_c, pad_c)), mode="reflect")
    else:
        fft_input = work_filled

    rows, cols = fft_input.shape
    f_shifted = fftshift(fft2(fft_input))
    mask = make_notch_mask(
        rows,
        cols,
        notch_width=notch_width,
        center_radius=center_radius,
        soft=soft_notch,
        sigma=soft_sigma,
    )
    f_filtered = f_shifted * mask
    fft_back = np.real(ifft2(ifftshift(f_filtered))).astype(np.float32)

    if pad_r > 0 or pad_c > 0:
        img_back = fft_back[pad_r : pad_r + work_filled.shape[0], pad_c : pad_c + work_filled.shape[1]]
    else:
        img_back = fft_back

    # Conservative correction: remove the estimated grid component while
    # preserving broader signal where correction magnitude is low.
    correction = work_filled - img_back
    valid = ~work_nan

    strength = float(np.clip(correction_strength, 0.0, 1.0))
    if adaptive_quantile is None:
        weight = 1.0
    else:
        q = float(np.clip(adaptive_quantile, 0.0, 0.999999))
        abs_corr = np.abs(correction)
        if np.any(valid):
            thr = float(np.quantile(abs_corr[valid], q))
            maxv = float(np.max(abs_corr[valid]))
            if maxv > thr:
                weight = np.clip((abs_corr - thr) / (maxv - thr), 0.0, 1.0).astype(np.float32)
            else:
                weight = np.zeros_like(abs_corr, dtype=np.float32)
        else:
            weight = np.zeros_like(correction, dtype=np.float32)

    out_work = work_filled - (strength * weight * correction)
    out_work = out_work.astype(np.float32, copy=False)
    out_work[work_nan] = np.nan

    out = img.copy()
    out[r0:r1, c0:c1] = out_work
    out[nan_mask] = np.nan
    return out


# Get kwargs for filtering
def _resolve_filter_kwargs(
    n_threads: int | None,
    filter_kwargs: dict | None,
    fn_name: str,
):
    """Build validated kwargs for remove_grid_effect from filter_kwargs."""
    _REMOVE_GRID_EFFECT_KEYS = {
        "notch_width",
        "center_radius",
        "inpaint_size",
        "inpaint_init",
        "inpaint_max_iter",
        "inpaint_tol",
        "soft_notch",
        "soft_sigma",
        "adaptive_quantile",
        "correction_strength",
        "fft_pad_px",
    }

    if filter_kwargs is None:
        return None

    merged = {}
    if not isinstance(filter_kwargs, dict):
        raise TypeError(f"{fn_name}() expects filter_kwargs to be a dict or None.")
    merged.update(filter_kwargs)

    invalid = sorted(set(merged) - _REMOVE_GRID_EFFECT_KEYS)
    if invalid:
        bad = ", ".join(invalid)
        raise TypeError(
            f"{fn_name}() got unexpected keyword argument(s): {bad}. "
            "Use only remove_grid_effect parameters."
        )

    # Conservative defaults for stable comparisons across years/extents.
    if "notch_width" not in merged:
        merged["notch_width"] = 3
    if "soft_notch" not in merged:
        merged["soft_notch"] = True
    if "soft_sigma" not in merged:
        merged["soft_sigma"] = 2.0
    if "center_radius" not in merged:
        merged["center_radius"] = 35
    if "adaptive_quantile" not in merged:
        merged["adaptive_quantile"] = 0.9
    if "correction_strength" not in merged:
        merged["correction_strength"] = 0.6
    if "fft_pad_px" not in merged:
        merged["fft_pad_px"] = 32
    # Thread control is taken only from the main function argument.
    merged["n_threads"] = n_threads

    return merged
