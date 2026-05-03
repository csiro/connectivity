import numpy as np
from rasterio.windows import from_bounds
from rasterio.transform import Affine
from rasterio.features import geometry_mask


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


def clip_unit_interval(data):
    """Clip finite values to [0, 1] while preserving NaNs."""
    out = np.array(data, copy=True, dtype=np.float32)
    finite = np.isfinite(out)
    out[finite] = np.clip(out[finite], 0.0, 1.0)
    return out


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


def _nan_box_mean(data, size):
    """Fast NaN-aware box mean with clipped windows at array edges."""
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1

    img = np.asarray(data, dtype=np.float32)
    valid = np.isfinite(img)
    values = np.where(valid, img, 0.0).astype(np.float64, copy=False)
    counts = valid.astype(np.float64, copy=False)

    value_sum = np.pad(values, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    count_sum = np.pad(counts, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)

    rows, cols = img.shape
    half = size // 2
    r0 = np.maximum(np.arange(rows) - half, 0)
    r1 = np.minimum(np.arange(rows) + half + 1, rows)
    c0 = np.maximum(np.arange(cols) - half, 0)
    c1 = np.minimum(np.arange(cols) + half + 1, cols)

    sums = (
        value_sum[r1[:, None], c1[None, :]]
        - value_sum[r0[:, None], c1[None, :]]
        - value_sum[r1[:, None], c0[None, :]]
        + value_sum[r0[:, None], c0[None, :]]
    )
    ns = (
        count_sum[r1[:, None], c1[None, :]]
        - count_sum[r0[:, None], c1[None, :]]
        - count_sum[r1[:, None], c0[None, :]]
        + count_sum[r0[:, None], c0[None, :]]
    )

    out = np.full(img.shape, np.nan, dtype=np.float32)
    ok = ns > 0
    out[ok] = (sums[ok] / ns[ok]).astype(np.float32)
    return out


def _derive_grid_periods(periods, levels, shape):
    if periods is not None:
        candidates = periods
    elif levels is not None and len(levels) > 0:
        # Artifact periods come from the Rust window construction: at each
        # level L the focal window is positioned relative to the 2L-overview
        # block, so the artifact repeats every 2L base pixels.  The dominant
        # (most visible) artifact is 2*max_level; secondary is max_level itself
        # (from the second-coarsest level).  Finer levels produce increasingly
        # weak artifacts whose periods are max_level/2, max_level/4, etc. and
        # they are omitted because they are rarely visible and very short.
        max_level = int(max(levels))
        candidates = [max_level, 2 * max_level]
    else:
        candidates = [32, 64]

    max_dim = max(shape)
    out = []
    for p in candidates:
        p = int(round(p))
        if p > 1 and p <= max_dim and p not in out:
            out.append(p)
    return sorted(out)


def _phase_mean(sum_by_phase, count_by_phase, min_count):
    bias = np.zeros_like(sum_by_phase, dtype=np.float32)
    ok = count_by_phase >= float(min_count)
    if not np.any(ok):
        return bias, ok

    bias[ok] = (sum_by_phase[ok] / count_by_phase[ok]).astype(np.float32)
    shrink = count_by_phase[ok] / (count_by_phase[ok] + float(min_count))
    bias[ok] *= shrink.astype(np.float32)

    center = np.average(bias[ok], weights=count_by_phase[ok])
    bias[ok] -= np.float32(center)
    return bias, ok


def _estimate_phase_bias(
    residual,
    valid,
    row_phase,
    col_phase,
    period,
    min_count,
    iterations,
    allow_rows=True,
    allow_cols=True,
):
    residual = residual.astype(np.float32, copy=False)
    valid = valid.astype(bool, copy=False)
    row_counts = valid.sum(axis=1).astype(np.float64)
    col_counts = valid.sum(axis=0).astype(np.float64)

    row_count_by_phase = np.bincount(row_phase, weights=row_counts, minlength=period).astype(np.float64)
    col_count_by_phase = np.bincount(col_phase, weights=col_counts, minlength=period).astype(np.float64)

    row_bias = np.zeros(period, dtype=np.float32)
    col_bias = np.zeros(period, dtype=np.float32)
    row_ok = np.zeros(period, dtype=bool)
    col_ok = np.zeros(period, dtype=bool)

    for _ in range(max(1, int(iterations))):
        if allow_rows:
            adjusted = residual - col_bias[col_phase][None, :]
            row_sums = np.where(valid, adjusted, 0.0).sum(axis=1).astype(np.float64)
            row_sum_by_phase = np.bincount(row_phase, weights=row_sums, minlength=period).astype(np.float64)
            row_bias, row_ok = _phase_mean(row_sum_by_phase, row_count_by_phase, min_count)

        if allow_cols:
            adjusted = residual - row_bias[row_phase][:, None]
            col_sums = np.where(valid, adjusted, 0.0).sum(axis=0).astype(np.float64)
            col_sum_by_phase = np.bincount(col_phase, weights=col_sums, minlength=period).astype(np.float64)
            col_bias, col_ok = _phase_mean(col_sum_by_phase, col_count_by_phase, min_count)

    row_score = 0.0
    if np.any(row_ok):
        row_score = float(np.sqrt(np.average(row_bias[row_ok] ** 2, weights=row_count_by_phase[row_ok])))

    col_score = 0.0
    if np.any(col_ok):
        col_score = float(np.sqrt(np.average(col_bias[col_ok] ** 2, weights=col_count_by_phase[col_ok])))

    return row_bias, col_bias, max(row_score, col_score), row_score, col_score


def _index_runs(global_index, period):
    """Return contiguous array slices that share the same global grid block."""
    block_index = np.floor_divide(global_index, period).astype(np.int64, copy=False)
    breaks = np.flatnonzero(np.diff(block_index)) + 1
    starts = np.r_[0, breaks]
    stops = np.r_[breaks, block_index.size]
    return starts, stops


def _estimate_block_bias(residual, valid, row_index, col_index, period, min_count, smooth_blocks=5):
    """
    Estimate spatially local block offsets on globally anchored grid blocks.

    Phase correction handles repeated row/column offsets. Some artefacts are
    instead block-wise: a whole period-aligned rectangle is slightly high or
    low relative to its neighbourhood. This estimates that local residual mean
    after background removal, with shrinkage and zero-centering to avoid adding
    or removing a global offset.

    To avoid removing genuine ecological gradients that were not fully captured
    by the pixel-level background, the raw block means are smoothed over
    `smooth_blocks` neighbouring blocks and only the deviation from that smooth
    surface is treated as an artifact. Gradients that vary smoothly across
    blocks cancel in this step; abrupt block-aligned discontinuities do not.
    """
    period = int(period)
    min_count = max(1, int(min_count))
    smooth_blocks = max(1, int(smooth_blocks))
    row_starts, row_stops = _index_runs(row_index, period)
    col_starts, col_stops = _index_runs(col_index, period)

    n_row_blocks = len(row_starts)
    n_col_blocks = len(col_starts)

    block_counts = np.zeros((n_row_blocks, n_col_blocks), dtype=np.float64)
    block_raw = np.full((n_row_blocks, n_col_blocks), np.nan, dtype=np.float32)

    for bi, (r0, r1) in enumerate(zip(row_starts, row_stops)):
        for bj, (c0, c1) in enumerate(zip(col_starts, col_stops)):
            block_valid = valid[r0:r1, c0:c1]
            count = int(block_valid.sum())
            block_counts[bi, bj] = count
            if count >= min_count:
                mean = float(residual[r0:r1, c0:c1][block_valid].mean())
                shrink = count / (count + float(min_count))
                block_raw[bi, bj] = np.float32(mean * shrink)

    valid_blocks = np.isfinite(block_raw)
    correction = np.zeros_like(residual, dtype=np.float32)
    if not np.any(valid_blocks):
        return correction, 0.0

    # Subtract the smooth spatial trend from block means.  Genuine gradients
    # are spatially smooth at the block scale, so smooth_block ≈ block_raw
    # for them and the deviation ≈ 0.  Only abrupt block-aligned jumps
    # (the actual artifact) survive.
    smooth_block = _nan_box_mean(block_raw, smooth_blocks)
    block_artifact = np.full_like(block_raw, np.nan)
    block_artifact[valid_blocks] = block_raw[valid_blocks] - smooth_block[valid_blocks]

    vb_counts = block_counts[valid_blocks]
    vb_values = block_artifact[valid_blocks].astype(np.float64)
    center = float(np.average(vb_values, weights=vb_counts))
    vb_values -= center
    block_artifact[valid_blocks] = vb_values.astype(np.float32)

    score = float(np.sqrt(np.average(vb_values ** 2, weights=vb_counts)))

    for bi, (r0, r1) in enumerate(zip(row_starts, row_stops)):
        for bj, (c0, c1) in enumerate(zip(col_starts, col_stops)):
            if np.isfinite(block_artifact[bi, bj]):
                correction[r0:r1, c0:c1] = block_artifact[bi, bj]

    correction[~valid] = 0.0
    return correction, score


def remove_grid_bias(
    data,
    row0=0,
    col0=0,
    periods=None,
    levels=None,
    background_size=None,
    strength=1.0,
    max_correction=0.05,
    min_count=100,
    min_period_repeats=6,
    min_score=0.002,
    axis="both",
    robust_clip_quantile=0.95,
    iterations=3,
    max_periods=1,
    block_correction=False,
    block_min_count=None,
    block_min_period_repeats=2,
    block_strength=0.5,
    max_block_periods=1,
    block_smooth_window=5,
    clamping=True,
    return_diagnostics=False,
):
    """
    Remove row/column grid bias using global-coordinate phase statistics.

    This deterministic, mask-aware filter estimates a smooth background,
    measures residual row/column bias by
    global pixel phase (`global_row % period`, `global_col % period`), then
    subtracts a capped correction. NaNs are ignored when estimating bias and
    restored in the output.

    Parameters
    ----------
    data : np.ndarray
        Input 2D array.
    row0, col0 : int, optional
        Global raster row/column of `data[0, 0]`. Use tile offsets here to make
        phase classes invariant to tile size and tile layout.
    periods : sequence of int, optional
        Candidate grid periods in pixels. If omitted, candidates are derived
        from `levels`. The two structurally grounded periods are `max_level`
        and `2 * max_level`; the dominant (most visible) artifact is at
        `2 * max_level` because each level's window is positioned relative to
        the `2 × current_level` overview block boundary.
    levels : sequence of int, optional
        Overview levels used in the connectivity run. Used to derive candidate
        periods; only `max(levels)` matters.
    background_size : int, optional
        Window size for the NaN-aware smooth background. If omitted, set to
        `8 * max_level + 1` to cover 4 full cycles of the dominant period
        (`2 * max_level`). Covering 4 cycles ensures genuine ecological
        gradients are captured by the background rather than leaking into the
        residual and being mis-identified as grid bias.
    strength : float, optional
        Multiplier for the estimated correction.
    max_correction : float | None, optional
        Absolute cap applied to the final per-pixel correction.
        Set to `None` to disable capping.
    min_count : int, optional
        Minimum valid-cell count required for a row/column phase estimate.
    min_score : float, optional
        Minimum weighted-RMS bias score required before any correction is
        applied for a given period. When no real grid artifact is present the
        estimated scores stay at or below the statistical noise floor; this
        threshold prevents those noise-level estimates from modifying the data.
        Set to 0.0 to apply any non-zero correction (original behaviour).
        Default is 0.002.
    min_period_repeats : int, optional
        Minimum number of times a period must repeat along an axis before that
        axis can be corrected. This avoids treating broad geographic gradients
        as periodic bias when periods are large relative to the raster.
    axis : {"both", "rows", "cols"}, optional
        Which phase components to correct. Use "rows" for horizontal bands,
        "cols" for vertical bands, or "both" for both components.
    robust_clip_quantile : float, optional
        Quantile used to clip residuals before estimating phase bias.
    iterations : int, optional
        Alternating row/column bias fitting iterations per period.
    max_periods : int | None, optional
        Number of strongest candidate periods to apply. `None` applies all
        candidate periods with non-zero scores.
    block_correction : bool, optional
        Also correct locally varying period-aligned block offsets. This handles
        the non-separable 2-D component of the artifact — rectangular patches
        that cannot be decomposed into independent row and column offsets. In
        most cases the row/column phase correction is sufficient; enable this
        only when residual rectangular block artifacts are still visible after
        phase correction. Default is False.
    block_min_count : int, optional
        Minimum valid-cell count required inside a block before its local
        residual mean can be used. If omitted, a conservative value based on
        `min_count` is used.
    block_min_period_repeats : int, optional
        Minimum number of times a block period must repeat along at least one
        axis before local block correction can use it. This is intentionally
        separate from `min_period_repeats` because large rectangular artefacts
        can be visible even when a coarse period occurs only a few times.
    block_strength : float, optional
        Multiplier for the local block correction before the global `strength`
        and `max_correction` cap are applied.
    max_block_periods : int | None, optional
        Number of strongest block periods to apply. `None` applies all
        candidate periods with non-zero block scores.
    block_smooth_window : int, optional
        Neighbourhood size (in blocks, not pixels) used to smooth block means
        before extracting the local block artifact. The smooth surface captures
        genuine ecological gradients at the block scale; only deviations from
        it are treated as artifacts. Larger values are more conservative (less
        correction). Default is 5.
    clamping : bool, optional
        Clamp finite output values to the unit interval after correction.
    return_diagnostics : bool, optional
        Return `(array, diagnostics)` instead of only the corrected array.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, dict]
        Corrected array, optionally with diagnostics.
    """
    img = np.asarray(data)
    if img.ndim != 2:
        raise ValueError("remove_grid_bias expects a 2D array.")

    img = img.astype(np.float32, copy=True)
    img[~np.isfinite(img)] = np.nan
    valid = np.isfinite(img)
    if not np.any(valid):
        diagnostics = {
            "periods": [],
            "selected_periods": [],
            "scores": {},
            "valid_fraction": 0.0,
        }
        return (img, diagnostics) if return_diagnostics else img

    rows, cols = img.shape
    periods = _derive_grid_periods(periods, levels, img.shape)
    if len(periods) == 0:
        raise ValueError("No valid grid periods were provided or derived.")

    axis = str(axis).lower()
    if axis not in {"both", "rows", "cols"}:
        raise ValueError("axis must be one of 'both', 'rows', or 'cols'.")

    min_period_repeats = max(1, int(min_period_repeats))
    period_axis = {}
    for period in periods:
        allow_rows = axis in {"both", "rows"} and (rows / float(period)) >= min_period_repeats
        allow_cols = axis in {"both", "cols"} and (cols / float(period)) >= min_period_repeats
        period_axis[period] = (allow_rows, allow_cols)

    if background_size is None:
        usable_periods = [p for p in periods if any(period_axis[p])]
        base_period = min(usable_periods) if usable_periods else min(periods)
        # The dominant artifact period is 2 * max_level (= 2 * base_period),
        # because each level's window is positioned relative to the 2×-coarser
        # overview block boundary. We need 4 cycles of that dominant period in
        # the background window to cleanly separate it from the genuine gradient.
        background_size = 8 * base_period + 1
    background_size = int(background_size)
    if background_size % 2 == 0:
        background_size += 1

    background = _nan_box_mean(img, background_size)
    fallback = np.float32(np.nanmean(img))
    background[~np.isfinite(background)] = fallback

    residual = img - background
    residual[~valid] = 0.0

    q = float(np.clip(robust_clip_quantile, 0.0, 1.0))
    if q < 1.0:
        clip_at = float(np.nanquantile(np.abs(residual[valid]), q))
        if clip_at > 0.0:
            residual = np.clip(residual, -clip_at, clip_at).astype(np.float32)

    row_index = int(row0) + np.arange(rows)
    col_index = int(col0) + np.arange(cols)

    scored = []
    phase_cache = {}
    for period in periods:
        row_phase = (row_index % period).astype(np.int64)
        col_phase = (col_index % period).astype(np.int64)
        phase_cache[period] = (row_phase, col_phase)
        allow_rows, allow_cols = period_axis[period]
        row_bias, col_bias, score, row_score, col_score = _estimate_phase_bias(
            residual,
            valid,
            row_phase,
            col_phase,
            period,
            min_count=min_count,
            iterations=iterations,
            allow_rows=allow_rows,
            allow_cols=allow_cols,
        )
        scored.append((period, score, row_score, col_score, allow_rows, allow_cols, row_bias, col_bias))

    min_score = max(0.0, float(min_score))
    scored.sort(key=lambda item: item[1], reverse=True)
    if max_periods is None:
        selected = [item for item in scored if item[1] > min_score]
    else:
        selected = [item for item in scored if item[1] > min_score][: max(0, int(max_periods))]

    correction = np.zeros_like(img, dtype=np.float32)
    residual_current = residual.copy()
    for period, _, _, _, allow_rows, allow_cols, _, _ in selected:
        row_phase, col_phase = phase_cache[period]
        row_bias, col_bias, _, _, _ = _estimate_phase_bias(
            residual_current,
            valid,
            row_phase,
            col_phase,
            period,
            min_count=min_count,
            iterations=iterations,
            allow_rows=allow_rows,
            allow_cols=allow_cols,
        )
        period_correction = row_bias[row_phase][:, None] + col_bias[col_phase][None, :]
        period_correction = np.where(valid, period_correction, 0.0).astype(np.float32)
        correction += period_correction
        residual_current = residual_current - period_correction

    block_scored = []
    selected_blocks = []
    if block_correction:
        if block_min_count is None:
            block_min_count = max(int(min_count) * 5, 1000)
        block_min_count = max(1, int(block_min_count))
        block_min_period_repeats = max(1, int(block_min_period_repeats))

        block_candidates = [
            p for p in periods
            if (
                (rows / float(p)) >= block_min_period_repeats
                or (cols / float(p)) >= block_min_period_repeats
            )
        ]
        for period in block_candidates:
            _, block_score = _estimate_block_bias(
                residual_current,
                valid,
                row_index,
                col_index,
                period,
                min_count=block_min_count,
                smooth_blocks=block_smooth_window,
            )
            block_scored.append((period, block_score))

        block_scored.sort(key=lambda item: item[1], reverse=True)
        if max_block_periods is None:
            selected_blocks = [item for item in block_scored if item[1] > min_score]
        else:
            selected_blocks = [
                item for item in block_scored if item[1] > min_score
            ][: max(0, int(max_block_periods))]

        for period, _ in selected_blocks:
            block_bias, _ = _estimate_block_bias(
                residual_current,
                valid,
                row_index,
                col_index,
                period,
                min_count=block_min_count,
                smooth_blocks=block_smooth_window,
            )
            block_bias = (np.float32(block_strength) * block_bias).astype(np.float32)
            correction += block_bias
            residual_current = residual_current - block_bias

    correction *= np.float32(strength)
    if max_correction is not None:
        cap = max(0.0, float(max_correction))
        correction = np.clip(correction, -cap, cap).astype(np.float32)

    out = img - correction
    out[~valid] = np.nan
    if clamping:
        out = clip_unit_interval(out)

    diagnostics = {
        "periods": periods,
        "selected_periods": [int(item[0]) for item in selected],
        "selected_components": [
            {
                "period": int(period),
                "rows": bool(allow_rows),
                "cols": bool(allow_cols),
            }
            for period, _, _, _, allow_rows, allow_cols, _, _ in selected
        ],
        "scores": {int(period): float(score) for period, score, *_ in scored},
        "row_scores": {int(period): float(row_score) for period, _, row_score, *_ in scored},
        "col_scores": {int(period): float(col_score) for period, _, _, col_score, *_ in scored},
        "selected_block_periods": [int(item[0]) for item in selected_blocks],
        "block_scores": {int(period): float(score) for period, score in block_scored},
        "block_min_count": int(block_min_count) if block_correction else None,
        "block_min_period_repeats": int(block_min_period_repeats) if block_correction else None,
        "block_strength": float(block_strength) if block_correction else 0.0,
        "period_axis_enabled": {
            int(period): {"rows": bool(allow_rows), "cols": bool(allow_cols)}
            for period, _, _, _, allow_rows, allow_cols, _, _ in scored
        },
        "background_size": int(background_size),
        "min_score": float(min_score),
        "valid_fraction": float(np.mean(valid)),
        "max_abs_correction": float(np.nanmax(np.abs(correction[valid]))) if np.any(valid) else 0.0,
        "mean_abs_correction": float(np.nanmean(np.abs(correction[valid]))) if np.any(valid) else 0.0,
    }
    return (out, diagnostics) if return_diagnostics else out


# Get kwargs for filtering
def _resolve_filter_kwargs(
    filter_kwargs: dict | None,
    fn_name: str,
    *,
    row0: int,
    col0: int,
    levels: list[int],
):
    """Build validated kwargs for remove_grid_bias from filter_kwargs."""
    _REMOVE_GRID_BIAS_KEYS = {
        "periods",
        "background_size",
        "strength",
        "max_correction",
        "min_count",
        "min_period_repeats",
        "min_score",
        "axis",
        "robust_clip_quantile",
        "iterations",
        "max_periods",
        "block_correction",
        "block_min_count",
        "block_min_period_repeats",
        "block_strength",
        "max_block_periods",
        "block_smooth_window",
        "clamping",
    }

    if filter_kwargs is None:
        return None

    merged = {}
    if not isinstance(filter_kwargs, dict):
        raise TypeError(f"{fn_name}() expects filter_kwargs to be a dict or None.")
    merged.update(filter_kwargs)

    invalid = sorted(set(merged) - _REMOVE_GRID_BIAS_KEYS)
    if invalid:
        bad = ", ".join(invalid)
        raise TypeError(
            f"{fn_name}() got unexpected keyword argument(s): {bad}. "
            "Use only remove_grid_bias filtering parameters."
        )

    merged["row0"] = int(row0)
    merged["col0"] = int(col0)
    merged["levels"] = list(levels)

    if "clamping" not in merged:
        merged["clamping"] = True

    return merged
