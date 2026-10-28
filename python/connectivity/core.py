from concurrent.futures import ThreadPoolExecutor

import numpy as np
import geopandas as gpd
import rasterio
from rust_conn import connectivity
from .rastio import read_raster, write_raster
from .utils import (
    _normalise_window_mode,
    _make_traversal,
    _check_continuous_levels,
    _connected_habitat,
    _crop_array,
    _round_to_pow2,
)

_MAX_BERI_READ_WORKERS = 8


# Connectedness main funciton
def connectedness(
        condition_file: str,
        pa_file: str | None = None,
        pa_to_pa: bool = False,
        polygon_mask: gpd.GeoDataFrame | None = None,
        closed_border: bool = False,
        margin_px: int = 32,
        lambdas: list[float] = [2, 20, 200],
        max_cost: float = 2.0, 
        window_size: int = 3, 
        outer_window: int = 9,
        window_mode: str = "circular",
        levels: list[int] = [2, 4, 8, 16, 32],
        scale: float | tuple | None = None,
        resistance_file: str | None = None,
        resistance_scale: float | None = None,
        option: int = 3,
        n_threads: int | None = None,
        filename: str = "",
    ):
    """Computes a multi-scale habitat and PARC connectedness metrics
    
    This based on habiat condition using a hierarchical neighborhood-based over multiple resolution
    levels (raster overviews).

    This algorithm operates on the overview layers of a GeoTIFF file (including 
    Cloud-Optimized GeoTIFFs). Please ensure that these overview layers are generated 
    using the `average`, not `nearest` resampling method. Use the `create_overviews()` 
    function to generate the required overview layers correctly.
    
    The approximate one-sided search reach in the condition raster is computed as:
    max-reach = outer_window * max(levels) * resolution

    Parameters
    ----------
    condition_file : str
        Path to the input habitat-condition raster file. Values should range from 0 to 1 and can be 
        adjusted using the `scale` parameter.
    pa_file : str, optional
        Path to the raster file containing protected-area (PA) proportions. This file is required to
        calculate PARC-connectedness. If provided, the function will compute PARC-connectedness instead
        of standard habitat connectedness. If `None`, only habitat connectedness is calculated.
    pa_to_pa : bool, optional
        PARC edition only (requires `pa_file`). If True, restrict the source-destination
        connectivity so that only graph nodes falling on a protected area contribute to the
        connectedness *numerator*. Paths still route through the intermediate non-PA landscape
        and their adjusted/intact distances are unchanged; the *denominator* also keeps the full
        reachable area exactly as when False, so only the numerator changes. A destination cell
        outside every PA contributes nothing to the numerator, and a coarse cell straddling a PA
        boundary contributes only its protected fraction. An isolated PA (one that can reach no
        other PA) scores 0. If False (default), every reachable cell is counted, i.e. the
        original PARC behaviour. Ignored when `pa_file` is None.
    polygon_mask : gpd.GeoDataFrame, optional
        A GeoDataFrame containing polygon geometry that defines the area of
        interest. If provided, analysis is limited to this area; if None,
        the entire image is processed. See `closed_border` for boundary behavior.
    closed_border : bool
        Specifies how polygon boundaries are handled when `polygon_mask`
        is provided. If True, only cells inside the polygon are included.
        If False, cells outside the polygon within a buffer are considered for the analysis.
    margin_px : int, optional
        Extra pixel expansion for the valid analysis mask in non-closed mode.
        This expands computed (non-masked) context around `polygon_mask` to reduce
        NaN halo effects before post-filtering. Ignored in closed-border mode.
        In non-closed mode it must satisfy:
        `margin_px <= (outer_window + 3) * max(levels)`.
        Default is 32.
    lambdas : list of float, optional
        The bandwidth values for the connectivity kernels. Controls the distance over 
        which the condition is used in the connectivity as a measure of organism 
        dispersal. Default is [2, 20, 200].
    max_cost : float, optional
        The cost of moving through a fully-degraded site. Used in weighting the path traversal
        (default = 2, i.e. twice the cost of passing through an intact cell). Applied as
        `w = (1.0 - max_cost) * t + max_cost`, where `t` is the traversal value: the condition
        value by default, or `1 - resistance` when `resistance_file` is given (see below).
    window_size : int
        Odd local window width used for non-coarsest levels. For instance,
        window_size=3 produces an effective 6×6 current-level window in the
        multi-resolution framework.
        Default is 3.
    outer_window : int, optional
        Odd coarsest-level window width used to set the long-range search reach.
        Must be greater than or equal to window_size.
        Default is 9.
    window_mode : {"circular", "square"}, optional
        Multi-resolution window construction mode. "circular" uses
        source-centered circular annuli with fractional area/count support at
        annulus boundaries. "square" uses the same source-centered fractional
        construction with square annuli. Default is "circular".
    levels : list of int
        List of overview levels used for multi-scale analysis. Must form a continuous
        power-of-two sequence (1 is added internally).
        Default is [2, 4, 8, 16, 32].
    scale : float or tuple, optional
        Scaling factor(s) applied to the condition and PA rasters.
        - If a single float is provided, it is applied only to the condition raster.
        - If a tuple of two values is provided, the first value scales the condition raster
        and the second value scales the PA raster.
        - Each element may be ``None``, ``0``, or ``1`` to indicate no scaling for that raster.
    resistance_file : str, optional
        Path to an optional resistance raster (values in [0, 1], high = harder to cross). When
        provided it is decoupled from condition and used *only* for path traversal: the weight
        becomes `w = (1.0 - max_cost) * (1 - resistance) + max_cost`. Condition still drives all
        indicator values (habitat area and the value multiplier). If ``None`` (default), the
        condition raster is used for traversal too, i.e. the original behaviour (bit-identical).
        Cells valid in condition but missing in resistance fall back to condition for traversal
        and emit a warning (no cell is dropped).
    resistance_scale : float, optional
        Scaling factor applied to the resistance raster (divides it, like ``scale`` for
        condition) so its values land in [0, 1]. If ``None``, 0, or 1, it is used unchanged.
    option : int, optional
        Option flag to generate the connected condition from connectedness and input habitat-condition 
        (this is ignored for PARC-connectedness).
        Default is 3:
            - 1: connectedness
            - 2: connectedness * condition
            - 3: sqrt(connectedness * condition)
    n_threads : int, optional
        The number of CPU cores for parallel processing in Rust connectivity.
        Default is None (all available cores).
    filename : str, optional
        Path to save the output file. If empty, the result is not written to disk.
        Default is "".

    Returns
    -------
    Connectedness or connected-habiat array (depending on the 'option') and optianlly saves it to disk.

    Notes
    -----
    - Ensure that `polygon_mask`, if provided, is spatially aligned with `condition_file`.
    - Appropriate choice of `lambdas` and `levels` significantly affects the output connectedness.
    - This function is suitable for applications in spatial pattern analysis, texture segmentation,

    """
    window_mode = _normalise_window_mode(window_mode)

    # Before reading raster, fix neighbours window
    if outer_window < window_size:
        print(f"Notice: 'outer_window' was smaller than 'window_size' and has been adjusted to {window_size}.")
        outer_window = window_size

    # PA target-gating only applies to PARC-connectedness (needs a PA file to define targets).
    if pa_to_pa and pa_file is None:
        print("Notice: 'pa_to_pa' is ignored because no 'pa_file' was provided.")
    pa_to_pa = bool(pa_to_pa) and (pa_file is not None)

    # Take of care of two scales in case PA has a different scale
    if isinstance(scale, tuple):
        s1, s2 = (scale + (None,))[:2]   # pad with None and take first 2
    else:
        s1, s2 = scale, None

    if levels is None:
        raise ValueError("levels must be provided")

    # An early check for the overall shapes of grids.
    if pa_file is not None and polygon_mask is None:
        # Only check when no masks; grids could have differnce shape but the masked version could be identical.
        with rasterio.open(condition_file) as ds1, rasterio.open(pa_file) as ds2:
            if ds1.shape != ds2.shape:
                raise ValueError(
                    f"Shape mismatch: {condition_file} {ds1.shape} vs {pa_file} {ds2.shape}"
                )

    # Round to nearest power of 2 (GDAL/rasterio overviews use 2, 4, 8, ...)
    levels = sorted({1, *(_round_to_pow2(x) for x in levels)})
    _check_continuous_levels(levels)

    # For closed border there'll be no padding
    pad_size = 0 if closed_border else outer_window

    # Read condition raster overviews; this checks levels as well
    cond_array, mask_array, affine_dict, is_geo, tile_row0, tile_col0 = read_raster(
        file_path=condition_file,
        polygon=polygon_mask, 
        levels=levels, 
        scale=s1, # only for condition raster
        expand_px=pad_size,
        valid_margin_px=margin_px,
    )

    # Return early if the analysis window has no usable condition cells.
    if np.isnan(cond_array).all() or np.all(mask_array):
        out_array = np.full(mask_array.shape, np.nan, dtype=np.float32)
    else:
        # Process PA-array for PARC-connectedness
        if pa_file is not None:
            # Read PA raster overviews; this checks levels as well
            pa_array, *_ = read_raster(
                file_path=pa_file, 
                polygon=polygon_mask, 
                levels=levels, 
                scale=s2, 
                expand_px=pad_size,
                valid_margin_px=margin_px,
            )
            # Ensure dimension of the arrays the same
            if cond_array.shape != pa_array.shape:
                raise ValueError(
                    "The shape of the condition and trans_grids[0] doesn't match.\n"
                    f"cond_array shape: {cond_array.shape}\n"
                    f"trans_grids[0] shape: {pa_array[0].shape}"
                )
            
            # Update the condition dict with the max(c, p) for each cell in each level
            cond_array = np.maximum(cond_array, pa_array)

            # Filter for protected areas, prop > 0 or NaN; and any NaN in condition stays NaN;
            mask_array = np.where((pa_array > 0) & ~np.isnan(cond_array), False, True).astype(bool)

        # After applying polygon/PA masks, there may be no valid cells left to analyse.
        if np.all(mask_array):
            out_array = np.full(mask_array.shape, np.nan, dtype=np.float32)
        else:
            # Optional resistance raster -> traversal values (decoupled from condition; used
            # only for path weighting). None keeps the resistance-free behaviour unchanged.
            res_array = None
            if resistance_file is not None:
                res_array, *_ = read_raster(
                    file_path=resistance_file,
                    polygon=polygon_mask,
                    levels=levels,
                    scale=resistance_scale,
                    expand_px=pad_size,
                    valid_margin_px=margin_px,
                )
                if cond_array.shape != res_array.shape:
                    raise ValueError(
                        f"Shape mismatch: condition {cond_array.shape} vs "
                        f"resistance {res_array.shape}"
                    )
            traversal = _make_traversal(cond_array, res_array)

            # The base Rust connectivity funciton
            conn_array = connectivity(
                condition = cond_array,
                mask = mask_array,
                transgrid_list = None,
                transforms = affine_dict,
                levels = levels,
                lambdas = lambdas,
                is_geo = is_geo,
                max_cost = max_cost,
                window_size = window_size,
                outer_window = outer_window,
                offsets = (tile_row0, tile_col0),
                traversal = traversal,
                n_threads = n_threads,
                window_mode = window_mode,
                pa_to_pa = pa_to_pa,
            )

            # Calculate the connected-habitat or just return the PARC-connectedness
            if pa_file is None:
                out_array = _connected_habitat(conn_array, cond_array, option=option)
            else:
                out_array = conn_array

    tr = affine_dict[1]
    # Crop array back to the polygon mask
    if not closed_border and polygon_mask is not None:
        out_array, tr = _crop_array(out_array, affine_dict[1], polygon_mask)

    if len(filename) > 3:
        write_raster(out_array, outfile=filename, template=condition_file, transform=tr)

    return out_array



# BERI main funciton
def beri(
        condition_file: str,
        current_file: str,
        future_files: list[str] | None = None,
        polygon_mask: gpd.GeoDataFrame | None = None,
        closed_border: bool = False,
        margin_px: int = 32,
        lambdas: list[float] = [2, 20, 200],
        max_cost: float = 2.0, 
        window_size: int = 3, 
        outer_window: int = 9,
        window_mode: str = "circular",
        levels: list[int] = [2, 4, 8, 16, 32],
        scale: float | None = None,
        resistance_file: str | None = None,
        resistance_scale: float | None = None,
        n_threads: int | None = None,
        filename: str = "",
    ):
    """Computes the Bioclimatic Ecosystem Resilience Index (BERI)

    BERI measures the capacity of ecosystems to retain biodiversity under 
    projected future climate scenarios. It combines spatial connectedness 
    of habitat condition with projected species turnover, comparing current 
    and future scenarios to assess resilience.

    This algorithm operates on the overview layers of a raster file that are generated on-the-fly.
    
    The approximate one-sided search reach in the condition raster is computed as:
    max-reach = outer_window * max(levels) * resolution

    Parameters
    ----------
    condition_file : str
        Path to the input habitat-condition raster file. Values should range from 0 to 1 and can be 
        adjusted using the `scale` parameter. 
    current_file : str
        Path to the current compositional turnover layer (e.g., dissimilarity surface under current climate).
    future_files : list of str, optional
        List of file paths representing future compositional turnover layers (scenarios).
        Each should be aligned with the spatial resolution and extent of `current_file` and `condition_file`.
    polygon_mask : gpd.GeoDataFrame, optional
        A GeoDataFrame containing polygon geometry that defines the area of
        interest. If provided, analysis is limited to this area; if None,
        the entire image is processed. See `closed_border` for boundary behavior.
    closed_border : bool
        Specifies how polygon boundaries are handled when `polygon_mask`
        is provided. If True, only cells inside the polygon are included.
        If False, cells outside the polygon within a buffer are considered for the analysis.
    margin_px : int, optional
        Extra pixel expansion for the valid analysis mask in non-closed mode.
        This expands computed (non-masked) context around `polygon_mask` to reduce
        NaN halo effects before post-filtering. Ignored in closed-border mode.
        In non-closed mode it must satisfy:
        `margin_px <= (outer_window + 3) * max(levels)`.
        Default is 32.
    lambdas : list of float, optional
        The bandwidth values for the connectivity kernels. Controls the distance over 
        which the condition is used in the connectivity as a measure of organism 
        dispersal. Default is [2, 20, 200].
    max_cost : float, optional
        The cost of moving through a fully-degraded site. Used in weighting the path traversal
        (default = 2, i.e. twice the cost of passing through an intact cell). Applied as
        `w = (1.0 - max_cost) * t + max_cost`, where `t` is the traversal value: the condition
        value by default, or `1 - resistance` when `resistance_file` is given (see below).
    window_size : int
        Odd local window width used for non-coarsest levels. For instance,
        window_size=3 produces an effective 6x6 current-level window in the
        multi-resolution framework.
        Default is 3.
    outer_window : int, optional
        Odd coarsest-level window width used to set the long-range search reach.
        Must be greater than or equal to window_size.
        Default is 9.
    window_mode : {"circular", "square"}, optional
        Multi-resolution window construction mode. "circular" uses
        source-centered circular annuli with fractional area/count support at
        annulus boundaries. "square" uses the same source-centered fractional
        construction with square annuli. Default is "circular".
    levels : list of int
        List of overview levels used for multi-scale analysis. Must form a continuous
        power-of-two sequence (1 is added internally).
        Default is [2, 4, 8, 16, 32].
    scale : float, optional
        Scaling factor for condition raster. If None, 0, or 1, condition raster is used unchanged;
        otherwise it is divided by scale.
    resistance_file : str, optional
        Path to an optional resistance raster (values in [0, 1], high = harder to cross). When
        provided it is decoupled from condition and used *only* for path traversal: the weight
        becomes `w = (1.0 - max_cost) * (1 - resistance) + max_cost`. Condition still drives all
        indicator values. If ``None`` (default), condition is used for traversal too, i.e. the
        original behaviour (bit-identical). Cells valid in condition but missing in resistance
        fall back to condition for traversal and emit a warning (no cell is dropped).
    resistance_scale : float, optional
        Scaling factor applied to the resistance raster (divides it, like ``scale``) so its
        values land in [0, 1]. If ``None``, 0, or 1, it is used unchanged.
    n_threads : int, optional
        The number of CPU cores for parallel processing in Rust connectivity.
        Default is None (all available cores).
    filename : str, optional
        Path to save the resulting BERI raster. If empty, the output is not written to disk.
        Default is "".

    Returns
    -------
    None
        The function processes spatial data to compute BERI and optionally writes the result to a file.
        No value is returned unless modified to do so.

    Notes
    -----
    - All raster inputs must be spatially aligned and use the same coordinate reference system.
    - BERI is derived by aggregating scenario results to capture the capacity of the ecosystem to maintain biodiversity.

    """
    window_mode = _normalise_window_mode(window_mode)

    # Before reading raster, fix neighbours window
    if outer_window < window_size:
        print(f"Notice: 'outer_window' was smaller than 'window_size' and has been adjusted to {window_size}.")
        outer_window = window_size

    # Avoid mutable default/list aliasing across repeated calls (e.g., tiled loops).
    if future_files is None:
        future_files = []

    # Get only one scale if confused with connectedness function
    if isinstance(scale, tuple):
        scale = scale[0]

    if levels is None:
        raise ValueError("levels must be provided")
    # Round to nearest power of 2 (GDAL/rasterio overviews use 2, 4, 8, ...)
    levels = sorted({1, *(_round_to_pow2(x) for x in levels)})
    _check_continuous_levels(levels)

    # For closed border there'll be no padding
    pad_size = 0 if closed_border else outer_window

    # An early check for the overall shapes of grids.
    if polygon_mask is not None:
        # Only check when no masks; grids could have differnce shape but the masked version could be identical.
        for src in future_files:
            with rasterio.open(condition_file) as ds1, rasterio.open(src) as ds2:
                if ds1.shape != ds2.shape:
                    raise ValueError(
                        f"Shape mismatch: {condition_file} {ds1.shape} vs {src} {ds2.shape}"
                    )

    # Read raster overview as a dictionary; this checks levels as well.
    cond_array, mask_array, affine_dict, is_geo, tile_row0, tile_col0  = read_raster(
        file_path=condition_file,
        polygon=polygon_mask, 
        levels=levels, 
        scale=scale, # only for condition raster
        expand_px=pad_size,
        valid_margin_px=margin_px,
    )

    # Return early if the analysis window has no usable condition cells.
    if np.isnan(cond_array).all() or np.all(mask_array):
        out_array = np.full(mask_array.shape, np.nan, dtype=np.float32)
    else:
        # Build scenario list without mutating caller input; current always first.
        scenario_files = [current_file, *future_files]
        # Just get the cond_dict for the transgrids; Ignore the affine_dict
        # the scale parameter is not used here
        def _read_transgrid(file_path):
            return read_raster(
                file_path=file_path,
                polygon=polygon_mask,
                levels=levels,
                expand_px=pad_size,
                valid_margin_px=margin_px,
            )[0]

        if len(scenario_files) == 1:
            trans_grids = [_read_transgrid(scenario_files[0])]
        else:
            max_workers = min(len(scenario_files), _MAX_BERI_READ_WORKERS)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                trans_grids = list(executor.map(_read_transgrid, scenario_files))
        
        # Ensure rows/cols of the arrays the same
        if cond_array.shape[:2] != trans_grids[0].shape[:2]:
            raise ValueError(
                "The shape of the condition and trans_grids[0] doesn't match.\n"
                f"cond_array shape: {cond_array.shape}\n"
                f"trans_grids[0] shape: {trans_grids[0].shape}"
            )

        # Optional resistance raster -> traversal values (decoupled from condition; used only
        # for path weighting). None keeps the resistance-free behaviour unchanged.
        res_array = None
        if resistance_file is not None:
            res_array, *_ = read_raster(
                file_path=resistance_file,
                polygon=polygon_mask,
                levels=levels,
                scale=resistance_scale,
                expand_px=pad_size,
                valid_margin_px=margin_px,
            )
            if cond_array.shape[:2] != res_array.shape[:2]:
                raise ValueError(
                    f"Shape mismatch: condition {cond_array.shape[:2]} vs "
                    f"resistance {res_array.shape[:2]}"
                )
        traversal = _make_traversal(cond_array, res_array)

        # The base Rust connectivity funciton
        out_array = connectivity(
            condition = cond_array,
            mask = mask_array,
            transgrid_list = trans_grids,
            transforms = affine_dict,
            levels = levels,
            lambdas = lambdas,
            is_geo = is_geo,
            max_cost = max_cost,
            window_size = window_size,
            outer_window = outer_window,
            offsets = (tile_row0, tile_col0),
            traversal = traversal,
            n_threads = n_threads,
            window_mode = window_mode,
        )

    tr = affine_dict[1]
    # Crop array back to the polygon mask
    if not closed_border and polygon_mask is not None:
        out_array, tr = _crop_array(out_array, affine_dict[1], polygon_mask)

    if len(filename) > 3:
        write_raster(out_array, outfile=filename, template=condition_file, transform=tr)

    return out_array
