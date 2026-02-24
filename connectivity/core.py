import numpy as np
import geopandas as gpd
import rasterio
from rust_conn import connectivity
from .rastio import read_raster, write_raster, load_weight_overviews
from .utils import smoothing_filter, fn, crop_array, round_to_pow2


# Connectedness main funciton
def connectedness(
        condition_file: str,
        pa_file: str | None = None, 
        polygon_mask: gpd.GeoDataFrame | None = None,
        closed_border: bool = False,
        lambdas: list[float] = [2, 20, 200],
        max_cost: float = 2.0, 
        window_size: int = 3, 
        outer_window: int = 9,
        levels: list[int] = [2, 4, 8, 16, 32],
        sigma: float | None = 1,
        scale: float | tuple | None = None,
        option: int = 3,
        weight_overviews: dict[int, str] | None = None,
        use_num_cells: bool = True,
        n_threads: int | None = None,
        filename: str = ""
    ):
    """Computes a multi-scale habitat and PARC connectedness metrics 
    
    This based on habiat condition using a hierarchical neighborhood-based over multiple resolution
    levels (raster overviews), and optionally applies Gaussian smoothing. 

    This algorithm operates on the overview layers of a GeoTIFF file (including 
    Cloud-Optimized GeoTIFFs). Please ensure that these overview layers are generated 
    using the `average`, not `nearest` resampling method. Use the `create_overviews()` 
    function to generate the required overview layers correctly.
    
    The maximum distance/raduis the algorithm searches for cells (in the condition raster) to
    calculate connectivity is computed as: 
    max-distance = outer_window * max(levels) * resolution

    Parameters
    ----------
    condition_file : str
        Path to the input habitat-condition raster file. Values should range from 0 to 1 and can be 
        adjusted using the `scale` parameter.
    pa_file : str, optional
        Path to the raster file containing protected-area (PA) proportions. This file is required to 
        calculate PARC-connectedness. If provided, the function will compute PARC-connectedness instead 
        of standard habitat connectedness. If `None`, only habitat connectedness is calculated.
    polygon_mask : gpd.GeoDataFrame, optional
        A GeoDataFrame containing polygon geometry that defines the area of
        interest. If provided, analysis is limited to this area; if None,
        the entire image is processed. See `closed_border` for boundary behavior.
    closed_border : bool
        Specifies how polygon boundaries are handled when `polygon_mask`
        is provided. If True, only cells inside the polygon are included.
        If False, cells outside the polygon within a buffer are considered for the analysis.
    lambdas : list of float, optional
        The bandwidth values for the connectivity kernels. Controls the distance over 
        which the condition is used in the connectivity as a measure of organism 
        dispersal. Default is [2, 20, 200].
    max_cost : float, optional
        The cost of moving through a removed site (cell with condition zero). Used in weighting habitat condition
        for connectivity computation (default = 2, i.e. twice the cost of passing through intact cells). 
        Applied as: `w = (1.0 - max_cost) * condition + max_cost`.
    window_size : int
        Radius of the local neighborhood (in pixels). For instance, a radius of 3 produces an effective 6×6 window in 
        the multi-resolution framework. Must be an odd number.
        Default is 3.
    outer_window : int, optional
        Radius of the neighborhood at the coarsest (largest) resolution level, used to capture broader connectivity context.
        Must be an odd number greater than or equal to window_size.
        Default is 9.
    levels : list of int
        List of overview levels used for multi-scale analysis. Must be powers of 2 (1 is ignored).
        Default is [2, 4, 8, 16, 32].
    sigma : float, optional
        Standard deviation of the Gaussian kernel used for smoothing. 
        Default is 1. Zero or None for disabling smoothing.
    scale : float or tuple, optional
        Scaling factor(s) applied to the condition and PA rasters.
        - If a single float is provided, it is applied only to the condition raster.
        - If a tuple of two values is provided, the first value scales the condition raster
        and the second value scales the PA raster.
        - Each element may be ``None``, ``0``, or ``1`` to indicate no scaling for that raster.
    option : int, optional
        Option flag to generate the connected condition from connectedness and input habitat-condition 
        (this is ignored for PARC-connectedness).
        Default is 3:
            - 1: connectedness
            - 2: connectedness * condition
            - 3: sqrt(connectedness * condition)
    n_threads : int, optional
        The number of CPU cores for parallel processing. 
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
    # Before reading raster, fix neighbours window
    if outer_window < window_size:
        print(f"Notice: 'outer_window' was smaller than 'window_size' and has been adjusted to {window_size}.")
        outer_window = window_size

    # Take of care of two scales in case PA has a different scale
    if isinstance(scale, tuple):
        s1, s2 = (scale + (None,))[:2]   # pad with None and take first 2
    else:
        s1, s2 = scale, None

    # An early check for the overall shapes of grids.
    if pa_file is not None:
        if levels is None:
            levels = common_levels(condition_file, pa_file)

        if polygon_mask is None:
            # Only check when no masks; grids could have differnce shape but the masked version could be identical.
            with rasterio.open(condition_file) as ds1, rasterio.open(pa_file) as ds2:
                if ds1.shape != ds2.shape:
                    raise ValueError(
                        f"Shape mismatch: {condition_file} {ds1.shape} vs {pa_file} {ds2.shape}"
                    )

    if levels is None:
        raise ValueError("levels must be provided")
    # Round to nearest power of 2 (GDAL/rasterio overviews use 2, 4, 8, ...)
    levels = sorted({1, *(round_to_pow2(x) for x in levels)})

    # For closed border there'll be no padding
    pad_size = 0 if closed_border else outer_window

    # Read condition raster overviews; this checks levels as well
    cond_array, mask_array, affine_dict, is_geo, tile_row0, tile_col0 = read_raster(
        file_path=condition_file,
        polygon=polygon_mask, 
        levels=levels, 
        scale=s1, # only for condition raster
        expand_px=pad_size
    )

    # Extra check and return early if condition is all NA; e.g. in a tile
    if np.isnan(cond_array).all():
        out_array = cond_array
    else:
        # Load externally-generated global overview weights into {level: array}.
        # These are passed to Rust instead of internally computing cell_weights.
        weight_map = None
        if weight_overviews is not None:
            weight_map = load_weight_overviews(
                overview_files=weight_overviews,
                levels=levels,
                tile_row0=tile_row0,
                tile_col0=tile_col0,
                tile_shape=cond_array.shape[:2],
            )

        # Process PA-array for PARC-connectedness
        if pa_file is not None:
            # Read PA raster overviews; this checks levels as well
            pa_array, *_ = read_raster(
                file_path=pa_file, 
                polygon=polygon_mask, 
                levels=levels, 
                scale=s2, 
                expand_px=pad_size
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
            mask_array = np.where((pa_array > 0) & ~np.isnan(cond_array), False, True).astype(np.bool)

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
            cell_weights = weight_map,
            use_num_cells = use_num_cells,
            n_threads = n_threads,
        )

        # Smooth the output array with Gaussian filtering
        if sigma is not None and sigma != 0:
            conn_array = smoothing_filter(conn_array, sigma=max(sigma, 1))

        # Calculate the connected-habitat or just return the PARC-connectedness
        if pa_file is None:
            out_array = fn(conn_array, cond_array, option=option)
        else:
            out_array = conn_array

    tr = affine_dict[1]
    # Crop array back to the polygon mask
    if not closed_border and polygon_mask is not None:
        out_array, tr = crop_array(out_array, affine_dict[1], polygon_mask)

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
        lambdas: list[float] = [2, 20, 200],
        max_cost: float = 2.0, 
        window_size: int = 3, 
        outer_window: int = 9,
        levels: list[int] = [2, 4, 8, 16, 32],
        sigma: float | None = 1,
        scale: float | None = None,
        weight_overviews: dict[int, str] | None = None,
        use_num_cells: bool = True,
        n_threads: int | None = None,
        filename: str = ""
    ):
    """Computes the Bioclimatic Ecosystem Resilience Index (BERI)

    BERI measures the capacity of ecosystems to retain biodiversity under 
    projected future climate scenarios. It combines spatial connectedness 
    of habitat condition with projected species turnover, comparing current 
    and future scenarios to assess resilience.

    This algorithm operates on the overview layers of a raster file that are generated on-the-fly.
    
    The maximum distance/raduis the algorithm searches for cells (in the condition 
    raster) to calculate connectivity in BERI is computed as:
    max-distance = outer_window * max(levels) * resolution

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
    lambdas : list of float, optional
        The bandwidth values for the connectivity kernels. Controls the distance over 
        which the condition is used in the connectivity as a measure of organism 
        dispersal. Default is [2, 20, 200].
    max_cost : float, optional
        The cost of moving through a removed site (cell with condition zero). Used in weighting habitat condition
        for connectivity computation (default = 2, i.e. twice the cost of passing through intact cells). 
        Applied as: `w = (1.0 - max_cost) * condition + max_cost`.
    window_size : int
        Radius of the local neighborhood (in pixels). For instance, a radius of 3 produces an effective 6×6 window in 
        the multi-resolution framework. Must be an odd number.
        Default is 3.
    outer_window : int, optional
        Radius of the neighborhood at the coarsest (largest) resolution level, used to capture broader connectivity context.
        Must be an odd number greater than or equal to window_size.
        Default is 9.
    levels : list of int
        List of overview levels used for multi-scale analysis. Should be powers of 2 (1 is ignored).
        Default is [2, 4, 8, 16, 32].
    sigma : float, optional
        Standard deviation for the Gaussian kernel if smoothing is applied to input layers.
        Default is 1. Zero or None for disabling smoothing.
    scale : float, optional
        Scaling factor for condition raster. If None, 0, or 1, condition raster is used unchanged; 
        otherwise it is divided by scale.
    n_threads : int, optional
        The number of CPU cores for parallel processing. 
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

    # # Check for common levels
    # if levels is None:
    #     levels = common_levels(condition_file, current_file)

    if levels is None:
        raise ValueError("levels must be provided")
    # Round to nearest power of 2 (GDAL/rasterio overviews use 2, 4, 8, ...)
    levels = sorted({1, *(round_to_pow2(x) for x in levels)})

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
        expand_px=pad_size
    )

    # Extra check and return early if condition is all NA; e.g. in a tile
    if np.isnan(cond_array).all():
        out_array = cond_array
    else:
        # Load externally-generated global overview weights into {level: array}.
        # These are passed to Rust instead of internally computing cell_weights.
        weight_map = None
        if weight_overviews is not None:
            weight_map = load_weight_overviews(
                overview_files=weight_overviews,
                levels=levels,
                tile_row0=tile_row0,
                tile_col0=tile_col0,
                tile_shape=cond_array.shape[:2],
            )

        # Build scenario list without mutating caller input.
        scenario_files = [current_file, *future_files]
        # Just get the cond_dict for the transgrids; Ignore the affine_dict
        # the scale parameter is not used here
        trans_grids = [
            read_raster(file_path=i, polygon=polygon_mask, levels=levels, expand_px=pad_size)[0]
            for i in scenario_files
        ]
        
        # Ensure rows/cols of the arrays the same
        if cond_array.shape[:2] != trans_grids[0].shape[:2]:
            raise ValueError(
                "The shape of the condition and trans_grids[0] doesn't match.\n"
                f"cond_array shape: {cond_array.shape}\n"
                f"trans_grids[0] shape: {trans_grids[0].shape}"
            )

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
            cell_weights = weight_map,
            use_num_cells = use_num_cells,
            n_threads = n_threads,
        )

        # Smooth the output array with Gaussian filtering
        if sigma is not None and sigma != 0:
            out_array = smoothing_filter(out_array, sigma=max(sigma, 1))

    tr = affine_dict[1]
    # Crop array back to the polygon mask
    if not closed_border and polygon_mask is not None:
        out_array, tr = crop_array(out_array, affine_dict[1], polygon_mask)

    if len(filename) > 3:
        write_raster(out_array, outfile=filename, template=condition_file, transform=tr)

    return out_array
