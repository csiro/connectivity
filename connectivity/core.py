import numpy as np
import geopandas as gpd
from rust_conn import connectivity
from .rastio import read_raster, write_raster
from .utils import check_grids, smoothing_filter, fn, crop_array


# Connectedness main funciton
def connectedness(
        condition_file: str,
        pa_file: str | None = None, 
        polygon_mask: gpd.GeoDataFrame | None = None,
        lambdas: list[float] = [2, 20, 200],
        max_cost: float = 2.0, 
        window_size: int = 3, 
        outer_window: int = 9,
        levels: list[int] | None = None,
        sigma: float | None = 1,
        scale: float | tuple | None = None,
        option: int = 3,
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
        adjusted using the `scale` parameter. The file must be a GeoTIFF (including COGs) with overview
        levels generated using the average aggregation method for multi-scale analysis.
    pa_file : str, optional
        Path to the raster file containing protected-area (PA) proportions. This file is required to 
        calculate PARC-connectedness. If provided, the function will compute PARC-connectedness instead 
        of standard habitat connectedness. If `None`, only habitat connectedness is calculated.
    polygon_mask : str, optional
        Path to a polygon shapefile or mask file used to limit the analysis area.
        If None, the entire image is processed.
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
    levels : list of int, optional
        List of overview levels used for multi-scale analysis. Must be powers of 2 (1 is ignored).
        Default is None (uses all overview levels).
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

    # Read condition raster overviews; this checks levels as well
    cond_dict, affine_dict, is_geo = read_raster(
        file=condition_file, 
        polygon=polygon_mask, 
        levels=levels, 
        scale=s1, # only for condition raster
        expand_px=outer_window # * max_level?
    )

    # Process PA-array for PARC-connectedness
    if pa_file is None:
        pa_mask = None
    else:
        # Read PA raster overviews; this checks levels as well
        pa_dict, _, _ = read_raster(file=pa_file, polygon=polygon_mask, levels=levels, scale=s2, expand_px=outer_window)
        # Ensure both dictionaries have the same keys
        if cond_dict.keys() != pa_dict.keys():
            raise ValueError("Input condition and PA date do not have identical overviews.")
        # Ensure dimension of the girds the same
        if not check_grids(cond_dict[1], pa_dict[1]):
            raise(ValueError("The shape of the condition and PA data doesn't match."))
        
        # Update the condition dict with the max(c, p) for each cell in each level
        for k in cond_dict:
            cond_dict[k] = np.maximum(cond_dict[k], pa_dict[k])

        # Filter for protected areas, prop > 0 or NaN; and any NaN in condition stays NaN;
        pa_mask = np.where((pa_dict[1] > 0) & ~np.isnan(cond_dict[1]), 1.0, np.nan).astype(np.float32)

    # The base Rust connectivity funciton
    conn_array = connectivity(
        condition = cond_dict,
        pa_array = pa_mask, 
        transgrid_list = [{}], # empty dict-list to compute connectedness in Rust, insead of BERI
        transforms = affine_dict,
        lambdas = lambdas, 
        is_geo = is_geo,
        max_cost = max_cost,
        window_size = window_size,
        outer_window = outer_window,
        n_threads = n_threads,
    )

    # Smooth the output array with Gaussian filtering
    if sigma is not None and sigma != 0:
        sigma = max(sigma, 1)
        conn_array = smoothing_filter(conn_array, sigma=sigma)

    # Calculate the connected-habitat or just return the PARC-connectedness
    if pa_file is None:
        out_array = fn(conn_array, cond_dict[1], option=option)
    else:
        out_array = conn_array

    tr = None
    # Crop array back to the polygon mask
    if polygon_mask is not None:
        out_array, tr = crop_array(out_array, affine_dict[1], polygon_mask)

    if len(filename) > 3:
        write_raster(out_array, outfile=filename, template=condition_file, transform=tr)

    return out_array



# BERI main funciton
def beri(
        condition_file: str,
        current_file: str,
        future_files: list[str] = [],
        polygon_mask: gpd.GeoDataFrame | None = None,
        lambdas: list[float] = [2, 20, 200],
        max_cost: float = 2.0, 
        window_size: int = 3, 
        outer_window: int = 9,
        levels: list[int] | None = None,
        sigma: float | None = 1,
        scale: float | None = None,
        n_threads: int | None = None,
        filename: str = ""
    ):
    """Computes the Bioclimatic Ecosystem Resilience Index (BERI)

    BERI measures the capacity of ecosystems to retain biodiversity under 
    projected future climate scenarios. It combines spatial connectedness 
    of habitat condition with projected species turnover, comparing current 
    and future scenarios to assess resilience.

    This algorithm operates on the overview layers of a GeoTIFF file (including 
    Cloud-Optimized GeoTIFFs). Please ensure that these overview layers are generated 
    using the `average`, not `nearest` resampling method. Use the `create_overviews()` 
    function to generate the required overview layers correctly.
    
    The maximum distance/raduis the algorithm searches for cells (in the condition 
    raster) to calculate connectivity in BERI is computed as:
    max-distance = outer_window * max(levels) * resolution

    Parameters
    ----------
    condition_file : str
        Path to the input habitat-condition raster file. Values should range from 0 to 1 and can be 
        adjusted using the `scale` parameter. The file must be a GeoTIFF (including COGs) with overview
        levels generated using the average aggregation method for multi-scale analysis.
    current_file : str
        Path to the current compositional turnover layer (e.g., dissimilarity surface under current climate).
    future_files : list of str, optional
        List of file paths representing future compositional turnover layers (scenarios).
        Each should be aligned with the spatial resolution and extent of `current_file` and `condition_file`.
    polygon_mask : str, optional
        Path to a polygon shapefile or mask used to limit the analysis area.
        If None, the entire input extent is analyzed.
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
    levels : list of int, optional
        List of overview levels used for multi-scale analysis. Should be powers of 2 (1 is ignored).
        Default is None (uses all overview levels).
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

    # get only one scale if confused with connectedness function
    if isinstance(scale, tuple):
        scale = scale[0]

    # Read raster overview as a dictionary; this checks levels as well.
    cond_dict, affine_dict, is_geo  = read_raster(
        file=condition_file,
        polygon=polygon_mask, 
        levels=levels, 
        scale=scale, # only for condition raster
        expand_px=outer_window
    )

    # Insert current climate as the first element in the list (this is important) before reading
    future_files.insert(0, current_file)
    # Just get the cond_dict for the transgrids; Ignore the affine_dict
    # the scale parameter is not used here
    trans_grids = [
        read_raster(file=i, polygon=polygon_mask, levels=levels, expand_px=outer_window)[0]
        for i in future_files
    ]
    
    # Ensure dimension of the girds the same
    if not check_grids(cond_dict[1], trans_grids[0][1]):
        raise(ValueError("The shape of the condition and transgrids doesn't match."))

    # The base Rust connectivity funciton
    out_array = connectivity(
        condition = cond_dict,
        pa_array = None,             # only use for PARC-connectedness; keep None otherwise
        transgrid_list = trans_grids,
        transforms = affine_dict,
        lambdas = lambdas, 
        is_geo = is_geo,
        max_cost = max_cost,
        window_size = window_size,
        outer_window = outer_window,
        n_threads = n_threads,
    )

    # Smooth the output array with Gaussian filtering
    if sigma is not None and sigma != 0:
        sigma = max(sigma, 1)
        out_array = smoothing_filter(out_array, sigma=sigma)

    tr = None
    # Crop array back to the polygon mask
    if polygon_mask is not None:
        out_array, tr = crop_array(out_array, affine_dict[1], polygon_mask)

    if len(filename) > 3:
        write_raster(out_array, outfile=filename, template=condition_file, transform=tr)

    return out_array

