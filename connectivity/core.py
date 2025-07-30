import numpy as np
from rust_conn import connectivity
from .rastio import read_raster, write_raster
from .utils import check_grids, smoothing_filter, fn


# Connectedness main funciton
def connectedness(
        condition_file,
        polygon_mask=None,
        lambdas=[2, 20, 200],
        scale=2.0, 
        nb_size=3, 
        last_nb_size=9,
        levels=[1, 2, 4, 8, 16],
        sigma=None,
        option=3,
        n_threads=None,
        filename=""
    ):
    """
    Computes a multi-scale connectedness metrics 
    
    This based on condition using a hierarchical neighborhood-based over multiple resolution
    levels (COG overviews), and optionally applies Gaussian smoothing. 

    This algorithm operates on the overview layers of Cloud Optimized 
    GeoTIFF (COG) files. Please ensure that these overview layers are generated
    using `mean` aggregation, not `nearest neighbor` resampling. Use `create_overviews()`
    function for generting correct overview layers.
    
    The maximum distance the algorithm searches for cells (in the condition raster) to
    calculate connectivity is computed as: max-distance = last_nb_size * max(levels) * resolution

    Parameters
    ----------
    condition_file : str
        Path to the input condition COG file (it must be a COG file containing the 
        overview levels used in the function).
    polygon_mask : str, optional
        Path to a polygon shapefile or mask file used to limit the analysis area.
        If None, the entire image is processed.
    lambdas : list of float, optional
        The bandwidth values for the connectivity kernels. Controls the distance over 
        which the condition is used in the connectivity as a measure of organism 
        dispersal. Default is [2, 20, 200].
    scale : float, optional
        Used in weighting the condition for connectivity computation (default = 2). 
        Applied as: `w = (1.0 - scale) * condition + scale`.
    nb_size : int, optional
        Neighborhood size (e.g., 3 for 3x3). Determines the local window for most scales.
        Default is 3.
    last_nb_size : int, optional
        Neighborhood size for the final (largest) scale. Used to capture broader context.
        Default is 9.
    levels : list of int, optional
        List of overview levels used for multi-scale analysis. Must be powers of 2.
        Default is [1, 2, 4, 8, 16].
    sigma : float, optional
        Standard deviation of the Gaussian kernel used for smoothing. 
        Default is None (smoothing disabled).
    option : int, optional
        Option flag to generate the connected condition from connectedness and input condition.
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
    if last_nb_size < nb_size:
        last_nb_size = nb_size

    # Read raster overviews as a dictionary
    data_dict = read_raster(file=condition_file, gdf=polygon_mask, levels=levels, expand_px=last_nb_size)

    conn_array = connectivity(
        data_dict = data_dict,
        trans_list = [{}], # empty dict in a list to calacualate connectedness in Rust
        lambdas = lambdas, 
        scale = scale,
        nb_size = nb_size,
        last_nb_size = last_nb_size,
        n_threads = n_threads,
    )

    # Smooth the output array with Gaussian filtering
    if sigma is not None:
        sigma = max(sigma, 1)
        conn_array = smoothing_filter(conn_array, sigma=sigma)

    # Calculate connected habitat
    out_array = fn(conn_array, data_dict[1], option=option)

    if len(filename) > 3:
        write_raster(out_array, outfile=filename, template=condition_file)

    return out_array


# BERI main funciton
def beri(
        condition_file,
        current_file,
        future_files=[],
        polygon_mask=None,
        lambdas=[2, 20, 200],
        scale=2.0, 
        nb_size=3, 
        last_nb_size=9,
        levels=[1, 2, 4, 8, 16],
        sigma=None,
        n_threads=None,
        filename=""
    ):
    """
    Computes the Bioclimatic Ecosystem Resilience Index (BERI).

    BERI measures the capacity of ecosystems to retain biodiversity under 
    projected future climate scenarios. It combines spatial connectedness 
    of habitat condition with projected species turnover, comparing current 
    and future scenarios to assess resilience.

    This algorithm operates on the overview layers of Cloud Optimized 
    GeoTIFF (COG) files. Please ensure that these overview layers are generated
    using `mean` aggregation, not `nearest neighbor` resampling. Use `create_overviews()`
    function for generting correct overview layers.
    
    The maximum distance the algorithm searches for cells (in the condition 
    raster) to calculate connectivity in BERI is computed as:
    max-distance = last_nb_size * max(levels) * resolution

    Parameters
    ----------
    condition_file : str
        Path to the input condition COG file representing current habitat condition.
        Must be a Cloud-Optimized GeoTIFF (COG) with overview levels used for multi-scale analysis.
    current_file : str
        Path to the current compositional turnover layer (e.g., dissimilarity surface under current climate).
    future_files : list of str, optional
        List of file paths representing future compositional turnover layers (scenarios).
        Each should be aligned with the spatial resolution and extent of `current_file`.
    polygon_mask : str, optional
        Path to a polygon shapefile or mask used to limit the analysis area.
        If None, the entire input extent is analyzed.
    lambdas : list of float, optional
        The bandwidth values for the connectivity kernels. Controls the distance over 
        which the condition is used in the connectivity as a measure of organism 
        dispersal. Default is [2, 20, 200].
    scale : float, optional
        Weighting factor for the condition in the connectivity calculation. 
        Used as: `w = (1.0 - scale) * condition + scale`.
        Default is 2.0.
    nb_size : int, optional
        Neighborhood size (e.g., 3 for 3x3). Determines the local window for most scales.
        Default is 3.
    last_nb_size : int, optional
        Neighborhood size for the final (largest) scale to capture broader connectivity context.
        Default is 9.
    levels : list of int, optional
        List of overview levels used for multi-scale analysis. Should be powers of 2.
        Default is [1, 2, 4, 8, 16].
    sigma : float, optional
        Standard deviation for the Gaussian kernel if smoothing is applied to input layers.
        Default is None (no smoothing).
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
    if last_nb_size < nb_size:
        last_nb_size = nb_size

    # Read raster overview as a dictionary
    data_dict = read_raster(file=condition_file, gdf=polygon_mask, levels=levels, expand_px=last_nb_size)

    # Insert current climate as the first element in the list (this is important) before reading
    future_files.insert(0, current_file)
    trans_grids = [read_raster(file=i, gdf=polygon_mask, levels=levels, expand_px=last_nb_size) for i in future_files]
    
    # fix this.....
    if not check_grids(data_dict[1], trans_grids[0][1]):
        raise(ValueError("The shape of the condition and transgrids doesn't match."))

    out_array = connectivity(
        data_dict = data_dict,
        trans_list = trans_grids,
        lambdas = lambdas, 
        scale = scale,
        nb_size = nb_size,
        last_nb_size = last_nb_size,
        n_threads = n_threads,
    )

    # Smooth the output array with Gaussian filtering
    if sigma is not None:
        out_array = smoothing_filter(out_array, sigma=sigma)

    if len(filename) > 3:
        write_raster(out_array, outfile=filename, template=condition_file)

    return out_array

