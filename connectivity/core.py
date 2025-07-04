import numpy as np
from rust_conn import connectivity
from .rastio import read_raster, write_raster
from .utils import check_grids, smoothing_filter


def connectedness(
        condition_file,
        polygon_mask=None,
        lambdas=[0.2, 2.0, 20.0],
        scale=2.0, 
        nb_size=3, 
        last_nb_size=9,
        levels=[1, 2, 4, 8, 16],
        smooth=True, 
        sigma=3,
        filename=""
    ):
    """
    Habitat Condition Connectedness

    
    sigma: smoothing Gaussian filter sigma; set 0 to avoid smoothing

    """
    # read raster overview as a dictionary
    data_dict = read_raster(file=condition_file, gdf=polygon_mask, levels=levels, expand_px=last_nb_size)

    out_array = connectivity(
        data_dict = data_dict,
        trans_list = [{}], # empty dict in a list to calacualate connectedness
        lambdas = lambdas, 
        scale = scale,
        nb_size = nb_size,
        last_nb_size = last_nb_size
    )

    # smooth the output array with Gaussian filtering
    if smooth:
        out_array = smoothing_filter(out_array, sigma=sigma)

    if len(filename) > 3:
        write_raster(out_array, outfile=filename, template=condition_file)

    return out_array



def beri(
        condition_file,
        current_file,
        polygon_mask=None,
        future_files=[],
        lambdas=[0.2, 2.0, 20.0],
        scale=2.0, 
        nb_size=3, 
        last_nb_size=9,
        levels=[1, 2, 4, 8, 16],
        smooth=False, 
        sigma=3,
        filename=""
    ):
    """
    The Bioclimatic Ecosystem Resilience Index (BERI)

    sigma: smoothing Gaussian filter sigma; set 0 to avoid smoothing
    
    """
    # read raster overview as a dictionary
    data_dict = read_raster(file=condition_file, gdf=polygon_mask, levels=levels, expand_px=last_nb_size)

    # insert current climate as the first element in the list (this is important)
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
        last_nb_size = last_nb_size
    )

    # smooth the output array with Gaussian filtering
    if smooth:
        out_array = smoothing_filter(out_array, sigma=sigma)


    if len(filename) > 3:
        write_raster(out_array, outfile=filename, template=condition_file)

    return out_array

