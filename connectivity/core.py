import numpy as np
from rust_conn import connectivity
from .raster_io import read_overviews, write_raster, check_grids, mask_gird, smoothing_filter


def connectedness(
        condition_file, 
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
    data_dict = read_overviews(file_path=condition_file, levels=levels, nan_nodata=True)

    out_array = connectivity(
        data_dict = data_dict,
        trans_list = [{}], # empty dict in a list to calacualate connectedness
        lambdas = lambdas, 
        scale = scale,
        nb_size = nb_size,
        last_nb_size = last_nb_size
    )

    # set the correct nan mask
    out_array = mask_gird(out_array, condition_file)

    # smooth the output array with Gaussian filtering
    if smooth:
        out_array = smoothing_filter(out_array, sigma=sigma)

    if len(filename) > 3:
        write_raster(out_array, outfile=filename, template=condition_file)

    return out_array



def beri(
        condition_file,
        current_file,
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
    data_dict = read_overviews(file_path=condition_file, levels=levels, nan_nodata=True)

    # insert current climate as the first element in the list (this is important)
    future_files.insert(0, current_file)
    trans_grids = [read_overviews(file_path=i, levels=levels, nan_nodata=True) for i in future_files]

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

    # set the correct nan mask
    out_array = mask_gird(out_array, condition_file)

    # smooth the output array with Gaussian filtering
    if smooth:
        out_array = smoothing_filter(out_array, sigma=sigma)


    if len(filename) > 3:
        write_raster(out_array, outfile=filename, template=condition_file)

    return out_array

