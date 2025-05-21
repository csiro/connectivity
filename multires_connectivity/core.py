import numpy as np
from scipy.ndimage import gaussian_filter
from rust_connectivity import _connectivity as rust_fn
from .raster_io import read_overviews, write_raster, check_grids

def connectedness(
        condition_file, 
        lambdas=[0.2, 2.0, 20.0],
        scale=2.0, 
        nb_size=3, 
        last_nb_size=9,
        levels=[1, 2, 4, 8, 16],
        mask=True, 
        smooth=True,
        filename=""
    ):
    # read raster overview as a dictionary
    data_dict = read_overviews(file_path=condition_file, levels=levels, fill_nodata=True)

    out_array = rust_fn(
        data_dict = data_dict,
        trans_list = [{}], # empty dict in a list to calacualate connectedness
        lambdas = lambdas, 
        scale = scale,
        nb_size = nb_size,
        last_nb_size = last_nb_size
    )

    # fill the NAs with 0 to perform smoothing
    if smooth:
        arr_filled = np.where(np.isnan(out_array), 0, out_array)
        out_array = gaussian_filter(arr_filled, sigma=3)

    # mask the output
    if mask:
        mask = np.where(data_dict[1].copy() < -9990, 0, 1)
        out_array = np.where(mask == 0, np.nan, out_array)

    if len(filename) > 2:
        write_raster(out_array, outfile=filename, template=path)

    return out_array




def beri(
        condition_file,
        current_file,
        future_files = [],
        lambdas=[0.2, 2.0, 20.0],
        scale=2.0, 
        nb_size=3, 
        last_nb_size=9,
        levels=[1, 2, 4, 8, 16],
        mask=True, 
        smooth=True,
        filename=""
    ):

    # insert current climate as the first element in the list (this is important)
    future_files.insert(0, current_file)

    # read raster overview as a dictionary
    data_dict = read_overviews(file_path=condition_file, levels=levels, fill_nodata=True)
    trans_grids = [read_overviews(file_path=i, levels=levels, fill_nodata=True) for i in future_files]

    # fix this.....
    if check_grids(data_dict[1], trans_grids[0][1]):
        raise(ValueError)

    out_array = rust_fn(
        data_dict = data_dict,
        trans_list = trans_grids,
        lambdas = lambdas, 
        scale = scale,
        nb_size = nb_size,
        last_nb_size = last_nb_size
    )

    # fill the NAs with 0 to perform smoothing
    if smooth:
        arr_filled = np.where(np.isnan(out_array), 0, out_array)
        out_array = gaussian_filter(arr_filled, sigma=3)

    # mask the output
    if mask:
        # mask_gird()
        mask = np.where(data_dict[1].copy() < -9990, 0, 1) # fix this....
        out_array = np.where(mask == 0, np.nan, out_array)

    if len(filename) > 2:
        write_raster(out_array, outfile=filename, template=path)

    return out_array