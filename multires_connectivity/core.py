import numpy as np
from scipy.ndimage import gaussian_filter
from rust_connectivity import _connectivity as fn_conn
from .raster_io import read_overviews, write_raster

def connectivity(
        path, 
        disp_rate,
        scale=2.0, 
        nb_size=3, 
        last_nb_size=9,
        levels=[1, 2, 4, 8, 16],
        mask=True, 
        smooth=True,
        filename=""
    ):
    # read raster overview as a dictionary
    data_dict = read_overviews(file_path=path, levels=levels)

    # for now just replace the NaNs with 0
    data_dict = {k: np.where(v < -9990, 0, v) for k, v in data_dict.items()}

    out_array = fn_conn(data_dict, lambda_val=disp_rate, scale=scale, nb_size=nb_size, last_nb_size=last_nb_size)

    # fill the NAs with 0 to perform smoothing
    if smooth:
        arr_filled = np.where(np.isnan(out_array), 0, out_array)
        out_array = gaussian_filter(arr_filled, sigma=3)

    # mask the output
    if mask:
        mask = np.where(data_dict[1].copy() < -9990, 0, 1)
        out_array = np.where(mask == 0, np.nan, out_array)

    if len(filename < 2):
        write_raster(out_array, outfile=filename, template=path)

    return out_array
