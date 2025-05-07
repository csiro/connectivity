import numpy as np
from multires_connectivity import connectivity

from functools import reduce
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter

def random_image(size=50, sig=3):
    def scale(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    random_field = np.random.randint(0, size**2/10, size=(size, size)).astype(np.float32)
    # random_field = np.random.uniform(0, 1, size=(size, size)).astype(np.float64)
    smoothed = gaussian_filter(random_field, sigma=sig)
    out = scale(smoothed)
    return scale(out + np.random.normal(0, 0.01, (size, size)))

def aggregate_2x2(arr, func=np.mean):
    # Crop to even dimensions if needed
    h, w = arr.shape
    h -= h % 2
    w -= w % 2
    arr = arr[:h, :w]
    # Reshape to 2x2 blocks
    reshaped = arr.reshape(h // 2, 2, w // 2, 2)
    # Apply the aggregation function
    return func(func(reshaped, axis=1), axis=2)


def test_multi_level_window():
    """
    Test the Rust 'multi_level_window' function by creating a dictionary
    of numpy arrays and printing the last one.
    """

    np.random.seed(90)
    arr = random_image(32, 3)

    data_dict = {}
    for level in [1, 2, 4, 8]:
        if level < 2:
            data_dict[level] = arr.astype(np.float32)
        else:
            lower_level = level/2
            data_dict[level] = aggregate_2x2(data_dict[lower_level]).astype(np.float32)
    
    # Call the Rust function
    base_i = 1
    base_j = 1
    current_level = 2
    nb_size = 3
    last_nb_size = 3
    
    # Call the wrapped function
    array = connectivity(
        data_dict, 
        nb_size, 
        last_nb_size
    )
    
    
    return array
    

if __name__ == "__main__":
    print(test_multi_level_window())
