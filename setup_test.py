import numpy as np
from multires_connectivity import py_multi_level_window

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
    arr = random_image(128, 10)


    # # Create a dictionary of numpy arrays
    # data_dict = {
    #     1: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    #     2: np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    #     3: np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32)
    # }
    data_dict = {}
    for level in [1, 2, 4, 8, 16]:
        if level < 2:
            data_dict[level] = arr.astype(np.float32)
        else:
            lower_level = level/2
            data_dict[level] = aggregate_2x2(data_dict[lower_level]).astype(np.float32)
    
    # # Print the last numpy array in the dictionary
    # last_key = max(data_dict.keys())
    # print(f"Last numpy array (key={last_key}):")
    # print(data_dict[last_key])
    
    # Call the Rust function
    base_i = 0
    base_j = 0
    current_level = 2
    nb_size = 3
    last_nb_size = 2
    
    # Call the wrapped function
    row_indices, col_indices, values = py_multi_level_window(
        base_i, 
        base_j, 
        current_level, 
        data_dict, 
        nb_size, 
        last_nb_size
    )
    
    # Print the results
    print("\nResults from Rust function:")
    print(f"Row indices: {row_indices}")
    print(f"Column indices: {col_indices}")
    print(f"Values: {values}")
    
    return row_indices, col_indices, values
    

if __name__ == "__main__":
    test_multi_level_window()