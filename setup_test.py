import numpy as np
from multires_connectivity import py_multi_level_window

def test_multi_level_window():
    """
    Test the Rust 'multi_level_window' function by creating a dictionary
    of numpy arrays and printing the last one.
    """
    # Create a dictionary of numpy arrays
    data_dict = {
        1: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        2: np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        3: np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32)
    }
    
    # Print the last numpy array in the dictionary
    last_key = max(data_dict.keys())
    print(f"Last numpy array (key={last_key}):")
    print(data_dict[last_key])
    
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