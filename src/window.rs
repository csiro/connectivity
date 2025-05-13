use ndarray::{Array2};
use std::collections::HashMap;

/// Get the indices of cells at current_level that fall within the neighborhood
/// of its higher level containing point (i,j). Uses different neighborhood sizes
/// for highest level vs. other levels.
///
/// # Arguments
/// * `base_i`, `base_j` - Coordinates in the original resolution
/// * `current_level` - Current level (1, 2, 4, 8, etc.)
/// * `data_dict` - HashMap mapping levels to arrays
/// * `nb_size` - Size of the neighborhood for normal levels (3 for 3x3, 5 for 5x5, etc.).
///               Must be odd-numbered.
/// * `last_nb_size` - Size of the neighborhood for the highest level only.
///
/// # Returns
/// * Tuple of row indices, column indices, and values for the neighborhood as vectors
pub fn multi_level_window(
    base_i: i32,
    base_j: i32,
    current_level: i32,
    data_dict: &HashMap<i32, Array2<f32>>,
    nb_size: i32,
    last_nb_size: i32,
    // gdm_dict: &HashMap<i32, Array3<f32>>,
) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let agg_factor = 2;
    let higher_level = current_level * agg_factor;
    
    let current_array = &data_dict[&current_level];
    // let gdm_array = &gdm_dict[&current_level];
    let current_height = current_array.shape()[0] as i32;
    let current_width = current_array.shape()[1] as i32;
    
    let higher_center_i = base_i / higher_level;
    let higher_center_j = base_j / higher_level;
    let i = base_i / current_level;
    let j = base_j / current_level;
    
    // Find the maximum level
    let max_level = *data_dict.keys().max().unwrap_or(&current_level);
    let is_highest_level = current_level == max_level;
    
    // Ensure neighborhood sizes are odd
    if nb_size % 2 == 0 {
        panic!("Neighborhood size must be odd-numbered");
    }
    
    // If last_nb_size is lower than nb_size, use the regular nb_size
    let last_nb_size = if last_nb_size < nb_size {
        nb_size
    } else if last_nb_size % 2 == 0 {
        panic!("Highest neighborhood size must be odd-numbered");
    } else {
        last_nb_size
    };
    
    // Choose the appropriate neighborhood size
    let effective_nb_size = if is_highest_level {
        last_nb_size
    } else {
        nb_size
    };
    
    // Calculate radius based on effective neighborhood size
    let radius = effective_nb_size / 2;
    let exclusion_radius = nb_size / 2; // Always use the standard size for exclusion
    
    // Pre-allocate vectors with estimated capacity
    // We can give a conservative estimate to avoid multiple reallocations
    let max_nb_size = nb_size.max(last_nb_size);
    let estimated_capacity = (max_nb_size * max_nb_size) as usize;
    
    let mut row_indices = Vec::with_capacity(estimated_capacity);
    let mut col_indices = Vec::with_capacity(estimated_capacity);
    let mut values = Vec::with_capacity(estimated_capacity);
    // let mut gdmvalues = Vec::with_capacity(estimated_capacity);
    
    // Determine higher level dimensions
    let (higher_height, higher_width) = if data_dict.contains_key(&higher_level) {
        let higher_array = &data_dict[&higher_level];
        (higher_array.shape()[0] as i32, higher_array.shape()[1] as i32)
    } else {
        (current_height / agg_factor, current_width / agg_factor)
    };
    
    // Iterate over the NxN neighborhood
    for di in -radius..=radius {
        for dj in -radius..=radius {
            let higher_i = higher_center_i + di;
            let higher_j = higher_center_j + dj;
            
            if higher_i < 0 || higher_i >= higher_height || higher_j < 0 || higher_j >= higher_width {
                continue;
            }
            
            let start_i = higher_i * agg_factor;
            let start_j = higher_j * agg_factor;
            let end_i = (higher_i + 1) * agg_factor;
            let end_i = end_i.min(current_height);
            let end_j = (higher_j + 1) * agg_factor;
            let end_j = end_j.min(current_width);
            
            for curr_i in start_i..end_i {
                for curr_j in start_j..end_j {
                    // Skip cells in the exclusion zone (using standard neighborhood size)
                    if current_level > 1 && 
                       (curr_i - i).abs() <= exclusion_radius && 
                       (curr_j - j).abs() <= exclusion_radius {
                        continue;
                    }
                    
                    row_indices.push(curr_i);
                    col_indices.push(curr_j);
                    values.push(current_array[[curr_i as usize, curr_j as usize]]);
                    // let dissim: f32 = dissimilarity(gdm_array[[curr_i as usize, curr_j as usize]]);
                    // gdmvalues.push(dissim);
                }
            }
        }
    }
    
    (row_indices, col_indices, values)
}


// // Calcualte L1 distance of GDM layers
// fn dissimilarity() {
//     for j in 0..x.len() {

//     }
// }

