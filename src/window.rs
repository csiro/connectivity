use ndarray::{Array3, Array2, Array1, ArrayView1, s};
use std::collections::HashMap;
use std::f32::consts::E;
use std::iter::zip;


/// Get the indices of cells at current_level that fall within the neighborhood
/// of its higher level containing point (i,j). Uses different neighborhood sizes
/// for highest level vs. other levels.
///
/// # Arguments
/// * `base_i`, `base_j` - Coordinates in the original resolution
/// * `current_level` - Current level (1, 2, 4, 8, etc.)
/// * `win_size` - Size of the neighborhood for normal levels (3 for 3x3, 5 for 5x5, etc.).
///               Must be odd-numbered.
/// * `outer_win` - Size of the neighborhood for the highest level only.
/// * `cond_dict` - HashMap mapping levels to arrays
/// * `trans_vect`   - Transgrids values hashmap
/// * `trans_ij`     - The current climate values of i, j
///
/// # Returns
/// * Tuple of row-indices, column-indices, conditon, and similarity values for the neighborhood as vectors
pub fn build_window(
    base_i: i32,
    base_j: i32,
    current_level: i32,
    win_size: i32,
    outer_win: i32,
    cond_dict: &HashMap<i32, Array2<f32>>,
    trans_vect: &Vec<HashMap<i32, Array3<f32>>>,
    trans_ij: &Array1<f32>,
) -> (Vec<i32>, Vec<i32>, Vec<f32>, Vec<Vec<f32>>) {
    let agg_factor = 2;
    let higher_level = current_level * agg_factor;
    
    let current_array = &cond_dict[&current_level];
    let current_height = current_array.shape()[0] as i32;
    let current_width = current_array.shape()[1] as i32;
    
    let higher_center_i: i32 = base_i / higher_level;
    let higher_center_j: i32 = base_j / higher_level;
    let i: i32 = base_i / current_level;
    let j: i32 = base_j / current_level;
    
    // Ensure neighborhood sizes are odd
    if win_size % 2 == 0 {
        panic!("Window size must be odd-numbered");
    }
    
    // If outer_win is lower than win_size, use the regular win_size
    let outer_win = if outer_win < win_size {
        win_size
    } else if outer_win % 2 == 0 {
        panic!("Outer window size must be odd-numbered");
    } else {
        outer_win
    };
    
    // Find the maximum level
    let max_level = *cond_dict.keys().max().unwrap_or(&current_level);
    let is_highest_level = current_level == max_level;
    // Choose the appropriate neighborhood size
    let effective_win_size = if is_highest_level {
        outer_win
    } else {
        win_size
    };
    
    // Calculate radius based on effective neighborhood size
    let radius: i32 = effective_win_size / 2;
    let exclusion_radius: i32 = win_size / 2; // Always use the standard size for exclusion
    
    // Pre-allocate vectors with estimated capacity
    // We can give a conservative estimate to avoid multiple reallocations
    let max_win_size = win_size.max(outer_win);
    let estimated_capacity = (max_win_size * max_win_size) as usize;
    
    let mut row_indices = Vec::with_capacity(estimated_capacity);
    let mut col_indices = Vec::with_capacity(estimated_capacity);
    let mut values = Vec::with_capacity(estimated_capacity);
    
    // Determine higher level dimensions
    let (higher_height, higher_width) = if cond_dict.contains_key(&higher_level) {
        let higher_array = &cond_dict[&higher_level];
        (higher_array.shape()[0] as i32, higher_array.shape()[1] as i32)
    } else {
        (current_height / agg_factor, current_width / agg_factor)
    };
    
    // Iterate over the NxN neighborhood (this is in potential higher level)
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

                    // Skip the cell/segment if habitat condition is nan!
                    let condition: f32 = current_array[[curr_i as usize, curr_j as usize]];
                    if condition.is_nan() {
                        continue;
                    } else {
                        row_indices.push(curr_i);
                        col_indices.push(curr_j);
                        values.push(condition);
                    }                    
                }
            }
        }
    }

    // Now, separately process the transgrids for all scenarios; easier this way
    // Output: Vec_scenario<Vec_indices<sim>>
    let gdmvals: Vec<Vec<f32>> = trans_vect
        .iter()
        .map(|scenario_map| {
            if let Some(array) = scenario_map.get(&current_level) {
                zip(row_indices.iter(), col_indices.iter())
                    .map(|(&curr_i, &curr_j)| {
                        let seg_val: ArrayView1<f32> = array.slice(s![.., curr_i as usize, curr_j as usize]);
                        similarity(&trans_ij, &seg_val)
                    })
                    .collect::<Vec<f32>>()
            } else {
                // Return a vector of 0.0s the same length as the number of segments
                vec![0.0; row_indices.len()]
            }
        })
        .collect();
    
    (row_indices, col_indices, values, gdmvals)
}


// Calcualte similarity of the transgrid layers
#[inline]
fn similarity(a: &Array1<f32>, b: &ArrayView1<f32>) -> f32 {
    let l1_dist: f32 = (a - b).mapv(f32::abs).sum();
    E.powf(-l1_dist)
}

