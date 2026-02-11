use ndarray::{Array3, Array2, Array1, ArrayView1, s};
use std::collections::HashMap;
use std::iter::zip;


// Window kernel for one resolution
#[derive(Debug, Clone)]
pub struct FocalWindow {
    pub i_array: Vec<i32>, // i/col array of neighbour cells
    pub j_array: Vec<i32>, // j/row array ...
    pub values: Vec<f32>,  // condition values of the cell
    pub counts: Vec<f32>,  // cell count contribution to each level (habitat-area)
    pub sims: Vec<Vec<Option<f32>>>, // similiarity with the ij cell in the current climate; scen<cell<sim>>
}

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
impl FocalWindow {
    #[inline]
    pub fn from_data(
        base_i: i32,
        base_j: i32,
        current_level: i32,
        win_size: i32,
        outer_win: i32,
        cond_dict: &HashMap<i32, Array2<f32>>,
        trans_vect: &[HashMap<i32, Array3<f32>>],
        trans_ij: &Array1<f32>,
        cell_weights: &HashMap<i32, Array2<f32>>,
    ) -> Self {
        let agg_factor = 2;
        let higher_level = current_level * agg_factor;
        
        let count_array = cell_weights.get(&current_level).expect("Weights level not found!");
        let current_array = cond_dict.get(&current_level).expect("Condition level not found!");
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
        
        // Choose the appropriate neighborhood size based on level being procced
        let effective_win_size: i32 = {
            let max_level = cond_dict.keys().copied().max().unwrap_or(current_level);
            if current_level == max_level { outer_win } else { win_size }
        };
        
        
        // Calculate radius based on effective neighborhood size
        let radius: i32 = effective_win_size / 2;
        let exclusion_radius: i32 = win_size / 2; // Always use the standard size for exclusion
        
        // Pre-allocate vectors with estimated capacity
        // We can give a conservative estimate to avoid multiple reallocations
        let max_win_size = win_size.max(outer_win);
        let estimated_capacity = (max_win_size * max_win_size) as usize;
        
        let mut i_array = Vec::with_capacity(estimated_capacity);
        let mut j_array = Vec::with_capacity(estimated_capacity);
        let mut values = Vec::with_capacity(estimated_capacity);
        let mut counts = Vec::with_capacity(estimated_capacity);
        
        // Determine higher level dimensions
        let (higher_height, higher_width) = if cond_dict.contains_key(&higher_level) {
            let higher_array = cond_dict.get(&higher_level).expect("Condition level not found!");
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

                        // Cell weights; num valid cell for habitat-area contribution
                        let weight: f32 = count_array[[curr_i as usize, curr_j as usize]];
                        // Skip the cell/segment if habitat condition is nan!
                        let condition: f32 = current_array[[curr_i as usize, curr_j as usize]];
                        if condition.is_nan() {
                            continue;
                        } else {
                            i_array.push(curr_i);
                            j_array.push(curr_j);
                            values.push(condition);
                            counts.push(weight);
                        }                    
                    }
                }
            }
        }

        // Now, separately process the transgrids for all scenarios; easier this way
        // Output: Vec_scenario<Vec_indices<sim>>
        let sims: Vec<Vec<Option<f32>>> = trans_vect
            .iter()
            .map(|scenario_map| {
                if let Some(array) = scenario_map.get(&current_level) {
                    zip(i_array.iter(), j_array.iter())
                        .map(|(&ui, &uj)| {
                            let seg_val: ArrayView1<f32> = array.slice(s![ui as usize, uj as usize, ..]);
                            similarity(&trans_ij, &seg_val)
                        })
                        .collect::<Vec<Option<f32>>>()
                } else {
                    // Return a vector of 0.0s the same length as the number of segments
                    vec![Some(0.0); i_array.len()]
                }
            })
            .collect();
        
        Self { i_array, j_array, values, counts, sims }
    }
}


// Calcualte similarity of the transgrid layers and dealing with NaNs
// Any cell with nan coniditon is ignored in the step before this; now any nan transgrid 
// will be processed as None and dealt with later in the metrics.rs
#[inline]
fn similarity(a: &Array1<f32>, b: &ArrayView1<f32>) -> Option<f32> {
    let mut l1: f32 = 0.0;

    // loop over each transform gird; return None even for one nan transgrid
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x.is_finite() && y.is_finite() {
            l1 += (x - y).abs();
        } else {
            return None;
        }
    }

    Some((-l1).exp())
}

