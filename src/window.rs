use ndarray::{s, Array1, Array2, Array3, ArrayView1};
use std::collections::HashMap;
use std::iter::zip;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowMode {
    Block,
    Fractional,
}

impl WindowMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "block" | "legacy" => Ok(Self::Block),
            "fractional"
            | "annulus"
            | "source-centered"
            | "source_centered"
            | "source-centred"
            | "source_centred"
            | "source-centered-annulus" => Ok(Self::Fractional),
            other => Err(format!(
                "Invalid window_mode '{other}'. Expected 'block' or 'fractional'."
            )),
        }
    }
}

// Window kernel for one resolution
#[derive(Debug, Clone)]
pub struct FocalWindow {
    pub i_array: Vec<i32>,           // i/col array of neighbour cells
    pub j_array: Vec<i32>,           // j/row array ...
    pub values: Vec<f32>,            // condition values of the cell
    pub counts: Vec<f32>,            // cell count contribution to each level (habitat-area)
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
        offsets: (usize, usize),
        cond_dict: &HashMap<i32, Array2<f32>>,
        trans_vect: &[HashMap<i32, Array3<f32>>],
        trans_ij: &Array1<f32>,
        cell_weights: &HashMap<i32, Array2<f32>>,
        window_mode: WindowMode,
    ) -> Self {
        let agg_factor = 2;
        let higher_level = current_level * agg_factor;

        let (tile_row0_u, tile_col0_u) = offsets;
        let tile_row0 = tile_row0_u as i32;
        let tile_col0 = tile_col0_u as i32;

        let level_index = |base_idx: i32, tile0: i32, level: i32| -> i32 {
            (tile0 + base_idx).div_euclid(level) - tile0.div_euclid(level)
        };

        let count_array = cell_weights
            .get(&current_level)
            .expect("Weights level not found!");
        let current_array = cond_dict
            .get(&current_level)
            .expect("Condition level not found!");
        let current_height = current_array.shape()[0] as i32;
        let current_width = current_array.shape()[1] as i32;
        let higher_center_i: i32 = level_index(base_i, tile_row0, higher_level);
        let higher_center_j: i32 = level_index(base_j, tile_col0, higher_level);
        let i: i32 = level_index(base_i, tile_row0, current_level);
        let j: i32 = level_index(base_j, tile_col0, current_level);

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
            if current_level == max_level {
                outer_win
            } else {
                win_size
            }
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

        match window_mode {
            WindowMode::Block => collect_block_window(
                higher_center_i,
                higher_center_j,
                i,
                j,
                current_level,
                higher_level,
                agg_factor,
                radius,
                exclusion_radius,
                current_height,
                current_width,
                cond_dict,
                count_array,
                current_array,
                &mut i_array,
                &mut j_array,
                &mut values,
                &mut counts,
            ),
            WindowMode::Fractional => {
                let base_array = cond_dict.get(&1).expect("Base condition level not found!");
                let base_height = base_array.shape()[0] as i32;
                let base_width = base_array.shape()[1] as i32;

                collect_fractional_annulus_window(
                    base_i,
                    base_j,
                    current_level,
                    win_size,
                    effective_win_size,
                    offsets,
                    current_height,
                    current_width,
                    base_height,
                    base_width,
                    count_array,
                    current_array,
                    &mut i_array,
                    &mut j_array,
                    &mut values,
                    &mut counts,
                )
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
                            let seg_val: ArrayView1<f32> =
                                array.slice(s![ui as usize, uj as usize, ..]);
                            similarity(&trans_ij, &seg_val)
                        })
                        .collect::<Vec<Option<f32>>>()
                } else {
                    // Return a vector of 0.0s the same length as the number of segments
                    vec![Some(0.0); i_array.len()]
                }
            })
            .collect();

        Self {
            i_array,
            j_array,
            values,
            counts,
            sims,
        }
    }
}

#[inline]
fn collect_block_window(
    higher_center_i: i32,
    higher_center_j: i32,
    i: i32,
    j: i32,
    current_level: i32,
    higher_level: i32,
    agg_factor: i32,
    radius: i32,
    exclusion_radius: i32,
    current_height: i32,
    current_width: i32,
    cond_dict: &HashMap<i32, Array2<f32>>,
    count_array: &Array2<f32>,
    current_array: &Array2<f32>,
    i_array: &mut Vec<i32>,
    j_array: &mut Vec<i32>,
    values: &mut Vec<f32>,
    counts: &mut Vec<f32>,
) {
    // Determine higher level dimensions
    let (higher_height, higher_width) = if cond_dict.contains_key(&higher_level) {
        let higher_array = cond_dict
            .get(&higher_level)
            .expect("Condition level not found!");
        (
            higher_array.shape()[0] as i32,
            higher_array.shape()[1] as i32,
        )
    } else {
        // Virtual 2x aggregation grid for the highest level: use ceil division so
        // trailing partial blocks are still represented at tile edges.
        (
            (current_height + agg_factor - 1) / agg_factor,
            (current_width + agg_factor - 1) / agg_factor,
        )
    };

    // Iterate over the NxN neighborhood (this is in potential higher level)
    for di in -radius..=radius {
        for dj in -radius..=radius {
            let higher_i = higher_center_i + di;
            let higher_j = higher_center_j + dj;

            if higher_i < 0 || higher_i >= higher_height || higher_j < 0 || higher_j >= higher_width
            {
                continue;
            }

            let start_i = higher_i * agg_factor;
            let start_j = higher_j * agg_factor;
            let end_i = ((higher_i + 1) * agg_factor).min(current_height);
            let end_j = ((higher_j + 1) * agg_factor).min(current_width);

            for curr_i in start_i..end_i {
                for curr_j in start_j..end_j {
                    // Skip cells in the exclusion zone (using standard neighborhood size)
                    if current_level > 1
                        && (curr_i - i).abs() <= exclusion_radius
                        && (curr_j - j).abs() <= exclusion_radius
                    {
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
}

#[inline]
fn collect_fractional_annulus_window(
    base_i: i32,
    base_j: i32,
    current_level: i32,
    win_size: i32,
    effective_win_size: i32,
    offsets: (usize, usize),
    current_height: i32,
    current_width: i32,
    base_height: i32,
    base_width: i32,
    count_array: &Array2<f32>,
    current_array: &Array2<f32>,
    i_array: &mut Vec<i32>,
    j_array: &mut Vec<i32>,
    values: &mut Vec<f32>,
    counts: &mut Vec<f32>,
) {
    let (tile_row0_u, tile_col0_u) = offsets;
    let tile_row0 = tile_row0_u as i32;
    let tile_col0 = tile_col0_u as i32;

    let level_origin_i = tile_row0.div_euclid(current_level);
    let level_origin_j = tile_col0.div_euclid(current_level);

    let tile_r0 = tile_row0 as f64;
    let tile_c0 = tile_col0 as f64;
    let tile_r1 = (tile_row0 + base_height) as f64;
    let tile_c1 = (tile_col0 + base_width) as f64;

    let source_r = tile_row0 as f64 + base_i as f64 + 0.5;
    let source_c = tile_col0 as f64 + base_j as f64 + 0.5;

    let outer = current_level as f64 * effective_win_size as f64;
    let inner = if current_level == 1 {
        0.0
    } else {
        current_level as f64 * win_size as f64 * 0.5
    };

    let search_r0 = (source_r - outer).max(tile_r0);
    let search_r1 = (source_r + outer).min(tile_r1);
    let search_c0 = (source_c - outer).max(tile_c0);
    let search_c1 = (source_c + outer).min(tile_c1);
    if search_r1 <= search_r0 || search_c1 <= search_c0 {
        return;
    }

    let global_i_start = (search_r0.floor() as i32).div_euclid(current_level);
    let global_i_end = (search_r1.ceil() as i32 - 1).div_euclid(current_level);
    let global_j_start = (search_c0.floor() as i32).div_euclid(current_level);
    let global_j_end = (search_c1.ceil() as i32 - 1).div_euclid(current_level);

    let local_i_start = (global_i_start - level_origin_i).max(0);
    let local_i_end = (global_i_end - level_origin_i).min(current_height - 1);
    let local_j_start = (global_j_start - level_origin_j).max(0);
    let local_j_end = (global_j_end - level_origin_j).min(current_width - 1);
    if local_i_end < local_i_start || local_j_end < local_j_start {
        return;
    }

    for curr_i in local_i_start..=local_i_end {
        let global_i = level_origin_i + curr_i;
        let cell_r0 = (global_i * current_level) as f64;
        let cell_r1 = cell_r0 + current_level as f64;
        let footprint_r0 = cell_r0.max(tile_r0);
        let footprint_r1 = cell_r1.min(tile_r1);

        for curr_j in local_j_start..=local_j_end {
            let global_j = level_origin_j + curr_j;
            let cell_c0 = (global_j * current_level) as f64;
            let cell_c1 = cell_c0 + current_level as f64;
            let footprint_c0 = cell_c0.max(tile_c0);
            let footprint_c1 = cell_c1.min(tile_c1);

            let fraction = square_annulus_overlap_fraction(
                footprint_r0,
                footprint_r1,
                footprint_c0,
                footprint_c1,
                source_r,
                source_c,
                inner,
                outer,
            );
            if fraction <= 0.0 {
                continue;
            }

            let condition = current_array[[curr_i as usize, curr_j as usize]];
            if condition.is_nan() {
                continue;
            }

            let weight = count_array[[curr_i as usize, curr_j as usize]];
            if !weight.is_finite() {
                continue;
            }

            i_array.push(curr_i);
            j_array.push(curr_j);
            values.push(condition);
            counts.push(weight * fraction);
        }
    }
}

#[inline]
fn square_annulus_overlap_fraction(
    r0: f64,
    r1: f64,
    c0: f64,
    c1: f64,
    center_r: f64,
    center_c: f64,
    inner: f64,
    outer: f64,
) -> f32 {
    let footprint_area = (r1 - r0) * (c1 - c0);
    if footprint_area <= 0.0 {
        return 0.0;
    }

    let outer_area = rect_square_overlap_area(r0, r1, c0, c1, center_r, center_c, outer);
    if outer_area <= 0.0 {
        return 0.0;
    }

    let inner_area = rect_square_overlap_area(r0, r1, c0, c1, center_r, center_c, inner);
    let annulus_area = (outer_area - inner_area).max(0.0);
    (annulus_area / footprint_area).clamp(0.0, 1.0) as f32
}

#[inline]
fn rect_square_overlap_area(
    r0: f64,
    r1: f64,
    c0: f64,
    c1: f64,
    center_r: f64,
    center_c: f64,
    half_extent: f64,
) -> f64 {
    if half_extent <= 0.0 {
        return 0.0;
    }

    let sr0 = center_r - half_extent;
    let sr1 = center_r + half_extent;
    let sc0 = center_c - half_extent;
    let sc1 = center_c + half_extent;

    let rows = r1.min(sr1) - r0.max(sr0);
    if rows <= 0.0 {
        return 0.0;
    }

    let cols = c1.min(sc1) - c0.max(sc0);
    if cols <= 0.0 {
        return 0.0;
    }

    rows * cols
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
