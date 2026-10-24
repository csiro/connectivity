//! Frozen baseline window builder (95554da `window.rs`), kept for tests.
//!
//! Identical to the baseline except that each admitted cell's area fraction is also
//! recorded in `fractions` so structural comparisons can check it.
use crate::window::{circle_frac, square_frac, WindowMode};
use ndarray::{s, Array1, Array2, Array3, ArrayView1};
use std::collections::HashMap;
use std::iter::zip;

// Window kernel for one resolution
#[derive(Debug, Clone)]
pub struct FocalWindow {
    pub i_array: Vec<i32>,                   // i/col array of neighbour cells
    pub j_array: Vec<i32>,                   // j/row array ...
    pub values: Vec<f32>,                    // condition values of the cell
    pub trav_values: Option<Vec<f32>>,       // traversal (resistance-derived) values; None => use condition
    pub counts: Vec<f32>,                    // cell count contribution to each level (habitat-area)
    pub fractions: Vec<f32>,                 // area fraction of each cell (test instrumentation)
    pub pa_values: Vec<f32>,                 // protected fraction of each cell (1.0 when pa_to_pa off); numerator-only
    pub sims: Vec<Vec<Option<f32>>>, // similiarity with the ij cell in the current climate; scen<cell<sim>>
    pub edge_cells: Option<Vec<(i32, i32)>>, // geometry-specific outer edge cells for fringe links
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
        max_level: i32,
        win_size: i32,
        outer_win: i32,
        offsets: (usize, usize),
        cond_dict: &HashMap<i32, Array2<f32>>,
        trans_vect: &[HashMap<i32, Array3<f32>>],
        trans_ij: &Array1<f32>,
        cell_weights: &HashMap<i32, Array2<f32>>,
        pa_dict: Option<&HashMap<i32, Array2<f32>>>,
        trav_dict: Option<&HashMap<i32, Array2<f32>>>,
        window_mode: WindowMode,
    ) -> Self {
        let count_array = cell_weights
            .get(&current_level)
            .expect("Weights level not found!");
        let current_array = cond_dict
            .get(&current_level)
            .expect("Condition level not found!");
        // Optional PA-membership array for this level (Some only under target-gating).
        let pa_array = pa_dict.map(|dict| {
            dict.get(&current_level)
                .expect("PA-membership level not found!")
        });
        // Optional traversal-value array for this level (Some only when a resistance raster
        // was supplied). Drives the path weight `w`; condition is used when absent.
        let trav_array = trav_dict.map(|dict| {
            dict.get(&current_level)
                .expect("Traversal level not found!")
        });
        let current_height = current_array.shape()[0] as i32;
        let current_width = current_array.shape()[1] as i32;

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
        let effective_win_size: i32 = if current_level == max_level {
            outer_win
        } else {
            win_size
        };

        // Pre-allocate vectors with estimated capacity
        // We can give a conservative estimate to avoid multiple reallocations
        let max_win_size = win_size.max(outer_win);
        let estimated_capacity = (max_win_size * max_win_size) as usize;

        let mut i_array = Vec::with_capacity(estimated_capacity);
        let mut j_array = Vec::with_capacity(estimated_capacity);
        let mut values = Vec::with_capacity(estimated_capacity);
        let mut counts = Vec::with_capacity(estimated_capacity);
        let mut fractions = Vec::with_capacity(estimated_capacity);
        let mut pa_values = Vec::with_capacity(estimated_capacity);
        // Only allocate the traversal vector when a resistance raster was supplied.
        let mut trav_values: Option<Vec<f32>> =
            trav_array.map(|_| Vec::with_capacity(estimated_capacity));
        let mut edge_cells = if window_mode == WindowMode::Circular {
            Some(Vec::with_capacity(estimated_capacity))
        } else {
            None
        };

        let base_array = cond_dict.get(&1).expect("Base condition level not found!");
        let base_height = base_array.shape()[0] as i32;
        let base_width = base_array.shape()[1] as i32;
        collect_annulus(
            base_i,
            base_j,
            current_level,
            win_size,
            effective_win_size,
            window_mode,
            offsets,
            current_height,
            current_width,
            base_height,
            base_width,
            count_array,
            current_array,
            pa_array,
            trav_array,
            &mut i_array,
            &mut j_array,
            &mut values,
            &mut counts,
            &mut fractions,
            &mut pa_values,
            trav_values.as_mut(),
            edge_cells.as_mut(),
        );

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
            trav_values,
            counts,
            fractions,
            pa_values,
            sims,
            edge_cells,
        }
    }
}

#[inline]
fn collect_annulus(
    base_i: i32,
    base_j: i32,
    current_level: i32,
    win_size: i32,
    effective_win_size: i32,
    window_mode: WindowMode,
    offsets: (usize, usize),
    current_height: i32,
    current_width: i32,
    base_height: i32,
    base_width: i32,
    count_array: &Array2<f32>,
    current_array: &Array2<f32>,
    pa_array: Option<&Array2<f32>>,
    trav_array: Option<&Array2<f32>>,
    i_array: &mut Vec<i32>,
    j_array: &mut Vec<i32>,
    values: &mut Vec<f32>,
    counts: &mut Vec<f32>,
    fractions: &mut Vec<f32>,
    pa: &mut Vec<f32>,
    mut trav_values: Option<&mut Vec<f32>>,
    mut edge_cells: Option<&mut Vec<(i32, i32)>>,
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

            let (fraction, is_outer_edge) = match window_mode {
                WindowMode::Circular => circle_frac(
                    footprint_r0,
                    footprint_r1,
                    footprint_c0,
                    footprint_c1,
                    source_r,
                    source_c,
                    inner,
                    outer,
                ),
                WindowMode::Square => (
                    square_frac(
                        footprint_r0,
                        footprint_r1,
                        footprint_c0,
                        footprint_c1,
                        source_r,
                        source_c,
                        inner,
                        outer,
                    ),
                    false,
                ),
            };
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

            // PARC target-gating: protected fraction of this destination cell (1.0 when gating is
            // off). Applied to the connectedness numerator only (see metrics.rs), so the
            // denominator keeps the full reachable area in both gated and ungated runs. Non-PA
            // cells -> 0 drop from the numerator but remain routable; a straddle cell contributes
            // its protected fraction.
            let pa_value = match pa_array {
                Some(pa_arr) => {
                    let f = pa_arr[[curr_i as usize, curr_j as usize]];
                    if f.is_finite() {
                        f
                    } else {
                        0.0
                    }
                }
                None => 1.0,
            };

            i_array.push(curr_i);
            j_array.push(curr_j);
            values.push(condition);
            counts.push(weight * fraction);
            fractions.push(fraction);
            pa.push(pa_value);
            // Traversal value for this cell, index-aligned with `values`. Falls back to the
            // condition value if resistance is missing/non-finite here (kept in the condition
            // domain, so admitted cells always have a finite traversal value).
            if let Some(tv) = trav_values.as_deref_mut() {
                let t = trav_array
                    .map(|arr| arr[[curr_i as usize, curr_j as usize]])
                    .filter(|v| v.is_finite())
                    .unwrap_or(condition);
                tv.push(t);
            }
            if is_outer_edge {
                if let Some(edges) = edge_cells.as_mut() {
                    edges.push((curr_i, curr_j));
                }
            }
        }
    }
}

// Calcualte similarity of the transgrid layers and dealing with NaNs
// Any cell with nan coniditon is ignored in the step before this; now any nan transgrid
// will be processed as None and dealt with later in the metrics.rs
#[inline]
pub fn similarity(a: &Array1<f32>, b: &ArrayView1<f32>) -> Option<f32> {
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
