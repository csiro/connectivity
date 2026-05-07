use ndarray::{s, Array1, Array2, Array3, ArrayView1};
use std::collections::HashMap;
use std::iter::zip;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowMode {
    Block,
    Square,
    Circular,
}

impl WindowMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "block" => Ok(Self::Block),
            "square" => Ok(Self::Square),
            "circular" => Ok(Self::Circular),
            other => Err(format!(
                "Invalid window_mode '{other}'. Expected 'block', 'square', or 'circular'."
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnnulusShape {
    Square,
    Circular,
}

// Window kernel for one resolution
#[derive(Debug, Clone)]
pub struct FocalWindow {
    pub i_array: Vec<i32>,                   // i/col array of neighbour cells
    pub j_array: Vec<i32>,                   // j/row array ...
    pub values: Vec<f32>,                    // condition values of the cell
    pub counts: Vec<f32>,                    // cell count contribution to each level (habitat-area)
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
        let effective_win_size: i32 = if current_level == max_level {
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

        let mut i_array = Vec::with_capacity(estimated_capacity);
        let mut j_array = Vec::with_capacity(estimated_capacity);
        let mut values = Vec::with_capacity(estimated_capacity);
        let mut counts = Vec::with_capacity(estimated_capacity);
        let mut edge_cells = if window_mode == WindowMode::Circular {
            Some(Vec::with_capacity(estimated_capacity))
        } else {
            None
        };

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
            WindowMode::Square | WindowMode::Circular => {
                let base_array = cond_dict.get(&1).expect("Base condition level not found!");
                let base_height = base_array.shape()[0] as i32;
                let base_width = base_array.shape()[1] as i32;
                let annulus_shape = if window_mode == WindowMode::Circular {
                    AnnulusShape::Circular
                } else {
                    AnnulusShape::Square
                };

                collect_fractional_annulus_window(
                    base_i,
                    base_j,
                    current_level,
                    win_size,
                    effective_win_size,
                    annulus_shape,
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
                    edge_cells.as_mut(),
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
            edge_cells,
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
    annulus_shape: AnnulusShape,
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

            let (fraction, is_outer_edge) = match annulus_shape {
                AnnulusShape::Square => (
                    square_annulus_overlap_fraction(
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
                AnnulusShape::Circular => circular_annulus_overlap_fraction_and_edge(
                    footprint_r0,
                    footprint_r1,
                    footprint_c0,
                    footprint_c1,
                    source_r,
                    source_c,
                    inner,
                    outer,
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

            i_array.push(curr_i);
            j_array.push(curr_j);
            values.push(condition);
            counts.push(weight * fraction);
            if is_outer_edge {
                if let Some(edges) = edge_cells.as_mut() {
                    edges.push((curr_i, curr_j));
                }
            }
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

#[inline]
fn circular_annulus_overlap_fraction_and_edge(
    r0: f64,
    r1: f64,
    c0: f64,
    c1: f64,
    center_r: f64,
    center_c: f64,
    inner: f64,
    outer: f64,
) -> (f32, bool) {
    let footprint_area = (r1 - r0) * (c1 - c0);
    if footprint_area <= 0.0 {
        return (0.0, false);
    }

    let outer_area = rect_circle_overlap_area(r0, r1, c0, c1, center_r, center_c, outer);
    if outer_area <= 0.0 {
        return (0.0, false);
    }

    let inner_area = rect_circle_overlap_area(r0, r1, c0, c1, center_r, center_c, inner);
    let annulus_area = (outer_area - inner_area).max(0.0);
    let fraction = (annulus_area / footprint_area).clamp(0.0, 1.0) as f32;
    let is_outer_edge = annulus_area > 0.0
        && rect_intersects_circle_boundary(r0, r1, c0, c1, center_r, center_c, outer);

    (fraction, is_outer_edge)
}

#[inline]
fn rect_circle_overlap_area(
    r0: f64,
    r1: f64,
    c0: f64,
    c1: f64,
    center_r: f64,
    center_c: f64,
    radius: f64,
) -> f64 {
    let rect_area = (r1 - r0) * (c1 - c0);
    if rect_area <= 0.0 || radius <= 0.0 {
        return 0.0;
    }

    let x0 = c0 - center_c;
    let x1 = c1 - center_c;
    let y0 = r0 - center_r;
    let y1 = r1 - center_r;
    let radius_sq = radius * radius;

    if rect_min_distance_sq_to_origin(x0, x1, y0, y1) >= radius_sq {
        return 0.0;
    }
    if rect_max_distance_sq_to_origin(x0, x1, y0, y1) <= radius_sq {
        return rect_area;
    }

    let x_parts = split_at_zero(x0, x1);
    let y_parts = split_at_zero(y0, y1);
    let mut area = 0.0;

    for &(xa, xb) in &x_parts {
        if xb <= xa {
            continue;
        }
        let (qx0, qx1) = if xb <= 0.0 { (-xb, -xa) } else { (xa, xb) };

        for &(ya, yb) in &y_parts {
            if yb <= ya {
                continue;
            }
            let (qy0, qy1) = if yb <= 0.0 { (-yb, -ya) } else { (ya, yb) };
            area += first_quadrant_rect_circle_area(qx0, qx1, qy0, qy1, radius);
        }
    }

    area.clamp(0.0, rect_area)
}

#[inline]
fn first_quadrant_rect_circle_area(x0: f64, x1: f64, y0: f64, y1: f64, radius: f64) -> f64 {
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }

    first_quadrant_circle_area(x1, y1, radius)
        - first_quadrant_circle_area(x0, y1, radius)
        - first_quadrant_circle_area(x1, y0, radius)
        + first_quadrant_circle_area(x0, y0, radius)
}

#[inline]
fn first_quadrant_circle_area(x: f64, y: f64, radius: f64) -> f64 {
    if x <= 0.0 || y <= 0.0 || radius <= 0.0 {
        return 0.0;
    }

    let x = x.min(radius);
    if x <= 0.0 {
        return 0.0;
    }

    if y >= radius {
        return circle_segment_antiderivative(x, radius);
    }

    let x_flat = (radius * radius - y * y).max(0.0).sqrt();
    let rect_width = x.min(x_flat);
    let mut area = rect_width * y;
    if x > x_flat {
        area += circle_segment_antiderivative(x, radius)
            - circle_segment_antiderivative(x_flat, radius);
    }

    area
}

#[inline]
fn circle_segment_antiderivative(x: f64, radius: f64) -> f64 {
    let x = x.clamp(0.0, radius);
    let y = (radius * radius - x * x).max(0.0).sqrt();
    let angle = (x / radius).clamp(-1.0, 1.0).asin();
    0.5 * (x * y + radius * radius * angle)
}

#[inline]
fn split_at_zero(a: f64, b: f64) -> [(f64, f64); 2] {
    if a < 0.0 && b > 0.0 {
        [(a, 0.0), (0.0, b)]
    } else {
        [(a, b), (0.0, 0.0)]
    }
}

#[inline]
fn rect_intersects_circle_boundary(
    r0: f64,
    r1: f64,
    c0: f64,
    c1: f64,
    center_r: f64,
    center_c: f64,
    radius: f64,
) -> bool {
    if radius <= 0.0 {
        return false;
    }

    let x0 = c0 - center_c;
    let x1 = c1 - center_c;
    let y0 = r0 - center_r;
    let y1 = r1 - center_r;
    let radius_sq = radius * radius;
    let eps = (radius_sq.abs() + 1.0) * 1.0e-12;

    rect_min_distance_sq_to_origin(x0, x1, y0, y1) <= radius_sq + eps
        && rect_max_distance_sq_to_origin(x0, x1, y0, y1) >= radius_sq - eps
}

#[inline]
fn rect_min_distance_sq_to_origin(x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    let dx = if x0 > 0.0 {
        x0
    } else if x1 < 0.0 {
        -x1
    } else {
        0.0
    };
    let dy = if y0 > 0.0 {
        y0
    } else if y1 < 0.0 {
        -y1
    } else {
        0.0
    };
    dx * dx + dy * dy
}

#[inline]
fn rect_max_distance_sq_to_origin(x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    let dx = x0.abs().max(x1.abs());
    let dy = y0.abs().max(y1.abs());
    dx * dx + dy * dy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_overlap_returns_full_rectangle_inside_disk() {
        let area = rect_circle_overlap_area(9.0, 11.0, 9.0, 11.0, 10.0, 10.0, 3.0);
        assert!((area - 4.0).abs() < 1.0e-12);
    }

    #[test]
    fn circle_overlap_matches_quarter_circle() {
        let radius = 3.0;
        let area = rect_circle_overlap_area(0.0, radius, 0.0, radius, 0.0, 0.0, radius);
        let expected = std::f64::consts::PI * radius * radius * 0.25;
        assert!((area - expected).abs() < 1.0e-12);
    }

    #[test]
    fn circular_annulus_marks_outer_boundary_not_inner_boundary() {
        let (_, outer_edge) =
            circular_annulus_overlap_fraction_and_edge(-0.5, 0.5, 2.5, 3.5, 0.0, 0.0, 1.0, 3.0);
        assert!(outer_edge);

        let (_, inner_edge) =
            circular_annulus_overlap_fraction_and_edge(-0.5, 0.5, 0.5, 1.5, 0.0, 0.0, 1.0, 3.0);
        assert!(!inner_edge);
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
