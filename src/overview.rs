use anyhow::{bail, Result};
use ndarray::{Array2, Array3};
use rayon::prelude::*;
use std::collections::HashMap;
use crate::affine::Affine;

#[derive(Debug, Clone, Copy)]
pub enum Resampling {
    Average,
    Count,
    Area,
    #[allow(dead_code)]
    Sum,
}

/// Check levels are a power of two with bitwise operation.
fn check_levels(levels: &[usize]) -> Result<()> {
    for &level in levels {
        if level == 0 || (level & (level - 1)) != 0 {
            bail!("Level {} is not a power of 2", level);
        }
    }
    Ok(())
}


/// Generate globally-anchored overviews for a SINGLE tile (polygon crop).
///
/// The key fix is that aggregation windows are aligned to the ORIGINAL dataset pixel grid
/// using `tile_row0` / `tile_col0`. This prevents seams when adjacent polygon runs are
/// later stitched, because overview cell boundaries are identical across independent runs.
///
/// Inputs:
/// - `base`: tile-local array where base[[0,0]] corresponds to global pixel (tile_row0, tile_col0)
/// - `offsets`: (tile_row0, tile_col0) in ORIGINAL dataset pixel coordinates (top-left of this tile)
///
/// Output:
/// - level -> overview array covering exactly the global overview cells that intersect this tile
///
/// Note:
/// - This function does NOT need the full dataset width/height.
/// - If you write these arrays to GeoTIFF, you must ensure the output transform is consistent
///   with the global overview grid origin (often easiest if you also track ov_row_start/ov_col_start).
pub fn make_overview(
    base: &Array2<f32>,
    levels: &[usize],
    offsets: (usize, usize),
    method: Resampling,
    base_transform: Option<&Affine>,
) -> Result<HashMap<i32, Array2<f32>>> {
    check_levels(levels)?;

    let (base_rows_u, base_cols_u) = base.dim();
    let base_rows = base_rows_u as i32;
    let base_cols = base_cols_u as i32;

    let (tile_row0_u, tile_col0_u) = offsets;
    let tile_row0 = tile_row0_u as i32;
    let tile_col0 = tile_col0_u as i32;

    let tile_r0 = tile_row0;
    let tile_c0 = tile_col0;
    let tile_r1 = tile_row0 + base_rows; // exclusive
    let tile_c1 = tile_col0 + base_cols; // exclusive

    // For area-weighted counts, precompute a per-row relative area factor: cos(latitude).
    // This applies only to geographic grids and requires the base-level transform.
    let row_area_factor: Option<Vec<f32>> = if matches!(method, Resampling::Area) {
        let tr = base_transform.ok_or_else(|| {
            anyhow::anyhow!("Area resampling requires base-level transform information.")
        })?;

        let mut factors = Vec::with_capacity(base_rows_u);
        for local_r in 0..base_rows_u {
            let global_r = tile_row0 + local_r as i32;
            let row_f = global_r as f64 + 0.5;
            let col_f = 0.5_f64;
            let lat_deg = tr.y_skew * col_f + tr.y_scale * row_f + tr.y_origin;
            let w = lat_deg.to_radians().cos().abs() as f32;
            factors.push(if w.is_finite() { w.max(0.0) } else { 0.0 });
        }
        Some(factors)
    } else {
        None
    };
    let row_area_factor = row_area_factor.as_deref();

    let mut result = HashMap::with_capacity(levels.len());

    for &f_u in levels {
        let f = f_u as i32;

        let ov_row_start = tile_r0.div_euclid(f);
        let ov_col_start = tile_c0.div_euclid(f);
        let ov_row_end = (tile_r1 - 1).div_euclid(f);
        let ov_col_end = (tile_c1 - 1).div_euclid(f);

        let out_rows = (ov_row_end - ov_row_start + 1) as usize;
        let out_cols = (ov_col_end - ov_col_start + 1) as usize;

        // Preallocate row-major output
        let mut raster = vec![f32::NAN; out_rows * out_cols];

        raster
            .par_chunks_mut(out_cols)
            .enumerate()
            .for_each(|(local_r, row)| {
                let ov_r = ov_row_start + local_r as i32;

                let block_r0 = ov_r * f;
                let block_r1 = block_r0 + f;

                let r0 = block_r0.max(tile_r0);
                let r1 = block_r1.min(tile_r1);
                if r1 <= r0 {
                    return;
                }

                let tile_r_start = (r0 - tile_r0) as usize;
                let tile_r_end = (r1 - tile_r0) as usize;

                for local_c in 0..out_cols {
                    let ov_c = ov_col_start + local_c as i32;

                    let block_c0 = ov_c * f;
                    let block_c1 = block_c0 + f;

                    let c0 = block_c0.max(tile_c0);
                    let c1 = block_c1.min(tile_c1);
                    if c1 <= c0 {
                        continue;
                    }

                    let tile_c_start = (c0 - tile_c0) as usize;
                    let tile_c_end = (c1 - tile_c0) as usize;

                    let mut valid_count = 0.0_f32;
                    let mut sum_values = 0.0_f32;
                    let mut area_count = 0.0_f32;

                    for r in tile_r_start..tile_r_end {
                        for c in tile_c_start..tile_c_end {
                            let v = base[[r, c]];
                            if v.is_finite() {
                                valid_count += 1.0;
                                match method {
                                    Resampling::Average | Resampling::Sum => {
                                        sum_values += v;
                                    }
                                    Resampling::Area => {
                                        if let Some(area_rows) = row_area_factor {
                                            area_count += area_rows[r];
                                        }
                                    }
                                    Resampling::Count => {}
                                }
                            }
                        }
                    }

                    if valid_count > 0.0 {
                        row[local_c] = match method {
                            Resampling::Average => sum_values / valid_count,
                            Resampling::Count => valid_count,
                            Resampling::Area => area_count,
                            Resampling::Sum => sum_values,
                        };
                    }
                }
            });

        result.insert(f as i32, Array2::from_shape_vec((out_rows, out_cols), raster)?);
    }

    Ok(result)
}



/// Generate globally-aligned overviews for 3D arrays (rows, cols, bands).
///
/// Fixes the same seam issue as the 2D version by:
/// - defining each overview cell by global indices floor(global_pixel / level)
/// - intersecting each global f×f block with the tile extent
/// - avoiding saturating_sub by doing intersections in signed coords
pub fn make_overview_3d(
    base: &Array3<f32>, // shape: (rows, cols, bands)
    levels: &[usize],
    offsets: (usize, usize), // (tile_row0, tile_col0) in ORIGINAL dataset pixel coords
    method: Resampling,
) -> Result<HashMap<i32, Array3<f32>>> {
    check_levels(levels)?;

    let (base_rows_u, base_cols_u, n_bands_u) = base.dim();
    let base_rows = base_rows_u as i32;
    let base_cols = base_cols_u as i32;
    let n_bands = n_bands_u as usize;

    let (tile_row0_u, tile_col0_u) = offsets;
    let tile_row0 = tile_row0_u as i32;
    let tile_col0 = tile_col0_u as i32;

    let tile_r0 = tile_row0;
    let tile_c0 = tile_col0;
    let tile_r1 = tile_row0 + base_rows; // exclusive
    let tile_c1 = tile_col0 + base_cols; // exclusive

    let mut result = HashMap::with_capacity(levels.len());

    for &f_u in levels {
        let f = f_u as i32;

        let ov_row_start = tile_r0.div_euclid(f);
        let ov_col_start = tile_c0.div_euclid(f);
        let ov_row_end = (tile_r1 - 1).div_euclid(f);
        let ov_col_end = (tile_c1 - 1).div_euclid(f);

        let out_rows = (ov_row_end - ov_row_start + 1) as usize;
        let out_cols = (ov_col_end - ov_col_start + 1) as usize;

        let row_stride = out_cols * n_bands;
        let mut raster = vec![f32::NAN; out_rows * row_stride];

        raster
            .par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(local_r, row)| {
                let ov_r = ov_row_start + local_r as i32;

                let block_r0 = ov_r * f;
                let block_r1 = block_r0 + f;

                let r0 = block_r0.max(tile_r0);
                let r1 = block_r1.min(tile_r1);
                if r1 <= r0 {
                    return;
                }

                let tile_r_start = (r0 - tile_r0) as usize;
                let tile_r_end = (r1 - tile_r0) as usize;

                for local_c in 0..out_cols {
                    let ov_c = ov_col_start + local_c as i32;

                    let block_c0 = ov_c * f;
                    let block_c1 = block_c0 + f;

                    let c0 = block_c0.max(tile_c0);
                    let c1 = block_c1.min(tile_c1);
                    if c1 <= c0 {
                        continue;
                    }

                    let tile_c_start = (c0 - tile_c0) as usize;
                    let tile_c_end = (c1 - tile_c0) as usize;

                    for band_idx in 0..n_bands {
                        let mut valid_count = 0.0_f32;
                        let mut sum_values = 0.0_f32;

                        for r in tile_r_start..tile_r_end {
                            for c in tile_c_start..tile_c_end {
                                let v = base[[r, c, band_idx]];
                                if v.is_finite() {
                                    valid_count += 1.0;
                                    if let Resampling::Average | Resampling::Sum = method {
                                        sum_values += v;
                                    }
                                }
                            }
                        }

                        if valid_count > 0.0 {
                            row[local_c * n_bands + band_idx] = match method {
                                Resampling::Average => sum_values / valid_count,
                                Resampling::Count => valid_count,
                                // Area-weighted counting is only used for 2D cell weights.
                                // For 3D overviews this falls back to plain valid counts.
                                Resampling::Area => valid_count,
                                Resampling::Sum => sum_values,
                            };
                        }
                    }
                }
            });

        result.insert(
            f as i32,
            Array3::from_shape_vec((out_rows, out_cols, n_bands), raster)?,
        );
    }

    Ok(result)
}
