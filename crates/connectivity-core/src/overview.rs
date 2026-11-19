use anyhow::{bail, Result};
use ndarray::{Array2, ArrayView2, ArrayView3};
#[cfg(test)]
use ndarray::Array3;
use rayon::prelude::*;
#[cfg(test)]
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
#[cfg(test)]
fn check_levels(levels: &[usize]) -> Result<()> {
    for &level in levels {
        if level == 0 || (level & (level - 1)) != 0 {
            bail!("Level {} is not a power of 2", level);
        }
    }
    Ok(())
}


/// Relative area, cos(latitude), of each of a geographic tile's `rows` from its base transform.
pub fn row_areas(tr: &Affine, rows: usize) -> Vec<f32> {
    let mut factors = Vec::with_capacity(rows);
    for local_r in 0..rows {
        let (_, lat_deg) = tr.xy(local_r as i32, 0);
        let w = lat_deg.to_radians().cos().abs() as f32;
        factors.push(if w.is_finite() { w.max(0.0) } else { 0.0 });
    }
    factors
}

/// Rows and columns of the level-`f` grid of a `rows` x `cols` tile at global `offsets`: every
/// level-`f` cell that intersects the tile.
pub fn level_dim(rows: usize, cols: usize, f: usize, offsets: (usize, usize)) -> (usize, usize) {
    let f = f as i32;
    let (r0, c0) = (offsets.0 as i32, offsets.1 as i32);
    let (r1, c1) = (r0 + rows as i32, c0 + cols as i32); // exclusive
    let out_rows = (r1 - 1).div_euclid(f) - r0.div_euclid(f) + 1;
    let out_cols = (c1 - 1).div_euclid(f) - c0.div_euclid(f) + 1;
    (out_rows as usize, out_cols as usize)
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
/// - level -> overview array covering exactly the global overview cells that intersect this tile;
///   level 1 is a copy of `base` (Average, Sum) or `base_weights` (Count, Area). `core` borrows
///   the base level instead and builds the others with `coarse_level`; this map serves the frozen
///   reference implementation.
#[cfg(test)]
pub fn make_overview(
    base: &Array2<f32>,
    levels: &[usize],
    offsets: (usize, usize),
    method: Resampling,
    base_transform: Option<&Affine>,
) -> Result<HashMap<i32, Array2<f32>>> {
    check_levels(levels)?;
    let areas = area_rows(method, base_transform, base.nrows())?;
    let mut result = HashMap::with_capacity(levels.len());
    for &f in levels {
        let level = match (f, method) {
            (1, Resampling::Average | Resampling::Sum) => base.clone(),
            (1, _) => base_weights(base.view(), method, areas.as_deref()),
            _ => coarse_level(base.view(), f, offsets, method, areas.as_deref()),
        };
        result.insert(f as i32, level);
    }
    Ok(result)
}

/// Per-row area factors (`row_areas`) for Area resampling, which needs the base transform.
pub fn area_rows(method: Resampling, base_transform: Option<&Affine>, rows: usize) -> Result<Option<Vec<f32>>> {
    if !matches!(method, Resampling::Area) {
        return Ok(None);
    }
    let tr = base_transform
        .ok_or_else(|| anyhow::anyhow!("Area resampling requires base-level transform information."))?;
    Ok(Some(row_areas(tr, rows)))
}

/// Level-1 weights of `base`: 1 (Count) or the row's area factor (Area) where `base` is finite,
/// else NaN.
pub fn base_weights(base: ArrayView2<f32>, method: Resampling, areas: Option<&[f32]>) -> Array2<f32> {
    Array2::from_shape_fn(base.dim(), |(r, c)| {
        if !base[[r, c]].is_finite() {
            f32::NAN
        } else if let Resampling::Area = method {
            areas.expect("Area resampling requires row area factors.")[r]
        } else {
            1.0
        }
    })
}

/// Level `f` (> 1) of `base`: each cell of the level grid aggregates the tile's base cells it
/// covers, in row-major f32 order (mean, count, area or sum of the finite values; NaN without any).
pub fn coarse_level(
    base: ArrayView2<f32>,
    f: usize,
    offsets: (usize, usize),
    method: Resampling,
    row_area_factor: Option<&[f32]>,
) -> Array2<f32> {
    let (base_rows_u, base_cols_u) = base.dim();
    let (out_rows, out_cols) = level_dim(base_rows_u, base_cols_u, f, offsets);
    let f = f as i32;
    let (tile_r0, tile_c0) = (offsets.0 as i32, offsets.1 as i32);
    let tile_r1 = tile_r0 + base_rows_u as i32; // exclusive
    let tile_c1 = tile_c0 + base_cols_u as i32; // exclusive
    let ov_row_start = tile_r0.div_euclid(f);
    let ov_col_start = tile_c0.div_euclid(f);

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

    Array2::from_shape_vec((out_rows, out_cols), raster).expect("level grid shape")
}


/// Generate globally-aligned overviews for 3D arrays (rows, cols, bands).
///
/// Fixes the same seam issue as the 2D version by:
/// - defining each overview cell by global indices floor(global_pixel / level)
/// - intersecting each global f×f block with the tile extent
/// - avoiding saturating_sub by doing intersections in signed coords
#[cfg(test)]
pub fn make_overview_3d(
    base: &Array3<f32>, // shape: (rows, cols, bands)
    levels: &[usize],
    offsets: (usize, usize), // (tile_row0, tile_col0) in ORIGINAL dataset pixel coords
    method: Resampling,
) -> Result<HashMap<i32, Array3<f32>>> {
    check_levels(levels)?;
    let mut result = HashMap::with_capacity(levels.len());
    for &f in levels {
        if f == 1 {
            result.insert(1, base.clone());
            continue;
        }
        let (rows, cols) = level_dim(base.dim().0, base.dim().1, f, offsets);
        let mut raster = vec![f32::NAN; rows * cols * base.dim().2];
        coarse_bands_into(base.view(), f, offsets, method, |_, _| true, &mut raster, 1, 0)?;
        result.insert(f as i32, Array3::from_shape_vec((rows, cols, base.dim().2), raster)?);
    }
    Ok(result)
}

/// Level `f` (> 1) of `base`'s bands over the tile cells (r, c) for which `keep` holds, written
/// to `out`, which holds the level grid's rows x columns x bands x `lanes` values (row-major), at
/// lane `lane`. Sums run per band in f32 in row-major order of the base cells, as the crossing
/// modes' parts sum their bands (`crossing::Base::add`); cells and bands without a value leave
/// `out` unchanged (NaN when it is initialised so).
#[allow(clippy::too_many_arguments)]
pub fn coarse_bands_into(
    base: ArrayView3<f32>,
    f: usize,
    offsets: (usize, usize),
    method: Resampling,
    keep: impl Fn(usize, usize) -> bool + Sync,
    out: &mut [f32],
    lanes: usize,
    lane: usize,
) -> Result<()> {
    if !matches!(method, Resampling::Average | Resampling::Sum) {
        bail!(
            "3D overviews support only Average and Sum resampling; got {:?}",
            method
        );
    }
    let is_average = matches!(method, Resampling::Average);

    let (base_rows_u, base_cols_u, n_bands) = base.dim();
    let (out_rows, out_cols) = level_dim(base_rows_u, base_cols_u, f, offsets);
    assert!(lane < lanes && out.len() >= out_rows * out_cols * n_bands * lanes, "packed level size");
    let f = f as i32;
    let (tile_r0, tile_c0) = (offsets.0 as i32, offsets.1 as i32);
    let tile_r1 = tile_r0 + base_rows_u as i32; // exclusive
    let tile_c1 = tile_c0 + base_cols_u as i32; // exclusive
    let ov_row_start = tile_r0.div_euclid(f);
    let ov_col_start = tile_c0.div_euclid(f);

    let row_stride = out_cols * n_bands * lanes;
    out[..out_rows * row_stride]
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
                            if v.is_finite() && keep(r, c) {
                                valid_count += 1.0;
                                sum_values += v;
                            }
                        }
                    }

                    if valid_count > 0.0 {
                        row[(local_c * n_bands + band_idx) * lanes + lane] = if is_average {
                            sum_values / valid_count
                        } else {
                            sum_values
                        };
                    }
                }
            }
        });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_uses_local_rows() {
        let base = Array2::<f32>::ones((2, 1));
        let transform = Affine {
            x_scale: 1.0,
            x_skew: 0.0,
            x_origin: 0.0,
            y_skew: 0.0,
            y_scale: -1.0,
            y_origin: 80.0,
        };

        let overview = make_overview(&base, &[1], (10, 0), Resampling::Area, Some(&transform))
            .expect("area overview should build");
        let weights = overview.get(&1).expect("level 1 weights should exist");

        let expected_row0 = 79.5_f32.to_radians().cos().abs();
        let expected_row1 = 78.5_f32.to_radians().cos().abs();

        assert!((weights[[0, 0]] - expected_row0).abs() < 1.0e-6);
        assert!((weights[[1, 0]] - expected_row1).abs() < 1.0e-6);
    }
}
