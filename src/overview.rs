use std::collections::HashMap;
use ndarray::{Array2, Array3};
use anyhow::{bail, Result};
use rayon::prelude::*;


#[derive(Debug, Clone, Copy)]
pub enum Resampling {
    Average,
    Count,
}


/// Check levels are a power fo two with bitwise operation.
fn check_levels(levels: &Vec<usize>) -> Result<()> {
    for &level in levels {
        if level == 0 || (level & (level - 1)) != 0 {
            bail!("Level {} is not a power of 2", level);
        }
    }
    Ok(())
}


/// Generate overviews for raster arrays
pub fn make_overview(
    base: &Array2<f32>,
    levels: &Vec<usize>,
    method: Resampling,
) -> Result<HashMap<i32, Array2<f32>>> {
    // Check levels are a power of two
    check_levels(levels)?;
    let (base_rows, base_cols) = base.dim();
    
    let mut result = HashMap::with_capacity(levels.len());
    
    for &f in levels.iter() {
        let out_rows = (base_rows + f - 1) / f;
        let out_cols = (base_cols + f - 1) / f;
        
        let raster: Vec<f32> = (0..out_rows)
            .into_par_iter()
            .flat_map(|out_r| {
                let mut row = vec![f32::NAN; out_cols];
                let base_r_start = out_r * f;
                let base_r_end = (base_r_start + f).min(base_rows);
                
                for out_c in 0..out_cols {
                    let base_c_start = out_c * f;
                    let base_c_end = (base_c_start + f).min(base_cols);
                    
                    let mut valid_count = 0.0;
                    let mut sum_values = 0.0;
                    for r in base_r_start..base_r_end {
                        for c in base_c_start..base_c_end {
                            let cell_value = base[[r, c]];
                            if !cell_value.is_nan() {
                                valid_count += 1.0;
                                if let Resampling::Average = method {
                                    sum_values += cell_value;
                                }
                            }
                        }
                    }
                    
                    if valid_count > 0.0 {
                        row[out_c] = match method {
                            Resampling::Average => sum_values / valid_count,
                            Resampling::Count => valid_count,
                        };
                    }
                }
                row
            })
            .collect();
        
        let array = Array2::from_shape_vec((out_rows, out_cols), raster)?;
        result.insert(f as i32, array);
    }
    
    Ok(result)
}


/// Generate overviews for 3D arrays (multi-band rasters such as GDM transgrids)
pub fn make_overview_3d(
    base: &Array3<f32>,  // shape: (rows, cols, bands)
    levels: &Vec<usize>,
    method: Resampling,
) -> Result<HashMap<i32, Array3<f32>>> {
    // Check levels are a power of two
    check_levels(levels)?;
    let (base_rows, base_cols, n_bands) = base.dim();
    
    let mut result = HashMap::new();
    
    for &f in levels.iter() {
        let out_rows = (base_rows + f - 1) / f;
        let out_cols = (base_cols + f - 1) / f;
        
        let raster: Vec<f32> = (0..out_rows)
            .into_par_iter()
            .flat_map(|out_r| {
                let mut row = vec![f32::NAN; out_cols * n_bands];
                let base_r_start = out_r * f;
                let base_r_end = (base_r_start + f).min(base_rows);
                
                for out_c in 0..out_cols {
                    let base_c_start = out_c * f;
                    let base_c_end = (base_c_start + f).min(base_cols);
                    
                    // Process each band (from multi-dimensional raster)
                    for band_idx in 0..n_bands {
                        let mut valid_count = 0.0;
                        let mut sum_values = 0.0;
                        
                        for r in base_r_start..base_r_end {
                            for c in base_c_start..base_c_end {
                                let cell_value = base[[r, c, band_idx]];
                                if !cell_value.is_nan() {
                                    valid_count += 1.0;
                                    if let Resampling::Average = method {
                                        sum_values += cell_value;
                                    }
                                }
                            }
                        }
                        
                        if valid_count > 0.0 {
                            row[out_c * n_bands + band_idx] = match method {
                                Resampling::Average => sum_values / valid_count,
                                Resampling::Count => valid_count,
                            };
                        }
                    }
                }
                row
            })
            .collect();
        
        let array = Array3::from_shape_vec((out_rows, out_cols, n_bands), raster)?;
        result.insert(f as i32, array);
    }
    
    Ok(result)
}

