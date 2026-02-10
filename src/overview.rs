use std::collections::HashMap;
use ndarray::Array2;
use anyhow::Result;
// use rayon::prelude::*;


#[derive(Debug, Clone, Copy)]
pub enum Resampling {
    Average,
    Count,
}

// fn check_levels(levels: &Vec<usize>) {

// }


// Generate overviews for raster arrays
pub fn make_overview(
    base: &Array2<f32>,
    levels: &Vec<usize>,
    method: Resampling,
) -> Result<HashMap<i32, Array2<f32>>> {
    let (base_rows, base_cols) = base.dim();
    
    let mut result = HashMap::with_capacity(levels.len());
    
    for &f in levels.iter() {
        let out_rows = (base_rows + f - 1) / f;
        let out_cols = (base_cols + f - 1) / f;
        
        let mut raster = Array2::from_elem((out_rows, out_cols), f32::NAN);
        
        for out_r in 0..out_rows {
            for out_c in 0..out_cols {
                let base_r_start = out_r * f;
                let base_r_end = (base_r_start + f).min(base_rows);
                let base_c_start = out_c * f;
                let base_c_end = (base_c_start + f).min(base_cols);
                
                let mut valid_count = 0.0;
                let mut sum_values = 0.0;
                for r in base_r_start..base_r_end {
                    for c in base_c_start..base_c_end {
                        let cell_value = base[[r, c]];
                        if cell_value.is_finite() {
                            valid_count += 1.0;
                            if let Resampling::Average = method {
                                sum_values += cell_value;
                            }
                        }
                    }
                }
                
                if valid_count > 0.0 {
                    let out = match method {
                        Resampling::Average => sum_values / valid_count,
                        Resampling::Count => valid_count,
                    };
                    raster[[out_r, out_c]] = out;
                }
            }
        }
        
        result.insert(f as i32, raster);
    }
    
    Ok(result)
}



// /// For each factor f in `cond_map` (including 1), build an Array2 where each cell stores
// /// the number of finite base (key=1) pixels inside that f×f block. If count==0 -> NaN.
// /// Output shape matches the existing overview shape in `cond_map` when present; otherwise
// /// falls back to ceil(rows/f), ceil(cols/f).
// pub fn count_cells(
//     cond_map: &HashMap<i32, Array2<f32>>,
// ) -> Result<HashMap<i32, Array2<f32>>> {
//     let base = cond_map
//         .get(&1)
//         .context("Missing base level key=1")?;
//     let (rows, cols) = base.dim();
    
//     let keys: Vec<i32> = cond_map.keys().copied().collect();
//     let pairs: Vec<(i32, Array2<f32>)> = keys
//         .into_par_iter()
//         .map(|f_i32| -> Result<(i32, Array2<f32>)> {
//             if f_i32 <= 0 {
//                 bail!("Overview factors must be positive (got {f_i32})");
//             }
//             let f = f_i32 as usize;
//             let out_rows = (rows + f - 1) / f;
//             let out_cols = (cols + f - 1) / f;
            
//             let mut buf = vec![f32::NAN; out_rows * out_cols];
//             buf.par_chunks_mut(out_cols)
//                 .enumerate()
//                 .for_each(|(orow, out_row)| {
//                     let r0 = orow * f;
//                     if r0 >= rows {
//                         return;
//                     }
//                     let r1 = ((orow + 1) * f).min(rows);
//                     for ocol in 0..out_cols {
//                         let c0 = ocol * f;
//                         if c0 >= cols {
//                             continue;
//                         }
//                         let c1 = ((ocol + 1) * f).min(cols);
//                         let mut count = 0u32;
//                         for r in r0..r1 {
//                             for c in c0..c1 {
//                                 if base[(r, c)].is_finite() {
//                                     count += 1;
//                                 }
//                             }
//                         }
//                         out_row[ocol] = if count > 0 {
//                             count as f32
//                         } else {
//                             f32::NAN
//                         };
//                     }
//                 });
//             let arr = Array2::from_shape_vec((out_rows, out_cols), buf)
//                 .context("Failed to reshape output array")?;
//             Ok((f_i32, arr))
//         })
//         .collect::<Result<Vec<_>>>()?;
//     Ok(pairs.into_iter().collect())
// }

