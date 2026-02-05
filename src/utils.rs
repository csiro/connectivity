use ndarray::{Array3, Array2, Array1, s};
use std::collections::HashMap;
use rayon::prelude::*;
use anyhow::{Result, anyhow};

/// Get the transgrid values for ij cell
pub fn get_current(trans_maps: &[HashMap<i32, Array3<f32>>], i: usize, j: usize) -> Array1<f32> {
    let trans_array = &trans_maps[0];

    if let Some(array3) = trans_array.get(&1) {
        if i < array3.shape()[0] && j < array3.shape()[1] {
            return array3.slice(s![i, j, ..]).to_owned(); // returns Array1<f32>
        }
    }
    // Return empty array if anything fails
    Array1::zeros(0)
}


/// For each overview factor `f` in `cond_map` (keys), compute the number of valid base (level=1)
/// pixels contributing to each overview pixel, following SUM-overview semantics:
/// - count = number of finite pixels in the corresponding fxf (edge-truncated) block
/// - if count == 0 -> NaN, else count as f32
///
/// Assumes `cond_map[&1]` exists and contains 1.0 for valid and NaN for invalid (or any finite values for valid).
pub fn count_cells(
    cond_map: &HashMap<i32, Array2<f32>>,
) -> Result<HashMap<i32, Array2<f32>>, &'static str> {
    let base = cond_map.get(&1).ok_or("Missing base level key=1")?;
    let (rows, cols) = base.dim();

    let keys: Vec<i32> = cond_map.keys().copied().collect();

    let pairs: Vec<(i32, Array2<f32>)> = keys
        .into_par_iter()
        .map(|f_i32| {
            if f_i32 <= 0 {
                return Err("Overview factors must be positive");
            }
            let f = f_i32 as usize;

            let out_rows = (rows + f - 1) / f;
            let out_cols = (cols + f - 1) / f;

            // Fill each output row independently in parallel.
            let mut buf = vec![f32::NAN; out_rows * out_cols];

            buf.par_chunks_mut(out_cols)
                .enumerate()
                .for_each(|(orow, out_row)| {
                    let r0 = orow * f;
                    let r1 = ((orow + 1) * f).min(rows);

                    // column-accumulator for this output row: u16 is enough up to f*f (max 65535)
                    let mut acc = vec![0u32; out_cols];

                    for r in r0..r1 {
                        // Count valids in this base row, binned into f-wide column blocks
                        for ocol in 0..out_cols {
                            let c0 = ocol * f;
                            let c1 = ((ocol + 1) * f).min(cols);

                            let mut ccount = 0u32;
                            for c in c0..c1 {
                                if base[(r, c)].is_finite() {
                                    ccount += 1;
                                }
                            }
                            acc[ocol] += ccount;
                        }
                    }

                    for ocol in 0..out_cols {
                        let count = acc[ocol];
                        out_row[ocol] = if count > 0 { count as f32 } else { f32::NAN };
                    }
                });

            let arr = Array2::from_shape_vec((out_rows, out_cols), buf)
                .map_err(|_| "Failed to reshape output array")?;

            Ok((f_i32, arr))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(pairs.into_iter().collect())
}


// Make sure cell-num mask is the same as input cond.
pub fn check_dims(
    a: &HashMap<i32, Array2<f32>>,
    b: &HashMap<i32, Array2<f32>>,
) -> Result<()> {
    // 1. Check key sets are identical
    if a.len() != b.len() {
        return Err(anyhow!(
            "HashMaps have different number of keys: {} vs {}",
            a.len(),
            b.len()
        ));
    }

    for (k, arr_a) in a {
        let arr_b = b.get(k).ok_or_else(|| {
            anyhow!("Key {} exists in first map but not in second", k)
        })?;

        // 2. Check dimensions
        if arr_a.dim() != arr_b.dim() {
            return Err(anyhow!(
                "Dimension mismatch for key {}: {:?} vs {:?}",
                k,
                arr_a.dim(),
                arr_b.dim()
            ));
        }
    }

    Ok(())
}

