use ndarray::{Array3, Array2, Array1, s};
use std::collections::HashMap;
use anyhow::{anyhow, Result};


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

