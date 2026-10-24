use ndarray::{s, Array2, Array3, ArrayView1};
use std::collections::HashMap;
use anyhow::{anyhow, Result};


/// Get the transgrid values for ij cell from the current climate's base level, as a view.
/// Empty when there are no transgrids or ij lies outside the array.
pub fn get_current(current: Option<&Array3<f32>>, i: usize, j: usize) -> ArrayView1<'_, f32> {
    if let Some(array3) = current {
        if i < array3.shape()[0] && j < array3.shape()[1] {
            return array3.slice(s![i, j, ..]);
        }
    }
    // Empty view if anything fails
    ArrayView1::from(&[])
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

