//! Checks shared with the frozen reference implementation (tests only).
use ndarray::Array2;
use std::collections::HashMap;
use anyhow::{anyhow, Result};


// Make sure cell-num mask is the same as input cond (the frozen reference's check).
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

