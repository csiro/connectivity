use anyhow::{bail, Result};
use ndarray::Array2;
use rayon::prelude::*;
use std::collections::VecDeque;

#[inline]
fn clamp_index(x: isize, len: usize) -> usize {
    if x < 0 {
        0
    } else if x >= len as isize {
        len - 1
    } else {
        x as usize
    }
}

fn transpose_into(input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
    output
        .par_chunks_mut(rows)
        .enumerate()
        .for_each(|(j, out_row)| {
            for i in 0..rows {
                out_row[i] = input[i * cols + j];
            }
        });
}

fn box_filter_rows_into(
    input: &[f32],
    output: &mut [f32],
    _rows: usize,
    cols: usize,
    left: usize,
    right: usize,
) {
    let window = (left + right + 1) as f32;

    output
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(i, out_row)| {
            let base = i * cols;
            let row = &input[base..base + cols];

            let mut sum = 0.0_f32;
            for off in -(left as isize)..=(right as isize) {
                let jj = clamp_index(off, cols);
                sum += row[jj];
            }
            out_row[0] = sum / window;

            for j in 1..cols {
                let out_idx = clamp_index(j as isize - 1 - left as isize, cols);
                let in_idx = clamp_index(j as isize + right as isize, cols);
                sum += row[in_idx] - row[out_idx];
                out_row[j] = sum / window;
            }
        });
}

fn nearest_fill(
    filled: &mut [f32],
    nan_mask: &[bool],
    nan_indices: &[usize],
    rows: usize,
    cols: usize,
) {
    let n = rows * cols;
    let mut source = vec![usize::MAX; n];
    let mut queue = VecDeque::with_capacity(n);

    for idx in 0..n {
        if !nan_mask[idx] {
            source[idx] = idx;
            queue.push_back(idx);
        }
    }

    let nbrs: [(isize, isize); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];

    while let Some(idx) = queue.pop_front() {
        let src = source[idx];
        let r = idx / cols;
        let c = idx % cols;

        for (dr, dc) in nbrs {
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            if nr < 0 || nr >= rows as isize || nc < 0 || nc >= cols as isize {
                continue;
            }

            let nidx = nr as usize * cols + nc as usize;
            if source[nidx] == usize::MAX {
                source[nidx] = src;
                queue.push_back(nidx);
            }
        }
    }

    for &idx in nan_indices {
        filled[idx] = filled[source[idx]];
    }
}

pub fn inpaint_nans_diffusion(
    img: &Array2<f32>,
    size: usize,
    max_iter: usize,
    tol: f32,
    init: &str,
) -> Result<Array2<f32>> {
    let (rows, cols) = img.dim();
    if rows == 0 || cols == 0 {
        return Ok(img.clone());
    }

    let mut filled: Vec<f32> = img.iter().copied().collect();
    let n = rows * cols;
    let mut nan_mask = vec![false; n];
    let mut nan_indices: Vec<usize> = Vec::new();
    let mut valid_count = 0usize;

    for (idx, &v) in filled.iter().enumerate() {
        if v.is_nan() {
            nan_mask[idx] = true;
            nan_indices.push(idx);
        } else {
            valid_count += 1;
        }
    }

    if nan_indices.is_empty() {
        return Ok(img.clone());
    }
    if valid_count == 0 {
        bail!("All pixels are NaN; cannot inpaint.");
    }

    match init {
        "nearest" => nearest_fill(&mut filled, &nan_mask, &nan_indices, rows, cols),
        "mean" => {
            let mean = filled
                .iter()
                .filter(|v| !v.is_nan())
                .copied()
                .sum::<f32>()
                / valid_count as f32;
            for &idx in &nan_indices {
                filled[idx] = mean;
            }
        }
        _ => bail!("init must be 'nearest' or 'mean'"),
    }

    if nan_indices.len() < 10 || max_iter == 0 {
        return Array2::from_shape_vec((rows, cols), filled)
            .map_err(|e| anyhow::anyhow!("Failed to build output array: {e}"));
    }

    let size = size.max(1);
    let left = size / 2;
    let right = size - left - 1;

    let mut tmp1 = vec![0.0_f32; n];
    let mut tmp2 = vec![0.0_f32; n];
    let mut tmp3 = vec![0.0_f32; n];
    let mut smoothed = vec![0.0_f32; n];

    for _ in 0..max_iter {
        // separable uniform filter with nearest-edge extension
        box_filter_rows_into(&filled, &mut tmp1, rows, cols, left, right);
        transpose_into(&tmp1, &mut tmp2, rows, cols);
        box_filter_rows_into(&tmp2, &mut tmp3, cols, rows, left, right);
        transpose_into(&tmp3, &mut smoothed, cols, rows);

        let mut max_diff = 0.0_f32;
        for &idx in &nan_indices {
            let old = filled[idx];
            let new = smoothed[idx];
            let d = (new - old).abs();
            if d > max_diff {
                max_diff = d;
            }
            filled[idx] = new;
        }

        if max_diff < tol {
            break;
        }
    }

    Array2::from_shape_vec((rows, cols), filled)
        .map_err(|e| anyhow::anyhow!("Failed to build output array: {e}"))
}
