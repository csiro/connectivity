use crate::affine::Affine;
use ndarray::parallel::prelude::*;
use ndarray::{s, Array2, Axis};

/// Compute fractional pixel coverage for a list of polygons.
///
/// Each polygon is `(exterior, holes)` with rings as `Vec<(f64, f64)>` of (x, y)
/// world-coordinate vertices. Polygons are assumed already-unioned on the Python side;
/// any residual overlap is clipped to 1.0.
///
/// # Returns
/// A 2D array of shape (nrows, ncols) with values in [0, 1].
pub fn pixel_coverage_array(
    polygons: &[(Vec<(f64, f64)>, Vec<Vec<(f64, f64)>>)],
    transform: &Affine,
    nrows: usize,
    ncols: usize,
) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((nrows, ncols));

    for (exterior, holes) in polygons {
        let ext_px = to_pixel(exterior, transform);
        accumulate_ring(&mut out, &ext_px, 1.0, nrows, ncols);
        for hole in holes {
            let hole_px = to_pixel(hole, transform);
            accumulate_ring(&mut out, &hole_px, -1.0, nrows, ncols);
        }
    }

    out.mapv_inplace(|v| v.clamp(0.0, 1.0));
    out
}

/// Convert a ring of (x, y) world coords to continuous (col, row) pixel coords.
fn to_pixel(ring: &[(f64, f64)], tr: &Affine) -> Vec<(f64, f64)> {
    ring.iter().map(|&(x, y)| tr.col_row(x, y)).collect()
}

/// Accumulate fractional coverage for a single ring into `out`. `sign` is +1 for
/// exterior rings and -1 for holes.
fn accumulate_ring(
    out: &mut Array2<f32>,
    ring: &[(f64, f64)],
    sign: f32,
    nrows: usize,
    ncols: usize,
) {
    if ring.len() < 3 {
        return;
    }

    // Bounding box of the ring in pixel coords (col=x, row=y after transform)
    let (mut xmin, mut ymin) = (f64::INFINITY, f64::INFINITY);
    let (mut xmax, mut ymax) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &(x, y) in ring {
        if x < xmin { xmin = x; }
        if x > xmax { xmax = x; }
        if y < ymin { ymin = y; }
        if y > ymax { ymax = y; }
    }

    // Clip bbox to raster extent
    let c0 = clamp_to_usize(xmin.floor(), ncols);
    let c1 = clamp_to_usize(xmax.ceil(), ncols);
    let r0 = clamp_to_usize(ymin.floor(), nrows);
    let r1 = clamp_to_usize(ymax.ceil(), nrows);
    if c0 >= c1 || r0 >= r1 {
        return;
    }

    // Parallel over rows in the bbox; each thread writes a disjoint row slice.
    out.slice_mut(s![r0..r1, c0..c1])
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(local_r, mut row)| {
            let r = r0 + local_r;
            let r_top = r as f64;
            let r_bot = r_top + 1.0;

            // Pre-clip ring to the row strip [r_top, r_bot] so per-pixel clips are cheaper.
            let strip = clip_half(ring, 1, r_top, true);
            let strip = clip_half(&strip, 1, r_bot, false);
            if strip.len() < 3 {
                return;
            }

            // Restrict cols to the strip's actual x-range, intersected with bbox cols.
            let mut sxmin = f64::INFINITY;
            let mut sxmax = f64::NEG_INFINITY;
            for &(x, _) in &strip {
                if x < sxmin { sxmin = x; }
                if x > sxmax { sxmax = x; }
            }
            let row_c0 = clamp_to_usize(sxmin.floor(), ncols).max(c0);
            let row_c1 = clamp_to_usize(sxmax.ceil(), ncols).min(c1);

            for c in row_c0..row_c1 {
                let x_left = c as f64;
                let x_right = x_left + 1.0;
                let cell = clip_half(&strip, 0, x_left, true);
                let cell = clip_half(&cell, 0, x_right, false);
                let area = shoelace_abs(&cell) as f32;
                if area > 0.0 {
                    row[c - c0] += sign * area;
                }
            }
        });
}

/// Sutherland-Hodgman clip against an axis-aligned half-plane.
/// `axis` is 0 for x and 1 for y. `keep_geq=true` keeps coord >= threshold;
/// otherwise keeps coord <= threshold.
fn clip_half(poly: &[(f64, f64)], axis: usize, threshold: f64, keep_geq: bool) -> Vec<(f64, f64)> {
    let n = poly.len();
    if n == 0 {
        return Vec::new();
    }

    let coord = |p: (f64, f64)| if axis == 0 { p.0 } else { p.1 };
    let inside = |p: (f64, f64)| {
        let v = coord(p);
        if keep_geq { v >= threshold } else { v <= threshold }
    };
    let intersect = |s: (f64, f64), p: (f64, f64)| -> (f64, f64) {
        let cs = coord(s);
        let cp = coord(p);
        let t = (threshold - cs) / (cp - cs);
        (s.0 + t * (p.0 - s.0), s.1 + t * (p.1 - s.1))
    };

    let mut out = Vec::with_capacity(n + 2);
    let mut s = poly[n - 1];
    let mut s_in = inside(s);
    for &p in poly {
        let p_in = inside(p);
        if p_in {
            if !s_in {
                out.push(intersect(s, p));
            }
            out.push(p);
        } else if s_in {
            out.push(intersect(s, p));
        }
        s = p;
        s_in = p_in;
    }
    out
}

/// Shoelace-formula area (orientation-independent).
fn shoelace_abs(poly: &[(f64, f64)]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let n = poly.len();
    let mut sum = 0.0;
    for i in 0..n {
        let (x1, y1) = poly[i];
        let (x2, y2) = poly[(i + 1) % n];
        sum += x1 * y2 - x2 * y1;
    }
    (sum * 0.5).abs()
}

/// Saturating cast from f64 to a usize clipped to [0, max].
fn clamp_to_usize(v: f64, max: usize) -> usize {
    if !v.is_finite() || v <= 0.0 {
        0
    } else if v >= max as f64 {
        max
    } else {
        v as usize
    }
}
