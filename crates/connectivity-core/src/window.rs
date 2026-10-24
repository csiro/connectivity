use ndarray::ArrayView1;

/// Shape of the annulus that each level contributes around a focal cell. Cells cut by the
/// annulus boundary count with their covered area fraction in both modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowMode {
    /// Source-centred circular annuli (the Python default).
    Circular,
    /// Source-centred square annuli.
    Square,
}

impl WindowMode {
    /// Parse `"circular"` or `"square"`, ignoring case and surrounding whitespace. The error
    /// is the message the Python binding raises as `ValueError`.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "square" => Ok(Self::Square),
            "circular" => Ok(Self::Circular),
            other => Err(format!(
                "Invalid window_mode '{other}'. Expected 'square' or 'circular'."
            )),
        }
    }
}

#[inline]
pub(crate) fn square_frac(
    r0: f64,
    r1: f64,
    c0: f64,
    c1: f64,
    center_r: f64,
    center_c: f64,
    inner: f64,
    outer: f64,
) -> f32 {
    let footprint_area = (r1 - r0) * (c1 - c0);
    if footprint_area <= 0.0 {
        return 0.0;
    }

    let outer_area = square_overlap(r0, r1, c0, c1, center_r, center_c, outer);
    if outer_area <= 0.0 {
        return 0.0;
    }

    let inner_area = square_overlap(r0, r1, c0, c1, center_r, center_c, inner);
    let annulus_area = (outer_area - inner_area).max(0.0);
    (annulus_area / footprint_area).clamp(0.0, 1.0) as f32
}

#[inline]
fn square_overlap(
    r0: f64,
    r1: f64,
    c0: f64,
    c1: f64,
    center_r: f64,
    center_c: f64,
    half_extent: f64,
) -> f64 {
    if half_extent <= 0.0 {
        return 0.0;
    }

    let sr0 = center_r - half_extent;
    let sr1 = center_r + half_extent;
    let sc0 = center_c - half_extent;
    let sc1 = center_c + half_extent;

    let rows = r1.min(sr1) - r0.max(sr0);
    if rows <= 0.0 {
        return 0.0;
    }

    let cols = c1.min(sc1) - c0.max(sc0);
    if cols <= 0.0 {
        return 0.0;
    }

    rows * cols
}

#[inline]
pub(crate) fn circle_frac(
    r0: f64,
    r1: f64,
    c0: f64,
    c1: f64,
    center_r: f64,
    center_c: f64,
    inner: f64,
    outer: f64,
) -> (f32, bool) {
    let footprint_area = (r1 - r0) * (c1 - c0);
    if footprint_area <= 0.0 {
        return (0.0, false);
    }

    let outer_area = circle_overlap(r0, r1, c0, c1, center_r, center_c, outer);
    if outer_area <= 0.0 {
        return (0.0, false);
    }

    let inner_area = circle_overlap(r0, r1, c0, c1, center_r, center_c, inner);
    let annulus_area = (outer_area - inner_area).max(0.0);
    let fraction = (annulus_area / footprint_area).clamp(0.0, 1.0) as f32;
    let is_outer_edge =
        annulus_area > 0.0 && crosses_circle(r0, r1, c0, c1, center_r, center_c, outer);

    (fraction, is_outer_edge)
}

#[inline]
fn circle_overlap(
    r0: f64,
    r1: f64,
    c0: f64,
    c1: f64,
    center_r: f64,
    center_c: f64,
    radius: f64,
) -> f64 {
    let rect_area = (r1 - r0) * (c1 - c0);
    if rect_area <= 0.0 || radius <= 0.0 {
        return 0.0;
    }

    let x0 = c0 - center_c;
    let x1 = c1 - center_c;
    let y0 = r0 - center_r;
    let y1 = r1 - center_r;
    let radius_sq = radius * radius;

    if min_dist_sq(x0, x1, y0, y1) >= radius_sq {
        return 0.0;
    }
    if max_dist_sq(x0, x1, y0, y1) <= radius_sq {
        return rect_area;
    }

    let x_parts = split_at_zero(x0, x1);
    let y_parts = split_at_zero(y0, y1);
    let mut area = 0.0;

    for &(xa, xb) in &x_parts {
        if xb <= xa {
            continue;
        }
        let (qx0, qx1) = if xb <= 0.0 { (-xb, -xa) } else { (xa, xb) };

        for &(ya, yb) in &y_parts {
            if yb <= ya {
                continue;
            }
            let (qy0, qy1) = if yb <= 0.0 { (-yb, -ya) } else { (ya, yb) };
            area += q1_rect_area(qx0, qx1, qy0, qy1, radius);
        }
    }

    area.clamp(0.0, rect_area)
}

#[inline]
fn q1_rect_area(x0: f64, x1: f64, y0: f64, y1: f64, radius: f64) -> f64 {
    if x1 <= x0 || y1 <= y0 {
        return 0.0;
    }

    q1_corner_area(x1, y1, radius) - q1_corner_area(x0, y1, radius) - q1_corner_area(x1, y0, radius)
        + q1_corner_area(x0, y0, radius)
}

#[inline]
fn q1_corner_area(x: f64, y: f64, radius: f64) -> f64 {
    if x <= 0.0 || y <= 0.0 || radius <= 0.0 {
        return 0.0;
    }

    let x = x.min(radius);
    if x <= 0.0 {
        return 0.0;
    }

    if y >= radius {
        return arc_integral(x, radius);
    }

    let x_flat = (radius * radius - y * y).max(0.0).sqrt();
    let rect_width = x.min(x_flat);
    let mut area = rect_width * y;
    if x > x_flat {
        area += arc_integral(x, radius) - arc_integral(x_flat, radius);
    }

    area
}

#[inline]
fn arc_integral(x: f64, radius: f64) -> f64 {
    let x = x.clamp(0.0, radius);
    let y = (radius * radius - x * x).max(0.0).sqrt();
    let angle = (x / radius).clamp(-1.0, 1.0).asin();
    0.5 * (x * y + radius * radius * angle)
}

#[inline]
fn split_at_zero(a: f64, b: f64) -> [(f64, f64); 2] {
    if a < 0.0 && b > 0.0 {
        [(a, 0.0), (0.0, b)]
    } else {
        [(a, b), (0.0, 0.0)]
    }
}

#[inline]
fn crosses_circle(
    r0: f64,
    r1: f64,
    c0: f64,
    c1: f64,
    center_r: f64,
    center_c: f64,
    radius: f64,
) -> bool {
    if radius <= 0.0 {
        return false;
    }

    let x0 = c0 - center_c;
    let x1 = c1 - center_c;
    let y0 = r0 - center_r;
    let y1 = r1 - center_r;
    let radius_sq = radius * radius;
    let eps = (radius_sq.abs() + 1.0) * 1.0e-12;

    min_dist_sq(x0, x1, y0, y1) <= radius_sq + eps && max_dist_sq(x0, x1, y0, y1) >= radius_sq - eps
}

#[inline]
fn min_dist_sq(x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    let dx = if x0 > 0.0 {
        x0
    } else if x1 < 0.0 {
        -x1
    } else {
        0.0
    };
    let dy = if y0 > 0.0 {
        y0
    } else if y1 < 0.0 {
        -y1
    } else {
        0.0
    };
    dx * dx + dy * dy
}

#[inline]
fn max_dist_sq(x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
    let dx = x0.abs().max(x1.abs());
    let dy = y0.abs().max(y1.abs());
    dx * dx + dy * dy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_inside_disk() {
        let area = circle_overlap(9.0, 11.0, 9.0, 11.0, 10.0, 10.0, 3.0);
        assert!((area - 4.0).abs() < 1.0e-12);
    }

    #[test]
    fn quarter_disk() {
        let radius = 3.0;
        let area = circle_overlap(0.0, radius, 0.0, radius, 0.0, 0.0, radius);
        let expected = std::f64::consts::PI * radius * radius * 0.25;
        assert!((area - expected).abs() < 1.0e-12);
    }

    // Scenarios summed together match one-at-a-time similarities bit for bit
    #[test]
    fn sims_match_scalar() {
        let mut rng = crate::testkit::Rng::new(3);
        let value = |rng: &mut crate::testkit::Rng| match rng.below(40) {
            0 => f32::NAN,
            1 => f32::INFINITY,
            _ => (rng.unit() * 4.0 - 2.0) as f32,
        };
        for _ in 0..2000 {
            let bands = 1 + rng.below(25);
            let current: Vec<f32> = (0..bands).map(|_| value(&mut rng)).collect();
            let n_scen = 1 + rng.below(12);
            let cells: Vec<Vec<f32>> = (0..n_scen)
                .map(|_| {
                    let len = if rng.chance(0.1) { rng.below(30) } else { bands };
                    (0..len).map(|_| value(&mut rng)).collect()
                })
                .collect();
            let refs: Vec<&[f32]> = cells.iter().map(|c| c.as_slice()).collect();
            let mut out = Vec::new();
            for group in refs.chunks(8) {
                similarities(&current, group, f32::NAN, &mut out);
            }
            for (k, cell) in cells.iter().enumerate() {
                let expected = similarity(&ArrayView1::from(&current[..]), &ArrayView1::from(&cell[..]));
                assert_eq!(out[k].to_bits(), expected.unwrap_or(f32::NAN).to_bits());
            }
        }
    }

    #[test]
    fn flags_outer_rim_only() {
        let (_, outer_edge) = circle_frac(-0.5, 0.5, 2.5, 3.5, 0.0, 0.0, 1.0, 3.0);
        assert!(outer_edge);

        let (_, inner_edge) = circle_frac(-0.5, 0.5, 0.5, 1.5, 0.0, 0.0, 1.0, 3.0);
        assert!(!inner_edge);
    }
}

// Calcualte similarity of the transgrid layers and dealing with NaNs
// Any cell with nan coniditon is ignored in the step before this; now any nan transgrid
// will be processed as None and dealt with later in the metrics.rs
#[inline]
pub(crate) fn similarity(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Option<f32> {
    let mut l1: f32 = 0.0;

    // loop over each transform gird; return None even for one nan transgrid
    for (&x, &y) in a.iter().zip(b.iter()) {
        if x.is_finite() && y.is_finite() {
            l1 += (x - y).abs();
        } else {
            return None;
        }
    }

    Some((-l1).exp())
}

/// `similarity` of one cell for every scenario at once, pushed to `out` (`missing` for None).
/// Each scenario's L1 distance is summed in band order exactly as in `similarity`; advancing
/// the scenarios together only overlaps independent sums. A non-finite pair makes that
/// scenario's result missing, as the early return does.
#[inline]
pub(crate) fn similarities(current: &[f32], cells: &[&[f32]], missing: f32, out: &mut Vec<f32>) {
    const LANES: usize = 8;
    for group in cells.chunks(LANES) {
        let n = current.len().min(group[0].len());
        if group.iter().any(|cell| cell.len().min(current.len()) != n) {
            for cell in group {
                let sim = similarity(&ArrayView1::from(current), &ArrayView1::from(*cell));
                out.push(sim.unwrap_or(missing));
            }
            continue;
        }
        let mut l1 = [0.0f32; LANES];
        let mut finite = [true; LANES];
        for (b, &x) in current[..n].iter().enumerate() {
            for (k, cell) in group.iter().enumerate() {
                let y = cell[b];
                finite[k] &= x.is_finite() & y.is_finite();
                l1[k] += (x - y).abs();
            }
        }
        for k in 0..group.len() {
            out.push(if finite[k] { (-l1[k]).exp() } else { missing });
        }
    }
}
