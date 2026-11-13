//! NoData crossing: the shared gap-width rule and the per-part values used by both modes.
//!
//! NoData cells in gaps at most `width` kilometres wide are crossable: they are traversable at a
//! set traversal value and carry no habitat area. The test is a morphological closing of the
//! land by a disc of diameter `width`, made once at base resolution with distances in
//! kilometres, so it does not depend on the overview grid. Mode 1 (`gate`) groups touching
//! traversable cells into landmasses; mode 2 (`strict`) splits coarse cells into pieces linked
//! only where their base cells touch.
//!
//! Both modes work on a frame that adds `offsets % max_level` empty rows and columns before the
//! tile, so every level's cells are aligned with the frame: level-`L` frame cell `(I, J)` covers
//! frame rows `I * L .. (I + 1) * L`, and its parent is `(I / 2, J / 2)`. The tile's level-`L`
//! cell `(i, j)` is frame cell `(i + pad.0 / L, j + pad.1 / L)`.
use crate::affine::Affine;
use ndarray::parallel::prelude::*;
use rayon::prelude::*;
use ndarray::{Array2, Array3};

/// Values of one part of a coarse cell: a mode 1 group part or a mode 2 piece.
#[derive(Clone, Debug)]
pub struct Part {
    pub cond: f32,  // mean condition of the part's land cells, NaN when it has none
    pub count: f32, // habitat area weight: land cells, or their cos-latitude areas
    pub trav: f32,  // mean finite traversal value of the part's cells, else its condition
    pub pa: f32,    // protected fraction of its land cells (PARC gating), else 1
    /// Mean of each scenario's bands over the part's land cells, scenario-major (BERI only)
    pub bands: Box<[f32]>,
}

impl Part {
    /// Condition and area weight of a node: crossable NoData alone has no habitat area.
    #[inline]
    pub fn habitat(&self) -> (f32, f32) {
        if self.count > 0.0 {
            (self.cond, self.count)
        } else {
            (0.0, 0.0)
        }
    }

    #[cfg(test)]
    pub fn bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.bands.len() * 4
    }
}

/// Running sums of one part, accumulated in f32 in row-major order as `overview` does, so a part
/// gets the values the overviews would give its cell if it held only that part.
#[derive(Clone, Default)]
pub struct Acc {
    cond: f32,
    land: f32,
    area: f32,
    trav: f32,
    cells: f32,
    pa: f32,
    bands: Vec<(f32, f32)>, // sum and finite count per scenario band
}

/// Base-resolution data of one call under a crossing mode, in frame coordinates.
pub struct Base<'a> {
    pub pad: (usize, usize),
    pub land: Array2<bool>,   // finite condition
    pub domain: Array2<bool>, // land or crossable NoData
    pub levels: Vec<i32>,     // ascending, from 1
    #[cfg(test)]
    pub crossable: usize,
    cond: &'a Array2<f32>, // tile
    trav: &'a Array2<f32>, // tile: traversal values, constant on crossable NoData, NaN off the domain
    areas: Option<Vec<f32>>, // cos-latitude area factor of each tile row (geographic grids)
    mask: Option<&'a Array2<bool>>, // PARC gating: `false` marks protected cells
    trans: &'a [Array3<f32>],
}

impl<'a> Base<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cross: &Array2<bool>,
        cond: &'a Array2<f32>,
        trav: &'a Array2<f32>,
        levels: &[i32],
        offsets: (usize, usize),
        areas: Option<Vec<f32>>,
        mask: Option<&'a Array2<bool>>,
        trans: &'a [Array3<f32>],
    ) -> Self {
        let top = *levels.last().unwrap() as usize;
        let pad = (offsets.0 % top, offsets.1 % top);
        let (h, w) = cond.dim();
        let dim = (h + pad.0, w + pad.1);
        let inside = |(r, c): (usize, usize)| (r >= pad.0 && c >= pad.1).then(|| (r - pad.0, c - pad.1));
        let land = Array2::from_shape_fn(dim, |rc| inside(rc).is_some_and(|ij| cond[ij].is_finite()));
        let domain = Array2::from_shape_fn(dim, |rc| inside(rc).is_some_and(|ij| cond[ij].is_finite() || cross[ij]));
        Self {
            pad,
            land,
            domain,
            levels: levels.to_vec(),
            #[cfg(test)]
            crossable: cross.iter().filter(|&&c| c).count(),
            cond,
            trav,
            areas,
            mask,
            trans,
        }
    }

    /// Frame offset of the tile's cells at level index `l`.
    #[inline]
    pub fn shift(&self, l: usize) -> (usize, usize) {
        let size = self.levels[l] as usize;
        (self.pad.0 / size, self.pad.1 / size)
    }

    /// Rows `r0..r1` and columns `c0..c1` of frame cell (i, j) at level size `size`.
    #[inline]
    pub fn footprint(&self, i: usize, j: usize, size: usize) -> (usize, usize, usize, usize) {
        let (h, w) = self.domain.dim();
        (i * size, ((i + 1) * size).min(h), j * size, ((j + 1) * size).min(w))
    }

    fn bands(&self) -> usize {
        self.trans.first().map_or(0, |t| t.dim().2) * self.trans.len()
    }

    /// Adds traversable frame cell (r, c) to `acc`.
    #[inline]
    pub fn add(&self, acc: &mut Acc, r: usize, c: usize) {
        let ij = (r - self.pad.0, c - self.pad.1);
        let t = self.trav[ij];
        if t.is_finite() {
            acc.trav += t;
            acc.cells += 1.0;
        }
        if !self.land[[r, c]] {
            return;
        }
        acc.cond += self.cond[ij];
        acc.land += 1.0;
        acc.area += self.areas.as_ref().map_or(1.0, |a| a[ij.0]);
        if let Some(mask) = self.mask {
            acc.pa += if mask[ij] { 0.0 } else { 1.0 };
        }
        if !self.trans.is_empty() {
            let n = self.trans[0].dim().2;
            if acc.bands.is_empty() {
                acc.bands = vec![(0.0, 0.0); self.bands()];
            }
            for (s, grid) in self.trans.iter().enumerate() {
                for b in 0..n {
                    let v = grid[[ij.0, ij.1, b]];
                    if v.is_finite() {
                        let slot = &mut acc.bands[s * n + b];
                        slot.0 += v;
                        slot.1 += 1.0;
                    }
                }
            }
        }
    }

    pub fn finish(&self, acc: &Acc) -> Part {
        let land = acc.land > 0.0;
        let mut bands = vec![f32::NAN; if land { self.bands() } else { 0 }];
        for (out, &(sum, n)) in bands.iter_mut().zip(&acc.bands) {
            if n > 0.0 {
                *out = sum / n;
            }
        }
        Part {
            cond: if land { acc.cond / acc.land } else { f32::NAN },
            count: acc.area,
            // Mean of the finite traversal values, else condition
            trav: if acc.cells > 0.0 { acc.trav / acc.cells } else { acc.cond / acc.land },
            pa: match self.mask {
                Some(_) if land => acc.pa / acc.land,
                Some(_) => 0.0,
                None => 1.0,
            },
            bands: bands.into_boxed_slice(),
        }
    }
}

/// Cell sizes in kilometres: height, and width of each row (it shrinks with latitude on
/// geographic grids).
pub fn cell_km(base: &Affine, rows: usize, is_geo: bool) -> (f64, Vec<f64>) {
    if is_geo {
        // Arc lengths on the 6371 km sphere of `distances::haversine`
        let km = 6371.0 * std::f64::consts::PI / 180.0;
        let widths = (0..rows)
            .map(|r| base.x_scale.abs() * km * base.xy(r as i32, 0).1.to_radians().cos().abs())
            .collect();
        (base.y_scale.abs() * km, widths)
    } else {
        (base.y_scale.abs() / 1000.0, vec![base.x_scale.abs() / 1000.0; rows])
    }
}

/// Squared distance in km² from point (i + oy / 2, j + ox / 2) of every cell (i, j) to the
/// nearest site, with rows `dy` apart and the cells of row i `dx[i]` apart; `oy` and `ox` are in
/// half cells, -1, 0 or 1. Sites are cell centres, or with `squares` whole cells (squares of
/// side one cell), for `oy` and `ox` 0 or 1. Exact separable transform: a column pass, then the
/// lower envelope of parabolas along each row; a square is a parabola with a flat bottom, which
/// three parabolas (its centre and edges) represent exactly at half-cell query points.
fn edt_sq(sites: &Array2<bool>, dy: f64, dx: &[f64], oy: i32, ox: i32, squares: bool) -> Array2<f32> {
    let (h, w) = sites.dim();
    let sites = sites.as_standard_layout();
    let sites = sites.as_slice().unwrap();
    const FAR: i32 = 1 << 20;
    // Rows up to the nearest site at or above each cell, then twice the vertical distance from
    // the query point: to the site's centre, or to its nearest edge for squares
    let mut g = vec![FAR; h * w];
    for i in 0..h {
        let (above, row) = g.split_at_mut(i * w);
        let row = &mut row[..w];
        let prev = if i > 0 { Some(&above[(i - 1) * w..]) } else { None };
        for j in 0..w {
            row[j] = if sites[i * w + j] {
                0
            } else {
                prev.map_or(FAR, |p| (p[j] + 1).min(FAR))
            };
        }
    }
    let mut below = vec![FAR; w]; // rows down to the nearest site at or below row i + 1
    let mut down = vec![FAR; w];
    let mut above = vec![FAR; w]; // rows up from row i - 1, before it is overwritten
    for i in (0..h).rev() {
        if i > 0 {
            above.copy_from_slice(&g[(i - 1) * w..i * w]);
        } else {
            above.fill(FAR);
        }
        let row = &mut g[i * w..(i + 1) * w];
        for j in 0..w {
            down[j] = if sites[i * w + j] { 0 } else { (below[j] + 1).min(FAR) };
            let up = row[j];
            let twice = match oy {
                0 => 2 * up.min(down[j]),
                1 => (2 * up + 1).min(2 * below[j] + 1),
                _ => (2 * above[j] + 1).min(2 * down[j] + 1),
            };
            row[j] = if squares { (twice - 1).max(0) } else { twice };
        }
        std::mem::swap(&mut below, &mut down);
    }
    let mut out = Array2::<f32>::zeros((h, w));
    let out_rows = out.as_slice_mut().unwrap().par_chunks_mut(w.max(1));
    out_rows.zip(g.par_chunks(w.max(1))).enumerate().for_each(|(i, (row, grow))| {
        let s = dx[i];
        let vert = |j: usize| (grow[j] as f64 * 0.5 * dy).powi(2);
        // Sites: positions (in cells) and squared vertical distances
        let (pos, f): (Vec<f64>, Vec<f64>) = if squares {
            (0..=2 * w)
                .map(|k| {
                    let v = if k % 2 == 1 {
                        vert(k / 2)
                    } else {
                        let left = if k >= 2 { vert(k / 2 - 1) } else { f64::INFINITY };
                        let right = if k / 2 < w { vert(k / 2) } else { f64::INFINITY };
                        left.min(right)
                    };
                    ((k as f64 - 1.0) * 0.5 * s, v)
                })
                .unzip()
        } else {
            (0..w).map(|j| (j as f64 * s, vert(j))).unzip()
        };
        let n = pos.len();
        let mut v = vec![0usize; n];
        let mut z = vec![0f64; n + 1];
        let mut k = 0usize;
        z[0] = f64::NEG_INFINITY;
        z[1] = f64::INFINITY;
        for q in 1..n {
            loop {
                let p = v[k];
                let cut = ((f[q] + pos[q] * pos[q]) - (f[p] + pos[p] * pos[p])) / (2.0 * (pos[q] - pos[p]));
                if cut <= z[k] {
                    k -= 1;
                } else {
                    k += 1;
                    v[k] = q;
                    z[k] = cut;
                    z[k + 1] = f64::INFINITY;
                    break;
                }
            }
        }
        k = 0;
        for (q, out) in row.iter_mut().enumerate() {
            let at = (q as f64 + 0.5 * ox as f64) * s;
            while z[k + 1] < at {
                k += 1;
            }
            let p = v[k];
            *out = ((at - pos[p]).powi(2) + f[p]) as f32;
        }
    });
    out
}

/// NoData cells lying in gaps at most `width` km wide: the closing of the land cells (as
/// squares) by a disc of diameter `width`, less the land itself. `dy` and `dx` are cell sizes
/// from `cell_km`.
///
/// A disc fits where its centre is more than `width / 2` from every land cell. Centres are tried
/// on the half-cell lattice (cell centres, edge midpoints and corners), and a NoData cell is
/// crossable when no fitting disc covers its centre. This is exact for straight channels along
/// rows or columns of any width, and close for other shapes.
pub fn crossable(land: &Array2<bool>, width: f64, dy: f64, dx: &[f64]) -> Array2<bool> {
    let dim = land.dim();
    if width <= 0.0 || land.is_empty() {
        return Array2::from_elem(dim, false);
    }
    // A tolerance of 0.001 squared cells keeps channels exactly `width` wide crossable
    let unit = dx.iter().copied().fold(dy, f64::min);
    let r2 = ((width * 0.5).powi(2) + 1.0e-3 * unit * unit) as f32;
    let mut covered = Array2::from_elem(dim, false);
    for (qy, qx) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
        let open = edt_sq(land, dy, dx, qy, qx, true).mapv(|d| d > r2);
        if open.iter().any(|&o| o) {
            covered.zip_mut_with(&edt_sq(&open, dy, dx, -qy, -qx, false), |c, &d| *c |= d <= r2);
        }
    }
    Array2::from_shape_fn(dim, |ij| !land[ij] && !covered[ij])
}

#[cfg(test)]
mod tests {
    use super::*;

    // A straight channel is crossable up to the width and not beyond it, along either axis
    #[test]
    fn channel_width() {
        let res = 0.25; // km
        for gap in 1..=10usize {
            for rows in [true, false] {
                let land = Array2::from_shape_fn((80, 60), |(i, j)| {
                    let k = if rows { i } else { j };
                    k < 30 || k >= 30 + gap
                });
                for cells in [0.5, 3.0, 4.0, 5.5, 8.0] {
                    let cross = crossable(&land, cells * res, res, &[res; 80]);
                    let n = cross.iter().filter(|&&c| c).count();
                    let side = if rows { 60 } else { 80 };
                    let expected = if gap as f64 <= cells { gap * side } else { 0 };
                    assert_eq!(n, expected, "gap {gap}, width {cells} cells, rows {rows}");
                }
            }
        }
    }

    // The transform matches brute force for point and square sites at every half-cell offset
    #[test]
    fn edt_brute() {
        let mut rng = crate::testkit::Rng::new(3);
        let (h, w) = (13, 17);
        let (dy, dx): (f64, Vec<f64>) = (0.3, (0..h).map(|i| 0.2 + 0.01 * i as f64).collect());
        for density in [0.02, 0.2] {
            let sites = Array2::from_shape_fn((h, w), |_| rng.chance(density));
            for (oy, ox, squares) in [(0, 0, true), (1, 0, true), (0, 1, true), (1, 1, true), (-1, 0, false), (0, -1, false), (1, 1, false), (0, 0, false)] {
                let fast = edt_sq(&sites, dy, &dx, oy, ox, squares);
                for ((i, j), &d) in fast.indexed_iter() {
                    let (y, x) = (i as f64 + 0.5 * oy as f64, j as f64 + 0.5 * ox as f64);
                    let mut best = f64::INFINITY;
                    for ((r, c), &site) in sites.indexed_iter() {
                        if site {
                            let gap = |t: f64| if squares { (t.abs() - 0.5).max(0.0) } else { t.abs() };
                            best = best.min((gap(y - r as f64) * dy).powi(2) + (gap(x - c as f64) * dx[i]).powi(2));
                        }
                    }
                    if best.is_finite() {
                        assert!((d as f64 - best).abs() <= 1.0e-5 * best.max(1.0), "{oy} {ox} {squares} ({i}, {j}): {d} vs {best}");
                    }
                }
            }
        }
    }

    // On a geographic grid a column gap of fixed degrees narrows in km towards the pole
    #[test]
    fn geo_width() {
        let base = Affine::from((0.01, 0.0, 140.0, 0.0, -0.01, -40.0));
        let (dy, dx) = cell_km(&base, 4000, true);
        assert!((dy - 1.1119).abs() < 1.0e-3);
        assert!((dx[0] - 1.1119 * (40.005f64).to_radians().cos()).abs() < 1.0e-3);
        // Three columns of NoData from 40 S to 80 S: 2.6 km wide at 40 S, 1.7 km at 60 S
        let land = Array2::from_shape_fn((4000, 20), |(_, j)| !(8..11).contains(&j));
        let cross = crossable(&land, 2.0, dy, &dx);
        let lat = |r: usize| base.xy(r as i32, 0).1;
        for r in [100, 2000, 3000, 3999] {
            let width = 3.0 * dx[r];
            assert_eq!(cross[[r, 9]], width <= 2.0, "row {r} at {:.1}, width {width:.2} km", lat(r));
        }
    }
}
