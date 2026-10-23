//! Guarded distance tables for one `connectivity()` call.
//!
//! An edge's legacy distance is `distance_km(xy_a(cell), xy_b(target))` with the supplied level
//! transforms. Without skew, x depends only on the column and y only on the row, and every edge
//! of a slot has the same real-arithmetic displacement between the two cell centres:
//! - same-level slot k in 0..8 joins (i, j) to (i + COLS[k], j + ROWS[k]); the isolated
//!   source's synthetic edge (i, j) -> (i + 1, j) is slot 1;
//! - slot 8 + s, with s = `template::link_slot`, joins a level-L cell with parent-grid parity
//!   (pr, pc) to the level-2L cell (δr, δc) away from its parent. Its x displacement is
//!   sx_L * (2 * δc - pc + 0.5) + (A_2L - A_L), A_L = ox_L - sx_L * origin_L, when
//!   sx_2L = 2 * sx_L exactly (likewise y), using the transforms' actual anchoring.
//! Projected tables are keyed by (level, slot); geographic ones also by the source's local row
//! at that level, because latitude enters the haversine.
//!
//! Rounding still ties the legacy f32 to absolute coordinates, so a slot keeps a value only when
//! every edge it can serve provably gets exactly the legacy f32:
//! - exact: the coordinate arithmetic is exact (dyadic scales and origins in range), or a
//!   geographic same-column move, so every edge calls `distance_km` with identical inputs;
//! - bounded: over the slot's whole domain, the legacy f64 distance stays within a conservative
//!   bound of the representative value, and that interval rounds to a single f32. The bound
//!   covers coordinate rounding (unit roundoff per operation) and allows 16 ulps of hypot error
//!   or 2^-40 relative haversine error, far above platform libm accuracy.
//! Every other slot (skew, non-finite or zero terms, inconsistent level scales, an interval that
//! spans an f32 rounding boundary, long arcs, invalid latitudes, over budget) is evaluated per
//! edge with `Affine::xy` and `distance_km`. A table distance therefore never differs from the
//! legacy one, and integer costs cannot change.
use crate::affine::Affine;
use crate::distances::distance_km64;
use crate::graph::{COLS, ROWS};
use crate::template::link_slot;

/// Same-level slots then link slots
pub const SLOTS: usize = 8 + 36;
/// Slot of the synthetic edge (i, j) -> (i + 1, j)
pub const SYNTHETIC: usize = 1;

// Slot status: table hits, then fallback reasons
pub const EXACT: u8 = 0;
pub const BOUNDED: u8 = 1;
pub const SKEW: u8 = 2;
pub const NONFINITE: u8 = 3;
pub const LEVELS: u8 = 4;
pub const STRADDLE: u8 = 5;
pub const RANGE: u8 = 6;
pub const BUDGET: u8 = 7;
pub const STATUSES: usize = 8;
#[cfg(test)]
pub const STATUS_NAMES: [&str; STATUSES] = ["exact", "bounded", "skew", "nonfinite", "levels", "straddle", "range", "budget"];

const U: f64 = f64::EPSILON / 2.0; // unit roundoff
const HYPOT_ERR: f64 = 16.0 * f64::EPSILON; // 16 ulps, relative
const HAVERSINE_ERR: f64 = 1.0 / (1u64 << 40) as f64; // relative
const SLACK: f64 = 1.0 + 1.0 / (1u64 << 30) as f64; // covers rounding in the bound itself
const EARTH_RADIUS: f64 = 6_371_000.0; // as in `distances::haversine`, metres
const MAX_ARC_KM: f64 = 6000.0; // haversine a <= 1/4

/// Lowest set-bit exponent e of a finite nonzero v (v is an integer multiple of 2^e).
fn lsb_exp(v: f64) -> Option<i32> {
    if v == 0.0 || !v.is_finite() {
        return None;
    }
    let bits = v.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let mant = bits & ((1u64 << 52) - 1);
    let (m, e) = if exp == 0 {
        (mant, -1074)
    } else {
        (mant | (1u64 << 52), exp - 1075)
    };
    Some(e + m.trailing_zeros() as i32)
}

/// One axis of a level transform, `fl(fl(scale * (k + 0.5)) + origin)`, over cells
/// k in [lo, hi].
#[derive(Clone, Copy)]
struct Axis {
    scale: f64,
    origin: f64,
    lo: i32,
    hi: i32,
}

impl Axis {
    /// Bound M on |scale * (k + 0.5)| (1 + u) + |coordinate| over the range; the coordinate
    /// rounding error is at most u * M.
    fn magnitude(&self) -> f64 {
        let at = |k: i32| {
            let p = self.scale * (k as f64 + 0.5);
            p.abs() * (1.0 + U) + (p + self.origin).abs()
        };
        at(self.lo).max(at(self.hi)) * SLACK
    }

    /// Exponent q when every coordinate is computed exactly as a multiple of 2^q.
    fn quantum(&self) -> Option<i32> {
        let half = lsb_exp(self.scale)? - 1;
        let q = lsb_exp(self.origin).map_or(half, |e| e.min(half));
        for k in [self.lo, self.hi] {
            let p = self.scale * (k as f64 + 0.5);
            if p.abs() >= 2f64.powi(52 + half) || (p + self.origin).abs() >= 2f64.powi(51 + q) {
                return None;
            }
        }
        Some(q)
    }
}

/// Both coordinates of one level over its domain (with a two-cell margin).
#[derive(Clone, Copy)]
struct Frame {
    t: Affine,
    x: Axis,
    y: Axis,
}

impl Frame {
    fn new(t: Affine, height: i32, width: i32) -> Self {
        Self {
            t,
            x: Axis { scale: t.x_scale, origin: t.x_origin, lo: -2, hi: width + 1 },
            y: Axis { scale: t.y_scale, origin: t.y_origin, lo: -2, hi: height + 1 },
        }
    }
}

/// Status of a level transform for table use.
fn check(t: &Option<Affine>) -> Result<Affine, u8> {
    let t = t.ok_or(NONFINITE)?;
    if t.x_skew != 0.0 || t.y_skew != 0.0 {
        return Err(SKEW);
    }
    let terms = [t.x_scale, t.x_origin, t.y_scale, t.y_origin];
    if terms.iter().any(|v| !v.is_finite()) || t.x_scale == 0.0 || t.y_scale == 0.0 {
        return Err(NONFINITE);
    }
    Ok(t)
}

/// Whether the closed interval [d - h, d + h] rounds to a single f32.
fn single_f32(d: f64, h: f64) -> bool {
    let (lo, hi) = ((d - h).next_down(), (d + h).next_up());
    (lo as f32).to_bits() == (hi as f32).to_bits()
}

/// Table entry for edges displaced like `src` -> `dst` (cells of frames a and b; the same frame
/// for a same-level slot).
fn entry(a: &Frame, b: &Frame, same: bool, src: (i32, i32), dst: (i32, i32), geo: bool) -> (f32, u8) {
    let (x1, y1) = a.t.xy(src.0, src.1);
    let (x2, y2) = b.t.xy(dst.0, dst.1);
    let d = distance_km64(x1, y1, x2, y2, geo);
    if !d.is_finite() {
        return (f32::NAN, NONFINITE);
    }
    let value = d as f32;

    if geo {
        // Latitudes are fixed per key; the bound assumes valid latitudes and short arcs
        if y1.abs() > 90.0 || y2.abs() > 90.0 || d > MAX_ARC_KM {
            return (f32::NAN, RANGE);
        }
        // Same column of the same level: identical longitudes, hence identical inputs
        if same && src.1 == dst.1 {
            return (value, EXACT);
        }
        // Longitude difference error in radians, then haversine Lipschitz and evaluation bounds
        let rad = std::f64::consts::PI / 180.0;
        let phi = 3.01 * U * rad * (a.x.magnitude() + b.x.magnitude()) * SLACK;
        let lipschitz = std::f64::consts::SQRT_2 * EARTH_RADIUS * phi;
        let f_up = (d * 1000.0 + 2.0 * lipschitz) * SLACK;
        let bound = (HAVERSINE_ERR * f_up + lipschitz * (1.0 + HAVERSINE_ERR)) / 1000.0 * SLACK;
        return if bound.is_finite() && single_f32(d, 2.0 * bound) {
            (value, BOUNDED)
        } else {
            (f32::NAN, if bound.is_finite() { STRADDLE } else { RANGE })
        };
    }

    // Exact coordinate differences on an axis give identical hypot inputs on that axis
    let error = |p: &Axis, q: &Axis| -> f64 {
        if let (Some(qp), Some(qq)) = (p.quantum(), q.quantum()) {
            let common = qp.min(qq);
            let fits = |axis: &Axis| axis.magnitude() < 2f64.powi(51 + common);
            if fits(p) && fits(q) {
                return 0.0;
            }
        }
        2.0 * U * (p.magnitude() + q.magnitude()) * SLACK
    };
    let (ex, ey) = (error(&a.x, &b.x), error(&a.y, &b.y));
    if ex == 0.0 && ey == 0.0 {
        return (value, EXACT);
    }
    let d_up = (d * 1000.0 + ex + ey) * SLACK;
    let bound = ((ex + ey) + (HYPOT_ERR + 2.0 * U) * (d_up + ex + ey)) / 1000.0 * SLACK;
    if bound.is_finite() && single_f32(d, 2.0 * bound) {
        (value, BOUNDED)
    } else {
        (f32::NAN, if bound.is_finite() { STRADDLE } else { RANGE })
    }
}

/// Entries of one table row: same-level slots from cell (row, 0), and link slots from a cell of
/// each parent-grid parity (projected: every row parity; geographic: this row's).
fn level_row(
    a: &Frame,
    next: Option<&Result<Frame, u8>>,
    origins: [(i32, i32); 2],
    row: i32,
    geo: bool,
    dist: &mut [f32],
    status: &mut [u8],
) {
    for k in 0..8 {
        (dist[k], status[k]) = entry(a, a, true, (row, 0), (row + COLS[k], ROWS[k]), geo);
    }
    let Some(next) = next else { return };
    let parities = if geo { [(origins[0].0 + row).rem_euclid(2); 2] } else { [0, 1] };
    for pr in parities.into_iter().take(if geo { 1 } else { 2 }) {
        for pc in 0..2 {
            // Source cell with this global parity, and its parent in the next level's local grid
            let src = (
                if geo { row } else { (pr - origins[0].0).rem_euclid(2) },
                (pc - origins[0].1).rem_euclid(2),
            );
            let parent = (
                (origins[0].0 + src.0).div_euclid(2) - origins[1].0,
                (origins[0].1 + src.1).div_euclid(2) - origins[1].1,
            );
            for dr in -1..=1 {
                for dc in -1..=1 {
                    let slot = 8 + link_slot((pr, pc), (dr, dc)) as usize;
                    (dist[slot], status[slot]) = match next {
                        Ok(b) => entry(a, b, false, src, (parent.0 + dr, parent.1 + dc), geo),
                        Err(reason) => (f32::NAN, *reason),
                    };
                }
            }
        }
    }
}

/// Distance tables of every level; a NaN entry means per-edge legacy evaluation.
pub struct Tables {
    geo: bool,
    dist: Box<[Box<[f32]>]>, // [level][row * SLOTS + slot]; empty when not built
    status: Box<[Box<[u8]>]>,
    pub fixed: usize, // per-level slice headers, allocated whatever the budget
    pub bytes: usize, // retained heap bytes, from allocation sizes
}

impl Tables {
    /// `dims` are the (rows, columns) of each level's grid and `fringe` whether a level links to
    /// the next. Levels are tabulated from the smallest while the retained heap fits `budget`;
    /// building allocates nothing else.
    pub fn new(
        levels: &[i32],
        transforms: &[Option<Affine>],
        dims: &[(usize, usize)],
        offsets: (usize, usize),
        fringe: &[bool],
        geo: bool,
        budget: usize,
    ) -> Self {
        let n_levels = transforms.len();
        let fixed = n_levels * (std::mem::size_of::<Box<[f32]>>() + std::mem::size_of::<Box<[u8]>>());
        let mut tables = Self {
            geo,
            dist: vec![Box::default(); n_levels].into_boxed_slice(),
            status: vec![Box::default(); n_levels].into_boxed_slice(),
            fixed,
            bytes: fixed,
        };
        for l in 0..n_levels {
            let rows = if geo { dims[l].0 } else { 1 };
            let bytes = rows * SLOTS * (std::mem::size_of::<f32>() + std::mem::size_of::<u8>());
            if tables.bytes + bytes > budget {
                continue;
            }
            tables.bytes += bytes;
            let mut dist = vec![f32::NAN; rows * SLOTS].into_boxed_slice();
            let mut status = vec![BUDGET; rows * SLOTS].into_boxed_slice();
            let frame = |l: usize| {
                check(&transforms[l]).map(|t| Frame::new(t, dims[l].0 as i32, dims[l].1 as i32))
            };
            match frame(l) {
                Err(reason) => status.fill(reason),
                Ok(a) => {
                    // Links need the next level's frame and exactly doubled scales
                    let next = fringe[l].then(|| {
                        frame(l + 1).and_then(|b| {
                            let doubled = b.t.x_scale == 2.0 * a.t.x_scale
                                && b.t.y_scale == 2.0 * a.t.y_scale;
                            if doubled { Ok(b) } else { Err(LEVELS) }
                        })
                    });
                    let up = fringe[l].then(|| levels[l + 1]).unwrap_or(1);
                    let origins = [levels[l], up].map(|level| {
                        ((offsets.0 as i32).div_euclid(level), (offsets.1 as i32).div_euclid(level))
                    });
                    for row in 0..rows as i32 {
                        let slots = &mut dist[row as usize * SLOTS..(row as usize + 1) * SLOTS];
                        let states = &mut status[row as usize * SLOTS..(row as usize + 1) * SLOTS];
                        level_row(&a, next.as_ref(), origins, row, geo, slots, states);
                    }
                }
            }
            tables.dist[l] = dist;
            tables.status[l] = status;
        }
        tables
    }

    /// Distance and status of `slot` for a source in local row `row` of level `l`.
    #[inline]
    pub fn get(&self, l: usize, row: i32, slot: usize) -> (f32, u8) {
        let idx = if self.geo { row as usize * SLOTS + slot } else { slot };
        match self.dist[l].get(idx) {
            Some(&d) => (d, self.status[l][idx]),
            None => (f32::NAN, BUDGET),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distances::distance_km;
    use crate::template::CACHE_BUDGET;
    use crate::testkit::{level_transforms, Rng};

    #[test]
    fn lsb_and_quantum() {
        assert_eq!(lsb_exp(1.0), Some(0));
        assert_eq!(lsb_exp(0.75), Some(-2));
        assert_eq!(lsb_exp(250.0), Some(1));
        assert_eq!(lsb_exp(0.0), None);
        let axis = |scale, origin| Axis { scale, origin, lo: -2, hi: 10_000 };
        assert_eq!(axis(250.0, 500_000.0).quantum(), Some(0));
        assert_eq!(axis(1.0 / 120.0, 143.7).quantum(), None);
        assert_eq!(axis(0.3, 1.0e7 + 0.1).quantum(), None);
    }

    // Table hits equal the legacy f32 distance for edges sampled across large domains, with
    // extreme origins, unequal scales and high latitudes; every other slot falls back
    #[test]
    fn sampled_agreement() {
        let mut rng = Rng::new(11);
        let (mut hits, mut fallbacks, mut checked) = (0usize, 0usize, 0usize);
        for _ in 0..150 {
            let geo = rng.chance(0.5);
            let base = if geo {
                let res = rng.pick(&[0.000_25, 1.0 / 120.0, 0.05, 1.0 / 3600.0]);
                let lat = rng.pick(&[-85.0, -43.0, 0.5, 60.0, 89.0]);
                (res, 0.0, rng.pick(&[-179.99, 0.0, 143.7, 179.9]), 0.0, -res * rng.pick(&[1.0, 0.5]), lat)
            } else {
                let res = rng.pick(&[0.01, 0.3, 30.0, 92.6, 250.0, 1000.0]);
                let x0 = rng.pick(&[0.0, 1.0e7 + 0.1, -2.5e6, 123_456.789, 3.0e8]);
                (res * rng.pick(&[1.0, 1.7]), 0.0, x0, 0.0, -res, rng.pick(&[1.0e7, -4.4e6]))
            };
            let levels = [1, 2, 4];
            let offsets = (rng.below(5000), rng.below(5000));
            let transforms = level_transforms(base, &levels, offsets);
            let transforms: Vec<Option<Affine>> = levels.iter().map(|l| transforms.get(&(*l as i32)).copied()).collect();
            let (h, w) = (1 + rng.below(20_000), 1 + rng.below(20_000));
            let dims: Vec<(usize, usize)> = levels.iter().map(|&l| (h.div_ceil(l) + 1, w.div_ceil(l) + 1)).collect();
            let levels: Vec<i32> = levels.iter().map(|&l| l as i32).collect();
            let tables = Tables::new(&levels, &transforms, &dims, offsets, &[true, true, false], geo, CACHE_BUDGET);

            for l in 0..2 {
                let (a, b) = (transforms[l].unwrap(), transforms[l + 1].unwrap());
                let origin = |level: i32| ((offsets.0 as i32).div_euclid(level), (offsets.1 as i32).div_euclid(level));
                let (o, ou) = (origin(levels[l]), origin(levels[l + 1]));
                for _ in 0..400 {
                    // Any cell of the level grid, often at the domain edges
                    let pick = |rng: &mut Rng, n: usize| match rng.below(4) {
                        0 => 0,
                        1 => n as i32 - 1,
                        _ => rng.below(n) as i32,
                    };
                    let (i, j) = (pick(&mut rng, dims[l].0 - 1), pick(&mut rng, dims[l].1 - 1));
                    let k = rng.below(8);
                    let dst = (i + COLS[k], j + ROWS[k]);
                    let (x1, y1) = a.xy(i, j);
                    let (x2, y2) = a.xy(dst.0, dst.1);
                    let legacy = distance_km(x1, y1, x2, y2, geo);
                    let (d, status) = tables.get(l, i, k);
                    checked += 1;
                    if status <= BOUNDED {
                        hits += 1;
                        assert_eq!(d.to_bits(), legacy.to_bits(), "{base:?} level {l} cell ({i},{j}) slot {k}");
                    } else {
                        fallbacks += 1;
                    }
                    // A link from the same cell to each of its four next-level candidates
                    let g = (o.0 + i, o.1 + j);
                    let parity = (g.0.rem_euclid(2), g.1.rem_euclid(2));
                    let (dr, dc) = (rng.below(3) as i32 - 1, rng.below(3) as i32 - 1);
                    let up = (g.0.div_euclid(2) + dr - ou.0, g.1.div_euclid(2) + dc - ou.1);
                    let (x2, y2) = b.xy(up.0, up.1);
                    let legacy = distance_km(x1, y1, x2, y2, geo);
                    let (d, status) = tables.get(l, i, 8 + link_slot(parity, (dr, dc)) as usize);
                    checked += 1;
                    if status <= BOUNDED {
                        hits += 1;
                        assert_eq!(d.to_bits(), legacy.to_bits(), "{base:?} level {l} link from ({i},{j})");
                    } else {
                        fallbacks += 1;
                    }
                }
            }
        }
        println!("sampled {checked} edges: {hits} table hits, {fallbacks} fallbacks");
        assert!(hits > checked / 2);
    }
}
