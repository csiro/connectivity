//! BERI climate of one call: every scenario's bands at every level, and node similarities.
//!
//! Storage: the base level of each scenario is the caller's array, borrowed when it is in standard
//! layout (each cell's bands contiguous) and otherwise copied once into it. Coarse levels hold the
//! band means of all scenarios together, `[cell][band][scenario]`, so one band of eight scenarios
//! is one contiguous load. The means are written straight into that layout, so no other copy of a
//! level exists while it is built.
//!
//! Similarities follow `window::similarity` exactly: per scenario, |current - cell| summed in f32 in
//! band order, `exp(-sum)` with scalar `exp`, and `NO_SIM` when any band pair is not finite.
//! Scenarios are advanced eight at a time, each in its own lane, which only overlaps independent
//! sums. A coarse level's array ends with `LANES` spare values so the last scenario group of the
//! last cell can always load eight values; lanes beyond the scenario count are never pushed.
//!
//! Finiteness is known before the sums: every cell keeps one bit per scenario, set when all its
//! bands are finite (one byte per group of eight scenarios), and the focal cell's bands are checked
//! once. A pair of band vectors is all finite exactly when both are, so a similarity is `NO_SIM`
//! exactly as before; the sum of a lane whose bit is clear is computed and dropped. The bits
//! describe the input values, not the sum: finite values whose sum overflows still give
//! `exp(-inf) = 0`, as before.
use crate::graph::NO_SIM;
use crate::overview::{self, Resampling};
use crate::window::similarities;
use anyhow::Result;
use ndarray::{Array2, Array3, CowArray, Ix3};
use rayon::prelude::*;
use std::mem::size_of;

/// Scenarios advanced together
const LANES: usize = 8;

pub struct Climate<'a> {
    base: Vec<CowArray<'a, f32, Ix3>>, // per scenario, standard layout
    coarse: Vec<Box<[f32]>>,           // per level index (empty at level 1): [cell][band][scenario]
    widths: Vec<usize>,                // columns of each level's grid
    finite: Vec<Box<[u8]>>,            // per level: [cell][scenario group], bit k: all bands finite
    scenarios: usize,
    groups: usize, // scenario groups of `LANES`
    bands: usize,
    /// Retained heap: base levels copied into standard layout, coarse means and descriptors.
    /// Building allocates nothing else, so this is also the construction peak.
    pub bytes: usize,
}

impl<'a> Climate<'a> {
    /// Climate of `levels` (ascending, from 1) with grids of `dims`; with `land`, coarse means
    /// use only cells where it is finite (the crossing modes' land climate).
    pub fn new(
        trans: &'a [Array3<f32>],
        levels: &[i32],
        dims: &[(usize, usize)],
        offsets: (usize, usize),
        land: Option<&Array2<f32>>,
    ) -> Result<Self> {
        let base: Vec<CowArray<f32, Ix3>> = trans.iter().map(|t| t.as_standard_layout()).collect();
        let (scenarios, bands) = (base.len(), base[0].dim().2);
        let mut bytes: usize = base.iter().filter(|b| b.is_owned()).map(|b| b.len() * 4).sum();
        let mut coarse = Vec::with_capacity(levels.len());
        coarse.push(Box::default());
        for (l, &level) in levels.iter().enumerate().skip(1) {
            let cells = dims[l].0 * dims[l].1;
            let mut means = vec![f32::NAN; cells * bands * scenarios + LANES].into_boxed_slice();
            for (s, grid) in base.iter().enumerate() {
                let keep = |r: usize, c: usize| land.is_none_or(|land| land[[r, c]].is_finite());
                overview::coarse_bands_into(grid.view(), level as usize, offsets, Resampling::Average, keep, &mut means, scenarios, s)?;
            }
            bytes += means.len() * 4;
            coarse.push(means);
        }
        let widths: Vec<usize> = dims.iter().map(|d| d.1).collect();
        let groups = scenarios.div_ceil(LANES);
        let mut finite = Vec::with_capacity(levels.len());
        for (l, &(h, w)) in dims.iter().enumerate() {
            let flags = if l == 0 {
                let slices: Vec<&[f32]> = base.iter().map(|b| b.as_slice().expect("standard layout")).collect();
                finite_bits(h * w, scenarios, |cell, s| slices[s][cell * bands..(cell + 1) * bands].iter().all(|v| v.is_finite()))
            } else {
                let means = &coarse[l];
                let at = |cell: usize, b: usize, s: usize| means[(cell * bands + b) * scenarios + s];
                finite_bits(h * w, scenarios, |cell, s| (0..bands).all(|b| at(cell, b, s).is_finite()))
            };
            bytes += flags.len();
            finite.push(flags);
        }
        bytes += base.capacity() * size_of::<CowArray<f32, Ix3>>()
            + coarse.capacity() * size_of::<Box<[f32]>>()
            + widths.capacity() * size_of::<usize>()
            + finite.capacity() * size_of::<Box<[u8]>>();
        Ok(Self { base, coarse, widths, finite, scenarios, groups, bands, bytes })
    }

    pub fn scenarios(&self) -> usize {
        self.scenarios
    }

    /// The current climate's bands (scenario 0) of base cell (i, j), and whether all are finite
    #[inline]
    pub fn current(&self, i: usize, j: usize) -> (&[f32], bool) {
        let bands = self.cell(0, i, j);
        (bands, bands.iter().all(|v| v.is_finite()))
    }

    #[inline]
    fn cell(&self, s: usize, i: usize, j: usize) -> &[f32] {
        let start = (i * self.widths[0] + j) * self.bands;
        &self.base[s].as_slice().expect("standard layout")[start..start + self.bands]
    }

    /// Similarity of `current` (`finite`: all its bands are) with cell `at` of level index l, for
    /// every scenario in order, pushed to `out`.
    #[inline]
    pub fn sims(&self, current: &[f32], finite: bool, l: usize, at: (usize, usize), out: &mut Vec<f32>) {
        let n = self.scenarios;
        if !finite {
            out.extend(std::iter::repeat_n(NO_SIM, n));
            return;
        }
        let cell = at.0 * self.widths[l] + at.1;
        let flags = &self.finite[l][cell * self.groups..(cell + 1) * self.groups];
        for (g, &bits) in flags.iter().enumerate() {
            let lanes = LANES.min(n - g * LANES);
            let mut l1 = [0.0f32; LANES];
            if l == 0 {
                // Base cells: each scenario's bands from its own array
                let mut cells: [&[f32]; LANES] = [&[]; LANES];
                for (k, cell) in cells.iter_mut().take(lanes).enumerate() {
                    *cell = self.cell(g * LANES + k, at.0, at.1);
                }
                for (b, &x) in current.iter().enumerate() {
                    for (sum, cell) in l1.iter_mut().zip(&cells[..lanes]) {
                        *sum += (x - cell[b]).abs();
                    }
                }
            } else {
                // Coarse cells: one band of eight scenarios is one load
                let block = &self.coarse[l][cell * self.bands * n..(cell + 1) * self.bands * n + LANES];
                for (b, &x) in current.iter().enumerate() {
                    let y: &[f32; LANES] = block[b * n + g * LANES..b * n + g * LANES + LANES].try_into().unwrap();
                    for k in 0..LANES {
                        l1[k] += (x - y[k]).abs();
                    }
                }
            }
            for (k, &sum) in l1[..lanes].iter().enumerate() {
                out.push(if bits >> k & 1 == 1 { (-sum).exp() } else { NO_SIM });
            }
        }
    }
}

/// One byte per group of `LANES` scenarios for each of `cells` cells, bit k set when `finite(cell,
/// scenario)` holds for scenario `group * LANES + k`.
fn finite_bits(cells: usize, scenarios: usize, finite: impl Fn(usize, usize) -> bool + Sync) -> Box<[u8]> {
    let groups = scenarios.div_ceil(LANES);
    let mut bits = vec![0u8; cells * groups].into_boxed_slice();
    bits.par_chunks_mut(groups).enumerate().for_each(|(cell, out)| {
        for s in 0..scenarios {
            out[s / LANES] |= (finite(cell, s) as u8) << (s % LANES);
        }
    });
    bits
}

/// Similarity of `current` with the bands of a part (`crossing::Part::bands`, scenario-major,
/// empty for a part without land), for every one of `scenarios`, pushed to `out`.
#[inline]
pub fn part_sims(current: &[f32], bands: &[f32], scenarios: usize, out: &mut Vec<f32>) {
    if bands.is_empty() {
        out.extend(std::iter::repeat_n(NO_SIM, scenarios));
        return;
    }
    let n = bands.len() / scenarios;
    for group in bands.chunks(n * LANES) {
        let mut cells: [&[f32]; LANES] = [&[]; LANES];
        let m = group.len() / n;
        for (k, cell) in cells.iter_mut().take(m).enumerate() {
            *cell = &group[k * n..(k + 1) * n];
        }
        similarities(current, &cells[..m], NO_SIM, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::Rng;
    use crate::window::similarity;
    use ndarray::ArrayView1;

    // Check both the packed means and their similarities with [0, 0]. Expectations come from
    // the fixtures below, not from an aggregation or similarity helper.
    fn assert_coarse_cell(climate: &Climate<'_>, l: usize, at: (usize, usize), expected: &[[f32; 2]]) {
        let n = expected.len();
        assert_eq!(climate.scenarios(), n);
        let start = (at.0 * climate.widths[l] + at.1) * 2 * n;
        let means = &climate.coarse[l][start..start + 2 * n];
        let mut sims = Vec::new();
        climate.sims(&[0.0, 0.0], true, l, at, &mut sims);
        assert_eq!(sims.len(), n);
        for (s, bands) in expected.iter().enumerate() {
            for (b, &want) in bands.iter().enumerate() {
                let got = means[b * n + s];
                if want.is_nan() {
                    assert!(got.is_nan(), "level index {l}, {at:?}, scenario {s}, band {b}: {got}");
                } else {
                    assert_eq!(got.to_bits(), want.to_bits(), "level index {l}, {at:?}, scenario {s}, band {b}");
                }
            }
            if bands.iter().all(|v| v.is_finite()) {
                let want = (-(bands[0].abs() + bands[1].abs())).exp();
                assert_eq!(sims[s].to_bits(), want.to_bits(), "level index {l}, {at:?}, scenario {s}");
            } else {
                assert!(sims[s].is_nan(), "level index {l}, {at:?}, scenario {s}: missing band");
            }
        }
    }

    #[test]
    fn aggregation_offsets_and_missing_bands() {
        let nan = f32::NAN;
        let trans = [Array3::from_shape_vec((3, 3, 2), vec![
            2.0, nan, 4.0, 20.0, 8.0, 28.0,
            6.0, 8.0, nan, f32::INFINITY, 10.0, f32::NEG_INFINITY,
            14.0, 16.0, 18.0, nan, 22.0, 44.0,
        ]).unwrap()];
        let land = ndarray::array![[0.5, nan, 0.5], [nan, 0.5, nan], [nan, nan, 0.5]];
        // Global rows 1..4, columns 3..6. At level 2 the local row/column groups are
        // [0], [1, 2]; at level 4 all rows merge, while the column groups stay separate.
        let dims = [(3, 3), (2, 2), (1, 2)];
        // Legacy aggregation uses every finite climate value, independently per band:
        // bottom-right level-2 sums/counts are 50/3 and 44/1; level-4 right is 62/5, 92/3.
        // Land-only aggregation keeps (0,0), (0,2), (1,1), (2,2). The bottom-left cell
        // has no land; (1,1) is land but has no finite climate bands.
        for (keep, level2, level4) in [
            (None, [[2.0, nan], [6.0, 24.0], [10.0, 12.0], [50.0 / 3.0, 44.0]],
                [[22.0 / 3.0, 12.0], [62.0 / 5.0, 92.0 / 3.0]]),
            (Some(&land), [[2.0, nan], [8.0, 28.0], [nan, nan], [22.0, 44.0]],
                [[2.0, nan], [15.0, 36.0]]),
        ] {
            let climate = Climate::new(&trans, &[1, 2, 4], &dims, (1, 3), keep).unwrap();
            // Land filtering applies only to coarse means: this off-land base cell stays intact.
            assert_eq!(climate.cell(0, 1, 0), &[6.0, 8.0]);
            for (l, cells) in [(1, &level2[..]), (2, &level4[..])] {
                for (cell, &bands) in cells.iter().enumerate() {
                    assert_coarse_cell(&climate, l, (cell / 2, cell % 2), &[bands]);
                }
            }
        }
    }

    #[test]
    fn aggregation_packed_scenario_boundaries() {
        for scenarios in [1, 7, 8, 9, 16, 17] {
            let trans: Vec<Array3<f32>> = (0..scenarios).map(|s| {
                let x = s as f32;
                let missing = s % 3 == 1;
                Array3::from_shape_vec((1, 2, 2), vec![
                    x / 8.0, if missing { f32::NAN } else { x / 16.0 },
                    (x + 2.0) / 8.0, if missing { f32::NAN } else { (x + 4.0) / 16.0 },
                ]).unwrap()
            }).collect();
            // Offset (1,1) splits the two base cells at level 2; level 4 averages them.
            // Distinct scenario/band values catch transposition and lane-order mistakes. Missing
            // bands vary across both scenario groups. The final cell exercises the spare lanes.
            let climate = Climate::new(&trans, &[1, 2, 4], &[(1, 2), (1, 2), (1, 1)], (1, 1), None).unwrap();
            // (level index, column, first-band numerator offset, second-band numerator offset)
            for (l, col, dx, dy) in [(1, 0, 0, 0), (1, 1, 2, 4), (2, 0, 1, 2)] {
                let expected: Vec<[f32; 2]> = (0..scenarios).map(|s| [
                    (s + dx) as f32 / 8.0,
                    if s % 3 == 1 { f32::NAN } else { (s + dy) as f32 / 16.0 },
                ]).collect();
                assert_coarse_cell(&climate, l, (0, col), &expected);
            }
        }
    }

    // The reported bytes are the heap the climate retains, and the peak while it is built, as the
    // allocator counts them: standard-layout input is borrowed, other layouts copied once
    #[test]
    fn bytes_measured() {
        use crate::testkit::alloc;
        use rayon::prelude::*;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        pool.install(|| {
            (0..4096).into_par_iter().for_each(|k| _ = std::hint::black_box(k));
            for band_major in [false, true] {
                let trans: Vec<Array3<f32>> = (0..3)
                    .map(|s| {
                        let a = Array3::from_shape_fn((4, 23, 31), |(b, r, c)| (r * 7 + c * 3 + b + s) as f32);
                        let a = a.permuted_axes([1, 2, 0]); // band-major
                        if band_major { a } else { a.as_standard_layout().into_owned() }
                    })
                    .collect();
                let levels = [1, 2, 4, 8];
                let dims: Vec<(usize, usize)> = levels.iter().map(|&l| overview::level_dim(23, 31, l as usize, (3, 5))).collect();
                let before = alloc::live();
                alloc::reset_peak();
                let climate = Climate::new(&trans, &levels, &dims, (3, 5), None).unwrap();
                let (retained, peak) = ((alloc::live() - before) as usize, (alloc::peak() - before) as usize);
                assert_eq!(retained, climate.bytes, "band-major {band_major}");
                assert_eq!(peak, climate.bytes, "band-major {band_major}");
                let copies = if band_major { 3 * 4 * 23 * 31 * 4 } else { 0 };
                assert!(climate.bytes >= copies && climate.bytes < copies + 3 * 4 * 23 * 31 * 4 / 2 + 1024);
            }
        });
    }

    // Base and packed coarse similarities, with their finite bits, equal the scalar `similarity` of
    // each scenario's own values or means, for 1 to 17 scenarios, 1 to 19 bands, non-finite,
    // extreme and signed-zero values
    #[test]
    fn packed_matches_scalar() {
        let mut rng = Rng::new(17);
        let value = |rng: &mut Rng| match rng.below(40) {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => -0.0,
            5 => f32::NEG_INFINITY,
            3 => 3.0e38,
            4 => -3.0e38,
            _ => rng.unit() as f32,
        };
        for scenarios in [1, 2, 3, 7, 8, 9, 16, 17] {
            for bands in [1, 3, 19] {
                let (h, w) = (6, 5);
                let trans: Vec<Array3<f32>> =
                    (0..scenarios).map(|_| Array3::from_shape_fn((h, w, bands), |_| value(&mut rng))).collect();
                let levels = [1, 2, 4];
                let dims: Vec<(usize, usize)> = levels.iter().map(|&l| overview::level_dim(h, w, l as usize, (1, 2))).collect();
                let climate = Climate::new(&trans, &levels, &dims, (1, 2), None).unwrap();
                for l in 0..levels.len() {
                    let means: Vec<Array3<f32>> = trans
                        .iter()
                        .map(|t| {
                            if l == 0 {
                                return t.clone();
                            }
                            let (rows, cols) = dims[l];
                            let mut out = vec![f32::NAN; rows * cols * bands];
                            let all = |_: usize, _: usize| true;
                            overview::coarse_bands_into(t.view(), levels[l] as usize, (1, 2), Resampling::Average, all, &mut out, 1, 0).unwrap();
                            Array3::from_shape_vec((rows, cols, bands), out).unwrap()
                        })
                        .collect();
                    for ((i, j), _) in means[0].slice(ndarray::s![.., .., 0]).indexed_iter() {
                        let current: Vec<f32> = (0..bands).map(|_| value(&mut rng)).collect();
                        let finite = current.iter().all(|v| v.is_finite());
                        let mut got = Vec::new();
                        climate.sims(&current, finite, l, (i, j), &mut got);
                        assert_eq!(got.len(), scenarios);
                        for (s, m) in means.iter().enumerate() {
                            let cell = m.slice(ndarray::s![i, j, ..]);
                            let expected = similarity(&ArrayView1::from(&current[..]), &cell).unwrap_or(NO_SIM);
                            assert_eq!(got[s].to_bits(), expected.to_bits(), "{scenarios} x {bands}, level {l}, ({i}, {j}), {s}");
                        }
                    }
                }
            }
        }
    }
}
