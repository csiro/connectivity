//! Test fixtures, a small deterministic PRNG and output comparisons.
use crate::affine::Affine;
use crate::legacy::{self, canonical, Prepared};
use crate::window::WindowMode;
use ndarray::{Array2, Array3};
use std::collections::HashMap;

/// SplitMix64: fixed seeds give the same fixtures on every platform.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }

    pub fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

/// Level transforms anchored like `rastio.read_raster`:
/// `base * Affine.scale(level) * Affine.translation(col0 // level, row0 // level)`,
/// with the `affine` package's operation order.
pub fn level_transforms(
    base: (f64, f64, f64, f64, f64, f64),
    levels: &[usize],
    offsets: (usize, usize),
) -> HashMap<i32, Affine> {
    let (a, b, c, d, e, f) = base;
    levels
        .iter()
        .map(|&level| {
            let l = level as f64;
            let (sa, sb, sd, se) = (a * l + b * 0.0, a * 0.0 + b * l, d * l + e * 0.0, d * 0.0 + e * l);
            let (tc, tr) = ((offsets.1 / level) as f64, (offsets.0 / level) as f64);
            let t = (sa, sb, sa * tc + sb * tr + c, sd, se, sd * tc + se * tr + f);
            (level as i32, Affine::from(t))
        })
        .collect()
}

#[derive(Clone)]
pub struct Fixture {
    pub cond: Array2<f32>,
    pub mask: Array2<bool>,
    pub trav: Option<Array2<f32>>,
    pub trans: Option<Vec<Array3<f32>>>,
    pub transforms: HashMap<i32, Affine>,
    pub levels: Vec<usize>,
    pub lambdas: Vec<f32>,
    pub is_geo: bool,
    pub max_cost: f32,
    pub window_size: i32,
    pub outer_window: i32,
    pub offsets: (usize, usize),
    pub mode: WindowMode,
    pub pa_to_pa: bool,
}

impl Fixture {
    pub fn prepared(&self) -> Prepared<'_> {
        Prepared::new(
            &self.cond,
            self.trav.as_ref(),
            self.trans.as_deref(),
            &self.transforms,
            &self.mask,
            &self.levels,
            &self.lambdas,
            self.is_geo,
            self.max_cost,
            self.window_size,
            self.outer_window,
            self.offsets,
            self.mode,
            self.pa_to_pa,
        )
        .unwrap()
    }

    pub fn legacy(&self) -> Array2<f32> {
        legacy::conn(
            &Some(self.cond.clone()),
            self.trav.as_ref(),
            self.trans.as_deref(),
            &self.transforms,
            &self.mask,
            &self.levels,
            &self.lambdas,
            self.is_geo,
            self.max_cost,
            self.window_size,
            self.outer_window,
            self.offsets,
            self.mode,
            self.pa_to_pa,
        )
        .unwrap()
    }

    pub fn canonical(&self, inject: Option<canonical::Inject>) -> Array2<f32> {
        canonical::conn(&self.prepared(), &self.cond, &self.mask, inject)
    }

    /// The multi-level fixture behind `multilevel_regression`.
    pub fn multilevel(mode: WindowMode, beri: bool, pa_to_pa: bool) -> Self {
        let cond = Array2::from_shape_fn((13, 14), |(i, j)| {
            if (i * 7 + j * 11) % 29 == 0 {
                f32::NAN
            } else {
                ((i * 7 + j * 13) % 10 + 1) as f32 / 10.0
            }
        });
        let mask = Array2::from_shape_fn((13, 14), |(i, j)| {
            !cond[[i, j]].is_finite() || (i * 3 + j * 5) % 7 == 0
        });
        let traversal = Array2::from_shape_fn((13, 14), |(i, j)| {
            if (i + j * 3) % 13 == 0 {
                f32::NAN
            } else {
                ((i * 5 + j * 3) % 11) as f32 / 10.0
            }
        });
        let transitions: Vec<Array3<f32>> = (0..2)
            .map(|scenario| {
                Array3::from_shape_fn((13, 14, 2), |(i, j, k)| {
                    if (i * 3 + j * 5 + scenario + k) % 31 == 0 {
                        f32::NAN
                    } else {
                        ((i * 7 + j * 3 + scenario * 11 + k * 2) % 23) as f32 / 23.0
                    }
                })
            })
            .collect();
        let levels = vec![1, 2, 4];
        let offsets = (3, 5);
        let transforms: HashMap<i32, Affine> = levels
            .iter()
            .map(|&level| {
                let scale = 1000.0 * level as f64;
                let row0 = offsets.0 / level * level;
                let col0 = offsets.1 / level * level;
                (
                    level as i32,
                    Affine::from((
                        scale,
                        0.0,
                        500_000.0 + col0 as f64 * 1000.0,
                        0.0,
                        -scale,
                        6_000_000.0 - row0 as f64 * 1000.0,
                    )),
                )
            })
            .collect();
        Self {
            cond,
            mask,
            trav: Some(traversal),
            trans: beri.then_some(transitions),
            transforms,
            levels,
            lambdas: vec![2.0, 20.0],
            is_geo: false,
            max_cost: 20.0,
            window_size: 3,
            outer_window: 5,
            offsets,
            mode,
            pa_to_pa,
        }
    }

    /// A randomized fixture: NaN holes, masks, optional traversal and BERI scenarios,
    /// non-aligned offsets and projected or geographic transforms.
    pub fn random(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let levels: Vec<usize> = match rng.below(4) {
            0 => vec![1, 2],
            1 => vec![1, 2, 4],
            2 => vec![1, 2, 4, 8],
            _ => vec![1],
        };
        let max_level = *levels.last().unwrap();
        let rows = 4 + rng.below(18 + 2 * max_level);
        let cols = 4 + rng.below(18 + 2 * max_level);
        let offsets = (rng.below(40), rng.below(40));
        let mode = if rng.chance(0.5) {
            WindowMode::Circular
        } else {
            WindowMode::Square
        };
        let window_size = rng.pick(&[1, 3, 3, 5]);
        let outer_window = window_size + 2 * rng.below(3) as i32;

        // Coarse value sets make tied and zero-cost edges common
        let coarse = rng.chance(0.4);
        let nan_rate = rng.pick(&[0.0, 0.05, 0.2]);
        let value = |rng: &mut Rng| -> f32 {
            if rng.chance(nan_rate) {
                f32::NAN
            } else if coarse {
                rng.pick(&[0.0, 0.5, 1.0])
            } else {
                rng.unit() as f32
            }
        };
        let cond = Array2::from_shape_fn((rows, cols), |_| value(&mut rng));
        let mask_rate = rng.pick(&[0.0, 0.1, 0.4]);
        let keep_nan_sources = rng.chance(0.2);
        let mask = Array2::from_shape_fn((rows, cols), |(i, j)| {
            (!keep_nan_sources && cond[[i, j]].is_nan()) || rng.chance(mask_rate)
        });
        let trav = rng.chance(0.4).then(|| Array2::from_shape_fn((rows, cols), |_| value(&mut rng)));
        let trans = rng.chance(0.4).then(|| {
            let scenarios = 1 + rng.below(3);
            let bands = 1 + rng.below(3);
            let band_nan = rng.pick(&[0.0, 0.02, 0.1]);
            (0..scenarios)
                .map(|_| {
                    Array3::from_shape_fn((rows, cols, bands), |_| {
                        if rng.chance(band_nan) {
                            f32::NAN
                        } else {
                            rng.unit() as f32
                        }
                    })
                })
                .collect()
        });

        let is_geo = rng.chance(0.4);
        let base = if is_geo {
            let res = rng.pick(&[0.0025, 1.0 / 120.0, 0.05, 0.25]);
            let lon = 100.0 + 79.0 * rng.unit();
            let lat = -45.0 + 30.0 * rng.unit();
            (res, 0.0, lon, 0.0, -res * rng.pick(&[1.0, 1.0, 0.75]), lat)
        } else {
            let res = rng.pick(&[30.0, 250.0, 1000.0, 0.3, 92.6]);
            let x0 = rng.pick(&[0.0, 500_000.0, -2_000_123.4, 1.0e7 + 0.1]);
            let y0 = rng.pick(&[6_000_000.0, -4_500_000.75, 1.0e7]);
            (res * rng.pick(&[1.0, 1.0, 1.5]), 0.0, x0, 0.0, -res, y0)
        };
        let transforms = level_transforms(base, &levels, offsets);
        let lambdas = match rng.below(4) {
            0 => vec![],
            1 => vec![2.0],
            _ => vec![2.0, 20.0, 50.0],
        };
        let max_cost = rng.pick(&[0.0, 1.0, 2.0, 2.0, 20.0]);
        let pa_to_pa = rng.chance(0.3);

        Self {
            cond,
            mask,
            trav,
            trans,
            transforms,
            levels,
            lambdas,
            is_geo,
            max_cost,
            window_size,
            outer_window,
            offsets,
            mode,
            pa_to_pa,
        }
    }
}

/// Difference statistics between a new and a reference output.
#[derive(Debug, Default)]
pub struct Diff {
    pub cells: usize,
    pub differing: usize,
    pub mask_mismatch: usize,
    pub max_abs: f64,
    pub mean_abs: f64,
    pub max_rel: f64,
    pub mean_rel: f64,
    pub zero_ref: usize,
    pub zero_ref_differing: usize,
}

pub fn diff(new: &Array2<f32>, reference: &Array2<f32>) -> Diff {
    let mut d = Diff::default();
    let (mut sum_abs, mut sum_rel, mut rel_cells) = (0.0, 0.0, 0usize);
    for (&a, &b) in new.iter().zip(reference) {
        if a.is_finite() != b.is_finite() {
            d.mask_mismatch += 1;
            continue;
        }
        if !b.is_finite() {
            continue;
        }
        d.cells += 1;
        let abs = (a as f64 - b as f64).abs();
        if a.to_bits() != b.to_bits() {
            d.differing += 1;
        }
        sum_abs += abs;
        d.max_abs = d.max_abs.max(abs);
        if b == 0.0 {
            d.zero_ref += 1;
            if a != 0.0 {
                d.zero_ref_differing += 1;
            }
        } else {
            let rel = abs / (b as f64).abs();
            sum_rel += rel;
            rel_cells += 1;
            d.max_rel = d.max_rel.max(rel);
        }
    }
    d.mean_abs = sum_abs / d.cells.max(1) as f64;
    d.mean_rel = sum_rel / rel_cells.max(1) as f64;
    d
}

/// True when both arrays hold the same bits everywhere.
pub fn same_bits(a: &Array2<f32>, b: &Array2<f32>) -> bool {
    a.dim() == b.dim() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

impl Fixture {
    /// Focused variants of a randomized fixture: isolated sources, unmasked NaN sources, empty
    /// lambdas, tiny tiles, every parent-grid parity, NaN traversal and scenario bands.
    pub fn focused(seed: u64) -> Vec<Self> {
        let base = Self::random(seed);
        let mut out = Vec::new();

        // Sparse lattice of valid cells: every source is isolated
        for pa_to_pa in [false, true] {
            let mut f = base.clone();
            f.cond.indexed_iter_mut().for_each(|((i, j), v)| {
                if i % 3 != 0 || j % 3 != 0 {
                    *v = f32::NAN;
                }
            });
            f.mask = f.cond.mapv(|v| v.is_nan());
            f.pa_to_pa = pa_to_pa;
            out.push(f);
        }

        // NaN focal cells left unmasked, and empty lambdas
        let mut f = base.clone();
        f.mask.fill(false);
        out.push(f.clone());
        f.lambdas.clear();
        out.push(f);

        // Tiny tiles at several levels and offsets
        for (shape, offsets) in [((1, 1), (3, 5)), ((1, 5), (7, 1)), ((2, 3), (0, 0)), ((3, 1), (5, 6))] {
            let mut f = base.clone();
            f.levels = vec![1, 2, 4, 8];
            let rows = shape.0.min(f.cond.nrows());
            let cols = shape.1.min(f.cond.ncols());
            f.cond = f.cond.slice(ndarray::s![..rows, ..cols]).to_owned();
            f.mask = f.mask.slice(ndarray::s![..rows, ..cols]).to_owned();
            f.trav = f.trav.map(|t| t.slice(ndarray::s![..rows, ..cols]).to_owned());
            f.trans = f
                .trans
                .map(|ts| ts.iter().map(|t| t.slice(ndarray::s![..rows, ..cols, ..]).to_owned()).collect());
            f.offsets = offsets;
            let base_t = f.transforms[&1];
            let t = (base_t.x_scale, base_t.x_skew, base_t.x_origin, base_t.y_skew, base_t.y_scale, base_t.y_origin);
            f.transforms = level_transforms(t, &f.levels, offsets);
            out.push(f);
        }

        // All four parent-grid parities at levels 1 and 2
        for o0 in 0..4 {
            for o1 in 0..4 {
                let mut f = base.clone();
                f.offsets = (o0, o1);
                let base_t = f.transforms[&1];
                let t = (base_t.x_scale, base_t.x_skew, base_t.x_origin, base_t.y_skew, base_t.y_scale, base_t.y_origin);
                f.transforms = level_transforms(t, &f.levels, f.offsets);
                out.push(f);
            }
        }

        // Band-major transgrids, as a `np.moveaxis` view arrives from Python
        let mut f = base.clone();
        f.trans = f.trans.map(|ts| {
            ts.iter()
                .map(|t| {
                    let (rows, cols, bands) = t.dim();
                    let band_major = Array3::from_shape_fn((bands, rows, cols), |(b, i, j)| t[[i, j, b]]);
                    band_major.permuted_axes([1, 2, 0])
                })
                .collect()
        });
        out.push(f);

        // Traversal entirely NaN (condition fallback) and a NaN scenario band
        let mut f = base.clone();
        f.trav = Some(f.cond.mapv(|_| f32::NAN));
        if let Some(trans) = f.trans.as_mut() {
            trans[0].index_axis_mut(ndarray::Axis(2), 0).fill(f32::NAN);
        }
        out.push(f);
        out
    }
}

/// Per-thread allocation counts and live heap bytes, for steady-state and memory checks.
pub mod alloc {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static COUNT: Cell<usize> = const { Cell::new(0) };
        static LIVE: Cell<isize> = const { Cell::new(0) };
        static PEAK: Cell<isize> = const { Cell::new(0) };
    }

    fn grow(bytes: isize) {
        let live = LIVE.with(|l| {
            l.set(l.get() + bytes);
            l.get()
        });
        PEAK.with(|p| p.set(p.get().max(live)));
    }

    pub struct Counting;

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            COUNT.with(|c| c.set(c.get() + 1));
            grow(layout.size() as isize);
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            COUNT.with(|c| c.set(c.get() + 1));
            grow(layout.size() as isize);
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            grow(-(layout.size() as isize));
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            COUNT.with(|c| c.set(c.get() + 1));
            grow(new_size as isize - layout.size() as isize);
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static GLOBAL: Counting = Counting;

    /// Allocations and reallocations made by this thread so far
    pub fn count() -> usize {
        COUNT.with(|c| c.get())
    }

    /// Bytes this thread has allocated and not freed
    pub fn live() -> isize {
        LIVE.with(|l| l.get())
    }

    /// Start a new peak measurement at the current live bytes
    pub fn reset_peak() {
        PEAK.with(|p| p.set(live()));
    }

    /// Highest live bytes since `reset_peak`
    pub fn peak() -> isize {
        PEAK.with(|p| p.get())
    }
}

/// Minimal `.npy` reading (little-endian f32 or bool, C order) and raw f32 output.
pub mod npy {
    use ndarray::{ArrayD, IxDyn};
    use std::path::Path;

    fn read(path: &Path) -> (String, Vec<usize>, Vec<u8>) {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(&bytes[..6], b"\x93NUMPY", "{}", path.display());
        let (len, start) = if bytes[6] == 1 {
            (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
        } else {
            (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12)
        };
        let header = std::str::from_utf8(&bytes[start..start + len]).unwrap().to_string();
        assert!(header.contains("'fortran_order': False"), "{header}");
        let descr = header.split("'descr': '").nth(1).unwrap().split('\'').next().unwrap().to_string();
        let shape = header.split("'shape': (").nth(1).unwrap().split(')').next().unwrap();
        let shape = shape.split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse().unwrap()).collect();
        (descr, shape, bytes[start + len..].to_vec())
    }

    pub fn f32s(path: &Path) -> ArrayD<f32> {
        let (descr, shape, data) = read(path);
        assert_eq!(descr, "<f4");
        let values = data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        ArrayD::from_shape_vec(IxDyn(&shape), values).unwrap()
    }

    pub fn bools(path: &Path) -> ArrayD<bool> {
        let (descr, shape, data) = read(path);
        assert_eq!(descr, "|b1");
        ArrayD::from_shape_vec(IxDyn(&shape), data.iter().map(|&b| b != 0).collect()).unwrap()
    }

    /// Raw little-endian f32 values, row-major
    pub fn write_f32(path: &Path, values: &ndarray::Array2<f32>) {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes).unwrap();
    }
}

/// A captured `rust_conn.connectivity()` call (see the benchmark harness's `rust.txt`).
pub fn load_workload(dir: &std::path::Path) -> Fixture {
    use ndarray::Ix2;
    let text = std::fs::read_to_string(dir.join("rust.txt")).unwrap();
    let mut f = Fixture {
        cond: Array2::zeros((0, 0)),
        mask: Array2::from_elem((0, 0), false),
        trav: None,
        trans: None,
        transforms: HashMap::new(),
        levels: vec![],
        lambdas: vec![],
        is_geo: false,
        max_cost: 0.0,
        window_size: 0,
        outer_window: 0,
        offsets: (0, 0),
        mode: WindowMode::Circular,
        pa_to_pa: false,
    };
    let f32_bits = |s: &str| f32::from_bits(s.parse().unwrap());
    for line in text.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        let rest = &parts[1..];
        match parts[0] {
            "levels" => f.levels = rest.iter().map(|s| s.parse().unwrap()).collect(),
            "lambdas" => f.lambdas = rest.iter().map(|s| f32_bits(s)).collect(),
            "is_geo" => f.is_geo = rest[0] == "1",
            "max_cost" => f.max_cost = f32_bits(rest[0]),
            "window_size" => f.window_size = rest[0].parse().unwrap(),
            "outer_window" => f.outer_window = rest[0].parse().unwrap(),
            "offsets" => f.offsets = (rest[0].parse().unwrap(), rest[1].parse().unwrap()),
            "window_mode" => f.mode = WindowMode::parse(rest[0]).unwrap(),
            "pa_to_pa" => f.pa_to_pa = rest[0] == "1",
            "transform" => {
                let v: Vec<f64> = rest[1..].iter().map(|s| f64::from_bits(s.parse().unwrap())).collect();
                f.transforms.insert(rest[0].parse().unwrap(), Affine::from((v[0], v[1], v[2], v[3], v[4], v[5])));
            }
            "condition" => f.cond = npy::f32s(&dir.join(rest[0])).into_dimensionality::<Ix2>().unwrap(),
            "mask" => f.mask = npy::bools(&dir.join(rest[0])).into_dimensionality::<Ix2>().unwrap(),
            "traversal" => f.trav = Some(npy::f32s(&dir.join(rest[0])).into_dimensionality::<Ix2>().unwrap()),
            "transgrid" => f.trans.get_or_insert_with(Vec::new).push(
                npy::f32s(&dir.join(rest[0])).into_dimensionality::<ndarray::Ix3>().unwrap(),
            ),
            other => panic!("unknown key {other}"),
        }
    }
    f
}

impl Fixture {
    /// Projected or geographic grids whose edge costs sit on either side of integer rounding
    /// thresholds: pixel sizes just below, at and above multiples of 5 m, uniform weights, and
    /// origins large relative to the pixel size.
    pub fn thresholds(seed: u64) -> Self {
        let mut f = Self::random(seed ^ 0x5eed);
        let mut rng = Rng::new(seed);
        let res = rng.pick(&[4.999_999_9, 5.0, 5.000_000_1, 14.999_99, 15.0, 15.000_01, 2.5, 0.125]);
        let origin = rng.pick(&[0.0, 1.0e7 + 0.1, 123_456.789, -9.0e6, 4.0e6]);
        let base = if f.is_geo {
            let deg = res / 111_000.0;
            (deg, 0.0, 140.0 + rng.unit(), 0.0, -deg, -(10.0 + 30.0 * rng.unit()))
        } else {
            (res, 0.0, origin, 0.0, -res * rng.pick(&[1.0, 1.0, 2.0]), origin + 5.0e6)
        };
        f.transforms = level_transforms(base, &f.levels, f.offsets);
        f.max_cost = rng.pick(&[0.0, 1.0, 2.0]);
        f.cond.mapv_inplace(|v| if v.is_nan() { v } else { rng.pick(&[0.5f32, 1.0]) });
        f
    }

    /// A randomized fixture with skewed transforms (distance tables must fall back).
    pub fn skewed(seed: u64) -> Self {
        let mut f = Self::random(seed ^ 0x5ce3);
        let t = f.transforms[&1];
        let skew = t.x_scale.abs() * 1.0e-3;
        let base = (t.x_scale, skew, t.x_origin, -skew, t.y_scale, t.y_origin);
        f.transforms = level_transforms(base, &f.levels, f.offsets);
        f
    }
}
