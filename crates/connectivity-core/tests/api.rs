//! The public API on the multi-level fixture of the unit tests, through crate paths only.
use connectivity_core::ndarray::{Array2, Array3};
use connectivity_core::{
    connectivity, level_transforms, pixel_coverage, Affine, Config, Crossing, CrossingMode, Inputs, WindowMode,
};
use std::collections::HashMap;

struct Tile {
    condition: Array2<f32>,
    mask: Array2<bool>,
    traversal: Array2<f32>,
    transgrids: Vec<Array3<f32>>,
    transforms: HashMap<i32, Affine>,
}

const LEVELS: [usize; 3] = [1, 2, 4];
const OFFSETS: (usize, usize) = (3, 5);

/// `testkit::Fixture::multilevel`, built with the public helpers.
fn tile() -> Tile {
    let condition = Array2::from_shape_fn((13, 14), |(i, j)| {
        if (i * 7 + j * 11) % 29 == 0 {
            f32::NAN
        } else {
            ((i * 7 + j * 13) % 10 + 1) as f32 / 10.0
        }
    });
    let mask = Array2::from_shape_fn((13, 14), |(i, j)| {
        !condition[[i, j]].is_finite() || (i * 3 + j * 5) % 7 == 0
    });
    let traversal = Array2::from_shape_fn((13, 14), |(i, j)| {
        if (i + j * 3) % 13 == 0 {
            f32::NAN
        } else {
            ((i * 5 + j * 3) % 11) as f32 / 10.0
        }
    });
    let transgrids = (0..2)
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
    let base = (1000.0, 0.0, 500_000.0, 0.0, -1000.0, 6_000_000.0);
    let transforms = level_transforms(base, &LEVELS, OFFSETS);
    Tile { condition, mask, traversal, transgrids, transforms }
}

fn inputs<'a>(t: &'a Tile, beri: bool) -> Inputs<'a> {
    Inputs {
        condition: &t.condition,
        mask: &t.mask,
        traversal: Some(&t.traversal),
        transgrids: beri.then_some(t.transgrids.as_slice()),
        transforms: &t.transforms,
    }
}

fn config(window_mode: WindowMode, pa_to_pa: bool) -> Config<'static> {
    Config {
        levels: &LEVELS,
        lambdas: &[2.0, 20.0],
        is_geo: false,
        max_cost: 20.0,
        window_size: 3,
        outer_window: 5,
        offsets: OFFSETS,
        window_mode,
        pa_to_pa,
        crossing: None,
    }
}

fn same_bits(a: &Array2<f32>, b: &Array2<f32>) -> bool {
    a.dim() == b.dim() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

// The frozen sample values of the unit tests' `multilevel_values`
#[test]
fn multilevel_samples() {
    let t = tile();
    let cells = [(0, 1), (3, 5), (6, 8), (12, 13)];
    for (mode, beri, pa_to_pa, expected) in [
        (WindowMode::Circular, false, false, [0.035036724, 0.060928639, 0.057472181, 0.023413222]),
        (WindowMode::Circular, false, true, [0.031013189, 0.051940750, 0.050136760, 0.017623197]),
        (WindowMode::Circular, true, false, [0.287769228, 0.330879152, 0.336351663, 0.270322204]),
        (WindowMode::Square, false, false, [0.036365397, 0.062699854, 0.057887629, 0.023465034]),
        (WindowMode::Square, false, true, [0.032288730, 0.053552385, 0.050358951, 0.017685590]),
        (WindowMode::Square, true, false, [0.288273066, 0.334522188, 0.339534730, 0.269480884]),
    ] {
        let out = connectivity(&inputs(&t, beri), &config(mode, pa_to_pa)).unwrap();
        for (cell, expected) in cells.iter().zip(expected) {
            let actual = out[*cell];
            assert!((actual - expected).abs() <= 1.0e-6, "{mode:?} {beri} {pa_to_pa} {cell:?}: {actual}");
        }
        for (cell, &masked) in t.mask.indexed_iter() {
            assert_eq!(out[cell].is_nan(), masked);
        }
    }
}

// Any thread count of the caller's pool gives the same bits
#[test]
fn thread_pools() {
    let t = tile();
    for beri in [false, true] {
        let run = |threads| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| connectivity(&inputs(&t, beri), &config(WindowMode::Circular, true)).unwrap())
        };
        let one = run(1);
        assert!(same_bits(&one, &run(3)) && same_bits(&one, &run(8)));
    }
}

#[test]
fn optional_inputs() {
    let t = tile();
    let mut plain = inputs(&t, false);
    plain.traversal = None;
    let expected = connectivity(&plain, &config(WindowMode::Circular, false)).unwrap();
    plain.transgrids = Some(&[]);
    assert!(same_bits(&connectivity(&plain, &config(WindowMode::Circular, false)).unwrap(), &expected));
    let empty = Config { lambdas: &[], ..config(WindowMode::Circular, false) };
    let zeros = connectivity(&plain, &empty).unwrap();
    assert!(t.mask.iter().zip(&zeros).all(|(&m, &v)| if m { v.is_nan() } else { v == 0.0 }));
}

#[test]
fn default_config() {
    let c = Config::default();
    assert_eq!(c.levels, &[1, 2, 4, 8, 16, 32]);
    assert_eq!(c.lambdas, &[2.0, 20.0, 200.0]);
    assert_eq!((c.max_cost, c.window_size, c.outer_window), (2.0, 3, 9));
    assert_eq!((c.is_geo, c.offsets, c.window_mode, c.pa_to_pa), (false, (0, 0), WindowMode::Circular, false));
    assert_eq!(c.crossing, Some(Crossing { mode: CrossingMode::Strict, width: 0.5, cost: 4.0 }));
}

fn crossing(c: Config<'static>, width: f32, cost: f32) -> Config<'static> {
    Config { crossing: Some(Crossing { mode: CrossingMode::Strict, width, cost }), ..c }
}

#[test]
fn crossing_modes() {
    assert_eq!(CrossingMode::parse(" LandMass "), Ok(CrossingMode::Landmass));
    assert_eq!(CrossingMode::parse("strict"), Ok(CrossingMode::Strict));
    assert_eq!(
        CrossingMode::parse("none"),
        Err("Invalid nodata_crossing 'none'. Expected 'landmass' or 'strict'.".to_string())
    );
}

// Through the public API: without NoData every crossing mode gives the plain output bit for
// bit; an island 6 cells (6 km) from the mainland is unaffected by it at a width of 5 km and
// affected at 7 km, in both modes
#[test]
fn crossing_island() {
    let t = tile();
    let full = t.condition.mapv(|v| if v.is_nan() { 0.4 } else { v });
    let plain = config(WindowMode::Circular, false);
    let run = |cond: &Array2<f32>, mask: &Array2<bool>, c: &Config| {
        connectivity(&Inputs { condition: cond, mask, ..inputs(&t, false) }, c).unwrap()
    };
    let expected = run(&full, &t.mask, &plain);
    for mode in [CrossingMode::Landmass, CrossingMode::Strict] {
        let c = Config { crossing: Some(Crossing { mode, width: 3.0, cost: 4.0 }), ..plain };
        assert!(same_bits(&run(&full, &t.mask, &c), &expected), "{mode:?}");
    }

    let (h, w) = (24, 20);
    let land = |mainland: bool| {
        Array2::from_shape_fn((h, w), |(i, j)| match i {
            0..=5 if mainland => 0.5,
            12..=14 if (5..15).contains(&j) => 0.9 - 0.02 * j as f32,
            _ => f32::NAN,
        })
    };
    let (with, alone) = (land(true), land(false));
    let island = alone.mapv(|v| v.is_finite());
    let mask = island.mapv(|v| !v);
    let transforms = level_transforms((1000.0, 0.0, 0.0, 0.0, -1000.0, 0.0), &LEVELS, (0, 0));
    for mode in [CrossingMode::Landmass, CrossingMode::Strict] {
        for (width, linked) in [(5.0, false), (7.0, true)] {
            let c = Config { offsets: (0, 0), crossing: Some(Crossing { mode, width, cost: 4.0 }), ..plain };
            let score = |cond: &Array2<f32>| {
                let inputs = Inputs { condition: cond, mask: &mask, traversal: None, transgrids: None, transforms: &transforms };
                connectivity(&inputs, &c).unwrap()
            };
            let (a, b) = (score(&with), score(&alone));
            let changed = island.indexed_iter().filter(|(ij, &f)| f && a[*ij].to_bits() != b[*ij].to_bits()).count();
            let cells = island.iter().filter(|&&f| f).count();
            assert_eq!(changed, if linked { cells } else { 0 }, "{mode:?} width {width}");
        }
    }
}

/// A one-row strip of 1 km cells: land A (focal) and B with condition 0.5, NoData X between them.
fn strip() -> (Array2<f32>, Array2<bool>, HashMap<i32, Affine>) {
    let condition = Array2::from_shape_vec((1, 3), vec![0.5, f32::NAN, 0.5]).unwrap();
    let mask = Array2::from_shape_vec((1, 3), vec![false, true, true]).unwrap();
    let transforms = level_transforms((1000.0, 0.0, 0.0, 0.0, -1000.0, 1000.0), &[1], (0, 0));
    (condition, mask, transforms)
}

fn strip_config(max_cost: f32, crossing: Option<Crossing>) -> Config<'static> {
    Config { levels: &[1], lambdas: &[2.0, 20.0], max_cost, window_size: 3, outer_window: 3, crossing, ..Config::default() }
}

// The crossing cost is the path weight of crossed NoData per kilometre, independent of max_cost:
// from A, the only route to B steps 1 km into X at the crossing cost c, then 1 km into B at B's
// weight, so a_B = c + (1 - max_cost) 0.5 + max_cost against an intact g_B = 2. The focal cell is
// not a destination and X adds no habitat area, so connectedness is the mean over lambdas of
// 0.5 exp(-a_B / lambda) / exp(-2 / lambda), worked out here independently
#[test]
fn crossing_cost() {
    let (condition, mask, transforms) = strip();
    let inputs = Inputs { condition: &condition, mask: &mask, traversal: None, transgrids: None, transforms: &transforms };
    let expected = |max_cost: f64, cost: f64| {
        let a = cost + (1.0 - max_cost) * 0.5 + max_cost;
        [2.0f64, 20.0].iter().map(|l| 0.5 * (-a / l).exp() / (-2.0 / l).exp()).sum::<f64>() / 2.0
    };
    // Costs of exactly 1 and 4, below, at and above max_cost, and max_cost 1 (land weight 1)
    for (max_cost, cost) in [(2.0, 1.0), (2.0, 4.0), (1.0, 1.0), (1.0, 4.0), (20.0, 4.0), (2.0, 30.0), (4.0, 4.0)] {
        for mode in [CrossingMode::Strict, CrossingMode::Landmass] {
            let config = strip_config(max_cost, Some(Crossing { mode, width: 1.5, cost }));
            let out = connectivity(&inputs, &config).unwrap();
            let want = expected(max_cost as f64, cost as f64);
            assert!((out[[0, 0]] as f64 - want).abs() < 1.0e-6, "{mode:?} max_cost {max_cost} cost {cost}: {} vs {want}", out[[0, 0]]);
            assert!(out[[0, 1]].is_nan() && out[[0, 2]].is_nan());
        }
    }
    // Cost 1 on intact land (condition 1): every adjusted path equals the intact one, a score of 1
    let intact = condition.mapv(|v| if v.is_nan() { v } else { 1.0 });
    let config = strip_config(2.0, Some(Crossing { mode: CrossingMode::Strict, width: 1.5, cost: 1.0 }));
    let out = connectivity(&Inputs { condition: &intact, ..inputs }, &config).unwrap();
    assert!((out[[0, 0]] - 1.0).abs() < 1.0e-6, "{}", out[[0, 0]]);
    // Not crossed, under legacy NoData or a width below the gap: B is out of reach, so A scores as
    // without B (an isolated cell's synthetic neighbour, as in earlier versions), whatever the cost
    let mut alone = condition.clone();
    alone[[0, 2]] = f32::NAN;
    for crossing in [None, Some(Crossing { mode: CrossingMode::Strict, width: 0.0, cost: 4.0 })] {
        let config = strip_config(2.0, crossing);
        let out = connectivity(&inputs, &config).unwrap();
        let without = connectivity(&Inputs { condition: &alone, ..inputs }, &config).unwrap();
        assert_eq!(out[[0, 0]].to_bits(), without[[0, 0]].to_bits(), "{crossing:?}");
        assert!((out[[0, 0]] as f64 - expected(2.0, 4.0)).abs() > 1.0e-3);
    }
}

// A straight channel `gap` km wide across a raster of 1 km cells is crossed at a width equal to
// it or above, and not just below it: the far bank changes the focal cells' scores, or none
fn channel(gap: usize, far: bool) -> Array2<f32> {
    Array2::from_shape_fn((41, 12 + gap), |(i, j)| match j {
        _ if (6..6 + gap).contains(&j) => f32::NAN,
        _ if j >= 6 + gap && !far => f32::NAN,
        _ => 0.3 + ((i * 7 + j * 3) % 10) as f32 * 0.07,
    })
}

#[test]
fn crossing_widths() {
    for gap in [1, 3] {
        let (with, without) = (channel(gap, true), channel(gap, false));
        // Focal cells on the near bank, away from the channel's ends
        let mask = Array2::from_shape_fn(with.dim(), |(i, j)| !((12..29).contains(&i) && (3..6).contains(&j)));
        let transforms = level_transforms((1000.0, 0.0, 0.0, 0.0, -1000.0, 41_000.0), &LEVELS, (0, 0));
        for mode in [CrossingMode::Strict, CrossingMode::Landmass] {
            for (width, crossed) in [(gap as f32 - 0.01, false), (gap as f32, true), (gap as f32 + 0.01, true)] {
                let config = Config {
                    levels: &LEVELS, window_size: 3, outer_window: 5,
                    crossing: Some(Crossing { mode, width, cost: 4.0 }), ..Config::default()
                };
                let run = |cond: &Array2<f32>| {
                    let inputs = Inputs { condition: cond, mask: &mask, traversal: None, transgrids: None, transforms: &transforms };
                    connectivity(&inputs, &config).unwrap()
                };
                let (a, b) = (run(&with), run(&without));
                let focal = mask.iter().filter(|&&m| !m).count();
                let changed = mask.indexed_iter().filter(|(ij, &m)| !m && a[*ij].to_bits() != b[*ij].to_bits()).count();
                assert_eq!(changed, if crossed { focal } else { 0 }, "{mode:?} gap {gap} width {width}");
            }
        }
    }
}

// Crossing settings that cannot take effect change nothing: any width and cost without a mode
// (legacy NoData handling, `None`), and any cost when nothing is crossable (strict, width 0).
// Strict mode at width 0 is still not `None`: coarse levels link a nearby island to the mainland
// across NoData only under `None`
#[test]
fn crossing_inactive() {
    let (h, w) = (24, 20);
    let land = |mainland: bool| {
        Array2::from_shape_fn((h, w), |(i, j)| match i {
            0..=5 if mainland => 0.5,
            8..=10 if (5..15).contains(&j) => 0.9 - 0.02 * j as f32,
            _ => f32::NAN,
        })
    };
    let (with, alone) = (land(true), land(false));
    let island = alone.mapv(|v| v.is_finite());
    let mask = island.mapv(|v| !v);
    let transforms = level_transforms((1000.0, 0.0, 0.0, 0.0, -1000.0, 0.0), &LEVELS, (0, 0));
    let config = |crossing| Config { levels: &LEVELS, window_size: 3, outer_window: 5, crossing, ..Config::default() };
    let run = |cond: &Array2<f32>, crossing| {
        let inputs = Inputs { condition: cond, mask: &mask, traversal: None, transgrids: None, transforms: &transforms };
        connectivity(&inputs, &config(crossing)).unwrap()
    };
    let strict = |width, cost| Some(Crossing { mode: CrossingMode::Strict, width, cost });
    let zero = run(&with, strict(0.0, 4.0));
    for cost in [1.0, 2.5, 1000.0] {
        assert!(same_bits(&run(&with, strict(0.0, cost)), &zero), "cost {cost}");
    }
    let changed = |a: &Array2<f32>, b: &Array2<f32>| island.indexed_iter().filter(|(ij, &f)| f && a[*ij].to_bits() != b[*ij].to_bits()).count();
    // Strict at width 0: the island is unaffected by the mainland; legacy: coarse levels link it
    assert_eq!(changed(&zero, &run(&alone, strict(0.0, 4.0))), 0);
    let legacy = run(&with, None);
    assert!(changed(&legacy, &run(&alone, None)) > 0);
    assert!(changed(&legacy, &zero) > 0);
    // Legacy is the default engine of earlier versions: the plain multi-level fixture's frozen values
    // are checked in `multilevel_samples`, with `crossing: None`
}

// Crossed NoData only carries movement: its BERI climate, traversal and protected-area mask do not
// change any score, with and without PARC target gating
#[test]
fn crossing_inert() {
    let (condition, _, transforms) = strip();
    let condition = condition.mapv(|v| if v.is_nan() { v } else { v + 0.2 });
    let grid = |x: f32| Array3::from_shape_fn((1, 3, 2), |(_, j, k)| if j == 1 { x } else { 0.2 * (j + k) as f32 + 0.1 });
    let traversal = |x: f32| Array2::from_shape_vec((1, 3), vec![0.4, x, 0.9]).unwrap();
    let config = |pa_to_pa| Config { pa_to_pa, ..strip_config(2.0, Some(Crossing { mode: CrossingMode::Strict, width: 1.5, cost: 4.0 })) };
    for pa_to_pa in [false, true] {
        let mut outputs = Vec::new();
        for (x, masked) in [(0.1, false), (0.9, true), (f32::NAN, true)] {
            let mask = Array2::from_shape_vec((1, 3), vec![false, masked, false]).unwrap();
            let (grids, trav) = (vec![grid(x), grid(x * 0.5)], traversal(x));
            for beri in [false, true] {
                let inputs = Inputs {
                    condition: &condition, mask: &mask, traversal: Some(&trav),
                    transgrids: beri.then_some(grids.as_slice()), transforms: &transforms,
                };
                outputs.push((beri, connectivity(&inputs, &config(pa_to_pa)).unwrap()));
            }
        }
        for beri in [false, true] {
            let runs: Vec<_> = outputs.iter().filter(|(b, _)| *b == beri).map(|(_, out)| out).collect();
            assert!(runs[0][[0, 0]] > 0.0);
            // X itself is NoData: NaN when masked and 0 (nothing reached) when not
            for out in &runs[1..] {
                assert_eq!(out[[0, 0]].to_bits(), runs[0][[0, 0]].to_bits(), "beri {beri} pa_to_pa {pa_to_pa}");
                assert_eq!(out[[0, 2]].to_bits(), runs[0][[0, 2]].to_bits(), "beri {beri} pa_to_pa {pa_to_pa}");
            }
        }
    }
}

#[test]
fn window_modes() {
    assert_eq!(WindowMode::parse(" SqUaRe "), Ok(WindowMode::Square));
    assert_eq!(WindowMode::parse("circular"), Ok(WindowMode::Circular));
    assert_eq!(
        WindowMode::parse("hexagon"),
        Err("Invalid window_mode 'hexagon'. Expected 'square' or 'circular'.".to_string())
    );
}

// Level transforms match rasterio's `base * scale(L) * translation(col0 // L, row0 // L)`
#[test]
fn tile_transforms() {
    let base = Affine::from((250.0, 0.0, 500_000.0, 0.0, -250.0, 6_000_000.0));
    let transforms = level_transforms(base, &[1, 2, 4], (5, 11));
    for (level, (x0, y0)) in [(1, (502_750.0, 5_998_750.0)), (2, (502_500.0, 5_999_000.0)), (4, (502_000.0, 5_999_000.0))] {
        let t = transforms[&level];
        let s = 250.0 * level as f64;
        assert_eq!((t.x_scale, t.x_skew, t.x_origin, t.y_skew, t.y_scale, t.y_origin), (s, 0.0, x0, 0.0, -s, y0));
        assert_eq!(t.xy(0, 0), (x0 + s / 2.0, y0 - s / 2.0));
    }
    assert_eq!(base.col_row(500_625.0, 5_999_000.0), (2.5, 4.0));
}

// Invalid settings fail before any work, also when every cell is masked
#[test]
fn setting_errors() {
    let t = tile();
    let all_masked = Array2::from_elem((13, 14), true);
    let base: HashMap<i32, Affine> = HashMap::from([(1, t.transforms[&1])]);
    let coarse: HashMap<i32, Affine> = t.transforms.iter().filter(|(&l, _)| l != 1).map(|(&l, &a)| (l, a)).collect();
    let c = config(WindowMode::Circular, false);
    let sheared = level_transforms((1000.0, 300.0, 500_000.0, 0.0, -1000.0, 6_000_000.0), &LEVELS, OFFSETS);
    let geo = Config { is_geo: true, ..crossing(c, 1.0, 4.0) };
    for (config, transforms, message) in [
        (Config { window_size: 4, ..c }, &t.transforms, "Window size must be odd-numbered"),
        (Config { window_size: -1, outer_window: -1, ..c }, &t.transforms, "Window size must be positive"),
        (Config { outer_window: 6, ..c }, &t.transforms, "Outer window size must be odd-numbered"),
        (Config { levels: &[1, 3], ..c }, &t.transforms, "Level 3 is not a power of 2"),
        (Config { levels: &[2, 4], ..c }, &t.transforms, "Base condition level not found!"),
        (Config { levels: &[1, 4], ..c }, &t.transforms, "levels must form a continuous power-of-two sequence"),
        (c, &base, "Missing level in Affine set."),
        (c, &coarse, "Missing base-level transform (key=1)."),
        (crossing(c, -1.0, 4.0), &t.transforms, "Crossing width must be a finite number of kilometres >= 0, got -1"),
        (crossing(c, f32::NAN, 4.0), &t.transforms, "Crossing width must be a finite number of kilometres >= 0, got NaN"),
        (crossing(c, f32::INFINITY, 4.0), &t.transforms, "Crossing width must be a finite number of kilometres >= 0, got inf"),
        (crossing(c, 1.0, 0.5), &t.transforms, "Crossing cost must be a finite number >= 1, got 0.5"),
        (crossing(c, 1.0, 0.0), &t.transforms, "Crossing cost must be a finite number >= 1, got 0"),
        (crossing(c, 1.0, -4.0), &t.transforms, "Crossing cost must be a finite number >= 1, got -4"),
        (crossing(c, 1.0, f32::NAN), &t.transforms, "Crossing cost must be a finite number >= 1, got NaN"),
        (crossing(c, 1.0, f32::INFINITY), &t.transforms, "Crossing cost must be a finite number >= 1, got inf"),
        // A crossing cost is checked whatever the width: also at 0, where nothing is crossed
        (crossing(c, 0.0, 0.5), &t.transforms, "Crossing cost must be a finite number >= 1, got 0.5"),
        (
            crossing(c, 1.0, 4.0),
            &sheared,
            "NoData crossing needs perpendicular rows and columns, but the base transform is sheared: (a, b, d, e) = (1000, 300, 0, -1000)",
        ),
        (geo, &sheared, "NoData crossing needs a north-up geographic grid (transform terms b and d of 0), got b = 300, d = 0"),
    ] {
        for mask in [&t.mask, &all_masked] {
            let inputs = Inputs { mask, transforms, ..inputs(&t, false) };
            assert_eq!(connectivity(&inputs, &config).unwrap_err().to_string(), message);
        }
    }
}

// Arrays that do not match condition fail instead of indexing out of bounds or running
#[test]
fn shape_errors() {
    use connectivity_core::ndarray::s;
    let t = tile();
    let mask = t.mask.slice(s![..12, ..]).to_owned();
    let traversal = t.traversal.slice(s![..12, ..]).to_owned();
    let short = vec![t.transgrids[0].slice(s![..12, .., ..]).to_owned(), t.transgrids[1].clone()];
    let bands = vec![t.transgrids[0].clone(), Array3::zeros((13, 14, 3))];
    let empty = vec![Array3::zeros((13, 14, 0)); 2];
    for (inputs, message) in [
        (Inputs { mask: &mask, ..inputs(&t, false) }, "Dimension mismatch for mask: (12, 14) vs (13, 14)"),
        (Inputs { traversal: Some(&traversal), ..inputs(&t, false) }, "Dimension mismatch for key 1: (12, 14) vs (13, 14)"),
        (Inputs { transgrids: Some(&short), ..inputs(&t, true) }, "Dimension mismatch for transgrid 0: (12, 14, 2) vs (13, 14, 2)"),
        (Inputs { transgrids: Some(&bands), ..inputs(&t, true) }, "Dimension mismatch for transgrid 1: (13, 14, 3) vs (13, 14, 2)"),
        (Inputs { transgrids: Some(&empty), ..inputs(&t, true) }, "Transgrids must have at least one band"),
    ] {
        let error = connectivity(&inputs, &config(WindowMode::Circular, false)).unwrap_err();
        assert_eq!(error.to_string(), message);
    }
}

// A path cost beyond u32::MAX on any worker thread comes back as an error
#[test]
fn cost_overflow() {
    let t = tile();
    let config = Config { max_cost: 3.0e7, ..config(WindowMode::Circular, false) };
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let error = pool.install(|| connectivity(&inputs(&t, false), &config)).unwrap_err();
        assert_eq!(error.to_string(), "Path cost overflow; check max_cost and resistance scaling!");
    }
}

#[test]
fn coverage_fractions() {
    // 10 m pixels from (1000, 2000); pixel (row 1, col 2) spans x 1020..1030, y 1980..1990
    let transform = Affine::from((10.0, 0.0, 1000.0, 0.0, -10.0, 2000.0));
    let square = |x0: f64, y0: f64, x1: f64, y1: f64| vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
    let polygons = vec![
        (square(1020.0, 1990.0, 1030.0, 1980.0), vec![]),
        (square(1040.0, 1990.0, 1045.0, 1980.0), vec![]),
        (square(1000.0, 1960.0, 1040.0, 1940.0), vec![square(1010.0, 1955.0, 1030.0, 1945.0)]),
    ];
    let out = pixel_coverage(&polygons, &transform, 6, 6);
    assert_eq!((out[[1, 2]], out[[1, 4]]), (1.0, 0.5));
    for row in 4..6 {
        assert_eq!(out.row(row).to_vec(), vec![1.0, 0.5, 0.5, 1.0, 0.0, 0.0]);
    }
    assert_eq!(out.sum(), 7.5);
    assert_eq!(pixel_coverage(&[], &transform, 2, 3), Array2::<f32>::zeros((2, 3)));
}
